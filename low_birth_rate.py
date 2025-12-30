from __future__ import annotations

import os
import re
import json
import time
import argparse
from uuid import uuid4
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv

from pinecone import Pinecone
try:
    from pinecone import ServerlessSpec  # pinecone >= 3.x
except Exception:  # pragma: no cover
    ServerlessSpec = None  # type: ignore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------
# 0) Text utils
# ---------------------------
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------
# 1) PDF → pages → chunks (+metadata)
# ---------------------------
def load_pdf_pages(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # page 단위
    for d in docs:
        d.page_content = clean_text(d.page_content)
    return docs


def split_into_chunks(
    docs_by_page: List[Document],
    chunk_size: int = 400,      # 학습 노트(02) 값에 맞춤
    chunk_overlap: int = 30,    # 학습 노트(02) 값에 맞춤
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs_by_page)


def enrich_chunks(chunks: List[Document], source_name: str) -> List[Document]:
    for i, d in enumerate(chunks):
        d.metadata = d.metadata or {}
        d.metadata["source"] = source_name
        d.metadata["chunk_id"] = i
    return chunks


# ---------------------------
# 2) Embeddings / LLM
# ---------------------------
def get_embeddings() -> OpenAIEmbeddings:
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 또는 환경변수로 설정하세요.")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def get_llm() -> ChatOpenAI:
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 또는 환경변수로 설정하세요.")
    return ChatOpenAI(model=model, api_key=api_key)


# ---------------------------
# 3) Pinecone index 준비
# ---------------------------
def get_pinecone() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY가 없습니다. .env 또는 환경변수로 설정하세요.")
    return Pinecone(api_key=api_key)


def ensure_index_exists(pc: Pinecone, index_name: str, dimension: int) -> None:
    """없으면 생성. 이미 있으면 아무것도 안 함."""
    try:
        existing_names = {i["name"] if isinstance(i, dict) else getattr(i, "name", None) for i in pc.list_indexes()}
    except Exception:
        existing_names = set()
        for i in pc.list_indexes():
            name = i.get("name") if isinstance(i, dict) else getattr(i, "name", None)
            if name:
                existing_names.add(name)

    if index_name in existing_names:
        return

    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    if ServerlessSpec is not None:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    else:
        pc.create_index(name=index_name, dimension=dimension, metric="cosine")


def get_index(pc: Pinecone, index_name: str):
    return pc.Index(index_name)


# ---------------------------
# 4) PDF 청킹 → 임베딩 → Pinecone upsert
# ---------------------------
def index_pdf_to_pinecone(
    pdf_path: str,
    index_name: str,
    namespace: str,
    batch_size: int = 100,
    chunk_size: int = 400,
    chunk_overlap: int = 30,
) -> Dict[str, Any]:
    source_name = os.path.basename(pdf_path)

    pages = load_pdf_pages(pdf_path)
    chunks = split_into_chunks(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = enrich_chunks(chunks, source_name=source_name)

    embedding = get_embeddings()
    dim = len(embedding.embed_query("dimension_check"))

    pc = get_pinecone()
    ensure_index_exists(pc, index_name=index_name, dimension=dim)
    index = get_index(pc, index_name)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    total = 0

    for d in chunks:
        texts.append(d.page_content)
        meta = dict(d.metadata)
        meta["chunk_text"] = d.page_content  # text_key 용
        metadatas.append(meta)

        if len(texts) >= batch_size:
            ids = [str(uuid4()) for _ in range(len(texts))]
            vecs = embedding.embed_documents(texts)
            index.upsert(vectors=zip(ids, vecs, metadatas), namespace=namespace)
            total += len(texts)
            texts, metadatas = [], []
            time.sleep(1)

    if texts:
        ids = [str(uuid4()) for _ in range(len(texts))]
        vecs = embedding.embed_documents(texts)
        index.upsert(vectors=zip(ids, vecs, metadatas), namespace=namespace)
        total += len(texts)

    return {
        "pdf_path": pdf_path,
        "pages": len(pages),
        "chunks": len(chunks),
        "upserted": total,
        "index_name": index_name,
        "namespace": namespace,
        "dimension": dim,
    }


# ---------------------------
# 5) 질문 → Top-k 검색 → 컨텍스트 결합
# ---------------------------
def build_vector_store(index_name: str) -> PineconeVectorStore:
    pc = get_pinecone()
    index = get_index(pc, index_name)
    embedding = get_embeddings()
    return PineconeVectorStore(index=index, embedding=embedding, text_key="chunk_text")


def retrieve_top_k(question: str, index_name: str, namespace: str, k: int) -> List[Document]:
    vs = build_vector_store(index_name)
    return vs.similarity_search(query=question, k=k, namespace=namespace)


def build_context_and_sources(docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for d in docs:
        if d.page_content:
            context_parts.append(d.page_content)
        sources.append(
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "chunk_id": d.metadata.get("chunk_id"),
            }
        )
    return "\n\n".join(context_parts).strip(), sources


# ---------------------------
# 6) “컨텍스트 내에서만” 답변 생성
# ---------------------------
def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate([
        ("system", "당신은 전문 분석가입니다. 질문에 대해 최적의 답을 해주세요."),
        ("human",
         """{question}에 대해 설명해 주세요.
아래의 context에서만 정보를 사용하고 다른 정보는 사용하지 마세요. 없으면 없다고 말해주세요.
context:
{context}""")
    ])


def answer_with_rag(question: str, index_name: str, namespace: str, k: int) -> Dict[str, Any]:
    docs = retrieve_top_k(question, index_name=index_name, namespace=namespace, k=k)
    context, sources = build_context_and_sources(docs)

    if not context:
        return {"answer": "알 수 없습니다.", "sources": sources}

    llm = get_llm()
    prompt = build_prompt()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": question, "context": context}).strip()
    if not answer:
        answer = "알 수 없습니다."

    return {"answer": answer, "sources": sources}


# ---------------------------
# 7) CLI(터미널 전용)
# ---------------------------
def main():
    load_dotenv(override=True) 

    parser = argparse.ArgumentParser(description="PDF RAG (Pinecone) - low_birth_rate.py (CLI only)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="PDF를 청킹/임베딩 후 Pinecone에 업서트")
    p_index.add_argument("--pdf", default=os.getenv("PDF_PATH", "./asiabrief_3-26.pdf"))
    p_index.add_argument("--index", default=os.getenv("INDEX_NAME", "wiki"))
    p_index.add_argument("--namespace", default=os.getenv("NAMESPACE", "wiki_ns1"))
    p_index.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "100")))
    p_index.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "400")))
    p_index.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "30")))

    p_ask = sub.add_parser("ask", help="질문을 받아 RAG로 답변(터미널 출력)")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--index", default=os.getenv("INDEX_NAME", "wiki"))
    p_ask.add_argument("--namespace", default=os.getenv("NAMESPACE", "wiki_ns1"))
    p_ask.add_argument("--top_k", type=int, default=int(os.getenv("TOP_K", "3")))

    args = parser.parse_args()

    if args.cmd == "index":
        if not os.path.exists(args.pdf):
            raise ValueError(f"File path {args.pdf} is not a valid file or url")
        info = index_pdf_to_pinecone(
            pdf_path=args.pdf,
            index_name=args.index,
            namespace=args.namespace,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print("[INDEX DONE]")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print("\n다음:")
        print(f'  python low_birth_rate.py ask "한국 저출산의 원인이 무엇입니까?" --top_k {args.top_k}')
        return

    if args.cmd == "ask":
        result = answer_with_rag(args.question, index_name=args.index, namespace=args.namespace, k=args.top_k)
        print("\n[ANSWER]")
        print(result["answer"])
        print("\n[SOURCES]")
        for s in result.get("sources", []):
            print(f"- source={s.get('source')} page={s.get('page')} chunk_id={s.get('chunk_id')}")
        return


if __name__ == "__main__":
    main()
