from __future__ import annotations

import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

try:
    from .llm_factory import get_llm, resolve_model, resolve_provider
except ImportError:
    from llm_factory import get_llm, resolve_model, resolve_provider

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "course_materials"
EMBEDDER_NAME = "intfloat/multilingual-e5-small"

TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDER_NAME),
    collection_name=COLLECTION_NAME,
)


def baseline_pure_llm(
    question: str, provider: str | None = None, model: str | None = None
) -> dict:
    llm = get_llm(provider, model_override=model)
    messages = [
        SystemMessage(content="Ты — полезный ассистент"),
        HumanMessage(content=f"Ответь на вопрос студента: {question}"),
    ]
    response = llm.invoke(messages)
    return {
        "answer": response.content,
        "sources": [],
        "provider": resolve_provider(provider),
        "model": resolve_model(provider, model),
    }


def baseline_vector_search(question: str) -> dict:
    documents = vectorstore.similarity_search(question, k=1)
    if not documents:
        return {"answer": "", "sources": [], "provider": None, "model": None}

    document = documents[0]
    return {
        "answer": document.page_content[:500],
        "sources": [document.metadata],
        "provider": None,
        "model": None,
    }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _get_bm25_index() -> tuple[BM25Okapi, list[str], list[dict]]:
    stored = vectorstore.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    tokenized = [tokenize(document) for document in documents]
    return BM25Okapi(tokenized), documents, metadatas


def baseline_bm25(question: str) -> dict:
    bm25, documents, metadatas = _get_bm25_index()
    tokens = tokenize(question)
    if not documents or not tokens:
        return {"answer": "", "sources": [], "provider": None, "model": None}

    scores = bm25.get_scores(tokens)
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    return {
        "answer": documents[best_index][:500],
        "sources": [metadatas[best_index]],
        "provider": None,
        "model": None,
    }
