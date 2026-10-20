from __future__ import annotations

import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "course_materials"
EMBEDDER_NAME = "intfloat/multilingual-e5-small"
MODEL_PATH = "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
N_THREADS = 6
N_CTX = 4096

TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDER_NAME),
    collection_name=COLLECTION_NAME,
)

llm = ChatLlamaCpp(
    model_path=MODEL_PATH,
    temperature=0,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=0,
    max_tokens=512,
    stop=["<|im_end|>"],
    verbose=False,
)


def baseline_pure_llm(question: str) -> dict:
    messages = [
        SystemMessage(content="Ты — полезный ассистент"),
        HumanMessage(content=f"Ответь на вопрос студента: {question}"),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "sources": []}


def baseline_vector_search(question: str) -> dict:
    documents = vectorstore.similarity_search(question, k=1)
    if not documents:
        return {"answer": "", "sources": []}

    document = documents[0]
    return {
        "answer": document.page_content[:500],
        "sources": [document.metadata],
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
        return {"answer": "", "sources": []}

    scores = bm25.get_scores(tokens)
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    return {
        "answer": documents[best_index][:500],
        "sources": [metadatas[best_index]],
    }
