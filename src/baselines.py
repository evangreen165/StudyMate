from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "course_materials"
EMBEDDER_NAME = "intfloat/multilingual-e5-small"
MODEL_PATH = "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
N_THREADS = 8
N_CTX = 8192

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
    verbose=False,
)


def baseline_pure_llm(question: str) -> dict:
    messages = [
        SystemMessage(content="Ты — полезный ассистент"),
        HumanMessage(content=f"Ответь на вопрос студента: {question}"),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "sources": []}


def baseline_tfidf(question: str) -> dict:
    documents = vectorstore.similarity_search(question, k=1)
    if not documents:
        return {"answer": "", "sources": []}

    document = documents[0]
    return {
        "answer": document.page_content[:500],
        "sources": [document.metadata],
    }
