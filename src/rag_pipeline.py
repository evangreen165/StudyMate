from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "course_materials"
EMBEDDER_NAME = "intfloat/multilingual-e5-small"
MODEL_PATH = "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
TOP_K = 4
N_THREADS = 6
N_CTX = 4096

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDER_NAME),
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

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

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — ассистент по учебным материалам курса. Отвечай ТОЛЬКО на основе "
            "предоставленного контекста. Если в контексте нет ответа — скажи: "
            '"В материалах курса это не освещено". Не придумывай информацию.',
        ),
        (
            "human",
            "Контекст:\n{context}\n\nВопрос: {question}",
        ),
    ]
)


def answer_question(question: str) -> dict:
    chunks = retriever.invoke(question)
    context_text = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
    messages = PROMPT.format_messages(context=context_text, question=question)
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [
            {
                "document": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            }
            for chunk in chunks
        ],
        "retrieved_chunks": [chunk.page_content for chunk in chunks],
    }
