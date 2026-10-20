from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

try:
    from .llm_factory import get_llm, resolve_model, resolve_provider
except ImportError:
    from llm_factory import get_llm, resolve_model, resolve_provider

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "course_materials"
EMBEDDER_NAME = "intfloat/multilingual-e5-small"
TOP_K = 4

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDER_NAME),
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

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


def answer_question(
    question: str, provider: str | None = None, model: str | None = None
) -> dict:
    llm = get_llm(provider, model_override=model)
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
        "provider": resolve_provider(provider),
        "model": resolve_model(provider, model),
    }
