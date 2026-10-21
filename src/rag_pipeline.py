from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

try:
    from .dataset_manager import (
        dataset_status,
        get_archived_dataset_ids,
    )
    from .llm_factory import get_llm, resolve_model, resolve_provider
except ImportError:
    from dataset_manager import (
        dataset_status,
        get_archived_dataset_ids,
    )
    from llm_factory import get_llm, resolve_model, resolve_provider

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "course_materials"
EMBEDDER_NAME = "intfloat/multilingual-e5-small"
logger = logging.getLogger("studymate.api")

TOP_K = 4

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDER_NAME),
    collection_name=COLLECTION_NAME,
)
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — ассистент по учебным материалам курса. Отвечай ТОЛЬКО на основе "
            "предоставленного контекста. Если в контексте нет ответа — скажи: "
            '"В материалах курса это не освещено". Не придумывай информацию. '
            "Отвечай развёрнуто и полно: 3–7 предложений или структурированный список, "
            "если вопрос требует. Не обрывай ответ на середине. Если информации в "
            "контексте недостаточно для полного ответа — скажи, что именно не освещено, "
            "но не обрывай мысль.",
        ),
        (
            "human",
            "Контекст:\n{context}\n\nВопрос: {question}",
        ),
    ]
)


def answer_question(
    question: str,
    provider: str | None = None,
    model: str | None = None,
    dataset_id: str | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> dict:
    llm = get_llm(
        provider,
        model_override=model,
        max_tokens_override=max_tokens,
        api_key=api_key,
    )

    if dataset_id:
        status = dataset_status(dataset_id)
        if status not in ("indexed", "active"):
            raise ValueError(
                f"Датасет {dataset_id} не найден или архивирован"
            )
        search_filter = {"dataset_id": {"$eq": dataset_id}}
    else:
        archived = get_archived_dataset_ids()
        search_filter = (
            {"dataset_id": {"$nin": archived}} if archived else None
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K, "filter": search_filter}
    )
    chunks = retriever.invoke(question)
    context_text = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
    messages = PROMPT.format_messages(context=context_text, question=question)
    response = llm.invoke(messages)

    finish_reason = (response.response_metadata or {}).get("finish_reason")
    if finish_reason == "length":
        logger.warning(
            "response_truncated",
            extra={
                "provider": resolve_provider(provider),
                "max_tokens": max_tokens or None,
            },
        )

    metadata = response.response_metadata or {}
    usage = metadata.get("usage") or metadata.get("token_usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or metadata.get("tokens_evaluated")
    completion_tokens = usage.get("completion_tokens") or metadata.get("tokens_predicted")
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens is not None or completion_tokens is not None
            else None
        )
        usage_payload = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    else:
        usage_payload = None

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
        "dataset_id": dataset_id,
        "usage": usage_payload,
        "finish_reason": finish_reason,
    }
