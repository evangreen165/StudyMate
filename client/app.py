from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="StudyMate", layout="centered")
st.title("StudyMate")
st.caption("RAG-ассистент по учебным материалам курса")


@st.cache_data(ttl=30)
def load_providers() -> list[dict]:
    response = requests.get(f"{API_URL}/providers", timeout=10)
    response.raise_for_status()
    return response.json()["providers"]


@st.cache_data(ttl=300)
def load_models(provider: str) -> list[dict]:
    response = requests.get(
        f"{API_URL}/models", params={"provider": provider}, timeout=20
    )
    response.raise_for_status()
    return response.json().get("models", [])


@st.cache_data(ttl=60)
def load_datasets() -> list[dict]:
    response = requests.get(f"{API_URL}/datasets", timeout=20)
    response.raise_for_status()
    return response.json().get("datasets", [])


try:
    providers = load_providers()
except Exception as exc:
    st.error(f"Не удалось подключиться к API {API_URL}: {exc}")
    st.stop()

by_name = {provider["name"]: provider for provider in providers}
names = list(by_name)
configured_names = [name for name in names if by_name[name]["configured"]]
default_index = names.index(configured_names[0]) if configured_names else 0


def format_provider(name: str) -> str:
    item = by_name[name]
    if item["configured"]:
        return name
    return f"{name} (⚠ ключ не задан)"


provider = st.sidebar.radio(
    "Провайдер LLM",
    names,
    index=default_index,
    format_func=format_provider,
)

models: list[dict] = []
models_error = None
try:
    models = load_models(provider)
except Exception as exc:
    models_error = str(exc)

free_only = st.sidebar.checkbox(
    "Только бесплатные",
    value=False,
    disabled=provider == "local",
)
if provider != "local" and free_only:
    models = [item for item in models if item.get("is_free") is True]

model_value = ""
if models:
    if provider == "local":
        by_path = {item["path"]: item for item in models}
        options = list(by_path)

        def format_local(path: str) -> str:
            item = by_path[path]
            quant = item.get("quant") or "?"
            return f"{item['name']} ({item['size_mb']} MB, {quant})"

        selected = st.sidebar.selectbox(
            "Модель", options, format_func=format_local
        )
        model_value = selected
        st.sidebar.caption(f"Локальная модель: {by_path[selected]['name']}")
    else:
        by_id = {item["id"]: item for item in models}
        options = list(by_id)

        def format_cloud(model_id: str) -> str:
            item = by_id[model_id]
            context = item.get("context_length") or "?"
            if item.get("is_free"):
                return f"🟢 {model_id} (free, ctx={context})"
            pricing = item.get("pricing") or {}
            prompt = pricing.get("prompt_per_m")
            if prompt is not None:
                return f"{model_id} (ctx={context}, ${prompt:.2f}/M in)"
            return f"{model_id} (ctx={context}, цена не указана)"

        selected = st.sidebar.selectbox(
            "Модель", options, format_func=format_cloud
        )
        model_value = selected
else:
    if models_error:
        st.sidebar.warning(f"Не удалось загрузить каталог моделей: {models_error}")
    elif provider != "local" and free_only:
        st.sidebar.warning("Нет бесплатных моделей для выбранного провайдера")
    else:
        st.sidebar.warning("Каталог моделей пуст")
    model_value = st.sidebar.text_input(
        "Модель (введите вручную)",
        value=by_name[provider]["default_model"],
    )

if provider == "local" and models:
    pass
elif provider == "local":
    st.sidebar.caption(
        f"Локальная модель: {Path(by_name[provider]['default_model']).name}"
    )
elif not by_name[provider]["configured"]:
    st.sidebar.warning("Провайдер не настроен: добавьте ключ в .env")

datasets: list[dict] = []
try:
    datasets = load_datasets()
except Exception as exc:
    st.sidebar.warning(f"Не удалось загрузить датасеты: {exc}")

active_datasets = [item for item in datasets if item.get("status") == "active"]
dataset_by_id = {item["id"]: item for item in active_datasets}
dataset_options = ["__all__"] + list(dataset_by_id)


def format_dataset(dataset_id: str) -> str:
    if dataset_id == "__all__":
        return "Все датасеты"
    item = dataset_by_id.get(dataset_id, {})
    chunks = item.get("chunks")
    suffix = f", {chunks} чанков" if chunks is not None else ""
    return f"{item.get('name', dataset_id)}{suffix}"


selected_dataset = st.sidebar.selectbox(
    "Датасет", dataset_options, format_func=format_dataset
)
dataset_id = None if selected_dataset == "__all__" else selected_dataset

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])
        if message.get("sources"):
            with st.expander("Источники"):
                for source in message["sources"]:
                    document = (
                        source.get("document")
                        or source.get("source")
                        or "неизвестный документ"
                    )
                    st.write(f"- {document} — стр. {source.get('page')}")

question = st.chat_input(
    "Вопрос по материалам курса",
    disabled=not by_name[provider]["configured"],
)

if question:
    question = question.strip()
    if not question:
        st.warning("Введите вопрос.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        payload = {
            "question": question,
            "provider": provider,
            "model": model_value.strip() or None,
            "dataset_id": dataset_id,
        }

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat", json=payload, timeout=180
                    )
                except Exception as exc:
                    st.error(f"Ошибка соединения с API: {exc}")
                    st.stop()

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(detail)
            else:
                data = response.json()
                st.markdown(data["answer"])
                st.caption(f"Ответ: {data['provider']} / {data['model']}")

                sources = data.get("sources", [])
                if sources:
                    with st.expander("Источники"):
                        for source in sources:
                            document = (
                                source.get("document")
                                or source.get("source")
                                or "неизвестный документ"
                            )
                            st.write(f"- {document} — стр. {source.get('page')}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "meta": f"Ответ: {data['provider']} / {data['model']}",
                        "sources": sources,
                    }
                )
