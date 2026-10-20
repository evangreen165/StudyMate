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
model_value = st.sidebar.text_input(
    "Модель (опционально, для облачных)",
    disabled=provider == "local",
)

if provider == "local":
    model_name = Path(by_name[provider]["default_model"]).name
    st.sidebar.caption(f"Локальная модель: {model_name}")
elif not by_name[provider]["configured"]:
    st.sidebar.warning("Провайдер не настроен: добавьте ключ в .env")

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
