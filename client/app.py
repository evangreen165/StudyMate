from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_TOTAL_MB = 500

st.set_page_config(page_title="StudyMate", layout="wide")
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


@st.cache_data(ttl=30)
def load_datasets() -> dict:
    response = requests.get(f"{API_URL}/datasets", timeout=20)
    response.raise_for_status()
    return response.json()


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

if provider == "local" and not models:
    st.sidebar.caption(
        f"Локальная модель: {Path(by_name[provider]['default_model']).name}"
    )
elif not by_name[provider]["configured"]:
    st.sidebar.warning("Провайдер не настроен: добавьте ключ в .env")

datasets_payload: dict = {"datasets": [], "aggregate": {"totals": {}}}
try:
    datasets_payload = load_datasets()
except Exception as exc:
    st.sidebar.warning(f"Не удалось загрузить датасеты: {exc}")

datasets = datasets_payload.get("datasets", [])
active_datasets = [item for item in datasets if item.get("status") in ("indexed", "active")]
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

tab_chat, tab_materials = st.tabs(["Чат", "Материалы курса"])

with tab_chat:
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
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
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
                                st.write(
                                    f"- {document} — стр. {source.get('page')}"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "meta": f"Ответ: {data['provider']} / {data['model']}",
                            "sources": sources,
                        }
                    )

with tab_materials:
    totals = (datasets_payload.get("aggregate") or {}).get("totals", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Документов", totals.get("documents", 0))
    col2.metric("Страниц", totals.get("pages", 0))
    col3.metric("Чанков", totals.get("chunks", 0))

    st.markdown("### Загрузить новый корпус")
    new_name = st.text_input("Название корпуса", placeholder="Например: Лекции по базам данных")
    new_description = st.text_area("Описание", height=60)
    uploaded_files = st.file_uploader(
        "PDF-файлы",
        type=["pdf"],
        accept_multiple_files=True,
        help="До 50 файлов, до 500 МБ суммарно",
    )
    total_size = sum((file.size or 0) for file in uploaded_files)
    st.caption(
        "После загрузки нажмите «Индексировать и получить паспорт» — индексация большого корпуса может занять несколько минут."
    )
    can_upload = (
        bool(uploaded_files)
        and bool(new_name.strip())
        and total_size <= MAX_TOTAL_MB * 1024 * 1024
    )
    if uploaded_files and not new_name.strip():
        st.warning("Введите название корпуса")
    if uploaded_files and total_size > MAX_TOTAL_MB * 1024 * 1024:
        st.error(f"Суммарный размер превышает {MAX_TOTAL_MB} МБ")

    if st.button("Загрузить файлы", disabled=not can_upload):
        file_parts = [
            ("files", (file.name, file.getbuffer(), "application/pdf"))
            for file in uploaded_files
        ]
        form_data = {"dataset_name": new_name.strip(), "description": new_description.strip()}
        try:
            response = requests.post(
                f"{API_URL}/datasets/upload",
                data=form_data,
                files=file_parts,
                timeout=300,
            )
        except Exception as exc:
            st.error(f"Ошибка загрузки: {exc}")
        else:
            if response.status_code == 200:
                st.success("Файлы загружены, статус: uploaded")
                load_datasets.clear()
                st.rerun()
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(detail)

    st.markdown("### Датасеты")
    if not datasets:
        st.info("Датасеты не зарегистрированы")
    for item in datasets:
        cols = st.columns([2, 1, 1, 3])
        cols[0].write(item.get("name", item.get("id")))
        cols[1].write(item.get("status", "?"))
        cols[2].write(item.get("chunks"))
        with cols[3]:
            status = item.get("status")
            if status == "uploaded":
                if st.button(
                    "Индексировать и получить паспорт",
                    key=f"idx_{item['id']}",
                ):
                    with st.status(
                        "Индексация... (может занять несколько минут)"
                    ):
                        st.write("Чанкинг → эмбеддинги → запись в Chroma")
                        try:
                            response = requests.post(
                                f"{API_URL}/datasets/{item['id']}/index",
                                timeout=900,
                            )
                        except Exception as exc:
                            st.error(f"Ошибка индексации: {exc}")
                        else:
                            if response.status_code == 200:
                                st.session_state[f"passport_{item['id']}"] = response.json()
                                load_datasets.clear()
                                st.rerun()
                            else:
                                try:
                                    detail = response.json().get("detail", response.text)
                                except Exception:
                                    detail = response.text
                                st.error(detail)
            elif status == "indexed" or status == "active":
                if st.button("Показать паспорт", key=f"pass_{item['id']}"):
                    try:
                        response = requests.get(
                            f"{API_URL}/datasets/{item['id']}/passport",
                            timeout=30,
                        )
                        if response.status_code == 200:
                            st.session_state[f"passport_{item['id']}"] = response.json()
                            st.rerun()
                        else:
                            st.error(response.json().get("detail", response.text))
                    except Exception as exc:
                        st.error(f"Ошибка паспорта: {exc}")
                if st.button(
                    "Переиндексировать (force)",
                    key=f"reidx_{item['id']}",
                ):
                    with st.status("Переиндексация..."):
                        st.write("Удаление старых векторов → повторная индексация")
                        try:
                            response = requests.post(
                                f"{API_URL}/datasets/{item['id']}/index",
                                params={"force": "true"},
                                timeout=900,
                            )
                        except Exception as exc:
                            st.error(f"Ошибка: {exc}")
                        else:
                            if response.status_code == 200:
                                st.session_state[f"passport_{item['id']}"] = response.json()
                                load_datasets.clear()
                                st.rerun()
                            else:
                                st.error(response.json().get("detail", response.text))

        passport = st.session_state.get(f"passport_{item['id']}")
        if passport:
            st.json(passport)
            if passport.get("files"):
                st.dataframe(passport["files"], width="stretch")

    st.markdown("### Удалить датасет")
    all_ids = [item["id"] for item in datasets]
    if all_ids:
        delete_id = st.selectbox(
            "Датасет для удаления",
            all_ids,
            index=len(all_ids) - 1,
            key="delete_ds",
        )
        hard_delete = st.checkbox("Также удалить векторы (hard)")
        if st.button("Удалить", key="delete_btn"):
            try:
                response = requests.delete(
                    f"{API_URL}/datasets/{delete_id}",
                    params={"hard": str(hard_delete).lower()},
                    timeout=120,
                )
            except Exception as exc:
                st.error(f"Ошибка удаления: {exc}")
            else:
                if response.status_code == 200:
                    st.success(f"Датасет {delete_id} удалён")
                    for key in list(st.session_state):
                        if key.startswith("passport_"):
                            del st.session_state[key]
                    load_datasets.clear()
                    st.rerun()
                else:
                    st.error(response.json().get("detail", response.text))
