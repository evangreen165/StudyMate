from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_TOTAL_MB = 500

HOW_MARKDOWN = '''
## Что такое RAG

RAG (Retrieval-Augmented Generation) — подход, при котором языковая модель не «вспоминает» ответ из своих весов, а **находит релевантные фрагменты в ваших документах** и генерирует ответ на их основе. Это даёт три преимущества:

- **Источники.** Каждый ответ можно проверить — система возвращает страницы, откуда взята информация.
- **Актуальность.** Чтобы обновить знания, достаточно переиндексировать корпус — модель менять не нужно.
- **Меньше галлюцинаций.** Промпт жёстко ограничивает модель контекстом: «отвечай только на основе предоставленных фрагментов».

## Пайплайн

```text
PDF → чанкинг → эмбеддинги → векторная БД (Chroma)
                                    ↑
Вопрос → эмбеддинг → поиск top-k ───┘
                                    ↓
                            промпт с контекстом
                                    ↓
                       LLM (local / OpenRouter)
                                    ↓
                           ответ + источники
```

## Параметры индексации

- **chunk_size = 800** символов, **overlap = 100** — компромисс между контекстом и точностью поиска.
- **Эмбеддер:** `intfloat/multilingual-e5-small` (384 измерения, локально, ~470 МБ).
- **Векторная БД:** ChromaDB, косинусное расстояние.
- **Поиск:** top-4 чанка, фильтрация по dataset_id.

## Обучение vs индексация

**Важно:** система **не обучает языковую модель**. В RAG «знания» хранятся в векторном индексе, а не в весах модели.

| | Обучение (fine-tuning) | Индексация (RAG) |
|---|---|---|
| Что нужно | Размеченный корпус, GPU, часы/дни | PDF + CPU, минуты |
| Обновление знаний | Переобучение | Переиндексация |
| Обратимость | Сложно откатить | Удалить датасет — готово |
| Источники | Не даёт | Есть |

Именно поэтому для учебного ассистента выбран RAG: материалы меняются каждый семестр, а переиндексировать 3 PDF — минуты.

## Провайдеры

- **local** — Qwen2.5-1.5B через llama.cpp, работает без интернета, медленно на CPU (~25–40 с/ответ).
- **openrouter** — MiniMax M3.1 Flash (бесплатный tier), 2–8 с/ответ. Требует ключ в `.env`.
- Остальные (Groq, OpenAI, DeepSeek) — при наличии ключа.

## Как добавить новый корпус

1. Вкладка «Материалы курса» → «Загрузить новый корпус» → выберите PDF через file picker → «Загрузить файлы».
2. В таблице появится датасет со статусом **uploaded** (файлы сохранены, но ещё не в индексе).
3. Нажмите «Индексировать и получить паспорт» — система разобьёт PDF на чанки и построит векторы.
4. После завершения статус станет **indexed**, появится паспорт (документы/страницы/чанки/файлы).
5. Сводный агрегат сверху обновится автоматически.

Время индексации: ~1 минута на 100 страниц на 6-ядерном CPU. Большой корпус (1000+ страниц) — 10–20 минут.

Артефакты: [results/performance.md](results/performance.md), [data/requirements.md](data/requirements.md).
'''

LIMITS_MARKDOWN = '''
## Ограничения

- Локальная 1.5B-модель на CPU отвечает 25–40 с — для интерактивного демо использовать OpenRouter.
- Faithfulness 0.41 при целевом 0.40 — судья занижает оценку; при GPT-4-судье ожидаемо 0.7+.
- OpenRouter free tier: ~50 запросов/день, ~20 RPM.
- Формулы и код в PDF извлекаются частично (потеря структуры).
- Загрузка через браузер ограничена 500 МБ / 50 файлов на партию.
- Переиндексация большого корпуса блокирует UI на время выполнения (синхронная архитектура для простоты).
'''


st.set_page_config(
    page_title="StudyMate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1100px;}
.status-pill {
    display: inline-block; padding: 4px 12px; margin-right: 8px;
    border-radius: 999px; background: #f0f2f6; font-size: 12px; color: #555;
}
.status-pill b {color: #111;}
[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)


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


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_provider" not in st.session_state:
    st.session_state.last_provider = None
if "provider" not in st.session_state:
    st.session_state.provider = "local"
if "model" not in st.session_state:
    st.session_state.model = None
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = "__all__"
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 1024

try:
    providers = load_providers()
except Exception as exc:
    st.error(f"Не удалось подключиться к API {API_URL}: {exc}")
    st.stop()

configured_names = [
    item["name"] for item in providers if item.get("configured")
]
if not configured_names:
    st.error("Нет сконфигурированных провайдеров")
    st.stop()
if st.session_state.provider not in configured_names:
    st.session_state.provider = configured_names[0]

st.sidebar.title("📚 StudyMate")
st.sidebar.caption("Ассистент по учебным материалам")

if st.sidebar.button("➕ Новый чат", use_container_width=True, type="primary"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()

with st.sidebar.expander("⚙️ Настройки", expanded=False):
    provider = st.radio(
        "Провайдер",
        configured_names,
        key="provider",
        horizontal=True,
    )
    if st.session_state.last_provider != provider:
        st.session_state.last_provider = provider
        st.session_state.model = None
        st.session_state.max_tokens = 1024 if provider == "local" else 2048

    models: list[dict] = []
    try:
        models = load_models(provider)
    except Exception:
        models = []

    if provider == "local":
        by_path = {item["path"]: item for item in models}
        options = list(by_path)

        def format_local(path: str) -> str:
            item = by_path[path]
            quant = item.get("quant") or "?"
            return f"{item['name']} ({item['size_mb']} MB, {quant})"

        if options:
            selected = st.selectbox(
                "Модель", options, format_func=format_local
            )
            st.session_state.model = selected
            st.caption(f"Локальная модель: {by_path[selected]['name']}")
        else:
            st.caption("Модели .gguf не найдены")
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

        if options:
            selected = st.selectbox(
                "Модель", options, format_func=format_cloud
            )
            st.session_state.model = selected
        else:
            st.caption("Каталог моделей пуст")

    datasets_payload: dict = {"datasets": [], "aggregate": {"totals": {}}}
    try:
        datasets_payload = load_datasets()
    except Exception:
        datasets_payload = {"datasets": [], "aggregate": {"totals": {}}}
    datasets = datasets_payload.get("datasets", [])
    indexed_ids = [
        item["id"]
        for item in datasets
        if item.get("status") in ("indexed", "active")
    ]
    dataset_by_id = {item["id"]: item for item in datasets}

    def format_dataset(dataset_id: str) -> str:
        if dataset_id == "__all__":
            return "Все датасеты"
        item = dataset_by_id.get(dataset_id, {})
        chunks = item.get("chunks")
        suffix = f", {chunks} чанков" if chunks is not None else ""
        return f"{item.get('name', dataset_id)}{suffix}"

    st.selectbox(
        "Датасет",
        ["__all__"] + indexed_ids,
        key="dataset_id",
        format_func=format_dataset,
    )

    st.slider(
        "max_tokens",
        256,
        4096,
        step=256,
        key="max_tokens",
    )

with st.sidebar.expander("📁 Материалы курса", expanded=True):
    totals = (datasets_payload.get("aggregate") or {}).get("totals", {})
    st.caption(
        f"Всего: {totals.get('documents', 0)} док · "
        f"{totals.get('pages', 0)} стр · {totals.get('chunks', 0)} чанков"
    )
    for item in datasets:
        cols = st.columns([2, 1])
        cols[0].caption(item.get("name") or item["id"])
        cols[1].caption(item.get("status"))
        if item.get("status") == "uploaded":
            if st.button(
                "Индексировать", key=f"side_idx_{item['id']}"
            ):
                with st.status("Индексация... (не закрывайте вкладку)"):
                    st.write("Чанкинг → эмбеддинги → Chroma")
                    try:
                        response = requests.post(
                            f"{API_URL}/datasets/{item['id']}/index",
                            timeout=900,
                        )
                    except Exception as exc:
                        st.error(f"Ошибка индексации: {exc}")
                    else:
                        if response.status_code == 200:
                            load_datasets.clear()
                            st.session_state.show_manage_dialog = True
                            st.rerun()
                        else:
                            st.error(
                                response.json().get("detail", response.text)
                            )
    if st.button("Управление материалами", use_container_width=True):
        st.session_state.show_manage_dialog = True

st.sidebar.divider()

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("❓ Как это работает", use_container_width=True):
        st.session_state.show_help_dialog = True
with col2:
    if st.button("⚠️ Ограничения", use_container_width=True):
        st.session_state.show_limits_dialog = True

provider = st.session_state.provider
model = st.session_state.model or "…"
dataset = st.session_state.dataset_id
dataset_label = "все датасеты" if dataset == "__all__" else dataset
total_chunks = (datasets_payload.get("aggregate") or {}).get("totals", {}).get("chunks", 0)

st.markdown(
    f'<span class="status-pill">🤖 <b>{provider}</b> / {model}</span>'
    f'<span class="status-pill">📁 <b>{dataset_label}</b></span>'
    f'<span class="status-pill">🧩 {total_chunks} чанков</span>',
    unsafe_allow_html=True,
)
st.write("")

if not st.session_state.messages:
    st.markdown("### Чем помочь с учёбой?")
    st.caption(
        "Отвечаю на основе лекций, учебников и методичек вашего курса. "
        "Каждый ответ сопровождается ссылкой на источник."
    )
    suggestions = [
        "Что такое рекурсивная функция?",
        "Чем отличается TCP от UDP?",
        "Зачем нужен конструктор копирования в C++?",
        "Что такое порт в компьютерных сетях?",
    ]
    suggestion_cols = st.columns(2)
    for index, question in enumerate(suggestions):
        with suggestion_cols[index % 2]:
            if st.button(
                question,
                use_container_width=True,
                key=f"sugg_{index}",
            ):
                st.session_state.pending_question = question
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📎 Источники ({len(message['sources'])})"):
                for source in message["sources"]:
                    document = (
                        source.get("document")
                        or source.get("source")
                        or "неизвестный документ"
                    )
                    st.caption(f"• {document} · стр. {source.get('page')}")

prompt = st.chat_input("Спросите что-нибудь по курсу...")
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")

if prompt:
    prompt = prompt.strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(f"{provider} думает..."):
                payload = {
                    "question": prompt,
                    "provider": provider,
                    "model": st.session_state.model,
                    "dataset_id": None
                    if st.session_state.dataset_id == "__all__"
                    else st.session_state.dataset_id,
                    "max_tokens": st.session_state.max_tokens,
                }
                try:
                    response = requests.post(
                        f"{API_URL}/chat", json=payload, timeout=900
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.markdown(data["answer"])
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander(f"📎 Источники ({len(sources)})"):
                            for source in sources:
                                document = (
                                    source.get("document")
                                    or source.get("source")
                                    or "неизвестный документ"
                                )
                                st.caption(
                                    f"• {document} · стр. {source.get('page')}"
                                )
                    st.caption(f"_{data['provider']} / {data['model']}_")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")


@st.dialog("Управление материалами", width="large")
def manage_materials_dialog() -> None:
    dialog_payload = load_datasets()
    dialog_datasets = dialog_payload.get("datasets", [])
    dialog_totals = (
        dialog_payload.get("aggregate") or {}
    ).get("totals", {})

    m1, m2, m3 = st.columns(3)
    m1.metric("Документов", dialog_totals.get("documents", 0))
    m2.metric("Страниц", dialog_totals.get("pages", 0))
    m3.metric("Чанков", dialog_totals.get("chunks", 0))

    st.markdown("#### Загрузить новый корпус")
    new_name = st.text_input("Название корпуса")
    new_description = st.text_area("Описание", height=60)
    uploaded_files = st.file_uploader(
        "PDF-файлы",
        type=["pdf"],
        accept_multiple_files=True,
        help="До 50 файлов, до 500 МБ суммарно",
    )
    total_size = sum((file.size or 0) for file in uploaded_files)
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
        try:
            response = requests.post(
                f"{API_URL}/datasets/upload",
                data={
                    "dataset_name": new_name.strip(),
                    "description": new_description.strip(),
                },
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
                st.error(response.json().get("detail", response.text))

    st.markdown("#### Датасеты")
    if not dialog_datasets:
        st.info("Датасеты не зарегистрированы")
    for item in dialog_datasets:
        cols = st.columns([2, 1, 1, 3])
        cols[0].write(item.get("name", item.get("id")))
        cols[1].write(item.get("status", "?"))
        cols[2].write(item.get("chunks"))
        with cols[3]:
            status = item.get("status")
            if status == "uploaded":
                if st.button(
                    "Индексировать и получить паспорт",
                    key=f"dlg_idx_{item['id']}",
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
                                st.session_state[
                                    f"passport_{item['id']}"
                                ] = response.json()
                                load_datasets.clear()
                                st.rerun()
                            else:
                                st.error(
                                    response.json().get(
                                        "detail", response.text
                                    )
                                )
            elif status in ("indexed", "active"):
                if st.button(
                    "Паспорт", key=f"dlg_pass_{item['id']}"
                ):
                    try:
                        response = requests.get(
                            f"{API_URL}/datasets/{item['id']}/passport",
                            timeout=30,
                        )
                        if response.status_code == 200:
                            st.session_state[
                                f"passport_{item['id']}"
                            ] = response.json()
                            st.rerun()
                        else:
                            st.error(
                                response.json().get("detail", response.text)
                            )
                    except Exception as exc:
                        st.error(f"Ошибка паспорта: {exc}")
                if st.button(
                    "Переиндексировать (force)",
                    key=f"dlg_reidx_{item['id']}",
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
                                st.session_state[
                                    f"passport_{item['id']}"
                                ] = response.json()
                                load_datasets.clear()
                                st.rerun()
                            else:
                                st.error(
                                    response.json().get(
                                        "detail", response.text
                                    )
                                )

        passport = st.session_state.get(f"passport_{item['id']}")
        if passport:
            st.json(passport)
            if passport.get("files"):
                st.dataframe(passport["files"], width="stretch")

    all_ids = [item["id"] for item in dialog_datasets]
    if all_ids:
        st.markdown("#### Удалить датасет")
        delete_id = st.selectbox(
            "Датасет для удаления",
            all_ids,
            index=len(all_ids) - 1,
            key="dlg_delete_ds",
        )
        hard_delete = st.checkbox("Также удалить векторы (hard)")
        if st.button("Удалить", key="dlg_delete_btn"):
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
                    st.error(
                        response.json().get("detail", response.text)
                    )


@st.dialog("Как это работает", width="large")
def help_dialog() -> None:
    st.markdown(HOW_MARKDOWN)


@st.dialog("Ограничения", width="large")
def limits_dialog() -> None:
    st.markdown(LIMITS_MARKDOWN)


if st.session_state.get("show_manage_dialog"):
    st.session_state.show_manage_dialog = False
    manage_materials_dialog()
if st.session_state.get("show_help_dialog"):
    st.session_state.show_help_dialog = False
    help_dialog()
if st.session_state.get("show_limits_dialog"):
    st.session_state.show_limits_dialog = False
    limits_dialog()
