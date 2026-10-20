# StudyMate

RAG-ассистент по учебным материалам курса: отвечает на вопросы по загруженным PDF-конспектам и лекциям с опорой на найденные фрагменты.

## Структура проекта

```
StudyMate/
├── data/
│   └── course_materials/   # учебные материалы курса (PDF и др.)
├── src/                    # логика RAG: загрузка, чанкинг, эмбеддинги, поиск
├── server/                 # FastAPI-сервер
├── client/                 # Streamlit-клиент
├── requirements.txt
├── .env.example
└── README.md
```

## Установка

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Скопируйте `.env.example` в `.env`; для локального стека ключи внешних API не требуются.

## Локальная модель

Для генерации ответов используется GGUF-модель Qwen2.5-1.5B-Instruct (Q4_K_M) и `llama-cpp-python`. Файл модели кладётся в `models/` и не коммитится в git:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Qwen2.5-1.5B-Instruct-GGUF', filename='Qwen2.5-1.5B-Instruct-Q4_K_M.gguf', local_dir='models')"
```

## Индексация корпуса

```bash
python src/ingest.py
python src/ingest.py --rebuild
```

Скрипт нарезает PDF на чанки по 800 символов с перекрытием 100, строит коллекцию `course_materials` в `chroma_db/` и сохраняет метаданные `source` и `page`. Эмбеддинги считаются локально моделью `intfloat/multilingual-e5-small` (384 измерения); OpenAI API не используется.
