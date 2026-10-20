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

Скопируйте `.env.example` в `.env` и укажите `OPENAI_API_KEY`.
