# StudyMate

RAG-ассистент по учебным материалам курса: отвечает на вопросы по загруженным PDF-конспектам и лекциям с опорой на найденные фрагменты.

## Структура проекта

```
StudyMate/
├── data/
│   └── course_materials/   # учебные материалы курса (PDF и др.)
├── models/                 # локальные GGUF-модели (не коммитятся)
├── results/                # отчёты smoke-тестов и сравнений
├── src/                    # RAG, индексация, провайдеры LLM, baseline
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

Скопируйте `.env.example` в `.env` и заполните нужные ключи. `.env` не коммитится.

## Выбор LLM-провайдера

Провайдер задаётся переменной `LLM_PROVIDER` в `.env` (по умолчанию `local`) или параметром `provider` в API/UI. Модель можно переопределить через `LLM_MODEL` или параметр `model` (для облачных провайдеров).

| Провайдер | Модель по умолчанию | Переменная с ключом | Где получить ключ |
| --- | --- | --- | --- |
| `local` | `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | — | модель скачивается локально |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` | platform.openai.com |
| `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | console.groq.com |
| `openrouter` | `minimax/minimax-m3.1-flash:free` | `OPENROUTER_API_KEY` | openrouter.ai/keys |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | platform.deepseek.com |

OpenRouter free tier: примерно 20 запросов в минуту и 50 в сутки. Бесплатный доступ к `minimax/minimax-m3.1-flash:free` может быть временным; для стабильной работы используйте `local` или платный тариф.

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

Скрипт нарезает PDF на чанки по 800 символов с перекрытием 100, строит коллекцию `course_materials` в `chroma_db/` и сохраняет метаданные `source` и `page`. Эмбеддинги считаются локально моделью `intfloat/multilingual-e5-small` (384 измерения).

## Запуск API и UI

```bash
uvicorn server.main:app --reload
streamlit run client/app.py
```

По умолчанию клиент обращается к `http://127.0.0.1:8000`; адрес можно изменить переменной `API_URL`.

Проверка API:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/providers
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Что такое рекурсия?\", \"provider\": \"openrouter\"}"
```

Логи запросов пишутся в `server/logs/app.log` (не коммитится).

## Проверка провайдеров

```bash
python src/smoke_test.py --provider local
python src/smoke_test.py --provider openrouter
```

Отчёты сохраняются в `results/smoke_test_<provider>_<model>.md`; сводное сравнение — в `results/smoke_test_comparison.md`.
## Оценка качества

RAGAS 0.4.3 не установился в этом окружении: зависимость `scikit-network` требует MSVC, а импорт несовместим с текущей `langchain-community`. Поэтому используется lightweight-эквивалент `src/evaluate_light.py`:

- **Faithfulness** — локальный судья Qwen2.5-1.5B разбивает ответ на утверждения и проверяет их по контексту (0–1).
- **Answer relevance** — тот же судья оценивает соответствие ответа вопросу по шкале 1–5.
- **Context recall** — доля ключевых слов эталонного ответа, найденных в контексте (0–1).

```bash
python src/evaluate_light.py --split val --provider local --judge local
python src/evaluate_light.py --split test --provider local --judge local
python src/evaluate_light.py --split test --provider openrouter --judge local --limit 18
python src/evaluate_baselines.py
```

Отчёты сохраняются в `results/eval_<split>_<provider>_judge<judge>.md`; промежуточные JSON не коммитятся. Судья local слабый, а context recall не учитывает синонимы и морфологию — метрики следует рассматривать как ориентировочные.
