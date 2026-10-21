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

## Качество и ограничения

Целевые метрики и их обоснование — в `data/requirements.md`. Faithfulness на тестовой выборке — 0.41 (local 0.408, OpenRouter 0.422) при цели ≥ 0.40. Локальный судья Qwen2.5-1.5B склонен занижать faithfulness, а context recall считается по ключевым словам, поэтому метрики ориентировочные.
## Каталог моделей

`GET /models?provider=<name>` возвращает нормализованный каталог:

- `local` — файлы `models/*.gguf` с полями `id`, `name`, `path`, `size_mb`, `quant`, `is_free`.
- `openrouter` — модели OpenRouter без ключа (поля `id`, `name`, `context_length`, `pricing`, `is_free`).
- `openai`, `groq`, `deepseek` — требуют соответствующий ключ; цены в `/models` обычно отсутствуют, поэтому `pricing=None`, `is_free=None` (цена не указана).

Ответ кешируется на 5 минут; каталог OpenRouter — на 1 час. В UI модели выбираются через selectbox, есть фильтр «Только бесплатные».

## Управление датасетами

- `GET /datasets` — список датасетов и агрегированный паспорт.
- `GET /datasets/{id}/passport` — паспорт датасета.
- `POST /datasets` — JSON `{"dataset_id", "name", "description", "source_dir"}`; `source_dir` — путь на сервере с PDF.
- `DELETE /datasets/{id}?hard=false` — soft delete (архивирование), `hard=true` — удаление чанков и папки.

Паспорта хранятся в `data/datasets/<id>/passport.json`, индекс — в `data/datasets/_index.json`. RAG-поиск фильтруется по `dataset_id`; при выборе «Все датасеты» архивные исключаются.

Через UI доступна загрузка PDF через file picker; `source_dir` (регистрация по пути на сервере) остаётся альтернативой для API/Docker-сценариев.

## Загрузка нового корпуса (через UI)

1. Вкладка «Материалы курса» → «Загрузить новый корпус» → название, описание, PDF через file picker (до 50 файлов / 500 МБ) → «Загрузить файлы».
2. Датасет появляется со статусом **uploaded** (файлы сохранены, индекса нет).
3. Кнопка «Индексировать и получить паспорт» → чанкинг (chunk_size=800, overlap=100) → эмбеддинги → Chroma → паспорт.
4. Статус становится **indexed**, агрегат сверху обновляется.
5. «Переиндексировать (force)» пересоздаёт векторы; «Удалить» — soft/hard.

Время индексации: ~1 минута на 100 страниц на 6-ядерном CPU.

## Интерфейс

- **Чат — главный экран.** Статус-строка сверху (провайдер / модель / датасет / число чанков), приветственный экран с кнопками-подсказками, история сообщений с источниками.
- **Sidebar:** «➕ Новый чат», свёрнутые «⚙️ Настройки» (провайдер, модель из `/models`, датасет, slider `max_tokens` 256–4096), развёрнутые «📁 Материалы курса» (компактная сводка и кнопка индексации для uploaded), кнопки «❓ Как это работает» и «⚠️ Ограничения» — открывают диалоги `st.dialog`.
- Значения настроек хранятся в `st.session_state` и передаются в `/chat` (`max_tokens` тоже).

## Развёртывание

### Быстрый старт (Docker, одна команда)

1. Скопируй `.env.example` в `.env` и заполни хотя бы один ключ:

   ```bash
   cp .env.example .env
   # OPENROUTER_API_KEY=sk-or-... (бесплатный tier, 50 запросов/день)
   ```

2. Скачай локальную модель (если планируешь провайдер local):

   ```bash
   huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --local-dir models/
   ```

3. Запусти:

   ```bash
   docker compose up --build -d
   ```

4. Открой:

   - Streamlit UI: http://localhost:8501
   - FastAPI Swagger: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

5. Логи: `docker compose logs -f`.

Образ собран на основе runtime-зависимостей (`requirements.txt`); dev-зависимости (`requirements-dev.txt`) в образ не входят.

### Локальный запуск (без Docker)

```bash
python -m uvicorn server.main:app --port 8000
streamlit run client/app.py --server.port 8501
```

## Архитектура контейнера

- Один образ `rag-course-assistant:latest`.
- `chroma_db/` и `data/datasets/` — volumes (`./chroma_db:/app/chroma_db`, `./data/datasets:/app/data/datasets`); в образе — seed-копии (`/app/_seed/...`), `entrypoint.sh` копирует их в volume при первом старте.
- `models/*.gguf` — volume `./models:/app/models:ro` (не в образе).
- `server/logs/` — volume, логи сохраняются на хосте.
- `.env` — `env_file`, никогда не попадает в образ.
- Entrypoint запускает uvicorn (8000) и streamlit (8501) параллельно.
- `data/datasets/` — volume `./data/datasets:/app/data/datasets` (паспорта и новые датасеты переживают перезапуск).
- Размер образа: 4.20 ГБ; запуск до healthy — 21.64 с.

## Производительность

Подробные замеры — в `results/performance.md`. Кратко: p50 `/chat` через OpenRouter — 2.741 с, RPS при concurrency=3 — 0.658, RAM контейнера — 1.92 GiB; запуск до healthy — 21.64 с. Образ 4.20 ГБ превышает целевые 2.5 ГБ — ограничение зафиксировано в отчёте.


## Как это работает

Подробное объяснение — во вкладке «Как это работает» Streamlit. Кратко: RAG (Retrieval-Augmented Generation) не обучает модель, а ищет релевантные чанки в векторном индексе Chroma (эмбеддер `intfloat/multilingual-e5-small`, top-4, фильтр по `dataset_id`) и отдаёт их LLM вместе с промптом «отвечай только на основе контекста». Знания обновляются переиндексацией, а не переобучением.

## Ограничения

Сводка — во вкладке «Ограничения» Streamlit. Ключевое: локальная 1.5B-модель на CPU отвечает 25–40 с; Faithfulness 0.41 (судья занижает); OpenRouter free tier ~50 запросов/день; формулы/код из PDF извлекаются частично; загрузка ограничена 500 МБ / 50 файлов; переиндексация большого корпуса синхронно блокирует UI.


## Метрики после переоценки (v2)

После увеличения `max_tokens` (local=1024, облачные=2048) local-показатели на тестовой выборке: Faithfulness 0.461, Answer Relevance 4.39, Context Recall 0.521. Целевые пороги сохранены (Faithfulness ≥ 0.40). Детали: `results/test_summary_v2.md`, `data/requirements.md`.

## Устранение неполадок

**«Не вижу новые элементы UI (провайдер, ключ, фильтр free, тему, max_tokens, статистику, чаты).»**

Вероятная причина — устаревший Docker-образ: `docker compose up -d` без `--build` использует старый образ. Обновите и обновите страницу с Ctrl+F5:

```bash
docker compose down
docker compose up --build -d
```

Затем проверьте, что в контейнере действительно новый код:

```bash
docker compose exec rag-assistant grep -n "Провайдер\|free_only\|Тёмная тема\|max_tokens\|Чаты" /app/client/app.py
```

При локальном запуске: `streamlit run client/app.py` должен указывать на тот же корень проекта `D:\K\system(StudyMate)`.

**Другие частые проблемы**

- История чатов хранится в `st.session_state` и очищается при перезапуске Streamlit — это ожидаемое поведение.
- Локальная 1.5B-модель на CPU может не успеть закончить ответ за `max_tokens` (в логах появляется `response_truncated`) — увеличьте `max_tokens` в настройках (до 4096).
- API-ключ из UI передаётся только в текущий запрос и не сохраняется между перезапусками.

