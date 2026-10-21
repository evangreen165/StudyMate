# Проверка local-провайдера в контейнере

- Модель: `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (941 MiB, смонтирована в `/app/models` read-only).
- Запрос: `POST /chat` с `provider=local`, вопрос «Что такое рекурсия?».
- Результат: HTTP 200, `provider=local`, `model=Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`, 4 источника.
- Время ответа: **42.41 с** (первый запрос, включая загрузку GGUF в RAM).
- В логах: `chat_request` и `chat_response` с `provider=local`, `elapsed_ms=42078`, `sources_count=4`.
- Ответ релевантный, опирается на найденные фрагменты; ограничение — скорость CPU-инференса.
