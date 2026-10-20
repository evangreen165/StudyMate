from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import PROVIDER_DEFAULTS, get_provider
from src.rag_pipeline import answer_question

app = FastAPI(title="StudyMate RAG API")

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("studymate.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)


class QuestionRequest(BaseModel):
    question: str = ""
    provider: str | None = None
    model: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
    provider: str
    model: str


def is_configured(name: str, defaults: dict) -> bool:
    if name == "local":
        return Path(defaults["default_model"]).exists()
    env_key = defaults.get("env_key")
    return bool(env_key and os.getenv(env_key))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/providers")
def providers() -> dict:
    items = []
    for name, defaults in PROVIDER_DEFAULTS.items():
        items.append(
            {
                "name": name,
                "configured": is_configured(name, defaults),
                "default_model": defaults["default_model"],
                "kind": "local" if name == "local" else "cloud",
            }
        )
    return {"providers": items}


@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest) -> AnswerResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="Вопрос не должен быть пустым"
        )

    provider = (request.provider or get_provider()).lower()
    if provider not in PROVIDER_DEFAULTS:
        available = ", ".join(PROVIDER_DEFAULTS)
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный провайдер: {provider}. Доступны: {available}",
        )

    defaults = PROVIDER_DEFAULTS[provider]
    if not is_configured(provider, defaults):
        if provider == "local":
            detail = (
                "Провайдер local не настроен: файл модели не найден "
                f"({defaults['default_model']})"
            )
        else:
            detail = (
                f"Провайдер {provider} не настроен: "
                f"отсутствует {defaults['env_key']}"
            )
        raise HTTPException(status_code=400, detail=detail)

    started = time.perf_counter()
    try:
        result = answer_question(
            question, provider=request.provider, model=request.model
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Ошибка провайдера %s", provider)
        raise HTTPException(
            status_code=502, detail=f"Ошибка провайдера: {exc}"
        )

    elapsed = time.perf_counter() - started
    logger.info(
        "provider=%s model=%s question=%r elapsed=%.2fs sources=%d",
        result["provider"],
        result["model"],
        question[:200],
        elapsed,
        len(result["sources"]),
    )
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        provider=result["provider"],
        model=result["model"],
    )
