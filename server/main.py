from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .logging_config import setup_logging
from src.config import PROVIDER_DEFAULTS, get_provider
from src.dataset_manager import (
    _load_index,
    _save_index,
    aggregate_passport,
    delete_dataset,
    get_passport,
    index_dataset_from_source,
    list_datasets,
    register_dataset,
)
from src.upload_manager import (
    generate_dataset_id,
    save_uploaded_files,
    validate_batch,
)
from src.model_catalog import fetch_models, list_local_models
from src.rag_pipeline import answer_question

setup_logging()
logger = logging.getLogger("studymate.api")

app = FastAPI(title="StudyMate RAG API")

_MODELS_CACHE: dict[str, tuple[float, dict]] = {}
_MODELS_CACHE_TTL = 300


class QuestionRequest(BaseModel):
    question: str = ""
    provider: str | None = None
    model: str | None = None
    dataset_id: str | None = None
    max_tokens: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
    provider: str
    model: str
    dataset_id: str | None = None


class DatasetRequest(BaseModel):
    dataset_id: str
    name: str
    description: str = ""
    source_dir: str


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


@app.get("/models")
def models(provider: str = "local") -> dict:
    name = provider.lower()
    if name not in PROVIDER_DEFAULTS:
        available = ", ".join(PROVIDER_DEFAULTS)
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный провайдер: {name}. Доступны: {available}",
        )

    now = time.time()
    cached = _MODELS_CACHE.get(name)
    if cached and now - cached[0] < _MODELS_CACHE_TTL:
        return cached[1]

    if name == "local":
        payload = {"provider": "local", "models": list_local_models()}
    else:
        defaults = PROVIDER_DEFAULTS[name]
        env_key = defaults.get("env_key")
        api_key = os.getenv(env_key) if env_key else None
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=f"Провайдер {name} не настроен: отсутствует {env_key}",
            )
        try:
            items = fetch_models(name, api_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception(
                "models_error",
                extra={"provider": name, "error_type": type(exc).__name__},
            )
            raise HTTPException(
                status_code=502, detail=f"Ошибка каталога моделей: {exc}"
            )
        payload = {"provider": name, "models": items}

    _MODELS_CACHE[name] = (now, payload)
    return payload


@app.get("/datasets")
def datasets() -> dict:
    return {"datasets": list_datasets(), "aggregate": aggregate_passport()}


@app.get("/datasets/{dataset_id}/passport")
def dataset_passport(dataset_id: str) -> dict:
    try:
        return get_passport(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/datasets")
def create_dataset(request: DatasetRequest) -> dict:
    try:
        passport = register_dataset(
            dataset_id=request.dataset_id,
            name=request.name,
            description=request.description,
            source_dir=request.source_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "dataset_index_error",
            extra={
                "dataset_id": request.dataset_id,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=500, detail=f"Ошибка индексации датасета: {exc}"
        )
    logger.info(
        "dataset_created",
        extra={
            "dataset_id": request.dataset_id,
            "chunks": passport.get("chunks"),
            "status": "active",
        },
    )
    logger.info(
        "dataset_indexed",
        extra={
            "dataset_id": request.dataset_id,
            "chunks": passport.get("chunks"),
        },
    )
    return passport


@app.post("/datasets/upload")
def upload_dataset(
    dataset_name: str = Form(...),
    description: str = Form(""),
    files: list[UploadFile] = File(...),
) -> dict:
    name = dataset_name.strip()
    if not name:
        raise HTTPException(
            status_code=400, detail="Название датасета не должно быть пустым"
        )
    try:
        total_size = validate_batch(files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    dataset_id = generate_dataset_id(name)
    try:
        saved = save_uploaded_files(dataset_id, files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    index = _load_index()
    entry = {
        "id": dataset_id,
        "name": name,
        "description": description,
        "status": "uploaded",
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "file_hashes": [item["sha256"] for item in saved["files"]],
        "files": saved["files"],
    }
    index.setdefault("datasets", []).append(entry)
    _save_index(index)

    logger.info(
        "dataset_uploaded",
        extra={
            "dataset_id": dataset_id,
            "files_count": len(saved["files"]),
            "total_size": total_size,
        },
    )
    return {
        "dataset_id": dataset_id,
        "name": name,
        "status": "uploaded",
        "files": saved["files"],
        "warnings": saved["warnings"],
    }


@app.post("/datasets/{dataset_id}/index")
def index_dataset(dataset_id: str, force: bool = False) -> dict:
    index = _load_index()
    entry = next(
        (item for item in index.get("datasets", []) if item["id"] == dataset_id),
        None,
    )
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Датасет не найден: {dataset_id}"
        )
    if entry.get("status") == "indexed" and not force:
        raise HTTPException(
            status_code=409,
            detail="датасет уже проиндексирован, используйте force=true",
        )

    started = time.perf_counter()
    logger.info(
        "dataset_indexing_started",
        extra={"dataset_id": dataset_id, "force": force},
    )
    try:
        passport = index_dataset_from_source(
            dataset_id,
            entry.get("name", dataset_id),
            entry.get("description", ""),
            force=force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "dataset_index_error",
            extra={"dataset_id": dataset_id, "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=500, detail=f"Ошибка индексации: {exc}"
        )
    duration = round(time.perf_counter() - started, 2)
    logger.info(
        "dataset_indexing_finished",
        extra={
            "dataset_id": dataset_id,
            "duration_s": duration,
            "chunks": passport.get("chunks"),
        },
    )
    return passport


@app.get("/datasets/{dataset_id}/status")
def dataset_status(dataset_id: str) -> dict:
    index = _load_index()
    entry = next(
        (item for item in index.get("datasets", []) if item["id"] == dataset_id),
        None,
    )
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Датасет не найден: {dataset_id}"
        )
    return {
        "id": entry["id"],
        "status": entry.get("status"),
        "chunks": entry.get("chunks"),
        "indexed_at": entry.get("indexed_at"),
    }


@app.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: str, hard: bool = False) -> dict:
    try:
        entry = delete_dataset(dataset_id, hard=hard)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    logger.info(
        "dataset_deleted",
        extra={
            "dataset_id": dataset_id,
            "status": "deleted" if hard else "archived",
        },
    )
    return {"dataset": entry, "hard": hard}


@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest) -> AnswerResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="Вопрос не должен быть пустым"
        )

    provider = (request.provider or get_provider()).lower()
    logger.info(
        "chat_request",
        extra={
            "provider": provider,
            "model": request.model or "",
            "question_len": len(question),
            "question_preview": question[:200],
            "dataset_id": request.dataset_id or "",
            "max_tokens": request.max_tokens,
        },
    )

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
            question,
            provider=request.provider,
            model=request.model,
            dataset_id=request.dataset_id,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "chat_error",
            extra={
                "provider": provider,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=502, detail=f"Ошибка провайдера: {exc}"
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "chat_response",
        extra={
            "provider": result["provider"],
            "model": result["model"],
            "elapsed_ms": elapsed_ms,
            "sources_count": len(result["sources"]),
            "answer_len": len(result["answer"]),
            "dataset_id": result.get("dataset_id") or "",
        },
    )
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        provider=result["provider"],
        model=result["model"],
        dataset_id=result.get("dataset_id"),
    )
