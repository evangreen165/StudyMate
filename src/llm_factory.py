from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

try:
    from .config import PROVIDER_DEFAULTS, get_provider
except ImportError:
    from config import PROVIDER_DEFAULTS, get_provider


def _physical_cores() -> int:
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        if cores:
            return cores
    except Exception:
        pass
    return 6


def resolve_provider(provider: str | None = None) -> str:
    return (provider or get_provider()).lower()


def resolve_model(
    provider: str | None = None, model_override: str | None = None
) -> str:
    name = resolve_provider(provider)
    defaults = PROVIDER_DEFAULTS[name]
    if name == "local":
        return Path(defaults["default_model"]).name
    return model_override or os.getenv("LLM_MODEL") or defaults["default_model"]


@lru_cache(maxsize=16)
def get_llm(
    provider: str | None = None, model_override: str | None = None
) -> BaseChatModel:
    name = resolve_provider(provider)
    if name not in PROVIDER_DEFAULTS:
        available = ", ".join(PROVIDER_DEFAULTS)
        raise ValueError(f"Неизвестный провайдер '{name}'. Доступны: {available}")

    defaults = PROVIDER_DEFAULTS[name]

    if name == "local":
        model_path = defaults["default_model"]
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Файл модели не найден: {model_path}. "
                "Скачайте GGUF-модель командой huggingface-cli из README."
            )
        return ChatLlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            n_threads=_physical_cores(),
            n_batch=512,
            n_ubatch=128,
            n_gpu_layers=0,
            temperature=0,
            max_tokens=200,
            stop=["<|im_end|>"],
            verbose=False,
        )

    env_key = defaults["env_key"]
    api_key = os.getenv(env_key) if env_key else None
    if not api_key:
        raise ValueError(
            f"Установите {env_key} в .env или выберите другого провайдера"
        )

    model = model_override or os.getenv("LLM_MODEL") or defaults["default_model"]
    headers = (
        {
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG Course Assistant",
        }
        if name == "openrouter"
        else {}
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=defaults["base_url"],
        temperature=0,
        max_tokens=512,
        default_headers=headers,
    )
