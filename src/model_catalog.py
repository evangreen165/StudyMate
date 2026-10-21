from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import requests

MODEL_CATALOG_URLS = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "deepseek": "https://api.deepseek.com/v1/models",
}

QUANT_RE = re.compile(r"Q\d_K_[MS]|Q\d_\d|F16|F32", re.IGNORECASE)
_CACHE_TTL_SECONDS = 3600
_CACHE: dict[tuple[str, str | None], tuple[float, list[dict]]] = {}


def _api_key_hash(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _to_per_million(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def _normalize_openrouter(item: dict) -> dict:
    pricing = item.get("pricing") or {}
    prompt = pricing.get("prompt")
    completion = pricing.get("completion")
    model_id = str(item.get("id", ""))
    is_free = (
        (str(prompt) == "0" and str(completion) == "0")
        or model_id.endswith(":free")
    )
    return {
        "id": model_id,
        "name": str(item.get("name") or model_id),
        "context_length": item.get("context_length")
        or (item.get("top_provider") or {}).get("context_length"),
        "pricing": {
            "prompt_per_m": _to_per_million(prompt),
            "completion_per_m": _to_per_million(completion),
        },
        "is_free": is_free,
    }


def _normalize_openai_compatible(item: dict) -> dict:
    model_id = str(item.get("id", ""))
    return {
        "id": model_id,
        "name": str(item.get("name") or model_id),
        "context_length": item.get("context_length"),
        "pricing": None,
        "is_free": None,
    }


def fetch_models(
    provider: str, api_key: str | None = None, timeout: int = 10
) -> list[dict]:
    """Возвращает нормализованный каталог моделей.

    Для Groq/OpenAI/DeepSeek цены в /models обычно отсутствуют, поэтому
    pricing=None и is_free=None: цену нужно брать из документации провайдера
    или отдельного конфига.
    """
    provider = provider.lower()
    if provider not in MODEL_CATALOG_URLS:
        raise ValueError(f"Неизвестный провайдер каталога: {provider}")

    cache_key = (provider, _api_key_hash(api_key))
    cached = _CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    headers = {}
    if provider != "openrouter":
        if not api_key:
            raise ValueError(f"Провайдер {provider} требует ключ")
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(
        MODEL_CATALOG_URLS[provider], headers=headers, timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or []

    if provider == "openrouter":
        models = [_normalize_openrouter(item) for item in data]
    else:
        models = [_normalize_openai_compatible(item) for item in data]

    _CACHE[cache_key] = (time.time(), models)
    return models


def list_local_models(models_dir: str = "models") -> list[dict]:
    directory = Path(models_dir)
    if not directory.exists():
        return []

    models = []
    for path in sorted(directory.glob("*.gguf")):
        name = path.stem
        match = QUANT_RE.search(name)
        models.append(
            {
                "id": path.name,
                "name": name,
                "path": str(path.resolve()),
                "size_mb": round(path.stat().st_size / 1_000_000, 1),
                "quant": match.group(0).upper() if match else None,
                "is_free": True,
            }
        )
    return sorted(models, key=lambda item: item["size_mb"])
