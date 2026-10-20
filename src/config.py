from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "local").lower()


PROVIDER_DEFAULTS = {
    "local": {
        "base_url": None,
        "default_model": os.getenv(
            "LOCAL_MODEL_PATH", "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        ),
        "env_key": None,
    },
    "openai": {
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "minimax/minimax-m3.1-flash:free",
        "env_key": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
}
