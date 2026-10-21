from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "app.log"

EXTRA_FIELDS = (
    "provider",
    "model",
    "question_len",
    "question_preview",
    "elapsed_ms",
    "sources_count",
    "answer_len",
    "error_type",
    "dataset_id",
    "chunks",
    "status",
    "files_count",
    "total_size",
    "duration_s",
    "force",
    "max_tokens",
    "finish_reason",
    "usage",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "module": record.name,
            "event": record.getMessage(),
        }
        for key in EXTRA_FIELDS:
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["traceback"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = JsonFormatter()
    file_handler = RotatingFileHandler(
        LOG_PATH, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
