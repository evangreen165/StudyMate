from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from src.dataset_manager import DATASETS_DIR, _load_index

MAX_FILE_MB = 100
MAX_TOTAL_MB = 500
MAX_FILES = 50
ALLOWED_EXT = {".pdf"}
PDF_MAGIC = b"%PDF-"


def _validate_single(file: UploadFile) -> None:
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Файл {filename}: разрешены только PDF")

    head = file.file.read(5)
    file.file.seek(0)
    if head != PDF_MAGIC:
        raise ValueError(f"Файл {filename}: не PDF (magic bytes не совпадают)")


def _file_size(file: UploadFile) -> int:
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    return size


def validate_batch(files: list[UploadFile]) -> int:
    if not files:
        raise ValueError("Файлы не переданы")
    if len(files) > MAX_FILES:
        raise ValueError(f"Слишком много файлов: {len(files)} (макс {MAX_FILES})")

    for file in files:
        _validate_single(file)

    total = sum(_file_size(file) for file in files)
    if total > MAX_TOTAL_MB * 1024 * 1024:
        raise ValueError(
            f"Суммарный размер {total / 1e6:.1f} МБ превышает лимит {MAX_TOTAL_MB} МБ"
        )
    for file in files:
        size = _file_size(file)
        if size > MAX_FILE_MB * 1024 * 1024:
            raise ValueError(
                f"Файл {file.filename}: размер {size / 1e6:.1f} МБ "
                f"превышает лимит {MAX_FILE_MB} МБ"
            )
    return total


def _known_hashes() -> set[str]:
    index = _load_index()
    hashes = set()
    for item in index.get("datasets", []):
        hashes.update(item.get("file_hashes", []))
    return hashes


def save_uploaded_files(dataset_id: str, files: list[UploadFile]) -> dict:
    dataset_dir = DATASETS_DIR / dataset_id
    source_dir = dataset_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    known = _known_hashes()
    saved = []
    warnings = []
    for file in files:
        filename = Path(file.filename or "file.pdf").name
        hasher = hashlib.sha256()
        size = 0
        with open(source_dir / filename, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
                size += len(chunk)
        file.file.close()
        sha256 = hasher.hexdigest()
        if sha256 in known:
            warnings.append(f"{filename}: файл уже существует в другом датасете")
        else:
            known.add(sha256)
        saved.append({"name": filename, "size_bytes": size, "sha256": sha256})

    return {"files": saved, "warnings": warnings}


def generate_dataset_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")
    if not slug or len(slug) < 3:
        slug = "ds_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{timestamp}"
