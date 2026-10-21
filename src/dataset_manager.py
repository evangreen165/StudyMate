from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "data" / "datasets"
INDEX_PATH = DATASETS_DIR / "_index.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "course_materials"
EMBEDDER_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
DATASET_ID_RE = re.compile(r"^[a-z0-9_-]{3,40}$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text('{"datasets": []}\n', encoding="utf-8")


def _load_index() -> dict:
    ensure_dirs()
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(index: dict) -> None:
    INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _dataset_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def _passport_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / "passport.json"


def _relative_passport(dataset_id: str) -> str:
    return _passport_path(dataset_id).relative_to(BASE_DIR).as_posix()


def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def _migrate_legacy_metadata() -> None:
    try:
        collection = _collection()
    except Exception:
        return
    sample = collection.get(limit=1, include=["metadatas"])
    metadatas = sample.get("metadatas") or []
    if metadatas and metadatas[0] and "dataset_id" in metadatas[0]:
        return

    stored = collection.get(include=["metadatas"])
    ids = stored.get("ids") or []
    metadatas = stored.get("metadatas") or []
    for start in range(0, len(ids), 500):
        batch_ids = ids[start : start + 500]
        batch_metas = [
            {
                **(meta or {}),
                "dataset_id": "course_materials",
                "dataset_name": "Учебные материалы курса",
            }
            for meta in metadatas[start : start + 500]
        ]
        if batch_ids:
            collection.update(ids=batch_ids, metadatas=batch_metas)


def ensure_legacy_dataset() -> dict:
    ensure_dirs()
    index = _load_index()
    entry = next(
        (item for item in index.get("datasets", []) if item["id"] == "course_materials"),
        None,
    )
    if entry is None:
        passport_file = _passport_path("course_materials")
        if not passport_file.exists():
            raise FileNotFoundError(
                f"Нет паспорта для course_materials: {passport_file}"
            )
        passport = json.loads(passport_file.read_text(encoding="utf-8"))
        entry = {
            "id": "course_materials",
            "name": passport.get("name", "Учебные материалы курса"),
            "description": passport.get("description", ""),
            "created_at": passport.get("created_at", _now()),
            "passport_path": _relative_passport("course_materials"),
            "status": "indexed",
        }
        index.setdefault("datasets", []).append(entry)
        _save_index(index)
    _migrate_legacy_metadata()
    return entry


def _tokenize_quant(filename: str) -> str | None:
    match = re.search(r"Q\d_K_[MS]|Q\d_\d|F16|F32", filename, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _build_passport(
    dataset_id: str,
    name: str,
    description: str,
    documents: list,
    chunks: list,
) -> dict:
    files: dict[str, dict] = {}
    total_chars = 0
    for document in documents:
        source = document.metadata.get("source", "unknown.pdf")
        chars = len(document.page_content)
        total_chars += chars
        item = files.setdefault(
            source, {"name": source, "pages": 0, "chars": 0, "chunks": 0}
        )
        item["pages"] += 1
        item["chars"] += chars
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown.pdf")
        files.setdefault(
            source, {"name": source, "pages": 0, "chars": 0, "chunks": 0}
        )["chunks"] += 1

    pages = len(documents)
    return {
        "dataset_id": dataset_id,
        "name": name,
        "description": description,
        "created_at": _now(),
        "documents": len(files),
        "pages": pages,
        "chars": total_chars,
        "chunks": len(chunks),
        "avg_page_chars": round(total_chars / pages, 1) if pages else 0.0,
        "files": sorted(files.values(), key=lambda item: item["name"]),
    }


def _passport_from_chunks(dataset_id: str, name: str, description: str) -> dict:
    collection = _collection()
    stored = collection.get(
        where={"dataset_id": {"$eq": dataset_id}},
        include=["documents", "metadatas"],
    )
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if not documents:
        raise ValueError(f"В индексе нет чанков для датасета {dataset_id}")

    files: dict[str, dict] = {}
    pages_by_file: dict[str, set] = {}
    for text, meta in zip(documents, metadatas):
        source = (meta or {}).get("source", "unknown.pdf")
        page = (meta or {}).get("page", 0)
        item = files.setdefault(
            source, {"name": source, "pages": 0, "chars": 0, "chunks": 0}
        )
        item["chars"] += len(text)
        item["chunks"] += 1
        pages_by_file.setdefault(source, set()).add(page)
    for source, pages in pages_by_file.items():
        files[source]["pages"] = len(pages)

    total_chars = sum(item["chars"] for item in files.values())
    pages = sum(item["pages"] for item in files.values())
    return {
        "dataset_id": dataset_id,
        "name": name,
        "description": description,
        "created_at": _now(),
        "documents": len(files),
        "pages": pages,
        "chars": total_chars,
        "chunks": len(documents),
        "avg_page_chars": round(total_chars / pages, 1) if pages else 0.0,
        "files": sorted(files.values(), key=lambda item: item["name"]),
    }


def register_dataset(
    dataset_id: str,
    name: str,
    description: str,
    source_dir: str,
    skip_indexing: bool = False,
) -> dict:
    if not DATASET_ID_RE.match(dataset_id):
        raise ValueError(
            "dataset_id должен содержать только [a-z0-9_-] и иметь длину 3–40"
        )
    ensure_dirs()
    index = _load_index()
    existing = next(
        (item for item in index.get("datasets", []) if item["id"] == dataset_id),
        None,
    )
    if existing is not None:
        if skip_indexing and _passport_path(dataset_id).exists():
            return json.loads(_passport_path(dataset_id).read_text(encoding="utf-8"))
        raise ValueError(f"Датасет {dataset_id} уже существует")

    dataset_dir = _dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if skip_indexing:
        if dataset_id == "course_materials":
            ensure_legacy_dataset()
            passport = json.loads(_passport_path(dataset_id).read_text(encoding="utf-8"))
        else:
            passport = _passport_from_chunks(dataset_id, name, description)
    else:
        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"source_dir не найден: {source_dir}")
        pdfs = sorted(source_path.glob("*.pdf"))
        if not pdfs:
            raise ValueError(f"В {source_dir} нет PDF-файлов")

        source_copy = dataset_dir / "source"
        source_copy.mkdir(parents=True, exist_ok=True)
        documents = []
        for pdf in pdfs:
            shutil.copy2(pdf, source_copy / pdf.name)
            for page_index, document in enumerate(PyPDFLoader(str(pdf)).load()):
                document.metadata["source"] = pdf.name
                document.metadata["page"] = page_index + 1
                document.metadata["dataset_id"] = dataset_id
                document.metadata["dataset_name"] = name
                documents.append(document)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
        )
        chunks = [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_NAME)
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        vectorstore.add_documents(chunks)
        passport = _build_passport(dataset_id, name, description, documents, chunks)

    _passport_path(dataset_id).write_text(
        json.dumps(passport, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    entry = {
        "id": dataset_id,
        "name": name,
        "description": description,
        "created_at": passport.get("created_at", _now()),
        "passport_path": _relative_passport(dataset_id),
        "status": "indexed",
    }
    if existing is None:
        index.setdefault("datasets", []).append(entry)
    else:
        index["datasets"] = [
            entry if item["id"] == dataset_id else item
            for item in index["datasets"]
        ]
    _save_index(index)
    return passport


def get_passport(dataset_id: str) -> dict:
    path = _passport_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Паспорт не найден: {dataset_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_datasets() -> list[dict]:
    ensure_dirs()
    index = _load_index()
    datasets = []
    for item in index.get("datasets", []):
        entry = dict(item)
        try:
            passport = get_passport(item["id"])
            entry.update(
                {
                    "documents": passport.get("documents"),
                    "pages": passport.get("pages"),
                    "chunks": passport.get("chunks"),
                    "chars": passport.get("chars"),
                }
            )
        except FileNotFoundError:
            pass
        datasets.append(entry)
    return datasets


def aggregate_passport() -> dict:
    datasets = [
        item for item in list_datasets()
        if item.get("status") in ("indexed", "active")
    ]
    totals = {
        "datasets": len(datasets),
        "documents": sum(item.get("documents") or 0 for item in datasets),
        "pages": sum(item.get("pages") or 0 for item in datasets),
        "chars": sum(item.get("chars") or 0 for item in datasets),
        "chunks": sum(item.get("chunks") or 0 for item in datasets),
    }
    return {"totals": totals, "datasets": datasets}


def index_dataset_from_source(
    dataset_id: str, name: str, description: str, force: bool = False
) -> dict:
    dataset_dir = _dataset_dir(dataset_id)
    source_dir = dataset_dir / "source"
    if not source_dir.exists():
        raise ValueError(f"Нет загруженных файлов для датасета {dataset_id}")
    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"В {source_dir} нет PDF-файлов")

    if force:
        try:
            _collection().delete(where={"dataset_id": {"$eq": dataset_id}})
        except Exception:
            pass

    documents = []
    for pdf in pdfs:
        for page_index, document in enumerate(PyPDFLoader(str(pdf)).load()):
            document.metadata["source"] = pdf.name
            document.metadata["page"] = page_index + 1
            document.metadata["dataset_id"] = dataset_id
            document.metadata["dataset_name"] = name
            documents.append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_NAME)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.add_documents(chunks)
    passport = _build_passport(dataset_id, name, description, documents, chunks)

    passport_path = _passport_path(dataset_id)
    passport_path.parent.mkdir(parents=True, exist_ok=True)
    passport_path.write_text(
        json.dumps(passport, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    index = _load_index()
    entries = index.get("datasets", [])
    for item in entries:
        if item["id"] == dataset_id:
            item["status"] = "indexed"
            item["indexed_at"] = passport.get("created_at")
            item["name"] = name
            item["description"] = description
            item["passport_path"] = _relative_passport(dataset_id)
            item["documents"] = passport.get("documents")
            item["pages"] = passport.get("pages")
            item["chunks"] = passport.get("chunks")
    _save_index(index)
    return passport


def delete_dataset(dataset_id: str, hard: bool = False) -> dict:
    index = _load_index()
    entry = next(
        (item for item in index.get("datasets", []) if item["id"] == dataset_id),
        None,
    )
    if entry is None:
        raise ValueError(f"Датасет не найден: {dataset_id}")

    if hard:
        try:
            collection = _collection()
            collection.delete(where={"dataset_id": {"$eq": dataset_id}})
        except Exception:
            pass
        shutil.rmtree(_dataset_dir(dataset_id), ignore_errors=True)
        index["datasets"] = [
            item for item in index["datasets"] if item["id"] != dataset_id
        ]
    else:
        entry = {**entry, "status": "archived"}
        index["datasets"] = [
            entry if item["id"] == dataset_id else item
            for item in index["datasets"]
        ]
    _save_index(index)
    return entry


def get_active_dataset_ids() -> list[str]:
    return [
        item["id"]
        for item in list_datasets()
        if item.get("status") in ("indexed", "active")
    ]


def get_archived_dataset_ids() -> list[str]:
    return [
        item["id"]
        for item in list_datasets()
        if item.get("status") == "archived"
    ]


def dataset_status(dataset_id: str) -> str | None:
    entry = next(
        (item for item in list_datasets() if item["id"] == dataset_id), None
    )
    return entry.get("status") if entry else None
