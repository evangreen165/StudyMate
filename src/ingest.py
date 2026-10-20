from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
MATERIALS_DIR = BASE_DIR / "data" / "course_materials"
PERSIST_DIR = BASE_DIR / "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDER_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "course_materials"

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Индексация учебных материалов в Chroma."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Удалить PERSIST_DIR перед построением индекса.",
    )
    return parser.parse_args()


def load_documents() -> list:
    pdf_paths = sorted(MATERIALS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"В папке {MATERIALS_DIR} нет PDF-файлов.")

    documents = []
    for pdf_path in pdf_paths:
        for page_index, document in enumerate(PyPDFLoader(str(pdf_path)).load()):
            document.metadata["source"] = pdf_path.name
            document.metadata["page"] = page_index + 1
            documents.append(document)
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def build_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        existing = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        existing.delete_collection()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )


def get_vector_dimension(
    vectorstore: Chroma, embeddings: HuggingFaceEmbeddings
) -> int:
    stored = vectorstore.get(limit=1, include=["embeddings"])
    vectors = stored.get("embeddings")
    if vectors is not None and len(vectors):
        return len(vectors[0])
    return len(embeddings.embed_query("размерность вектора"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    if args.rebuild and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print(f"Удалена директория: {PERSIST_DIR}")

    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_NAME)
    vectorstore = build_vectorstore(chunks, embeddings)
    dimension = get_vector_dimension(vectorstore, embeddings)

    unique_documents = len({chunk.metadata["source"] for chunk in chunks})
    print(f"Чанков: {len(chunks)}")
    print(f"Уникальных документов: {unique_documents}")
    print(f"Модель эмбеддингов: {EMBEDDER_NAME}")
    print(f"Размерность вектора: {dimension}")


if __name__ == "__main__":
    main()
