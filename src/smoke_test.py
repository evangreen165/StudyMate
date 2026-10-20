from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

try:
    from baselines import (
        baseline_bm25,
        baseline_pure_llm,
        baseline_vector_search,
    )
    from rag_pipeline import answer_question
except ImportError:
    from src.baselines import (
        baseline_bm25,
        baseline_pure_llm,
        baseline_vector_search,
    )
    from src.rag_pipeline import answer_question

try:
    from llm_factory import resolve_model, resolve_provider
except ImportError:
    from src.llm_factory import resolve_model, resolve_provider

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

QUESTIONS = [
    {
        "document": "Основы алгоритмизации и программирования",
        "question": "Что такое рекурсивная функция?",
    },
    {
        "document": "Компьютерные сети (Олифер)",
        "question": "Чем отличается TCP от UDP?",
    },
    {
        "document": "ООП",
        "question": "Зачем нужен конструктор копирования в C++?",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-тест RAG-ассистента.")
    parser.add_argument(
        "--provider",
        choices=["local", "openai", "groq", "openrouter", "deepseek"],
        default=None,
        help="Провайдер LLM (по умолчанию из LLM_PROVIDER).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Модель для облачных провайдеров (по умолчанию из настроек).",
    )
    return parser.parse_args()


def short_text(text: str, limit: int = 200) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "—"

    formatted = []
    for source in sources:
        document = source.get("document") or source.get("source") or "неизвестный документ"
        page = source.get("page")
        label = f"{document}, стр. {page}" if page is not None else str(document)
        if label not in formatted:
            formatted.append(label)
    return "; ".join(formatted)


def sanitize_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def build_systems(provider: str | None, model: str | None) -> list[tuple[str, object]]:
    return [
        (
            "RAG",
            lambda question: answer_question(
                question, provider=provider, model=model
            ),
        ),
        (
            "Pure LLM",
            lambda question: baseline_pure_llm(
                question, provider=provider, model=model
            ),
        ),
        ("Vector search (Chroma)", baseline_vector_search),
        ("BM25", baseline_bm25),
    ]


def run_smoke_test(provider: str | None, model: str | None) -> list[dict]:
    rows = []
    systems = build_systems(provider, model)
    for question_index, item in enumerate(QUESTIONS, start=1):
        for system_name, system in systems:
            started = time.perf_counter()
            try:
                result = system(item["question"])
                error = None
            except Exception as exc:  # noqa: BLE001
                result = {"answer": f"Ошибка: {exc}", "sources": []}
                error = str(exc)
            elapsed = time.perf_counter() - started

            rows.append(
                {
                    "question_index": question_index,
                    "document": item["document"],
                    "question": item["question"],
                    "system": system_name,
                    "provider": result.get("provider"),
                    "model": result.get("model"),
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "retrieved_chunks": result.get("retrieved_chunks", []),
                    "elapsed": elapsed,
                    "error": error,
                }
            )
            print(
                f"[{question_index}/{len(QUESTIONS)}] {system_name}: "
                f"{elapsed:.2f} с"
            )
    return rows


def build_report(
    rows: list[dict], provider: str, model: str
) -> str:
    lines = [
        "# Smoke-тест RAG-ассистента",
        "",
        f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Параметры",
        "",
        f"- Провайдер: `{provider}`",
        f"- Модель: `{model}`",
        "- Эмбеддер: `intfloat/multilingual-e5-small` (384 измерения)",
        "- Векторная БД: Chroma, коллекция `course_materials`",
        "- RAG: `TOP_K=4`",
        "- BM25: `rank-bm25`, лексический поиск top-1 по всем чанкам",
        "",
        "## Сводная таблица",
        "",
        "| № | Документ | Вопрос | Система | Провайдер | Модель | Время, с | Ответ (первые 200 символов) | Источники |",
        "| ---: | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f"| {row['question_index']} | {row['document']} | {row['question']} | "
            f"{row['system']} | {row['provider'] or '—'} | {row['model'] or '—'} | "
            f"{row['elapsed']:.2f} | "
            f"{short_text(row['answer']).replace('|', '/')} | "
            f"{format_sources(row['sources']).replace('|', '/')} |"
        )

    lines += ["", "## Подробные результаты", ""]

    for question_index, item in enumerate(QUESTIONS, start=1):
        lines += [
            f"### Вопрос {question_index}: {item['question']}",
            "",
            f"Документ: {item['document']}.",
            "",
        ]
        for row in [entry for entry in rows if entry["question_index"] == question_index]:
            lines += [
                f"#### {row['system']}",
                "",
                f"Провайдер: `{row['provider'] or '—'}`; модель: `{row['model'] or '—'}`.",
                "",
                f"Время ответа: {row['elapsed']:.2f} с.",
                "",
                "**Ответ:**",
                "",
                row["answer"].strip() or "—",
                "",
                f"**Источники:** {format_sources(row['sources'])}.",
                "",
            ]

            if row["retrieved_chunks"]:
                lines += ["**Найденные фрагменты:**", ""]
                for chunk_index, chunk in enumerate(row["retrieved_chunks"], start=1):
                    lines += [
                        f"{chunk_index}. Фрагмент {chunk_index}:",
                        "",
                        "```text",
                        chunk.strip(),
                        "```",
                        "",
                    ]

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    provider = resolve_provider(args.provider)
    model = resolve_model(args.provider, args.model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = (
        RESULTS_DIR / f"smoke_test_{provider}_{sanitize_model(model)}.md"
    )

    rows = run_smoke_test(args.provider, args.model)
    report = build_report(rows, provider, model)
    report_path.write_text(report, encoding="utf-8")

    print()
    print("| № | Вопрос | Система | Провайдер | Модель | Время, с | Ответ (200) | Источники |")
    print("| ---: | --- | --- | --- | --- | ---: | --- | --- |")
    for row in rows:
        print(
            f"| {row['question_index']} | {short_text(row['question'], 60)} | "
            f"{row['system']} | {row['provider'] or '—'} | {row['model'] or '—'} | "
            f"{row['elapsed']:.2f} | "
            f"{short_text(row['answer']).replace('|', '/')} | "
            f"{format_sources(row['sources']).replace('|', '/')} |"
        )
    print()
    print(f"Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
