from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from baselines import baseline_pure_llm, baseline_tfidf
from rag_pipeline import answer_question

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
REPORT_PATH = RESULTS_DIR / "smoke_test.md"

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

SYSTEMS = [
    ("RAG", answer_question),
    ("Pure LLM", baseline_pure_llm),
    ("TF-IDF/Chroma", baseline_tfidf),
]


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


def run_smoke_test() -> list[dict]:
    rows = []
    for question_index, item in enumerate(QUESTIONS, start=1):
        for system_name, system in SYSTEMS:
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


def build_report(rows: list[dict]) -> str:
    lines = [
        "# Smoke-тест RAG-ассистента",
        "",
        f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Параметры",
        "",
        "- LLM: `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`",
        "- Эмбеддер: `intfloat/multilingual-e5-small` (384 измерения)",
        "- Векторная БД: Chroma, коллекция `course_materials`",
        "- RAG: `TOP_K=4`, `N_CTX=8192`, `N_THREADS=8`",
        "",
        "## Сводная таблица",
        "",
        "| № | Документ | Вопрос | Система | Время, с | Ответ (первые 200 символов) | Источники |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f"| {row['question_index']} | {row['document']} | {row['question']} | "
            f"{row['system']} | {row['elapsed']:.2f} | "
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_smoke_test()
    report = build_report(rows)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print()
    print("| № | Вопрос | Система | Время, с | Ответ (200) | Источники |")
    print("| ---: | --- | --- | ---: | --- | --- |")
    for row in rows:
        print(
            f"| {row['question_index']} | {short_text(row['question'], 60)} | "
            f"{row['system']} | {row['elapsed']:.2f} | "
            f"{short_text(row['answer']).replace('|', '/')} | "
            f"{format_sources(row['sources']).replace('|', '/')} |"
        )
    print()
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
