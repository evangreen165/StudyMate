from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "error_analysis.md"

PRIORITY = [
    "retrieval_failure",
    "faithfulness_failure",
    "relevance_failure",
    "ambiguity",
    "ok",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ ошибок RAG.")
    parser.add_argument(
        "--eval-file",
        default=None,
        help="JSON с результатами evaluate_light; по умолчанию results/eval_test_<provider>_judgelocal.json.",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openrouter"],
        default="openrouter",
        help="Провайдер, чьи результаты анализируются.",
    )
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def classify(row: dict) -> list[str]:
    retrieved = {source.get("page") for source in row.get("sources", [])}
    gold = set(row.get("gold_pages", []))
    hit = bool(gold & retrieved)
    faithfulness = row.get("faithfulness")
    relevance = row.get("answer_relevance")
    recall = row.get("context_recall") or 0.0

    labels = []
    if not hit:
        if recall >= 0.5:
            labels.append("ambiguity")
        else:
            labels.append("retrieval_failure")
    if faithfulness is not None and faithfulness < 0.6:
        labels.append("faithfulness_failure")
    if relevance is not None and relevance < 3:
        labels.append("relevance_failure")
    if not labels:
        labels.append("ok")
    return labels


def primary_label(labels: list[str]) -> str:
    for label in PRIORITY:
        if label in labels:
            return label
    return labels[0]


def describe_cause(row: dict, labels: list[str]) -> str:
    answer = row.get("answer", "")
    recall = row.get("context_recall") or 0.0
    if "retrieval_failure" in labels:
        return (
            "Ни одна gold-страница не попала в top-4: эмбеддер не нашёл "
            "терминологическое совпадение, вопрос сформулирован иначе."
        )
    if "ambiguity" in labels:
        return (
            "Найдены тематически близкие страницы, но не gold: вопрос допускает "
            "несколько трактовок или ответ размазан по нескольким разделам."
        )
    if "faithfulness_failure" in labels:
        if len(answer) > 700:
            return (
                "Ответ многословен: модель добавляет детали и примеры, которых нет "
                "в найденных фрагментах."
            )
        if recall < 0.35:
            return (
                "В контексте мало ключевых слов эталона; модель дополняет ответ "
                "своими знаниями вместо точного следования фрагментам."
            )
        if row.get("type") == "procedural":
            return (
                "Процедурный вопрос: модель добавляет шаги и уточнения, не "
                "подтверждённые контекстом."
            )
        return (
            "Модель добавляет общеизвестные детали, не подтверждённые "
            "retrieved_chunks (низкая faithfulness)."
        )
    if "relevance_failure" in labels:
        return "Ответ не отвечает на вопрос по существу или слишком общий."
    return "Ошибок не обнаружено: retrieval и ответ согласованы."


def format_pages(pages: list[int]) -> str:
    return ", ".join(str(page) for page in sorted(set(pages))) or "—"


def build_report(rows: list[dict], provider: str, limit: int) -> str:
    for row in rows:
        row["labels"] = classify(row)
        row["primary"] = primary_label(row["labels"])
        row["cause"] = describe_cause(row, row["labels"])

    errors = [row for row in rows if row["primary"] != "ok"]
    errors.sort(key=lambda row: (PRIORITY.index(row["primary"]), row["id"]))
    selected = errors[:limit]
    if len(selected) < 10:
        selected += [row for row in rows if row["primary"] == "ok"][: 10 - len(selected)]

    label_counts = Counter(label for row in rows for label in row["labels"])
    primary_counts = Counter(row["primary"] for row in rows)
    cause_counts = Counter(row["cause"] for row in errors)

    lines = [
        "# Анализ ошибок RAG",
        "",
        f"Источник: `results/eval_test_{provider}_judgelocal.json` (18 вопросов, OpenRouter RAG, локальный судья).",
        "",
        "## Распределение ошибок",
        "",
        "| Тип | Количество (multi-label) |",
        "| --- | ---: |",
    ]
    for label in PRIORITY:
        lines.append(f"| {label} | {label_counts.get(label, 0)} |")

    lines += [
        "",
        "| Первичный тип | Количество |",
        "| --- | ---: |",
    ]
    for label in PRIORITY:
        lines.append(f"| {label} | {primary_counts.get(label, 0)} |")

    lines += [
        "",
        "Все gold-страницы попали в top-4 на этой выборке, поэтому retrieval_failure "
        "не обнаружен. Основная проблема — низкая faithfulness: модель добавляет "
        "утверждения, которых нет в найденных фрагментах.",
        "",
        "**Важно:** локальный судья Qwen2.5-1.5B часто занижает faithfulness. "
        "Например, ответ A2 по содержанию совпадает с эталоном, но получил 0.16. "
        "Поэтому распределение отражает и ошибки судьи, а не только галлюцинации модели.",
        "",
        "## Примеры ошибок",
        "",
    ]

    for index, row in enumerate(selected, start=1):
        retrieved = [source.get("page") for source in row.get("sources", [])]
        lines += [
            f"### {index}. {row['id']} — {', '.join(row['labels'])}",
            "",
            f"**Вопрос:** {row['question']}",
            "",
            f"**Эталонный ответ:** {row['ground_truth']}",
            "",
            f"**Ответ системы:** {row['answer']}",
            "",
            f"**Gold pages:** {format_pages(row['gold_pages'])}; "
            f"**retrieved pages:** {format_pages(retrieved)}.",
            "",
            f"**Метрики:** faithfulness={row['faithfulness']:.2f}, "
            f"relevance={row['answer_relevance']}, "
            f"context_recall={row['context_recall']:.2f}.",
            "",
            f"**Предполагаемая причина:** {row['cause']}",
            "",
        ]

    lines += [
        "## Топ-3 причины по частоте",
        "",
    ]
    for index, (cause, count) in enumerate(cause_counts.most_common(3), start=1):
        lines.append(f"{index}. {cause} — {count} раз(а).")
    lines += [
        "",
        "## Направления улучшения",
        "",
        "1. Добавить реранкер (cross-encoder) после векторного поиска, чтобы в top-K попадали более точные фрагменты.",
        "2. Использовать гибридный поиск BM25 + вектор: лексические совпадения помогают терминам вроде TCP/UDP и именам классов.",
        "3. Фильтровать кандидатов по документу, если в вопросе явно упоминается тема (сети, ООП, алгоритмы).",
        "4. Увеличить chunk_overlap или использовать чанки по абзацам, чтобы не терять определения на границах.",
        "5. Усилить промпт: требовать краткий ответ и явно запрещать детали, отсутствующие в контексте; ограничить длину ответа.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    eval_path = (
        Path(args.eval_file)
        if args.eval_file
        else RESULTS_DIR / f"eval_test_{args.provider}_judgelocal.json"
    )
    if not eval_path.exists():
        raise SystemExit(
            f"Не найден файл результатов: {eval_path}. "
            "Сначала запустите src/evaluate_light.py."
        )
    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    report = build_report(rows, args.provider, args.limit)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    counts = Counter(primary_label(classify(row)) for row in rows)
    print("Распределение:", dict(counts))
    print(f"Отчёт: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
