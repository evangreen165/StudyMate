from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

try:
    from .baselines import baseline_bm25, baseline_vector_search
except ImportError:
    from baselines import baseline_bm25, baseline_vector_search

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
STOPWORDS = set(
    "и в во на с со по для что как это не а или при из к у о об от до за же ли то "
    "все его ее их также был была были быть может можно нужно который которая".split()
)


def keywords(text: str) -> set[str]:
    return {word for word in TOKEN_RE.findall(text.lower()) if word not in STOPWORDS and len(word) > 2}


def context_recall(ground_truth: str, context: str) -> float:
    reference = keywords(ground_truth)
    if not reference:
        return 0.0
    return len(reference & keywords(context)) / len(reference)


def load_test_set() -> list[dict]:
    path = DATA_DIR / "test_set.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    items = load_test_set()
    systems = [
        ("vector_search", baseline_vector_search, "Векторный поиск Chroma, top-1, первые 500 символов чанка."),
        ("BM25", baseline_bm25, "Лексический поиск rank-bm25, top-1, первые 500 символов чанка."),
    ]

    summaries = []
    details = []
    for name, function, comment in systems:
        recalls = []
        for item in items:
            result = function(item["question"])
            recall = context_recall(item["ground_truth"], result.get("answer", ""))
            recalls.append(recall)
            details.append((name, item["id"], item["question"], recall))
        summaries.append((name, statistics.mean(recalls) if recalls else 0.0, comment))

    lines = [
        "# Baseline-поиск на тестовой выборке",
        "",
        f"Вопросов: {len(items)}. Метрика: context recall по пересечению ключевых слов.",
        "",
        "| Baseline | Context recall | Комментарий |",
        "| --- | ---: | --- |",
    ]
    for name, value, comment in summaries:
        lines.append(f"| {name} | {value:.3f} | {comment} |")

    lines += [
        "",
        "## По вопросам",
        "",
        "| Baseline | ID | Вопрос | Context recall |",
        "| --- | --- | --- | ---: |",
    ]
    for name, question_id, question, recall in details:
        lines.append(f"| {name} | {question_id} | {question} | {recall:.2f} |")

    lines += [
        "",
        "## Ограничения",
        "",
        "- Используется только текст первого найденного чанка (первые 500 символов), без LLM-судьи.",
        "- Context recall по ключевым словам не учитывает синонимы, морфологию и порядок слов.",
        "- Pure LLM baseline не оценивался, чтобы не расходовать квоту OpenRouter и не зависеть от обрезки reasoning.",
        "",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "baselines_test.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    for name, value, _ in summaries:
        print(f"{name}: context_recall={value:.3f}")
    print(f"Отчёт: {output}")


if __name__ == "__main__":
    main()
