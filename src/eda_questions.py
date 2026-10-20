from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "eda_questions.md"

SPLITS = ["test_set_full", "val_set", "test_set"]


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"Файл не найден: {path}")
    items = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: {exc}") from exc
    return items


def duplicate_values(items: list[dict], field: str) -> list[str]:
    counts = Counter(item[field] for item in items)
    return sorted(value for value, count in counts.items() if count > 1)


def mean_length(items: list[dict], field: str) -> float:
    if not items:
        return 0.0
    return statistics.mean(len(item[field]) for item in items)


def distribution(items: list[dict], field: str) -> str:
    counts = Counter(item[field] for item in items)
    return ", ".join(f"`{value}` — {count}" for value, count in sorted(counts.items()))


def build_report(splits: dict[str, list[dict]]) -> str:
    lines = [
        "# EDA тестового набора вопросов",
        "",
        "## Объём выборок",
        "",
        "| Выборка | Вопросов | Средняя длина question, символов | Средняя длина ground_truth, символов | Дубликаты question | Дубликаты ground_truth |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, items in splits.items():
        lines.append(
            f"| {name} | {len(items)} | {mean_length(items, 'question'):.1f} | "
            f"{mean_length(items, 'ground_truth'):.1f} | "
            f"{len(duplicate_values(items, 'question'))} | "
            f"{len(duplicate_values(items, 'ground_truth'))} |"
        )

    lines += ["", "## Распределение по документам", ""]
    for name, items in splits.items():
        lines.append(f"**{name}:** {distribution(items, 'doc')}.")
    lines += ["", "## Распределение по типам вопросов", ""]
    for name, items in splits.items():
        lines.append(f"**{name}:** {distribution(items, 'type')}.")

    lines += ["", "## Проверка дубликатов", ""]
    duplicates_found = False
    for name, items in splits.items():
        for field in ("question", "ground_truth"):
            duplicates = duplicate_values(items, field)
            if duplicates:
                duplicates_found = True
                lines.append(f"- **{name}, {field}:** {len(duplicates)} дубликат(ов).")
                for value in duplicates:
                    lines.append(f"  - {value}")
    if not duplicates_found:
        lines.append("Дубликатов вопросов и эталонных ответов не обнаружено.")

    test_items = splits["test_set"]
    lines += [
        "",
        "## Тестовая выборка",
        "",
        f"- Вопросов: {len(test_items)}.",
        f"- Диапазон gold_pages: {min((min(item['gold_pages']) for item in test_items), default=0)}–{max((max(item['gold_pages']) for item in test_items), default=0)}.",
        "- Тестовая выборка используется только для финальной оценки.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    splits = {name: load_split(name) for name in SPLITS}
    report = build_report(splits)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    for name, items in splits.items():
        print(f"{name}: {len(items)}")
    print(f"Отчёт сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
