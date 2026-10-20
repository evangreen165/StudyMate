from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
MATERIALS_DIR = BASE_DIR / "data" / "course_materials"
OUTPUT_PATH = BASE_DIR / "data" / "dataset_passport.md"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
LOW_TEXT_THRESHOLD = 200
MATH_SYMBOLS = "=∫∑√±×÷≈≤≥"


@dataclass
class FileStats:
    name: str
    title: str
    author: str
    pages: int
    chars: int
    chunks: int
    low_text_pages: list[tuple[int, int]] = field(default_factory=list)
    math_symbols: int = 0
    cyrillic_chars: int = 0
    latin_chars: int = 0

    @property
    def avg_chars(self) -> float:
        return self.chars / self.pages if self.pages else 0.0


def format_number(value: float, digits: int = 0) -> str:
    if digits:
        return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    return f"{int(round(value)):,}".replace(",", " ")


def is_cyrillic(char: str) -> bool:
    lowered = char.lower()
    return "а" <= lowered <= "я" or lowered == "ё"


def is_latin(char: str) -> bool:
    lowered = char.lower()
    return "a" <= lowered <= "z"


def plural(number: int, one: str, few: str, many: str) -> str:
    if number % 10 == 1 and number % 100 != 11:
        return one
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return few
    return many


def load_stats(splitter: RecursiveCharacterTextSplitter) -> list[FileStats]:
    pdf_paths = sorted(MATERIALS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"В папке {MATERIALS_DIR} нет PDF-файлов.")

    stats: list[FileStats] = []
    for pdf_path in pdf_paths:
        documents = PyPDFLoader(str(pdf_path)).load()
        if not documents:
            continue

        text = "\n".join(document.page_content for document in documents)
        metadata = documents[0].metadata
        low_text_pages = [
            (index + 1, len(document.page_content))
            for index, document in enumerate(documents)
            if len(document.page_content.strip()) < LOW_TEXT_THRESHOLD
        ]
        stats.append(
            FileStats(
                name=pdf_path.name,
                title=str(metadata.get("title") or "—").strip(),
                author=str(metadata.get("author") or "—").strip(),
                pages=len(documents),
                chars=sum(len(document.page_content) for document in documents),
                chunks=len(splitter.split_documents(documents)),
                low_text_pages=low_text_pages,
                math_symbols=sum(text.count(symbol) for symbol in MATH_SYMBOLS),
                cyrillic_chars=sum(1 for char in text if is_cyrillic(char)),
                latin_chars=sum(1 for char in text if is_latin(char)),
            )
        )
    return stats


def build_report(stats: list[FileStats]) -> str:
    total_pages = sum(item.pages for item in stats)
    total_chars = sum(item.chars for item in stats)
    total_chunks = sum(item.chunks for item in stats)
    avg_page_chars = total_chars / total_pages if total_pages else 0.0

    cyrillic = sum(item.cyrillic_chars for item in stats)
    latin = sum(item.latin_chars for item in stats)

    total_low_text_pages = sum(len(item.low_text_pages) for item in stats)
    total_math_symbols = sum(item.math_symbols for item in stats)

    lines = [
        "# Паспорт корпуса учебных материалов",
        "",
        "Отчёт сформирован скриптом `src/dataset_passport.py`.",
        "",
        "## Сводные метрики",
        "",
        "| Показатель | Значение |",
        "| --- | ---: |",
        f"| Число PDF-документов | {len(stats)} |",
        f"| Суммарное число страниц | {format_number(total_pages)} |",
        f"| Суммарное число символов | {format_number(total_chars)} |",
        f"| Средняя длина страницы, символов | {format_number(avg_page_chars, 1)} |",
        f"| Число чанков (`chunk_size={CHUNK_SIZE}`, `chunk_overlap={CHUNK_OVERLAP}`) | {format_number(total_chunks)} |",
        "",
        "## Детализация по файлам",
        "",
        "| Файл | Название | Автор | Страниц | Символов | Средняя длина страницы | Чанков |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for item in stats:
        lines.append(
            f"| `{item.name}` | {item.title} | {item.author} | {format_number(item.pages)} | "
            f"{format_number(item.chars)} | {format_number(item.avg_chars, 1)} | {format_number(item.chunks)} |"
        )

    lines += [
        "",
        "## Описание корпуса",
        "",
        f"**Источник материалов.** Локальная папка `data/course_materials/`. "
        f"Корпус содержит {len(stats)} {plural(len(stats), 'PDF-файл', 'PDF-файла', 'PDF-файлов')} "
        f"общим объёмом {format_number(total_pages)} "
        f"{plural(total_pages, 'страница', 'страницы', 'страниц')}.",
        "",
        "**Состав корпуса.**",
        "",
    ]
    for item in stats:
        title = item.title if item.title != "—" else "название не указано в метаданных"
        author = item.author if item.author != "—" else "автор не указан"
        lines.append(f"- `{item.name}` — {title} ({author}).")
    lines += [
        "",
        f"**Язык.** Основной язык материалов — русский; в тексте встречаются англоязычные "
        f"технические термины и названия технологий. Доля кириллических символов среди "
        f"букв составляет {format_number(cyrillic / max(cyrillic + latin, 1) * 100, 1)} %.",
        "",
        "**Формат хранения.** PDF с текстовым слоем. Текст извлекается `PyPDFLoader` "
        f"(на базе `pypdf`), разбиение выполняется `RecursiveCharacterTextSplitter` "
        f"с `chunk_size={CHUNK_SIZE}` и `chunk_overlap={CHUNK_OVERLAP}`.",
        "",
        "**Ограничения.**",
        "",
    ]

    if total_low_text_pages:
        lines.append(
            f"- Страницы с малым объёмом текста (менее {LOW_TEXT_THRESHOLD} символов): "
            f"{total_low_text_pages} шт. Это титульные листы, разделители частей и страницы "
            "с преобладанием рисунков или кода."
        )
        for item in stats:
            if item.low_text_pages:
                pages = ", ".join(
                    f"{page} ({length} симв.)" for page, length in item.low_text_pages
                )
                lines.append(f"  - `{item.name}`: страницы {pages}.")
    else:
        lines.append(
            f"- Страниц с малым объёмом текста (менее {LOW_TEXT_THRESHOLD} символов) не обнаружено."
        )

    lines += [
        f"- В корпусе встречаются математические символы и формулы "
        f"(условный счётчик символов `{MATH_SYMBOLS}`: {format_number(total_math_symbols)} шт.). "
        "При извлечении текста формулы могут терять структуру и превращаться в "
        "последовательность символов, что снижает качество поиска по ним.",
        "- Таблицы, схемы, рисунки и листинги кода извлекаются частично: визуальная "
        "структура и связи между элементами не сохраняются.",
        "- Переносы слов и разрывы строк PDF могут попадать в текст, создавая шум в чанках.",
        "- Англоязычные термины и аббревиатуры полезно учитывать при выборе модели "
        "эмбеддингов и формулировке запросов.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    stats = load_stats(splitter)
    report = build_report(stats)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    total_pages = sum(item.pages for item in stats)
    total_chars = sum(item.chars for item in stats)
    total_chunks = sum(item.chunks for item in stats)
    print(f"Документов: {len(stats)}")
    print(f"Страниц: {total_pages}")
    print(f"Символов: {total_chars}")
    print(f"Чанков: {total_chunks}")
    print(f"Отчёт сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
