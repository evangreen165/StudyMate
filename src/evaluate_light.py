from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from .config import PROVIDER_DEFAULTS
    from .llm_factory import _physical_cores, get_llm
    from .rag_pipeline import answer_question
except ImportError:
    from config import PROVIDER_DEFAULTS
    from llm_factory import _physical_cores, get_llm
    from rag_pipeline import answer_question

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
STOPWORDS = set(
    "и в во на с со по для что как это не а или при из к у о об от до за же ли то "
    "все его ее их также был была были быть может можно нужно который которая".split()
)

JUDGE_SYSTEM = (
    "Ты — строгий экзаменатор. Оцени ответ ассистента только по предоставленному "
    "контексту. Не добавляй своих знаний."
)

JUDGE_PROMPT = """Вопрос: {question}

Эталонный ответ: {ground_truth}

Контекст:
{context}

Ответ ассистента:
{answer}

Разбей ответ на утверждения. Для каждого реши, подтверждается ли оно контекстом.
Оцени, насколько ответ отвечает на вопрос, по шкале 1–5.
Верни ТОЛЬКО JSON без пояснений:
{{"faithful": <число подтверждённых утверждений>, "total": <общее число утверждений>, "relevance": <целое 1–5>, "comment": "<кратко>"}}"""

JUDGE_PROMPT_SIMPLE = """Вопрос: {question}
Контекст:
{context}
Ответ:
{answer}

Сколько утверждений ответа подтверждается контекстом? Сколько всего утверждений?
Насколько ответ релевантен вопросу (1–5)?
Верни ТОЛЬКО JSON: {{"faithful": 0, "total": 1, "relevance": 3}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight-оценка RAG.")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--provider", choices=["local", "openai", "groq", "openrouter", "deepseek"], default="local")
    parser.add_argument("--judge", choices=["local", "openai", "groq", "openrouter", "deepseek"], default="local")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}_set.jsonl"
    if not path.exists():
        raise SystemExit(f"Файл не найден: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keywords(text: str) -> set[str]:
    return {word for word in TOKEN_RE.findall(text.lower()) if word not in STOPWORDS and len(word) > 2}


def context_recall(ground_truth: str, context: str) -> float:
    reference = keywords(ground_truth)
    if not reference:
        return 0.0
    found = keywords(context)
    return len(reference & found) / len(reference)


@lru_cache(maxsize=4)
def get_judge_llm(provider: str):
    if provider == "local":
        model_path = PROVIDER_DEFAULTS["local"]["default_model"]
        if not Path(model_path).exists():
            raise RuntimeError(f"Файл модели не найден: {model_path}")
        return ChatLlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            n_threads=_physical_cores(),
            n_batch=512,
            n_ubatch=128,
            n_gpu_layers=0,
            temperature=0,
            max_tokens=1024,
            stop=["<|im_end|>"],
            verbose=False,
        )
    return get_llm(provider)


def parse_judge_response(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        faithful = int(data["faithful"])
        total = int(data["total"])
        relevance = int(data["relevance"])
    except (KeyError, TypeError, ValueError):
        return None
    if total <= 0 or faithful < 0 or faithful > total or not 1 <= relevance <= 5:
        return None
    return {
        "faithful": faithful,
        "total": total,
        "relevance": relevance,
        "comment": str(data.get("comment", ""))[:300],
    }


def judge_answer(judge: str, question: str, ground_truth: str, context: str, answer: str) -> dict:
    llm = get_judge_llm(judge)
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                context=context,
                answer=answer,
            )
        ),
    ]
    response = llm.invoke(messages)
    parsed = parse_judge_response(response.content)
    if parsed is not None:
        return parsed

    retry_messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=JUDGE_PROMPT_SIMPLE.format(
                question=question, context=context, answer=answer
            )
        ),
    ]
    retry = llm.invoke(retry_messages)
    parsed = parse_judge_response(retry.content)
    if parsed is not None:
        return parsed

    return {
        "faithful": None,
        "total": None,
        "relevance": None,
        "comment": f"Не удалось разобрать ответ судьи: {retry.content[:200]}",
    }


def run_evaluation(split: str, provider: str, judge: str, limit: int | None) -> list[dict]:
    items = load_split(split)
    if limit:
        items = items[:limit]

    rows = []
    for index, item in enumerate(items, start=1):
        started = time.perf_counter()
        result = answer_question(item["question"], provider=provider)
        rag_time = time.perf_counter() - started
        context = "\n\n---\n\n".join(result["retrieved_chunks"])

        started = time.perf_counter()
        judged = judge_answer(
            judge,
            item["question"],
            item["ground_truth"],
            context,
            result["answer"],
        )
        judge_time = time.perf_counter() - started

        faithfulness = None
        if judged["total"]:
            faithfulness = judged["faithful"] / judged["total"]

        row = {
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "gold_pages": item["gold_pages"],
            "doc": item["doc"],
            "type": item["type"],
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": result["retrieved_chunks"],
            "context_recall": context_recall(item["ground_truth"], context),
            "faithfulness": faithfulness,
            "answer_relevance": judged["relevance"],
            "judge_comment": judged["comment"],
            "judge_raw": judged,
            "provider": result.get("provider"),
            "model": result.get("model"),
            "rag_time_s": rag_time,
            "judge_time_s": judge_time,
        }
        rows.append(row)
        print(
            f"[{index}/{len(items)}] {item['id']}: "
            f"recall={row['context_recall']:.2f} "
            f"faith={row['faithfulness'] if row['faithfulness'] is None else round(row['faithfulness'], 2)} "
            f"rel={row['answer_relevance']} "
            f"rag={rag_time:.1f}s judge={judge_time:.1f}s"
        )
    return rows


def mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def build_markdown(rows: list[dict], split: str, provider: str, judge: str) -> str:
    faith = mean([row["faithfulness"] for row in rows])
    relevance = mean([row["answer_relevance"] for row in rows])
    recall = mean([row["context_recall"] for row in rows])
    lines = [
        "# Оценка RAG (lightweight evaluator)",
        "",
        f"- Выборка: `{split}`",
        f"- Провайдер RAG: `{provider}`",
        f"- Судья: `{judge}`",
        f"- Вопросов: {len(rows)}",
        "",
        "## Сводные метрики",
        "",
        "| Метрика | Среднее |",
        "| --- | ---: |",
        f"| Faithfulness | {faith:.3f} |" if faith is not None else "| Faithfulness | — |",
        f"| Answer relevance (1–5) | {relevance:.2f} |" if relevance is not None else "| Answer relevance (1–5) | — |",
        f"| Context recall (keywords) | {recall:.3f} |" if recall is not None else "| Context recall (keywords) | — |",
        "",
        "## По вопросам",
        "",
        "| ID | Вопрос | Faithfulness | Relevance | Context recall | RAG, с | Судья, с |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        faith_value = "—" if row["faithfulness"] is None else f"{row['faithfulness']:.2f}"
        relevance_value = "—" if row["answer_relevance"] is None else str(row["answer_relevance"])
        lines.append(
            f"| {row['id']} | {row['question']} | {faith_value} | {relevance_value} | "
            f"{row['context_recall']:.2f} | {row['rag_time_s']:.1f} | {row['judge_time_s']:.1f} |"
        )
    lines += [
        "",
        "## Ограничения",
        "",
        "- Faithfulness и relevance выставляет локальная модель Qwen2.5-1.5B; судья слабый и может ошибаться.",
        "- Context recall считается по пересечению ключевых слов, без учёта синонимов и морфологии.",
        "- RAGAS 0.4.3 не установился: scikit-network требует MSVC, а импорт несовместим с langchain-community; использован lightweight-эквивалент.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    rows = run_evaluation(args.split, args.provider, args.judge, args.limit)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = f"eval_{args.split}_{args.provider}_judge{args.judge}"
    json_path = RESULTS_DIR / f"{base}.json"
    md_path = RESULTS_DIR / f"{base}.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(rows, args.split, args.provider, args.judge), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Отчёт: {md_path}")


if __name__ == "__main__":
    main()
