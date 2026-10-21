from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Замер производительности /chat.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


def load_questions(n: int) -> list[str]:
    path = DATA_DIR / "test_set.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item["question"] for item in items[:n]]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def post_question(base_url: str, provider: str, question: str) -> dict:
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"question": question, "provider": provider},
            timeout=240,
        )
        elapsed = time.perf_counter() - started
        if response.status_code == 200:
            data = response.json()
            return {
                "elapsed": elapsed,
                "ok": True,
                "provider": data.get("provider"),
                "model": data.get("model"),
                "sources": len(data.get("sources", [])),
            }
        return {
            "elapsed": elapsed,
            "ok": False,
            "status": response.status_code,
            "error": response.text[:300],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "elapsed": time.perf_counter() - started,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    questions = load_questions(args.n)
    base_url = args.url.rstrip("/")

    wall_started = time.perf_counter()
    if args.concurrency <= 1:
        results = [post_question(base_url, args.provider, question) for question in questions]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            results = list(
                executor.map(
                    lambda question: post_question(base_url, args.provider, question),
                    questions,
                )
            )
    wall_time = time.perf_counter() - wall_started

    times = [item["elapsed"] for item in results]
    success = sum(1 for item in results if item["ok"])
    failed = len(results) - success
    stats = {
        "min": min(times) if times else 0.0,
        "p50": percentile(times, 50),
        "p95": percentile(times, 95),
        "max": max(times) if times else 0.0,
        "mean": statistics.mean(times) if times else 0.0,
    }
    rps = len(results) / wall_time if wall_time > 0 else 0.0

    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": args.provider,
        "n": len(questions),
        "concurrency": args.concurrency,
        "success": success,
        "failed": failed,
        "wall_time_s": wall_time,
        "rps": rps,
        "stats": stats,
        "times": times,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"performance_{args.provider}.json"
    payload = {"provider": args.provider, "url": base_url, "runs": []}
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
    payload.setdefault("runs", []).append(run)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"provider={args.provider} n={run['n']} concurrency={args.concurrency}")
    print(f"success={success} failed={failed} wall={wall_time:.2f}s rps={rps:.3f}")
    for key, value in stats.items():
        print(f"{key}={value:.3f}s")
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
