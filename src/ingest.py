from __future__ import annotations

import argparse
import json
import sys

try:
    from src.dataset_manager import dataset_status, get_passport, register_dataset
except ImportError:
    from dataset_manager import dataset_status, get_passport, register_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Индексация PDF в датасет.")
    parser.add_argument("--dataset-id", default="course_materials")
    parser.add_argument("--dataset-name", default="Учебные материалы курса")
    parser.add_argument(
        "--description",
        default="Базовый корпус из трёх PDF: основы алгоритмизации, компьютерные сети, ООП.",
    )
    parser.add_argument("--source-dir", default="data/course_materials")
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Только создать паспорт для уже проиндексированного датасета.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    if (
        not args.skip_indexing
        and dataset_status(args.dataset_id) is not None
    ):
        print(
            f"Датасет {args.dataset_id} уже зарегистрирован; "
            "переиндексация не требуется."
        )
        print(json.dumps(get_passport(args.dataset_id), ensure_ascii=False, indent=2))
        return

    passport = register_dataset(
        dataset_id=args.dataset_id,
        name=args.dataset_name,
        description=args.description,
        source_dir=args.source_dir,
        skip_indexing=args.skip_indexing,
    )
    print(json.dumps(passport, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
