"""Десять вопросов; сохранение настоящих ответов и сравнение без поиска."""

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError
from weaviate.exceptions import WeaviateBaseError

from rag_pipeline import DEFAULT_MAX_DISTANCE
from runtime import open_pipeline

ROOT = Path(__file__).resolve().parent


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Демонстрация ответов по базе знаний")
    parser.add_argument(
        "--compare", action="store_true", help="Сравнить с генерацией без контекста"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/demo.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=os.getenv("MAX_DISTANCE", DEFAULT_MAX_DISTANCE),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cases = json.loads((ROOT / "examples/questions.json").read_text(encoding="utf-8"))
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "top_k": args.top_k,
        "max_distance": args.max_distance,
        "cases": [],
    }
    try:
        with open_pipeline(args.top_k, args.max_distance) as pipeline:
            report["model"] = pipeline.generator.model
            report["embedding_model"] = pipeline.retriever.embedder.model
            for case in cases:
                result = pipeline.answer(case["question"])
                record = {**case, "rag": result.model_dump()}
                if args.compare and case["answerable"]:
                    # Те же вопрос, модель, инструкции и температура; меняется контекст.
                    record["without_retrieval"] = pipeline.generator.generate(
                        case["question"], []
                    ).model_dump()
                report["cases"].append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"{case['id']}: {result.status}\n{result.answer}\n", flush=True)
    except OpenAIError:
        parser.error(
            "Не удалось выполнить запрос к модели; уже полученные результаты сохранены"
        )
    except (OSError, ValueError, RuntimeError, WeaviateBaseError) as exc:
        parser.error(str(exc))
    return (
        1
        if any(case["rag"]["status"] == "invalid_output" for case in report["cases"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
