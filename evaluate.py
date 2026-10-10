"""Сравнение двух размеров фрагментов и top_k на одинаковых вопросах."""

import argparse
import json
import logging
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAIError
from weaviate.exceptions import WeaviateBaseError

from chunking import chunk_documents
from embedder import Embedder
from loader import load_documents
from weaviate_store import WeaviateStore, connect

ROOT = Path(__file__).resolve().parent


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Сравнение настроек поиска")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/retrieval_settings.json"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cases = json.loads((ROOT / "examples/questions.json").read_text(encoding="utf-8"))
    documents = load_documents(ROOT / "knowledge_base")
    records = []
    embedder = Embedder.from_env()
    try:
        question_vectors = embedder.embed([case["question"] for case in cases])
        with connect() as client:
            for size, overlap in [(400, 40), (800, 80)]:
                name = "SettingsTest" + uuid4().hex
                try:
                    chunks = chunk_documents(documents, size, overlap)
                    vectors = embedder.embed([chunk["text"] for chunk in chunks])
                    store = WeaviateStore(client, name)
                    store.sync(chunks, vectors, embedder.model, embedder.space_id)
                    for top_k in [1, 3]:
                        results = []
                        for case, vector in zip(cases, question_vectors, strict=True):
                            found = store.search(
                                vector, embedder.model, embedder.space_id, top_k
                            )
                            results.append(
                                {
                                    **case,
                                    "context": [item.model_dump() for item in found],
                                }
                            )
                        hits = sum(
                            case["answerable"]
                            and any(
                                chunk["source_name"] == case["expected_source"]
                                for chunk in case["context"]
                            )
                            for case in results
                        )
                        record = {
                            "chunk_size": size,
                            "overlap": overlap,
                            "top_k": top_k,
                            "chunks": len(chunks),
                            "expected_source_hits": hits,
                            "answerable_questions": sum(
                                case["answerable"] for case in cases
                            ),
                            "cases": results,
                        }
                        records.append(record)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        print(
                            f"Размер {size}, top_k {top_k}: ожидаемый источник найден для {hits}/5 вопросов",
                            flush=True,
                        )
                finally:
                    if client.collections.exists(name):
                        client.collections.delete(name)
    except OpenAIError:
        parser.error("Не удалось получить эмбеддинги; готовые результаты сохранены")
    except (OSError, ValueError, RuntimeError, WeaviateBaseError) as exc:
        parser.error(str(exc))
    finally:
        embedder.client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
