"""Демонстрация загрузки и разбиения локальной базы знаний."""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAIError
from weaviate.exceptions import WeaviateBaseError

from chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_documents
from embedder import Embedder
from loader import load_documents
from weaviate_store import WeaviateStore, connect


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовка текстовой базы знаний")
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "knowledge_base",
        help="Папка с документами UTF-8",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--examples", type=int, default=3, help="Число примеров фрагментов"
    )
    parser.add_argument(
        "--index", action="store_true", help="Загрузить фрагменты в Weaviate"
    )
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("Число примеров не может быть отрицательным")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        documents = load_documents(args.folder)
        chunks = chunk_documents(documents, args.chunk_size, args.overlap)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    for document in documents:
        print(f"{document['source_name']}: {len(document['text'])} символов")
    print(f"Загружено документов: {len(documents)}")
    print(f"Фрагментов: {len(chunks)}")
    print(f"Размер: до {args.chunk_size} символов; перекрытие: до {args.overlap}")
    for chunk in chunks[: args.examples]:
        print(json.dumps(chunk, ensure_ascii=False, indent=2))
    if args.index:
        try:
            embedder = Embedder.from_env()
            try:
                vectors = embedder.embed([chunk["text"] for chunk in chunks])
            finally:
                embedder.client.close()
            with connect() as client:
                store = WeaviateStore(
                    client, os.getenv("WEAVIATE_COLLECTION", "KnowledgeChunk")
                )
                result = store.sync(chunks, vectors, embedder.model, embedder.space_id)
            print(result.model_dump_json(indent=2))
        except APIStatusError as exc:
            parser.error(f"Сервис эмбеддингов вернул HTTP {exc.status_code}")
        except APIConnectionError:
            parser.error("Не удалось подключиться к сервису эмбеддингов")
        except OpenAIError:
            parser.error("Сервис эмбеддингов вернул некорректный ответ")
        except (WeaviateBaseError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    return 0 if documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
