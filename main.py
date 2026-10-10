"""Загрузка, индексация, поиск и ответы по локальной базе знаний."""

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
from rag_pipeline import DEFAULT_MAX_DISTANCE
from runtime import open_pipeline
from weaviate_store import WeaviateStore, connect


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser(description="Ответы по текстовой базе знаний")
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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--index", action="store_true", help="Загрузить фрагменты в Weaviate"
    )
    actions.add_argument("--search", help="Найти фрагменты по вопросу")
    actions.add_argument("--ask", help="Ответить на вопрос с источниками")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=os.getenv("MAX_DISTANCE", DEFAULT_MAX_DISTANCE),
    )
    parser.add_argument("--json", action="store_true", help="Вывод в JSON")
    parser.add_argument(
        "--show-context", action="store_true", help="Показать найденный текст"
    )
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("Число примеров не может быть отрицательным")
    if args.top_k <= 0:
        parser.error("top_k должен быть положительным")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        if args.search is not None or args.ask is not None:
            with open_pipeline(args.top_k, args.max_distance) as pipeline:
                if args.search is not None:
                    context = pipeline.retriever.retrieve(args.search, args.top_k)
                    if args.json:
                        print(
                            json.dumps(
                                [chunk.model_dump() for chunk in context],
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                    else:
                        print_context(context)
                else:
                    result = pipeline.answer(args.ask)
                    if args.json:
                        print(result.model_dump_json(indent=2))
                    else:
                        print(result.answer)
                        for source in result.sources:
                            print(
                                f"\nИсточник: {source.source_name} — {source.section_title}"
                            )
                            print(
                                f"Фрагмент: {source.chunk_id}\nЦитата: {source.quote}"
                            )
                        if args.show_context:
                            print_context(result.context)
                    return 1 if result.status == "invalid_output" else 0
            return 0

        documents = load_documents(args.folder)
        chunks = chunk_documents(documents, args.chunk_size, args.overlap)
        for document in documents:
            print(f"{document['source_name']}: {len(document['text'])} символов")
        print(f"Загружено документов: {len(documents)}")
        print(f"Фрагментов: {len(chunks)}")
        print(f"Размер: до {args.chunk_size} символов; перекрытие: до {args.overlap}")
        for chunk in chunks[: args.examples]:
            print(json.dumps(chunk, ensure_ascii=False, indent=2))
        if args.index:
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
        parser.error(f"Сервис модели вернул HTTP {exc.status_code}")
    except APIConnectionError:
        parser.error("Не удалось подключиться к сервису модели")
    except OpenAIError:
        parser.error("Сервис модели вернул некорректный ответ")
    except (OSError, WeaviateBaseError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0 if documents else 1


def print_context(context):
    for index, chunk in enumerate(context, 1):
        print(f"\n{index}. {chunk.source_name} — {chunk.section_title}")
        print(f"Фрагмент: {chunk.chunk_id}; расстояние: {chunk.distance:.4f}")
        print(chunk.text)


if __name__ == "__main__":
    raise SystemExit(main())
