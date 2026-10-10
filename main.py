"""Демонстрация загрузки и разбиения локальной базы знаний."""

import argparse
import json
import logging
from pathlib import Path

from chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_documents
from loader import load_documents


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
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("Число примеров не может быть отрицательным")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
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
    return 0 if documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
