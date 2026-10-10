"""Демонстрация загрузки локальной базы знаний."""

import argparse
import logging
from pathlib import Path

from loader import load_documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка текстовой базы знаний")
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "knowledge_base",
        help="Папка с документами UTF-8",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        documents = load_documents(args.folder)
    except OSError as exc:
        parser.error(str(exc))

    for document in documents:
        print(f"{document['source_name']}: {len(document['text'])} символов")
    print(f"Загружено документов: {len(documents)}")
    return 0 if documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
