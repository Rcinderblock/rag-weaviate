"""Загрузка и очистка локальных текстовых документов с сохранением источника."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)
_INLINE_WHITESPACE = re.compile(r"[^\S\r\n]+")


def clean_text(text: str) -> str:
    """Удалить пустые строки и лишние пробелы, сохранив границы остальных строк."""
    lines = []
    for line in text.splitlines():
        cleaned = _INLINE_WHITESPACE.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def load_documents(folder: str | Path) -> list[dict[str, str]]:
    """Прочитать текстовые файлы UTF-8 и вернуть очищенный текст с именем источника."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Папка документов не найдена: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Ожидалась папка документов: {folder}")

    documents = []
    files = sorted(
        (path for path in folder.iterdir() if path.is_file()),
        key=lambda path: path.name,
    )
    for path in files:
        if path.suffix.lower() != ".txt":
            continue
        try:
            text = clean_text(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            logger.warning("Не удалось прочитать %s: %s", path.name, exc)
            continue
        if not text:
            logger.warning("Пустой документ пропущен: %s", path.name)
            continue
        documents.append({"source_name": path.name, "text": text})
    return documents
