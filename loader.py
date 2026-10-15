"""Загрузка и очистка локальных текстовых документов с сохранением источника."""

import logging
import re
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)
_INLINE_WHITESPACE = re.compile(r"[^\S\r\n]+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t\u00a0]*\n(?:[ \t\u00a0]*\n)*")


class Document(TypedDict):
    source_name: str
    text: str
    paragraphs: list[str]


def clean_text(text: str) -> str:
    """Удалить пустые строки и лишние пробелы, сохранив границы остальных строк."""
    lines = []
    for line in text.splitlines():
        cleaned = _INLINE_WHITESPACE.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def load_documents(folder: str | Path, *, strict: bool = False) -> list[Document]:
    """Прочитать текстовые файлы UTF-8 и вернуть очищенный текст с именем источника."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Папка документов не найдена: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Ожидалась папка документов: {folder}")

    documents: list[Document] = []
    files = sorted(
        (path for path in folder.iterdir() if path.is_file()),
        key=lambda path: path.name,
    )
    for path in files:
        if path.suffix.lower() != ".txt":
            continue
        try:
            original = path.read_text(encoding="utf-8")
            text = clean_text(original)
        except (OSError, UnicodeError) as exc:
            if strict:
                raise ValueError(f"Не удалось прочитать документ: {path.name}") from exc
            logger.warning("Не удалось прочитать %s: %s", path.name, exc)
            continue
        if not text:
            if strict:
                raise ValueError(f"Пустой документ: {path.name}")
            logger.warning("Пустой документ пропущен: %s", path.name)
            continue
        # Сохраняем границы абзацев до удаления пустых строк из общего текста.
        paragraphs = [
            cleaned
            for paragraph in _PARAGRAPH_BREAK.split(original)
            if (cleaned := clean_text(paragraph))
        ]
        documents.append(
            {"source_name": path.name, "text": text, "paragraphs": paragraphs}
        )
    return documents
