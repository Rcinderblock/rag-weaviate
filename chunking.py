"""Разбиение документов по явным заголовкам, абзацам и предложениям."""

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import TypedDict
from uuid import NAMESPACE_URL, uuid5

from loader import Document

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 80
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


class Chunk(TypedDict):
    document_id: str
    source_name: str
    chunk_id: str
    section_title: str
    text: str


@dataclass
class _Section:
    title: str
    paragraphs: list[str]


def _sections(document: Document) -> list[_Section]:
    """Распознать заголовки # … ######; текст без них считать одним разделом."""
    sections = []
    headings: list[tuple[int, str]] = []
    paragraphs: list[str] = []
    title = document["source_name"]

    for paragraph in document["paragraphs"]:
        lines: list[str] = []
        for line in paragraph.splitlines():
            heading = _HEADING.match(line)
            if heading is None:
                lines.append(line)
                continue
            if lines:
                paragraphs.append("\n".join(lines))
                lines = []
            if paragraphs:
                sections.append(_Section(title, paragraphs))
                paragraphs = []
            level = len(heading[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading[2]))
            title = " / ".join(name for _, name in headings)
        if lines:
            paragraphs.append("\n".join(lines))

    if paragraphs:
        sections.append(_Section(title, paragraphs))
    return sections


def _split_body(body: str, size: int, overlap: int) -> list[str]:
    """Предпочесть границы абзацев, затем предложений, затем слов."""
    boundaries = [
        [match.start() for match in re.finditer(pattern, body)]
        for pattern in (r"\n\n", r"(?<=[.!?])\s+", r"\s+")
    ]
    parts = []
    start = 0
    while start < len(body):
        end = min(start + size, len(body))
        if end < len(body):
            # Не создаём крошечный фрагмент ради ранней границы абзаца.
            minimum_end = start + max(1, size // 2)
            for positions in boundaries:
                index = bisect_right(positions, end) - 1
                if index >= 0 and positions[index] >= minimum_end:
                    end = positions[index]
                    break
        parts.append(body[start:end].strip())
        if end == len(body):
            break

        next_start = max(start + 1, end - overlap)
        # Начало перекрытия сдвигаем к целому слову. Очень длинное слово
        # без пробелов приходится резать по символам, чтобы соблюдать предел.
        end_inside_word = not body[end - 1].isspace() and not body[end].isspace()
        if (
            not end_inside_word
            and next_start > 0
            and not body[next_start - 1].isspace()
            and not body[next_start].isspace()
        ):
            while next_start < end and not body[next_start].isspace():
                next_start += 1
        while next_start < len(body) and body[next_start].isspace():
            next_start += 1
        start = next_start
    return parts


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Размер и перекрытие заданы в символах; заголовок входит в размер."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Размер фрагмента должен быть положительным целым числом")
    if type(overlap) is not int or not 0 <= overlap < chunk_size // 2:
        raise ValueError("Перекрытие должно быть целым числом меньше половины размера")

    chunks: list[Chunk] = []
    for document in documents:
        document_id = uuid5(NAMESPACE_URL, document["source_name"])
        for section_index, section in enumerate(_sections(document)):
            prefix = section.title + "\n"
            body_size = chunk_size - len(prefix)
            if body_size < 2 or overlap >= body_size // 2:
                raise ValueError(
                    f"Увеличьте размер или уменьшите перекрытие: заголовок "
                    f"раздела в {document['source_name']} занимает {len(prefix)} символов"
                )
            body = "\n\n".join(section.paragraphs)
            for part_index, part in enumerate(_split_body(body, body_size, overlap)):
                text = prefix + part
                chunk_id = uuid5(document_id, f"{section_index}:{part_index}:{text}")
                chunks.append(
                    {
                        "document_id": str(document_id),
                        "source_name": document["source_name"],
                        "chunk_id": str(chunk_id),
                        "section_title": section.title,
                        "text": text,
                    }
                )
    return chunks
