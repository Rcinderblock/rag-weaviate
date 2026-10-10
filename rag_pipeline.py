"""Вопрос → поиск → проверка контекста → ответ с подтверждёнными источниками."""

import logging
import math

from llm_client import Generator
from retriever import Retriever
from schemas import GeneratedAnswer, RagResult, RetrievedChunk, Source

logger = logging.getLogger(__name__)
DEFAULT_MAX_DISTANCE = 0.55


def bind_sources(
    generated: GeneratedAnswer, context: list[RetrievedChunk]
) -> list[Source]:
    if generated.abstained:
        if generated.citations:
            raise ValueError("Отказ не должен содержать использованные источники")
        return []
    if not generated.citations:
        raise ValueError("Ответ не содержит подтверждающих цитат")
    by_id = {chunk.chunk_id: chunk for chunk in context}
    sources = []
    seen = set()
    for citation in generated.citations:
        if citation.chunk_id not in by_id or citation.chunk_id in seen:
            raise ValueError("Модель указала неизвестный или повторяющийся источник")
        chunk = by_id[citation.chunk_id]
        quote = " ".join(citation.quote.split())
        if not quote or quote not in " ".join(chunk.text.split()):
            raise ValueError("Цитата не найдена в указанном источнике")
        if quote in chunk.section_title:
            raise ValueError("Один заголовок не подтверждает ответ")
        sources.append(
            Source(
                chunk_id=chunk.chunk_id,
                quote=citation.quote,
                source_name=chunk.source_name,
                section_title=chunk.section_title,
            )
        )
        seen.add(citation.chunk_id)
    return sources


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 3,
        max_distance: float = DEFAULT_MAX_DISTANCE,
    ):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k должен быть положительным целым числом")
        if (
            not isinstance(max_distance, (int, float))
            or isinstance(max_distance, bool)
            or not math.isfinite(max_distance)
            or not 0 <= max_distance <= 2
        ):
            raise ValueError("Порог расстояния должен быть числом от 0 до 2")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.max_distance = max_distance

    def answer(self, question: str) -> RagResult:
        context = self.retriever.retrieve(question, self.top_k)
        accepted = [chunk for chunk in context if chunk.distance <= self.max_distance]
        logger.info("Порог прошли %d/%d фрагментов", len(accepted), len(context))
        if not accepted:
            return RagResult(
                question=question,
                status="insufficient_context",
                answer="В базе знаний не найдено достаточно подходящей информации для ответа.",
                sources=[],
                context=context,
            )
        try:
            generated = self.generator.generate(question, accepted)
            sources = bind_sources(generated, accepted)
        except ValueError:
            logger.warning("Ответ модели не прошёл проверку структуры или источников")
            return RagResult(
                question=question,
                status="invalid_output",
                answer="Модель вернула ответ, который не удалось подтвердить источниками.",
                sources=[],
                context=context,
            )
        return RagResult(
            question=question,
            answer=generated.answer,
            status="insufficient_context" if generated.abstained else "answered",
            sources=sources,
            context=context,
        )
