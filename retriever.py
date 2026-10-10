"""Поиск подходящих фрагментов по вектору вопроса."""

import logging

from embedder import Embedder
from schemas import RetrievedChunk
from weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder, store: WeaviateStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Вопрос не может быть пустым")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k должен быть положительным целым числом")
        expected_dimensions = self.store.check_model(
            self.embedder.model, self.embedder.space_id
        )
        vector = self.embedder.embed([question.strip()])[0]
        if len(vector) != expected_dimensions:
            raise ValueError("Размерность вектора вопроса не совпадает с индексом")
        result = self.store.search(
            vector, self.embedder.model, self.embedder.space_id, top_k
        )
        logger.info(
            "Лучшее расстояние: %s", result[0].distance if result else "нет результатов"
        )
        return result
