"""Получение векторных представлений через совместимый с OpenAI API."""

import hashlib
import logging
import math
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


def validate_vectors(vectors: list[list[float]], expected_count: int) -> int:
    """Проверить количество, размерность и числовые значения векторов."""
    if not isinstance(vectors, list) or len(vectors) != expected_count or not vectors:
        raise ValueError("Количество векторов не соответствует числу текстов")
    if any(not isinstance(vector, list) for vector in vectors):
        raise ValueError("API вернул неправильный формат вектора")
    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError("API вернул пустой вектор")
    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("Векторы имеют разную размерность")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Вектор содержит некорректные числовые значения")
        if not any(value != 0 for value in vector):
            raise ValueError("API вернул нулевой вектор")
    return dimension


class Embedder:
    def __init__(self, client: OpenAI, model: str):
        if not model.strip():
            raise ValueError("Не задана модель эмбеддингов")
        self.client = client
        self.model = model.strip()
        # Разные модели и разные сервисы не должны смешиваться в одном индексе.
        identity = f"{str(client.base_url).rstrip('/')}|{self.model}"
        self.space_id = hashlib.sha256(identity.encode()).hexdigest()

    @classmethod
    def from_env(cls) -> "Embedder":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("Укажите OPENAI_API_KEY в .env или переменных окружения")
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=60,
            max_retries=2,
        )
        return cls(client, os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Размер группы текстов должен быть положительным числом")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Нельзя получить эмбеддинг пустого текста")
        if not texts:
            return []

        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.model, input=batch, encoding_format="float"
            )
            data = response.data
            if not isinstance(data, list) or len(data) != len(batch):
                raise ValueError("API вернул неправильное число эмбеддингов")
            indices = [getattr(item, "index", None) for item in data]
            if any(type(index) is not int for index in indices) or sorted(
                indices
            ) != list(range(len(batch))):
                raise ValueError("API вернул неправильные индексы эмбеддингов")
            vectors.extend(
                getattr(item, "embedding", None)
                for item in sorted(data, key=lambda item: item.index)
            )
            logger.info("Получены эмбеддинги: %d/%d", len(vectors), len(texts))

        validate_vectors(vectors, len(texts))
        return vectors
