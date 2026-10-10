"""Хранение фрагментов, векторов и метаданных в локальной Weaviate."""

import json
import logging
import math
import os
from uuid import UUID

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import MetadataQuery

from chunking import Chunk
from embedder import validate_vectors
from schemas import IndexingResult, RetrievedChunk

logger = logging.getLogger(__name__)
_PROPERTIES = ("document_id", "source_name", "chunk_id", "section_title", "text")


def connect():
    return weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    )


class WeaviateStore:
    def __init__(
        self, client: weaviate.WeaviateClient, collection_name: str = "KnowledgeChunk"
    ):
        self.client = client
        self.collection_name = collection_name

    def _collection(
        self, model: str, space_id: str, dimensions: int, create: bool = True
    ):
        description = json.dumps(
            {"embedding_model": model, "space_id": space_id, "dimensions": dimensions},
            sort_keys=True,
        )
        if not self.client.collections.exists(self.collection_name):
            if not create:
                raise ValueError("Индекс не найден. Сначала запустите main.py --index")
            logger.info("Создание коллекции %s", self.collection_name)
            return self.client.collections.create(
                name=self.collection_name,
                description=description,
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    )
                ),
                properties=[
                    Property(name=name, data_type=DataType.TEXT) for name in _PROPERTIES
                ],
            )
        collection = self.client.collections.use(self.collection_name)
        if collection.config.get().description != description:
            raise ValueError(
                "Коллекция использует другую модель или размерность. "
                "Задайте новое имя WEAVIATE_COLLECTION и заново создайте индекс"
            )
        return collection

    def check_model(self, model: str, space_id: str) -> int:
        if not self.client.collections.exists(self.collection_name):
            raise ValueError("Индекс не найден. Сначала запустите main.py --index")
        collection = self.client.collections.use(self.collection_name)
        try:
            config = json.loads(collection.config.get().description or "")
        except (ValueError, TypeError) as exc:
            raise ValueError("Индекс не содержит конфигурации эмбеддингов") from exc
        if (
            not isinstance(config, dict)
            or config.get("embedding_model") != model
            or config.get("space_id") != space_id
        ):
            raise ValueError("Модель запроса не совпадает с моделью индекса")
        dimensions = config.get("dimensions")
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("Индекс содержит неправильную размерность векторов")
        return dimensions

    def search(
        self, vector: list[float], model: str, space_id: str, top_k: int = 3
    ) -> list[RetrievedChunk]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k должен быть положительным целым числом")
        dimensions = validate_vectors([vector], 1)
        collection = self._collection(model, space_id, dimensions, create=False)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_properties=list(_PROPERTIES),
            return_metadata=MetadataQuery(distance=True),
        )
        result = [
            RetrievedChunk(**obj.properties, distance=obj.metadata.distance)
            for obj in response.objects
        ]
        logger.info("Поиск вернул %d фрагментов", len(result))
        return sorted(result, key=lambda chunk: chunk.distance)

    def sync(
        self, chunks: list[Chunk], vectors: list[list[float]], model: str, space_id: str
    ) -> IndexingResult:
        """Обновить снимок базы знаний; убрать устаревшее после успешной записи."""
        if not chunks:
            raise ValueError(
                "Нет фрагментов для индексации; существующий индекс сохранён"
            )
        dimensions = validate_vectors(vectors, len(chunks))
        expected = {}
        for chunk, vector in zip(chunks, vectors, strict=True):
            if any(
                not isinstance(chunk.get(name), str) or not chunk[name].strip()
                for name in _PROPERTIES
            ):
                raise ValueError("Фрагмент содержит пустой текст или метаданные")
            uuid = str(UUID(chunk["chunk_id"]))
            UUID(chunk["document_id"])
            if uuid in expected:
                raise ValueError("Повторяющийся идентификатор фрагмента")
            expected[uuid] = (chunk, vector)

        collection = self._collection(model, space_id, dimensions)
        inserted = updated = deleted = 0
        for uuid, (chunk, vector) in expected.items():
            properties = {name: chunk[name] for name in _PROPERTIES}
            if collection.data.exists(uuid):
                collection.data.replace(uuid=uuid, properties=properties, vector=vector)
                updated += 1
            else:
                collection.data.insert(uuid=uuid, properties=properties, vector=vector)
                inserted += 1

        # Новые данные уже записаны: теперь можно удалить старые версии фрагментов.
        obsolete = [
            str(obj.uuid)
            for obj in collection.iterator(return_properties=["chunk_id"])
            if str(obj.uuid) not in expected
        ]
        for uuid in obsolete:
            collection.data.delete_by_id(uuid)
            deleted += 1

        actual = {
            str(obj.uuid): obj for obj in collection.iterator(include_vector=True)
        }
        if set(actual) != set(expected):
            raise RuntimeError("Состав индекса не совпадает с текущей базой знаний")
        for uuid, (chunk, vector) in expected.items():
            obj = actual[uuid]
            saved_vector = obj.vector.get("default", [])
            if any(obj.properties.get(name) != chunk[name] for name in _PROPERTIES):
                raise RuntimeError("Метаданные фрагмента изменились при сохранении")
            if len(saved_vector) != dimensions or not all(
                math.isclose(saved, original, rel_tol=1e-5, abs_tol=1e-7)
                for saved, original in zip(saved_vector, vector, strict=True)
            ):
                raise RuntimeError(
                    "Сохранённый вектор не совпадает с полученным от API"
                )

        logger.info(
            "Индекс проверен: %d фрагментов, %d измерений", len(actual), dimensions
        )
        return IndexingResult(
            chunks=len(actual),
            dimensions=dimensions,
            inserted=inserted,
            updated=updated,
            deleted=deleted,
            verified=True,
        )
