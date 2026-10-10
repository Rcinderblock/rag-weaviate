"""Хранение фрагментов, векторов и метаданных в локальной Weaviate."""

import json
import logging
import math
import os
from uuid import UUID

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from chunking import Chunk
from embedder import validate_vectors
from schemas import IndexingResult

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

    def _collection(self, model: str, space_id: str, dimensions: int):
        description = json.dumps(
            {"embedding_model": model, "space_id": space_id, "dimensions": dimensions},
            sort_keys=True,
        )
        if not self.client.collections.exists(self.collection_name):
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
