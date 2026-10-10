"""Подключение компонентов приложения и закрытие клиентов."""

import os
from contextlib import contextmanager

from embedder import Embedder
from llm_client import Generator
from rag_pipeline import DEFAULT_MAX_DISTANCE, RagPipeline
from retriever import Retriever
from weaviate_store import WeaviateStore, connect


@contextmanager
def open_pipeline(top_k: int = 3, max_distance: float = DEFAULT_MAX_DISTANCE):
    embedder = Embedder.from_env()
    try:
        generator = Generator.from_env()
        try:
            with connect() as client:
                store = WeaviateStore(
                    client, os.getenv("WEAVIATE_COLLECTION", "KnowledgeChunk")
                )
                yield RagPipeline(
                    Retriever(embedder, store), generator, top_k, max_distance
                )
        finally:
            generator.client.close()
    finally:
        embedder.client.close()
