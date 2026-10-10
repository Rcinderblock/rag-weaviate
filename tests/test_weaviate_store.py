"""Проверка записи в настоящую локальную базу на тестовых векторах."""

import os
import unittest
from uuid import NAMESPACE_URL, uuid4, uuid5

from weaviate_store import WeaviateStore, connect


@unittest.skipUnless(
    os.getenv("TEST_WEAVIATE") == "1", "Нужен TEST_WEAVIATE=1 и локальный Weaviate"
)
class WeaviateIntegrationTests(unittest.TestCase):
    def test_reindex_updates_removes_obsolete_and_rejects_another_model(self):
        name = "IndexTest" + uuid4().hex
        document_id = str(uuid5(NAMESPACE_URL, "test.txt"))

        def chunk(text):
            return {
                "document_id": document_id,
                "source_name": "test.txt",
                "chunk_id": str(uuid5(NAMESPACE_URL, text)),
                "section_title": "Правила",
                "text": text,
            }

        first, second = chunk("Первое правило"), chunk("Второе правило")
        with connect() as client:
            store = WeaviateStore(client, name)
            try:
                result = store.sync(
                    [first, second],
                    [[1.0, 2.0], [3.0, 4.0]],
                    "test-model",
                    "test-space",
                )
                self.assertEqual(
                    (result.inserted, result.updated, result.chunks), (2, 0, 2)
                )
                result = store.sync(
                    [first, second],
                    [[1.0, 2.0], [3.0, 4.0]],
                    "test-model",
                    "test-space",
                )
                self.assertEqual(
                    (result.inserted, result.updated, result.chunks), (0, 2, 2)
                )
                found = store.search([1.0, 2.0], "test-model", "test-space", top_k=1)
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0].chunk_id, first["chunk_id"])
                self.assertEqual(found[0].source_name, "test.txt")
                self.assertTrue(abs(found[0].distance) < 1e-5)

                changed = chunk("Обновлённое правило")
                result = store.sync([changed], [[5.0, 6.0]], "test-model", "test-space")
                self.assertEqual(
                    (result.inserted, result.deleted, result.chunks), (1, 2, 1)
                )
                for model, space in [
                    ("other-model", "test-space"),
                    ("test-model", "other-space"),
                ]:
                    with (
                        self.subTest(model=model, space=space),
                        self.assertRaises(ValueError),
                    ):
                        store.sync([changed], [[5.0, 6.0]], model, space)
                with self.assertRaises(ValueError):
                    store.sync([changed], [[5.0]], "test-model", "test-space")
                with self.assertRaises(ValueError):
                    store.sync([], [], "test-model", "test-space")
                self.assertEqual(
                    client.collections.use(name)
                    .aggregate.over_all(total_count=True)
                    .total_count,
                    1,
                )
            finally:
                if client.collections.exists(name):
                    client.collections.delete(name)

    def test_search_does_not_create_missing_index(self):
        name = "MissingIndexTest" + uuid4().hex
        with connect() as client:
            store = WeaviateStore(client, name)
            with self.assertRaises(ValueError):
                store.search([1.0, 2.0], "test-model", "test-space")
            self.assertFalse(client.collections.exists(name))


if __name__ == "__main__":
    unittest.main()
