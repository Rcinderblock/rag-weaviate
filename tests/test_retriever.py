import unittest
from unittest.mock import Mock

from retriever import Retriever


class RetrievalTests(unittest.TestCase):
    def components(self):
        embedder, store = Mock(), Mock()
        embedder.model = "test-model"
        embedder.space_id = "test-space"
        embedder.embed.return_value = [[1.0, 2.0]]
        store.check_model.return_value = 2
        return embedder, store

    def test_invalid_question_and_limits_do_not_call_embedding_api(self):
        embedder, store = self.components()
        for question, top_k in [
            ("", 3),
            ("  ", 3),
            (None, 3),
            ("Вопрос", 0),
            ("Вопрос", True),
        ]:
            with (
                self.subTest(question=question, top_k=top_k),
                self.assertRaises(ValueError),
            ):
                Retriever(embedder, store).retrieve(question, top_k)
        embedder.embed.assert_not_called()

    def test_different_model_is_rejected_before_embedding_request(self):
        embedder, store = self.components()
        store.check_model.side_effect = ValueError("Другая модель")
        with self.assertRaises(ValueError):
            Retriever(embedder, store).retrieve("Вопрос")
        embedder.embed.assert_not_called()

    def test_wrong_query_vector_dimension_is_not_sent_to_database(self):
        embedder, store = self.components()
        embedder.embed.return_value = [[1.0]]
        with self.assertRaises(ValueError):
            Retriever(embedder, store).retrieve("Вопрос")
        store.search.assert_not_called()

    def test_query_vector_and_limit_are_forwarded_to_database(self):
        embedder, store = self.components()
        store.search.return_value = []
        self.assertEqual(Retriever(embedder, store).retrieve(" Вопрос ", 2), [])
        embedder.embed.assert_called_once_with(["Вопрос"])
        store.search.assert_called_once_with([1.0, 2.0], "test-model", "test-space", 2)


if __name__ == "__main__":
    unittest.main()
