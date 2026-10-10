import json
import unittest

import httpx
from openai import OpenAI

from embedder import Embedder, validate_vectors


class EmbeddingTests(unittest.TestCase):
    def client(self, handler):
        client = OpenAI(
            api_key="test-key",
            base_url="https://embedding.example/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        self.addCleanup(client.close)
        return client

    def test_sdk_response_order_and_batches_keep_vectors_matched_to_text(self):
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            data = [
                {
                    "index": index,
                    "embedding": [float(len(text)), 1.0],
                    "object": "embedding",
                }
                for index, text in enumerate(payload["input"])
            ]
            return httpx.Response(200, json={"data": list(reversed(data))})

        embedder = Embedder(self.client(handler), "test-model")
        vectors = embedder.embed(["Раз", "Два слова", "Три слова тут"], batch_size=2)
        self.assertEqual(vectors, [[3.0, 1.0], [9.0, 1.0], [13.0, 1.0]])
        self.assertEqual([len(request["input"]) for request in requests], [2, 1])
        self.assertTrue(all(request["model"] == "test-model" for request in requests))

    def test_bad_api_indices_counts_and_missing_vectors_are_rejected(self):
        invalid_data = [
            [{"index": 0, "embedding": [1.0]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
            [{"index": 0, "embedding": [1.0]}, {"index": 3, "embedding": [2.0]}],
            [{"index": 0}, {"index": 1, "embedding": [2.0]}],
        ]
        for data in invalid_data:
            with self.subTest(data=data):
                client = self.client(
                    lambda request, data=data: httpx.Response(200, json={"data": data})
                )
                with self.assertRaises(ValueError):
                    Embedder(client, "test-model").embed(["Первый", "Второй"])

    def test_invalid_vectors_are_rejected_before_storage(self):
        for vectors in [
            [],
            [[]],
            [[0.0, 0.0]],
            [[float("nan")]],
            [[float("inf")]],
            [[True]],
            [["1"]],
            [None],
            [[1.0], [1.0, 2.0]],
        ]:
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                validate_vectors(vectors, max(1, len(vectors)))
        self.assertEqual(validate_vectors([[1.0, -2.0], [3.0, 4.0]], 2), 2)

    def test_dimensions_cannot_change_between_api_batches(self):
        responses = iter(
            [
                {"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
                {"data": [{"index": 0, "embedding": [3.0]}]},
            ]
        )
        client = self.client(lambda request: httpx.Response(200, json=next(responses)))
        with self.assertRaisesRegex(ValueError, "размерность"):
            Embedder(client, "test-model").embed(["Первый", "Второй"], batch_size=1)

    def test_empty_input_does_not_call_api_and_blank_text_is_rejected(self):
        requests = []
        client = self.client(lambda request: requests.append(request))
        embedder = Embedder(client, "test-model")
        self.assertEqual(embedder.embed([]), [])
        with self.assertRaises(ValueError):
            embedder.embed(["  "])
        with self.assertRaises(ValueError):
            embedder.embed(["Текст"], batch_size=0)
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
