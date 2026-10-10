import json
import unittest

import httpx
from openai import APIStatusError, OpenAI

from llm_client import Generator
from tests.test_rag_pipeline import chunk


class GeneratorTests(unittest.TestCase):
    def client(self, content, finish_reason="stop", status=200, record=None):
        def handler(request):
            if record is not None:
                record.append(json.loads(request.content))
            return httpx.Response(
                status,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "error": {"message": "test error"} if status != 200 else None,
                },
            )

        client = OpenAI(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.addCleanup(client.close)
        return client

    def test_sdk_passes_question_context_and_validates_structured_response(self):
        record = []
        content = json.dumps(
            {
                "answer": "300 рублей",
                "abstained": False,
                "citations": [
                    {"chunk_id": "a", "quote": "Стоимость доставки — 300 рублей."}
                ],
            }
        )
        result = Generator(self.client(content, record=record), "test-model").generate(
            "Цена?", [chunk()]
        )
        self.assertFalse(result.abstained)
        payload = json.loads(record[0]["messages"][1]["content"])
        self.assertEqual(payload["question"], "Цена?")
        self.assertEqual(payload["context"][0]["text"], chunk().text)
        self.assertEqual(record[0]["response_format"]["json_schema"]["strict"], True)

    def test_bad_json_duplicate_keys_nonfinite_and_wrong_schema_are_rejected(self):
        for content in [
            "not json",
            "{}",
            '{"answer":"x","answer":"y","abstained":true,"citations":[]}',
            '{"answer":"x","abstained":"false","citations":[]}',
            '{"answer":"x","abstained":false,"citations":[],"extra":NaN}',
            '{"answer":"x","abstained":false,"citations":[],"extra":Infinity}',
            '{"answer":"   ","abstained":true,"citations":[]}',
        ]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                Generator(self.client(content), "test-model").generate("Вопрос", [])

    def test_empty_or_truncated_model_response_is_not_accepted(self):
        for content, finish in [
            (None, "stop"),
            ("", "stop"),
            ("{}", "length"),
            ("{}", "content_filter"),
        ]:
            with (
                self.subTest(content=content, finish=finish),
                self.assertRaises(ValueError),
            ):
                Generator(self.client(content, finish), "test-model").generate(
                    "Вопрос", []
                )

    def test_api_failure_is_not_misrepresented_as_missing_knowledge(self):
        with self.assertRaises(APIStatusError):
            Generator(self.client(None, status=503), "test-model").generate(
                "Вопрос", []
            )


if __name__ == "__main__":
    unittest.main()
