"""Вызов модели и проверка структуры её ответа."""

import json
import logging
import os

from openai import OpenAI

from prompts import build_messages
from schemas import GeneratedAnswer, RetrievedChunk

logger = logging.getLogger(__name__)


def _reject_constant(value: str):
    raise ValueError("JSON содержит недопустимую числовую константу")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON содержит повторяющиеся ключи")
        result[key] = value
    return result


class Generator:
    def __init__(self, client: OpenAI, model: str):
        if not model.strip():
            raise ValueError("Не задана модель генерации")
        self.client = client
        self.model = model.strip()

    @classmethod
    def from_env(cls) -> "Generator":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("Укажите OPENAI_API_KEY в .env или переменных окружения")
        return cls(
            OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                timeout=60,
                max_retries=2,
            ),
            os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        )

    def generate(self, question: str, context: list[RetrievedChunk]) -> GeneratedAnswer:
        logger.info("Генерация ответа по %d фрагментам", len(context))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(question, context),
            temperature=0,
            max_tokens=1200,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_answer",
                    "strict": True,
                    "schema": GeneratedAnswer.model_json_schema(),
                },
            },
        )
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ValueError("Модель не завершила ответ")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Модель вернула пустой ответ")
        data = json.loads(
            content, parse_constant=_reject_constant, object_pairs_hook=_unique_keys
        )
        return GeneratedAnswer.model_validate(data)
