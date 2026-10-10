import unittest
from unittest.mock import Mock

from rag_pipeline import RagPipeline, bind_sources
from schemas import Evidence, GeneratedAnswer, RetrievedChunk


def chunk(identifier="a", distance=0.4):
    return RetrievedChunk(
        document_id="document",
        source_name="rules.txt",
        chunk_id=identifier,
        section_title="Оплата",
        text="Оплата\nСтоимость доставки — 300 рублей.",
        distance=distance,
    )


def generated(identifier="a", quote="Стоимость доставки — 300 рублей."):
    return GeneratedAnswer(
        answer="Доставка стоит 300 рублей.",
        abstained=False,
        citations=[Evidence(chunk_id=identifier, quote=quote)],
    )


class PipelineTests(unittest.TestCase):
    def pipeline(self, context, response=None):
        retriever = Mock()
        retriever.retrieve.return_value = context
        generator = Mock()
        generator.generate.return_value = response or generated()
        return RagPipeline(retriever, generator), generator

    def test_sources_are_bound_to_actual_retrieved_metadata(self):
        sources = bind_sources(
            generated(quote="Стоимость доставки — 300\nрублей."), [chunk()]
        )
        self.assertEqual(sources[0].source_name, "rules.txt")
        self.assertEqual(sources[0].section_title, "Оплата")
        self.assertEqual(sources[0].chunk_id, "a")

    def test_only_relevant_chunks_are_sent_to_model(self):
        relevant, noise = chunk(), chunk("noise", 0.9)
        pipeline, generator = self.pipeline([relevant, noise])
        result = pipeline.answer("Сколько стоит доставка?")
        generator.generate.assert_called_once_with(
            "Сколько стоит доставка?", [relevant]
        )
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.context, [relevant, noise])
        self.assertEqual(len(result.sources), 1)

    def test_empty_and_weak_retrieval_do_not_call_generator(self):
        for context in [[], [chunk(distance=0.9)]]:
            with self.subTest(context=context):
                pipeline, generator = self.pipeline(context)
                result = pipeline.answer("Вопрос")
                self.assertEqual(result.status, "insufficient_context")
                self.assertEqual(result.sources, [])
                generator.generate.assert_not_called()

    def test_semantic_refusal_is_preserved_for_similar_but_unanswerable_question(self):
        response = GeneratedAnswer(
            answer="В базе нет срока рассмотрения.", abstained=True, citations=[]
        )
        pipeline, _ = self.pipeline([chunk()], response)
        result = pipeline.answer("Сколько длится рассмотрение?")
        self.assertEqual(result.status, "insufficient_context")
        self.assertEqual(result.answer, response.answer)
        self.assertEqual(result.sources, [])

    def test_invented_missing_duplicate_and_heading_only_evidence_is_rejected(self):
        responses = [
            generated("unknown"),
            generated(quote="Цена — 500 рублей."),
            generated(quote="Оплата"),
            generated(quote="  "),
            GeneratedAnswer(answer="300 рублей", abstained=False, citations=[]),
            GeneratedAnswer(
                answer="300 рублей",
                abstained=False,
                citations=[generated().citations[0]] * 2,
            ),
            GeneratedAnswer(
                answer="Не знаю", abstained=True, citations=generated().citations
            ),
        ]
        for response in responses:
            with self.subTest(response=response):
                pipeline, _ = self.pipeline([chunk()], response)
                result = pipeline.answer("Цена?")
                self.assertEqual(result.status, "invalid_output")
                self.assertEqual(result.sources, [])

    def test_bad_model_structure_returns_distinct_failure_status(self):
        pipeline, generator = self.pipeline([chunk()])
        generator.generate.side_effect = ValueError("Invalid JSON")
        result = pipeline.answer("Цена?")
        self.assertEqual(result.status, "invalid_output")
        self.assertNotEqual(result.status, "insufficient_context")

    def test_invalid_search_settings_are_rejected(self):
        for top_k, distance in [
            (0, 0.5),
            (True, 0.5),
            (3, -1),
            (3, 3),
            (3, float("nan")),
            (3, float("inf")),
            (3, True),
        ]:
            with (
                self.subTest(top_k=top_k, distance=distance),
                self.assertRaises(ValueError),
            ):
                RagPipeline(Mock(), Mock(), top_k, distance)


if __name__ == "__main__":
    unittest.main()
