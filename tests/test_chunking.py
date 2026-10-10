import re
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from uuid import UUID

from chunking import chunk_documents
from loader import load_documents


class ChunkingTests(unittest.TestCase):
    def load(self, text, source_name="sample.txt"):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / source_name).write_text(text, encoding="utf-8")
            return load_documents(folder)

    def bodies(self, chunks):
        return [chunk["text"].split("\n", 1)[1] for chunk in chunks]

    def test_short_sections_keep_related_facts_together_and_do_not_mix(self):
        documents = self.load(
            "# Сервис\n\n## Город\n"
            "Цена: 300 рублей. Вес: до 5 кг. Срок: 2 дня.\n\n"
            "## Регион\nЦена: 650 рублей. Вес: до 10 кг. Срок: 3–5 дней."
        )
        chunks = chunk_documents(documents)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section_title"], "Сервис / Город")
        self.assertEqual(chunks[1]["section_title"], "Сервис / Регион")
        self.assertIn("300 рублей. Вес: до 5 кг. Срок: 2 дня.", chunks[0]["text"])
        self.assertNotIn("650", chunks[0]["text"])
        self.assertNotIn("300", chunks[1]["text"])

    def test_long_document_preserves_all_words_and_repeats_only_small_suffixes(self):
        words = [f"условие{index:03}" for index in range(200)]
        documents = self.load("# Правила\n\n" + " ".join(words))
        chunks = chunk_documents(documents, chunk_size=250, overlap=35)
        bodies = self.bodies(chunks)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(chunk["text"]) <= 250 for chunk in chunks))
        found = []
        for body in bodies:
            found.extend(body.split())
        self.assertEqual(set(found), set(words))
        self.assertEqual(found[0], words[0])
        self.assertEqual(found[-1], words[-1])
        for previous, current in pairwise(bodies):
            previous_words, current_words = previous.split(), current.split()
            overlap_words = [word for word in current_words if word in previous_words]
            self.assertGreater(len(overlap_words), 0)
            overlap_text = " ".join(overlap_words)
            self.assertLessEqual(len(overlap_text), 35)
            self.assertTrue(previous.endswith(overlap_text))
            self.assertTrue(current.startswith(overlap_text))

    def test_paragraph_and_sentence_boundaries_are_preferred(self):
        first = "Первый абзац содержит полезное правило. " * 2
        second = "Второй абзац содержит другое правило. " * 3
        documents = self.load("# Правила\n\n" + first + "\n\n" + second)
        chunks = chunk_documents(documents, chunk_size=160, overlap=0)
        bodies = self.bodies(chunks)
        self.assertEqual(bodies[0], first.strip())
        self.assertTrue(all(body.endswith(".") for body in bodies))
        self.assertEqual(
            " ".join(" ".join(bodies).split()),
            " ".join((first + second).split()),
        )

    def test_headings_are_kept_in_every_part_of_long_section(self):
        documents = self.load(
            "# Доставка\n## Отмена\n### После забора\n" + "Возврат платный. " * 50
        )
        chunks = chunk_documents(documents, chunk_size=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["section_title"], "Доставка / Отмена / После забора")
            self.assertTrue(chunk["text"].startswith(chunk["section_title"] + "\n"))
            self.assertLessEqual(len(chunk["text"]), 200)

    def test_plain_text_and_long_unbroken_word_are_not_lost(self):
        original = "А" * 500
        documents = self.load(original)
        chunks = chunk_documents(documents, chunk_size=100, overlap=0)
        self.assertEqual("".join(self.bodies(chunks)), original)
        self.assertTrue(all(len(chunk["text"]) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk["section_title"] == "sample.txt" for chunk in chunks))

    def test_ids_are_stable_unique_and_keep_source(self):
        documents = self.load("# Сервис\n## Раздел\nПравило.\n## Раздел\nПравило.")
        chunks = chunk_documents(documents)
        self.assertEqual(chunks, chunk_documents(documents))
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 2)
        self.assertEqual(len({chunk["document_id"] for chunk in chunks}), 1)
        for chunk in chunks:
            UUID(chunk["document_id"])
            UUID(chunk["chunk_id"])
            self.assertEqual(chunk["source_name"], "sample.txt")
        renamed = self.load(documents[0]["text"], "other.txt")
        self.assertNotEqual(
            chunks[0]["document_id"], chunk_documents(renamed)[0]["document_id"]
        )

    def test_invalid_settings_have_clear_errors(self):
        documents = self.load("# Раздел\nТекст.")
        for size, overlap in [(0, 0), (-1, 0), (200, -1), (200, 100), (True, 0)]:
            with (
                self.subTest(size=size, overlap=overlap),
                self.assertRaises(ValueError),
            ):
                chunk_documents(documents, size, overlap)
        with self.assertRaisesRegex(ValueError, "заголовок"):
            chunk_documents(self.load("# " + "А" * 200 + "\nТекст."), 100, 10)

    def test_sample_corpus_preserves_facts_and_metadata(self):
        folder = Path(__file__).resolve().parents[1] / "knowledge_base"
        documents = load_documents(folder)
        chunks = chunk_documents(documents)
        self.assertEqual(len(documents), 5)
        self.assertEqual(
            {chunk["source_name"] for chunk in chunks},
            {document["source_name"] for document in documents},
        )
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))
        self.assertTrue(all(len(chunk["text"]) <= 800 for chunk in chunks))
        for document in documents:
            chunks_text = " ".join(
                chunk["text"]
                for chunk in chunks
                if chunk["source_name"] == document["source_name"]
            )
            source_words = re.findall(r"\w+", document["text"])
            self.assertTrue(set(source_words) <= set(re.findall(r"\w+", chunks_text)))


if __name__ == "__main__":
    unittest.main()
