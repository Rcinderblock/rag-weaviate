import tempfile
import unittest
from pathlib import Path

from loader import clean_text, load_documents


class DocumentLoadingTests(unittest.TestCase):
    def test_cleanup_preserves_heading_and_list_boundaries(self):
        original = (
            "  Доставка   по Москве.\r\n\r\n"
            "  - Срок:\t 2 дня.\r\n"
            "  - Цена: 300\u00a0рублей.  \r\n"
        )
        self.assertEqual(
            clean_text(original),
            "Доставка по Москве.\n- Срок: 2 дня.\n- Цена: 300 рублей.",
        )

    def test_loading_preserves_utf8_sources_and_ignores_other_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "b.TXT").write_text("  Срок:   2 дня. ", encoding="utf-8")
            (folder / "a.txt").write_text("Стоимость: 300 рублей.", encoding="utf-8")
            (folder / "notes.md").write_text("Не входит в базу.", encoding="utf-8")
            self.assertEqual(
                load_documents(folder),
                [
                    {
                        "source_name": "a.txt",
                        "text": "Стоимость: 300 рублей.",
                        "paragraphs": ["Стоимость: 300 рублей."],
                    },
                    {
                        "source_name": "b.TXT",
                        "text": "Срок: 2 дня.",
                        "paragraphs": ["Срок: 2 дня."],
                    },
                ],
            )

    def test_bad_and_empty_documents_are_reported_without_losing_valid_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "bad.txt").write_bytes(b"\xff")
            (folder / "empty.txt").write_text(" \n\t\n", encoding="utf-8")
            (folder / "valid.txt").write_text("Условия доставки.", encoding="utf-8")
            with self.assertLogs("loader", level="WARNING") as captured:
                documents = load_documents(folder)
            self.assertEqual(
                documents,
                [
                    {
                        "source_name": "valid.txt",
                        "text": "Условия доставки.",
                        "paragraphs": ["Условия доставки."],
                    }
                ],
            )
            self.assertTrue(any("bad.txt" in message for message in captured.output))
            self.assertTrue(any("empty.txt" in message for message in captured.output))

    def test_paragraph_boundaries_survive_before_empty_line_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "document.txt").write_bytes(
                "# Доставка\r\n \r\nПервый   абзац.\r\nПродолжение.\r\n"
                "\u00a0\r\n\r\nВторой абзац.".encode("utf-8")
            )
            document = load_documents(folder)[0]
            self.assertNotIn("\n\n", document["text"])
            self.assertEqual(
                document["paragraphs"],
                ["# Доставка", "Первый абзац.\nПродолжение.", "Второй абзац."],
            )


if __name__ == "__main__":
    unittest.main()
