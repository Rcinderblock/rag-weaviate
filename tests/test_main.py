import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from main import main


class IndexingCommandTests(unittest.TestCase):
    def test_incomplete_document_loading_stops_before_api_and_database_calls(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "valid.txt").write_text("Правила доставки.", encoding="utf-8")
            (folder / "broken.txt").write_bytes(b"\xff")
            output = io.StringIO()
            with (
                patch("sys.argv", ["main.py", str(folder), "--index"]),
                patch("main.load_dotenv"),
                patch("main.Embedder.from_env") as embedder,
                patch("main.connect") as connect,
                redirect_stderr(output),
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("broken.txt", output.getvalue())
            embedder.assert_not_called()
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
