from pathlib import Path
import unittest

from query_cache import PIPELINE_FILES, pipeline_fingerprint


class PipelineFingerprintTests(unittest.TestCase):
    def test_pipeline_files_can_be_fingerprinted_and_are_staged_for_the_container(self):
        root = Path(__file__).resolve().parent.parent
        files = set((root / "environment-setup" / "appservice-context-files.txt").read_text(
            encoding="utf-8"
        ).splitlines())
        self.assertTrue(set(PIPELINE_FILES).issubset(files))
        fingerprint = pipeline_fingerprint()
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(fingerprint, pipeline_fingerprint())


if __name__ == "__main__":
    unittest.main()
