import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from local_ocr import recognize_page


class LocalOcrRuntimeTests(unittest.TestCase):
    def test_app_service_uses_separate_ocr_environment_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / ".venv" / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            executable.parent.mkdir(parents=True)
            executable.touch()
            result = SimpleNamespace(
                returncode=0, stdout=json.dumps({"page": 1, "lines": []}), stderr=""
            )
            with patch.dict(os.environ, {"RFQ_APP_SERVICE": "true", "DUERR_API_KEY": "fixture"}), \
                    patch("local_ocr.ocr_root", return_value=root), \
                    patch("local_ocr.subprocess.run", return_value=result) as run:
                self.assertEqual(recognize_page(root / "fixture.pdf", 1), [])
            command = run.call_args.args[0]
            self.assertEqual(command[0], str(executable))
            self.assertNotIn("DUERR_API_KEY", run.call_args.kwargs["env"])

    def test_builtin_app_service_uses_current_python_when_no_ocr_venv_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            result = SimpleNamespace(
                returncode=0, stdout=json.dumps({"page": 1, "lines": []}), stderr=""
            )
            with patch.dict(os.environ, {"RFQ_APP_SERVICE": "true"}), \
                    patch("local_ocr.ocr_root", return_value=Path(directory)), \
                    patch("local_ocr.subprocess.run", return_value=result) as run:
                self.assertEqual(recognize_page(Path(directory) / "fixture.pdf", 1), [])
            self.assertEqual(run.call_args.args[0][0], sys.executable)

    def test_worker_prioritizes_isolated_headless_opencv_and_rejects_missing_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stubs = root / "stubs"
            stubs.mkdir()
            for name in ("fitz.py", "numpy.py"):
                (stubs / name).write_text("", encoding="utf-8")
            (stubs / "rapidocr_onnxruntime.py").write_text(
                "import cv2\nraise RuntimeError(cv2.MARKER)\n", encoding="utf-8"
            )
            cv2 = root / "headless" / "cv2"
            cv2.mkdir(parents=True)
            (cv2 / "__init__.py").write_text(
                "MARKER = 'isolated-headless-opencv-loaded'\n", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment.update(PYTHONPATH=str(stubs), PYTHONIOENCODING="utf-8",
                               RFQ_OPENCV_HEADLESS_DIR=str(cv2.parent))
            command = [sys.executable, "-B", str(Path(__file__).resolve().parent.parent
                       / "local_ocr.py"), "--worker", str(root / "fixture.pdf"), "1"]
            result = subprocess.run(command, env=environment, capture_output=True,
                                    encoding="utf-8", timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("isolated-headless-opencv-loaded", result.stderr)

            environment["RFQ_OPENCV_HEADLESS_DIR"] = str(root / "missing")
            result = subprocess.run(command, env=environment, capture_output=True,
                                    encoding="utf-8", timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("headless OpenCV assets are missing or invalid", result.stderr)


if __name__ == "__main__":
    unittest.main()
