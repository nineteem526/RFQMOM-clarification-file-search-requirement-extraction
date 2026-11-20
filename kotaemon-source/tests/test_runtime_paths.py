import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from runtime_paths import (
    embedding_model_path, nltk_data_path, ocr_root, runtime_home, runtime_venv,
    tiktoken_data_path,
)
from preview_kotaemon import icon_directory


class RuntimePathTests(unittest.TestCase):
    def test_existing_windows_default_and_explicit_runtime_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LOCALAPPDATA": directory}, clear=True):
                self.assertEqual(runtime_home(), Path(directory) / "RFQKotaemon")
                self.assertEqual(
                    tiktoken_data_path(), Path(directory) / "RFQKotaemon" / "resources" / "tiktoken"
                )
            with patch.dict(os.environ, {
                "RFQ_RUNTIME_HOME": directory,
                "RFQ_RUNTIME_VENV": str(Path(directory) / "python"),
                "RFQ_EMBEDDING_DIR": str(Path(directory) / "models"),
                "RFQ_TIKTOKEN_DATA": str(Path(directory) / "tokens"),
            }, clear=True):
                self.assertEqual(runtime_home(), Path(directory))
                self.assertEqual(runtime_venv(), Path(directory) / "python")
                self.assertEqual(tiktoken_data_path(), Path(directory) / "tokens")
                self.assertEqual(
                    embedding_model_path("unused"),
                    Path(directory) / "models",
                )

    def test_linux_model_default_uses_runtime_home(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"RFQ_RUNTIME_HOME": directory}, clear=True):
                self.assertEqual(
                    embedding_model_path(
                        "%LOCALAPPDATA%\\RFQKotaemon\\models\\bge"
                    ),
                    Path(directory) / "models" / "bge",
                )

    def test_app_service_defaults_read_only_assets_from_deployment_and_data_from_home(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {
                "RFQ_APP_SERVICE": "true",
                "APP_PATH": directory,
            }, clear=True):
                self.assertEqual(runtime_home(), Path("/home/rfq-data"))
                self.assertEqual(runtime_venv(), Path(sys.prefix))
                self.assertEqual(nltk_data_path(), Path(directory) / "_appservice-assets" / "nltk")
                self.assertEqual(
                    tiktoken_data_path(),
                    Path(directory) / "_appservice-assets" / "tiktoken",
                )
                self.assertEqual(
                    embedding_model_path("unused"),
                    Path(directory) / "_appservice-assets" / "embedding",
                )
                self.assertEqual(
                    ocr_root(), Path(directory) / "_appservice-assets" / "ocr",
                )

    def test_icon_assets_support_both_builtin_python_and_custom_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_icons = root / "kotaemon" / "libs" / "ktem" / "ktem" / "assets" / "icons"
            packaged_icons = root / "app" / "_appservice-assets" / "icons"
            for path in (source_icons, packaged_icons):
                path.mkdir(parents=True)
                for name in ("dark_mode", "expand", "rename", "delete", "new"):
                    (path / f"{name}.svg").touch()
            with patch("preview_kotaemon.sys.prefix", str(root / ".venv")):
                with patch.dict(os.environ, {"RFQ_APP_SERVICE": "true"}, clear=True):
                    self.assertEqual(icon_directory(), source_icons)
                with patch.dict(os.environ, {
                    "RFQ_APP_SERVICE": "true", "APP_PATH": str(root / "app"),
                }, clear=True):
                    self.assertEqual(icon_directory(), packaged_icons)

    def test_relative_override_and_missing_runtime_root_are_rejected(self):
        with patch.dict(os.environ, {"RFQ_RUNTIME_HOME": "relative"}, clear=True):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                runtime_home()
        with patch.dict(os.environ, {"RFQ_TIKTOKEN_DATA": "relative"}, clear=True):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                tiktoken_data_path()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "RFQ_RUNTIME_HOME"):
                runtime_home()


if __name__ == "__main__":
    unittest.main()
