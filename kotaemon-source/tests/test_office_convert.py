from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from office_convert import convert_snapshot


class OfficeConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "snapshot.doc"
        self.path.write_bytes(b"\xd0\xcf\x11\xe0" + b"synthetic")

    def test_unrecognized_header_does_not_start_office(self):
        self.path.write_bytes(b"not an Office document")
        with patch("office_convert.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "文件头"):
                convert_snapshot(self.path)
        run.assert_not_called()

    def test_linux_legacy_conversion_is_explicitly_rejected(self):
        with patch("office_convert.sys.platform", "linux"), \
                patch("office_convert.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "需要Windows版Microsoft Office"):
                convert_snapshot(self.path)
        run.assert_not_called()

    def test_worker_error_is_explicit_and_does_not_modify_source(self):
        original = self.path.read_bytes()
        result = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"fixture conversion error")
        with patch("office_convert.subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(ValueError, "fixture conversion error"):
                convert_snapshot(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(run.call_args.kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW)
        self.assertNotIn("DUERR_API_KEY", run.call_args.kwargs["env"])

    def test_timeout_without_office_pid_is_explicit(self):
        with patch("office_convert.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("fixture", 90)):
            with self.assertRaisesRegex(ValueError, "超时"):
                convert_snapshot(self.path)

    def test_receipt_keeps_original_hash_and_new_file_relation(self):
        def convert(command, **kwargs):
            Path(command[5]).write_bytes(b"synthetic converted content")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        with patch("office_convert.subprocess.run", side_effect=convert):
            target, receipt = convert_snapshot(self.path)
        self.assertTrue(target.exists())
        self.assertEqual(receipt["source"], self.path.name)
        self.assertEqual(receipt["converted"], target.name)
        self.assertTrue(receipt["read_only_source"])


if __name__ == "__main__":
    unittest.main()
