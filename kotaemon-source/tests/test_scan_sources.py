from pathlib import Path
import tempfile
import unittest

from scan_sources import inventory


class ScanTests(unittest.TestCase):
    def test_inventory_changes_duplicates_and_no_source_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "协议.docx").write_bytes(b"fixture")
            (root / "协 议.pdf").write_bytes(b"other")
            (root / "copy.docx").write_bytes(b"fixture")
            (root / "~$协议.docx").write_bytes(b"lock")
            first = inventory(root)
            self.assertEqual(len(first["files"]), 3)
            self.assertEqual(len(first["ignored"]), 1)
            self.assertEqual(len(first["byte_identical_groups"]), 1)
            self.assertEqual(len(first["possible_version_pairs"]), 1)
            self.assertFalse(first["ingestion_approved"])
            second = inventory(root, first)
            self.assertTrue(all(row["change"] == "未变化" for row in second["files"]))
            (root / "协 议.pdf").write_bytes(b"changed")
            third = inventory(root, second)
            row = next(row for row in third["files"] if row["path"] == "协 议.pdf")
            self.assertEqual(row["change"], "内容变化")
            self.assertEqual((root / "协议.docx").read_bytes(), b"fixture")

    def test_wrong_root_rejected_and_empty_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(inventory(directory)["files"], [])
            with self.assertRaises(ValueError):
                inventory(directory, {"source_root": "different"})


if __name__ == "__main__":
    unittest.main()
