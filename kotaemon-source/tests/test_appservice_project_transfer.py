from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "environment-setup"))
import export_appservice_project as exporter
import import_appservice_project as importer


class AppServiceProjectTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.project = self.root / "data" / "projects" / "fixture"
        self.source.mkdir()
        for directory in ("originals/id", "scans", "ingestions", "exports"):
            (self.project / directory).mkdir(parents=True, exist_ok=True)
        original = self.source / "资料" / "document.txt"
        original.parent.mkdir()
        original.write_bytes(b"fixture evidence")
        (self.source / "duplicate.txt").write_bytes(original.read_bytes())
        (self.source / "~$lock.docx").write_bytes(b"temporary")
        snapshot = self.project / "originals" / "id" / original.name
        snapshot.write_bytes(original.read_bytes())
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        self.database = self.project / "project.sqlite3"
        with closing(sqlite3.connect(self.database)) as conn, conn:
            conn.executescript("""
                CREATE TABLE state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE versions(id TEXT PRIMARY KEY, path TEXT, sha TEXT,
                    snapshot TEXT, status TEXT, UNIQUE(path,sha));
                CREATE TABLE aliases(path TEXT, sha TEXT, canonical TEXT REFERENCES versions(id),
                    PRIMARY KEY(path,sha));
                CREATE TABLE results(id TEXT PRIMARY KEY);
            """)
            conn.execute("INSERT INTO state VALUES ('source',?)", (str(self.source.resolve()),))
            conn.execute("INSERT INTO state VALUES ('scan_fingerprint','old')")
            conn.execute("INSERT INTO versions VALUES (?,?,?,?,?)", (
                "id", str(original.relative_to(self.source)), digest,
                str(snapshot.relative_to(self.project)), "ready",
            ))
            conn.execute("INSERT INTO aliases VALUES (?,?,?)", ("duplicate.txt", digest, "id"))
        (self.project / "scans" / "scan.json").write_text('{"audit":"scan"}', encoding="utf-8")
        (self.project / "ingestions" / "receipt.json").write_text(
            '{"audit":"receipt"}', encoding="utf-8"
        )
        self.archive = self.root / "transfer.zip"

    def export(self):
        return exporter.export_project(self.project, self.source, self.archive)

    def test_round_trip_preserves_source_and_normalizes_paths(self):
        before = (self.source / "资料" / "document.txt").read_bytes()
        result = self.export()
        self.assertEqual(result["source_files"], 2)
        target = self.root / "imported"
        importer.import_project(self.archive, result["sha256"], target)
        project = target / "data" / "projects" / "fixture"
        with closing(sqlite3.connect(project / "project.sqlite3")) as conn:
            path, snapshot = conn.execute("SELECT path,snapshot FROM versions").fetchone()
            self.assertEqual(path, "资料/document.txt")
            self.assertEqual(snapshot, "originals/id/document.txt")
            self.assertEqual(
                conn.execute("SELECT value FROM state WHERE key='source'").fetchone()[0],
                "/home/rfq/source",
            )
            self.assertIsNone(
                conn.execute("SELECT value FROM state WHERE key='scan_fingerprint'").fetchone()
            )
        self.assertEqual((self.source / "资料" / "document.txt").read_bytes(), before)
        self.assertEqual((target / "source" / "资料" / "document.txt").read_bytes(), before)
        self.assertFalse((target / "source" / "~$lock.docx").exists())
        self.assertTrue((project / "migration-audit" / "scans" / "scan.json").is_file())
        self.assertTrue((project / "migration-audit" / "ingestions" / "receipt.json").is_file())
        self.assertEqual(list((project / "scans").iterdir()), [])

    def test_existing_archive_and_destination_are_never_overwritten(self):
        self.archive.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.export()
        self.assertEqual(self.archive.read_bytes(), b"preserve")
        self.archive.unlink()
        result = self.export()
        target = self.root / "existing"
        target.mkdir()
        (target / "marker").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            importer.import_project(self.archive, result["sha256"], target)
        self.assertEqual((target / "marker").read_bytes(), b"preserve")

    def test_an_existing_target_created_during_validation_is_not_replaced(self):
        result = self.export()
        target = self.root / "raced"
        commit = importer.rename_without_replacement

        def race(source, destination):
            destination.mkdir()
            (destination / "marker").write_bytes(b"preserve")
            commit(source, destination)

        with patch.object(importer, "rename_without_replacement", side_effect=race):
            with self.assertRaises(OSError):
                importer.import_project(self.archive, result["sha256"], target)
        self.assertEqual((target / "marker").read_bytes(), b"preserve")

    def test_missing_index_and_damaged_snapshot_fail_before_export(self):
        with closing(sqlite3.connect(self.database)) as conn, conn:
            conn.execute("DELETE FROM aliases")
        with self.assertRaisesRegex(ValueError, "not been successfully indexed"):
            self.export()
        self.assertFalse(self.archive.exists())
        with closing(sqlite3.connect(self.database)) as conn, conn:
            digest = conn.execute("SELECT sha FROM versions").fetchone()[0]
            conn.execute("INSERT INTO aliases VALUES (?,?,?)", ("duplicate.txt", digest, "id"))
        (self.project / "originals" / "id" / "document.txt").write_bytes(b"damaged")
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.archive.exists())

    def test_corruption_and_path_traversal_fail_without_installation(self):
        result = self.export()
        target = self.root / "imported"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            importer.import_project(self.archive, "0" * 64, target)
        bad = self.root / "bad.zip"
        content = b"bad"
        name = "source/../../outside"
        manifest = {
            "schema": 1, "target_root": "/home/rfq", "project": "fixture",
            "source_files": 1,
            "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}},
        }
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr(name, content)
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            importer.import_project(bad, importer.file_hash(bad), target)
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
