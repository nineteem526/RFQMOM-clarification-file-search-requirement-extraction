"""Export an already indexed Windows project for a first App Service import."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_backup import backup
from scan_sources import inventory


CLOUD_ROOT = "/home/rfq"


def portable_relative(value):
    value = value.replace("\\", "/")
    parts = value.split("/")
    if (not value or value.startswith("/") or any(part in {"", ".", ".."} for part in parts)
            or ":" in parts[0] or "\x00" in value):
        raise ValueError(f"Unsafe project-relative path: {value!r}")
    return PurePosixPath(*parts).as_posix()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_source(root, row, destination):
    relative = Path(row["path"])
    source = root / relative
    if (not source.resolve(strict=True).is_relative_to(root) or source.is_symlink()
            or getattr(source.lstat(), "st_file_attributes", 0) & 0x400):
        raise ValueError("A source file is a link or has left the authorized directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        before = os.fstat(incoming.fileno())
        digest = hashlib.sha256()
        while chunk := incoming.read(1024 * 1024):
            digest.update(chunk)
            outgoing.write(chunk)
        after = os.fstat(incoming.fileno())
    current = source.stat()
    identity = lambda item: (item.st_size, item.st_mtime_ns, item.st_ino)
    if (any(identity(item) != identity(before) for item in (after, current))
            or before.st_size != row["bytes"] or before.st_mtime_ns != row["mtime_ns"]
            or digest.hexdigest() != row["sha256"]
            or file_hash(destination) != row["sha256"]):
        raise ValueError("A source file changed during export; no archive was published.")
    os.utime(destination, ns=(before.st_atime_ns, before.st_mtime_ns))


def prepare_database(project, source, scan):
    database = project / "project.sqlite3"
    with closing(sqlite3.connect(database)) as conn:
        binding = conn.execute("SELECT value FROM state WHERE key='source'").fetchone()
        if not binding or Path(binding[0]).resolve() != source:
            raise ValueError("The indexed project is bound to a different source folder.")
        if conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]:
            raise ValueError("Only a newly indexed project without saved queries can be exported.")
        versions = conn.execute("SELECT id,path,sha,snapshot,status FROM versions").fetchall()
        if any(row[4] not in {"ready", "limited"} for row in versions):
            raise ValueError("Some document versions failed indexing; review them before export.")
        available = {(portable_relative(path), sha) for _, path, sha, _, _ in versions}
        for path, sha, status in conn.execute(
            "SELECT a.path,a.sha,v.status FROM aliases a "
            "JOIN versions v ON v.id=a.canonical"
        ):
            if status in {"ready", "limited"}:
                available.add((portable_relative(path), sha))
        missing = [
            row for row in scan["files"]
            if (portable_relative(row["path"]), row["sha256"]) not in available
        ]
        if missing:
            raise ValueError(f"{len(missing)} source files have not been successfully indexed.")

        with conn:
            for identity, path, _, snapshot, _ in versions:
                conn.execute(
                    "UPDATE versions SET path=?,snapshot=? WHERE id=?",
                    (portable_relative(path), portable_relative(snapshot), identity),
                )
            for path, sha in conn.execute("SELECT path,sha FROM aliases").fetchall():
                conn.execute(
                    "UPDATE aliases SET path=? WHERE path=? AND sha=?",
                    (portable_relative(path), path, sha),
                )
            conn.execute("UPDATE state SET value=? WHERE key='source'", (CLOUD_ROOT + "/source",))
            conn.execute("DELETE FROM state WHERE key='scan_fingerprint'")
        if (conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or conn.execute("PRAGMA foreign_key_check").fetchall()):
            raise ValueError("Exported project database failed integrity checks.")
        for snapshot, expected in conn.execute("SELECT snapshot,sha FROM versions"):
            path = project.joinpath(*PurePosixPath(snapshot).parts).resolve(strict=True)
            if not path.is_relative_to(project) or file_hash(path) != expected:
                raise ValueError("An exported original snapshot failed hash verification.")


def export_project(project_dir, source_dir, output):
    project_dir = Path(project_dir).resolve(strict=True)
    source_dir = Path(source_dir).resolve(strict=True)
    output = Path(output).resolve()
    if not source_dir.is_dir() or not project_dir.is_dir():
        raise ValueError("Project and source must be existing directories.")
    if project_dir.parent.name != "projects" or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,47}", project_dir.name
    ):
        raise ValueError("Project must be a valid directory under a projects directory.")
    data_root = project_dir.parent.parent
    if output.suffix.lower() != ".zip" or not output.parent.is_dir():
        raise ValueError("Export must be a ZIP in an existing output directory.")
    if output.exists() or output.is_relative_to(source_dir) or output.is_relative_to(data_root):
        raise ValueError("Export must be new and outside the source and project data directories.")
    scan = inventory(source_dir)
    if (scan["failures"] or not scan["files"]
            or any(item["reason"] != "temporary/lock file" for item in scan["ignored"])):
        raise ValueError("Source inventory is incomplete or contains unsafe links.")

    snapshot = backup(data_root, project_dir.name)
    with tempfile.TemporaryDirectory(prefix="rfq-export-", dir=output.parent) as temp:
        payload = Path(temp)
        source = payload / "source"
        project = payload / "data" / "projects" / project_dir.name
        source.mkdir()
        project.mkdir(parents=True)
        for row in scan["files"]:
            destination = source.joinpath(*PurePosixPath(portable_relative(row["path"])).parts)
            copy_source(source_dir, row, destination)
        current = inventory(source_dir)
        expected = {(row["path"], row["sha256"]) for row in scan["files"]}
        actual = {(row["path"], row["sha256"]) for row in current["files"]}
        if (current["failures"] or actual != expected
                or any(item["reason"] != "temporary/lock file" for item in current["ignored"])):
            raise ValueError("The authorized source folder changed during export.")
        for name in ("project.sqlite3", "originals", "scans", "ingestions", "exports"):
            item = snapshot / name
            if item.is_dir():
                shutil.copytree(item, project / name)
            elif item.is_file() and name == "project.sqlite3":
                shutil.copyfile(item, project / name)
        if not (project / "project.sqlite3").is_file() or not (project / "originals").is_dir():
            raise ValueError("Verified project backup is missing required files.")
        prepare_database(project, source_dir, scan)
        audit = project / "migration-audit"
        for name in ("scans", "ingestions"):
            item = project / name
            if item.is_dir():
                audit.mkdir(exist_ok=True)
                item.rename(audit / name)

        files = {
            path.relative_to(payload).as_posix(): {
                "bytes": path.stat().st_size, "sha256": file_hash(path)
            }
            for path in sorted(payload.rglob("*")) if path.is_file()
        }
        manifest = {"schema": 1, "target_root": CLOUD_ROOT, "project": project_dir.name,
                    "source_files": len(scan["files"]), "files": files}
        created = False
        try:
            with output.open("xb") as stream:
                created = True
                with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=6) as archive:
                    archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
                    for name in files:
                        archive.write(payload.joinpath(*PurePosixPath(name).parts), name)
        except Exception:
            if created:
                output.unlink(missing_ok=True)
            raise
    return {"path": str(output), "sha256": file_hash(output), "source_files": len(scan["files"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("project-dir", "source-dir", "output"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_project(args.project_dir, args.source_dir, args.output)))


if __name__ == "__main__":
    main()
