"""Verify and install an exported project into a previously unused App Service home."""

import argparse
from contextlib import closing
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile


CLOUD_ROOT = "/home/rfq"
PROJECT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,47}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_name(name):
    parts = name.split("/")
    if (not name or name.startswith("/") or "\\" in name or "\x00" in name
            or any(part in {"", ".", ".."} for part in parts) or ":" in parts[0]):
        raise ValueError("Archive contains an unsafe path.")
    return PurePosixPath(*parts)


def validate_project(payload, manifest):
    project = payload / "data" / "projects" / manifest["project"]
    database = project / "project.sqlite3"
    if not database.is_file():
        raise ValueError("Archive is missing the project database.")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        if (conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or conn.execute("PRAGMA foreign_key_check").fetchall()):
            raise ValueError("Imported database failed integrity checks.")
        binding = conn.execute("SELECT value FROM state WHERE key='source'").fetchone()
        if not binding or binding[0] != CLOUD_ROOT + "/source":
            raise ValueError("Imported database has an unexpected source binding.")
        if conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]:
            raise ValueError("Imported database unexpectedly contains saved queries.")
        versions = conn.execute("SELECT path,sha,snapshot,status FROM versions").fetchall()
        available = set()
        for path, digest, snapshot, status in versions:
            path = relative_name(path).as_posix()
            snapshot = relative_name(snapshot)
            if status not in {"ready", "limited"}:
                raise ValueError("Imported database contains failed document versions.")
            original = project.joinpath(*snapshot.parts).resolve(strict=True)
            if not original.is_relative_to(project) or file_hash(original) != digest:
                raise ValueError("An imported original snapshot failed hash verification.")
            available.add((path, digest))
        for path, digest, status in conn.execute(
            "SELECT a.path,a.sha,v.status FROM aliases a "
            "JOIN versions v ON v.id=a.canonical"
        ):
            if status in {"ready", "limited"}:
                available.add((relative_name(path).as_posix(), digest))
        for name, metadata in manifest["files"].items():
            if name.startswith("source/"):
                if (name[len("source/"):], metadata["sha256"]) not in available:
                    raise ValueError("An imported source file is not indexed.")
    for name in ("scans", "ingestions", "exports"):
        (project / name).mkdir(exist_ok=True)


def rename_without_replacement(source, target):
    if os.name == "nt":
        os.rename(source, target)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("Atomic no-replacement directory rename is unavailable.")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def import_project(archive_path, expected_sha, target=None):
    archive_path = Path(archive_path).resolve(strict=True)
    if not SHA256.fullmatch(expected_sha) or file_hash(archive_path) != expected_sha:
        raise ValueError("Transfer archive SHA-256 does not match the local export.")
    target = Path(target) if target is not None else Path(CLOUD_ROOT)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Import target already exists; refusing to overwrite it.")
    parent = target.parent.resolve(strict=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (len(names) != len(set(names)) or "manifest.json" not in names
                or archive.getinfo("manifest.json").file_size > 8 * 1024 * 1024):
            raise ValueError("Archive has duplicate entries or an invalid manifest.")
        manifest = json.loads(archive.read("manifest.json"))
        if (manifest.get("schema") != 1 or manifest.get("target_root") != CLOUD_ROOT
                or not isinstance(manifest.get("project"), str)
                or not PROJECT_ID.fullmatch(manifest["project"])
                or not isinstance(manifest.get("files"), dict)):
            raise ValueError("Archive manifest has an unsupported project or target.")
        files = manifest["files"]
        if set(names) != set(files) | {"manifest.json"}:
            raise ValueError("Archive entries do not match the manifest.")
        project_prefix = f"data/projects/{manifest['project']}/"
        required_bytes = 0
        for name, metadata in files.items():
            relative_name(name)
            entry = archive.getinfo(name)
            mode = (entry.external_attr >> 16) & 0o170000
            if (not (name.startswith("source/") or name.startswith(project_prefix))
                    or not isinstance(metadata, dict)
                    or type(metadata.get("bytes")) is not int or metadata["bytes"] < 0
                    or not isinstance(metadata.get("sha256"), str)
                    or not SHA256.fullmatch(metadata["sha256"])
                    or entry.file_size != metadata["bytes"] or entry.is_dir()
                    or mode == stat.S_IFLNK or entry.flag_bits & 1):
                raise ValueError("Archive contains an invalid entry.")
            required_bytes += metadata["bytes"]
        if (type(manifest.get("source_files")) is not int
                or manifest["source_files"] < 1
                or sum(name.startswith("source/") for name in files) != manifest["source_files"]
                or required_bytes > shutil.disk_usage(parent).free):
            raise ValueError("Archive source count or available disk space is invalid.")

        with tempfile.TemporaryDirectory(prefix=".rfq-import-", dir=parent) as temp:
            payload = Path(temp) / "payload"
            payload.mkdir()
            for name, metadata in files.items():
                destination = payload.joinpath(*relative_name(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(name) as incoming, destination.open("xb") as outgoing:
                    while chunk := incoming.read(1024 * 1024):
                        digest.update(chunk)
                        outgoing.write(chunk)
                if digest.hexdigest() != metadata["sha256"]:
                    raise ValueError("An imported file failed hash verification.")
            validate_project(payload, manifest)
            rename_without_replacement(payload, target)
    return {"target": str(target), "project": manifest["project"],
            "source_files": manifest["source_files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(import_project(args.archive, args.sha256)))


if __name__ == "__main__":
    main()
