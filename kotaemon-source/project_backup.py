"""Verified offline/online project backups; restores never overwrite a project."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import uuid

from file_lock import FileLockError, exclusive_file_lock
from runtime_paths import runtime_home


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "backup-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"备份文件校验失败：{name}")
    # A verification must not create or modify a business database.
    with closing(sqlite3.connect(
        (directory / "project.sqlite3").as_uri() + "?mode=ro", uri=True
    )) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("备份SQLite完整性校验失败。")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("备份SQLite引用关系校验失败。")
        for snapshot, expected in db.execute("SELECT snapshot,sha FROM versions"):
            path = (directory / snapshot).resolve(strict=True)
            if not path.is_relative_to(directory) or digest(path) != expected:
                raise ValueError("备份原件快照与数据库哈希不符。")
    return directory


def backup(data_root, project_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", project_id):
        raise ValueError("项目ID无效。")
    source = Path(data_root) / "projects" / project_id
    if not (source / "project.sqlite3").is_file():
        raise ValueError("项目数据库不存在。")
    try:
        lock = exclusive_file_lock(
            source / "service.lock",
            "请先关闭项目服务，再备份；未复制不一致的数据。",
        )
        with lock:
            target = Path(data_root) / "backups" / (
                project_id + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-" + uuid.uuid4().hex[:8]
            )
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(
                "service.lock", "backup-manifest.json", "local-service.json", "service-output.log"
            ))
            manifest = {"project": project_id, "files": {
                str(path.relative_to(target)): digest(path)
                for path in target.rglob("*") if path.is_file()
            }}
            with (target / "backup-manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
            verify(target)
            return target
    except FileLockError:
        raise ValueError("请先关闭项目服务，再备份；未复制不一致的数据。") from None


def restore(backup_path, target):
    source = verify(backup_path)
    target = Path(target).resolve()
    if target.exists():
        raise ValueError("恢复目标必须不存在；禁止覆盖现有项目数据。")
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("恢复目录不能与备份目录互相包含。")
    shutil.copytree(source, target)
    verify(target)
    return target


def live_backup(workspace):
    """Freeze business writes and use SQLite's online backup API."""
    with workspace.lock:
        target = workspace.root.parent.parent / "backups" / (
            workspace.project_id + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid.uuid4().hex[:8]
        )
        shutil.copytree(
            workspace.root, target,
            ignore=shutil.ignore_patterns(
                "service.lock", "project.sqlite3*", "backup-manifest.json",
                "local-service.json", "service-output.log",
            ),
        )
        with workspace.connection() as source, closing(sqlite3.connect(target / "project.sqlite3")) as dest:
            source.backup(dest)
        manifest = {"project": workspace.project_id, "files": {
            str(path.relative_to(target)): digest(path) for path in target.rglob("*") if path.is_file()
        }}
        with (target / "backup-manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        return verify(target)


def available_backups(data_root):
    root = Path(data_root) / "backups"
    return [(path.parent.name, path.parent.name)
            for path in sorted(root.glob("*/backup-manifest.json"), reverse=True)]


def restore_project(data_root, backup_name, identity, authorized_source):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", identity):
        raise ValueError("恢复项目ID无效。")
    base = Path(data_root).resolve()
    source = (base / "backups" / backup_name).resolve(strict=True)
    if source.parent != base / "backups":
        raise ValueError("只允许从本工具备份目录选择备份。")
    verify(source)
    with closing(sqlite3.connect((source / "project.sqlite3").as_uri() + "?mode=ro", uri=True)) as conn:
        binding = conn.execute("SELECT value FROM state WHERE key='source'").fetchone()
    if not binding or Path(binding[0]).resolve() != Path(authorized_source).resolve():
        raise ValueError("备份源目录不在当前授权范围。")
    return restore(source, base / "projects" / identity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="project-demo")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    elif args.restore:
        if args.target is None:
            parser.error("--restore requires --target pointing to a new directory")
        result = restore(args.restore, args.target)
    else:
        result = backup(runtime_home(), args.project)
    print(str(result))


if __name__ == "__main__":
    main()
