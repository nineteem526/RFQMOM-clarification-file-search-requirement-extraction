"""Read-only source inventory. A scan is never ingestion approval."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import unicodedata
import uuid

from runtime_paths import source_scan_path

FORMAT_STATUS = {
    ".txt": "本地UTF-8文本行；长行分块需核对相邻内容",
    ".docx": "本地正文段落及表格行；文本框、修订等未覆盖，无页码",
    ".xlsx": "原始单元格坐标及顶部行；多级表头、合并语义及公式计算未完成",
    ".pdf": "本地文本层及物理页码；OCR、表格结构及高亮未完成",
    ".doc": "本机Word只读转换副本；失败明确显示",
    ".xls": "本机Excel只读转换副本；公式不重算，链接不更新",
    ".wps": "尝试本机Word只读转换；不兼容的WPS格式明确失败",
}


def inventory(root, previous=None):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Source must be a directory.")
    if previous and previous.get("source_root") != str(root):
        raise ValueError("Previous inventory belongs to another source directory.")
    old = {row["path"]: row for row in (previous or {}).get("files", [])}
    files, failures, ignored = [], [], []
    visited = set()

    def walk(directory):
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        except OSError as error:
            failures.append({"path": str(directory.relative_to(root)),
                             "error": type(error).__name__})
            return
        for path in children:
            relative = str(path.relative_to(root))
            if path.name.startswith("~$") or path.suffix.lower() in {".tmp", ".lock"}:
                ignored.append({"path": relative, "reason": "temporary/lock file"})
                continue
            try:
                info = path.lstat()
                if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                    ignored.append({"path": relative, "reason": "link/reparse point"})
                    continue
                if stat.S_ISDIR(info.st_mode):
                    walk(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    ignored.append({"path": relative, "reason": "not a regular file"})
                    continue
                with path.open("rb") as stream:
                    before = os.fstat(stream.fileno())
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    after = os.fstat(stream.fileno())
                current = path.stat()
                if any(
                    (entry.st_size, entry.st_mtime_ns, entry.st_ino)
                    != (before.st_size, before.st_mtime_ns, before.st_ino)
                    for entry in (after, current)
                ):
                    raise ValueError("File changed during hashing; rescan required.")
                prior = old.get(relative)
                change = (
                    "新增（相对上次扫描，非入库状态）" if prior is None else
                    "内容变化" if prior["sha256"] != digest else
                    "元信息变化" if prior["mtime_ns"] != before.st_mtime_ns else
                    "未变化"
                )
                files.append({
                    "path": relative, "bytes": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                    "source_modified_utc": datetime.fromtimestamp(
                        before.st_mtime, timezone.utc
                    ).isoformat(),
                    "date_basis": "文件修改时间兜底，未经人工确认",
                    "sha256": digest, "change": change,
                    "parser_status": FORMAT_STATUS.get(
                        path.suffix.lower(), "格式未验证／不支持录入"
                    ),
                    "ingested": False,
                })
                visited.add(relative)
            except (OSError, ValueError) as error:
                failures.append({"path": relative, "error": str(error)})

    walk(root)
    groups = {}
    stems = {}
    for row in files:
        groups.setdefault(row["sha256"], []).append(row["path"])
        path = Path(row["path"])
        if path.suffix.lower() in {".doc", ".docx", ".pdf", ".wps"}:
            normalized = unicodedata.normalize(
                "NFKC", str(path.with_suffix(""))
            ).casefold()
            key = "".join(normalized.split())
            stems.setdefault(key, []).append(row["path"])
    return {
        "schema": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "comparison": "previous scan, NOT imported versions",
        "files": files, "failures": failures, "ignored": ignored,
        "byte_identical_groups": [paths for paths in groups.values() if len(paths) > 1],
        "possible_version_pairs": [
            paths for paths in stems.values()
            if len(paths) > 1 and any(Path(path).suffix.lower() == ".pdf" for path in paths)
        ],
        "previous_paths_not_successfully_seen": sorted(set(old) - visited),
        "coverage_complete": not failures,
        "content_parsed": False, "ai_called": False, "ingestion_approved": False,
    }


def scan_authorized_sources():
    from project_registry import authorized_source

    root = authorized_source()
    reports = source_scan_path()
    reports.mkdir(parents=True, exist_ok=True)
    # Unique immutable reports avoid changing a user's source or an older result.
    existing = sorted(reports.glob("scan-*.json"))
    previous = None
    if existing:
        previous = json.loads(existing[-1].read_text(encoding="utf-8"))
    result = inventory(root, previous)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = reports / f"scan-{name}-{uuid.uuid4().hex[:8]}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result, output = scan_authorized_sources()
    print(json.dumps({
        "report": str(output), "file_count": len(result["files"]),
        "failures": result["failures"], "ignored_count": len(result["ignored"]),
        "byte_identical_groups": result["byte_identical_groups"],
        "possible_version_pairs": result["possible_version_pairs"],
        "ai_called": False, "ingestion_approved": False,
    }, ensure_ascii=False, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
