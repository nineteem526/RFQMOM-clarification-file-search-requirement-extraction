"""Modern multi-project FastAPI web server for the RFQ Engineering Knowledge Workbench.

Each project uses an isolated directory and independent SQLite database under
%LOCALAPPDATA%\\RFQKotaemon\\projects\\<project_id>\\ so switching projects in the
header immediately switches the active database, documents, chats, comparisons,
saved results, and settings.
"""

import argparse
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from html import escape
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import sys
import uuid

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pilot_runtime import prepare_runtime
from runtime_paths import ocr_root
from document_chunker import (
    IMAGE_SUFFIXES,
    ParsedDocument,
    bind_table_row as _bind_table_row,
    parse_document as _parse_document_impl,
)
from rag_engine import (
    embed_text,
    embedding_health,
    generate_structured_answer,
    hybrid_search,
    llm_config_status,
    load_local_llm_config,
    masked_local_llm_config,
    pack_vector,
    save_local_llm_config,
    split_text_with_overlap,
    update_admin_pin,
    verify_admin_pin,
)

DEFAULT_SETTINGS = {
    "chat_model": "GPT-4o（企业内网）",
    "embedding_model": "BAAI/bge-small-zh-v1.5（本地离线）",
    "retrieval_mode": "混合检索（语义 + 关键词）",
    "similarity_threshold": 0.62,
    "top_k": 10,
    "ocr_enabled": True,
    "online_answers_enabled": False,
    "chunk_size": 480,
    "chunk_overlap": 64,
    "data_path": "",
}

DEFAULT_FOLDERS = [
    "RFQ 询价文件",
    "技术协议 / B12",
    "澄清文件",
    "会议纪要",
    "其他资料",
]
QUERY_CACHE_FORMAT = 3
QUERY_MODES = {"快速问答": 0, "扩展候选检索": 12, "更多候选检索": 20}
LEGACY_QUERY_MODES = {"主题逐文件检查": "扩展候选检索", "深度全文检查": "更多候选检索"}
CHUNKS_FOR_DISPLAY = (
    "SELECT id, doc_id, page, section, content, highlight, "
    "COALESCE(chunk_type, 'text') AS chunk_type FROM chunks "
    "WHERE doc_id=? ORDER BY page, rowid"
)


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _today_cn():
    now = datetime.now()
    return f"{now.month}月{now.day}日"


def _safe_project_id(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower()).strip("-")
    if not slug:
        slug = f"proj-{uuid.uuid4().hex[:6]}"
    return slug[:48]


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _parse_document(
    path: Path,
    chunk_size: int = 480,
    chunk_overlap: int = 64,
    ocr_enabled: bool = True,
) -> ParsedDocument:
    """Extract readable text/table/OCR chunks using specialized per-format parsers."""
    return _parse_document_impl(
        path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
    )


def _extract_text_from_file(path: Path, chunk_size=480, chunk_overlap=64, ocr_enabled=True) -> list[dict]:
    """Keep the retrieval benchmark on the same parser as uploads."""
    return _parse_document(path, chunk_size, chunk_overlap, ocr_enabled).chunks


class ProjectStore:
    """Isolated per-project SQLite database and file repository."""

    def __init__(self, runtime_root: Path, project_id: str):
        self.runtime_root = runtime_root
        self.project_id = project_id
        self.root = runtime_root / "projects" / project_id
        self.files_dir = self.root / "files"
        self.backups_dir = self.root / "backups"
        self.exports_dir = self.root / "exports"
        self.db_path = self.root / "workbench.db"
        self.meta_path = self.root / "project_meta.json"
        for d in (self.root, self.files_dir, self.backups_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS folders (
                    name TEXT PRIMARY KEY,
                    sort_order INTEGER DEFAULT 100
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    folder TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    version TEXT DEFAULT 'Rev.0',
                    pages INTEGER DEFAULT 1,
                    size_bytes INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    status TEXT DEFAULT '已索引',
                    indexed INTEGER DEFAULT 1,
                    file_path TEXT DEFAULT '',
                    parse_warnings TEXT DEFAULT '[]',
                    sha256 TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page INTEGER DEFAULT 1,
                    section TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    highlight TEXT DEFAULT '',
                    chunk_type TEXT DEFAULT 'text',
                    embedding BLOB DEFAULT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT DEFAULT '',
                    role TEXT NOT NULL,
                    question TEXT DEFAULT '',
                    time_str TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS saved_items (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    sources_text TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES ('_revision', '1')")
            for idx, folder in enumerate(DEFAULT_FOLDERS):
                conn.execute(
                    "INSERT OR IGNORE INTO folders(name, sort_order) VALUES (?, ?)",
                    (folder, idx),
                )
            # Migrate existing project databases cleanly if embedding or chunk_type column is missing
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
            if "embedding" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB DEFAULT NULL")
            if "chunk_type" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN chunk_type TEXT DEFAULT 'text'")
            chat_cols = {row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)")}
            if "session_id" not in chat_cols:
                conn.execute("ALTER TABLE chat_messages ADD COLUMN session_id TEXT DEFAULT ''")
            self._adopt_orphan_messages(conn)
            doc_cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
            if "parse_warnings" not in doc_cols:
                conn.execute("ALTER TABLE documents ADD COLUMN parse_warnings TEXT DEFAULT '[]'")
                # Older PDF page counts and sparse Excel rows may be wrong.
                # Keep the original files so users can reparse them in place.
                conn.execute(
                    """
                    UPDATE documents SET indexed=0, status='待重新解析', pages=0, parse_warnings=?
                    WHERE file_path <> '' AND (
                        UPPER(file_type) IN ('PDF', 'XLSX', 'XLSM')
                        OR EXISTS (SELECT 1 FROM chunks WHERE doc_id=documents.id AND section='文件摘要')
                    )
                    """,
                    (json.dumps(["请重新解析原件以更新页码、索引和解析状态。"], ensure_ascii=False),),
                )
                self._bump_revision(conn)

    def get_meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "id": self.project_id,
            "name": "RFQ 演示项目" if self.project_id == "rfq-demo" else self.project_id,
            "description": "工程询价文档智能处理独立项目库",
            "created_at": _now_str(),
        }

    def save_meta(self, meta: dict):
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_settings(self) -> dict:
        cfg = dict(DEFAULT_SETTINGS)
        cfg["data_path"] = str(self.root)
        has_explicit_online = False
        with self._conn() as conn:
            for row in conn.execute("SELECT key, value FROM settings WHERE key <> '_revision'"):
                if row["key"] == "online_answers_enabled":
                    has_explicit_online = True
                try:
                    cfg[row["key"]] = json.loads(row["value"])
                except Exception:
                    cfg[row["key"]] = row["value"]
        if not has_explicit_online:
            local_cfg = load_local_llm_config()
            if (
                local_cfg.get("provider") in {"siliconflow", "company-azure"}
                and local_cfg.get("default_online_enabled", True)
                and llm_config_status().get("configured")
            ):
                cfg["online_answers_enabled"] = True
        return cfg

    def get_revision(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='_revision'").fetchone()
            return int(row["value"]) if row and str(row["value"]).isdigit() else 1

    @staticmethod
    def _bump_revision(conn):
        conn.execute("UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='_revision'")
        conn.execute("DELETE FROM query_cache")

    def update_settings(self, patch: dict) -> dict:
        if "online_answers_enabled" in patch and type(patch["online_answers_enabled"]) is not bool:
            raise ValueError("在线回答开关必须为布尔值。")
        updates = {k: v for k, v in patch.items() if k in DEFAULT_SETTINGS and k != "data_path"}
        with self._conn() as conn:
            for k, v in updates.items():
                conn.execute(
                    "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
                    (k, json.dumps(v, ensure_ascii=False)),
                )
            if updates:
                self._bump_revision(conn)
        return self.get_settings()

    def list_folders(self) -> list[str]:
        with self._conn() as conn:
            return [r["name"] for r in conn.execute("SELECT name FROM folders ORDER BY sort_order, name")]

    def add_folder(self, name: str):
        clean = name.strip()
        if not clean:
            raise ValueError("文件夹名称不能为空")
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO folders(name, sort_order) VALUES (?, 50)", (clean,))

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS raw_chunk_count
                FROM documents d
                ORDER BY d.updated_at DESC, d.name ASC
                """
            ).fetchall()
            result = []
            for r in rows:
                doc = self._document_dict(r)
                raw_cnt = int(doc.pop("raw_chunk_count", 0) or 0)
                doc["chunk_count"] = raw_cnt if doc.get("indexed") else 0
                result.append(doc)
            return result

    @staticmethod
    def _document_dict(row):
        doc = dict(row)
        doc["parse_warnings"] = json.loads(doc.get("parse_warnings") or "[]")
        return doc

    def get_document_preview(self, doc_id: str, page: int | None = None) -> dict:
        with self._conn() as conn:
            doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not doc:
                raise KeyError("未找到指定文档")
            chunks = [
                dict(r) for r in conn.execute(
                    CHUNKS_FOR_DISPLAY,
                    (doc_id,),
                )
            ] if doc["indexed"] else []
        doc_dict = self._document_dict(doc)
        total_chunks = len(chunks)
        doc_dict["chunk_count"] = total_chunks
        chunk_summary = {
            "total": total_chunks,
            "text": sum(1 for c in chunks if (c.get("chunk_type") or "text") == "text"),
            "table": sum(1 for c in chunks if c.get("chunk_type") == "table"),
            "ocr": sum(1 for c in chunks if c.get("chunk_type") == "ocr"),
        }
        fp = doc_dict.get("file_path") or ""
        can_preview_image = bool(
            fp and Path(fp).exists() and Path(fp).suffix.lower() in ({".pdf"} | IMAGE_SUFFIXES)
        )
        selected_chunk = None
        if page is not None:
            selected_chunk = next((c for c in chunks if c["page"] == page), None)
        elif chunks:
            selected_chunk = chunks[0]
        return {
            "document": doc_dict,
            "chunks": chunks,
            "active_chunk": selected_chunk,
            "total_chunks": total_chunks,
            "chunk_summary": chunk_summary,
            "can_preview_image": can_preview_image,
        }

    def _embed_safe(self, text: str) -> bytes | None:
        try:
            return pack_vector(embed_text(text))
        except Exception:
            return None

    def ensure_embeddings(self) -> int:
        """Lazily backfill BGE 512-dim embeddings for any chunks missing vectors."""
        updated = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT c.id, c.content FROM chunks c JOIN documents d ON d.id=c.doc_id "
                "WHERE c.embedding IS NULL AND d.indexed=1"
            ).fetchall()
            for r in rows:
                blob = self._embed_safe(r["content"])
                if blob is not None:
                    conn.execute("UPDATE chunks SET embedding=? WHERE id=?", (blob, r["id"]))
                    updated += 1
            if updated:
                self._bump_revision(conn)
        return updated

    @staticmethod
    def _parse_state(parsed):
        status = "解析失败" if not parsed.chunks else ("部分索引，需核对" if parsed.warnings else "已索引")
        return status, int(bool(parsed.chunks)), json.dumps(parsed.warnings, ensure_ascii=False)

    def _store_parsed_chunks(self, conn, doc_id: str, chunks: list[dict]):
        for idx, item in enumerate(chunks, 1):
            emb = self._embed_safe(item["content"])
            conn.execute(
                "INSERT INTO chunks(id, doc_id, page, section, content, highlight, chunk_type, embedding) VALUES (?, ?, ?, ?, ?, '', ?, ?)",
                (
                    f"{doc_id}-c{idx}",
                    doc_id,
                    item["page"],
                    item["section"],
                    item["content"],
                    item.get("chunk_type", "text"),
                    emb,
                ),
            )

    def ingest_upload(self, filename: str, content: bytes, folder: str, version: str = "Rev.0") -> dict:
        safe_name = Path(filename).name
        if not safe_name:
            raise ValueError("无效文件名")
        self.add_folder(folder)
        cfg = self.get_settings()
        doc_id = uuid.uuid4().hex[:12]
        target = self.files_dir / f"{doc_id}_{safe_name}"
        target.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        parsed = _parse_document(
            target,
            chunk_size=int(cfg.get("chunk_size", 480)),
            chunk_overlap=int(cfg.get("chunk_overlap", 64)),
            ocr_enabled=bool(cfg.get("ocr_enabled", True)),
        )
        ext = target.suffix.lstrip(".").upper() or "FILE"
        now = _now_str()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents(id, name, folder, file_type, version, pages, size_bytes, updated_at,
                                      status, indexed, parse_warnings, file_path, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, safe_name, folder, ext, version or "Rev.0", parsed.pages, len(content), now,
                 *self._parse_state(parsed), str(target), sha),
            )
            self._store_parsed_chunks(conn, doc_id, parsed.chunks)
            self._bump_revision(conn)
        return self.get_document_preview(doc_id)["document"]

    def reparse_document(self, doc_id: str) -> dict:
        cfg = self.get_settings()
        with self._conn() as conn:
            doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not doc:
                raise KeyError("未找到指定文档")
            file_path = Path(doc["file_path"]) if doc["file_path"] else None
            now = _now_str()
            if file_path and file_path.exists():
                parsed = _parse_document(
                    file_path,
                    chunk_size=int(cfg.get("chunk_size", 480)),
                    chunk_overlap=int(cfg.get("chunk_overlap", 64)),
                    ocr_enabled=bool(cfg.get("ocr_enabled", True)),
                )
                conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                self._store_parsed_chunks(conn, doc_id, parsed.chunks)
                conn.execute(
                    "UPDATE documents SET pages=?, updated_at=?, status=?, indexed=?, parse_warnings=? WHERE id=?",
                    (parsed.pages, now, *self._parse_state(parsed), doc_id),
                )
            elif not file_path:
                # Re-embed existing chunks for seeded docs
                rows = conn.execute("SELECT id, content FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
                for r in rows:
                    emb = self._embed_safe(r["content"])
                    conn.execute("UPDATE chunks SET embedding=? WHERE id=?", (emb, r["id"]))
                conn.execute("UPDATE documents SET updated_at=?, status='已索引', indexed=1 WHERE id=?", (now, doc_id))
            else:
                conn.execute(
                    "UPDATE documents SET updated_at=?, status='解析失败', indexed=0, parse_warnings=? WHERE id=?",
                    (now, json.dumps(["原件缺失，无法重新解析。"], ensure_ascii=False), doc_id),
                )
            self._bump_revision(conn)
        return self.get_document_preview(doc_id)

    def reparse_batch(self, only_pending: bool = True) -> dict:
        with self._conn() as conn:
            if only_pending:
                rows = conn.execute(
                    "SELECT id FROM documents WHERE file_path <> '' "
                    "AND (indexed = 0 OR status IN ('待重新解析', '解析失败')) ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM documents WHERE file_path <> '' ORDER BY updated_at DESC"
                ).fetchall()
        doc_ids = [r["id"] for r in rows]
        results = []
        indexed = partial = failed = 0
        for doc_id in doc_ids:
            preview = self.reparse_document(doc_id)
            doc = preview["document"]
            st = doc.get("status", "")
            if st == "已索引":
                indexed += 1
            elif st == "部分索引，需核对":
                partial += 1
            else:
                failed += 1
            results.append({
                "id": doc_id,
                "name": doc.get("name", ""),
                "status": st,
                "chunk_count": preview.get("total_chunks", 0),
                "pages": doc.get("pages", 0),
            })
        return {
            "processed": len(doc_ids),
            "indexed": indexed,
            "partial": partial,
            "failed": failed,
            "results": results,
        }

    @staticmethod
    def _make_session_title(question: str) -> str:
        clean = re.sub(r"\s+", " ", (question or "").strip())
        if not clean:
            return "新对话"
        return clean if len(clean) <= 28 else clean[:28] + "…"

    @classmethod
    def _adopt_orphan_messages(cls, conn):
        orphans = conn.execute(
            "SELECT question, created_at FROM chat_messages "
            "WHERE COALESCE(session_id, '') = '' ORDER BY rowid ASC"
        ).fetchall()
        if not orphans:
            return
        first_q = orphans[0]["question"] or "历史对话"
        first_ts = orphans[0]["created_at"] or _now_str()
        last_ts = orphans[-1]["created_at"] or first_ts
        title = cls._make_session_title(first_q)
        existing = conn.execute("SELECT id FROM chat_sessions WHERE id='sess-legacy'").fetchone()
        if existing:
            conn.execute("UPDATE chat_sessions SET updated_at=? WHERE id='sess-legacy'", (last_ts,))
        else:
            conn.execute(
                "INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("sess-legacy", title, first_ts, last_ts),
            )
        conn.execute("UPDATE chat_messages SET session_id='sess-legacy' WHERE COALESCE(session_id, '') = ''")

    @staticmethod
    def _fetch_sessions(conn, session_id: str | None = None) -> list[dict]:
        where = "WHERE s.id = ?" if session_id else ""
        params = (session_id,) if session_id else ()
        rows = conn.execute(
            f"""
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count,
                   COALESCE((
                       SELECT m2.question FROM chat_messages m2
                       WHERE m2.session_id = s.id ORDER BY m2.rowid DESC LIMIT 1
                   ), '') AS last_question
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            {where}
            GROUP BY s.id
            ORDER BY s.updated_at DESC, COALESCE(MAX(m.rowid), 0) DESC, s.rowid DESC
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def list_chat_sessions(self) -> list[dict]:
        with self._conn() as conn:
            self._adopt_orphan_messages(conn)
            return self._fetch_sessions(conn)

    def create_chat_session(self, title: str = "新对话") -> dict:
        sid = f"sess-{uuid.uuid4().hex[:10]}"
        clean_title = re.sub(r"\s+", " ", (title or "").strip())[:60] or "新对话"
        now = _now_str()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, clean_title, now, now),
            )
            return self._fetch_sessions(conn, sid)[0]

    def rename_chat_session(self, session_id: str, title: str) -> dict:
        clean_title = re.sub(r"\s+", " ", (title or "").strip())[:60]
        if not clean_title:
            raise ValueError("对话标题不能为空")
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE chat_sessions SET title=? WHERE id=?",
                (clean_title, session_id),
            )
            if cur.rowcount == 0:
                raise KeyError("未找到指定对话会话")
            return self._fetch_sessions(conn, session_id)[0]

    def delete_chat_session(self, session_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
            cur = conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
            if cur.rowcount == 0:
                raise KeyError("未找到指定对话会话")
        return True

    def get_chat_history(self, session_id: str | None = None) -> list[dict]:
        with self._conn() as conn:
            self._adopt_orphan_messages(conn)
            target_sid = (session_id or "").strip()
            if not target_sid:
                row = conn.execute(
                    """
                    SELECT s.id FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.id
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC, COALESCE(MAX(m.rowid), 0) DESC, s.rowid DESC
                    LIMIT 1
                    """
                ).fetchone()
                if not row:
                    return []
                target_sid = row["id"]
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id=? ORDER BY rowid ASC",
                (target_sid,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "question": r["question"],
                    "time": r["time_str"],
                    "payload": json.loads(r["payload_json"]),
                }
                for r in rows
            ]

    def clear_chat(self, session_id: str | None = None):
        with self._conn() as conn:
            if session_id:
                conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
                conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
            else:
                conn.execute("DELETE FROM chat_messages")
                conn.execute("DELETE FROM chat_sessions")

    def _record_chat_turn(
        self,
        conn,
        session_id: str | None,
        question: str,
        now_time: str,
        now_full: str,
        answer_payload: dict,
    ) -> tuple[str, str, str]:
        req_sid = (session_id or "").strip()
        if req_sid:
            existing = conn.execute(
                "SELECT id, title FROM chat_sessions WHERE id=?",
                (req_sid,),
            ).fetchone()
            if existing:
                active_sid = req_sid
                active_title = (
                    self._make_session_title(question)
                    if existing["title"] == "新对话"
                    else existing["title"]
                )
                conn.execute(
                    "UPDATE chat_sessions SET title=?, updated_at=? WHERE id=?",
                    (active_title, now_full, active_sid),
                )
            else:
                active_sid = req_sid
                active_title = self._make_session_title(question)
                conn.execute(
                    "INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (active_sid, active_title, now_full, now_full),
                )
        else:
            active_sid = f"sess-{uuid.uuid4().hex[:10]}"
            active_title = self._make_session_title(question)
            conn.execute(
                "INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (active_sid, active_title, now_full, now_full),
            )
        msg_id = uuid.uuid4().hex[:12]
        conn.execute(
            "INSERT INTO chat_messages(id, session_id, role, question, time_str, payload_json, created_at) "
            "VALUES (?, ?, 'turn', ?, ?, ?, ?)",
            (msg_id, active_sid, question, now_time, json.dumps(answer_payload, ensure_ascii=False), now_full),
        )
        return msg_id, active_sid, active_title

    def ask_question(self, question: str, mode: str = "快速问答", session_id: str | None = None) -> dict:
        q = question.strip()
        if not q:
            raise ValueError("请输入要查询的问题")
        mode = LEGACY_QUERY_MODES.get(mode, mode)
        if mode not in QUERY_MODES:
            raise ValueError("未知查询模式。")
        self.ensure_embeddings()
        cfg = self.get_settings()
        rev = self.get_revision()
        llm = llm_config_status() if cfg.get("online_answers_enabled") is True else {"configured": False}
        cache_sig = hashlib.sha256(
            json.dumps({
                "q": q,
                "mode": mode,
                "format": QUERY_CACHE_FORMAT,
                "provider": llm.get("cache_identity", ("local",)),
                "rm": cfg.get("retrieval_mode"),
                "thr": cfg.get("similarity_threshold"),
                "k": cfg.get("top_k"),
                "rev": rev,
            }, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        now_time = datetime.now().strftime("%H:%M")
        now_full = _now_str()

        with self._conn() as conn:
            cached_row = conn.execute(
                "SELECT payload_json, created_at FROM query_cache WHERE cache_key=? AND revision=?",
                (cache_sig, rev),
            ).fetchone()
        if cached_row:
            answer_payload = json.loads(cached_row["payload_json"])
            answer_payload["cached"] = True
            answer_payload["cached_from"] = cached_row["created_at"]
            with self._conn() as conn:
                msg_id, active_sid, active_title = self._record_chat_turn(
                    conn, session_id, q, now_time, now_full, answer_payload
                )
            return {
                "id": msg_id,
                "session_id": active_sid,
                "session_title": active_title,
                "role": "turn",
                "question": q,
                "time": now_time,
                "payload": answer_payload,
            }

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.doc_id, c.page, c.section, c.content, c.highlight, c.embedding AS embedding_blob,
                       d.name AS doc_name, d.version, d.pages AS doc_pages, d.folder, d.file_type
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE d.indexed = 1
                """
            ).fetchall()
        candidates = [dict(r) for r in rows]

        top_k = max(int(cfg.get("top_k", 10)), QUERY_MODES[mode])

        search_res = hybrid_search(
            query=q,
            candidates=candidates,
            retrieval_mode=str(cfg.get("retrieval_mode", "混合检索（语义 + 关键词）")),
            similarity_threshold=float(cfg.get("similarity_threshold", 0.62)),
            top_k=top_k,
        )
        answer_payload = generate_structured_answer(
            question=q,
            hits=search_res["hits"],
            project_name=self.get_meta()["name"],
            retrieval_meta={
                "mode": search_res["mode"],
                "latency_ms": search_res["latency_ms"],
                "bge_used": search_res["bge_used"],
                "total_candidates": search_res["total_candidates"],
                "hit_count": len(search_res["hits"]),
                "refused_missing_attrs": search_res["refused_missing_attrs"],
                "evidence_count": min(len(search_res["hits"]), 5),
                "scope": "候选片段检索，未检查全文",
            },
            mode=mode,
            allow_external=llm["configured"],
        )

        now_time = datetime.now().strftime("%H:%M")
        now_full = _now_str()
        with self._conn() as conn:
            current_revision = conn.execute("SELECT value FROM settings WHERE key='_revision'").fetchone()[0]
            if int(current_revision) == rev and (
                not llm["configured"] or str(answer_payload.get("llm_provider", "")).startswith(("企业模型（", "硅基流动（"))
            ):
                conn.execute(
                    "INSERT OR REPLACE INTO query_cache(cache_key, revision, payload_json, created_at) VALUES (?, ?, ?, ?)",
                    (cache_sig, rev, json.dumps(answer_payload, ensure_ascii=False), now_full),
                )
            msg_id, active_sid, active_title = self._record_chat_turn(
                conn, session_id, q, now_time, now_full, answer_payload
            )
        return {
            "id": msg_id,
            "session_id": active_sid,
            "session_title": active_title,
            "role": "turn",
            "question": q,
            "time": now_time,
            "payload": answer_payload,
        }

    def compare_documents(self, left_id: str, right_id: str, keyword: str = "", sections: list[str] | None = None) -> dict:
        with self._conn() as conn:
            left_doc = conn.execute("SELECT * FROM documents WHERE id=?", (left_id,)).fetchone()
            right_doc = conn.execute("SELECT * FROM documents WHERE id=?", (right_id,)).fetchone()
            if not left_doc or not right_doc:
                raise KeyError("请选择两个有效的项目文档")
            if not left_doc["indexed"] or not right_doc["indexed"]:
                raise ValueError("请先重新解析未索引的文档，再进行原文对照。")
            left_chunks = [dict(r) for r in conn.execute(CHUNKS_FOR_DISPLAY, (left_id,))]
            right_chunks = [dict(r) for r in conn.execute(CHUNKS_FOR_DISPLAY, (right_id,))]

        all_sections = []
        for c in left_chunks + right_chunks:
            if c["section"] and c["section"] not in all_sections:
                all_sections.append(c["section"])

        kw = keyword.strip()
        kw_terms = [t.lower() for t in kw.split() if t.strip()]
        sec_filter = set(sections or [])

        def keep(chunk):
            if sec_filter and chunk["section"] not in sec_filter:
                return False
            if kw_terms and not any(t in (chunk["content"] + " " + chunk["section"]).lower() for t in kw_terms):
                return False
            return True

        l_list = [c for c in left_chunks if keep(c)]
        r_list = [c for c in right_chunks if keep(c)]

        used_r = set()
        groups: dict[str, list[dict]] = {}
        diff_summary = []

        for lc in l_list:
            best_idx = -1
            best_score = 0.0
            for i, rc in enumerate(r_list):
                if i in used_r:
                    continue
                same_sec = 0.35 if lc["section"] == rc["section"] else 0.0
                ratio = SequenceMatcher(None, lc["content"], rc["content"]).ratio() + same_sec
                if ratio > best_score:
                    best_score = ratio
                    best_idx = i
            rc = r_list[best_idx] if best_idx >= 0 and best_score >= 0.42 else None
            if rc:
                used_r.add(best_idx)
            sec_title = lc["section"] or (rc["section"] if rc else "条款对照")
            is_diff = (rc is None) or (lc["content"].strip() != rc["content"].strip())
            groups.setdefault(sec_title, []).append({
                "left": lc,
                "right": rc,
                "is_diff": is_diff,
            })
            if is_diff and len(diff_summary) < 6:
                diff_summary.append({
                    "title": sec_title,
                    "detail": (
                        f"左侧与右侧在“{sec_title}”条款存在表述或参数差异，请核对高亮数值与新增要求。"
                        if rc else f"仅在左侧文档“{sec_title}”检索到该条款，右侧无直接配对片段。"
                    ),
                })

        for i, rc in enumerate(r_list):
            if i not in used_r:
                sec_title = rc["section"] or "新增条款"
                groups.setdefault(sec_title, []).append({
                    "left": None,
                    "right": rc,
                    "is_diff": True,
                })
                if len(diff_summary) < 6:
                    diff_summary.append({
                        "title": sec_title,
                        "detail": f"右侧文档在“{sec_title}”新增或补充了条款说明，请结合原件核实生效状态。",
                    })

        conclusion = (
            f"{right_doc['name']} 与 {left_doc['name']} 共检出 {len(diff_summary)} 项主要差异，"
            "涉及技术参数或条款补充；左右顺序不代表效力高低，最终以审批状态与原件为准。"
            if diff_summary else
            "当前筛选范围内未检出明显条款差异；可更换关键词或勾选更多章节进行比对。"
        )
        return {
            "left_doc": dict(left_doc),
            "right_doc": dict(right_doc),
            "available_sections": all_sections,
            "groups": [{"section": k, "rows": v} for k, v in groups.items()],
            "summary": diff_summary,
            "conclusion": conclusion,
            "keyword": kw,
        }

    def list_saved(self, kind: str = "all", keyword: str = "") -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM saved_items ORDER BY created_at DESC, rowid DESC").fetchall()
        items = []
        kw = keyword.strip().lower()
        for r in rows:
            if kind != "all" and r["kind"] != kind:
                continue
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            d["project_id"] = self.project_id
            d["project_name"] = self.get_meta()["name"]
            if kw and kw not in (d["title"] + " " + d["summary"] + " " + d["sources_text"]).lower():
                continue
            items.append(d)
        return items

    def save_item(self, kind: str, title: str, summary: str, sources_text: str, payload: dict) -> dict:
        item_id = uuid.uuid4().hex[:12]
        now = _now_str()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO saved_items(id, kind, title, summary, sources_text, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (item_id, kind, title, summary, sources_text, now, json.dumps(payload, ensure_ascii=False)),
            )
        return {
            "id": item_id,
            "kind": kind,
            "title": title,
            "summary": summary,
            "sources_text": sources_text,
            "created_at": now,
            "payload": payload,
            "project_id": self.project_id,
            "project_name": self.get_meta()["name"],
        }

    def export_saved_xlsx(self, item_id: str) -> tuple[str, bytes]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM saved_items WHERE id=?", (item_id,)).fetchone()
        if not row:
            raise KeyError("未找到指定已保存记录")
        payload = json.loads(row["payload_json"])
        import openpyxl  # type: ignore

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "工程核查结果"
        ws.append(["项目名称", self.get_meta()["name"]])
        ws.append(["记录标题", row["title"]])
        ws.append(["保存时间", row["created_at"]])
        ws.append(["来源文件", row["sources_text"]])
        ws.append([])
        ws.append(["检索结论", payload.get("conclusion", row["summary"])])
        ws.append([])
        if payload.get("table"):
            ws.append(["项目", "内容 / 要求"])
            for tr in payload["table"]:
                ws.append([tr.get("item", ""), tr.get("value", "")])
            ws.append([])
        if payload.get("citations"):
            ws.append(["引用编号", "文件名", "页码 / 章节", "原文摘录"])
            for c in payload["citations"]:
                ws.append([f"[{c.get('index', '')}]", c.get("file_name", ""),
                           c.get("location") or c.get("section", ""), c.get("content", "")])
        buf = io.BytesIO()
        wb.save(buf)
        safe_title = re.sub(r'[\\/:*?"<>|]+', "_", row["title"])[:32]
        return f"{self.project_id}_{safe_title}.xlsx", buf.getvalue()

    def create_backup(self) -> dict:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.project_id}_backup_{stamp}.zip"
        backup_path = self.backups_dir / backup_name
        shutil.make_archive(str(backup_path.with_suffix("")), "zip", root_dir=str(self.root), base_dir=".")
        return {
            "name": backup_name,
            "path": str(backup_path),
            "created_at": _now_str(),
            "size": _format_size(backup_path.stat().st_size),
        }


def _backfill_demo_chunks(store: ProjectStore):
    """Ensure every seeded synthetic demo document (with file_path == '') has previewable chunks."""
    extra_demo_chunks = [
        ("chk-rfq0-1", "doc-rfq-0", 11, "2.1 供货范围", "2.1  Scope of Supply\n\n2.1.1  Supplier shall design, manufacture and deliver the complete automation control cabinet system.", "Scope of Supply", "text"),
        ("chk-tech-1", "doc-rfq-tech", 8, "3.2 环境条件", "3.2  Environmental Conditions\n\n3.2.1  Ambient temperature: 0 °C to +45 °C; Relative humidity <= 85% non-condensing.", "0 °C to +45 °C", "text"),
        ("chk-draw-1", "doc-rfq-draw", 4, "第 4 页 · 图纸说明", "[图纸说明] 主电气控制柜外形尺寸：800 mm (W) × 600 mm (D) × 2200 mm (H)，底部进线。", "2200 mm (H)", "ocr"),
        ("chk-eval-1", "doc-rfq-eval", 3, "第 3 页 · 表格 1", "[表格 1] 评审项 | 权重 | 评分标准\n评审项: 技术响应偏离 | 权重: 45% | 评分标准: 满足所有强制电气与防护条款", "45%", "table"),
        ("chk-b12p-1", "doc-b12-param", 5, "第 5 页 · 表格 1", "[表格 1] 参数名称 | 设计要求 | 备注\n参数名称: 供电电源 | 设计要求: 380V AC / 24V DC | 备注: 含双路冗余", "380V AC / 24V DC", "table"),
        ("chk-clar01-1", "doc-clar-01", 2, "澄清条目 Q&A-04", "Q-04: 控制柜备用端子数量要求？\nA-04: 每一排端子排应至少预留 20% 备用端子。", "20% 备用端子", "text"),
        ("chk-mom-te-1", "doc-mom-te", 2, "第 2 页 · 技术会议纪要", "会议决议 TE-02：现场接线盒与主柜之间采用屏蔽控制电缆，接地电阻 <= 4 Ω。", "<= 4 Ω", "text"),
        ("chk-mom-df-1", "doc-mom-df", 3, "第 3 页 · 设计冻结纪要", "设计冻结项 DF-05：所有铭牌采用中英文双语激光蚀刻不锈钢材质。", "中英文双语", "text"),
        ("chk-xparam-1", "doc-xlsx-param", 1, "工作表 电气参数", "[表头] 编号 | 参数项 | 设定值 | 单位\n编号: P-01 | 参数项: 额定绝缘电压 | 设定值: 690 | 单位: V", "690", "table"),
        ("chk-xequip-1", "doc-xlsx-equip", 1, "工作表 设备清单", "[表头] 位号 | 设备名称 | 数量 | 防护等级\n位号: CAB-01 | 设备名称: 主控PLC柜 | 数量: 2 | 防护等级: IP54", "IP54", "table"),
    ]
    seeded_ids = tuple({row[1] for row in extra_demo_chunks})
    placeholders = ",".join("?" for _ in seeded_ids)
    with store._conn() as conn:
        missing_targets = {
            r[0]
            for r in conn.execute(
                f"SELECT d.id FROM documents d WHERE d.file_path = '' AND d.id IN ({placeholders}) "
                "AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.id)",
                seeded_ids,
            )
        }
        if missing_targets:
            for c in extra_demo_chunks:
                if c[1] in missing_targets:
                    conn.execute(
                        "INSERT OR IGNORE INTO chunks(id, doc_id, page, section, content, highlight, chunk_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        c,
                    )
            store._bump_revision(conn)


def seed_demo_project(runtime_root: Path):
    """Seed a clean synthetic demo project matching the 5 prototype screens (no real company data)."""
    store = ProjectStore(runtime_root, "rfq-demo")
    if store.list_documents():
        _backfill_demo_chunks(store)
        return
    store.save_meta({
        "id": "rfq-demo",
        "name": "RFQ 演示项目",
        "description": "虚构工程询价示例项目库（不含任何企业真实资料）",
        "created_at": "2024-11-20 14:26",
    })
    demo_docs = [
        ("doc-rfq-a", "RFQ Rev.A.pdf", "RFQ 询价文件", "PDF", "Rev.A", 126, 9751756, "2024-11-20 14:26", "已索引"),
        ("doc-rfq-0", "RFQ Rev.0.pdf", "RFQ 询价文件", "PDF", "Rev.0", 118, 8920100, "2024-11-15 09:32", "已索引"),
        ("doc-rfq-tech", "RFQ 附件 - 技术要求.pdf", "RFQ 询价文件", "PDF", "Rev.A", 42, 3420000, "2024-11-14 16:08", "已索引"),
        ("doc-rfq-draw", "RFQ 附件 - 图纸及布置.pdf", "RFQ 询价文件", "PDF", "Rev.A", 18, 6120000, "2024-11-14 16:07", "已索引"),
        ("doc-rfq-eval", "RFQ 评分标准.pdf", "RFQ 询价文件", "PDF", "Rev.0", 12, 1280000, "2024-11-12 11:20", "已索引"),
        ("doc-b12-spec", "B12 技术规范.docx", "技术协议 / B12", "DOCX", "Rev.B", 64, 2450000, "2024-11-18 16:37", "已索引"),
        ("doc-b12-param", "B12 附件 1 - 参数要求.pdf", "技术协议 / B12", "PDF", "Rev.B", 28, 1890000, "2024-11-18 15:10", "已索引"),
        ("doc-clar-01", "Clarification 01.pdf", "澄清文件", "PDF", "Rev.0", 8, 920000, "2024-11-16 10:15", "已索引"),
        ("doc-clar-02", "Clarification 02.pdf", "澄清文件", "PDF", "Rev.0", 6, 840000, "2024-11-19 10:21", "已索引"),
        ("doc-mom-te", "TE MOM.pdf", "会议纪要", "PDF", "Rev.0", 9, 760000, "2024-11-18 11:05", "已索引"),
        ("doc-mom-df", "DF MOM.pdf", "会议纪要", "PDF", "Rev.0", 7, 680000, "2024-11-18 10:50", "已索引"),
        ("doc-xlsx-param", "参数表.xlsx", "其他资料", "XLSX", "Rev.A", 4, 420000, "2024-11-17 14:12", "已索引"),
        ("doc-xlsx-equip", "设备清单.xlsx", "其他资料", "XLSX", "Rev.A", 5, 510000, "2024-11-17 14:00", "已索引"),
    ]
    with store._conn() as conn:
        for row in demo_docs:
            conn.execute(
                """
                INSERT OR IGNORE INTO documents(id, name, folder, file_type, version, pages, size_bytes, updated_at, status, indexed, file_path, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, '', ?)
                """,
                (*row, hashlib.sha256(row[0].encode()).hexdigest()),
            )
        demo_chunks = [
            (
                "chk-rfq-93", "doc-rfq-a", 58, "9.3 电气要求",
                "9.3  Electrical Requirements\n\n"
                "9.3.1  The electrical equipment shall comply with the requirements of relevant international standards and the specifications stated in this RFQ.\n\n"
                "9.3.2  Electrical control cabinets shall be suitable for the specified environmental conditions. The enclosure protection rating shall be IP54.\n\n"
                "9.3.3  The control cabinets shall be constructed of corrosion-resistant materials and shall provide adequate space for cable termination, maintenance and future expansion.\n\n"
                "9.3.4  All electrical components shall be clearly labeled, and the wiring shall be in accordance with the project wiring diagrams.",
                "The enclosure protection rating shall be IP54.",
            ),
            (
                "chk-rfq-94", "doc-rfq-a", 61, "9.4 接口与通信",
                "9.4.1  控制系统应支持 Modbus TCP 协议，便于与上位系统的数据交互。\n\n"
                "9.4.2  通信接口应至少提供 1 个以太网接口 (RJ45)，速率不低于 100 Mbps。",
                "Modbus TCP",
            ),
            (
                "chk-rfq-95", "doc-rfq-a", 64, "9.5 测试要求",
                "9.5.1  设备出厂前应进行工厂验收测试 (FAT)，测试内容包括功能测试和基本性能验证。\n\n"
                "9.5.2  测试应形成报告，并随设备交付。",
                "工厂验收测试 (FAT)",
            ),
            (
                "chk-clar-93", "doc-clar-02", 3, "9.3 电气要求",
                "9.3.2  电气控制柜应适用于本项目的环境条件，其外壳防护等级应不低于 IP55（适用于室外布置区域；室内控制柜按原 IP54 执行）。\n\n"
                "9.3.3  控制柜应采用耐腐蚀材料制造，并应在电缆接线、维护和未来扩展方面提供足够的空间。",
                "IP55",
            ),
            (
                "chk-clar-94", "doc-clar-02", 4, "9.4 接口与通信",
                "9.4.1  控制系统应支持 Modbus TCP 和 OPC UA 协议，便于与上位系统的数据交互。\n\n"
                "9.4.2  通信接口应至少提供 1 个以太网接口 (RJ45)，速率不低于 100 Mbps。",
                "Modbus TCP 和 OPC UA",
            ),
            (
                "chk-clar-95", "doc-clar-02", 5, "9.5 测试要求",
                "9.5.1  设备出厂前应进行工厂验收测试 (FAT)，测试内容包括功能测试、性能验证和通信联调。\n\n"
                "9.5.2  测试应形成报告，并随设备交付。报告应包含测试环境、测试用例及结果记录。",
                "通信联调",
            ),
            (
                "chk-b12-1", "doc-b12-spec", 12, "4.2 钢构荷载与防腐",
                "4.2.1  钢结构平台设计均布荷载为 3.0 kN/m²，防腐采用环氧底漆 + 聚氨酯面漆，干膜总厚度不小于 160 μm。",
                "3.0 kN/m²",
            ),
        ]
        for c in demo_chunks:
            conn.execute(
                "INSERT OR IGNORE INTO chunks(id, doc_id, page, section, content, highlight) VALUES (?, ?, ?, ?, ?, ?)",
                c,
            )

        demo_answer = {
            "mode": "快速问答",
            "conclusion": (
                "根据已检索的 RFQ 和澄清文件，示例资料中电控柜的防护等级为 **IP54** [1]。"
                "但具体适用位置和审批状态仍应结合原文核实 [2]。"
            ),
            "basis": [
                {"index": 1, "source": "RFQ Rev.A（第 58 页，9.3 电气要求）", "detail": "明确规定电控柜的外壳防护等级为 IP54 [1]。"},
                {"index": 2, "source": "Clarification 02（第 3 页）", "detail": "对防护等级的适用范围作出补充说明，确认室外布置区域及补充要求 [2]。"},
            ],
            "table": [
                {"item": "设备", "value": "电控柜 (Electrical Control Cabinet)"},
                {"item": "防护等级", "value": "IP54"},
                {"item": "主要来源", "value": "RFQ Rev.A 第 58 页 [1]；Clarification 02 第 3 页 [2]"},
            ],
            "citations": [
                {
                    "index": 1, "chunk_id": "chk-rfq-93", "doc_id": "doc-rfq-a",
                    "file_name": "RFQ Rev.A.pdf", "version": "Rev.A", "page": 58, "total_pages": 126,
                    "section": "9.3 电气要求", "content": demo_chunks[0][4], "highlight": demo_chunks[0][5],
                },
                {
                    "index": 2, "chunk_id": "chk-clar-93", "doc_id": "doc-clar-02",
                    "file_name": "Clarification 02.pdf", "version": "Rev.0", "page": 3, "total_pages": 6,
                    "section": "防护等级说明", "content": demo_chunks[3][4], "highlight": demo_chunks[3][5],
                },
            ],
        }
        conn.execute(
            "INSERT OR IGNORE INTO chat_messages(id, role, question, time_str, payload_json, created_at) VALUES (?, 'turn', ?, '14:26', ?, '2024-11-20 14:26')",
            ("msg-demo-1", "电控柜的防护等级要求是什么？", json.dumps(demo_answer, ensure_ascii=False)),
        )
        store._adopt_orphan_messages(conn)

        saved_seed = [
            (
                "sav-1", "qa", "电控柜的防护等级要求是什么？",
                "明确电控柜的防护等级为 IP54，并说明适用范围及依据文件。",
                "RFQ Rev.A.pdf; Clarification 02.pdf", "2024-11-20 14:26", demo_answer,
            ),
            (
                "sav-2", "qa", "控制柜的电气要求有哪些？",
                "汇总控制柜的电气设备要求、环境适应性及标识要求。",
                "RFQ Rev.A.pdf; B12 技术规范.docx", "2024-11-19 17:03", demo_answer,
            ),
            (
                "sav-3", "compare", "室外布置区域的适用防护等级说明",
                "对室外布置区域应采用的防护等级作出补充说明，并给出文件依据。",
                "Clarification 02.pdf", "2024-11-19 10:21", demo_answer,
            ),
            (
                "sav-4", "qa", "B12 项目的主要技术参数汇总",
                "汇总 B12 项目中关键设备的主要技术参数及边界条件。",
                "B12 附件 1 - 参数要求.pdf; B12 技术规范.docx", "2024-11-18 16:37", demo_answer,
            ),
            (
                "sav-5", "export", "项目交付资料清单",
                "整理项目需提交的技术文件和交付资料清单。",
                "DF MOM.pdf; TE MOM.pdf", "2024-11-18 11:05", demo_answer,
            ),
        ]
        for s in saved_seed:
            conn.execute(
                "INSERT OR IGNORE INTO saved_items(id, kind, title, summary, sources_text, created_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (s[0], s[1], s[2], s[3], s[4], s[5], json.dumps(s[6], ensure_ascii=False)),
            )
    _backfill_demo_chunks(store)


def create_web_app(runtime_root: Path | None = None) -> FastAPI:
    root = runtime_root or prepare_runtime()
    seed_demo_project(root)

    web_dir = Path(__file__).resolve().parent / "web_ui"
    api = FastAPI(title="工程项目知识助手")
    api.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    def get_store(project_id: str) -> ProjectStore:
        clean = _safe_project_id(project_id)
        return ProjectStore(root, clean)

    @api.get("/project-health")
    def health():
        return {"app": "rfq-workbench-web", "status": "ok", "runtime_root": str(root)}

    @api.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((web_dir / "index.html").read_text(encoding="utf-8"))

    @api.get("/api/projects")
    def list_projects():
        projects_root = root / "projects"
        projects_root.mkdir(parents=True, exist_ok=True)
        items = []
        for entry in sorted(projects_root.iterdir()):
            if not entry.is_dir():
                continue
            store = ProjectStore(root, entry.name)
            meta = store.get_meta()
            docs = store.list_documents()
            items.append({
                "id": entry.name,
                "name": meta.get("name", entry.name),
                "description": meta.get("description", ""),
                "doc_count": len(docs),
                "db_path": str(store.db_path),
            })
        items.sort(key=lambda x: (0 if x["id"] == "rfq-demo" else 1, x["name"]))
        return {"projects": items}

    @api.post("/api/projects")
    def create_project(payload: dict):
        name = str(payload.get("name", "")).strip()
        raw_id = str(payload.get("id", "")).strip() or name
        if not name:
            raise HTTPException(400, "项目名称不能为空")
        pid = _safe_project_id(raw_id)
        store = ProjectStore(root, pid)
        meta = {
            "id": pid,
            "name": name,
            "description": str(payload.get("description", "")).strip() or "独立项目数据库",
            "created_at": _now_str(),
        }
        store.save_meta(meta)
        return {"project": {**meta, "doc_count": len(store.list_documents()), "db_path": str(store.db_path)}}

    @api.get("/api/projects/{project_id}/documents")
    def get_documents(project_id: str):
        store = get_store(project_id)
        folders = store.list_folders()
        docs = store.list_documents()
        tree = []
        for folder in folders:
            f_docs = [d for d in docs if d["folder"] == folder]
            tree.append({"name": folder, "count": len(f_docs), "documents": f_docs})
        return {
            "project": store.get_meta(),
            "folders": folders,
            "tree": tree,
            "documents": docs,
            "total": len(docs),
            "last_updated": _today_cn(),
        }

    @api.post("/api/projects/{project_id}/folders")
    def create_folder(project_id: str, payload: dict):
        store = get_store(project_id)
        try:
            store.add_folder(str(payload.get("name", "")))
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        return {"folders": store.list_folders()}

    @api.post("/api/projects/{project_id}/documents/upload")
    async def upload_documents(
        project_id: str,
        folder: str = Form("RFQ 询价文件"),
        version: str = Form("Rev.0"),
        files: list[UploadFile] = File(...),
    ):
        store = get_store(project_id)
        uploaded = []
        for f in files:
            content = await f.read()
            uploaded.append(store.ingest_upload(f.filename or "untitled.txt", content, folder, version))
        return {"uploaded": uploaded, "count": len(uploaded)}

    @api.get("/api/projects/{project_id}/documents/{doc_id}/preview")
    def document_preview(project_id: str, doc_id: str, page: int | None = None):
        store = get_store(project_id)
        try:
            return store.get_document_preview(doc_id, page=page)
        except KeyError as err:
            raise HTTPException(404, str(err)) from None

    @api.post("/api/projects/{project_id}/documents/reparse-all")
    def reparse_all_documents(project_id: str, payload: dict | None = None):
        store = get_store(project_id)
        only_pending = bool((payload or {}).get("only_pending", True))
        return store.reparse_batch(only_pending=only_pending)

    @api.post("/api/projects/{project_id}/documents/{doc_id}/reparse")
    def reparse_document(project_id: str, doc_id: str):
        store = get_store(project_id)
        try:
            return store.reparse_document(doc_id)
        except KeyError as err:
            raise HTTPException(404, str(err)) from None

    @api.get("/api/projects/{project_id}/documents/{doc_id}/page-image/{page}")
    def document_page_image(project_id: str, doc_id: str, page: int):
        store = get_store(project_id)
        try:
            info = store.get_document_preview(doc_id)
        except KeyError as err:
            raise HTTPException(404, str(err)) from None
        fp = info["document"].get("file_path")
        if not fp or not Path(fp).exists():
            raise HTTPException(404, "仅本地已上传的 PDF 或图片文件支持图像预览")
        src = Path(fp)
        suffix = src.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz  # type: ignore

                with fitz.open(fp) as pdf:
                    if not 1 <= page <= len(pdf):
                        raise HTTPException(404, "页码超出原始 PDF 范围")
                    png_bytes = pdf[page - 1].get_pixmap(dpi=130).tobytes("png")
                return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
            except HTTPException:
                raise
            except Exception as err:
                raise HTTPException(500, f"渲染 PDF 原页失败: {err}") from None
        if suffix in IMAGE_SUFFIXES:
            try:
                from PIL import Image, ImageOps  # type: ignore

                with Image.open(src) as img:
                    n_frames = int(getattr(img, "n_frames", 1) or 1)
                    if not 1 <= page <= n_frames:
                        raise HTTPException(404, "帧序号超出图片范围")
                    if suffix == ".png" and n_frames == 1:
                        return StreamingResponse(io.BytesIO(src.read_bytes()), media_type="image/png")
                    img.seek(page - 1)
                    frame = ImageOps.exif_transpose(img).convert("RGB")
                    buf = io.BytesIO()
                    frame.save(buf, format="PNG")
                    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/png")
            except HTTPException:
                raise
            except Exception as err:
                raise HTTPException(500, f"渲染图片失败: {err}") from None
        raise HTTPException(404, "仅本地已上传的 PDF 或图片文件支持图像预览")

    @api.get("/api/projects/{project_id}/source/{doc_id}")
    def download_source(project_id: str, doc_id: str):
        store = get_store(project_id)
        try:
            info = store.get_document_preview(doc_id)
        except KeyError as err:
            raise HTTPException(404, str(err)) from None
        fp = info["document"].get("file_path")
        if fp and Path(fp).exists():
            return FileResponse(fp, filename=info["document"]["name"])
        text = "\n\n".join(f"--- {c['section']} (第 {c['page']} 页) ---\n{c['content']}" for c in info["chunks"])
        return HTMLResponse(
            f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>{escape(info['document']['name'])}</title>"
            f"<body style='font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:860px;margin:32px auto;padding:0 20px;line-height:1.8'>"
            f"<h2>{escape(info['document']['name'])} ({escape(info['document']['version'])})</h2>"
            f"<pre style='white-space:pre-wrap;background:#f8fafc;padding:20px;border-radius:10px;border:1px solid #e5e7eb'>{escape(text)}</pre></body></html>"
        )

    @api.get("/api/projects/{project_id}/chat/sessions")
    def list_chat_sessions(project_id: str):
        store = get_store(project_id)
        return {"sessions": store.list_chat_sessions()}

    @api.post("/api/projects/{project_id}/chat/sessions")
    def create_chat_session(project_id: str, payload: dict | None = None):
        store = get_store(project_id)
        title = str((payload or {}).get("title", "新对话"))
        return {"session": store.create_chat_session(title)}

    @api.patch("/api/projects/{project_id}/chat/sessions/{session_id}")
    def rename_chat_session(project_id: str, session_id: str, payload: dict):
        store = get_store(project_id)
        try:
            return {"session": store.rename_chat_session(session_id, str(payload.get("title", "")))}
        except KeyError as err:
            raise HTTPException(404, str(err)) from None
        except ValueError as err:
            raise HTTPException(400, str(err)) from None

    @api.delete("/api/projects/{project_id}/chat/sessions/{session_id}")
    def delete_chat_session(project_id: str, session_id: str):
        store = get_store(project_id)
        try:
            store.delete_chat_session(session_id)
            return {"ok": True}
        except KeyError as err:
            raise HTTPException(404, str(err)) from None

    @api.get("/api/projects/{project_id}/chat/history")
    def chat_history(project_id: str, session_id: str | None = None):
        store = get_store(project_id)
        sessions = store.list_chat_sessions()
        active_sid = (
            session_id
            if (session_id and any(s["id"] == session_id for s in sessions))
            else (sessions[0]["id"] if sessions else None)
        )
        messages = store.get_chat_history(active_sid) if active_sid else []
        return {"session_id": active_sid, "sessions": sessions, "messages": messages}

    @api.delete("/api/projects/{project_id}/chat/history")
    def clear_chat_history(project_id: str, session_id: str | None = None):
        store = get_store(project_id)
        store.clear_chat(session_id=session_id)
        return {"ok": True}

    @api.post("/api/projects/{project_id}/ask")
    def ask(project_id: str, payload: dict):
        store = get_store(project_id)
        try:
            turn = store.ask_question(
                str(payload.get("question", "")),
                mode=str(payload.get("mode", "快速问答")),
                session_id=payload.get("session_id"),
            )
            return {"turn": turn, "session_id": turn["session_id"]}
        except ValueError as err:
            raise HTTPException(400, str(err)) from None

    @api.post("/api/projects/{project_id}/compare")
    def compare(project_id: str, payload: dict):
        store = get_store(project_id)
        try:
            return store.compare_documents(
                str(payload.get("left_id", "")),
                str(payload.get("right_id", "")),
                keyword=str(payload.get("keyword", "")),
                sections=payload.get("sections"),
            )
        except (KeyError, ValueError) as err:
            raise HTTPException(400, str(err)) from None

    @api.get("/api/projects/{project_id}/saved")
    def list_saved(project_id: str, kind: str = "all", keyword: str = ""):
        store = get_store(project_id)
        all_items = store.list_saved("all", "")
        filtered = store.list_saved(kind, keyword)
        counts = {
            "all": len(all_items),
            "qa": sum(1 for x in all_items if x["kind"] == "qa"),
            "compare": sum(1 for x in all_items if x["kind"] == "compare"),
            "export": sum(1 for x in all_items if x["kind"] == "export"),
        }
        return {"items": filtered, "counts": counts}

    @api.post("/api/projects/{project_id}/saved")
    def save_result(project_id: str, payload: dict):
        store = get_store(project_id)
        item = store.save_item(
            kind=str(payload.get("kind", "qa")),
            title=str(payload.get("title", "未命名结果")),
            summary=str(payload.get("summary", "")),
            sources_text=str(payload.get("sources_text", "")),
            payload=payload.get("payload") or {},
        )
        return {"item": item}

    @api.get("/api/projects/{project_id}/saved/{item_id}/export")
    def export_saved(project_id: str, item_id: str):
        store = get_store(project_id)
        try:
            filename, data = store.export_saved_xlsx(item_id)
        except KeyError as err:
            raise HTTPException(404, str(err)) from None
        from urllib.parse import quote

        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
        )

    def _build_runtime_status(store: ProjectStore, cfg: dict, is_admin: bool = False) -> dict:
        docs = store.list_documents()
        with store._conn() as conn:
            chunk_cnt, vec_cnt = conn.execute(
                "SELECT COUNT(*), COUNT(c.embedding) FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.indexed=1"
            ).fetchone()
        from local_ocr import is_ocr_available

        emb = embedding_health()
        llm = llm_config_status()
        ocr_status = "已关闭"
        if cfg.get("ocr_enabled"):
            ocr_status = "已启用（本地离线 RapidOCR 就绪）" if is_ocr_available() else "本地 OCR 未安装"
        last_call_note = ""
        if cfg.get("online_answers_enabled") is True and llm.get("configured"):
            with store._conn() as conn:
                last_row = conn.execute(
                    "SELECT payload_json FROM chat_messages ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
            if last_row:
                try:
                    lp = json.loads(last_row["payload_json"])
                    ast = lp.get("answer_status")
                    if ast == "llm_success":
                        last_call_note = " · 最近调用成功"
                    elif ast == "llm_call_failed":
                        code = (lp.get("diagnostic") or {}).get("code", "error")
                        last_call_note = f" · 最近调用失败({code})"
                    elif ast == "verification_failed":
                        last_call_note = " · 最近回答未通过校验"
                except Exception:
                    pass
            if not last_call_note:
                last_call_note = "（尚未在当前项目实测调用）"
        if cfg.get("online_answers_enabled") is True and llm.get("configured"):
            model_status = (llm["status_text"] if is_admin else "在线智能问答已就绪") + last_call_note
        elif cfg.get("online_answers_enabled") is True:
            model_status = llm["status_text"] if is_admin else "在线模型未配置（本地检索模式）"
        else:
            model_status = "本项目使用本地检索"
        return {
            "model_api": model_status,
            "vector_service": "本地 BGE 512维就绪" if emb.get("ok") else "向量模型未就绪",
            "index_service": f"运行中（文档 {len(docs)} 篇 · 向量块 {vec_cnt}/{chunk_cnt}）",
            "ocr_service": ocr_status,
            "updated_at": _now_str(),
        }

    admin_tokens: set[str] = set()

    def _is_admin(token: str | None) -> bool:
        return bool(token and token in admin_tokens)

    @api.get("/api/admin/status")
    def get_admin_status(x_admin_token: str | None = Header(default=None)):
        local_cfg = load_local_llm_config()
        return {
            "is_admin": _is_admin(x_admin_token),
            "has_custom_pin": bool(local_cfg.get("admin_pin_hash")),
        }

    @api.post("/api/admin/login")
    def admin_login(payload: dict):
        pin = str(payload.get("pin", ""))
        if not verify_admin_pin(pin):
            raise HTTPException(403, "管理员口令错误。")
        token = secrets.token_urlsafe(32)
        admin_tokens.add(token)
        return {"token": token, "is_admin": True}

    @api.post("/api/admin/logout")
    def admin_logout(x_admin_token: str | None = Header(default=None)):
        if x_admin_token:
            admin_tokens.discard(x_admin_token)
        return {"ok": True, "is_admin": False}

    @api.put("/api/admin/pin")
    def change_admin_pin(payload: dict, x_admin_token: str | None = Header(default=None)):
        if not _is_admin(x_admin_token):
            raise HTTPException(403, "需要管理员权限。")
        try:
            update_admin_pin(
                str(payload.get("current_pin", "")),
                str(payload.get("new_pin", "")),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @api.get("/api/llm-config")
    def get_llm_config(x_admin_token: str | None = Header(default=None)):
        st = llm_config_status()
        if not _is_admin(x_admin_token):
            return {
                "is_admin": False,
                "llm_status": {
                    "configured": bool(st.get("configured")),
                    "status_text": "在线模型已就绪" if st.get("configured") else "本地检索模式",
                },
            }
        return {"is_admin": True, "llm_config": masked_local_llm_config(), "llm_status": st}

    @api.put("/api/llm-config")
    def put_llm_config(payload: dict, x_admin_token: str | None = Header(default=None)):
        if not _is_admin(x_admin_token):
            raise HTTPException(403, "需要管理员权限。")
        try:
            default_online = payload.get("default_online_enabled")
            saved = save_local_llm_config(
                provider=str(payload.get("provider", "")),
                model=str(payload.get("model", "")),
                api_key=str(payload.get("api_key", "")),
                clear_key=bool(payload.get("clear_key", False)),
                default_online_enabled=bool(default_online) if default_online is not None else None,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"is_admin": True, "llm_config": saved, "llm_status": llm_config_status()}

    @api.get("/api/projects/{project_id}/settings")
    def get_settings(project_id: str, x_admin_token: str | None = Header(default=None)):
        store = get_store(project_id)
        cfg = store.get_settings()
        is_admin = _is_admin(x_admin_token)
        return {"settings": cfg, "status": _build_runtime_status(store, cfg, is_admin=is_admin), "is_admin": is_admin}

    @api.put("/api/projects/{project_id}/settings")
    def put_settings(project_id: str, payload: dict, x_admin_token: str | None = Header(default=None)):
        store = get_store(project_id)
        is_admin = _is_admin(x_admin_token)
        try:
            updated = store.update_settings(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"settings": updated, "status": _build_runtime_status(store, updated, is_admin=is_admin), "is_admin": is_admin}

    @api.post("/api/projects/{project_id}/backup")
    def backup_project(project_id: str):
        store = get_store(project_id)
        return {"backup": store.create_backup()}

    return api


def main():
    parser = argparse.ArgumentParser(description="RFQ Engineering Knowledge Workbench Web Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7866)
    args = parser.parse_args()
    app = create_web_app()
    print(f"RFQ Workbench Web running at http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
