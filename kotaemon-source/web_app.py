"""Modern multi-project FastAPI web server for the RFQ Engineering Knowledge Workbench.

Each project uses an isolated directory and independent SQLite database under
%LOCALAPPDATA%\\RFQKotaemon\\projects\\<project_id>\\ so switching projects in the
header immediately switches the active database, documents, chats, comparisons,
saved results, and settings.
"""

import argparse
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pilot_runtime import prepare_runtime
from rag_engine import (
    embed_text,
    embedding_health,
    generate_structured_answer,
    hybrid_search,
    llm_config_status,
    pack_vector,
    split_text_with_overlap,
)

DEFAULT_SETTINGS = {
    "chat_model": "GPT-4o（企业内网）",
    "embedding_model": "BAAI/bge-small-zh-v1.5（本地离线）",
    "retrieval_mode": "混合检索（语义 + 关键词）",
    "similarity_threshold": 0.62,
    "top_k": 10,
    "ocr_enabled": True,
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


def _extract_text_from_file(
    path: Path,
    chunk_size: int = 480,
    chunk_overlap: int = 64,
    ocr_enabled: bool = True,
) -> list[dict]:
    """Extract readable text pages/sections locally with overlap-aware chunking."""
    suffix = path.suffix.lower()
    raw_sections: list[tuple[int, str, str]] = []

    if suffix in {".txt", ".md", ".csv", ".json", ".log"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        raw_sections.append((1, "正文", text))

    elif suffix == ".pdf":
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(str(path))
            for idx, page in enumerate(reader.pages, 1):
                txt = (page.extract_text() or "").strip()
                if not txt and ocr_enabled:
                    try:
                        from local_ocr import ocr_pdf_page  # type: ignore

                        txt = (ocr_pdf_page(path, idx - 1) or "").strip()
                    except Exception:
                        txt = ""
                if txt:
                    raw_sections.append((idx, f"第 {idx} 页", txt))
        except Exception:
            pass

    elif suffix == ".docx":
        try:
            import docx  # type: ignore

            document = docx.Document(str(path))
            full_text = "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
            if full_text:
                raw_sections.append((1, "Word 正文", full_text))
            for t_idx, table in enumerate(document.tables, 1):
                t_rows = []
                for row in table.rows:
                    vals = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    if any(vals):
                        t_rows.append(" | ".join(vals))
                if t_rows:
                    raw_sections.append((t_idx + 1, f"表格 {t_idx}", "\n".join(t_rows)))
        except Exception:
            pass

    elif suffix in {".xlsx", ".xlsm"}:
        try:
            import openpyxl  # type: ignore

            wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
            for s_idx, sheet_name in enumerate(wb.sheetnames, 1):
                ws = wb[sheet_name]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    vals = [str(c).strip() for c in row if c is not None and str(c).strip()]
                    if vals:
                        rows.append(" | ".join(vals))
                if rows:
                    raw_sections.append((s_idx, f"工作表 {sheet_name}", "\n".join(rows)))
        except Exception:
            pass

    chunks: list[dict] = []
    for page_no, sec_name, body in raw_sections:
        parts = split_text_with_overlap(body, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for p_idx, part in enumerate(parts, 1):
            label = sec_name if len(parts) == 1 else f"{sec_name} ({p_idx}/{len(parts)})"
            chunks.append({"page": page_no, "section": label, "content": part})

    if chunks:
        return chunks
    return [{
        "page": 1,
        "section": "文件摘要",
        "content": f"已上传文件 {path.name}（{_format_size(path.stat().st_size)}）。当前格式需要人工核对原件或安装对应解析器。",
    }]


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

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

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
                    sha256 TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page INTEGER DEFAULT 1,
                    section TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    highlight TEXT DEFAULT '',
                    embedding BLOB DEFAULT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
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
                """
            )
            for idx, folder in enumerate(DEFAULT_FOLDERS):
                conn.execute(
                    "INSERT OR IGNORE INTO folders(name, sort_order) VALUES (?, ?)",
                    (folder, idx),
                )
            # Migrate existing project databases cleanly if embedding column is missing
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
            if "embedding" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB DEFAULT NULL")

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
        with self._conn() as conn:
            for row in conn.execute("SELECT key, value FROM settings"):
                try:
                    cfg[row["key"]] = json.loads(row["value"])
                except Exception:
                    cfg[row["key"]] = row["value"]
        return cfg

    def update_settings(self, patch: dict) -> dict:
        current = self.get_settings()
        for k, v in patch.items():
            if k in DEFAULT_SETTINGS:
                current[k] = v
        with self._conn() as conn:
            for k, v in current.items():
                conn.execute(
                    "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
                    (k, json.dumps(v, ensure_ascii=False)),
                )
        return current

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
                "SELECT * FROM documents ORDER BY updated_at DESC, name ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document_preview(self, doc_id: str, page: int | None = None) -> dict:
        with self._conn() as conn:
            doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not doc:
                raise KeyError("未找到指定文档")
            chunks = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM chunks WHERE doc_id=? ORDER BY page ASC, rowid ASC",
                    (doc_id,),
                )
            ]
        doc_dict = dict(doc)
        total_chunks = len(chunks)
        selected_chunk = None
        if page is not None:
            selected_chunk = next((c for c in chunks if c["page"] == page), None)
        if selected_chunk is None and chunks:
            selected_chunk = chunks[0]
        return {
            "document": doc_dict,
            "chunks": chunks,
            "active_chunk": selected_chunk,
            "total_chunks": total_chunks,
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
            rows = conn.execute("SELECT id, content FROM chunks WHERE embedding IS NULL").fetchall()
            for r in rows:
                blob = self._embed_safe(r["content"])
                if blob is not None:
                    conn.execute("UPDATE chunks SET embedding=? WHERE id=?", (blob, r["id"]))
                    updated += 1
        return updated

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
        extracted = _extract_text_from_file(
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
                INSERT INTO documents(id, name, folder, file_type, version, pages, size_bytes, updated_at, status, indexed, file_path, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, '已索引', 1, ?, ?)
                """,
                (doc_id, safe_name, folder, ext, version or "Rev.0", len(extracted), len(content), now, str(target), sha),
            )
            for idx, item in enumerate(extracted, 1):
                emb = self._embed_safe(item["content"])
                conn.execute(
                    "INSERT INTO chunks(id, doc_id, page, section, content, highlight, embedding) VALUES (?, ?, ?, ?, ?, '', ?)",
                    (f"{doc_id}-c{idx}", doc_id, item["page"], item["section"], item["content"], emb),
                )
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
                extracted = _extract_text_from_file(
                    file_path,
                    chunk_size=int(cfg.get("chunk_size", 480)),
                    chunk_overlap=int(cfg.get("chunk_overlap", 64)),
                    ocr_enabled=bool(cfg.get("ocr_enabled", True)),
                )
                conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                for idx, item in enumerate(extracted, 1):
                    emb = self._embed_safe(item["content"])
                    conn.execute(
                        "INSERT INTO chunks(id, doc_id, page, section, content, highlight, embedding) VALUES (?, ?, ?, ?, ?, '', ?)",
                        (f"{doc_id}-c{idx}", doc_id, item["page"], item["section"], item["content"], emb),
                    )
                conn.execute(
                    "UPDATE documents SET pages=?, updated_at=?, status='已索引', indexed=1 WHERE id=?",
                    (len(extracted), now, doc_id),
                )
            else:
                # Re-embed existing chunks for seeded docs
                rows = conn.execute("SELECT id, content FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
                for r in rows:
                    emb = self._embed_safe(r["content"])
                    conn.execute("UPDATE chunks SET embedding=? WHERE id=?", (emb, r["id"]))
                conn.execute("UPDATE documents SET updated_at=?, status='已索引', indexed=1 WHERE id=?", (now, doc_id))
        return self.get_document_preview(doc_id)

    def get_chat_history(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM chat_messages ORDER BY rowid ASC").fetchall()
            result = []
            for r in rows:
                result.append({
                    "id": r["id"],
                    "role": r["role"],
                    "question": r["question"],
                    "time": r["time_str"],
                    "payload": json.loads(r["payload_json"]),
                })
            return result

    def clear_chat(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM chat_messages")

    def ask_question(self, question: str, mode: str = "快速问答") -> dict:
        q = question.strip()
        if not q:
            raise ValueError("请输入要查询的问题")
        cfg = self.get_settings()
        self.ensure_embeddings()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.doc_id, c.page, c.section, c.content, c.highlight, c.embedding AS embedding_blob,
                       d.name AS doc_name, d.version, d.pages AS doc_pages, d.folder
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE d.indexed = 1
                """
            ).fetchall()
        candidates = [dict(r) for r in rows]

        top_k = int(cfg.get("top_k", 10))
        if mode == "主题逐文件检查":
            top_k = max(top_k, 12)
        elif mode == "深度全文检查":
            top_k = max(top_k, 20)

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
            },
            mode=mode,
        )

        now_time = datetime.now().strftime("%H:%M")
        now_full = _now_str()
        msg_id = uuid.uuid4().hex[:12]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_messages(id, role, question, time_str, payload_json, created_at) VALUES (?, 'turn', ?, ?, ?, ?)",
                (msg_id, q, now_time, json.dumps(answer_payload, ensure_ascii=False), now_full),
            )
        return {
            "id": msg_id,
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
            left_chunks = [dict(r) for r in conn.execute("SELECT * FROM chunks WHERE doc_id=? ORDER BY page, rowid", (left_id,))]
            right_chunks = [dict(r) for r in conn.execute("SELECT * FROM chunks WHERE doc_id=? ORDER BY page, rowid", (right_id,))]

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
                ws.append([f"[{c.get('index', '')}]", c.get("file_name", ""), f"第 {c.get('page', '')} 页 · {c.get('section', '')}", c.get("content", "")])
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


def seed_demo_project(runtime_root: Path):
    """Seed a clean synthetic demo project matching the 5 prototype screens (no real company data)."""
    store = ProjectStore(runtime_root, "rfq-demo")
    if store.list_documents():
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

    @api.post("/api/projects/{project_id}/documents/{doc_id}/reparse")
    def reparse_document(project_id: str, doc_id: str):
        store = get_store(project_id)
        try:
            return store.reparse_document(doc_id)
        except KeyError as err:
            raise HTTPException(404, str(err)) from None

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
            f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>{info['document']['name']}</title>"
            f"<body style='font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:860px;margin:32px auto;padding:0 20px;line-height:1.8'>"
            f"<h2>{info['document']['name']} ({info['document']['version']})</h2>"
            f"<pre style='white-space:pre-wrap;background:#f8fafc;padding:20px;border-radius:10px;border:1px solid #e5e7eb'>{text}</pre></body></html>"
        )

    @api.get("/api/projects/{project_id}/chat/history")
    def chat_history(project_id: str):
        store = get_store(project_id)
        return {"messages": store.get_chat_history()}

    @api.delete("/api/projects/{project_id}/chat/history")
    def clear_chat_history(project_id: str):
        store = get_store(project_id)
        store.clear_chat()
        return {"ok": True}

    @api.post("/api/projects/{project_id}/ask")
    def ask(project_id: str, payload: dict):
        store = get_store(project_id)
        try:
            turn = store.ask_question(
                str(payload.get("question", "")),
                mode=str(payload.get("mode", "快速问答")),
            )
            return {"turn": turn}
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
        except KeyError as err:
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

    def _build_runtime_status(store: ProjectStore, cfg: dict) -> dict:
        docs = store.list_documents()
        with store._conn() as conn:
            chunk_cnt = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            vec_cnt = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        emb = embedding_health()
        llm = llm_config_status()
        return {
            "model_api": llm["status_text"],
            "vector_service": "本地 BGE 512维就绪" if emb.get("ok") else "向量模型未就绪",
            "index_service": f"运行中（文档 {len(docs)} 篇 · 向量块 {vec_cnt}/{chunk_cnt}）",
            "ocr_service": "运行中" if cfg.get("ocr_enabled") else "已关闭",
            "updated_at": _now_str(),
        }

    @api.get("/api/projects/{project_id}/settings")
    def get_settings(project_id: str):
        store = get_store(project_id)
        cfg = store.get_settings()
        return {"settings": cfg, "status": _build_runtime_status(store, cfg)}

    @api.put("/api/projects/{project_id}/settings")
    def put_settings(project_id: str, payload: dict):
        store = get_store(project_id)
        updated = store.update_settings(payload)
        return {"settings": updated, "status": _build_runtime_status(store, updated)}

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
