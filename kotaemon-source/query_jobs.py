"""Persistent, project-scoped query checkpoints with cooperative cancellation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import threading
import time
import uuid
from query_cache import pipeline_fingerprint, request_key, reusable


def _now():
    return datetime.now(timezone.utc).isoformat()


class QueryCancelled(ValueError):
    pass


class QueryJobs:
    def __init__(self, workspace):
        self.workspace = workspace
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="project-query")
        self.lock = threading.Lock()
        self.futures = {}
        self.pipeline = pipeline_fingerprint()
        with workspace.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS query_sessions(
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created TEXT NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS query_jobs(
                    id TEXT PRIMARY KEY, question TEXT NOT NULL, context TEXT NOT NULL,
                    mode TEXT NOT NULL, revision INTEGER NOT NULL,
                    status TEXT NOT NULL, cancel INTEGER NOT NULL DEFAULT 0,
                    progress REAL NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT '',
                    elapsed REAL NOT NULL DEFAULT 0, result TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS query_checkpoints(
                    job TEXT NOT NULL REFERENCES query_jobs(id),
                    batch TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job,batch));
            """)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(query_jobs)")}
            if "session_id" not in columns:
                conn.execute("ALTER TABLE query_jobs ADD COLUMN session_id TEXT")
            if "created" not in columns:
                conn.execute("ALTER TABLE query_jobs ADD COLUMN created TEXT NOT NULL DEFAULT ''")
            for name in ("cache_key", "cached_from"):
                if name not in columns:
                    conn.execute(f"ALTER TABLE query_jobs ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
            if "reuse_answer" not in columns:
                conn.execute("ALTER TABLE query_jobs ADD COLUMN reuse_answer INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS query_jobs_cache ON query_jobs(cache_key,status)"
            )
            conn.execute(
                "UPDATE query_jobs SET created=COALESCE(NULLIF(created,''), "
                "(SELECT created FROM results WHERE results.id=query_jobs.result), '')"
            )
            legacy = conn.execute(
                "SELECT id,question,created FROM query_jobs WHERE session_id IS NULL"
            ).fetchall()
            for row in legacy:
                conn.execute(
                    "INSERT OR IGNORE INTO query_sessions(id,title,created,updated) VALUES (?,?,?,?)",
                    (row["id"], row["question"][:72], row["created"], row["created"]),
                )
                conn.execute("UPDATE query_jobs SET session_id=? WHERE id=?",
                             (row["id"], row["id"]))
            conn.execute(
                "UPDATE query_jobs SET status='interrupted', "
                "error='服务曾中断，可在资料未变化时继续。' "
                "WHERE status IN ('queued','running')"
            )

    def start(self, question, mode, context="", session_id=None, reuse_answer=True):
        if mode not in {"快速问答", "主题逐文件检查", "深度全文检查"}:
            raise ValueError("未知查询方式。")
        if not isinstance(question, str) or not question.strip() or len(question) > 400:
            raise ValueError("请输入1至400字符的问题。")
        if not isinstance(context, str) or len(context) > 400:
            raise ValueError("追问上下文问题最多400字符。")
        if not isinstance(reuse_answer, bool):
            raise ValueError("回答复用选项无效。")
        question, context = question.strip(), context.strip()
        identity = uuid.uuid4().hex
        timestamp = _now()
        with self.workspace.lock, self.workspace.connection() as conn:
            cache_key = request_key(self.workspace, question, mode, context, self.pipeline)
            if session_id is None:
                session_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO query_sessions(id,title,created,updated) VALUES (?,?,?,?)",
                    (session_id, question.strip()[:72], timestamp, timestamp),
                )
            elif conn.execute(
                "SELECT 1 FROM query_sessions WHERE id=?", (session_id,)
            ).fetchone() is None:
                raise ValueError("所选会话不属于当前项目或已不存在。")
            conn.execute(
                "UPDATE query_sessions SET title=CASE WHEN title='' THEN ? ELSE title END, "
                "updated=? WHERE id=?",
                (question.strip()[:72], timestamp, session_id),
            )
            conn.execute(
                "INSERT INTO query_jobs(id,question,context,mode,revision,status,session_id,created,"
                "cache_key,reuse_answer) VALUES (?,?,?,?,?,'queued',?,?,?,?)",
                (identity, question, context, mode, self.workspace.revision(), session_id, timestamp,
                 cache_key, int(reuse_answer)),
            )
        self._submit(identity)
        return identity

    def _submit(self, identity):
        with self.lock:
            if identity in self.futures and not self.futures[identity].done():
                raise ValueError("该任务仍在运行。")
            self.futures[identity] = self.executor.submit(self._run, identity)

    def get(self, identity):
        with self.workspace.connection() as conn:
            row = conn.execute("SELECT * FROM query_jobs WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("任务不属于当前项目。")
        return dict(row)

    def list(self):
        with self.workspace.connection() as conn:
            return [(f"{row['status']} | {row['question'][:55]}", row["id"])
                    for row in conn.execute("SELECT id,status,question FROM query_jobs ORDER BY rowid DESC LIMIT 100")]

    def sessions(self, search=""):
        if not isinstance(search, str):
            raise ValueError("历史搜索内容无效。")
        term = search.strip().lower()
        statement = (
            "SELECT s.id,s.title,s.created,s.updated,"
            "(SELECT j.status FROM query_jobs j WHERE j.session_id=s.id "
            "ORDER BY j.rowid DESC LIMIT 1) AS status "
            "FROM query_sessions s WHERE ?='' OR instr(lower(s.title),?)>0 OR EXISTS "
            "(SELECT 1 FROM query_jobs j WHERE j.session_id=s.id "
            "AND instr(lower(j.question),?)>0) "
            "ORDER BY CASE WHEN s.updated='' THEN s.created ELSE s.updated END DESC, "
            "s.rowid DESC"
        )
        if not term:
            statement += " LIMIT 200"
        with self.workspace.connection() as conn:
            rows = conn.execute(
                statement,
                (term, term, term),
            ).fetchall()
        return [dict(row) for row in rows]

    def session_choices(self, search=""):
        choices = []
        for row in self.sessions(search):
            try:
                created = datetime.fromisoformat(row["created"])
                date_label = created.astimezone().strftime("%Y-%m-%d")
            except ValueError:
                date_label = "既有记录"
            title = row["title"] or "新会话"
            choices.append((f"{date_label}\n{' '.join(title.split())}", row["id"]))
        return choices

    def session_jobs(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("请选择一个历史会话。")
        with self.workspace.connection() as conn:
            if conn.execute(
                "SELECT 1 FROM query_sessions WHERE id=?", (session_id,)
            ).fetchone() is None:
                raise ValueError("会话不属于当前项目或已不存在。")
            return [dict(row) for row in conn.execute(
                "SELECT * FROM query_jobs WHERE session_id=? ORDER BY rowid",
                (session_id,),
            )]

    def cancel(self, identity):
        row = self.get(identity)
        if row["status"] not in {"queued", "running"}:
            raise ValueError("该任务已结束，无需停止；可以查看结果或继续中断任务。")
        with self.workspace.connection() as conn:
            conn.execute(
                "UPDATE query_jobs SET cancel=1 WHERE id=? AND status IN ('queued','running')",
                (identity,),
            )
        return "已请求停止；正在进行的公司API请求返回或超时后停止，已完成批次会保留。"

    def resume(self, identity):
        row = self.get(identity)
        if row["status"] not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("只有失败、取消或中断的任务可以继续。")
        if row["revision"] != self.workspace.revision():
            raise ValueError("资料或元数据已变化，请重新提问，不能混用旧检查点。")
        if row["cache_key"] != request_key(
            self.workspace, row["question"], row["mode"], row["context"], self.pipeline
        ):
            raise ValueError("查询策略、模型配置或扫描状态已变化，请重新提问，不混用旧检查点。")
        with self.workspace.connection() as conn:
            conn.execute("UPDATE query_jobs SET status='queued',cancel=0,error='' WHERE id=?", (identity,))
        self._submit(identity)
        return identity

    def _run(self, identity):
        row = self.get(identity)
        started = time.monotonic()

        def update(**values):
            values["elapsed"] = row["elapsed"] + time.monotonic() - started
            with self.workspace.connection() as conn:
                conn.execute(
                    "UPDATE query_jobs SET " + ",".join(f"{key}=?" for key in values) + " WHERE id=?",
                    (*values.values(), identity),
                )
                conn.execute(
                    "UPDATE query_sessions SET updated=? WHERE id=?",
                    (_now(), row["session_id"]),
                )

        def check():
            current = self.get(identity)
            if current["cancel"]:
                raise QueryCancelled("已停止；可继续未完成批次。")
            if current["revision"] != self.workspace.revision():
                raise ValueError("资料已变化，任务停止；请重新查询。")

        def checkpoint(batch, compute):
            check()
            with self.workspace.connection() as conn:
                previous = conn.execute(
                    "SELECT payload FROM query_checkpoints WHERE job=? AND batch=?",
                    (identity, batch),
                ).fetchone()
            if previous:
                return json.loads(previous[0])
            value = compute()
            with self.workspace.connection() as conn:
                conn.execute("INSERT INTO query_checkpoints VALUES (?,?,?)",
                             (identity, batch, json.dumps(value, ensure_ascii=False)))
            check()
            return value

        def progress(value, desc):
            if value < 1:
                check()
            update(progress=value, stage=desc)

        try:
            with self.workspace.lock:
                check()
                if row["cache_key"] != request_key(
                    self.workspace, row["question"], row["mode"], row["context"], self.pipeline
                ):
                    raise ValueError("查询配置或扫描状态已变化，请重新提问。")
                update(status="running")
                if row["reuse_answer"]:
                    with self.workspace.connection() as conn:
                        previous = conn.execute(
                            "SELECT id,result FROM query_jobs WHERE cache_key=? AND status='completed' "
                            "AND cached_from='' ORDER BY rowid DESC LIMIT 1", (row["cache_key"],),
                        ).fetchone()
                    if previous:
                        saved = self.workspace.result(previous["result"])
                        if not saved["stale"] and reusable(saved):
                            check()
                            update(status="completed", result=previous["result"],
                                   cached_from=previous["id"], progress=1,
                                   stage="已复用相同条件下的回答；未调用模型")
                            return
                identity_result, _, _, _ = self.workspace.ask(
                    row["question"], comprehensive=row["mode"] != "快速问答",
                    deep=row["mode"] == "深度全文检查", generate_answer=True,
                    context_question=row["context"], checkpoint=checkpoint,
                    progress=progress, cancel_check=check,
                    recognize_intent=True,
                )
                check()
                update(status="completed", result=identity_result, progress=1, stage="查询完成")
        except QueryCancelled as error:
            update(status="cancelled", error=str(error), stage="已停止，可继续")
        except Exception as error:
            # Persist the failure for an explicit UI display and a resumable retry.
            update(status="failed", error=f"{type(error).__name__}: {error}", stage="失败，可重试")
