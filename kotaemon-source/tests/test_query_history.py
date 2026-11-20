import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from pilot_runtime import prepare_runtime

prepare_runtime()

from project_workspace import ProjectWorkspace
from query_jobs import QueryJobs


class QueryHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.source = base / "source"
        self.source.mkdir()
        self.workspace = ProjectWorkspace(base / "data", "fixture", self.source)
        self.jobs = QueryJobs(self.workspace)
        self.addCleanup(self.jobs.executor.shutdown, wait=True)

    def fake_ask(self, question, **_kwargs):
        identity = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.workspace.connection() as conn:
            conn.execute(
                "INSERT INTO results "
                "(id,created,revision,question,mode,coverage,evidence,saved,answer) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (identity, timestamp, self.workspace.revision(), question,
                 "快速问答", "[]", "[]", 0, ""),
            )
        return identity, [], [], []

    def test_followups_share_persistent_searchable_session(self):
        with patch.object(self.workspace, "ask", side_effect=self.fake_ask):
            first = self.jobs.start("钢构荷载要求", "快速问答")
            self.jobs.futures[first].result(timeout=10)
            session_id = self.jobs.get(first)["session_id"]
            second = self.jobs.start(
                "供货范围有哪些", "快速问答", "钢构荷载要求", session_id=session_id
            )
            self.jobs.futures[second].result(timeout=10)

        rows = self.jobs.session_jobs(session_id)
        self.assertEqual([row["id"] for row in rows], [first, second])
        self.assertTrue(all(row["status"] == "completed" for row in rows))
        self.assertEqual(self.jobs.sessions()[0]["title"], "钢构荷载要求")
        self.assertEqual(
            [value for _, value in self.jobs.session_choices("供货范围")],
            [session_id],
        )

        self.jobs.executor.shutdown(wait=True)
        reopened = QueryJobs(self.workspace)
        self.addCleanup(reopened.executor.shutdown, wait=True)
        self.assertEqual(len(reopened.session_jobs(session_id)), 2)

    def test_legacy_jobs_migrate_to_single_query_sessions(self):
        identity = "a" * 32
        legacy_workspace = ProjectWorkspace(
            Path(self.temp.name) / "legacy-data", "legacy", self.source
        )
        with legacy_workspace.connection() as conn:
            conn.executescript("""
                CREATE TABLE query_jobs(
                    id TEXT PRIMARY KEY, question TEXT NOT NULL, context TEXT NOT NULL,
                    mode TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL,
                    cancel INTEGER NOT NULL DEFAULT 0, progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT '', elapsed REAL NOT NULL DEFAULT 0,
                    result TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '');
                CREATE TABLE query_checkpoints(
                    job TEXT NOT NULL, batch TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(job,batch));
            """)
            conn.execute(
                "INSERT INTO query_jobs(id,question,context,mode,revision,status,result) "
                "VALUES (?,?,?,?,?,'completed',?)",
                (identity, "旧查询：PLC控制电压", "", "快速问答",
                 legacy_workspace.revision(), ""),
            )
        migrated = QueryJobs(legacy_workspace)
        self.addCleanup(migrated.executor.shutdown, wait=True)

        session = migrated.sessions()[0]
        self.assertEqual(session["id"], identity)
        self.assertEqual(session["title"], "旧查询：PLC控制电压")
        self.assertEqual(migrated.session_jobs(identity)[0]["session_id"], identity)
        self.assertEqual(migrated.session_choices()[0][0][:4], "既有记录")

    def test_history_labels_separate_date_and_preserve_full_title(self):
        title = "钢构荷载及防腐要求" * 12
        with patch.object(self.jobs, "sessions", return_value=[{
            "id": "fixture", "created": "2026-10-29T00:00:00+00:00",
            "title": title,
        }]):
            label, identity = self.jobs.session_choices()[0]
        date_label, displayed_title = label.split("\n", 1)
        self.assertRegex(date_label, r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(displayed_title, title)
        self.assertEqual(identity, "fixture")

    def test_search_reaches_sessions_older_than_recent_window(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        with self.workspace.connection() as conn:
            for index in range(201):
                identity = f"{index:032x}"
                timestamp = (start + timedelta(seconds=index)).isoformat()
                title = f"Archive marker {index:03d}"
                conn.execute(
                    "INSERT INTO query_sessions(id,title,created,updated) VALUES (?,?,?,?)",
                    (identity, title, timestamp, timestamp),
                )
                conn.execute(
                    "INSERT INTO query_jobs"
                    "(id,question,context,mode,revision,status,session_id,created) "
                    "VALUES (?,?,?,?,?,'completed',?,?)",
                    (identity, title, "", "快速问答", self.workspace.revision(),
                     identity, timestamp),
                )

        recent = self.jobs.sessions()
        self.assertEqual(len(recent), 200)
        self.assertNotIn(f"{0:032x}", [row["id"] for row in recent])
        matches = self.jobs.sessions("Archive marker 000")
        self.assertEqual([row["id"] for row in matches], [f"{0:032x}"])


if __name__ == "__main__":
    unittest.main()
