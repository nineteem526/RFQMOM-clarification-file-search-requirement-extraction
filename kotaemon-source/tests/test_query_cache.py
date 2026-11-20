import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from pilot_runtime import prepare_runtime

prepare_runtime()

from project_workspace import ProjectWorkspace, now
from query_cache import request_key, reusable
from query_jobs import QueryJobs
from test_extended_features import FixtureChat


class QueryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.workspace = ProjectWorkspace(self.root / "data", "fixture", self.source)
        self.jobs = self.manager(self.workspace)
        self.calls = 0
        self.coverage = []
        self.answer = '{"sections":[]}'

    def manager(self, workspace):
        jobs = QueryJobs(workspace)
        self.addCleanup(jobs.executor.shutdown, wait=True)
        return jobs

    def fake_ask(self, question, **_kwargs):
        self.calls += 1
        identity = uuid.uuid4().hex
        with self.workspace.connection() as conn:
            conn.execute(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,0,?)",
                (identity, now(), self.workspace.revision(), question, "快速问答",
                 json.dumps(self.coverage), '[{"id":"fixture","text":"虚构，待批准"}]', self.answer),
            )
        return identity, [], [], self.coverage

    def run_job(self, question="PLC 型号 S7-1500 的要求", mode="快速问答", context="", **kwargs):
        identity = self.jobs.start(question, mode, context, **kwargs)
        self.jobs.futures[identity].result(timeout=30)
        result = self.jobs.get(identity)
        self.assertEqual(result["status"], "completed", result)
        return result

    def test_exact_repeat_reuses_snapshot_without_retrieval_or_ai(self):
        with patch.object(self.workspace, "ask", side_effect=self.fake_ask):
            first = self.run_job()
            second = self.run_job()
        self.assertEqual(self.calls, 1)
        self.assertEqual(second["cached_from"], first["id"])
        self.assertEqual(second["result"], first["result"])
        self.assertIn("未调用模型", second["stage"])
        self.assertNotEqual(second["session_id"], first["session_id"])
        self.assertFalse(self.workspace.result(first["result"])["saved"])

    def test_fresh_query_and_exact_not_semantic_matching(self):
        with patch.object(self.workspace, "ask", side_effect=self.fake_ask):
            self.run_job()
            fresh = self.run_job(reuse_answer=False)
            latest = self.run_job()
            self.run_job("PLC型号S7-1500的要求")
            self.run_job(context="电气要求")
            self.run_job(mode="深度全文检查")
        self.assertEqual(self.calls, 5)
        self.assertEqual(latest["cached_from"], fresh["id"])

    def test_metadata_scan_and_pipeline_changes_invalidate_without_rewriting_history(self):
        with patch.object(self.workspace, "ask", side_effect=self.fake_ask):
            first = self.run_job()
            with self.workspace.connection() as conn:
                conn.execute("UPDATE state SET value='1' WHERE key='revision'")
            self.run_job()
            self.workspace.scan()
            self.run_job()
            self.jobs.pipeline = "changed-prompt-or-retrieval"
            self.run_job()
        self.assertEqual(self.calls, 4)
        self.assertTrue(self.workspace.result(first["result"])["stale"])

    def test_model_config_invalidates_but_key_rotation_never_enters_fingerprint(self):
        def key():
            return request_key(self.workspace, "问题", "快速问答", "", self.jobs.pipeline)
        original = key()
        with patch.dict("os.environ", {"DUERR_API_KEY": "synthetic-only-not-a-key"}):
            self.assertEqual(key(), original)
        for setting, value in (
            ("DUERR_MODEL", "different-model"),
            ("DUERR_API_VERSION", "different-api"),
            ("DUERR_API_ENDPOINT", "https://invalid.example"),
        ):
            with self.subTest(setting=setting), patch.dict("os.environ", {setting: value}):
                self.assertNotEqual(key(), original)

    def test_no_cache_for_failed_cancelled_empty_or_incomplete_results(self):
        for coverage in (
            [{"status": "未完成"}], [{"scan_failures": ["failure"]}],
            [{"not_indexed_as_current_bytes": ["changed.txt"]}],
            [{"not_sent_to_ai": 2}], [{"errors": ["API failed"]}],
            [{"parser_coverage_at_query": [{"status": "failed"}]}],
        ):
            with self.subTest(coverage=coverage):
                self.assertFalse(reusable({"answer": "answer", "evidence": [1], "coverage": coverage}))
        with patch.object(self.workspace, "ask", side_effect=ValueError("synthetic failure")):
            failed = self.jobs.start("PLC", "快速问答")
            self.jobs.futures[failed].result(timeout=10)
        with patch.object(self.workspace, "ask", side_effect=self.fake_ask):
            self.run_job("PLC")
            with self.workspace.connection() as conn:
                conn.execute("UPDATE query_jobs SET status='cancelled' WHERE question='PLC'")
            self.run_job("PLC")
            self.answer = ""
            self.run_job("PLC空")
            self.run_job("PLC空")
            self.answer = "answer"
            self.coverage = [{"status": "未完成"}]
            self.run_job("PLC未完成")
            self.run_job("PLC未完成")
        self.assertEqual(self.calls, 6)

    def test_restart_reuses_but_other_project_does_not(self):
        with patch.object(self.workspace, "ask", side_effect=self.fake_ask):
            first = self.run_job()
            self.jobs.executor.shutdown(wait=True)
            self.jobs = self.manager(self.workspace)
            second = self.run_job()
        self.assertEqual(second["cached_from"], first["id"])
        other = ProjectWorkspace(self.root / "data", "other", self.source)
        other_jobs = self.manager(other)
        with patch.object(other, "ask", side_effect=ValueError("no external request")) as ask:
            identity = other_jobs.start(first["question"], first["mode"])
            other_jobs.futures[identity].result(timeout=10)
        ask.assert_called_once()
        self.assertEqual(other_jobs.get(identity)["status"], "failed")

    def test_queued_identical_queries_share_only_completed_success(self):
        with patch.object(self.workspace, "ask", side_effect=self.fake_ask):
            with self.workspace.lock:
                identities = [self.jobs.start("PLC", "快速问答") for _ in range(2)]
            for identity in identities:
                self.jobs.futures[identity].result(timeout=10)
        self.assertEqual(self.calls, 1)
        self.assertTrue(self.jobs.get(identities[1])["cached_from"])

    def test_real_pipeline_uses_original_cited_result_on_second_run(self):
        (self.source / "fixture.txt").write_text(
            "虚构设备控制器必须采用 S7-1500，24 V DC 为讨论方案，待批准。", encoding="utf-8",
        )
        scan = self.workspace.scan()
        self.workspace.ingest(scan["scan_id"], ["fixture.txt"], True)
        chat = FixtureChat()
        with patch("project_workspace.CompanyAzureChat", return_value=chat):
            first = self.run_job("PLC S7-1500 要求")
            calls = len(chat.messages)
            second = self.run_job("PLC S7-1500 要求")
        self.assertGreater(calls, 0)
        self.assertEqual(len(chat.messages), calls)
        result = self.workspace.result(second["result"])
        self.assertEqual(first["result"], second["result"])
        self.assertIn("待批准", result["evidence"][0]["text"])
        self.assertEqual(result["coverage"][0]["exact_terms"], ["S7-1500"])

    def test_resume_rejects_changed_pipeline(self):
        with patch.object(self.workspace, "ask", side_effect=ValueError("synthetic")):
            identity = self.jobs.start("PLC", "快速问答")
            self.jobs.futures[identity].result(timeout=10)
        self.jobs.pipeline = "new-version"
        with self.assertRaisesRegex(ValueError, "策略"):
            self.jobs.resume(identity)


if __name__ == "__main__":
    unittest.main()
