import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pilot_runtime import prepare_runtime

prepare_runtime()

from evidence_tools import compare_versions, evidence_context
from project_backup import live_backup, restore_project, verify
from project_parser import parse_local
from project_registry import create_project, projects
from project_workspace import ProjectWorkspace
from query_jobs import QueryJobs


class FixtureChat:
    def __init__(self):
        self.selection_calls = 0
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        payload = json.loads(messages[1].content)
        evidence = payload["evidence"]
        system = messages[0].content
        if '"ids"' in system:
            self.selection_calls += 1
            result = {"ids": [item["id"] for item in evidence]}
        elif '"excerpts"' in system:
            self.selection_calls += 1
            result = {"excerpts": [{"id": item["id"], "quote": item["text"]} for item in evidence]}
        else:
            if "本次生成给普通用户直接阅读的概览" in system:
                evidence = evidence[:2]
            result = {"sections": [{"title": "原文", "points": [
                {"text": "待核对原文", "citations": [
                    {"id": item["id"], "quote": item["text"]}
                ]} for item in evidence
            ]}]}
        return SimpleNamespace(content=json.dumps(result, ensure_ascii=False))


class ExtendedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.data = self.base / "data"
        self.workspace = ProjectWorkspace(self.data, "fixture", self.source)

    def ingest(self, name="old.txt", count=1, value="3.0"):
        path = self.source / name
        path.write_text("\n".join(f"虚构钢构荷载{value} kN/m²，第{i}条，待批准。"
                                  for i in range(count)), encoding="utf-8")
        scan = self.workspace.scan()
        report = self.workspace.ingest(scan["scan_id"], [name], True)
        self.assertNotIn("error", report[0])
        return path

    def manager(self):
        jobs = QueryJobs(self.workspace)
        self.addCleanup(jobs.executor.shutdown, wait=True)
        return jobs

    def test_deep_checks_every_chunk_and_resumes_only_missing_batches(self):
        self.ingest(count=10)
        jobs = self.manager()

        class FailSecondSelection(FixtureChat):
            def invoke(self, messages):
                if '"ids"' in messages[0].content and self.selection_calls == 1:
                    self.selection_calls += 1
                    raise ValueError("synthetic API failure")
                return super().invoke(messages)

        chat = FailSecondSelection()
        with patch("project_workspace.CompanyAzureChat", return_value=chat):
            identity = jobs.start("钢构荷载", "深度全文检查")
            jobs.futures[identity].result(timeout=30)
            self.assertEqual(jobs.get(identity)["status"], "failed")
            with self.workspace.connection() as conn:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM query_checkpoints WHERE job=? AND batch!='intent-v1'", (identity,)
                ).fetchone()[0], 1)
            jobs.resume(identity)
            jobs.futures[identity].result(timeout=30)
        task = jobs.get(identity)
        self.assertEqual(task["status"], "completed", task)
        result = self.workspace.result(task["result"])
        self.assertEqual(len(result["evidence"]), 10)
        self.assertEqual(result["coverage"][0]["checked_chunks"], 10)
        self.assertEqual(result["coverage"][0]["not_sent_to_ai"], 0)
        self.assertEqual(chat.selection_calls, 3)
        answer = json.loads(result["answer"])
        self.assertEqual(sum(len(s["points"]) for s in answer["sections"]), 10)
        self.assertEqual(sum(len(s["points"]) for s in answer["overview"]), 2)
        self.workspace.save_result(task["result"])
        from openpyxl import load_workbook
        book = load_workbook(self.workspace.export(task["result"]))
        try:
            labels = [row[0] for row in book["回答"].iter_rows(min_row=2, values_only=True)]
            self.assertEqual(sum(label.startswith("完整条款") for label in labels), 10)
            self.assertEqual(sum(label.startswith("重点概览") for label in labels), 2)
        finally:
            book.close()

    def test_cancellation_persists_completed_batch_and_can_resume(self):
        self.ingest(count=2)
        jobs = self.manager()
        entered = threading.Event()
        release = threading.Event()

        class SlowChat(FixtureChat):
            def invoke(self, messages):
                if self.selection_calls == 0:
                    entered.set()
                    if not release.wait(10):
                        raise ValueError("fixture timeout")
                return super().invoke(messages)

        chat = SlowChat()
        with patch("project_workspace.CompanyAzureChat", return_value=chat):
            identity = jobs.start("荷载", "深度全文检查")
            self.assertTrue(entered.wait(10))
            jobs.cancel(identity)
            release.set()
            jobs.futures[identity].result(timeout=20)
            self.assertEqual(jobs.get(identity)["status"], "cancelled")
            jobs.resume(identity)
            jobs.futures[identity].result(timeout=20)
        self.assertEqual(jobs.get(identity)["status"], "completed")
        self.assertEqual(chat.selection_calls, 1)

    def test_revision_change_blocks_resume(self):
        self.ingest()
        jobs = self.manager()
        with patch("project_workspace.CompanyAzureChat", side_effect=ValueError("fixture")):
            identity = jobs.start("荷载", "快速问答")
            jobs.futures[identity].result(timeout=15)
        version = self.workspace.coverage()[0]
        self.workspace.update_version(version["id"], "2026-10-01", "", "", True)
        with self.assertRaisesRegex(ValueError, "变化"):
            jobs.resume(identity)

    def test_missing_meeting_source_does_not_turn_protocol_into_mom(self):
        self.ingest(name="技术协议.txt")
        chat = FixtureChat()
        identity, documents, _, coverage = self.workspace.ask(
            "最新MOM对钢构怎么说？", chat=chat, generate_answer=True
        )
        self.assertFalse(documents)
        self.assertFalse(chat.messages)
        self.assertTrue(coverage[0]["meeting_source_missing"])
        self.assertFalse(self.workspace.result(identity)["answer"])

    def test_follow_up_uses_user_question_and_original_evidence_only(self):
        self.ingest()
        chat = FixtureChat()
        identity, documents, _, _ = self.workspace.ask(
            "这个数值待批准吗", chat=chat, context_question="钢构荷载要求", generate_answer=True
        )
        self.assertTrue(documents)
        payload = json.loads(chat.messages[0][1].content)
        self.assertIn("钢构荷载要求", payload["question"])
        self.assertIn("当前问题", payload["question"])
        self.assertIn("follow_up_context_question",
                      self.workspace.result(identity)["coverage"][2])

    def test_version_settings_preserve_snapshots_and_history(self):
        source = self.ingest()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        row = self.workspace.coverage()[0]
        self.workspace.update_version(row["id"], "2026-10-01", "钢构协议", "Rev.A", False)
        self.assertEqual(self.workspace._index()[0], [])
        updated = self.workspace.coverage()[0]
        self.assertEqual(updated["approval"], row["approval"])
        self.assertEqual(updated["business_date"], "2026-10-01")
        self.assertTrue(self.workspace.source_snapshot(row["id"]).is_file())
        self.workspace.update_version(row["id"], "", "钢构协议", "Rev.A", True)
        self.assertTrue(self.workspace._index()[0])
        self.assertIn("兜底", self.workspace.coverage()[0]["date_basis"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_context_and_comparison_keep_original_conditions(self):
        self.ingest(count=6)
        self.ingest(name="new.txt", value="5.0")
        rows = self.workspace.coverage()
        old = next(row for row in rows if row["path"] == "old.txt")
        new = next(row for row in rows if row["path"] == "new.txt")
        markdown, details = compare_versions(self.workspace, old["id"], new["id"], "荷载")
        self.assertIn("3.0", markdown)
        self.assertIn("5.0", markdown)
        self.assertIn("待批准", markdown)
        self.assertIn("不自动", markdown)
        self.assertTrue(details["pairs"])
        doc = self.workspace._index()[0][2]
        context = evidence_context(self.workspace, doc.id_)
        self.assertIn("相邻", context)
        self.assertIn(doc.content, context)
        with self.assertRaises(ValueError):
            evidence_context(self.workspace, "outside-project")

    def test_live_backup_new_project_restore_and_scope(self):
        self.ingest()
        backup = live_backup(self.workspace)
        verify(backup)
        restored = restore_project(self.data, backup.name, "restored", self.source)
        reopened = ProjectWorkspace(self.data, "restored", self.source)
        self.assertTrue(restored.is_dir())
        self.assertEqual(len(reopened._index()[0]), 1)
        verify(live_backup(reopened))
        with self.assertRaises(ValueError):
            restore_project(self.data, backup.name, "fixture", self.source)
        with self.assertRaises(ValueError):
            restore_project(self.data, backup.name, "wrong-scope", self.base)

    def test_registry_only_allows_authorized_folder(self):
        with patch("project_registry.authorized_source", return_value=self.source):
            self.assertEqual(projects(self.data), ["fixture"])
            create_project(self.data, "second")
            self.assertEqual(projects(self.data), ["fixture", "second"])
            with self.assertRaises(ValueError):
                create_project(self.data, "second")
            with self.assertRaises(ValueError):
                create_project(self.data, "..\\escape")

    def test_restart_marks_unfinished_tasks_interrupted(self):
        jobs = self.manager()
        with self.workspace.connection() as conn:
            conn.execute(
                "INSERT INTO query_jobs(id,question,context,mode,revision,status) "
                "VALUES ('fixture','荷载','','快速问答',?,'running')",
                (self.workspace.revision(),),
            )
        jobs.executor.shutdown(wait=True)
        restarted = self.manager()
        self.assertEqual(restarted.get("fixture")["status"], "interrupted")

    def test_local_ocr_recognizes_real_chinese_raster_and_preserves_source(self):
        from local_ocr import ocr_root
        if not (ocr_root() / ".venv" / "Scripts" / "python.exe").is_file():
            self.skipTest("Independent local OCR environment is not installed.")
        import fitz

        path = self.source / "raster.pdf"
        text_pdf = fitz.open()
        page = text_pdf.new_page()
        page.insert_text((40, 100), "虚构控制电压", fontsize=20, fontname="china-s")
        page.insert_text((220, 100), "24 V DC", fontsize=20, fontname="helv")
        page.insert_text((40, 145), "虚构钢构荷载", fontsize=20, fontname="china-s")
        page.insert_text((220, 145), "3.0 kN", fontsize=20, fontname="helv")
        pixels = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        raster = fitz.open()
        raster.new_page().insert_image(page.rect, stream=pixels)
        raster.save(path)
        raster.close()
        text_pdf.close()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        chunks, _ = parse_local(path, use_ocr=True)
        text = "\n".join(chunk["text"] for chunk in chunks)
        self.assertIn("24", text)
        self.assertIn("3.0", text)
        self.assertIn("钢构", text)
        self.assertTrue(all("PDF物理第1页" in chunk["location"] for chunk in chunks))
        self.assertTrue(all("OCR坐标" in chunk["location"] for chunk in chunks))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_ocr_enabled_pdf_records_coordinates_and_failure_is_explicit(self):
        import fitz

        path = self.source / "image.pdf"
        pdf = fitz.open()
        pdf.new_page()
        pdf.save(path)
        pdf.close()
        with patch("local_ocr.recognize_page", return_value=[
            {"text": "虚构PLC 24 V DC", "confidence": 0.94,
             "box": [[10, 20], [180, 20], [180, 40], [10, 40]]},
        ]):
            chunks, warnings = parse_local(path, use_ocr=True)
        self.assertIn("OCR坐标", chunks[0]["location"])
        self.assertIn("24 V DC", chunks[0]["text"])
        self.assertTrue(any("OCR" in item for item in warnings))
        with patch("local_ocr.recognize_page", side_effect=ValueError("fixture OCR error")):
            with self.assertRaisesRegex(ValueError, "OCR失败.*fixture OCR error"):
                parse_local(path, use_ocr=True)


if __name__ == "__main__":
    unittest.main()
