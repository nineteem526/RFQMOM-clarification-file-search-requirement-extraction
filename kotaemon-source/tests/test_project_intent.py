import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pilot_runtime import prepare_runtime

prepare_runtime()

from project_intent import recognize, notice, answer_instruction
from project_workspace import ProjectWorkspace, render_answer
from query_jobs import QueryJobs
from citation_location import short_location
from project_answer import render_synthesis
from kotaemon.base import Document
from evidence_tools import evidence_context


class IntentTests(unittest.TestCase):
    def test_local_routes_do_not_call_model(self):
        chat = SimpleNamespace(invoke=lambda _: self.fail("Local case must not call company API"))
        cases = [
            ("推荐旅游路线", "", "out_of_scope"),
            ("你好", "", "help"),
            ("温度是多少？", "", "clarify"),
            ("这个呢", "", "clarify"),
            ("PLC电压是多少？", "", "parameter"),
            ("钢构设计有哪些要求？按荷载、防腐、供货边界整理。", "", "summary"),
            ("电气柜由谁负责安装？", "", "responsibility"),
            ("钢构这条要求出处在哪一页？", "", "citation"),
            ("新旧协议对比一下", "", "comparison"),
            ("温度是多少？", "烘干炉有哪些要求", "parameter"),
        ]
        for question, context, expected in cases:
            with self.subTest(question=question, context=context):
                route = recognize(question, context, chat)
                self.assertEqual(route["intent"], expected)
                self.assertEqual(route["method"], "local")
        self.assertIn("尚未实现", notice(recognize("客户供应商RFQ漏项比较")))
        self.assertIn("单位", answer_instruction(recognize("PLC电压是多少")))

    def test_unfamiliar_question_uses_only_question_context_no_documents(self):
        calls = []
        def invoke(messages):
            calls.append(json.loads(messages[1].content))
            return SimpleNamespace(content='{"intent":"lookup","clarification":""}')
        result = recognize("槽液循环过滤的选型依据", chat=SimpleNamespace(invoke=invoke))
        self.assertEqual(result["method"], "company-question-only")
        self.assertEqual(set(calls[0]), {"question", "previous_user_question"})
        self.assertNotIn("evidence", calls[0])

    def test_invalid_router_results_and_transport_errors_are_explicit(self):
        for response in [
            "not json", '{"intent":"execute","clarification":""}',
            '{"intent":"clarify","clarification":""}',
            '{"intent":"lookup","clarification":"猜一个"}',
        ]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                recognize("槽液循环过滤", chat=SimpleNamespace(
                    invoke=lambda _: SimpleNamespace(content=response)))
        def fail(_):
            raise ConnectionError("company unavailable")
        with self.assertRaises(ConnectionError):
            recognize("槽液循环过滤", chat=SimpleNamespace(invoke=fail))

    def test_stopped_routes_are_persisted_without_loading_index_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            workspace = ProjectWorkspace(base / "data", "fixture", source)
            jobs = QueryJobs(workspace)
            try:
                with patch.object(workspace, "_index", side_effect=AssertionError("No index")), \
                        patch("project_workspace.CompanyAzureChat") as chat:
                    for question, category in [
                        ("推荐旅游路线", "out_of_scope"), ("温度是多少", "clarify"),
                        ("对比新旧协议", "comparison"), ("你是谁", "help"),
                    ]:
                        identity = jobs.start(question, "深度全文检查")
                        jobs.futures[identity].result(timeout=15)
                        task = jobs.get(identity)
                        self.assertEqual(task["status"], "completed", task)
                        result = workspace.result(task["result"])
                        self.assertEqual(result["evidence"], [])
                        self.assertEqual(result["answer"], "")
                        self.assertEqual(result["coverage"][0]["intent"]["intent"], category)
                        self.assertNotIn("资料中未找到", render_answer([], result["coverage"]))
                        self.assertNotIn("来源范围不足", render_answer([], result["coverage"]))
                    chat.return_value.invoke.assert_not_called()
            finally:
                jobs.executor.shutdown(wait=True)


class CitationLocationTests(unittest.TestCase):
    def test_locations_are_source_native_not_inferred_pages(self):
        for location, expected in [
            ("PDF物理第3页（非印刷页码）", "PDF第3页"),
            ("PDF物理第4页；本地OCR；置信度0.9", "PDF第4页"),
            ("正文第199段（不提供页码）", "第199段"),
            ("本地转换副本 converted-a.docx / 正文第8段", "转换稿第8段"),
            ("正文表2，第5行", "表2行5"),
            ("工作表 参数!A5:G5；字符1-40", "参数!A5:G5"),
            ("文本第1行", ""),
        ]:
            with self.subTest(location=location):
                self.assertEqual(short_location({"location": location}), expected)

    def test_answer_citations_display_pdf_pages_and_word_paragraphs(self):
        for location, label in [
            ("PDF物理第3页（非印刷页码）", "PDF第3页"),
            ("正文第199段（不提供页码）", "第199段"),
        ]:
            doc = Document(id_="fixture", content="原文", metadata={"location": location})
            result = {"sections": [{"title": "依据", "points": [
                {"text": "条款", "citations": [{"id": doc.id_, "quote": doc.content}]}]}]}
            output = render_synthesis(result, [doc])
            self.assertIn(f"[1 · {label}](/context/fixture)", output)
            if "段" in location:
                self.assertNotIn("页", output)

    def test_compact_citation_details_show_approval_status(self):
        doc = Document(id_="fixture", content="原文", metadata={
            "file_name": "协议.docx",
            "date": "2026-02-13",
            "date_basis": "文件修改时间兜底",
            "location": "正文第7段（不提供页码）",
            "approval": "未确认；以原文措辞为准",
        })
        output = render_answer([doc], [], compact=True)
        self.assertIn("批准状态：未确认；以原文措辞为准", output)

        unmarked = Document(id_="unmarked", content="原文", metadata={
            "file_name": "旧资料.docx",
            "date": "2026-02-13",
            "date_basis": "文件修改时间兜底",
            "location": "正文第1段（不提供页码）",
        })
        output = render_answer([unmarked], [], compact=True)
        self.assertIn("批准状态：未标注", output)

    def test_pdf_context_has_exact_page_link_and_image_preview(self):
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            path = source / "fixture.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text((40, 80), "Fixture first page.")
            pdf.new_page().insert_text((40, 80), "Fixture voltage 24 V DC.")
            pdf.save(path)
            pdf.close()
            workspace = ProjectWorkspace(base / "data", "fixture", source)
            scan = workspace.scan()
            workspace.ingest(scan["scan_id"], [path.name], True)
            doc = next(d for d in workspace._index()[0] if "第2页" in d.metadata["location"])
            output = evidence_context(workspace, doc.id_)
            self.assertIn("#page=2", output)
            self.assertIn(f"/preview/{doc.id_}", output)
            self.assertIn("非印刷页码", output)
            self.assertTrue(workspace.pdf_evidence_preview(doc.id_).startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
