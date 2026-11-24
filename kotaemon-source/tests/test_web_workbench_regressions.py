import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient
from openpyxl import Workbook

from eval_benchmark import _snippet_in_chunks
from rag_engine import generate_structured_answer, llm_config_status, verify_grounded_answer
from web_app import ProjectStore, _parse_document, create_web_app


class WorkbenchRegressions(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        for key in ("DUERR_API_APPROVED_ENDPOINT", "DUERR_API_ENDPOINT", "DUERR_API_KEY"):
            os.environ.pop(key, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_failed_upload_and_empty_project_return_no_evidence(self):
        with TestClient(create_web_app(self.root)) as client:
            client.post("/api/projects", json={"id": "test", "name": "Test"}).raise_for_status()
            base = "/api/projects/test"
            req = {"question": "Rated voltage?", "mode": "深度全文检查"}
            turn = client.post(base + "/ask", json=req).raise_for_status().json()["turn"]["payload"]
            self.assertEqual(turn["citations"], [])
            self.assertEqual(turn["mode"], "更多候选检索")
            self.assertEqual(turn["retrieval"]["evidence_count"], 0)
            self.assertIn("未检查全文", turn["retrieval"]["scope"])
            response = client.post(base + "/documents/upload", files={"files": ("damaged.pdf", b"incomplete", "application/pdf")})
            doc = response.raise_for_status().json()["uploaded"][0]
            self.assertEqual(doc["indexed"], 0)
            self.assertEqual(doc["status"], "解析失败")
            self.assertTrue(doc["parse_warnings"])
            self.assertEqual(client.get(base + "/documents/" + doc["id"] + "/preview").json()["chunks"], [])
            turn = client.post(base + "/ask", json=req).raise_for_status().json()["turn"]["payload"]
            self.assertEqual(turn["retrieval"]["total_candidates"], 0)
            self.assertNotIn("cached", turn)

    def test_physical_pages_and_ocr(self):
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((40, 40), "Rated voltage: 18 V DC")
            pdf.new_page()
            data = pdf.tobytes()
        with TestClient(create_web_app(self.root)) as client, \
             patch.object(ProjectStore, "_embed_safe", return_value=None), \
             patch("local_ocr.recognize_page", return_value=[{"text": "IP54", "confidence": 0.94}]) as ocr:
            client.post("/api/projects", json={"id": "test", "name": "Test"}).raise_for_status()
            base = "/api/projects/test"
            client.put(base + "/settings", json={"chunk_size": 128}).raise_for_status()
            response = client.post(base + "/documents/upload", files={"files": ("manual.pdf", data, "application/pdf")})
            doc = response.raise_for_status().json()["uploaded"][0]
            self.assertEqual(doc["pages"], 2)
            self.assertEqual(doc["status"], "部分索引，需核对")
            self.assertEqual(ocr.call_args.args[1], 2)
            endpoint = base + "/documents/" + doc["id"]
            self.assertEqual(client.get(endpoint + "/page-image/2").headers["content-type"], "image/png")
            self.assertEqual(client.get(endpoint + "/page-image/3").status_code, 404)
            preview = client.get(endpoint + "/preview?page=2").raise_for_status().json()
            self.assertIn("IP54", preview["active_chunk"]["content"])
            self.assertEqual(client.get(base + "/source/" + doc["id"]).content, data)

    def test_workbook_keeps_blank_columns_and_non_pdf_location(self):
        book = Workbook()
        book.active.append(["设备", "电压", "防护等级"])
        book.active.append(["柜体", None, "IP54"])
        path = self.root / "sheet.xlsx"
        book.save(path)
        parsed = _parse_document(path)
        text = parsed.chunks[0]["content"]
        self.assertIn("防护等级: IP54", text)
        self.assertNotIn("电压: IP54", text)
        self.assertEqual(parsed.pages, 0)

    def test_document_comparison_serializes_only_display_fields(self):
        with TestClient(create_web_app(self.root)) as client, \
             patch.object(ProjectStore, "_embed_safe", return_value=b"\xff"):
            client.post("/api/projects", json={"id": "test", "name": "Test"}).raise_for_status()
            base = "/api/projects/test"
            files = [("files", (name, text)) for name, text in
                     (("left.txt", b"Cabinet IP54"), ("right.txt", b"Cabinet IP65"))]
            docs = client.post(base + "/documents/upload", files=files).raise_for_status().json()["uploaded"]
            payload = {"left_id": docs[0]["id"], "right_id": docs[1]["id"]}
            result = client.post(base + "/compare", json=payload).raise_for_status().json()
            self.assertTrue(result["groups"])
            self.assertNotIn("embedding", result["groups"][0]["rows"][0]["left"])

    def test_legacy_index_is_reparsed_without_reupload(self):
        store = ProjectStore(self.root, "legacy")
        path = self.root / "original.txt"
        path.write_text("Rated voltage: 18 V DC", encoding="utf-8")
        with sqlite3.connect(store.db_path) as conn:
            conn.execute("ALTER TABLE documents DROP COLUMN parse_warnings")
            conn.execute("INSERT INTO documents(id,name,folder,file_type,pages,updated_at,file_path) VALUES ('d','original.txt','test','TXT',7,'today',?)", (str(path),))
            conn.execute("INSERT INTO chunks(id,doc_id,section,content) VALUES ('old','d','文件摘要','placeholder')")
        conn.close()
        store = ProjectStore(self.root, "legacy")
        self.assertEqual(store.list_documents()[0]["status"], "待重新解析")
        self.assertEqual(store.get_document_preview("d")["chunks"], [])
        with patch.object(ProjectStore, "_embed_safe", return_value=None):
            result = store.reparse_document("d")
        self.assertEqual(result["document"]["indexed"], 1)
        self.assertEqual(result["chunks"][0]["content"], "Rated voltage: 18 V DC")

    def test_model_claims_require_exact_citations_and_values(self):
        citations = [{"index": 1, "content": "XFD12: D.C. 18 V. Enclosure IP54."},
                     {"index": 2, "content": "待批准建议：改为 IP65。"}]
        accepted = verify_grounded_answer({"conclusion": "XFD12：18 V DC，IP54 [1]"}, citations)
        self.assertTrue(accepted["passed"], accepted["warnings"])
        for proposal in (
            {"conclusion": "IP54"},
            {"conclusion": "IP99 [1]"},
            {"conclusion": "IP65 [1]"},
            {"conclusion": "18 V AC [1]"},
            {"conclusion": "IP65 已批准 [2]"},
            {"conclusion": "18 V DC [9]"},
            {"conclusion": "IP54 [1]", "table": [{"item": "防护等级", "value": "IP99 [1]"}]},
        ):
            with self.subTest(proposal=proposal):
                self.assertFalse(verify_grounded_answer(proposal, citations)["passed"])

    def test_bad_online_answer_is_discarded_and_voltage_is_quoted(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "manual.pdf", "page": 2,
                 "content": "[表格 1] Model: | XFD12\nDrilling capacities: XFD12, 13 mm\n"
                            "Fastening capacities: XFD12, 10 mm\nRated voltage: D.C. 18 V\nWeight: 2 kg",
                 "semantic_score": 0.9}]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused-test-value"}):
            self.assertFalse(llm_config_status()["configured"])
        with patch("rag_engine._call_external_llm", return_value={"conclusion": "XFD12 额定电压为 99 V [1]"}) as provider:
            question = "What is the rated voltage for the XFD12 model?"
            generate_structured_answer(question, hits, "demo", {})
            provider.assert_not_called()
            result = generate_structured_answer(question, hits, "demo", {}, allow_external=True)
            self.assertGreaterEqual(provider.call_count, 1)
        self.assertLessEqual(provider.call_count, 2)
        self.assertIn("18 V", result["conclusion"])
        self.assertNotIn("capacities", result["conclusion"])
        self.assertNotIn("99 V", result["conclusion"])
        self.assertTrue(result["verification"]["warnings"])

    def test_benchmark_requires_one_exact_snippet(self):
        expected = "Cabinet IP54 24 V DC"
        self.assertFalse(_snippet_in_chunks(expected, [{"content": "Cabinet IP99 24 V DC"}]))
        self.assertFalse(_snippet_in_chunks(expected, [{"content": "Cabinet IP54"}, {"content": "24 V DC"}]))
        self.assertFalse(_snippet_in_chunks("18 V", [{"content": "118 V"}]))
        self.assertTrue(_snippet_in_chunks(expected, [{"content": "Cabinet IP54\n24 V DC"}]))


if __name__ == "__main__":
    unittest.main()
