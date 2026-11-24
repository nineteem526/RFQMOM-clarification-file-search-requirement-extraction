import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import fitz
from fastapi.testclient import TestClient
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont

import local_ocr
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
        os.environ["RFQ_LOCAL_LLM_CONFIG"] = str(self.root / "local_llm_config.json")

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

    def test_pdf_structured_table_keeps_header_on_split_and_no_body_duplication(self):
        pdf_path = self.root / "table_spec.pdf"
        headers = ["Param", "Value", "Unit"]
        rows = [[f"Item-{i:02d}", f"{100 + i}", "V"] for i in range(1, 11)]
        all_rows = [headers] + rows
        with fitz.open() as pdf:
            page = pdf.new_page(width=595, height=842)
            page.insert_text((50, 48), "Outside prose note: verify cabinet wiring.")
            x0, y0, col_w, row_h = 50, 80, 140, 24
            for r_idx, row in enumerate(all_rows):
                ry = y0 + r_idx * row_h
                for c_idx, val in enumerate(row):
                    rx = x0 + c_idx * col_w
                    page.draw_rect(fitz.Rect(rx, ry, rx + col_w, ry + row_h))
                    page.insert_text((rx + 8, ry + 16), val)
            pdf.save(pdf_path)

        parsed = _parse_document(pdf_path, chunk_size=160, chunk_overlap=20, ocr_enabled=False)
        table_chunks = [c for c in parsed.chunks if c.get("chunk_type") == "table"]
        text_chunks = [c for c in parsed.chunks if c.get("chunk_type") == "text"]
        self.assertGreaterEqual(len(table_chunks), 2)
        for tc in table_chunks:
            self.assertIn("[表格 1] Param | Value | Unit", tc["content"])
        joined_tables = "\n".join(tc["content"] for tc in table_chunks)
        for i in range(1, 11):
            self.assertIn(f"Param: Item-{i:02d} | Value: {100 + i} | Unit: V", joined_tables)
        joined_text = "\n".join(tc["content"] for tc in text_chunks)
        self.assertIn("Outside prose note: verify cabinet wiring.", joined_text)
        self.assertNotIn("Item-01", joined_text)

    def test_pdf_clause_chunking_and_header_footer_suppression(self):
        pdf_path = self.root / "clauses.pdf"
        with fitz.open() as pdf:
            for p_idx in range(1, 6):
                page = pdf.new_page(width=595, height=842)
                page.insert_text((50, 35), "Project XYZ Standard Rev.A")
                body = (
                    f"9.{p_idx} Electrical Requirement Section {p_idx}\n"
                    + "All equipment shall comply with the project insulation specification and safety test procedures. " * 4
                )
                page.insert_textbox(fitz.Rect(50, 80, 540, 720), body)
                page.insert_text((50, 800), f"Confidential Engineering Spec Page {p_idx}")
            pdf.save(pdf_path)

        parsed = _parse_document(pdf_path, chunk_size=180, chunk_overlap=24, ocr_enabled=False)
        self.assertEqual(parsed.pages, 5)
        self.assertGreaterEqual(len(parsed.chunks), 5)
        joined = "\n".join(c["content"] for c in parsed.chunks)
        self.assertNotIn("Confidential Engineering Spec", joined)
        self.assertNotIn("Project XYZ Standard Rev.A", joined)
        self.assertIn("9.3 Electrical Requirement Section 3", joined)
        pages_seen = {c["page"] for c in parsed.chunks}
        self.assertEqual(pages_seen, {1, 2, 3, 4, 5})

    def test_scanned_pdf_low_text_page_triggers_ocr(self):
        img = Image.new("RGB", (500, 500), "white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        pdf_path = self.root / "scanned_with_pagenum.pdf"
        with fitz.open() as pdf:
            page = pdf.new_page(width=595, height=842)
            page.insert_image(fitz.Rect(40, 40, 540, 740), stream=img_bytes.getvalue())
            page.insert_text((290, 810), "2")
            pdf.save(pdf_path)

        with patch("local_ocr.recognize_page", return_value=[{"text": "Enclosure IP65", "confidence": 0.95}]) as ocr:
            parsed = _parse_document(pdf_path, ocr_enabled=True)
        ocr.assert_called_once()
        self.assertTrue(any(c.get("chunk_type") == "ocr" and "IP65" in c["content"] for c in parsed.chunks))

    def test_standalone_png_ocr_upload_and_image_preview(self):
        img = Image.new("RGB", (320, 120), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_pages = [{"page": 1, "lines": [{"text": "Control cabinet IP55", "confidence": 0.96, "box": [[10, 10], [200, 10], [200, 30], [10, 30]]}]}]
        with TestClient(create_web_app(self.root)) as client, \
             patch.object(ProjectStore, "_embed_safe", return_value=None), \
             patch("local_ocr.recognize_image", return_value=mock_pages) as rec_img:
            client.post("/api/projects", json={"id": "test", "name": "Test"}).raise_for_status()
            resp = client.post(
                "/api/projects/test/documents/upload",
                files={"files": ("cabinet.png", png_bytes, "image/png")},
            )
            doc = resp.raise_for_status().json()["uploaded"][0]
            rec_img.assert_called_once()
            self.assertEqual(doc["pages"], 1)
            self.assertEqual(doc["status"], "部分索引，需核对")
            self.assertEqual(doc["chunk_count"], 1)

            preview = client.get(f"/api/projects/test/documents/{doc['id']}/preview").raise_for_status().json()
            self.assertTrue(preview["can_preview_image"])
            self.assertEqual(preview["chunk_summary"]["ocr"], 1)
            self.assertIn("Control cabinet IP55", preview["chunks"][0]["content"])

            img_resp = client.get(f"/api/projects/test/documents/{doc['id']}/page-image/1")
            self.assertEqual(img_resp.status_code, 200)
            self.assertEqual(img_resp.headers["content-type"], "image/png")

    def test_macro_docx_and_html_parsing(self):
        macro_docx = self.root / "macro.docx"
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.ms-word.document.macroEnabled.main+xml"/>'
            '</Types>'
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>'
        )
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>'
            '<w:p><w:r><w:t>Main cabinet protection rating IP54.</w:t></w:r></w:p>'
            '<w:tbl>'
            '<w:tr><w:tc><w:p><w:r><w:t>Signal</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Protocol</w:t></w:r></w:p></w:tc></w:tr>'
            '<w:tr><w:tc><w:p><w:r><w:t>SCADA</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Modbus TCP</w:t></w:r></w:p></w:tc></w:tr>'
            '</w:tbl>'
            '<w:p><w:r><w:txbxContent><w:p><w:r><w:t>Textbox note: ambient 45C.</w:t></w:r></w:p></w:txbxContent></w:r></w:p>'
            '</w:body></w:document>'
        )
        with zipfile.ZipFile(macro_docx, "w") as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("word/document.xml", doc_xml)

        parsed_doc = _parse_document(macro_docx, ocr_enabled=False)
        doc_text = "\n".join(c["content"] for c in parsed_doc.chunks)
        self.assertIn("Main cabinet protection rating IP54.", doc_text)
        self.assertIn("Textbox note: ambient 45C.", doc_text)
        self.assertIn("Signal: SCADA | Protocol: Modbus TCP", doc_text)

        html_path = self.root / "spec.html"
        html_path.write_text(
            "<html><head><script>var secret='ignore';</script></head><body>"
            "<h1>Technical Spec</h1><p>General requirement: stainless steel enclosure.</p>"
            "<table><tr><th>Item</th><th>Value</th></tr>"
            "<tr><td>Row1</td><td>380V AC</td></tr>"
            "<tr><td>Row2</td><td>24V DC</td></tr>"
            "<tr><td>Row3</td><td>690V Insulation</td></tr>"
            "<tr><td>Row4</td><td>IP55 Outdoor</td></tr></table></body></html>",
            encoding="utf-8",
        )
        parsed_html = _parse_document(html_path, chunk_size=120, chunk_overlap=0, ocr_enabled=False)
        html_table_chunks = [c for c in parsed_html.chunks if c.get("chunk_type") == "table"]
        self.assertGreaterEqual(len(html_table_chunks), 2)
        for tc in html_table_chunks:
            self.assertIn("[HTML表格 1] Item | Value", tc["content"])
        joined_html = "\n".join(c["content"] for c in parsed_html.chunks)
        self.assertIn("stainless steel enclosure", joined_html)
        self.assertNotIn("secret='ignore'", joined_html)

    def test_batch_reparse_endpoint_recovers_pending_documents(self):
        with TestClient(create_web_app(self.root)) as client, \
             patch.object(ProjectStore, "_embed_safe", return_value=None):
            store = ProjectStore(self.root, "legacy2")
            p1 = self.root / "a.txt"
            p2 = self.root / "b.html"
            p1.write_text("Rated frequency 50 Hz.", encoding="utf-8")
            p2.write_text("<p>Ground resistance <= 4 ohm.</p>", encoding="utf-8")
            with sqlite3.connect(store.db_path) as conn:
                conn.execute(
                    "INSERT INTO documents(id,name,folder,file_type,pages,updated_at,status,indexed,file_path) VALUES "
                    "('d1','a.txt','f','TXT',0,'today','待重新解析',0,?),"
                    "('d2','b.html','f','HTML',0,'today','待重新解析',0,?)",
                    (str(p1), str(p2)),
                )
            conn.close()
            res = client.post("/api/projects/legacy2/documents/reparse-all", json={"only_pending": True}).raise_for_status().json()
            self.assertEqual(res["processed"], 2)
            self.assertEqual(res["indexed"], 2)
            docs = client.get("/api/projects/legacy2/documents").raise_for_status().json()["documents"]
            self.assertTrue(all(d["indexed"] == 1 and d["chunk_count"] >= 1 for d in docs))

    def test_real_offline_rapidocr_end_to_end(self):
        if not local_ocr.is_ocr_available():
            self.skipTest("Offline RapidOCR environment is not installed.")
        img = Image.new("RGB", (600, 200), "white")
        draw = ImageDraw.Draw(img)
        font = None
        for fpath in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
            if Path(fpath).exists():
                font = ImageFont.truetype(fpath, 42)
                break
        draw.text((40, 60), "IP54  18V DC", fill="black", font=font)
        png_path = self.root / "real_ocr.png"
        img.save(png_path)
        pdf_path = self.root / "real_ocr.pdf"
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=300)
            page.insert_image(fitz.Rect(20, 30, 560, 210), filename=str(png_path))
            pdf.save(pdf_path)

        parsed_img = _parse_document(png_path, ocr_enabled=True)
        parsed_pdf = _parse_document(pdf_path, ocr_enabled=True)
        self.assertIn("IP54", "\n".join(c["content"] for c in parsed_img.chunks))
        self.assertIn("IP54", "\n".join(c["content"] for c in parsed_pdf.chunks))


if __name__ == "__main__":
    unittest.main()
