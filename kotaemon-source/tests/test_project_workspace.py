import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree

from pilot_runtime import prepare_runtime

prepare_runtime()

from openpyxl import Workbook, load_workbook
from docx import Document as WordDocument
from project_parser import parse_local
from project_workspace import ProjectWorkspace


class SelectAll:
    def invoke(self, messages):
        payload = json.loads(messages[1].content)
        if '"ids"' in messages[0].content:
            return SimpleNamespace(content=json.dumps({
                "ids": [entry["id"] for entry in payload["evidence"]]
            }))
        return SimpleNamespace(content=json.dumps({"excerpts": [
            {"id": entry["id"], "quote": entry["text"]} for entry in payload["evidence"]
        ]}))


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.workspace = ProjectWorkspace(self.base / "data", "project-a", self.source)

    def ingest_text(self, name="fixture.txt", text="虚构平台荷载3.0 kN/m²，已确认。"):
        path = self.source / name
        path.write_text(text, encoding="utf-8")
        scan = self.workspace.scan()
        report = self.workspace.ingest(scan["scan_id"], [name], True)
        self.assertNotIn("error", report[0], report)
        return path

    def test_confirmation_changed_scan_and_version_snapshots(self):
        path = self.source / "fixture.txt"
        path.write_text("old requirement", encoding="utf-8")
        scan = self.workspace.scan()
        with self.assertRaises(ValueError):
            self.workspace.ingest(scan["scan_id"], [path.name])
        path.write_text("new requirement", encoding="utf-8")
        result = self.workspace.ingest(scan["scan_id"], [path.name], True)
        self.assertIn("error", result[0])
        self.assertEqual(self.workspace.coverage(), [])
        self.ingest_text()
        original = self.workspace.coverage()[0]
        self.ingest_text(text="虚构平台荷载5.0 kN/m²，待批准。")
        self.assertEqual(len(self.workspace.coverage()), 2)
        with self.workspace.connection() as conn:
            snapshot = conn.execute(
                "SELECT snapshot FROM versions WHERE id=?", (original["id"],)
            ).fetchone()[0]
        self.assertIn("3.0", (self.workspace.root / snapshot).read_text(encoding="utf-8"))

    def test_status_counts_and_query_progress(self):
        self.assertIn("可检索版本 0 个", self.workspace.status_summary())
        self.ingest_text()
        status = self.workspace.status_summary()
        self.assertIn("可检索版本 1 个", status)
        self.assertIn("有效片段 1 条", status)
        self.assertNotIn("尚未扫描", status)
        updates = []
        identity, _, _, _ = self.workspace.ask(
            "荷载", chat=SelectAll(),
            progress=lambda value, desc: updates.append((value, desc)),
        )
        self.assertEqual(updates[-1][0], 1)
        self.assertEqual([value for value, _ in updates], sorted(value for value, _ in updates))
        self.assertTrue(any("公司AI核对" in desc for _, desc in updates))
        self.workspace.save_result(identity)
        self.assertIn("已保存结果 1 个", self.workspace.status_summary())

    def test_gradio_progress_must_not_be_boolean_tested(self):
        import gradio as gr

        path = self.source / "fixture.txt"
        path.write_text("虚构平台荷载3.0 kN/m²。", encoding="utf-8")
        scan = self.workspace.scan()
        report = self.workspace.ingest(
            scan["scan_id"], [path.name], True, progress=gr.Progress()
        )
        self.assertNotIn("error", report[0])
        _, documents, _, _ = self.workspace.ask(
            "平台荷载", chat=SelectAll(), progress=gr.Progress()
        )
        self.assertTrue(documents)

    def test_low_relevance_vector_results_do_not_become_evidence(self):
        from project_workspace import render_answer

        self.ingest_text()
        documents = self.workspace._index()[0]
        low_hit = SimpleNamespace(id_=documents[0].id_, score=0.58)

        class UnexpectedChat:
            def invoke(self, messages):
                raise AssertionError("Low-relevance evidence must not reach the model.")

        with patch.object(
            self.workspace, "_index",
            return_value=(
                documents, None,
                lambda *args, **kwargs: [low_hit],
                lambda *args, **kwargs: [],
            ),
        ):
            identity, selected, candidates, coverage = self.workspace.ask(
                "qzxv-unlikely-token", chat=UnexpectedChat()
            )
        self.assertEqual(selected, [])
        self.assertEqual(candidates, [])
        self.assertEqual(coverage[0]["below_vector_relevance_threshold"], 1)
        self.assertIn("当前资料中未找到相关证据", render_answer([], coverage))
        self.assertEqual(self.workspace.result(identity)["evidence"], [])

    def test_keyword_hits_are_kept_when_vector_score_is_low(self):
        self.ingest_text()
        documents = self.workspace._index()[0]
        low_hit = SimpleNamespace(id_=documents[0].id_, score=0.58)

        with patch.object(
            self.workspace, "_index",
            return_value=(
                documents, None,
                lambda *args, **kwargs: [low_hit],
                lambda *args, **kwargs: documents,
            ),
        ):
            _, selected, candidates, _ = self.workspace.ask(
                "qzxv-unlikely-token", chat=SelectAll()
            )
        self.assertTrue(selected)
        self.assertTrue(candidates)

    def test_project_isolation_persistence_and_wrong_binding(self):
        self.ingest_text()
        source_b = self.base / "source-b"
        source_b.mkdir()
        b = ProjectWorkspace(self.base / "data", "project-b", source_b)
        with self.assertRaisesRegex(ValueError, "没有可检索证据"):
            b.ask("荷载", chat=SelectAll())
        result_id, docs, _, _ = self.workspace.ask("荷载", chat=SelectAll())
        self.assertTrue(all(doc.metadata["project"] == "project-a" for doc in docs))
        with self.assertRaises(ValueError):
            b.result(result_id)
        reopened = ProjectWorkspace(self.base / "data", "project-a", self.source)
        self.assertEqual(len(reopened._index()[0]), 1)
        with self.assertRaises(ValueError):
            ProjectWorkspace(self.base / "data", "project-a", source_b)

    def test_model_identity_mismatch_rejected(self):
        with self.workspace.connection() as conn:
            conn.execute("UPDATE state SET value='different' WHERE key='embedding_model'")
        with self.assertRaisesRegex(ValueError, "嵌入模型"):
            ProjectWorkspace(self.base / "data", "project-a", self.source)

    def test_changed_duplicate_not_registered_and_alias_coverage(self):
        self.ingest_text()
        alias = self.source / "alias.txt"
        alias.write_bytes((self.source / "fixture.txt").read_bytes())
        scan = self.workspace.scan()
        alias.write_text("changed after scan", encoding="utf-8")
        report = self.workspace.ingest(scan["scan_id"], ["alias.txt"], True)
        self.assertIn("error", report[0])
        with self.workspace.connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0], 0)
        alias.write_bytes((self.source / "fixture.txt").read_bytes())
        scan = self.workspace.scan()
        report = self.workspace.ingest(scan["scan_id"], ["alias.txt"], True)
        self.assertNotIn("error", report[0])
        _, _, _, coverage = self.workspace.ask("荷载", chat=SelectAll())
        self.assertNotIn("alias.txt", coverage[-1]["not_indexed_as_current_bytes"])

    def test_snapshot_review_staleness_and_safe_export(self):
        self.ingest_text(text="=HYPERLINK(\"https://invalid.example\",\"虚构证据\")")
        result_id, _, _, _ = self.workspace.ask("虚构证据", chat=SelectAll())
        with self.assertRaises(ValueError):
            self.workspace.export(result_id)
        self.workspace.save_result(result_id)
        self.workspace.review(result_id, 1, "已确认")
        before = self.workspace.result(result_id)
        target = self.workspace.export(result_id)
        book = load_workbook(target, data_only=False)
        self.addCleanup(book.close)
        cell = book["证据与出处"]["C2"]
        self.assertTrue(cell.value.startswith("="))
        self.assertEqual(cell.data_type, "s")
        self.assertNotIn("审核状态", [cell.value for cell in book["证据与出处"][1]])
        version = self.workspace.coverage()[0]
        self.workspace.set_metadata(version["id"], "2026-10-15", "待批准")
        after = self.workspace.result(result_id)
        self.assertTrue(after["stale"])
        self.assertEqual(before["evidence"], after["evidence"])
        with self.assertRaises(ValueError):
            self.workspace.review(result_id, 1, "已确认")

    def test_duplicate_pair_and_failed_format_visible(self):
        self.ingest_text()
        (self.source / "duplicate.txt").write_bytes((self.source / "fixture.txt").read_bytes())
        (self.source / "old.doc").write_bytes(b"unsupported fixture")
        scan = self.workspace.scan()
        result = self.workspace.ingest(scan["scan_id"], ["duplicate.txt", "old.doc"], True)
        self.assertEqual(result[0]["status"], "字节相同，不重复索引")
        self.assertIn("error", result[1])
        self.assertEqual(len(self.workspace._index()[0]), 1)
        self.assertTrue(any(v["status"] == "failed" for v in self.workspace.coverage()))

    def test_summary_batch_failure_is_visible(self):
        self.ingest_text()

        class BrokenChat:
            def invoke(self, messages):
                raise ValueError("fixture service failure")

        result_id, docs, _, coverage = self.workspace.ask(
            "荷载", comprehensive=True, chat=BrokenChat()
        )
        self.assertEqual(docs, [])
        self.assertEqual(coverage[0]["status"], "未完成")
        self.assertEqual(coverage[0]["checked_chunks"], 0)
        self.assertTrue(self.workspace.result(result_id)["coverage"][0]["errors"])

    def test_topic_mode_checks_each_file_and_reports_unchecked_chunks(self):
        self.ingest_text(text="\n".join(f"虚构平台荷载要求第{i}行。" for i in range(20)))

        class SelectIds:
            def invoke(self, messages):
                payload = json.loads(messages[1].content)
                return SimpleNamespace(content=json.dumps({
                    "ids": [item["id"] for item in payload["evidence"]]
                }))

        _, documents, _, coverage = self.workspace.ask(
            "平台荷载", comprehensive=True, chat=SelectIds()
        )
        self.assertEqual(len(documents), 8)
        self.assertEqual(coverage[0]["local_screened_chunks"], 20)
        self.assertEqual(coverage[0]["checked_chunks"], 8)
        self.assertEqual(coverage[0]["not_sent_to_ai"], 12)

    def test_table_coordinates_formula_and_merged_warning(self):
        path = self.source / "table.xlsx"
        book = Workbook()
        sheet = book.active
        sheet.title = "电气"
        sheet.append(["对象", "数值", "单位"])
        sheet.append(["PLC-S7-1500", 24, "V DC"])
        sheet.append(["缓存公式", "=1+1", None])
        sheet.append(["未缓存公式", "=2+2", None])
        sheet.append(["错误缓存公式", "=1/0", None])
        sheet.merge_cells("A6:C6")
        sheet["A6"] = "合并说明"
        book.save(path)
        book.close()

        ElementTree.register_namespace(
            "", "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        )
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        rewritten = path.with_suffix(".rewritten.xlsx")
        with ZipFile(path) as source, ZipFile(rewritten, "w", ZIP_DEFLATED) as target:
            for entry in source.infolist():
                contents = source.read(entry.filename)
                if entry.filename == "xl/worksheets/sheet1.xml":
                    root = ElementTree.fromstring(contents)
                    for cell in root.iter(f"{namespace}c"):
                        coordinate = cell.get("r")
                        if coordinate in {"B3", "B5"}:
                            value = cell.find(f"{namespace}v")
                            if value is None:
                                value = ElementTree.SubElement(cell, f"{namespace}v")
                            if coordinate == "B3":
                                value.text = "2"
                            else:
                                cell.set("t", "e")
                                value.text = "#DIV/0!"
                    contents = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
                target.writestr(entry, contents)
        shutil.move(rewritten, path)
        original_bytes = path.read_bytes()

        chunks, warnings = parse_local(path)
        self.assertTrue(any("B2=24；C2=V DC" in chunk["text"] for chunk in chunks))
        self.assertTrue(any("A2:C2" in chunk["location"] for chunk in chunks))
        self.assertTrue(any(
            "B3=[公式（=1+1）；已保存缓存值=2；未重算，可能陈旧]" in chunk["text"]
            for chunk in chunks
        ))
        self.assertTrue(any(
            "B4=[公式（=2+2）；没有已保存缓存值；未重算，可能陈旧]" in chunk["text"]
            for chunk in chunks
        ))
        self.assertTrue(any(
            "B5=[公式（=1/0）；缓存错误:#DIV/0!；未重算，可能陈旧]" in chunk["text"]
            for chunk in chunks
        ))
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertTrue(any("读取到已保存缓存值" in warning for warning in warnings))
        self.assertTrue(any("没有已保存缓存值" in warning for warning in warnings))
        self.assertTrue(any("缓存为错误值" in warning for warning in warnings))
        self.assertTrue(any("合并" in warning for warning in warnings))

    def test_word_table_and_pdf_physical_page(self):
        word_path = self.source / "fixture.docx"
        word = WordDocument()
        word.add_paragraph("虚构室外控制柜IP65，待批准。")
        table = word.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "荷载"
        table.cell(0, 1).text = "3.0 kN/m²"
        word.save(word_path)
        chunks, warnings = parse_local(word_path)
        self.assertTrue(any("待批准" in entry["text"] for entry in chunks))
        self.assertTrue(any("3.0 kN/m²" in entry["text"] for entry in chunks))
        self.assertTrue(any("表1" in entry["location"] for entry in chunks))
        self.assertTrue(warnings)

        import fitz

        pdf_path = self.source / "fixture.pdf"
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((60, 60), "Synthetic PLC: 24 V DC, pending approval.")
        pdf.new_page()
        pdf.save(pdf_path)
        pdf.close()
        chunks, warnings = parse_local(pdf_path)
        self.assertIn("PDF物理第1页", chunks[0]["location"])
        self.assertIn("24 V DC", chunks[0]["text"])
        self.assertTrue(any("第2页" in item for item in warnings))

    def test_pair_selection_and_scan_change_stales_result(self):
        self.ingest_text()
        result_id, _, _, _ = self.workspace.ask("荷载", chat=SelectAll())
        self.assertFalse(self.workspace.result(result_id)["stale"])
        (self.source / "协议.docx").write_bytes(b"fixture word")
        (self.source / "协议.pdf").write_bytes(b"fixture pdf")
        scan = self.workspace.scan()
        self.assertTrue(self.workspace.result(result_id)["stale"])
        report = self.workspace.ingest(
            scan["scan_id"], ["协议.docx", "协议.pdf"], confirmed=True
        )
        self.assertEqual(len(report), 2)
        self.assertTrue(all("possible_duplicate_sources" in row for row in report))
        # Invalid fixtures fail parsing individually, not the whole batch's pair check.
        self.assertTrue(all("error" in row for row in report))

    def test_long_text_split_preserves_all_characters(self):
        from project_parser import chunk_blocks
        from local_embedding import local_engine

        original = "虚构荷载要求和适用条件；" * 150
        chunks = chunk_blocks([{"text": original, "location": "第1段"}])
        self.assertEqual("".join(chunk["text"] for chunk in chunks), original)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(
            len(local_engine().tokenizer.encode(chunk["text"]).ids) <= 480
            for chunk in chunks
        ))

    def test_mixed_formats_and_possible_pair_ingest_together(self):
        import fitz

        word = WordDocument()
        word.add_paragraph("虚构荷载要求，待批准。")
        word.save(self.source / "协议.docx")
        pdf = fitz.open()
        pdf.new_page().insert_text((60, 60), "Synthetic load requirement.")
        pdf.save(self.source / "协议.pdf")
        pdf.close()
        book = Workbook()
        book.active.append(["设备", "数值", "单位"])
        book.active.append(["PLC", 24, "V DC"])
        book.save(self.source / "参数.xlsx")
        book.close()
        scan = self.workspace.scan()
        report = self.workspace.ingest(
            scan["scan_id"], ["协议.docx", "协议.pdf", "参数.xlsx"], confirmed=True
        )
        self.assertEqual(len(report), 3)
        self.assertTrue(all("error" not in row for row in report), report)
        self.assertTrue(report[0]["possible_duplicate_sources"])
        self.assertEqual(len(self.workspace.coverage()), 3)
        pdf_version = next(row for row in self.workspace.coverage() if row["path"] == "协议.pdf")
        with self.workspace.connection() as conn:
            chunk_id = conn.execute(
                "SELECT id FROM chunks WHERE version=?", (pdf_version["id"],)
            ).fetchone()[0]
        self.assertTrue(self.workspace.pdf_evidence_preview(chunk_id).startswith(b"\x89PNG"))
        self.assertTrue(self.workspace.source_snapshot(pdf_version["id"]).is_file())
        with self.assertRaises(ValueError):
            self.workspace.source_snapshot("..\\outside.pdf")

    def test_failed_version_retry_keeps_previous_snapshot(self):
        self.ingest_text()
        source = self.source / "fixture.txt"
        scan = self.workspace.scan()
        with self.workspace.connection() as conn:
            version = conn.execute("SELECT id,snapshot FROM versions").fetchone()
            conn.execute("DELETE FROM chunks")
            conn.execute("UPDATE versions SET status='failed'")
        report = self.workspace.ingest(scan["scan_id"], [source.name], True)
        self.assertNotIn("error", report[0], report)
        self.assertEqual(len(self.workspace.coverage()), 1)
        self.assertTrue((self.workspace.root / version["snapshot"]).is_file())
        with self.workspace.connection() as conn:
            retried = conn.execute("SELECT snapshot FROM versions").fetchone()[0]
        self.assertNotEqual(version["snapshot"], retried)

    def test_reader_upgrade_keeps_old_chunks_and_failed_upgrade_keeps_usable_index(self):
        self.ingest_text()
        original_ids = {doc.id_ for doc in self.workspace._index()[0]}
        scan = self.workspace.scan()
        with self.workspace.connection() as conn:
            conn.execute("UPDATE versions SET reader_revision=0")
        report = self.workspace.ingest(scan["scan_id"], ["fixture.txt"], True)
        self.assertNotIn("error", report[0])
        current_ids = {doc.id_ for doc in self.workspace._index()[0]}
        self.assertTrue(original_ids.isdisjoint(current_ids))
        self.assertEqual(len(current_ids), len(original_ids))
        with self.workspace.connection() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM chunks").fetchone()[0], 2)
            conn.execute("UPDATE versions SET reader_revision=0")
        with patch("project_workspace.parse_local", side_effect=ValueError("fixture parse failure")):
            report = self.workspace.ingest(scan["scan_id"], ["fixture.txt"], True)
        self.assertIn("error", report[0])
        self.assertEqual({doc.id_ for doc in self.workspace._index()[0]}, current_ids)
        self.assertNotEqual(self.workspace.coverage()[0]["status"], "failed")

    def test_synthesis_saved_exported_and_backup_restored(self):
        from project_backup import backup, restore, verify

        self.ingest_text()
        with patch("project_answer.synthesize", return_value={"sections": [{
            "title": "荷载", "points": [{"text": "原文荷载条件。",
                                      "citations": []}],
        }]}):
            identity, _, _, _ = self.workspace.ask(
                "荷载", chat=SelectAll(), generate_answer=True
            )
        self.workspace.save_result(identity)
        self.assertIn("sections", json.loads(self.workspace.result(identity)["answer"]))
        exported = self.workspace.export(identity)
        book = load_workbook(exported)
        self.assertIn("回答", book.sheetnames)
        book.close()
        copied = backup(self.base / "data", "project-a")
        restored = restore(copied, self.base / "restore")
        self.assertEqual(verify(restored), restored)
        with self.assertRaises(ValueError):
            restore(copied, restored)
        damaged = restored / "project.sqlite3"
        with damaged.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "校验"):
            verify(restored)


if __name__ == "__main__":
    unittest.main()
