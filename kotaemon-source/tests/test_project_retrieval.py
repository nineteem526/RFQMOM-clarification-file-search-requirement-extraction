from types import SimpleNamespace
import unittest

from pilot_runtime import prepare_runtime

prepare_runtime()

from kotaemon.base import Document
from project_retrieval import (
    query_variants, quick_candidates, nearby_context, is_meeting_source,
    exact_terms, exact_candidates,
)


class RetrievalTests(unittest.TestCase):
    def docs(self, values):
        return [Document(id_=f"v{i}-0", content=value, metadata={
            "version": f"v{i}", "file_name": f"fixture{i}.txt",
        }) for i, value in enumerate(values)]

    def test_exact_quantities_respect_numeric_unit_and_ac_dc_boundaries(self):
        docs = self.docs([
            "电压为 24 V DC，待批准。", "电压为240 V DC。",
            "电压为124 V DC。", "电压为24 kV DC。", "电压为24 V AC。",
            "电压为24.5 V DC。", "电压为24VDC，不得用于安全回路。",
        ])
        chosen, detail = exact_candidates("24VDC要求", docs)
        self.assertEqual(chosen, [docs[0], docs[-1]])
        self.assertEqual(detail["exact_matched_chunks"], 2)
        self.assertIn("不得", chosen[-1].content)
        area = self.docs(["走台100 kg/m²。", "走台100 kg/m。", "走台100 kg/m³。"])
        self.assertEqual(exact_candidates("100kg/m2", area)[0], [area[0]])

    def test_exact_models_standards_and_clauses_keep_versions(self):
        docs = self.docs([
            "采用 S7-1500；第 3.2.1 条，待批准，IEC 60204-1。",
            "不得采用 S7‑1500；第3.2.1条，旧要求。",
            "型号S7-15000，条款3.2.10，IEC 60204-11。",
        ])
        chosen, detail = exact_candidates("S7-1500，IEC60204-1，第3.2.1条", docs)
        self.assertEqual(chosen, docs[:2])
        self.assertEqual(detail["exact_matches_per_term"]["S7-1500"], 2)
        self.assertEqual(detail["exact_matches_per_term"]["IEC60204-1"], 1)
        self.assertEqual(detail["exact_matches_per_term"]["第3.2.1条"], 2)

    def test_exact_route_survives_low_vector_score_and_preserves_source(self):
        docs = self.docs(["采用IP54，待批准。", "采用IP540。"])
        chosen, details = quick_candidates(
            "IP54要求", docs,
            lambda *a, **k: [SimpleNamespace(id_=doc.id_, score=0.1) for doc in docs],
            lambda *a, **k: [], 0.62,
        )
        self.assertEqual(chosen, [docs[0]])
        self.assertEqual(details["exact_candidate_chunks"], 1)

    def test_literals_are_not_executed_and_candidates_are_bounded(self):
        docs = self.docs(["控制器IP54，待批准。"] * 50)
        chosen, detail = exact_candidates("IP54", docs)
        self.assertEqual(len(chosen), 24)
        self.assertEqual(detail["exact_omitted_chunks"], 26)
        self.assertEqual(exact_terms("钢构有哪些要求"), [])
        literal = self.docs(["原文 a+b[1]。", "原文 aaab1。"])
        self.assertEqual(exact_candidates('找“a+b[1]”', literal)[0], [literal[0]])

    def test_multiple_anchors_report_partial_matches_not_combined_requirements(self):
        docs = self.docs(["PLC S7-1500。", "电压24 V。", "S7-1500电压24 V，待批准。"])
        chosen, detail = exact_candidates("S7-1500 24V", docs)
        self.assertEqual({doc.id_ for doc in chosen}, {doc.id_ for doc in docs})
        self.assertEqual(detail["exact_matches_per_term"], {"24V": 2, "S7-1500": 2})
        self.assertEqual(chosen[0], docs[2])

    def test_si_prefixes_and_identifier_suffixes_are_not_conflated(self):
        docs = self.docs(["压力24 MPa。", "压力24 mPa。", "压力124 MPa。"])
        self.assertEqual(exact_candidates("24MPa要求", docs)[0], [docs[0]])
        self.assertEqual(exact_candidates("24mPa要求", docs)[0], [docs[1]])
        models = self.docs(["S7-1500.", "S7-1500-1。", "s7-1500T。"])
        self.assertEqual(exact_candidates("S7-1500", models)[0], [models[0]])

    def test_clause_header_is_found_without_invented_location(self):
        docs = self.docs(["3.2.1 控制柜应采用IP54。", "3.2.10 控制柜不含该系统。"])
        chosen, _ = exact_candidates("第3.2.1条写了什么", docs)
        self.assertEqual(chosen, docs[:1])
        self.assertEqual(chosen[0].content, "3.2.1 控制柜应采用IP54。")

    def test_exact_channel_does_not_mask_out_of_scope_retriever(self):
        docs = self.docs(["采用IP54。"])
        with self.assertRaisesRegex(ValueError, "边界"):
            quick_candidates("IP54", docs,
                             lambda *a, **k: [SimpleNamespace(id_="outside", score=0.9)],
                             lambda *a, **k: [], 0.62)

    def test_facets_expand_search_vocabulary_not_source_text(self):
        queries, subject = query_variants("钢构设计有哪些要求？按荷载、防腐、供货边界整理。")
        self.assertEqual(subject, "钢构")
        self.assertEqual(len(queries), 4)
        self.assertIn("钢平台 钢结构 供货范围", queries)

    def test_file_scoped_search_prioritizes_real_clauses_over_title(self):
        metadata = {"version": "v", "file_name": "电气控制技术要求.doc"}
        heading = Document(id_="g-0", content="电气控制", metadata=metadata)
        clause = Document(id_="g-1", content="所有紧急停止装置都必须采用双回路的方式，同时所有急停装置应独立接入安全PLC。",
                          metadata=metadata)
        hits = [SimpleNamespace(id_=doc.id_, score=score, metadata=doc.metadata)
                for doc, score in [(heading, 0.9), (clause, 0.6)]]
        chosen, detail = quick_candidates("电气控制有哪些要求？", [heading, clause],
                                         lambda *a, **k: hits, lambda *a, **k: [], 0.62)
        self.assertIn(clause, chosen)
        self.assertNotIn(heading, chosen)
        self.assertEqual(detail["filename_scoped_candidates"], 1)

    def test_adjacent_context_never_crosses_version_or_parse_generation(self):
        docs = [
            Document(id_=f"a-{i}", content=value, metadata={"version": "v"})
            for i, value in enumerate(["操作走台", "材质为铝合金，非通用钢平台。", "按100kg/m设计。"])
        ]
        other = Document(id_="b-0", content="另一次解析不应混入。", metadata={"version": "v"})
        result = nearby_context([docs[-1]], docs + [other])
        self.assertIn(docs[0], result)
        self.assertNotIn(other, result)

    def test_source_type_only_from_filename_or_human_label(self):
        self.assertFalse(is_meeting_source({"file_name": "技术协议.docx"}))
        self.assertFalse(is_meeting_source({"file_name": "thermometer.docx"}))
        self.assertTrue(is_meeting_source({"file_name": "会议纪要2026.docx"}))
        self.assertTrue(is_meeting_source({"version_label": "MOM 2026-10-19"}))


if __name__ == "__main__":
    unittest.main()
