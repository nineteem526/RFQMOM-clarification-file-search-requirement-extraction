import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pilot_runtime import prepare_runtime

prepare_runtime()

from synthetic_qa import (
    SyntheticKnowledge, evidence_html, fixture_documents, validate_excerpts,
)


class SyntheticQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.knowledge = SyntheticKnowledge(cls.root)

    def test_table_structure_and_repeatability(self):
        documents = fixture_documents(self.root)
        self.assertEqual([d.id_ for d in documents],
                         [d.id_ for d in self.knowledge.documents])
        self.assertEqual(len(documents), 7)
        row = next(d for d in documents if "PLC-S7-1500" in d.content)
        self.assertIn("数值：24；单位：V DC", row.content)
        self.assertIn("A3:G3", row.metadata["location"])
        self.assertIn("表头 A1:G1", row.metadata["location"])

    def test_real_fts_and_vector_retrieval(self):
        for query, terms in [
            ("钢结构平台荷载", ["3.0 kN/m²", "5.0 kN/m²", "待业主批准"]),
            ("PLC-S7-1500", ["PLC-S7-1500", "V DC"]),
            ("IF-002", ["IF-002", "不含现场接线"]),
        ]:
            with self.subTest(query=query):
                text = "\n".join(d.content for d in self.knowledge.retrieve(query))
                for term in terms:
                    self.assertIn(term, text)
        matches = self.knowledge.doc_store.query("PLC-S7-1500")
        self.assertIn("PLC-S7-1500", matches[0].content)
        self.assertEqual(
            self.knowledge.doc_store.query("PLC-S7-1500", doc_ids=[]), []
        )

    def test_no_silent_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_documents(root)
            path = root / "虚构RFQ.txt"
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\nchanged")
            with self.assertRaisesRegex(ValueError, "content changed"):
                fixture_documents(root)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("changed"))

    def test_exact_citations_only(self):
        documents = self.knowledge.documents
        doc = documents[0]
        accepted = {"excerpts": [{"id": doc.id_, "quote": doc.content}]}
        self.assertEqual(validate_excerpts(json.dumps(accepted), documents), [doc])
        for data in [
            {"excerpts": [{"id": "unknown", "quote": doc.content}]},
            {"excerpts": [{"id": doc.id_, "quote": "3.0 kN/m²"}]},
            {"excerpts": accepted["excerpts"] * 2},
            {"excerpts": [], "answer": "unsupported"},
            {"excerpts": None},
        ]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    validate_excerpts(json.dumps(data), documents)
        with self.assertRaises(ValueError):
            validate_excerpts("truncated {", documents)
        self.assertEqual(validate_excerpts('{"excerpts":[]}', documents), [])

    def test_answer_uses_retrieved_documents_only(self):
        knowledge = self.knowledge

        class FakeChat:
            def invoke(self, messages):
                self.payload = json.loads(messages[1].content)
                first = self.payload["retrieved_evidence"][0]
                return SimpleNamespace(content=json.dumps({
                    "excerpts": [{"id": first["id"], "quote": first["text"]}]
                }))

        chat = FakeChat()
        selected, retrieved = knowledge.answer("PLC-S7-1500电压", chat=chat)
        self.assertEqual(set(chat.payload), {"question", "retrieved_evidence"})
        self.assertEqual(
            {entry["id"] for entry in chat.payload["retrieved_evidence"]},
            {doc.id_ for doc in retrieved},
        )
        self.assertTrue(selected)

    def test_positions_displayed_and_invalid_query_rejected(self):
        html = evidence_html(self.knowledge.documents)
        self.assertIn("SHA256", html)
        self.assertIn("A3:G3", html)
        for query in ("", "  ", "x" * 401):
            with self.assertRaises(ValueError):
                self.knowledge.retrieve(query)


if __name__ == "__main__":
    unittest.main()
