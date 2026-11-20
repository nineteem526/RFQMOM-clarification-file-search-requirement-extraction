import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pilot_runtime import prepare_runtime

prepare_runtime()

from kotaemon.base import Document
from project_answer import synthesize, render_synthesis, prepare_presentation


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document(content="虚构平台设计荷载5.0 kN/m²，待业主批准，不替代旧要求。",
                            id_="fixture", metadata={})
        self.answer = {"sections": [{"title": "荷载", "points": [{
            "text": "平台拟采用5.0 kN/m²，待业主批准，不替代旧要求。",
            "citations": [{"id": "fixture", "quote": self.doc.content}],
        }]}]}

    def invoke(self, answer):
        chat = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(answer)))
        return synthesize("荷载", [self.doc], chat)

    def test_valid_answer_references_actual_evidence(self):
        result = self.invoke(self.answer)
        self.assertIn("[1]", render_synthesis(result, [self.doc]))
        self.assertIn("待业主批准", render_synthesis(result, [self.doc]))

    def test_invalid_source_quote_number_and_empty_citations_rejected(self):
        for field, value in [("id", "external"), ("quote", "虚构已批准")]:
            answer = copy.deepcopy(self.answer)
            answer["sections"][0]["points"][0]["citations"][0][field] = value
            with self.assertRaises(ValueError):
                self.invoke(answer)
        answer = copy.deepcopy(self.answer)
        answer["sections"][0]["points"][0]["text"] = "采用9.9 kN/m²。"
        with self.assertRaisesRegex(ValueError, "数字"):
            self.invoke(answer)
        answer["sections"][0]["points"][0]["citations"] = []
        with self.assertRaises(ValueError):
            self.invoke(answer)

    def test_malformed_or_success_shaped_output_rejected(self):
        for answer in [{"sections": []}, {"answer": "ok"}, None, "ok"]:
            with self.assertRaises(ValueError):
                self.invoke(answer)

    def test_pending_approval_cannot_be_written_as_final_requirement(self):
        answer = copy.deepcopy(self.answer)
        answer["sections"][0]["points"][0]["text"] = "平台荷载确定采用5.0 kN/m²。"
        with self.assertRaisesRegex(ValueError, "待定条件"):
            self.invoke(answer)

    def test_number_must_be_in_actual_quote_not_another_part_of_chunk(self):
        self.doc.content = "额定功率50 kW；平台荷载5.0 kN/m²，待业主批准。"
        answer = copy.deepcopy(self.answer)
        answer["sections"][0]["points"][0]["citations"][0]["quote"] = "额定功率50 kW"
        with self.assertRaisesRegex(ValueError, "数字"):
            self.invoke(answer)

    def test_substring_number_does_not_validate_a_different_value(self):
        self.doc.content = "平台荷载50 kN/m²，待业主批准。"
        with self.assertRaisesRegex(ValueError, "数字"):
            self.invoke(self.answer_with_quote(self.doc.content))

    def answer_with_quote(self, quote):
        answer = copy.deepcopy(self.answer)
        answer["sections"][0]["points"][0]["citations"][0]["quote"] = quote
        return answer

    def test_protocol_must_not_be_presented_as_latest_mom(self):
        self.doc.metadata["file_name"] = "设备技术协议.docx"
        answer = copy.deepcopy(self.answer)
        answer["sections"][0]["title"] = "最新MOM要求"
        with self.assertRaisesRegex(ValueError, "非会议来源"):
            self.invoke(answer)

    def test_one_reported_correction_uses_original_evidence_and_revalidates(self):
        rejected = copy.deepcopy(self.answer)
        rejected["sections"][0]["points"][0]["text"] = "荷载9.9 kN/m²，待批准。"
        messages = []
        notices = []
        def invoke(request):
            messages.append(request)
            return SimpleNamespace(content=json.dumps(rejected if len(messages) == 1 else self.answer))
        result = synthesize("荷载", [self.doc], SimpleNamespace(invoke=invoke),
                            on_correction=notices.append)
        self.assertEqual(result, self.answer)
        self.assertEqual(len(messages), 2)
        self.assertEqual(len(notices), 1)
        payload = json.loads(messages[1][1].content)
        self.assertEqual(payload["evidence"][0]["text"], self.doc.content)
        self.assertIn("validation_error", payload)

    def test_id_only_citation_is_filled_with_exact_original_locally(self):
        answer = copy.deepcopy(self.answer)
        del answer["sections"][0]["points"][0]["citations"][0]["quote"]
        result = self.invoke(answer)
        self.assertEqual(result["sections"][0]["points"][0]["citations"][0]["quote"],
                         self.doc.content)

    def test_short_answer_needs_no_extra_model_request(self):
        chat = SimpleNamespace(invoke=lambda _: self.fail("Short answer must not be rewritten"))
        self.assertIs(prepare_presentation("荷载", self.answer, [self.doc], chat), self.answer)

    def test_long_answer_has_bounded_overview_and_unchanged_details(self):
        detailed = copy.deepcopy(self.answer)
        detailed["sections"][0]["points"] *= 15
        original = copy.deepcopy(detailed)
        result = prepare_presentation(
            "荷载", detailed, [self.doc],
            SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(self.answer))),
        )
        self.assertEqual(detailed, original)
        self.assertEqual(len(result["sections"][0]["points"]), 15)
        self.assertEqual(result["overview"], self.answer["sections"])
        rendered = render_synthesis(result, [self.doc])
        self.assertIn("重点概览", rendered)
        self.assertIn("<details><summary>展开完整条款", rendered)
        self.assertIn("待业主批准", rendered)
        self.assertIn("[1](/context/fixture)", rendered)

    def test_overview_limits_are_enforced_not_just_prompted(self):
        for count, length in [(9, 0), (1, 201)]:
            answer = copy.deepcopy(self.answer)
            answer["sections"][0]["points"] *= count
            if length:
                answer["sections"][0]["points"][0]["text"] += "条件" * length
            with self.subTest(count=count, length=length), self.assertRaisesRegex(ValueError, "概览"):
                synthesize("荷载", [self.doc], SimpleNamespace(
                    invoke=lambda _: SimpleNamespace(content=json.dumps(answer))), concise=True)

    def test_overview_cannot_introduce_an_uncited_source(self):
        detailed = copy.deepcopy(self.answer)
        detailed["sections"][0]["points"] *= 9
        unrelated = Document(id_="other", content=self.doc.content, metadata={})
        answer = copy.deepcopy(self.answer)
        answer["sections"][0]["points"][0]["citations"][0]["id"] = "other"
        with self.assertRaisesRegex(ValueError, "以外"):
            prepare_presentation(
                "荷载", detailed, [self.doc, unrelated],
                SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(answer))),
            )


if __name__ == "__main__":
    unittest.main()
