import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import AzureOpenAI

from samples import SAMPLES
from workbench import (
    MAX_CONTEXT_CHARS, Store, WorkbenchError, completion_text,
    connection_test, context_payload, extract, make_client, split_text,
    validate_endpoint, validate_result,
)

TEST_ENDPOINT = "https://company-api.example.test"
COMPANY_ENDPOINT = TEST_ENDPOINT


def response(text, reason="stop"):
    return SimpleNamespace(choices=[
        SimpleNamespace(
            finish_reason=reason,
            message=SimpleNamespace(content=text, refusal=None),
        )
    ])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "test.sqlite3"
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.folder.cleanup()

    def seed(self):
        for sample in SAMPLES:
            self.store.import_sample(sample)

    def test_empty_and_whitespace(self):
        self.assertEqual(self.store.documents(), [])
        self.assertEqual(self.store.search("钢构"), [])
        self.assertEqual(self.store.search(" \n"), [])
        with self.assertRaises(WorkbenchError):
            split_text(" \n ")

    def test_single_document_idempotency_and_persistence(self):
        self.assertTrue(self.store.import_sample(SAMPLES[0]))
        before = self.store.evidence()
        self.assertFalse(self.store.import_sample(SAMPLES[0]))
        self.assertEqual(self.store.evidence(), before)
        reopened = Store(self.path)
        try:
            self.assertEqual(reopened.evidence(), before)
        finally:
            reopened.close()

    def test_versions_preserved_and_stage_filtered(self):
        self.seed()
        self.assertEqual(len(self.store.documents()), 6)
        te = self.store.search("钢构", ("TE",))
        self.assertEqual({item.version for item in te}, {"R1", "R2"})
        self.assertTrue(all(item.stage == "TE" for item in te))
        self.assertTrue(all(item.stage == "DF" for item in self.store.search("钢构", ("DF",))))

    def test_expected_exact_search_sources_and_aliases(self):
        for sample in SAMPLES:
            if sample.initial:
                self.store.import_sample(sample)
        items = self.store.search("钢构")
        texts = [item.text for item in items]
        self.assertTrue(any("5 kN/m²" in text for text in texts))
        self.assertTrue(any("3 kN/m²" in text for text in texts))
        self.assertTrue(any("客户负责基础施工" in text for text in texts))
        self.assertEqual(
            {item.document_key for item in items},
            {"DEMO_RFQ", "DEMO_B12", "DEMO_CLARIFICATION"},
        )
        self.assertGreater(
            len(self.store.search("电气控制")), len(self.store.search("电气控制", expand=False))
        )
        self.assertEqual(self.store.search("不存在的词"), [])
        self.assertTrue(self.store.search("IP54"))
        self.assertTrue(self.store.search("PLC"))
        self.assertEqual(self.store.search("'; DROP TABLE documents; --"), [])
        self.assertEqual(len(self.store.documents()), 3)

    def test_locator_points_to_exact_original_lines(self):
        self.seed()
        for item in self.store.evidence():
            sample = next(s for s in SAMPLES if
                          (s.document_key, s.version) == (item.document_key, item.version))
            start, end = item.locator.removeprefix("第 ").removesuffix(" 行").split("–")
            self.assertEqual(
                item.text, "\n".join(sample.text.splitlines()[int(start) - 1:int(end)])
            )
        self.assertEqual(split_text("single"), [("第 1–1 行", "single")])

    def test_reject_non_demo_without_writing(self):
        with self.assertRaises(WorkbenchError):
            self.store.import_sample(replace(SAMPLES[0], text="customer data"))
        self.assertEqual(self.store.documents(), [])

    def test_transaction_failure_does_not_leave_partial_document(self):
        self.store.connection.execute("""
            CREATE TRIGGER deny_chunk BEFORE INSERT ON chunks BEGIN
            SELECT RAISE(ABORT, 'fixture failure'); END;
        """)
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.import_sample(SAMPLES[0])
        self.assertEqual(self.store.documents(), [])
        self.assertEqual(self.store.evidence(), [])

    def test_corrupt_index_blocks_api(self):
        self.seed()
        self.store.verify_demo()
        with self.store.connection:
            self.store.connection.execute("UPDATE chunks SET text='changed' WHERE id=1")
        with self.assertRaises(WorkbenchError):
            self.store.verify_demo()
        with self.assertRaises(WorkbenchError):
            context_payload([self.store.evidence()[0]])

    def test_corrupt_metadata_blocks_api(self):
        self.seed()
        with self.store.connection:
            self.store.connection.execute("UPDATE documents SET filename='private name' WHERE id=1")
        with self.assertRaises(WorkbenchError):
            self.store.verify_demo()
        with self.assertRaises(WorkbenchError):
            context_payload([self.store.evidence()[0]])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.folder.name) / "test.sqlite3")
        for sample in SAMPLES:
            self.store.import_sample(sample)
        self.evidence = self.store.search("荷载")

    def tearDown(self):
        self.store.close()
        self.folder.cleanup()

    def valid(self):
        item = self.evidence[0]
        return {
            "items": [{
                "type": "requirement", "description": "虚构平台要求",
                "evidence": [{"source_id": item.source_id, "quote": item.text}],
            }],
            "limitations": ["没有标准原文，不能提供条款建议。"],
        }

    def test_json_and_quote_validation(self):
        valid = self.valid()
        self.assertEqual(validate_result(json.dumps(valid), self.evidence), valid)
        for text in ("not json", "[]", '{"items":[]}', '{"items":[],"limitations":[]}'):
            with self.subTest(text=text), self.assertRaises(WorkbenchError):
                validate_result(text, self.evidence)
        for quote, source_id in [("invented requirement", self.evidence[0].source_id), ("", 1),
                                 (self.evidence[0].text, 999), (self.evidence[0].text, True)]:
            data = self.valid()
            data["items"][0]["evidence"] = [{"source_id": source_id, "quote": quote}]
            with self.subTest(quote=quote, source_id=source_id), self.assertRaises(WorkbenchError):
                validate_result(json.dumps(data), self.evidence)

    def test_conflict_requires_distinct_documents(self):
        data = self.valid()
        data["items"][0]["type"] = "conflict"
        with self.assertRaises(WorkbenchError):
            validate_result(json.dumps(data), self.evidence)
        other = next(item for item in self.evidence
                     if item.document_key != self.evidence[0].document_key)
        data["items"][0]["evidence"].append({"source_id": other.source_id, "quote": other.text})
        self.assertEqual(validate_result(json.dumps(data), self.evidence), data)

    def test_empty_items_require_limitations(self):
        data = {"items": [], "limitations": ["所选范围没有可提取要求。"]}
        self.assertEqual(validate_result(json.dumps(data), self.evidence), data)

    def test_standard_advice_is_not_accepted(self):
        data = self.valid()
        data["items"][0]["type"] = "standard_advice"
        with self.assertRaises(WorkbenchError):
            validate_result(json.dumps(data), self.evidence)

    def test_payload_rejects_duplicates_empty_and_limit(self):
        with self.assertRaises(WorkbenchError):
            context_payload([])
        with self.assertRaises(WorkbenchError):
            context_payload([self.evidence[0], self.evidence[0]])
        payload = context_payload(self.evidence)
        self.assertLess(len(payload), MAX_CONTEXT_CHARS)
        with patch("workbench.MAX_CONTEXT_CHARS", len(payload) - 1):
            with self.assertRaises(WorkbenchError):
                context_payload(self.evidence)
        with patch("workbench.MAX_CONTEXT_CHARS", len(payload)):
            self.assertEqual(context_payload(self.evidence), payload)

    def test_company_only_endpoint(self):
        allowed_host = patch.dict("os.environ", {"DUERR_API_ALLOWED_HOST": "company-api.example.test"})
        allowed_host.start()
        self.addCleanup(allowed_host.stop)
        self.assertEqual(validate_endpoint(TEST_ENDPOINT + "/"), TEST_ENDPOINT)
        for endpoint in (
            "https://example.com", TEST_ENDPOINT.replace("https:", "http:"),
            TEST_ENDPOINT + "/v1", TEST_ENDPOINT + "?x=1",
            TEST_ENDPOINT + "#fragment",
            COMPANY_ENDPOINT.replace("https://", "https://user:password@"),
            TEST_ENDPOINT + ":444",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(WorkbenchError):
                validate_endpoint(endpoint)

    def test_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(WorkbenchError):
                make_client()

    def test_completion_failures_and_precise_smoke_output(self):
        for value, reason in (("partial", "length"), ("", "stop"), (None, "stop"),
                              ("blocked", "content_filter")):
            with self.subTest(value=value, reason=reason), self.assertRaises(WorkbenchError):
                completion_text(response(value, reason))
        with self.assertRaises(WorkbenchError):
            completion_text(SimpleNamespace(choices=[]))
        client = MagicMock()
        client.chat.completions.create.return_value = response("DUERR_API_TEST_OK")
        connection_test(client)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        client.chat.completions.create.return_value = response("almost")
        with self.assertRaises(WorkbenchError):
            connection_test(client)

    def test_extract_mocked_sdk_azure_request_shape(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "demo", "object": "chat.completion", "created": 0, "model": "gpt-5.2",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(self.valid())}}],
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            with AzureOpenAI(
                azure_endpoint=TEST_ENDPOINT, api_version="2025-04-01-preview",
                api_key="fake-fixture-not-a-real-key", max_retries=0, http_client=http_client,
            ) as client:
                with patch.dict("os.environ", {"DUERR_MODEL": "gpt-5.2"}):
                    self.assertEqual(extract(client, self.evidence), self.valid())
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.url.path, "/openai/deployments/gpt-5.2/chat/completions")
        self.assertEqual(request.url.params["api-version"], "2025-04-01-preview")
        self.assertEqual(request.headers["api-key"], "fake-fixture-not-a-real-key")
        self.assertNotIn("authorization", request.headers)
        body = json.loads(request.content)
        self.assertEqual(body["temperature"], 1)
        self.assertEqual(body["max_completion_tokens"], 6000)
        self.assertNotIn("response_format", body)
        self.assertIn("没有标准原文", body["messages"][0]["content"])
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        self.assertEqual([item["text"] for item in payload],
                         [item.text for item in self.evidence])


if __name__ == "__main__":
    unittest.main()
