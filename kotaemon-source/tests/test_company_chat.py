import json
import os
import unittest
from unittest.mock import patch

from pilot_runtime import prepare_runtime

prepare_runtime()

import httpx
from openai import AzureOpenAI
from theflow.utils.modules import serialize

from company_chat import CompanyAzureChat, CompanyChatError, constrain_request
from check_company_api import ENDPOINT
from company_chat import configuration_status, probe_connection
from types import SimpleNamespace


def complete(text="虚构测试完成", finish_reason="stop"):
    return {
        "id": "fixture", "object": "chat.completion", "created": 1, "model": "fixture",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"total_tokens": 12, "prompt_tokens": 8, "completion_tokens": 4},
    }


class CompanyChatTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "DUERR_API_KEY": "fixture-key-not-a-credential",
            "DUERR_API_ENDPOINT": ENDPOINT,
            "DUERR_API_VERSION": "2025-04-01-preview",
            "DUERR_MODEL": "GPT-5.2",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_configuration_status_has_no_secret_or_network_request(self):
        with patch.object(CompanyAzureChat, "invoke") as invoke:
            status = configuration_status()
            self.assertIn("已就绪", status)
            self.assertIn("不代表", status)
            self.assertNotIn("fixture-key-not-a-credential", status)
            with patch.dict(os.environ, {"DUERR_API_KEY": ""}):
                self.assertIn("未就绪", configuration_status())
            with patch.dict(os.environ, {"DUERR_API_ENDPOINT": "https://unapproved.invalid"}):
                self.assertIn("未就绪", configuration_status())
            invoke.assert_not_called()

    def test_connection_probe_only_sends_fixed_content_and_requires_exact_response(self):
        with patch.object(CompanyAzureChat, "invoke",
                          return_value=SimpleNamespace(text="DUERR_API_TEST_OK")) as invoke:
            self.assertIn("成功", probe_connection())
            invoke.assert_called_once_with("Reply with exactly: DUERR_API_TEST_OK")
        with patch.object(CompanyAzureChat, "invoke",
                          return_value=SimpleNamespace(text="other output")):
            with self.assertRaises(CompanyChatError):
                probe_connection()
        with patch.object(CompanyAzureChat, "invoke", side_effect=CompanyChatError("HTTP 401")):
            with self.assertRaisesRegex(CompanyChatError, "401"):
                probe_connection()

    def test_azure_sdk_mapping_and_secret_not_in_spec(self):
        seen = []

        def respond(request):
            seen.append(request)
            return httpx.Response(200, json=complete())

        chat = CompanyAzureChat()
        spec = serialize(chat)
        self.assertNotIn("fixture-key-not-a-credential", json.dumps(spec))
        http_client = httpx.Client(
            transport=httpx.MockTransport(respond),
            event_hooks={"request": [constrain_request]},
        )
        with patch("company_chat.company_http_client", return_value=http_client):
            result = chat.invoke("虚构问题")
        self.assertEqual(result.text, "虚构测试完成")
        request = seen[0]
        self.assertEqual(request.url.path, "/openai/deployments/GPT-5.2/chat/completions")
        self.assertEqual(request.url.params["api-version"], "2025-04-01-preview")
        self.assertEqual(request.headers["api-key"], "fixture-key-not-a-credential")
        self.assertNotIn("authorization", request.headers)
        payload = json.loads(request.content)
        self.assertEqual(payload["temperature"], 1)
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertFalse(payload["stream"])
        self.assertNotIn("max_tokens", payload)

    def test_truncated_refused_or_invalid_response_rejected(self):
        invalid = [complete("", "stop"), complete("partial", "length"), {},
                   complete("blocked", "content_filter")]
        refused = complete()
        refused["choices"][0]["message"]["refusal"] = "no"
        invalid.append(refused)
        invalid_usage = complete()
        invalid_usage["usage"] = None
        invalid.append(invalid_usage)
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(CompanyChatError):
                    CompanyAzureChat().prepare_output(response)

    def test_no_network_for_missing_key_or_unapproved_endpoint(self):
        for values in ({"DUERR_API_KEY": ""},
                       {"DUERR_API_ENDPOINT": "https://example.com"}):
            with patch.dict(os.environ, values):
                with patch.object(CompanyAzureChat, "prepare_client") as client:
                    with self.assertRaises(CompanyChatError):
                        CompanyAzureChat().invoke("not sent")
                    client.assert_not_called()

    def test_failure_no_retry_no_fallback_or_error_body(self):
        seen = []

        def respond(request):
            seen.append(request)
            return httpx.Response(401, json={"error": {"message": "sensitive fixture"}})

        client = AzureOpenAI(
            azure_endpoint=ENDPOINT, api_key=os.environ["DUERR_API_KEY"],
            api_version="2025-04-01-preview", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        with patch.object(CompanyAzureChat, "prepare_client", return_value=client):
            with self.assertRaisesRegex(CompanyChatError, r"HTTP 401") as error:
                list(CompanyAzureChat().stream("fixture"))
        self.assertNotIn("sensitive", str(error.exception))
        self.assertEqual(len(seen), 1)

    def test_unsupported_parameters_fail(self):
        with self.assertRaises(CompanyChatError):
            CompanyAzureChat().invoke("fixture", tools=[])


if __name__ == "__main__":
    unittest.main()
