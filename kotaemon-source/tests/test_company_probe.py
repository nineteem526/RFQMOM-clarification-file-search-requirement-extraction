import io
import json
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

from check_company_api import (
    ENDPOINT,
    EXPECTED,
    NoRedirects,
    ProbeError,
    make_request,
    run,
    validate_response,
)


def complete(content=EXPECTED, finish_reason="stop"):
    return json.dumps({
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content},
        }]
    }).encode()


class CompanyProbeTests(unittest.TestCase):
    def test_explicit_azure_mapping(self):
        request = make_request({
            "DUERR_API_KEY": "fixture-key-not-a-credential",
            "DUERR_MODEL": "GPT-5.2",
        })
        self.assertEqual(
            request.full_url,
            ENDPOINT + "/openai/deployments/GPT-5.2/chat/completions"
            "?api-version=2025-04-01-preview",
        )
        self.assertEqual(request.get_header("Api-key"), "fixture-key-not-a-credential")
        self.assertIsNone(request.get_header("Authorization"))
        payload = json.loads(request.data)
        self.assertEqual(payload["max_completion_tokens"], 256)
        self.assertEqual(payload["temperature"], 1)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("response_format", payload)

    def test_invalid_configuration_never_sends(self):
        for environ in (
            {},
            {"DUERR_API_KEY": "x", "DUERR_API_ENDPOINT": "https://example.com"},
            {"DUERR_API_KEY": "x", "DUERR_MODEL": "../external"},
            {"DUERR_API_KEY": "x", "DUERR_API_VERSION": "not-a-version"},
            {"DUERR_API_KEY": "x\nsecret"},
        ):
            with self.subTest(environ=list(environ)):
                opener = Mock()
                with self.assertRaises(ProbeError):
                    run(environ, opener)
                opener.open.assert_not_called()

    def test_complete_exact_response(self):
        validate_response(complete())

    def test_bad_responses_fail_closed(self):
        responses = [
            b"not json", b"null", b"[]", b"{}", b'{"choices":[]}',
            complete(""), complete("unverified answer"),
            complete(finish_reason="length"), complete(finish_reason="content_filter"),
            json.dumps({"choices": [{"finish_reason": "stop", "message": {
                "content": EXPECTED, "refusal": "refused",
            }}]}).encode(),
        ]
        for body in responses:
            with self.subTest(body=body):
                with self.assertRaises(ProbeError):
                    validate_response(body)

    def test_failure_no_retry_no_error_body_disclosure(self):
        for error in (
            HTTPError(ENDPOINT, 401, "sensitive-response", {}, io.BytesIO(b"secret")),
            URLError("sensitive-network-details"),
        ):
            opener = Mock()
            opener.open.side_effect = error
            with self.assertRaises(ProbeError) as caught:
                run({"DUERR_API_KEY": "fixture-key-not-a-credential"}, opener)
            self.assertNotIn("sensitive", str(caught.exception))
            self.assertEqual(opener.open.call_count, 1)

    def test_redirect_not_followed(self):
        self.assertIsNone(NoRedirects().redirect_request(
            None, None, 302, "redirect", {}, "https://example.com"
        ))


if __name__ == "__main__":
    unittest.main()
