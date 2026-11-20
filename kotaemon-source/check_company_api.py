"""Fixed-content company API probe; not a Kotaemon integration or RAG test."""

import json
import os
import re
import ssl
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener, HTTPSHandler


ENDPOINT = os.environ.get(
    "DUERR_API_APPROVED_ENDPOINT", "https://company-api.example.invalid"
).rstrip("/")
EXPECTED = "DUERR_API_TEST_OK"
MAX_RESPONSE_BYTES = 1_048_576


class ProbeError(ValueError):
    pass


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def make_request(environ):
    endpoint = environ.get("DUERR_API_ENDPOINT", ENDPOINT).rstrip("/")
    if endpoint != ENDPOINT:
        raise ProbeError("ENDPOINT_NOT_APPROVED")
    key = environ.get("DUERR_API_KEY", "")
    if not key or key.strip() != key or "\r" in key or "\n" in key:
        raise ProbeError("API_KEY_MISSING_OR_INVALID")
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        raise ProbeError("API_KEY_MISSING_OR_INVALID") from None
    version = environ.get("DUERR_API_VERSION", "2025-04-01-preview")
    deployment = environ.get("DUERR_MODEL", "gpt-5.2")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:-preview)?", version):
        raise ProbeError("INVALID_API_VERSION")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", deployment):
        raise ProbeError("INVALID_DEPLOYMENT")
    url = (
        f"{endpoint}/openai/deployments/{quote(deployment, safe='')}"
        f"/chat/completions?{urlencode({'api-version': version})}"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "Follow the user's test instruction."},
            {"role": "user", "content": f"Reply with exactly: {EXPECTED}"},
        ],
        "temperature": 1,
        "max_completion_tokens": 256,
    }
    return Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"api-key": key, "Content-Type": "application/json"},
        method="POST",
    )


def validate_response(body):
    try:
        result = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise ProbeError("INVALID_JSON_RESPONSE") from None
    if not isinstance(result, dict):
        raise ProbeError("INVALID_RESPONSE_SHAPE")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ProbeError("MISSING_OR_UNEXPECTED_CHOICES")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ProbeError("INCOMPLETE_OR_FILTERED_RESPONSE")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ProbeError("INVALID_MESSAGE")
    if message.get("refusal") or message.get("tool_calls") or message.get("function_call"):
        raise ProbeError("REFUSAL_OR_UNEXPECTED_TOOL_CALL")
    content = message.get("content")
    if not isinstance(content, str) or content.strip() != EXPECTED:
        raise ProbeError("EXACT_OUTPUT_CHECK_FAILED")


def run(environ, opener=None):
    request = make_request(environ)
    if opener is None:
        opener = build_opener(
            HTTPSHandler(context=ssl.create_default_context()), NoRedirects()
        )
    try:
        with opener.open(request, timeout=60) as response:
            if response.status != 200:
                raise ProbeError(f"HTTP_{response.status}")
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        status = error.code
        error.close()
        raise ProbeError(f"HTTP_{status}") from None
    except (URLError, TimeoutError, OSError):
        raise ProbeError("NETWORK_OR_TLS_FAILURE") from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProbeError("RESPONSE_TOO_LARGE")
    validate_response(body)


def main():
    try:
        run(os.environ)
    except ProbeError as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": "passed", "check": EXPECTED}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
