"""Kotaemon Azure extension with runtime-only credentials and complete responses."""

import os
import re
from urllib.parse import urlsplit

import httpx
from openai import APIConnectionError, APIStatusError, AzureOpenAI, OpenAIError

from kotaemon.base import Param
from kotaemon.llms import AzureChatOpenAI

from check_company_api import ENDPOINT


class CompanyChatError(ValueError):
    pass


def constrain_request(request):
    if (request.url.scheme != "https"
            or request.url.host != urlsplit(ENDPOINT).hostname
            or not request.url.path.startswith("/openai/deployments/")
            or not request.url.path.endswith("/chat/completions")
            or request.method != "POST"):
        raise CompanyChatError("Blocked non-company-chat request.")
    request.headers.pop("authorization", None)


def company_http_client():
    return httpx.Client(
        verify=True, follow_redirects=False, timeout=60.0,
        event_hooks={"request": [constrain_request]},
    )


def company_config():
    endpoint = os.environ.get("DUERR_API_ENDPOINT", ENDPOINT).rstrip("/")
    if endpoint != ENDPOINT or urlsplit(endpoint).scheme != "https":
        raise CompanyChatError("Company endpoint is not approved.")
    version = os.environ.get("DUERR_API_VERSION", "2025-04-01-preview")
    deployment = os.environ.get("DUERR_MODEL", "gpt-5.2")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:-preview)?", version):
        raise CompanyChatError("Invalid company API version.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", deployment):
        raise CompanyChatError("Invalid company deployment name.")
    key = os.environ.get("DUERR_API_KEY", "")
    if not key or key != key.strip() or not key.isascii() or "\r" in key or "\n" in key:
        raise CompanyChatError("DUERR_API_KEY is missing or invalid.")
    return endpoint, version, deployment, key


def configuration_status():
    """Report configuration presence, never credentials or implied connectivity."""
    try:
        company_config()
    except CompanyChatError:
        return (
            "公司AI配置未就绪：请在启动服务的环境中配置有效的 DUERR_API_KEY，"
            "并检查公司端点、部署和API版本。无需在网页输入密钥。"
        )
    return (
        "公司AI配置已就绪：密钥来自启动进程继承的 DUERR_API_KEY 环境变量，"
        "不会显示或写入项目文件。配置存在不代表当前连接成功；可点击下方连接检查。"
    )


def probe_connection():
    response = CompanyAzureChat(completion_budget=256).invoke(
        "Reply with exactly: DUERR_API_TEST_OK"
    )
    if response.text.strip() != "DUERR_API_TEST_OK":
        raise CompanyChatError("公司AI连接检查响应不符，不能报告连接成功。")
    return "公司AI连接检查成功；仅发送固定测试文字，未发送项目资料。此状态仅代表本次检查。"


class CompanyAzureChat(AzureChatOpenAI):
    # Kotaemon persists the model spec. Only this sentinel may be persisted.
    api_key: str = Param("runtime-environment-only")
    azure_endpoint: str = Param(ENDPOINT)
    azure_deployment: str = Param("runtime-environment-only")
    api_version: str = Param("runtime-environment-only")
    completion_budget: int = Param(4096)

    def prepare_params(self, **kwargs):
        if kwargs:
            raise CompanyChatError("Unsupported company chat request parameters.")
        if not isinstance(self.completion_budget, int) or not 1 <= self.completion_budget <= 16000:
            raise CompanyChatError("Invalid completion token budget.")
        _, _, deployment, _ = company_config()
        return {
            "model": deployment,
            "temperature": 1,
            "max_completion_tokens": self.completion_budget,
        }

    def prepare_client(self, async_version=False):
        if async_version:
            raise CompanyChatError("Async transport is not enabled in this pilot.")
        endpoint, version, _, key = company_config()
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=version,
            api_key=key,
            max_retries=0,
            timeout=60.0,
            http_client=company_http_client(),
        )

    def invoke(self, messages, *args, **kwargs):
        if args:
            raise CompanyChatError("Unsupported positional chat arguments.")
        params = self.prepare_params(**kwargs)
        prepared = self.prepare_message(messages)
        try:
            with self.prepare_client() as client:
                response = client.chat.completions.create(
                    messages=prepared, stream=False, **params
                )
        except APIStatusError as error:
            raise CompanyChatError(f"Company API HTTP {error.status_code}.") from None
        except APIConnectionError:
            raise CompanyChatError("Company API network or TLS connection failed.") from None
        except OpenAIError:
            raise CompanyChatError("Company API request failed.") from None
        return self.prepare_output(response.model_dump())

    def prepare_output(self, response):
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise CompanyChatError("Company API returned unexpected choices.")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise CompanyChatError("Company API output is incomplete or filtered; no answer accepted.")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise CompanyChatError("Company API returned an invalid message.")
        text = message.get("content")
        if (not isinstance(text, str) or not text.strip() or message.get("refusal")
                or message.get("tool_calls") or message.get("function_call")):
            raise CompanyChatError("Company API returned empty, refused or unsupported output.")
        usage = response.get("usage")
        if not isinstance(usage, dict) or any(
            not isinstance(usage.get(key), int) or isinstance(usage.get(key), bool)
            or usage[key] < 0
            for key in ("total_tokens", "prompt_tokens", "completion_tokens")
        ):
            raise CompanyChatError("Company API returned invalid usage metadata.")
        return super().prepare_output(response)

    def stream(self, messages, *args, **kwargs):
        # Deliver only a completely validated answer; no unvalidated partial text.
        yield self.invoke(messages, *args, **kwargs)

    async def ainvoke(self, messages, *args, **kwargs):
        import asyncio

        return await asyncio.to_thread(self.invoke, messages, *args, **kwargs)

    async def astream(self, messages, *args, **kwargs):
        yield await self.ainvoke(messages, *args, **kwargs)
