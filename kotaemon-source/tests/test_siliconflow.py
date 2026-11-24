import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import rag_engine
from web_app import ProjectStore


class SiliconFlowTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {
            "RFQ_LLM_PROVIDER": "siliconflow",
            "SILICONFLOW_API_KEY": "test-key",
            "SILICONFLOW_MODEL": "deepseek-ai/DeepSeek-V3",
        }, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_configuration_and_cache_identity(self):
        status = rag_engine.llm_config_status()
        self.assertTrue(status["configured"])
        self.assertEqual(status["provider"], "siliconflow")
        self.assertEqual(status["cache_identity"][-1], "deepseek-ai/DeepSeek-V3")
        os.environ["SILICONFLOW_MODEL"] = ""
        self.assertFalse(rag_engine.llm_config_status()["configured"])

    def test_request_and_verified_answer(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "x.pdf", "page": 1,
                 "content": "Rated voltage: 18 V DC", "semantic_score": 0.9}]
        result = {"conclusion": "额定电压为 18 V DC [1]", "basis": [], "table": []}
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                                    message=SimpleNamespace(content=json.dumps(result)))])
        with patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.return_value = response
            answer = rag_engine.generate_structured_answer("电压是多少", hits, "test", {}, allow_external=True)
        self.assertTrue(answer["verification"]["passed"])
        self.assertTrue(answer["llm_provider"].startswith("硅基流动"))
        self.assertEqual(client.call_args.kwargs["base_url"], "https://api.siliconflow.cn/v1")
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-ai/DeepSeek-V3")
        self.assertIn("Rated voltage: 18 V DC", request["messages"][1]["content"])

    def test_opt_in_and_invalid_answer_fall_back(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "x.pdf", "page": 1,
                 "content": "Rated voltage: 18 V DC", "semantic_score": 0.9}]
        with patch("rag_engine._call_external_llm", return_value={"conclusion": "99 V [1]"}) as call:
            rag_engine.generate_structured_answer("电压", hits, "test", {})
            call.assert_not_called()
            answer = rag_engine.generate_structured_answer("电压", hits, "test", {}, allow_external=True)
        self.assertNotIn("99 V", answer["conclusion"])
        self.assertTrue(answer["verification"]["warnings"])


if __name__ == "__main__":
    unittest.main()
