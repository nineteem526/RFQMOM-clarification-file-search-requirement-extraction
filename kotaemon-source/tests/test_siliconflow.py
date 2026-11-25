import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import rag_engine


class SiliconFlowTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {
            "RFQ_LLM_PROVIDER": "siliconflow",
            "SILICONFLOW_API_KEY": "test-key",
            "SILICONFLOW_MODEL": "deepseek-ai/DeepSeek-V4.1-Flash",
        }, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_configuration_and_cache_identity(self):
        status = rag_engine.llm_config_status()
        self.assertTrue(status["configured"])
        self.assertEqual(status["provider"], "siliconflow")
        self.assertEqual(status["cache_identity"][-1], "deepseek-ai/DeepSeek-V4.1-Flash")
        os.environ["SILICONFLOW_MODEL"] = ""
        self.assertFalse(rag_engine.llm_config_status()["configured"])

    def test_request_and_verified_answer_with_markdown_json_fence(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "makita.pdf", "page": 2,
                 "content": "Model: XFD12\nRated voltage: D.C. 18 V", "semantic_score": 0.9}]
        fenced = "```json\n" + json.dumps({
            "conclusion": "1. XFD12 额定电压为 18V DC，详见第 2 页 [1]。请核对原文。",
            "basis": [{"index": 1, "source": "makita.pdf", "detail": "原文标注 D.C. 18 V"}],
            "table": [{"item": "额定电压", "value": "18 V [1]"}],
        }, ensure_ascii=False) + "\n```"
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=45, total_tokens=165)
        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=fenced))],
            usage=usage,
        )
        with patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.return_value = response
            answer = rag_engine.generate_structured_answer("电压是多少", hits, "test", {}, allow_external=True)
        self.assertTrue(answer["verification"]["passed"], answer["verification"]["warnings"])
        self.assertEqual(answer["answer_status"], "llm_success")
        self.assertTrue(answer["llm_provider"].startswith("硅基流动"))
        self.assertEqual(answer["llm_stats"]["api_calls"], 1)
        self.assertEqual(answer["llm_stats"]["total_tokens"], 165)
        self.assertEqual(client.call_args.kwargs["base_url"], "https://api.siliconflow.cn/v1")

    def test_verifier_allows_trailing_zero_and_unit_variants_without_false_reject(self):
        citations = [{
            "index": 1, "file_name": "rfq.txt", "location": "第1段",
            "content": "钢结构平台设计均布荷载为 3.0 kN/m²，干膜总厚度不小于 160 μm。本条已确认。",
        }]
        payload = {
            "conclusion": "钢结构平台均布荷载为 3 kN/m²，干膜厚度不小于 160 um [1]。",
            "basis": [{"index": 1, "detail": "荷载 3.0 kN/m²，干膜 160 μm"}],
            "table": [{"item": "荷载", "value": "3 kN/m² [1]"}],
        }
        ver = rag_engine.verify_grounded_answer(payload, citations)
        self.assertTrue(ver["passed"], ver["warnings"])

    def test_verifier_still_blocks_wrong_numbers_out_of_range_cites_and_unapproved(self):
        citations = [{
            "index": 1, "file_name": "mom.docx", "location": "第1段",
            "content": "室外电气控制柜建议改为 IP65，待批准；室内控制柜仍按 IP54 执行。",
        }]
        bad_pending = {"conclusion": "室外控制柜已批准采用 IP65 [1]。", "basis": [], "table": []}
        ver1 = rag_engine.verify_grounded_answer(bad_pending, citations)
        self.assertFalse(ver1["passed"])
        self.assertTrue(any("待批准" in w or "已批准" in w for w in ver1["warnings"]))

        bad_num = {"conclusion": "室外控制柜防护等级为 IP68 [1]，待批准。", "basis": [], "table": []}
        ver2 = rag_engine.verify_grounded_answer(bad_num, citations)
        self.assertFalse(ver2["passed"])
        self.assertTrue(any("ip68" in w.lower() for w in ver2["warnings"]))

        bad_cite = {"conclusion": "室外控制柜待批准 [9]。", "basis": [], "table": []}
        ver3 = rag_engine.verify_grounded_answer(bad_cite, citations)
        self.assertFalse(ver3["passed"])
        self.assertEqual(ver3["invalid_citations"], [9])

    def test_bounded_self_correction_recovers_on_second_call(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "x.pdf", "page": 1,
                 "content": "Rated voltage: 18 V DC", "semantic_score": 0.9}]
        first_bad = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps({
                "conclusion": "额定电压为 18 V DC",  # Missing citation [1]
                "basis": [], "table": [],
            })))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        )
        second_good = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps({
                "conclusion": "额定电压为 18 V DC [1]。",
                "basis": [], "table": [{"item": "额定电压", "value": "18 V DC [1]"}],
            })))],
            usage=SimpleNamespace(prompt_tokens=160, completion_tokens=40, total_tokens=200),
        )
        with patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.side_effect = [first_bad, second_good]
            answer = rag_engine.generate_structured_answer("电压是多少", hits, "test", {}, allow_external=True)
        self.assertEqual(answer["answer_status"], "llm_success")
        self.assertTrue(answer["verification"]["passed"])
        self.assertTrue(answer["llm_stats"]["retried"])
        self.assertEqual(answer["llm_stats"]["api_calls"], 2)
        self.assertEqual(answer["llm_stats"]["total_tokens"], 330)

    def test_opt_in_and_invalid_answer_fall_back(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "x.pdf", "page": 1,
                 "content": "Rated voltage: 18 V DC", "semantic_score": 0.9}]
        with patch("rag_engine._call_external_llm", return_value={"conclusion": "99 V [1]"}) as call:
            rag_engine.generate_structured_answer("电压", hits, "test", {})
            call.assert_not_called()
            answer = rag_engine.generate_structured_answer("电压", hits, "test", {}, allow_external=True)
        self.assertEqual(answer["answer_status"], "verification_failed")
        self.assertNotIn("99 V", answer["conclusion"])
        self.assertTrue(answer["verification"]["warnings"])
        self.assertIn("99v", answer["diagnostic"]["message"].lower())

    def test_error_classification_for_auth_timeout_truncated_and_invalid_json(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "x.pdf", "page": 1,
                 "content": "Rated voltage: 18 V DC", "semantic_score": 0.9}]

        class FakeAuthError(Exception):
            status_code = 401

        with patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.side_effect = FakeAuthError("Unauthorized")
            ans_auth = rag_engine.generate_structured_answer("电压", hits, "test", {}, allow_external=True)
        self.assertEqual(ans_auth["answer_status"], "llm_call_failed")
        self.assertEqual(ans_auth["diagnostic"]["code"], "auth_error")

        truncated = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"conclusion": "partial'))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=2048, total_tokens=2098),
        )
        with patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.return_value = truncated
            ans_trunc = rag_engine.generate_structured_answer("电压", hits, "test", {}, allow_external=True)
        self.assertEqual(ans_trunc["answer_status"], "llm_call_failed")
        self.assertEqual(ans_trunc["diagnostic"]["code"], "truncated")

        bad_json = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="不是 JSON 格式的回答"))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )
        with patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.return_value = bad_json
            ans_json = rag_engine.generate_structured_answer("电压", hits, "test", {}, allow_external=True)
        self.assertEqual(ans_json["answer_status"], "llm_call_failed")
        self.assertEqual(ans_json["diagnostic"]["code"], "invalid_json")


    def test_local_persisted_llm_config_works_without_env_vars(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            cfg_file = Path(tmp) / "local_llm_config.json"
            with patch.dict(os.environ, {
                "RFQ_LOCAL_LLM_CONFIG": str(cfg_file),
                "RFQ_RUNTIME_HOME": str(tmp),
            }, clear=True):
                saved = rag_engine.save_local_llm_config(
                    "siliconflow", "deepseek-ai/DeepSeek-V4.1-Flash", "sk-demo1234567890"
                )
                self.assertTrue(saved["has_key"])
                self.assertNotIn("demo123456", saved["key_mask"])
                st = rag_engine.llm_config_status()
                self.assertTrue(st["configured"])
                self.assertEqual(st["provider"], "siliconflow")
                self.assertEqual(st["model"], "deepseek-ai/DeepSeek-V4.1-Flash")
                # Updating model with empty api_key keeps existing key intact
                saved2 = rag_engine.save_local_llm_config("siliconflow", "deepseek-ai/DeepSeek-V3", "")
                self.assertTrue(saved2["has_key"])
                self.assertEqual(rag_engine.llm_config_status()["model"], "deepseek-ai/DeepSeek-V3")

    def test_admin_pin_role_isolation_and_new_project_auto_online(self):
        import tempfile
        from pathlib import Path
        from fastapi.testclient import TestClient
        from web_app import create_web_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_file = root / "local_llm_config.json"
            with patch.dict(os.environ, {
                "RFQ_LOCAL_LLM_CONFIG": str(cfg_file),
                "RFQ_RUNTIME_HOME": str(root),
            }, clear=True):
                with TestClient(create_web_app(root)) as client:
                    # 1. Anonymous user sees no admin credentials/model details and cannot PUT /api/llm-config
                    anon_cfg = client.get("/api/llm-config").raise_for_status().json()
                    self.assertFalse(anon_cfg["is_admin"])
                    self.assertNotIn("llm_config", anon_cfg)
                    put_denied = client.put("/api/llm-config", json={"provider": "siliconflow"})
                    self.assertEqual(put_denied.status_code, 403)

                    # 2. Wrong PIN rejected, default PIN unlocks admin session
                    bad_login = client.post("/api/admin/login", json={"pin": "wrong"})
                    self.assertEqual(bad_login.status_code, 403)
                    login = client.post("/api/admin/login", json={"pin": "admin123"}).raise_for_status().json()
                    headers = {"X-Admin-Token": login["token"]}

                    # 3. Admin saves SiliconFlow config + default_online_enabled=True
                    saved = client.put(
                        "/api/llm-config",
                        headers=headers,
                        json={
                            "provider": "siliconflow",
                            "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                            "api_key": "sk-admin1234567890",
                            "default_online_enabled": True,
                        },
                    ).raise_for_status().json()
                    self.assertTrue(saved["is_admin"])
                    self.assertTrue(saved["llm_config"]["has_key"])
                    self.assertEqual(saved["llm_config"]["model"], "deepseek-ai/DeepSeek-V4.1-Flash")

                    # 4. Newly created project inherits online_answers_enabled=True automatically
                    client.post("/api/projects", json={"id": "p-auto", "name": "Auto Project"}).raise_for_status()
                    proj_cfg = client.get("/api/projects/p-auto/settings").raise_for_status().json()
                    self.assertTrue(proj_cfg["settings"]["online_answers_enabled"])

                    # 5. Admin can update PIN, old PIN stops working, new PIN works
                    client.put("/api/admin/pin", headers=headers, json={"current_pin": "admin123", "new_pin": "6789"}).raise_for_status()
                    self.assertEqual(client.post("/api/admin/login", json={"pin": "admin123"}).status_code, 403)
                    self.assertEqual(client.post("/api/admin/login", json={"pin": "6789"}).status_code, 200)

                    # 6. Logout revokes token
                    client.post("/api/admin/logout", headers=headers).raise_for_status()
                    self.assertEqual(client.put("/api/llm-config", headers=headers, json={"provider": "local"}).status_code, 403)

    def test_concise_fallback_omits_redundant_basis_and_summary_table(self):
        hits = [{"id": "1", "doc_id": "1", "doc_name": "manual.pdf", "page": 2,
                 "content": "Rated voltage: D.C. 18 V", "semantic_score": 0.9}]
        ans = rag_engine.generate_structured_answer("What is the rated voltage?", hits, "demo", {})
        self.assertEqual(ans["basis"], [])
        self.assertEqual(ans["table"], [])
        self.assertTrue(ans["citations"])

    def test_verifier_accepts_split_table_value_unit_and_rejects_mismatch(self):
        citations = [
            {
                "index": 1,
                "file_name": "虚构电气表.xlsx",
                "location": "虚构电气表.xlsx 第3行",
                "content": "对象：PLC控制电源；型号：PLC-S7-1500；参数：电压；数值：24；单位：V DC；条件：控制回路；批准状态：已确认",
            },
            {
                "index": 2,
                "file_name": "spec.xlsx",
                "location": "工作表 参数",
                "content": "Item: Overall length | Value: 172 | Unit: mm",
            },
        ]
        self.assertTrue(
            rag_engine.verify_grounded_answer(
                {"conclusion": "PLC-S7-1500 控制电源电压为 24 V DC [1]，总长度为 172 mm [2]。"},
                citations,
            )["passed"]
        )
        # Wrong unit or wrong value must still be rejected strictly
        self.assertFalse(
            rag_engine.verify_grounded_answer(
                {"conclusion": "PLC-S7-1500 控制电源电压为 24 V AC [1]。"},
                citations,
            )["passed"]
        )
        self.assertFalse(
            rag_engine.verify_grounded_answer(
                {"conclusion": "PLC-S7-1500 控制电源电压为 48 V DC [1]。"},
                citations,
            )["passed"]
        )

    def test_verifier_allows_question_numbers_only_in_explicit_absence_refusal(self):
        citations = [
            {
                "index": 1,
                "file_name": "manual.pdf",
                "location": "Page 4",
                "content": "Charge the battery cartridge at room temperature (10 °C - 40 °C).",
            },
            {
                "index": 2,
                "file_name": "spec.txt",
                "location": "Section 1",
                "content": "Control cabinet enclosure protection rating: IP54.",
            },
        ]
        self.assertTrue(
            rag_engine.verify_grounded_answer(
                {"conclusion": "文档未提及 0 °C (32 °F) 以下操作准则，仅说明充电温度为 10 °C - 40 °C [1]。"},
                citations,
                question="Are there specific safety guidelines below 0 °C (32 °F)?",
            )["passed"]
        )
        # Affirmative sub-clauses cannot smuggle numbers/models from the question
        self.assertFalse(
            rag_engine.verify_grounded_answer(
                {"conclusion": "原文未规定 IP65，实际防护等级为 IP68 [2]。"},
                citations,
                question="Is cabinet IP65 or IP68?",
            )["passed"]
        )
        self.assertFalse(
            rag_engine.verify_grounded_answer(
                {"conclusion": "原文未规定 IP65 但实为 IP68 [2]。"},
                citations,
                question="Is cabinet IP65 or IP68?",
            )["passed"]
        )
        self.assertFalse(
            rag_engine.verify_grounded_answer(
                {"conclusion": "电控柜防护等级为 IP65 [2]。"},
                citations,
                question="Is cabinet IP65 or IP68?",
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
