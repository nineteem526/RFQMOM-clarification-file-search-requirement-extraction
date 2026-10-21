import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from workbench import Store

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class AppTests(unittest.TestCase):
    def test_demo_import_search_versions_and_missing_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("workbench.default_database", return_value=Path(folder) / "demo.sqlite3"):
                with patch.dict("os.environ", {"DUERR_API_KEY": ""}):
                    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
                    self.assertEqual(len(app.exception), 0)
                    app.sidebar.button[0].click().run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertIn("新增 3", app.success[0].value)
                    app.sidebar.button[0].click().run()
                    self.assertIn("新增 0", app.success[0].value)
                    app.text_input[0].set_value("钢构").run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertGreater(len(app.expander), 0)
                    self.assertTrue(any("5 kN/m²" in text.value for text in app.text))
                    app.sidebar.button[1].click().run()
                    app.multiselect[0].set_value(["TE"]).run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertTrue(any("尚未取得客户书面批准" in text.value for text in app.text))
                    test_button = next(button for button in app.button
                                       if button.label == "发送最小聊天测试")
                    test_button.click().run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertTrue(any("未配置 DUERR_API_KEY" in error.value for error in app.error))

    def test_extract_preview_approval_and_selection_change(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "demo.sqlite3"
            client = MagicMock()
            with patch("workbench.default_database", return_value=database):
                with patch("workbench.make_client", return_value=client):
                    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
                    app.sidebar.button[0].click().run()
                    app.text_input[0].set_value("钢构").run()
                    store = Store(database)
                    try:
                        sources = store.search("钢构")
                    finally:
                        store.close()
                    source = sources[0]
                    result = {
                        "items": [{
                            "type": "requirement", "description": "虚构审核描述",
                            "evidence": [{"source_id": source.source_id, "quote": source.text}],
                        }],
                        "limitations": ["无标准原文，无法提供标准建议。"],
                    }
                    api_response = SimpleNamespace(choices=[SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=json.dumps(result), refusal=None),
                    )])
                    client.__enter__.return_value.chat.completions.create.return_value = api_response
                    app.multiselect[1].set_value([source.source_id]).run()
                    extract_button = next(button for button in app.button
                                          if button.label == "由内部 AI 提取要求")
                    self.assertTrue(extract_button.disabled)
                    self.assertEqual(client.__enter__.return_value.chat.completions.create.call_count, 0)
                    self.assertTrue(any(
                        source.text == json.loads(block.value)[0]["text"]
                        for block in app.code if block.language == "json"
                    ))
                    app.checkbox[1].check().run()
                    next(button for button in app.button
                         if button.label == "由内部 AI 提取要求").click().run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(client.__enter__.return_value.chat.completions.create.call_count, 1)
                    self.assertTrue(any("虚构审核描述" == text.value for text in app.text))
                    app.multiselect[1].set_value([sources[1].source_id]).run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertFalse(app.checkbox[1].value)
                    self.assertFalse(any("虚构审核描述" == text.value for text in app.text))
                    self.assertTrue(next(button for button in app.button
                                         if button.label == "由内部 AI 提取要求").disabled)


if __name__ == "__main__":
    unittest.main()
