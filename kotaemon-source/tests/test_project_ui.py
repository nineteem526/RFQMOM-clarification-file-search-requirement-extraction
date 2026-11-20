import os
import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import project_app


def evidence(filename="fixture.pdf"):
    return SimpleNamespace(
        id_="chunk1",
        content="荷载 5 kN/m²，拟采用，待批准。<script>alert(1)</script>",
        metadata={
            "version": "version1", "file_name": filename, "location": "PDF第3页",
            "date": "2026-10-19", "date_basis": "文件修改时间兜底",
            "approval": "未确认；以原文措辞为准",
            "version_label": "Rev.A", "document_group": "fixture", "sha256": "a" * 64,
        },
    )


class SourcePanelTests(unittest.TestCase):
    def test_project_context_counts_only_enabled_usable_versions_and_escapes_id(self):
        coverage = [
            {"enabled": True, "status": "ready"},
            {"enabled": True, "status": "limited"},
            {"enabled": False, "status": "ready"},
            {"enabled": True, "status": "failed"},
        ]
        context = project_app.project_context("<script>project</script>", coverage)
        self.assertIn("纳入查询 2 个资料版本", context)
        self.assertIn("其中 1 个存在解析限制", context)
        self.assertIn("&lt;script&gt;project&lt;/script&gt;", context)
        self.assertNotIn("<script>", context)
        self.assertIn("不代表原件完整解析", context)
        for rows in ([], [{"enabled": False, "status": "ready"}]):
            self.assertIn("暂无可查询资料", project_app.project_context("fixture", rows))

    def test_query_modes_explain_limits_and_followup_without_renaming_backend_values(self):
        for mode in ("快速问答", "主题逐文件检查", "深度全文检查"):
            self.assertIn(mode, project_app.query_mode_help(mode, False))
        self.assertIn("独立问题", project_app.query_mode_help("快速问答", False))
        self.assertIn("不以历史 AI 回答为证据", project_app.query_mode_help("快速问答", True))
        self.assertIn("8 个", project_app.query_mode_help("主题逐文件检查", False))
        self.assertIn("上千次", project_app.query_mode_help("深度全文检查", False))
        with self.assertRaisesRegex(ValueError, "查询方式"):
            project_app.query_mode_help("unknown", False)

    def test_cached_answer_notice_discloses_original_date_and_no_model_request(self):
        self.assertEqual(project_app.cache_notice({}, {}), "")
        notice = project_app.cache_notice({"cached_from": "job"}, {"created": "2026-01-01<script>"})
        self.assertIn("未调用模型", notice)
        self.assertIn("2026-01-01&lt;script&gt;", notice)
        self.assertIn("取消复用", notice)
        stale = project_app.cache_notice({}, {"stale": True})
        self.assertIn("资料已更新", stale)
        self.assertIn("历史回答快照", stale)

    def test_sources_keep_locations_dates_and_original_wording(self):
        result = project_app.source_panel([evidence()])
        for text in ("PDF第3页", "2026-10-19", "文件修改时间兜底",
                     "批准状态：未确认；以原文措辞为准",
                     "拟采用，待批准", "Rev.A", "SHA256"):
            self.assertIn(text, result)
        for path in ("/context/chunk1", "/source/version1", "/preview/chunk1"):
            self.assertIn(path, result)
        self.assertIn("不代表全部资料", result)

    def test_missing_approval_is_explicit_and_untrusted_status_is_escaped(self):
        doc = evidence("fixture.txt")
        doc.metadata["approval"] = "<script>pending</script>"
        result = project_app.source_panel([doc])
        self.assertIn("批准状态：&lt;script&gt;pending&lt;/script&gt;", result)
        self.assertNotIn("批准状态：<script>", result)

        doc.metadata.pop("approval")
        result = project_app.source_panel([doc])
        self.assertIn("批准状态：未标注", result)

    def test_source_text_and_names_are_escaped(self):
        result = project_app.source_panel([evidence('<img src=x onerror=alert(1)>.pdf')])
        self.assertNotIn("<script>", result)
        self.assertNotIn("<img ", result)
        self.assertIn("&lt;script&gt;", result)
        self.assertIn("&lt;img ", result)

    def test_non_pdf_does_not_offer_pdf_preview(self):
        doc = evidence("fixture.xlsx")
        doc.metadata["location"] = "Sheet1!A1:D8"
        result = project_app.source_panel([doc])
        self.assertNotIn("/preview/", result)
        self.assertIn("Sheet1!A1:D8", result)

    def test_empty_sources_are_explicit(self):
        self.assertIn("未检索项目原文", project_app.source_panel([]))


class WorkspaceLayoutTests(unittest.TestCase):
    def test_reparented_chat_remains_visible_and_sources_are_opt_in(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import runpy,sys; runpy.run_path(sys.argv[1])['verify_layout']()",
             str(Path(__file__).resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_single_project_mode_hides_project_creation_and_restore(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import runpy,sys; runpy.run_path(sys.argv[1])['verify_single_project_layout']()",
             str(Path(__file__).resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def verify_layout():
    from pilot_runtime import prepare_runtime

    # TheFlow settings are process-global; build in a fresh process like the service.
    prepare_runtime()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "authorized-source"
        source.mkdir()
        with patch.object(project_app, "prepare_runtime", return_value=Path(directory)), \
                patch("project_registry.authorized_source", return_value=source):
            demo, _workspace = project_app.build_app("ui-fixture")
        try:
            config = demo.get_config_file()
            blocks = {
                component["props"].get("elem_id"): component
                for component in config["components"]
                if component["props"].get("elem_id")
            }
            assert blocks["main-chat-bot"]["props"]["visible"]
            assert blocks["chat-input"]["props"]["visible"]
            assert not blocks["chat-info-panel"]["props"]["visible"]
            assert not blocks["toggle-sources"]["props"]["value"]
            for identity in ("history-scroll", "chat-toolbar", "chat-composer", "query-status"):
                assert identity in blocks
            def layout_ids(node):
                return [node["id"]] + [
                    identity for child in node.get("children", [])
                    for identity in layout_ids(child)
                ]
            identities = layout_ids(config["layout"])
            assert len(identities) == len(set(identities)), "Duplicate rendered component"
            new_conversation = next(
                function for function in demo.fns.values()
                if blocks["new-conversation"]["id"] in
                [target[0] for target in function.targets]
            )
            values = new_conversation.fn()
            assert len(values) == len(new_conversation.outputs)
            assert values[-3:] == (False, False, {"text": "", "files": []})
            verify_frontend_callbacks(demo, _workspace, blocks, config)
            verify_reuse_callbacks(demo, _workspace)
        finally:
            demo.close()


def verify_single_project_layout():
    from pilot_runtime import prepare_runtime

    prepare_runtime()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "authorized-source"
        source.mkdir()
        with patch.object(project_app, "prepare_runtime", return_value=Path(directory)), \
                patch("project_registry.authorized_source", return_value=source), \
                patch.dict(os.environ, {"RFQ_SINGLE_PROJECT_MODE": "true"}):
            demo, _workspace = project_app.build_app("ui-single")
        try:
            project_tab = next(
                block for block in demo.blocks.values()
                if getattr(block, "label", None) == "项目"
            )
            assert not project_tab.visible
            for label in (
                "恢复为新项目ID（小写字母／数字／短横线，现有项目不可覆盖）",
                "打开所选项目（本机独立入口）",
                "创建空项目（仍绑定授权目录）",
            ):
                control = next(
                    block for block in demo.blocks.values()
                    if getattr(block, "label", None) == label
                    or getattr(block, "value", None) == label
                )
                assert not control.interactive, label
        finally:
            demo.close()


def verify_frontend_callbacks(demo, workspace, blocks, config):
    functions = {getattr(item.fn, "__name__", ""): item for item in demo.fns.values()}
    for identity in ("cancel-query", "save-result", "select-all-documents", "ingest-documents"):
        assert not blocks[identity]["props"]["interactive"], identity
    assert blocks["query-mode"]["props"]["label"] == "查询方式"
    assert "ui-fixture" in blocks["workspace-header"]["props"]["value"]
    assert "暂无可查询资料" in blocks["workspace-context"]["props"]["value"]
    assert "原件完整解析" in blocks["workspace-context"]["props"]["value"]
    for label, question in project_app.QUESTION_EXAMPLES:
        button = next(item for item in config["components"]
                      if item["type"] == "button" and item["props"]["value"] == label)
        key, callback = next((key, fn) for key, fn in demo.fns.items()
                             if (button["id"], "click") in fn.targets)
        assert callback.fn() == {"text": question, "files": []}
        assert [item._id for item in callback.outputs] == [blocks["chat-input"]["id"]]
        assert not any(item.get("trigger_after") == key for item in config["dependencies"])

    prepare = functions["prepare_query"]
    values = prepare.fn()
    assert len(values) == len(prepare.outputs)
    assert all(not value["interactive"] for value in values[:-1])
    assert not values[-1]["visible"]
    finish = functions["finish_query"]
    for identity, history in (("", []), ("result", [("question", "answer")])):
        values = finish.fn(identity, history)
        assert len(values) == len(finish.outputs)
        assert not values[0]["interactive"]
        assert values[1]["interactive"] == bool(identity)
        assert all(value["interactive"] for value in values[2:-1])
        assert values[-1]["visible"] == (not bool(history))
    for name in ("respond", "resume"):
        key = next(key for key, fn in demo.fns.items() if fn is functions[name])
        cleanup = [item for item in config["dependencies"] if item.get("trigger_after") == key]
        assert len(cleanup) == 1
        assert not cleanup[0]["trigger_only_on_success"], "Restore controls on errors too"

    selection = functions["update_selection"]
    for selected in (None, [], ["fixture.txt"], ["a.txt", "b.txt"]):
        enabled, text = selection.fn(selected)
        assert enabled["interactive"] == bool(selected)
        assert (f"已选择 {len(selected)} 个" in text) if selected else ("尚未选择" in text)
    for paths in ([], ["fixture.txt"]):
        with patch.object(workspace, "scan", return_value={
            "scan_id": "fixture-scan", "files": [{"path": item} for item in paths],
            "failures": [],
        }):
            values = functions["scan"].fn()
        assert len(values) == len(functions["scan"].outputs)
        assert values[3]["value"] == []
        assert values[5]["interactive"] == bool(paths)
        assert not values[7]["interactive"]
        if not paths:
            assert "没有可选文件" in values[6]

    from query_jobs import QueryJobs
    running = {
        "session_id": "fixture-session", "elapsed": 0, "status": "running",
        "stage": "fixture", "progress": 0.5,
    }
    for status in ("failed", "cancelled", "interrupted"):
        terminal = {**running, "status": status, "error": "fixture end"}
        with patch.object(QueryJobs, "start", return_value="fixture-job"), \
             patch.object(QueryJobs, "get", side_effect=[running, running, terminal]), \
             patch.object(project_app.time, "sleep"):
            values = list(functions["respond"].fn(
                {"text": "虚构问题", "files": []}, [], "快速问答", False, "", True,
            ))
        assert values[0][-1]["interactive"]
        assert not values[-1][-1]["interactive"]
        assert values[-1][3] == ""
        assert status in values[-1][4]
        assert len(values[-1]) == len(functions["respond"].outputs)


def verify_reuse_callbacks(demo, workspace):
    from project_workspace import now
    import uuid

    calls = []

    def fake_ask(question, **_kwargs):
        calls.append(question)
        identity = uuid.uuid4().hex
        doc = evidence("fixture.txt")
        doc.content = "虚构要求，待批准。"
        doc.metadata["location"] = "第3段"
        citation = {"id": doc.id_, "quote": doc.content}
        answer = {"sections": [{"title": "虚构", "points": [
            {"text": "虚构要求，待批准。", "citations": [citation]},
        ]}]}
        with workspace.connection() as conn:
            conn.execute(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,0,?)",
                (identity, now(), workspace.revision(), question, "快速问答", "[]",
                 json.dumps([{"id": doc.id_, "text": doc.content, "metadata": doc.metadata}]),
                 json.dumps(answer)),
            )
        return identity, [], [], []

    respond = next(function for function in demo.fns.values()
                   if getattr(function.fn, "__name__", "") == "respond")
    load = next(function for function in demo.fns.values()
                if getattr(function.fn, "__name__", "") == "load_conversation")
    question = {"text": "PLC虚构要求", "files": []}
    with patch.object(workspace, "ask", side_effect=fake_ask):
        first = list(respond.fn(question, [], "快速问答", False, "", True))[-1]
        second = list(respond.fn(question, [], "快速问答", False, "", True))[-1]
        assert len(second) == len(respond.outputs)
        assert not second[-1]["interactive"]
        assert len(calls) == 1
        assert second[3] == first[3]
        assert "复用已有回答" in second[1][-1][1]
        restored = load.fn(second[7])
        assert "复用已有回答" in restored[1][-1][1]
        assert "待批准" in restored[2]
        third = list(respond.fn(question, [], "快速问答", False, "", False))[-1]
        assert len(calls) == 2
        assert third[3] != first[3]
        assert "复用已有回答" not in third[1][-1][1]


if __name__ == "__main__":
    unittest.main()
