"""Native Kotaemon ChatPage: read-only preview or fixed synthetic corpus QA."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

from pilot_runtime import prepare_runtime


def local_only(event, args):
    if event in {"socket.connect", "socket.connect_ex"}:
        address = args[1]
        if not isinstance(address, tuple) or address[0] not in {
            "127.0.0.1", "::1"
        }:
            raise RuntimeError("Preview forbids non-loopback network connections.")
    if event == "socket.getaddrinfo" and args[0] not in {
        "127.0.0.1", "::1", "localhost", None
    }:
        raise RuntimeError("Preview forbids external DNS lookups.")


def icon_directory():
    app_path = os.environ.get("APP_PATH")
    if os.environ.get("RFQ_APP_SERVICE", "").casefold() in {"1", "true"} and app_path:
        path = Path(app_path) / "_appservice-assets" / "icons"
        for name in ("dark_mode", "expand", "rename", "delete", "new"):
            if not (path / f"{name}.svg").is_file():
                raise RuntimeError(f"Kotaemon preview icon is missing: {name}")
        return path
    # The installed wheel omits icons; use the approved fixed source checkout.
    path = (
        Path(sys.prefix).parent / "kotaemon" / "libs" / "ktem"
        / "ktem" / "assets" / "icons"
    )
    for name in ("dark_mode", "expand", "rename", "delete", "new"):
        if not (path / f"{name}.svg").is_file():
            raise RuntimeError(f"Kotaemon preview icon is missing: {name}")
    return path


def company_network_boundary():
    import socket
    from urllib.parse import urlsplit
    from check_company_api import ENDPOINT

    hostname = urlsplit(ENDPOINT).hostname
    addresses = set()
    original_resolver = socket.getaddrinfo
    permitted_hosts = {"127.0.0.1", "::1", "localhost", hostname, None}

    def resolve(host, *args, **kwargs):
        normalized = host.decode("ascii") if isinstance(host, bytes) else host
        if normalized not in permitted_hosts:
            raise RuntimeError("Blocked DNS lookup outside localhost/company API.")
        result = original_resolver(host, *args, **kwargs)
        if normalized == hostname:
            addresses.update(item[4][0] for item in result)
        return result

    # Resolve company addresses only when making a request: offline scanning must work.
    socket.getaddrinfo = resolve

    def guard(event, args):
        if event in {"socket.connect", "socket.connect_ex"}:
            address = args[1]
            if isinstance(address, tuple) and (
                address[0] in {"127.0.0.1", "::1"}
                or (address[0] in addresses and address[1] == 443)
            ):
                return
            raise RuntimeError("Blocked connection outside localhost/company API.")
        if event == "socket.getaddrinfo":
            host = args[0].decode("ascii") if isinstance(args[0], bytes) else args[0]
            if host not in permitted_hosts:
                raise RuntimeError("Blocked DNS lookup outside localhost/company API.")

    return guard


def build_preview(synthetic=False):
    root = prepare_runtime()
    os.environ["THEFLOW_SETTINGS_MODULE"] = "preview_settings"
    os.environ["USE_LOW_LLM_REQUESTS"] = "true"
    from theflow.settings import settings
    import preview_settings

    if settings.KH_DATABASE != "sqlite://":
        raise RuntimeError("Preview requires its isolated in-memory database.")

    # One upstream component imports flowsettings directly instead of the loader.
    sys.modules["flowsettings"] = preview_settings
    sys.addaudithook(company_network_boundary() if synthetic else local_only)

    import gradio as gr
    import ktem
    from ktem.assets import KotaemonTheme
    from ktem.pages.chat import ChatPage
    from ktem.pages.chat import control
    from ktem.settings import BaseSettingGroup
    knowledge = None
    if synthetic:
        from synthetic_qa import SyntheticKnowledge

        knowledge = SyntheticKnowledge(root / "synthetic-only-v1" / "sources")

    assets = Path(ktem.__file__).resolve().parent / "assets"
    control.ASSETS_DIR = str(icon_directory())
    reasoning = BaseSettingGroup(
        settings={
            "use": {"name": "Reasoning", "value": "simple",
                    "choices": [("快速问答（未接通）", "simple")]},
            "lang": {"name": "Language", "value": "zh",
                     "choices": [("中文", "zh")]},
        },
        options={
            "simple": BaseSettingGroup(settings={
                "llm": {"name": "Model", "value": "",
                        "choices": [("公司 API（预览不调用）", "")]},
                "highlight_citation": {"name": "Citation", "value": "off",
                                       "choices": [("原文证据", "off")]},
            })
        },
    )
    app = SimpleNamespace(
        app_name=("工程项目知识问答 · 虚构资料问答" if synthetic
                  else "工程项目知识问答 · 只读界面预览"),
        index_manager=SimpleNamespace(indices=[]),
        default_settings=SimpleNamespace(reasoning=reasoning),
    )
    css = (assets / "css" / "main.css").read_text(encoding="utf-8")
    if synthetic:
        css += "\n#report-accordion, #chat-settings-expand { display: none; }\n"
    theme = KotaemonTheme(
        font=("Microsoft YaHei", "Segoe UI", "sans-serif"),
        font_mono=("Consolas", "monospace"),
    )
    with gr.Blocks(
        theme=theme, css=css, analytics_enabled=False,
        title=app.app_name,
    ) as demo:
        gr.Markdown((
            "## 工程项目知识问答 · 虚构资料试用\n"
            "**仅限虚构问题，不输入真实资料。** 已索引固定 TXT、DOCX、XLSX 样例；"
            "本地全文＋向量检索，点击发送会将问题和检索证据发往公司 API。"
            "仅提供经逐字核对的证据摘录，暂不提供自由生成汇总。"
            "上传、会话保存、真实文件录入均关闭。"
        ) if synthetic else (
            "## 工程项目知识问答 · Kotaemon 界面预览\n"
            "**只读预览，不是可用知识库。** 复用已安装 Kotaemon 的聊天页面；"
            "上传、发送、保存均禁用，不调用公司 API，不读取业务资料。"
        ))
        with gr.Tabs(selected="chat"):
            with gr.Tab("知识问答", id="chat"):
                page = ChatPage(app)
                page.chat_panel.chatbot.value = [(
                    "【静态虚构示例】钢构设计荷载有什么要求？",
                    "【人工编写的展示文字，非检索或 AI 结果】\n\n"
                    "虚构 RFQ：平台均布荷载 **3.0 kN/m²，已确认**。\n\n"
                    "虚构 MOM：建议提高至 **5.0 kN/m²，待批准**，"
                    "不能据此认定旧要求已被替代。\n\n"
                    "右侧显示对应的虚构原文示例。",
                )]
                page.info_panel.value = (
                    "<h3>虚构原文示例（未建立索引）</h3>"
                    "<p><b>虚构RFQ.txt</b> · 2026-09-01 · 第1段</p>"
                    "<blockquote>平台均布荷载为3.0 kN/m²。本条已确认。</blockquote>"
                    "<p><b>虚构MOM.txt</b> · 2026-10-01 · 第2段</p>"
                    "<blockquote>建议提高平台均布荷载至5.0 kN/m²，"
                    "待业主批准。</blockquote>"
                    "<p>以上日期和定位仅属于展示样例；PDF 跳转、高亮及"
                    "真实引用尚未接通。</p>"
                )
                page.chat_panel.text_input.placeholder = "只读预览：暂不能提问或上传"
                page.report_issue.report_btn.visible = False
                page.report_issue.more_detail.placeholder = "预览不提交反馈"
                if synthetic:
                    page.chat_panel.chatbot.value = []
                    page.chat_panel.chatbot.placeholder = (
                        "已录入固定虚构资料。可问：钢构平台荷载有哪些新旧要求？"
                        "或 PLC-S7-1500 的电压是多少？"
                    )
                    page.info_panel.value = (
                        "<h3>等待提问</h3><p>已索引3个固定虚构文件、7条证据。"
                        "可问钢构荷载、电气控制柜、PLC型号和接口供货边界。</p>"
                    )
                    page.chat_panel.text_input.placeholder = (
                        "仅输入虚构测试问题，例如：钢结构平台荷载有哪些新旧要求？"
                    )
                    with page.conv_column:
                        gr.Markdown(
                            "### 本次资料范围\n"
                            "虚构RFQ.txt：荷载、防腐\n\n"
                            "虚构MOM.docx：荷载、电气建议\n\n"
                            "虚构电气表.xlsx：型号、数值、单位、供货边界\n\n"
                            "固定测试项目；会话不保存。"
                        )
            if synthetic:
                with gr.Tab("资料扫描（不入库）"):
                    gr.Markdown(
                        "仅扫描当前工作目录下的“评审后”。读取元信息并计算字节哈希，"
                        "不解析正文、不调用AI、不复制或修改源文件。"
                        "首次全部标记新增；后续相对上次扫描比较，不代表已入库。"
                        "同名跨格式仅列为疑似版本对，不能据此认定内容相同。"
                    )
                    scan_button = gr.Button("手动扫描“评审后”")
                    scan_status = gr.Textbox(label="扫描状态", interactive=False)
                    scan_table = gr.Dataframe(
                        headers=["文件", "相对上次扫描", "解析限制", "源文件修改时间"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                    )
                    scan_details = gr.JSON(label="重复／疑似版本对／失败（仅本机）")
            with gr.Tab("接入状态"):
                gr.Markdown((
                    "已接通固定虚构资料的 TXT／DOCX／XLSX 解析、本地全文＋向量检索、"
                    "公司 API 证据选择，以及带原文校验的来源显示。\n\n"
                    "不是正式业务版本：任意文件上传、项目切换、PDF预览、长文分块、"
                    "自由汇总、逐文件主题覆盖、结果保存和审核导出仍未接通。"
                ) if synthetic else (
                    "公司聊天适配、本地嵌入组件已完成。\n\n"
                    "文件管理、混合检索、真实问答、引用跳转尚未接通。"
                    "项目隔离、版本日期及审核导出尚未实施。\n\n"
                    "本预览没有注册问答、录入、保存或反馈回调。"
                ))
        for component in demo.blocks.values():
            if isinstance(component, gr.components.Component) and hasattr(
                component, "interactive"
            ):
                component.interactive = False
        if synthetic:
            from company_chat import CompanyChatError
            from synthetic_qa import answer_markdown, evidence_html

            page.chat_panel.text_input.interactive = True
            scan_button.interactive = True

            def scan():
                from scan_sources import scan_authorized_sources

                try:
                    result, output = scan_authorized_sources()
                except (OSError, ValueError) as error:
                    raise gr.Error(f"扫描未完成：{error}") from None
                return (
                    f"扫描记录：{output}；文件{len(result['files'])}个，"
                    f"失败{len(result['failures'])}项。未入库，未调用AI。",
                    [[row["path"], row["change"], row["parser_status"],
                      row["source_modified_utc"]] for row in result["files"]],
                    {key: result[key] for key in (
                        "byte_identical_groups", "possible_version_pairs",
                        "failures", "ignored", "previous_paths_not_successfully_seen"
                    )},
                )

            scan_button.click(
                scan, inputs=[],
                outputs=[scan_status, scan_table, scan_details],
                concurrency_limit=1, api_name=False,
            )

            def respond(message, history):
                if not isinstance(message, dict) or message.get("files"):
                    raise gr.Error("此试用只接受文字，不接受文件上传。")
                question = message.get("text", "")
                try:
                    selected, retrieved = knowledge.answer(question)
                except (ValueError, CompanyChatError) as error:
                    raise gr.Error(str(error)) from None
                return (
                    {"text": "", "files": []},
                    (history or [])[-9:] + [(question, answer_markdown(selected))],
                    evidence_html(retrieved),
                )

            page.chat_panel.text_input.submit(
                respond,
                inputs=[page.chat_panel.text_input, page.chat_panel.chatbot],
                outputs=[
                    page.chat_panel.text_input, page.chat_panel.chatbot, page.info_panel
                ],
                concurrency_limit=1, api_name=False,
            )
    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    demo = build_preview(synthetic=args.synthetic)
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn

    server = FastAPI()

    @server.middleware("http")
    async def browser_boundary(request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if host not in {"127.0.0.1", "localhost"} or (
            origin and origin not in {
                f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"
            }
        ):
            response = JSONResponse({"detail": "Local same-origin access required."},
                                    status_code=403)
        elif request.url.path.rstrip("/") == "/upload" or (
            not args.synthetic and request.method not in {"GET", "HEAD"}
        ):
            response = JSONResponse(
                {"detail": "Read-only preview: uploads and actions are disabled."},
                status_code=405,
            )
        else:
            response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            "font-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; "
            "frame-src 'self'; object-src 'none'; form-action 'self'; "
            "base-uri 'self'; frame-ancestors 'none'"
        )
        return response

    server = gr.mount_gradio_app(
        server, demo, path="/", allowed_paths=[str(icon_directory())]
    )
    mode = "Synthetic-only Kotaemon QA" if args.synthetic else "Read-only Kotaemon preview"
    print(f"{mode}: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(server, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
