"""Project entry point reusing Kotaemon's ChatPage, with explicit ingestion."""

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import html
import json
from functools import wraps
import inspect
import time

from pilot_runtime import prepare_runtime
from citation_location import short_location
from file_lock import FileLockError, exclusive_file_lock


QUESTION_EXAMPLES = (
    ("查钢构要求", "钢构设计有哪些荷载、防腐和供货要求？"),
    ("查电气控制", "电气控制有哪些明确要求、例外和待确认事项？"),
    ("查供货边界", "供货范围中有哪些明确边界和待确认事项？请列出原文出处。"),
)
QUERY_MODE_DESCRIPTIONS = {
    "快速问答": "选取相关证据回答，不保证穷尽全部资料。",
    "主题逐文件检查": "按文件检查主题相关证据，每文件最多检查 8 个相关片段。",
    "深度全文检查": "逐批检查已解析文本，可能产生上千次公司 API 请求、耗时很长；不保证原件完整解析或零遗漏。",
}


def query_mode_help(mode, follow_up):
    if mode not in QUERY_MODE_DESCRIPTIONS:
        raise ValueError("请选择有效的查询方式。")
    context = ("续问已开启：仅沿用上一用户问题，不以历史 AI 回答为证据。"
               if follow_up else "当前为独立问题；开启续问可沿用上一用户问题。")
    return f"**{mode}**：{QUERY_MODE_DESCRIPTIONS[mode]} {context}"


def project_context(project_id, coverage):
    included = [row for row in coverage
                if row["enabled"] and row["status"] in {"ready", "limited"}]
    limited = sum(row["status"] == "limited" for row in included)
    status = (f"纳入查询 {len(included)} 个资料版本，其中 {limited} 个存在解析限制。"
              if included else "暂无可查询资料，请到“资料”扫描并录入文件。")
    return (
        f'<p><strong>当前项目：{html.escape(project_id)}</strong> · '
        f'{status} 仅检索本项目已启用且可用的资料；不代表原件完整解析。</p>'
    )


def source_panel(documents):
    if not documents:
        return '<p class="evidence-empty">本次未检索项目原文。</p>'
    sections = ['<p class="evidence-scope">本次检索证据，不代表全部资料。</p>']
    for number, doc in enumerate(documents, 1):
        meta = doc.metadata
        links = [
            f'<a href="/context/{doc.id_}" target="_blank" rel="noopener">相邻原文</a>',
            f'<a href="/source/{meta["version"]}" target="_blank" rel="noopener">原件快照</a>',
        ]
        if meta["file_name"].lower().endswith(".pdf"):
            links.insert(
                0,
                f'<a href="/preview/{doc.id_}" target="_blank" rel="noopener">'
                '引用页图像</a>',
            )
        extra = " · ".join(str(meta[key]) for key in
                           ("date_basis", "version_label", "document_group")
                           if meta.get(key))
        approval = html.escape(str(meta.get("approval") or "未标注"))
        sections.append(
            '<section class="source-card">'
            f'<h3><span class="source-number">[{number}]</span> '
            f'{html.escape(meta["file_name"])}</h3>'
            f'<p class="source-location">{html.escape(short_location(meta) or meta["location"])}'
            f' · {html.escape(str(meta["date"]))}'
            f'<span class="source-date-basis">{html.escape(meta["date_basis"])}</span></p>'
            f'<p class="source-approval">批准状态：{approval}</p>'
            f'<nav aria-label="证据 {number} 原文入口">{"".join(links)}</nav>'
            '<details><summary>证据原文</summary>'
            f'<blockquote>{html.escape(doc.content)}</blockquote></details>'
            '<details class="source-metadata"><summary>版本与校验</summary>'
            f'<p>{html.escape(meta["location"])}</p>'
            f'<p>批准状态：{approval}</p>'
            f'<p>{html.escape(extra)}</p>'
            f'<small>SHA256: {html.escape(meta["sha256"])}</small></details>'
            '</section>'
        )
    return "".join(sections)


def cache_notice(task, result):
    stale = ("**资料已更新：这是历史回答快照，请重新查询后再使用。**\n\n"
             if result.get("stale") else "")
    if not task.get("cached_from"):
        return stale
    return (
        stale + "**复用已有回答**：原回答生成于 " + html.escape(result["created"])
        + "；该次查询的问题、上下文、范围及资料状态与原回答相同，未调用模型。"
        "如需重新查询，请在“任务与覆盖”取消复用选项。\n\n"
    )


def build_app(project_id):
    root = prepare_runtime()
    single_project_mode = os.environ.get("RFQ_SINGLE_PROJECT_MODE", "").casefold() in {
        "1", "true",
    }
    os.environ["THEFLOW_SETTINGS_MODULE"] = "preview_settings"
    os.environ["USE_LOW_LLM_REQUESTS"] = "true"
    from theflow.settings import settings
    import preview_settings

    if settings.KH_DATABASE != "sqlite://":
        raise RuntimeError("Unexpected upstream database settings.")
    sys.modules["flowsettings"] = preview_settings
    from preview_kotaemon import company_network_boundary, icon_directory

    sys.addaudithook(company_network_boundary())
    import gradio as gr
    import ktem
    from ktem.assets import KotaemonTheme
    from ktem.pages.chat import ChatPage, control
    from ktem.settings import BaseSettingGroup
    from project_workspace import ProjectWorkspace, render_answer, coverage_incomplete
    from project_answer import render_synthesis
    from kotaemon.base import Document
    from company_chat import configuration_status, probe_connection, CompanyChatError
    from query_jobs import QueryJobs
    from evidence_tools import version_choices, compare_versions
    from project_registry import projects, create_project, open_project, stop_project, authorized_source
    from project_backup import live_backup, available_backups, restore_project

    source = authorized_source()
    workspace = ProjectWorkspace(root, project_id, source)
    jobs = QueryJobs(workspace)
    workspace_label = (
        "Azure App Service 单项目 PoC"
        if single_project_mode else "本机工作区"
    )
    control.ASSETS_DIR = str(icon_directory())
    app = SimpleNamespace(
        app_name="工程项目知识助手",
        index_manager=SimpleNamespace(indices=[]),
        default_settings=SimpleNamespace(reasoning=BaseSettingGroup(
            settings={
                "use": {"name": "方法", "value": "simple", "choices": [("证据问答", "simple")]},
                "lang": {"name": "语言", "value": "zh", "choices": [("中文", "zh")]},
            },
            options={"simple": BaseSettingGroup(settings={
                "llm": {"name": "公司模型", "value": "", "choices": [("公司 API", "")]},
                "highlight_citation": {"name": "引用", "value": "off",
                                       "choices": [("原文位置", "off")]},
            })},
        )),
    )
    assets = Path(ktem.__file__).resolve().parent / "assets"
    css = (assets / "css" / "main.css").read_text(encoding="utf-8")
    css += "\n" + Path(__file__).with_name("project_ui.css").read_text(encoding="utf-8")
    theme = KotaemonTheme(
        primary_hue="blue",
        neutral_hue="slate",
        font=("Segoe UI", "Microsoft YaHei", "sans-serif"),
        font_mono=("Consolas", "monospace"),
    ).set(
        body_background_fill="#ffffff",
        background_fill_primary="#ffffff",
        background_fill_secondary="#f5f6f8",
        body_text_color="#202733",
        block_background_fill="#ffffff",
        block_label_background_fill="transparent",
        block_label_text_color="#626d7d",
        block_title_text_color="#202733",
        block_label_padding="0px",
        block_label_margin="0px",
        layout_gap="12px",
        input_background_fill="#ffffff",
        input_border_color="#dce1e8",
        input_border_color_focus="#00468e",
        input_placeholder_color="#5b6875",
        checkbox_background_color_selected="#285f9e",
        checkbox_border_color_selected="#285f9e",
        checkbox_label_background_fill="transparent",
        checkbox_label_background_fill_selected="#e7edf5",
        checkbox_label_text_color_selected="#202733",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#edf1f6",
        button_secondary_border_color="#dce1e8",
        button_secondary_text_color="#202733",
    )

    def protect(function):
        if inspect.isgeneratorfunction(function):
            @wraps(function)
            def generate(*args):
                try:
                    yield from function(*args)
                except Exception as error:
                    raise gr.Error(f"{type(error).__name__}: {error}") from None
            return generate
        @wraps(function)
        def call(*args):
            try:
                return function(*args)
            except Exception as error:
                # Explicit UI boundary: failed work never returns success-shaped output.
                raise gr.Error(f"{type(error).__name__}: {error}") from None
        return call

    with gr.Blocks(
        theme=theme, css=css, title=app.app_name, analytics_enabled=False,
        js=Path(__file__).with_name("project_ui.js").read_text(encoding="utf-8"),
    ) as demo:
        gr.HTML(
            '<a class="workspace-skip" href="#chat-input">跳到提问</a>'
            '<h1 class="workspace-brand">工程项目知识助手</h1>'
            f'<span class="workspace-subtitle">{html.escape(workspace_label)} · '
            f'{html.escape(project_id)}</span>',
            elem_id="workspace-header",
        )
        scan_state = gr.State("")
        result_state = gr.State("")
        job_state = gr.State("")
        conversation_state = gr.State("")
        with gr.Tabs(selected="chat", elem_id="workspace-tabs"):
            with gr.Tab("问答", id="chat", elem_id="workspace-chat"):
                page = ChatPage(app)
                page.conv_column.parent.elem_id = "workspace-chat-layout"
                for child in page.conv_column.children:
                    child.visible = False
                page.followup_questions_ui.visible = False
                page.chat_panel.chatbot.likeable = False
                page.chat_panel.chatbot.height = None
                source_accordion = next(child for child in page.info_column.children
                                        if isinstance(child, gr.Accordion))
                source_accordion.label = "原文出处"
                source_accordion.open = True
                page.info_column.visible = False
                page.chat_panel.chatbot.value = []
                page.chat_panel.chatbot.placeholder = (
                    '<div class="chat-welcome"><span class="welcome-mark" aria-hidden="true">'
                    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
                    'stroke="currentColor" stroke-width="1.5"><circle cx="10.5" cy="10.5" r="6.5"/>'
                    '<path d="m16 16 5 5"/></svg></span><h2>从项目资料开始</h2>'
                    '<p>查找要求、核对参数，或追溯原文出处。</p>'
                    '<p class="welcome-example">确认上方资料状态，选择下方示例填入问题，再发送。</p></div>'
                )
                page.chat_panel.text_input.placeholder = "询问项目要求…"
                page.chat_panel.text_input.label = "项目问题"
                page.info_panel.value = '<p class="evidence-empty">回答后在这里查看原文和出处。</p>'
                with page.conv_column:
                    new_conversation = gr.Button(
                        "新建对话", elem_id="new-conversation"
                    )
                    history_search = gr.Textbox(
                        placeholder="搜索对话…", label="搜索历史对话",
                        show_label=False, container=False, elem_id="history-search",
                        lines=1, max_lines=1,
                    )
                    gr.HTML('<h2>最近对话</h2>', elem_id="history-heading")
                    with gr.Column(elem_id="history-scroll", min_width=0):
                        history_list = gr.Radio(
                            choices=jobs.session_choices(), label="历史会话",
                            show_label=False, container=False, elem_id="history-list",
                        )
                    with gr.Accordion("任务与覆盖", open=False, elem_id="query-tools"):
                        reuse_answer = gr.Checkbox(
                            label="优先复用相同条件下的回答", value=True,
                            info="取消后重新检索并请求公司模型。资料或配置变化后不会复用旧回答。",
                        )
                        with gr.Accordion("任务与重试", open=False):
                            task_list = gr.Dropdown(choices=jobs.list(), label="本项目任务")
                            refresh_tasks = gr.Button("刷新任务列表", size="sm")
                            resume_query = gr.Button("继续所选任务／重试失败批次", size="sm")
                        with gr.Accordion("覆盖与解析限制", open=False):
                            query_coverage = gr.JSON(label="本次覆盖范围")
                    with gr.Accordion("使用说明", open=False, elem_id="query-help"):
                        gr.Markdown(
                            "问题与证据发送至公司API；文件解析和向量检索在本机完成。\n\n"
                            "**快速问答**选取相关证据，不保证穷尽。**主题检查**每文件最多检查8个相关片段。"
                            "**深度检查**逐批检查已解析文本，可能产生上千次API请求、耗时很长；"
                            "均不代表原件完整解析或零遗漏。\n\n"
                            "追问只沿用上一用户问题，不以历史AI回答为证据。"
                            "回答不代替设计判断，条件和例外以原文为准。"
                        )
                chat_column = page.chat_panel.chatbot.parent
                with chat_column:
                    page.chat_panel.chatbot.unrender()
                with page.chat_panel.text_input.parent:
                    page.chat_panel.text_input.unrender()
                for child in chat_column.children:
                    child.visible = False
                page.chat_panel.chatbot.visible = True
                page.chat_panel.text_input.visible = True
                with chat_column:
                    with gr.Row(elem_id="chat-toolbar"):
                        gr.HTML('<h2>项目问答</h2>', elem_id="chat-title")
                        history_toggle = gr.Checkbox(
                            label="历史对话", value=False, container=False,
                            elem_id="toggle-history", scale=0, min_width=90,
                        )
                        sources_toggle = gr.Checkbox(
                            label="原文出处", value=False, container=False,
                            elem_id="toggle-sources", scale=0, min_width=90,
                        )
                    workspace_context = gr.HTML(
                        project_context(project_id, workspace.coverage()),
                        elem_id="workspace-context",
                    )
                    page.chat_panel.chatbot.render()
                    with gr.Row(elem_id="question-examples") as question_examples:
                        example_buttons = [
                            gr.Button(label, size="sm", min_width=100)
                            for label, _question in QUESTION_EXAMPLES
                        ]
                    with gr.Column(elem_id="chat-composer", min_width=0):
                        page.chat_panel.text_input.render()
                        with gr.Row(elem_id="composer-options"):
                            mode = gr.Dropdown(
                                ["快速问答", "主题逐文件检查", "深度全文检查"],
                                value="快速问答", label="查询方式", show_label=False,
                                container=False, scale=0, min_width=156, elem_id="query-mode",
                            )
                            follow_up = gr.Checkbox(
                                label="续问上一问题", value=False, container=False,
                                min_width=138, elem_id="follow-up",
                            )
                            cancel_query = gr.Button(
                                "停止", min_width=52, scale=0, size="sm",
                                elem_id="cancel-query", interactive=False,
                            )
                            save_button = gr.Button(
                                "保存结果", min_width=76, scale=0, size="sm",
                                elem_id="save-result", interactive=False,
                            )
                    mode_help = gr.Markdown(
                        query_mode_help("快速问答", False), elem_id="query-mode-help",
                    )
                    query_status = gr.Markdown("就绪", elem_id="query-status")

            with gr.Tab("资料", elem_id="workspace-documents"):
                gr.Markdown(
                    "### 项目资料\n扫描后选择文件录入。原件不修改，内容变化保留版本。"
                )
                project_status = gr.Markdown(
                    value=workspace.status_summary(), elem_id="document-status",
                )
                with gr.Accordion("录入与解析说明", open=False):
                    gr.Markdown(
                        "扫描不调用AI；录入在本机创建快照、解析和生成向量。"
                        "相同内容不重复索引；Word/PDF副本不自动视为独立依据。"
                        "DOC／XLS／WPS使用本机Office转换；复杂表格和OCR请核对原件。"
                    )
                gr.Markdown("#### 1. 扫描授权目录")
                scan_button = gr.Button(
                    f"扫描“{source.name}”目录", elem_classes=["workspace-primary-action"],
                    elem_id="scan-documents",
                )
                scan_summary = gr.Textbox(
                    label="扫描状态", interactive=False,
                    value="尚未扫描。扫描只列出可选文件，不录入、不调用 AI。",
                )
                with gr.Accordion("查看扫描明细、重复、疑似版本对和失败", open=False):
                    scan_details = gr.JSON(label="扫描清单（只读）")
                gr.Markdown("#### 2. 选择要录入的文件")
                selection_status = gr.Markdown(
                    "请先扫描目录，再选择需要录入的文件。", elem_id="selection-status",
                )
                selected_files = gr.CheckboxGroup(
                    choices=[], label="可选文件", elem_id="selected-documents",
                )
                select_all = gr.Button(
                    "选择全部文件", interactive=False,
                    elem_classes=["workspace-secondary-action"], elem_id="select-all-documents",
                )
                gr.Markdown("#### 3. 录入并查看结果")
                ingest_button = gr.Button(
                    "录入所选文件", interactive=False,
                    elem_classes=["workspace-primary-action"], elem_id="ingest-documents",
                )
                ocr_enabled = gr.Checkbox(value=True, label="无文本PDF页使用本地OCR（数字、型号及表格须对照原页）")
                ingest_summary = gr.Markdown()
                with gr.Accordion("逐文件录入结果", open=False):
                    ingest_report = gr.JSON(label="处理明细")
                with gr.Accordion("入库版本与解析覆盖", open=False):
                    coverage_button = gr.Button("刷新已入库版本与解析覆盖")
                    coverage_view = gr.JSON(value=workspace.coverage(), label="版本与未覆盖内容")
                with gr.Accordion("日期、版本归组与停用资料", open=False):
                    version_select = gr.Dropdown(choices=version_choices(workspace), label="资料版本")
                    refresh_versions = gr.Button("刷新版本选择")
                    version_date = gr.Textbox(label="人工确认业务日期 YYYY-MM-DD（留空恢复源文件修改时间）")
                    version_group = gr.Textbox(label="同一资料组（相同标签将Word/PDF或不同版本归组，不自动推断同版）")
                    version_label = gr.Textbox(label="版本标签，例如 RFQ Rev.A / MOM 2026-10-19")
                    version_enabled = gr.Checkbox(value=True, label="纳入后续检索（取消即停用，不删除快照和旧结果）")
                    update_version_button = gr.Button("保存日期和版本设置")
                    metadata_status = gr.Textbox(label="设置状态", interactive=False)
            with gr.Tab("原文对照"):
                gr.Markdown(
                    "### 原文对照\n选择两个版本按关键词并列查看。差异不自动代表冲突或替代。"
                )
                with gr.Row():
                    compare_left = gr.Dropdown(choices=version_choices(workspace), label="左侧版本")
                    compare_right = gr.Dropdown(choices=version_choices(workspace), label="右侧版本")
                compare_topic = gr.Textbox(label="比较关键词（空格分词，例如 荷载 防腐）")
                compare_button = gr.Button("对照两边原文（本地，不调用AI）")
                compare_output = gr.Markdown()
                with gr.Accordion("对照范围及原始片段", open=False):
                    compare_details = gr.JSON()
            with gr.Tab("已保存"):
                saved_results = gr.Dropdown(
                    choices=workspace.result_list(), label="结果快照"
                )
                with gr.Row():
                    load_result = gr.Button("查看快照")
                    refresh_results = gr.Button("刷新列表")
                    export_button = gr.Button("导出 XLSX")
                saved_answer = gr.Markdown()
                with gr.Accordion("保存的原文引用及覆盖明细", open=False):
                    result_view = gr.JSON(label="结果快照")
                export_status = gr.Textbox(label="导出路径", interactive=False, lines=1)
                export_file = gr.File(label="下载文件", interactive=False, height=100)
                gr.Markdown(
                    "快照保留当时引用，资料更新后提示重查。导出创建新文件，不修改业务工作簿。"
                )
            with gr.Tab("设置与备份"):
                with gr.Accordion("公司AI连接", open=False):
                    ai_configuration = gr.Markdown(value=configuration_status())
                    check_ai = gr.Button("检查公司AI连接（不发送项目资料）")
                    connection_status = gr.Textbox(label="本次连接检查", interactive=False)
                    refresh_status = gr.Button("刷新配置与资料状态（不发网络请求）")
                with gr.Accordion("数据位置与使用边界", open=False):
                    gr.Markdown(
                        "数据位置：`" + str(workspace.root) + "`\n\n"
                        "仅绑定本项目授权目录，不跨项目检索。问题与证据仅发送指定公司API。"
                        "引用使用真实段落、单元格或PDF物理页，不编造页码。"
                        "复杂表格、公式及OCR可能不完整，请核对解析限制。\n\n"
                        "备份含数据库、源文件快照、扫描记录及结果，不含API密钥。"
                        "恢复只创建新项目，不覆盖现有数据。"
                    )
                gr.Markdown(
                    "### 备份与恢复\n创建当前工作区一致性备份，恢复时使用新项目ID。"
                )
                backup_button = gr.Button("立即创建并校验本项目备份")
                backup_status = gr.Textbox(label="备份状态与路径", interactive=False)
                backup_select = gr.Dropdown(choices=available_backups(root), label="已有备份")
                refresh_backups = gr.Button("刷新备份列表")
                restore_id = gr.Textbox(
                    label="恢复为新项目ID（小写字母／数字／短横线，现有项目不可覆盖）",
                    visible=not single_project_mode,
                )
                restore_button = gr.Button(
                    "校验并恢复到新项目", visible=not single_project_mode,
                )
                restore_status = gr.Textbox(label="恢复状态", interactive=False)
            with gr.Tab("项目", visible=not single_project_mode):
                gr.Markdown(
                    f"### 本机项目\n项目独立存储。当前仅授权“{source.name}”，新建项目不扩大资料授权。"
                )
                project_select = gr.Dropdown(choices=projects(root), value=project_id, label="选择本机项目")
                project_open = gr.Button("打开所选项目（本机独立入口）")
                project_stop = gr.Button("停止所选其他项目（保留数据）")
                project_link = gr.Markdown()
                new_project_id = gr.Textbox(label="新项目ID")
                create_project_button = gr.Button("创建空项目（仍绑定授权目录）")
                project_message = gr.Textbox(label="项目操作状态", interactive=False)
                with gr.Accordion("同事访问（尚未开通）", open=False):
                    gr.Markdown(
                        "现有内网入口没有身份认证，仅在IT批准后使用；不自动修改防火墙。"
                        "企业接入须由IT部署HTTPS与单点登录，保护页面、API和原件下载。"
                        "配置模板见 enterprise-access.conf，不代表已部署。"
                    )

        # Only our explicitly wired actions are enabled; upstream settings and uploads stay off.
        for component in demo.blocks.values():
            if isinstance(component, gr.components.Component) and hasattr(component, "interactive"):
                component.interactive = False
        for component in (
            page.chat_panel.text_input, mode, scan_button, selected_files,
            coverage_button, refresh_results, saved_results, load_result,
            export_button,
            check_ai, refresh_status, new_conversation, history_search, history_list,
            follow_up, history_toggle, sources_toggle, task_list,
            refresh_tasks, resume_query, ocr_enabled, reuse_answer,
            version_select, refresh_versions, version_date, version_group, version_label,
            version_enabled, update_version_button, compare_left, compare_right, compare_topic,
            compare_button, backup_button, backup_select, refresh_backups, restore_id,
            restore_button, project_select, project_open, project_stop, new_project_id, create_project_button,
            *example_buttons,
        ):
            component.interactive = True
        if single_project_mode:
            for component in (
                restore_id, restore_button, project_select, project_open, project_stop,
                new_project_id, create_project_button,
            ):
                component.interactive = False
        for button, (_label, question) in zip(example_buttons, QUESTION_EXAMPLES):
            button.click(
                lambda text=question: {"text": text, "files": []},
                [], [page.chat_panel.text_input], api_name=False, queue=False,
            )
        for component in (mode, follow_up):
            component.change(
                protect(query_mode_help), [mode, follow_up], [mode_help],
                api_name=False, queue=False,
            )
        sources_toggle.change(
            lambda show: gr.update(visible=show), [sources_toggle], [page.info_column],
            api_name=False, queue=False,
        )

        def refresh_context():
            return project_context(project_id, workspace.coverage())

        refresh_status.click(
            protect(lambda: (configuration_status(), workspace.status_summary())),
            [], [ai_configuration, project_status], api_name=False,
        ).then(protect(refresh_context), [], [workspace_context], api_name=False)

        def check_connection():
            try:
                return probe_connection()
            except CompanyChatError as error:
                return f"公司AI连接检查失败：{error} 没有切换到外部模型。"

        check_ai.click(
            check_connection, [], [connection_status], api_name=False,
            concurrency_id="project", concurrency_limit=1,
        )

        @protect
        def scan():
            result = workspace.scan()
            count = len(result["files"])
            failures = len(result["failures"])
            return (
                result["scan_id"],
                f"可读取文件{count}个，失败{failures}项。扫描不是入库，也不是聊天请求。",
                result,
                gr.update(choices=[row["path"] for row in result["files"]], value=[]),
                workspace.status_summary(),
                gr.update(interactive=bool(count)),
                ("请选择需要录入的文件。" if count else
                 "本次扫描没有可选文件；请查看扫描明细和失败原因。"),
                gr.update(interactive=False),
            )

        scan_button.click(
            scan, [], [scan_state, scan_summary, scan_details, selected_files, project_status,
                       select_all, selection_status, ingest_button],
            api_name=False, concurrency_id="project", concurrency_limit=1,
        )
        select_all.click(
            protect(lambda identity: [entry["path"] for entry in workspace._load_scan(identity)["files"]]),
            [scan_state], [selected_files], api_name=False,
        )
        def update_selection(selected):
            count = len(selected or [])
            return gr.update(interactive=bool(count)), (
                f"已选择 {count} 个文件，确认后点击“录入所选文件”。"
                if count else "尚未选择文件；请先扫描目录，再勾选需要录入的文件。"
            )
        selected_files.change(
            update_selection, [selected_files], [ingest_button, selection_status],
            api_name=False, queue=False,
        )

        @protect
        def ingest(scan_id, selected, use_ocr, progress=gr.Progress()):
            report = workspace.ingest(scan_id, selected, confirmed=True, progress=progress,
                                      use_ocr=use_ocr)
            failed = sum("error" in row for row in report)
            duplicates = sum("canonical_path" in row for row in report)
            summary = (
                f"**录入结束：成功{len(report) - failed - duplicates}份，"
                f"相同内容跳过{duplicates}份，失败{failed}份。** "
                "原件未修改；失败原因及解析限制见下方明细。"
            )
            return report, workspace.coverage(), summary, workspace.status_summary()

        ingest_button.click(
            ingest, [scan_state, selected_files, ocr_enabled],
            [ingest_report, coverage_view, ingest_summary, project_status],
            api_name=False, concurrency_id="project", concurrency_limit=1,
        ).then(protect(refresh_context), [], [workspace_context], api_name=False)
        coverage_button.click(
            protect(workspace.coverage),
            [], [coverage_view], api_name=False,
        ).then(protect(refresh_context), [], [workspace_context], api_name=False)

        def stream_job(identity_job, question, history):
            initial_task = jobs.get(identity_job)
            conversation_id = initial_task["session_id"]
            started = time.monotonic()
            initial_elapsed = initial_task["elapsed"]
            while True:
                task = jobs.get(identity_job)
                elapsed = max(task["elapsed"], initial_elapsed + time.monotonic() - started)
                status = (f"{task['status']} | {task['stage']} | "
                          f"{task['progress']:.0%} | 已处理 {elapsed:.1f} 秒")
                if task["status"] == "completed":
                    identity = task["result"]
                    saved = workspace.result(identity)
                    chosen = [Document(id_=entry["id"], content=entry["text"], metadata=entry["metadata"])
                              for entry in saved["evidence"]]
                    coverage = saved["coverage"]
                    intent_label = next((entry["intent"]["label"] for entry in coverage
                                         if entry.get("intent")), "")
                    synthesis = json.loads(saved["answer"]) if saved["answer"] else None
                    answer = render_synthesis(synthesis, chosen) if synthesis else ""
                    cited_ids = {cite["id"] for section in synthesis["sections"]
                                 for point in section["points"] for cite in point["citations"]} if synthesis else None
                    yield (
                        {"text": "", "files": []},
                        (history or [])[-9:] + [(question, cache_notice(task, saved) + ((
                            answer + "\n\n" + render_answer(
                                chosen, coverage, compact=True, cited_ids=cited_ids
                            )
                        ) if answer else render_answer(chosen, coverage)))],
                        source_panel(chosen), identity,
                        (intent_label + " · " if intent_label else "") + status
                        + ("；覆盖范围与解析限制见清单。" if chosen else ""),
                        coverage, identity_job, conversation_id,
                        gr.update(choices=jobs.session_choices(), value=conversation_id),
                        gr.update(interactive=False),
                    )
                    return
                if task["status"] in {"failed", "cancelled", "interrupted"}:
                    yield ({"text": question, "files": []}, history or [],
                           "<p>未返回成功回答；可在任务区继续或重试。</p>", "",
                           status + "\n" + task["error"], [], identity_job,
                           conversation_id,
                           gr.update(choices=jobs.session_choices(), value=conversation_id),
                           gr.update(interactive=False))
                    return
                yield ({"text": "", "files": []}, history or [], "<p>正在查询，完成后显示原文。</p>",
                       "", status, [], identity_job, conversation_id,
                       gr.update(choices=jobs.session_choices(), value=conversation_id),
                       gr.update(interactive=True))
                time.sleep(0.4)

        @protect
        def respond(message, history, scope, use_context, conversation_id, reuse):
            if not isinstance(message, dict) or message.get("files"):
                raise ValueError("不接受聊天附件；请在资料页扫描并录入文件。")
            question = message.get("text", "")
            if use_context and not history:
                raise ValueError("没有上一问题，请取消追问选项或先完成一次查询。")
            context = history[-1][0] if use_context else ""
            identity_job = jobs.start(
                question, scope, context, conversation_id or None, reuse_answer=reuse
            )
            yield from stream_job(identity_job, question, history)

        query_outputs = [
            page.chat_panel.text_input, page.chat_panel.chatbot, page.info_panel,
            result_state, query_status, query_coverage, job_state,
            conversation_state, history_list,
            cancel_query,
        ]

        control_outputs = [
            cancel_query, save_button, page.chat_panel.text_input, new_conversation,
            history_list, mode, follow_up, reuse_answer, resume_query, question_examples,
        ]

        def prepare_query():
            return [gr.update(interactive=False) for _ in control_outputs[:-1]] + [
                gr.update(visible=False)
            ]

        def finish_query(identity, history):
            return [
                gr.update(interactive=False),
                gr.update(interactive=bool(identity)),
                *[gr.update(interactive=True) for _ in control_outputs[2:-1]],
                gr.update(visible=not bool(history)),
            ]

        page.chat_panel.text_input.submit(
            prepare_query, [], control_outputs, api_name=False, queue=False,
        ).then(
            respond, [page.chat_panel.text_input, page.chat_panel.chatbot, mode, follow_up,
                     conversation_state, reuse_answer],
            query_outputs,
            api_name=False, concurrency_id="project", concurrency_limit=1,
        ).then(
            finish_query, [result_state, page.chat_panel.chatbot], control_outputs,
            api_name=False, queue=False,
        )
        cancel_query.click(protect(jobs.cancel), [job_state], [query_status],
                           api_name=False, queue=False)
        refresh_tasks.click(protect(lambda: gr.update(choices=jobs.list())),
                            [], [task_list], api_name=False, queue=False)

        @protect
        def load_conversation(identity):
            rows = jobs.session_jobs(identity)
            history = []
            last_result = ""
            last_job = ""
            last_sources = '<p class="evidence-empty">本次未检索项目原文。</p>'
            last_coverage = []
            last_mode = "快速问答"
            for task in rows:
                last_job = task["id"]
                last_mode = task["mode"]
                if task["status"] == "completed" and task["result"]:
                    saved = workspace.result(task["result"])
                    documents = [
                        Document(content=entry["text"], id_=entry["id"],
                                 metadata=entry["metadata"])
                        for entry in saved["evidence"]
                    ]
                    synthesis = json.loads(saved["answer"]) if saved["answer"] else None
                    answer = render_synthesis(synthesis, documents) if synthesis else ""
                    cited_ids = (
                        {cite["id"] for section in synthesis["sections"]
                         for point in section["points"] for cite in point["citations"]}
                        if synthesis else None
                    )
                    answer = (answer + "\n\n" + render_answer(
                        documents, saved["coverage"], compact=True, cited_ids=cited_ids
                    )) if answer else render_answer(documents, saved["coverage"])
                    history.append((task["question"], cache_notice(task, saved) + answer))
                    last_sources = source_panel(documents)
                    last_coverage = saved["coverage"]
                    last_result = task["result"]
                else:
                    message = f"查询状态：{task['status']}"
                    if task["error"]:
                        message += f"\n\n{task['error']}"
                    history.append((task["question"], message))
                    last_sources = '<p class="evidence-empty">该查询没有可展示的原文证据。</p>'
                    last_coverage = []
                    last_result = ""
            return (
                identity, history, last_sources, last_result,
                f"已打开历史会话，共 {len(rows)} 次查询。",
                last_coverage, last_job, gr.update(value=last_mode), False,
            )

        history_list.select(
            load_conversation, [history_list],
            [conversation_state, page.chat_panel.chatbot, page.info_panel, result_state,
             query_status, query_coverage, job_state, mode, history_toggle],
            api_name=False,
        ).then(
            finish_query, [result_state, page.chat_panel.chatbot], control_outputs,
            api_name=False, queue=False,
        )
        new_conversation.click(
            lambda: (
                "", [], '<p class="evidence-empty">提出问题后，原文出处会显示在这里。</p>',
                "", "新对话", [], "", gr.update(value=None), "快速问答",
                False, False, {"text": "", "files": []},
            ),
            [], [conversation_state, page.chat_panel.chatbot, page.info_panel, result_state,
                 query_status, query_coverage, job_state, history_list, mode,
                 history_toggle, follow_up, page.chat_panel.text_input],
            api_name=False, queue=False,
        ).then(
            finish_query, [result_state, page.chat_panel.chatbot], control_outputs,
            api_name=False, queue=False,
        )
        history_search.input(
            protect(lambda term: gr.update(choices=jobs.session_choices(term))),
            [history_search], [history_list], api_name=False, queue=False,
        )

        @protect
        def resume(identity_job, history):
            task = jobs.get(identity_job)
            jobs.resume(identity_job)
            yield from stream_job(identity_job, task["question"], history)
        resume_query.click(
            prepare_query, [], control_outputs, api_name=False, queue=False,
        ).then(
            resume, [task_list, page.chat_panel.chatbot], query_outputs,
            api_name=False, concurrency_id="project", concurrency_limit=1,
        ).then(
            finish_query, [result_state, page.chat_panel.chatbot], control_outputs,
            api_name=False, queue=False,
        )

        @protect
        def save(identity):
            workspace.save_result(identity)
            return "结果已保存，可查看或导出。", gr.update(
                choices=workspace.result_list(), value=identity
            ), workspace.status_summary()

        save_button.click(
            save, [result_state], [query_status, saved_results, project_status], api_name=False
        )
        refresh_results.click(
            protect(lambda: gr.update(choices=workspace.result_list())),
            [], [saved_results], api_name=False,
        )
        def saved_result(identity):
            result = workspace.result(identity)
            result.pop("reviews", None)
            documents = [
                Document(content=entry["text"], id_=entry["id"], metadata=entry["metadata"])
                for entry in result["evidence"]
            ]
            synthesis = json.loads(result["answer"]) if result.get("answer") else None
            answer = render_synthesis(synthesis, documents) if synthesis else ""
            cited_ids = {cite["id"] for section in synthesis["sections"]
                         for point in section["points"] for cite in point["citations"]} if synthesis else None
            return result, (
                ("**资料已更新，建议重新查询；以下保留当时回答。**\n\n" if result["stale"] else "")
                + answer + "\n\n" + render_answer(
                    documents, result["coverage"], compact=bool(synthesis), cited_ids=cited_ids
                )
            )

        load_result.click(
            protect(saved_result), [saved_results], [result_view, saved_answer], api_name=False
        )
        @protect
        def export(identity):
            path = str(workspace.export(identity))
            return path, path

        export_button.click(
            export,
            [saved_results], [export_status, export_file], api_name=False,
        )
        def refresh_version_choices():
            choices = version_choices(workspace)
            return [gr.update(choices=choices) for _ in range(3)]
        refresh_versions.click(protect(refresh_version_choices), [],
                               [version_select, compare_left, compare_right], api_name=False)

        @protect
        def load_metadata(identity):
            row = next((row for row in workspace.coverage() if row["id"] == identity), None)
            if row is None:
                raise ValueError("请选择当前项目的版本。")
            return (row["business_date"] if row["date_basis"] == "人工确认业务日期" else "",
                    row["document_group"], row["version_label"], bool(row["enabled"]))
        version_select.change(load_metadata, [version_select],
                              [version_date, version_group, version_label, version_enabled],
                              api_name=False)

        @protect
        def update_metadata(identity, value, group, label, enabled):
            workspace.update_version(identity, value, group, label, enabled)
            return "已保存；旧结果不改写，已标记需重新查询。", workspace.coverage(), workspace.status_summary()
        update_version_button.click(
            update_metadata, [version_select, version_date, version_group, version_label, version_enabled],
            [metadata_status, coverage_view, project_status], api_name=False,
            concurrency_id="project", concurrency_limit=1,
        ).then(protect(refresh_context), [], [workspace_context], api_name=False)
        compare_button.click(
            protect(lambda left, right, topic: compare_versions(workspace, left, right, topic)),
            [compare_left, compare_right, compare_topic], [compare_output, compare_details],
            api_name=False, concurrency_id="project", concurrency_limit=1,
        )

        @protect
        def create_backup():
            path = live_backup(workspace)
            return f"备份已创建并通过哈希、数据库和快照校验：{path}", gr.update(
                choices=available_backups(root), value=path.name
            )
        backup_button.click(create_backup, [], [backup_status, backup_select],
                            api_name=False, concurrency_id="project", concurrency_limit=1)
        refresh_backups.click(protect(lambda: gr.update(choices=available_backups(root))),
                              [], [backup_select], api_name=False)

        @protect
        def restore_backup(name, identity):
            if single_project_mode:
                raise ValueError("Azure 单项目 PoC 不支持恢复或创建其他项目。")
            target = restore_project(root, name, identity, source)
            return f"已恢复并校验：{target}。现有项目未修改。", gr.update(
                choices=projects(root), value=identity)
        restore_button.click(restore_backup, [backup_select, restore_id],
                             [restore_status, project_select], api_name=False,
                             concurrency_id="project", concurrency_limit=1)

        @protect
        def new_project(identity):
            if single_project_mode:
                raise ValueError("Azure 单项目 PoC 不支持创建其他项目。")
            value = create_project(root, identity)
            return f"已创建空项目{value}；未自动扫描或录入。", gr.update(
                choices=projects(root), value=value
            )
        create_project_button.click(new_project, [new_project_id], [project_message, project_select],
                                    api_name=False, concurrency_id="project", concurrency_limit=1)

        @protect
        def switch_project(identity):
            if single_project_mode:
                raise ValueError("Azure 单项目 PoC 不支持切换项目。")
            if identity == project_id:
                return "当前已在此项目。"
            address = open_project(root, identity)
            return f"[打开独立项目 {html.escape(identity)}]({address})。本机入口不能直接分享给其他电脑。"
        project_open.click(switch_project, [project_select], [project_link], api_name=False,
                           concurrency_id="project", concurrency_limit=1)

        def stop_other_project(identity):
            if single_project_mode:
                raise ValueError("Azure 单项目 PoC 不支持停止其他项目。")
            return stop_project(root, identity, project_id)

        project_stop.click(
            protect(stop_other_project),
            [project_select], [project_message], api_name=False,
            concurrency_id="project", concurrency_limit=1,
        )
    return demo, workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("PORT", os.environ.get("WEBSITES_PORT", "7863"))),
    )
    parser.add_argument("--project", default=os.environ.get("RFQ_PROJECT_ID", "project-demo"))
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--lan-ip", help="Explicit RFC1918 IPv4 address for company-LAN access.")
    parser.add_argument(
        "--public-origin",
        default=os.environ.get("RFQ_PUBLIC_ORIGIN"),
        help="IT-approved HTTPS application origin.",
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("端口必须在1024至65535之间。")
    from network_access import validate_lan_ipv4, validate_public_origin

    app_service_mode = os.environ.get("RFQ_APP_SERVICE", "").casefold() in {"1", "true"}
    app_service_access_mode = os.environ.get("RFQ_APP_SERVICE_ACCESS_MODE", "entra").strip().casefold()
    if args.lan_ip:
        args.lan_ip = validate_lan_ipv4(args.lan_ip)
    if app_service_mode:
        if app_service_access_mode not in {"entra", "company-ip"}:
            raise ValueError("RFQ_APP_SERVICE_ACCESS_MODE must be entra or company-ip.")
        single_project_mode = os.environ.get("RFQ_SINGLE_PROJECT_MODE", "").casefold() in {
            "1", "true",
        }
        from project_registry import SOURCE_ENV, authorized_source

        if not single_project_mode:
            raise ValueError("Azure App Service must use RFQ_SINGLE_PROJECT_MODE=true.")
        if not os.environ.get(SOURCE_ENV):
            raise ValueError(f"Azure App Service requires an explicit {SOURCE_ENV} path.")
        source = authorized_source()
        if not source.is_absolute() or not source.is_dir():
            raise ValueError(f"{SOURCE_ENV} must be an existing absolute directory.")
        if args.public_origin is None:
            hostname = os.environ.get("WEBSITE_HOSTNAME")
            if hostname:
                args.public_origin = f"https://{hostname}"
    if args.public_origin:
        args.public_origin = validate_public_origin(args.public_origin)
        if args.lan_ip:
            raise ValueError("企业代理模式只允许本机监听，不能同时启用未认证内网监听。")
    if app_service_mode and not args.public_origin:
        raise ValueError("Azure App Service requires RFQ_PUBLIC_ORIGIN or WEBSITE_HOSTNAME.")
    if app_service_mode and args.lan_ip:
        raise ValueError("Azure App Service does not use unauthenticated LAN mode.")
    listen_host = "0.0.0.0" if app_service_mode else (args.lan_ip or "127.0.0.1")
    # Fail before expensive imports; never terminate another listener.
    import socket

    with socket.socket() as check:
        try:
            check.bind((listen_host, args.port))
        except OSError:
            raise SystemExit(
                f"端口{args.port}已占用。请访问现有服务，或用 --port 指定空闲端口。"
            ) from None
    # Acquire the service lease before migrations and task-recovery logic run.
    import re
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", args.project):
        raise ValueError("项目ID无效。")
    project_root = prepare_runtime() / "projects" / args.project
    project_root.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_file_lock(
            project_root / "service.lock",
            "此项目已有服务运行，拒绝第二个写入进程。",
        ):
            demo, workspace = build_app(args.project)
            (workspace.root / "local-service.json").write_text(
                json.dumps({"port": args.port, "pid": os.getpid()}), encoding="utf-8"
            )
            serve(
                demo, workspace, args.port, args.open_browser, args.lan_ip,
                args.public_origin, app_service_mode, app_service_access_mode,
            )
    except FileLockError as error:
        raise SystemExit(str(error)) from None


def serve(
    demo, workspace, port, open_browser=False, lan_ip=None, public_origin=None,
    app_service_mode=False, app_service_access_mode="entra",
):
    import gradio as gr
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, Response, HTMLResponse
    from preview_kotaemon import icon_directory
    from network_access import allowed_hosts, request_within_boundary, WebSocketBoundaryMiddleware
    import uvicorn

    server = FastAPI()
    hosts = allowed_hosts(lan_ip)
    if app_service_mode and app_service_access_mode not in {"entra", "company-ip"}:
        raise ValueError("Invalid App Service access mode.")
    app_service_ip_restricted = app_service_mode and app_service_access_mode == "company-ip"
    server.add_middleware(
        WebSocketBoundaryMiddleware, port=port, hosts=hosts, public_origin=public_origin,
        app_service_mode=app_service_mode,
        app_service_ip_restricted=app_service_ip_restricted,
    )
    listen_host = lan_ip or "127.0.0.1"
    access_url = (
        public_origin if app_service_mode
        else f"http://{lan_ip or '127.0.0.1'}:{port}"
    )
    if open_browser:
        @server.on_event("startup")
        async def open_local_browser():
            import threading
            import webbrowser

            threading.Timer(1.0, webbrowser.open, args=(access_url,)).start()

    @server.get("/project-health")
    def health():
        idle = workspace.lock.acquire(blocking=False)
        if idle:
            workspace.lock.release()
        return {
            "app": "rfq-kotaemon-project", "project": workspace.project_id,
            "access_mode": ("azure-app-service-company-ip" if app_service_ip_restricted
                            else "azure-app-service") if app_service_mode else (
                "enterprise-proxy" if public_origin else (
                    "company-lan" if lan_ip else "loopback")),
            "access_url": access_url,
            "pid": os.getpid(),
            "business_busy": not idle,
        }

    @server.get("/source/{version_id}")
    def source_snapshot(version_id: str):
        try:
            path = workspace.source_snapshot(version_id)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        return FileResponse(
            path, filename=path.name,
            media_type="application/pdf" if path.suffix.lower() == ".pdf" else None,
            content_disposition_type="inline" if path.suffix.lower() == ".pdf" else "attachment",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @server.get("/preview/{chunk_id}")
    def pdf_preview(chunk_id: str):
        try:
            image = workspace.pdf_evidence_preview(chunk_id)
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        return Response(image, media_type="image/png", headers={"Cache-Control": "no-store"})

    @server.get("/context/{chunk_id}")
    def source_context(chunk_id: str):
        from evidence_tools import evidence_context

        try:
            markup = evidence_context(workspace, chunk_id)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        return HTMLResponse(
            "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<title>引用上下文</title><body>" + markup + "</body></html>",
            headers={"Cache-Control": "no-store"},
        )

    @server.middleware("http")
    async def boundary(request, call_next):
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        allowed = request_within_boundary(
            host=host,
            origin=origin,
            client_ip=request.client.host if request.client else "",
            port=port,
            hosts=hosts,
            public_origin=public_origin,
            app_service_mode=app_service_mode,
            principal_header=request.headers.get("x-ms-client-principal"),
            app_service_ip_restricted=app_service_ip_restricted,
        )
        if not allowed:
            response = JSONResponse({"detail": "Local same-origin access required."}, 403)
        elif request.url.path.rstrip("/") == "/upload":
            response = JSONResponse({"detail": "Use explicit scan and ingestion."}, 405)
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
        server, demo, path="/",
        root_path="" if app_service_mode else (public_origin or ""),
        allowed_paths=[str(icon_directory()), str(workspace.root / "exports")]
    )
    print(f"Project service starting: {access_url}", flush=True)
    uvicorn.run(
        server, host="0.0.0.0" if app_service_mode else listen_host, port=port,
        log_level="info", proxy_headers=False,
    )


if __name__ == "__main__":
    main()
