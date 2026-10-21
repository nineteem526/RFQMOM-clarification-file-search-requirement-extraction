import io
import json
import sqlite3
import zipfile
from hashlib import sha256

import streamlit as st
from openai import APIError

from samples import SAMPLES
from workbench import (
    Store, WorkbenchError, connection_test, context_payload, default_database,
    extract, make_client, safe_api_error,
)


st.set_page_config(page_title="RFQ 要求工作台", page_icon="📑", layout="wide")
st.markdown("""
<style>
.stApp { background: #f1f5f8; color: #22394c; }
h1, h2, h3 { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }
h1 { font-size: 2rem !important; color: #153d60; }
[data-testid="stSidebar"] { background: #e3edf3; }
[data-testid="stMainBlockContainer"] { max-width: 1280px; }
div[data-testid="stExpander"] { background: #fff; border-left: 4px solid #36769c; }
.boundary { border-left: 5px solid #bc6b12; padding: 12px 16px;
background: #fff4dd; color: #604019; margin-bottom: 18px; }
button:focus-visible { outline: 3px solid #bc6b12 !important; outline-offset: 2px; }
</style>
""", unsafe_allow_html=True)


def sample_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for sample in SAMPLES:
            archive.writestr(sample.filename, sample.text.encode("utf-8"))
    return buffer.getvalue()


def run() -> None:
    st.title("RFQ 要求工作台")
    st.markdown(
        '<div class="boundary">虚构项目演示 · 真实文件导入与发送已禁用。'
        'AI 提取是审核草稿，不是工程批准或标准合规结论。</div>',
        unsafe_allow_html=True,
    )
    store = Store(default_database())
    try:
        render(store)
    finally:
        store.close()


def render(store: Store) -> None:
    with st.sidebar:
        st.header("项目文件")
        st.caption("示例项目 DEMO；全部版本并列保留，不自动认定替代关系。")
        if st.button("录入 RFQ / B12 / 澄清示例", width="stretch"):
            imported = sum(store.import_sample(sample) for sample in SAMPLES if sample.initial)
            st.success(f"新增 {imported} 个版本；重复版本未再次录入。")
        available = [sample for sample in SAMPLES if not sample.initial]
        choice = st.selectbox("后续文件版本", available, format_func=lambda item: item.filename)
        if st.button("录入所选 MOM 示例", width="stretch"):
            if store.import_sample(choice):
                st.success("新版本已录入，旧版本保留。")
            else:
                st.info("该版本已存在，未重复录入。")
        st.download_button("下载虚构源文件", sample_zip(), "rfq-demo-sources.zip", "application/zip")
        st.divider()
        st.caption("仅支持内置 TXT 示例。PDF、扫描件、Word、Excel 等真实文件解析待授权后开发。")
        st.caption(f"本机演示索引：{default_database()}")

    docs = store.documents()
    search_tab, files_tab, api_tab = st.tabs(["检索与提取", "文件与版本", "内部 AI 接入"])
    with files_tab:
        if not docs:
            st.info("从左侧录入虚构 RFQ、B12 和澄清文件开始。")
        else:
            st.dataframe(
                [{key: doc[key] for key in ("filename", "kind", "stage", "version", "date")}
                 for doc in docs],
                width="stretch", hide_index=True,
                column_config={"filename": "文件", "kind": "类型", "stage": "阶段",
                               "version": "版本", "date": "文件日期"},
            )
            chosen = st.selectbox("查看完整原文", docs, format_func=lambda doc: doc["filename"])
            sample = next(
                (s for s in SAMPLES if s.document_key == chosen["document_key"]
                 and s.version == chosen["version"]), None,
            )
            if sample:
                st.code(sample.text, language=None)
            else:
                st.error("未知文档；请检查演示索引，禁止发送。")
    with api_tab:
        st.subheader("仅连接公司内部代理")
        st.write("Azure Chat Completions · api-key 认证 · 环境变量读取 · 无外部回退")
        st.caption("在启动工具的终端按公司指南配置 DUERR_API_KEY；不要在页面或聊天中粘贴密钥。")
        st.write("最小聊天测试发送固定英文指令，不发送文件；会产生一次 API 用量。")
        if st.button("发送最小聊天测试"):
            try:
                with st.spinner("等待公司接口响应……"):
                    with make_client() as client:
                        connection_test(client)
                st.success("收到精确响应 DUERR_API_TEST_OK。此结果不代表要求提取能力已验证。")
            except (WorkbenchError, APIError) as error:
                st.error(safe_api_error(error))
        st.warning("尚未录入可核查的标准原文。国标 / ISO / 美标建议在此版禁用。")
    with search_tab:
        st.subheader("从描述找到要求与出处")
        st.caption("本地关键词检索；支持内置同义词，不是语义向量检索。空格分隔词按任一命中。")
        left, right = st.columns([3, 2])
        with left:
            query = st.text_input("词条或关键词", placeholder="例如：钢构、电气控制、荷载、PLC")
        with right:
            stages = st.multiselect("阶段（不选表示全部）", ["RFQ", "TE", "DF"])
        expand = st.checkbox("扩展内置同义词与相关术语", value=True)
        results = store.search(query, tuple(stages), expand)
        if not docs:
            st.info("尚无文件。请先在左侧录入虚构示例。")
        elif not query.strip():
            st.info("输入关键词，查找各文件中的相关描述。")
        elif not results:
            st.warning("没有命中。可更换关键词、取消阶段筛选或启用术语扩展；不代表项目没有此要求。")
        else:
            st.caption(f"命中 {len(results)} 个片段；原文优先，日期不决定文件优先级。")
            for item in results:
                with st.expander(
                    f"{item.filename} ｜ {item.locator} ｜ 出处 #{item.source_id}",
                    expanded=len(results) <= 4,
                ):
                    st.caption(f"{item.kind} / {item.stage} / {item.version} / {item.date}")
                    st.text(item.text)
            selected_ids = st.multiselect(
                "选择发送给公司 AI 的虚构片段",
                [item.source_id for item in results],
                format_func=lambda source_id: next(
                    f"#{item.source_id} {item.filename} {item.locator}"
                    for item in results if item.source_id == source_id
                ),
            )
            selected = [item for item in results if item.source_id in selected_ids]
            if selected:
                try:
                    store.verify_demo()
                    payload = context_payload(selected)
                except WorkbenchError as error:
                    st.error(str(error))
                    return
                st.caption(f"将发送 {len(selected)} 个虚构片段，含出处元数据，共 {len(payload)} 字符。")
                with st.expander("查看将发送的完整业务载荷", expanded=True):
                    st.code(payload, language="json")
                approved = st.checkbox(
                    "确认只发送以上虚构片段，允许产生公司 API 用量",
                    key="send_approved_" + sha256(payload.encode("utf-8")).hexdigest(),
                )
                if st.button("由内部 AI 提取要求", type="primary", disabled=not approved):
                    st.session_state.pop("analysis", None)
                    try:
                        with st.spinner("提取要求并校验原文证据……"):
                            with make_client() as client:
                                result = extract(client, selected)
                        st.session_state["analysis"] = {
                            "result": result, "sources": selected, "query": query,
                            "signature": payload,
                        }
                    except (WorkbenchError, APIError) as error:
                        st.error(safe_api_error(error))
            analysis = st.session_state.get("analysis")
            if analysis:
                current_ids = {item.source_id for item in selected}
                saved_ids = {item.source_id for item in analysis["sources"]}
                if (
                    current_ids == saved_ids and query == analysis["query"]
                    and context_payload(selected) == analysis["signature"]
                ):
                    render_analysis(analysis)
                else:
                    st.info("片段或查询已改变；上一次 AI 结果不属于当前选择，已隐藏。")


def render_analysis(analysis: dict) -> None:
    st.subheader("要求审核草稿")
    st.warning("引用已通过逐字原文校验；描述的正确性、遗漏及冲突判断仍需人工审核。")
    result = analysis["result"]
    sources = {item.source_id: item for item in analysis["sources"]}
    names = {"requirement": "要求", "clarification": "待确认", "conflict": "差异 / 冲突"}
    if not result["items"]:
        st.info("AI 未提取出有依据的条目，请查看局限性说明。")
    for item in result["items"]:
        st.write(f"**{names[item['type']]}**")
        st.text(item["description"])
        for ref in item["evidence"]:
            source = sources[ref["source_id"]]
            st.caption(f"#{source.source_id} {source.filename} · {source.locator}")
            st.text(ref["quote"])
        st.divider()
    st.write("**覆盖范围与局限性**")
    for limitation in result["limitations"]:
        st.text(limitation)
    export = {
        "notice": "虚构示例审核草稿，不可用于工程设计；须人工确认。",
        "query": analysis["query"],
        "result": result,
        "sources": [item.__dict__ for item in analysis["sources"]],
    }
    st.download_button(
        "下载含出处的提取结果",
        json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
        "demo-requirements.json", "application/json",
    )


if __name__ == "__main__":
    try:
        run()
    except (WorkbenchError, OSError, sqlite3.Error) as error:
        st.error(f"演示索引操作失败：{error}")
