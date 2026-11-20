"""Source-only neighbourhood views and cautious, deterministic version comparisons."""

from difflib import SequenceMatcher
import html
import re


def version_choices(workspace):
    return [(f"{row['path']} | {row['business_date']} | "
             f"{row['version_label'] or row['sha'][:8]}"
             + (" [已停用]" if not row["enabled"] else ""), row["id"])
            for row in workspace.coverage()]


def evidence_context(workspace, chunk_id):
    with workspace.connection() as conn:
        chunk = conn.execute("SELECT rowid,* FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if chunk is None:
            raise ValueError("引用不属于当前项目。")
        workspace.source_snapshot(chunk["version"])
        siblings = conn.execute(
            "SELECT id,content,location,rowid FROM chunks WHERE version=? AND "
            "substr(id,1,32)=? ORDER BY rowid",
            (chunk["version"], chunk_id[:32]),
        ).fetchall()
    index = next(i for i, row in enumerate(siblings) if row["id"] == chunk_id)
    picked = list(siblings[max(0, index - 3):index + 4])
    # Include all fragments at a row/table location as well as its neighbours.
    location = chunk["location"].split("；字符")[0]
    known = {row["id"] for row in picked}
    picked.extend(row for row in siblings if row["id"] not in known
                  and row["location"].split("；字符")[0] == location)
    picked.sort(key=lambda row: row["rowid"])
    page = re.search(r"PDF物理第(\d+)页", chunk["location"])
    page_link = (
        f"<p><a href='/source/{chunk['version']}#page={page[1]}'>"
        f"打开PDF物理第{page[1]}页（非印刷页码）</a> · "
        f"<a href='/preview/{chunk_id}'>查看引用页图像</a></p>"
        if page else
        "<p>Word使用段落／表格位置，Excel使用工作表和单元格；"
        "未建立固定排版页码映射，不编造页码。</p>"
    )
    return "<h2>引用及相邻原文</h2>" + page_link + \
        "<p>前后各最多三片段，并补齐同位置拆分片段；" \
        "不推断表头、单位或合并语义。OCR文字可能有误。</p>" \
        f"<p><a href='/source/{chunk['version']}'>打开原件快照核对完整结构</a></p>" + "".join(
            f"<h3>{html.escape(row['location'])}</h3>"
            f"<pre style='white-space:pre-wrap'>{html.escape(row['content'])}</pre>"
            for row in picked
        )


def compare_versions(workspace, left_id, right_id, topic):
    if left_id == right_id:
        raise ValueError("请选择两个不同版本。")
    versions = {row["id"]: row for row in workspace.coverage()}
    if left_id not in versions or right_id not in versions:
        raise ValueError("版本不属于当前项目。")
    if not isinstance(topic, str) or not topic.strip() or len(topic) > 160:
        raise ValueError("请输入1至160字符的比较关键词；多个词可用空格分开。")
    terms = [term.casefold() for term in topic.split() if term.strip()]
    def load(identity):
        workspace.source_snapshot(identity)
        with workspace.connection() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT id,content,location FROM chunks WHERE version=? AND active=1 ORDER BY rowid",
                (identity,),
            )]
        return rows, [row for row in rows if any(term in row["content"].casefold() for term in terms)]
    all_left, left = load(left_id)
    all_right, right = load(right_id)
    left_meta, right_meta = versions[left_id], versions[right_id]
    same_group = bool(left_meta["document_group"] and
                      left_meta["document_group"] == right_meta["document_group"])
    # Bound display work but disclose exactly what was not displayed.
    limit = 50
    compared_left, compared_right = left[:limit], right[:limit]
    used = set()
    pairs = []
    for old in compared_left:
        matches = [(SequenceMatcher(None, old["content"], new["content"], autojunk=False).ratio(), index)
                   for index, new in enumerate(compared_right) if index not in used]
        score, index = max(matches, default=(0, -1))
        new = compared_right[index] if score >= 0.35 else None
        if new:
            used.add(index)
        label = ("原文相同" if new and old["content"] == new["content"] else
                 "原文差异：需核对数值、条件和适用对象" if new else
                 "仅左侧检索到：不能据此判断已删除")
        pairs.append({"difference": label, "left": old, "right": new})
    pairs.extend({"difference": "仅右侧检索到：可能新增，也可能表述不同",
                  "left": None, "right": row}
                 for index, row in enumerate(compared_right) if index not in used)
    def heading(meta):
        return f"{meta['path']} | {meta['version_label']} | {meta['business_date']} | {meta['date_basis']}"
    lines = [
        "### 原文版本对照",
        "**左侧：** " + html.escape(heading(left_meta)),
        "**右侧：** " + html.escape(heading(right_meta)),
        ("已人工归入同一资料组。" if same_group else "尚未归入同一资料组，可能是不同适用资料。")
        + "左右不代表效力高低；同日不排序。文字差异不自动等于工程冲突，原文待定条件保持不变。",
        f"关键词命中：左{len(left)}/{len(all_left)}片段，右{len(right)}/{len(all_right)}片段；"
        f"本页各显示最多{limit}片段，未显示左{max(0,len(left)-limit)}、右{max(0,len(right)-limit)}。",
    ]
    for pair in pairs:
        lines.append("#### " + pair["difference"])
        for label, key in (("左侧", "left"), ("右侧", "right")):
            row = pair[key]
            lines.append(
                f"**{label}** " + (html.escape(row["location"]) + "\n\n"
                + html.escape(row["content"]) + f"\n\n[查看上下文](/context/{row['id']})"
                if row else "当前关键词范围内无可配对片段。")
            )
    if not pairs:
        lines.append("当前关键词范围内未找到可对照片段；请换用原文术语，不代表无此要求。")
    return "\n\n".join(lines), {"left": left_meta, "right": right_meta, "pairs": pairs,
                              "left_matches": len(left), "right_matches": len(right)}
