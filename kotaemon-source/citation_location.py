"""Compact, source-native citation labels; never infer pagination."""

import re


def short_location(metadata):
    location = metadata.get("location", "")
    page = re.search(r"PDF物理第(\d+)页", location)
    if page:
        return f"PDF第{page[1]}页"
    paragraph = re.search(r"正文第(\d+)段", location)
    converted = "本地转换副本" in location
    if paragraph:
        return ("转换稿" if converted else "") + f"第{paragraph[1]}段"
    table = re.search(r"正文表(\d+)，第(\d+)行", location)
    if table:
        return ("转换稿" if converted else "") + f"表{table[1]}行{table[2]}"
    sheet = re.search(r"工作表 (.+?)(?:；字符|$)", location)
    if sheet:
        return sheet[1]
    return ""


def citation_link(position, document):
    locator = short_location(document.metadata)
    label = str(position) + (" · " + locator if locator else "")
    # Metadata can contain Markdown punctuation, including sheet names.
    label = re.sub(r"([\\`*{}\[\]<>()!|_])", r"\\\1", label).replace("\n", " ")
    return f"[{label}](/context/{document.id_})"
