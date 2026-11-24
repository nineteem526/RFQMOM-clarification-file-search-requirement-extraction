"""Specialized multi-modal document parser and structure-aware chunker.

Supports:
- PDF text with clause/heading boundary splitting & repeating header/footer suppression
- PDF native tables (PyMuPDF find_tables) with header-preserving row-group chunking & body deduplication
- Scanned/low-text PDF pages & large embedded PDF images via offline local_ocr
- Standalone images (.png/.jpg/.jpeg/.bmp/.webp/.tif/.tiff) via offline local_ocr
- Word (.docx / macro-enabled OOXML .docx / .docm) including merged-cell tables & textboxes
- Excel (.xlsx / .xlsm) with smart header detection, merged header propagation, & header-preserving row chunks
- HTML (.html / .htm) separating structured tables and prose
- Legacy Office (.doc / .xls / .wps) via magic-byte sniffing or isolated office_convert snapshot conversion
"""

from contextlib import closing
from dataclasses import dataclass
import csv
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from rag_engine import split_text_with_overlap

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})

_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,4}\s+\S|"
    r"\d+(?:\.\d+){1,4}\s+\S|"
    r"\d+\.\s+\S|"
    r"\d+\s+[^\d\s].+|"
    r"第[一二三四五六七八九十百千0-9]+[章节条款部分](?:[\s：:].*)?|"
    r"[一二三四五六七八九十]+[、.]\s*\S|"
    r"(?:SECTION|CHAPTER|ANNEX|APPENDIX|PART)\s+[A-Z0-9]+\b.*"
    r")$"
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；.!?\n])")
_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass
class ParsedDocument:
    chunks: list[dict]
    pages: int
    warnings: list[str]


def bind_table_row(header: list[str], row: list[str]) -> str:
    """Bind non-empty cells in a row to their column headers without shifting columns."""
    parts = []
    for column, value in enumerate(row):
        clean_val = str(value).strip() if value is not None else ""
        if clean_val:
            hdr = str(header[column]).strip() if column < len(header) and header[column] is not None else ""
            name = hdr or f"列{column + 1}"
            parts.append(f"{name}: {clean_val}" if name != clean_val else clean_val)
    return " | ".join(parts)


def detect_table_header(rows: list[list[str]]) -> tuple[list[str], list[str], list[list[str]]]:
    """Identify optional title banners, the true header row, and data rows."""
    cleaned = [[_WHITESPACE_RE.sub(" ", str(c or "")).strip() for c in r] for r in rows]
    non_empty = [r for r in cleaned if any(r)]
    if not non_empty:
        return [], [], []

    inspect_n = min(5, len(non_empty))
    counts = [sum(1 for c in r if c) for r in non_empty[:inspect_n]]

    if len(non_empty) <= 2 and counts[0] >= 2:
        return [], non_empty[0], non_empty[1:]

    best_idx = 0
    if counts[0] == 1 and any(c >= 2 for c in counts[1:]):
        max_cols = max(counts)
        best_idx = next(i for i, c in enumerate(counts) if c == max_cols)

    banners = [
        " | ".join(c for c in r if c)
        for r in non_empty[:best_idx]
        if any(r)
    ]
    return banners, non_empty[best_idx], non_empty[best_idx + 1 :]


def chunk_table_rows(
    title: str,
    header: list[str],
    data_rows: list[list[str]],
    page_no: int,
    section_prefix: str,
    chunk_size: int = 480,
    banner_lines: list[str] | None = None,
    header_tag: str = "[表头]",
) -> list[dict]:
    """Chunk table rows cleanly while repeating column headers in every chunk."""
    size = max(120, min(int(chunk_size or 480), 1200))
    prefix_parts: list[str] = []
    if title and title.strip():
        prefix_parts.append(f"[表题] {title.strip()}")
    if banner_lines:
        prefix_parts.extend(f"[说明] {line}" for line in banner_lines if line and line.strip())

    clean_header = [str(c).strip() for c in header if c is not None and str(c).strip()]
    header_line = f"{header_tag} " + " | ".join(clean_header) if clean_header else header_tag
    prefix_parts.append(header_line)
    header_prefix = "\n".join(prefix_parts)

    bound_rows = [line for row in data_rows if (line := bind_table_row(header, row))]
    if not bound_rows:
        if not clean_header and not banner_lines:
            return []
        return [{
            "page": page_no,
            "section": section_prefix,
            "content": header_prefix,
            "chunk_type": "table",
        }]

    groups: list[list[str]] = []
    cur: list[str] = []
    cur_len = len(header_prefix)
    single_row_budget = max(80, size - len(header_prefix) - 2)

    for line in bound_rows:
        if len(header_prefix) + 1 + len(line) > size:
            if cur:
                groups.append(cur)
                cur = []
                cur_len = len(header_prefix)
            sub_parts = split_text_with_overlap(line, chunk_size=single_row_budget, chunk_overlap=min(32, single_row_budget // 4))
            for part in sub_parts:
                groups.append([part])
            continue

        if cur and cur_len + 1 + len(line) > size:
            groups.append(cur)
            last_row = cur[-1]
            if len(header_prefix) + 2 + len(last_row) + len(line) <= size:
                cur = [last_row, line]
                cur_len = len(header_prefix) + 2 + len(last_row) + len(line)
            else:
                cur = [line]
                cur_len = len(header_prefix) + 1 + len(line)
        else:
            cur.append(line)
            cur_len += 1 + len(line)

    if cur:
        groups.append(cur)

    total = len(groups)
    chunks: list[dict] = []
    for idx, grp in enumerate(groups, 1):
        label = section_prefix if total == 1 else f"{section_prefix} ({idx}/{total})"
        chunks.append({
            "page": page_no,
            "section": label,
            "content": header_prefix + "\n" + "\n".join(grp),
            "chunk_type": "table",
        })
    return chunks


def group_ocr_lines(lines: list[dict]) -> str:
    """Reconstruct reading order and column spacing from OCR line boxes."""
    valid = [item for item in (lines or []) if str(item.get("text", "")).strip()]
    if not valid:
        return ""

    boxed: list[tuple[float, float, float, float, str]] = []
    unboxed: list[str] = []
    for item in valid:
        txt = str(item["text"]).strip()
        box = item.get("box")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                boxed.append(((y0 + y1) / 2.0, x0, x1, max(1.0, y1 - y0), txt))
                continue
            except (TypeError, ValueError, IndexError):
                pass
        unboxed.append(txt)

    if not boxed:
        return "\n".join(unboxed)

    heights = sorted(entry[3] for entry in boxed)
    med_h = heights[len(heights) // 2] if heights else 12.0
    row_tol = max(6.0, 0.6 * med_h)

    boxed.sort(key=lambda item: (item[0], item[1]))
    visual_rows: list[list[tuple[float, float, float, float, str]]] = []
    row_ys: list[float] = []
    for entry in boxed:
        if visual_rows and abs(entry[0] - row_ys[-1]) <= row_tol:
            visual_rows[-1].append(entry)
            row_ys[-1] = sum(e[0] for e in visual_rows[-1]) / len(visual_rows[-1])
        else:
            visual_rows.append([entry])
            row_ys.append(entry[0])

    rendered: list[str] = []
    for row in visual_rows:
        row.sort(key=lambda e: e[1])
        parts = [row[0][4]]
        for prev, cur in zip(row, row[1:]):
            gap = cur[1] - prev[2]
            parts.append((" | " if gap > 2.2 * med_h else " ") + cur[4])
        rendered.append("".join(parts))

    out_pieces = [rendered[0]]
    for idx in range(1, len(rendered)):
        vgap = row_ys[idx] - row_ys[idx - 1]
        out_pieces.append(("\n\n" if vgap > 1.6 * med_h else "\n") + rendered[idx])

    if unboxed:
        out_pieces.append("\n" + "\n".join(unboxed))
    return "".join(out_pieces)


def _match_heading(line: str) -> str:
    stripped = line.strip()
    if not stripped or len(stripped) > 85:
        return ""
    if stripped.endswith(("。", "，", ";", "；", ",")) and not re.match(r"^\d+(?:\.\d+)+", stripped):
        return ""
    if _HEADING_RE.match(stripped):
        return stripped
    return ""


def _split_long_paragraph(para: str, size: int, overlap: int) -> list[str]:
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
    if len(sentences) <= 1:
        return split_text_with_overlap(para, chunk_size=size, chunk_overlap=overlap)

    pieces: list[str] = []
    buf = ""
    for sent in sentences:
        if len(sent) > size:
            if buf:
                pieces.append(buf.strip())
                buf = ""
            pieces.extend(split_text_with_overlap(sent, chunk_size=size, chunk_overlap=overlap))
            continue
        candidate = (buf + sent) if buf else sent
        if len(candidate) <= size:
            buf = candidate
        else:
            pieces.append(buf.strip())
            tail = buf[-overlap:].strip() if overlap > 0 else ""
            if tail and len(tail) + len(sent) + 1 <= size:
                buf = tail + "\n" + sent
            else:
                buf = sent
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def chunk_structured_text(
    text: str,
    page_no: int,
    section_prefix: str,
    chunk_size: int = 480,
    chunk_overlap: int = 64,
    active_heading: str = "",
    chunk_type: str = "text",
) -> tuple[list[dict], str]:
    """Split prose along headings & sentence boundaries while preserving short text verbatim."""
    clean = re.sub(r"\r\n?", "\n", text or "").strip()
    if not clean:
        return [], active_heading

    size = max(120, min(int(chunk_size or 480), 1200))
    overlap = max(0, min(int(chunk_overlap or 64), size // 2))

    # Update active heading from lines even for short pages so the next page inherits context
    latest_heading = active_heading
    for ln in clean.splitlines():
        h = _match_heading(ln)
        if h:
            latest_heading = h

    if len(clean) <= size:
        return [{
            "page": page_no,
            "section": section_prefix,
            "content": clean,
            "chunk_type": chunk_type,
        }], latest_heading

    # Pre-break blocks just before heading lines so headings stay at the start of chunks
    raw_lines = clean.splitlines()
    blocks: list[tuple[str, str]] = []  # (block_text, heading_if_starts_with_heading)
    cur_lines: list[str] = []
    cur_heading = ""
    for ln in raw_lines:
        if not ln.strip():
            if cur_lines:
                blocks.append(("\n".join(cur_lines).strip(), cur_heading))
                cur_lines = []
                cur_heading = ""
            continue
        h = _match_heading(ln)
        if h:
            if cur_lines:
                blocks.append(("\n".join(cur_lines).strip(), cur_heading))
            cur_lines = [ln.strip()]
            cur_heading = h
        else:
            cur_lines.append(ln.strip())
    if cur_lines:
        blocks.append(("\n".join(cur_lines).strip(), cur_heading))

    chunks_raw: list[tuple[str, str]] = []  # (content, heading)
    buf = ""
    buf_heading = active_heading
    running_heading = active_heading

    for block_text, block_heading in blocks:
        if block_heading:
            if buf and len(buf) >= int(size * 0.35):
                chunks_raw.append((buf, buf_heading))
                buf = ""
            running_heading = block_heading
            if not buf:
                buf_heading = running_heading

        if len(block_text) <= size:
            sep = "\n\n" if buf else ""
            if len(buf) + len(sep) + len(block_text) <= size:
                if not buf:
                    buf_heading = running_heading
                buf = buf + sep + block_text
                continue
            if buf:
                chunks_raw.append((buf, buf_heading))
                tail = buf[-overlap:].strip() if overlap > 0 and not block_heading else ""
                if tail and len(tail) + len(block_text) + 1 <= size:
                    buf = tail + "\n" + block_text
                    buf_heading = running_heading
                    continue
            buf = block_text
            buf_heading = running_heading
        else:
            if buf:
                chunks_raw.append((buf, buf_heading))
                buf = ""
            for piece in _split_long_paragraph(block_text, size, overlap):
                chunks_raw.append((piece, running_heading))
            buf_heading = running_heading

    if buf:
        chunks_raw.append((buf, buf_heading))

    total = len(chunks_raw)
    out: list[dict] = []
    for idx, (content, heading) in enumerate(chunks_raw, 1):
        body = content
        if idx > 1 and heading and heading not in content[:100] and not content.startswith("[章节]"):
            tag = f"[章节] {heading}\n"
            if len(tag) + len(body) <= size + 80:
                body = tag + body
        base_sec = section_prefix
        if heading and len(heading) <= 32 and heading not in base_sec:
            base_sec = f"{section_prefix} · {heading}"
        label = base_sec if total == 1 else f"{base_sec} ({idx}/{total})"
        out.append({
            "page": page_no,
            "section": label,
            "content": body,
            "chunk_type": chunk_type,
        })
    return out, running_heading


def _normalize_boilerplate_line(line: str) -> str:
    return re.sub(r"\d+", "#", _WHITESPACE_RE.sub(" ", line.strip()))


def _detect_pdf_boilerplate(doc) -> set[str]:
    """Detect repeating short top/bottom margin lines across multi-page PDFs."""
    n_pages = len(doc)
    if n_pages < 4:
        return set()
    counts: dict[str, int] = {}
    sample_pages = min(n_pages, 40)
    for idx in range(sample_pages):
        try:
            text = doc[idx].get_text("text") or ""
        except Exception:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        candidates = {lines[0], lines[-1]}
        seen_on_page: set[str] = set()
        for cand in candidates:
            if len(cand) <= 80:
                norm = _normalize_boilerplate_line(cand)
                if norm and norm != "#" and norm not in seen_on_page:
                    seen_on_page.add(norm)
                    counts[norm] = counts.get(norm, 0) + 1
    threshold = max(3, int(sample_pages * 0.6))
    return {norm for norm, cnt in counts.items() if cnt >= threshold}


def _rect_intersection_ratio(block_rect, table_rects: list) -> float:
    b_area = max(1.0, abs((block_rect[2] - block_rect[0]) * (block_rect[3] - block_rect[1])))
    covered = 0.0
    for tr in table_rects:
        ix0 = max(block_rect[0], tr[0])
        iy0 = max(block_rect[1], tr[1])
        ix1 = min(block_rect[2], tr[2])
        iy1 = min(block_rect[3], tr[3])
        if ix1 > ix0 and iy1 > iy0:
            covered += (ix1 - ix0) * (iy1 - iy0)
    return min(1.0, covered / b_area)


def _strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    if not boilerplate:
        return text
    kept = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if stripped and len(stripped) <= 80 and _normalize_boilerplate_line(stripped) in boilerplate:
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _decode_text_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_html_content(
    html_text: str,
    chunk_size: int,
    chunk_overlap: int,
    warnings: list[str],
) -> list[dict]:
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    chunks: list[dict] = []
    t_idx = 0
    for table in list(soup.find_all("table")):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [_WHITESPACE_RE.sub(" ", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(cells)
        if len(rows) >= 2:
            t_idx += 1
            banners, hdr, data_rows = detect_table_header(rows)
            chunks.extend(
                chunk_table_rows(
                    title="",
                    header=hdr,
                    data_rows=data_rows,
                    page_no=1,
                    section_prefix=f"HTML 表格 {t_idx}",
                    chunk_size=chunk_size,
                    banner_lines=banners,
                    header_tag=f"[HTML表格 {t_idx}]",
                )
            )
            table.decompose()

    text_blocks: list[str] = []
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = _WHITESPACE_RE.sub(" ", elem.get_text(" ", strip=True))
        if txt:
            text_blocks.append(txt)
    body_text = "\n\n".join(text_blocks) if text_blocks else _WHITESPACE_RE.sub(" ", soup.get_text("\n", strip=True))
    if body_text.strip():
        text_chunks, _ = chunk_structured_text(
            body_text,
            page_no=1,
            section_prefix="HTML 正文",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(text_chunks)

    if not chunks and not warnings:
        warnings.append("HTML 页面未提取到有效正文或表格。")
    return chunks


def _extract_word_xml_chunks(
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[dict], int]:
    """Parse OOXML word/document.xml directly (supports macro-enabled & textbox content)."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    w_ns = f"{{{ns['w']}}}"

    with zipfile.ZipFile(path) as zf:
        if "word/document.xml" not in zf.namelist():
            return [], 0
        root = ET.fromstring(zf.read("word/document.xml"))

    body = root.find("w:body", ns)
    if body is None:
        return [], 0

    paragraphs: list[str] = []
    chunks: list[dict] = []
    t_idx = 0

    for child in body:
        tag = child.tag
        if tag == f"{w_ns}p":
            txt = "".join(t.text or "" for t in child.findall(".//w:t", ns)).strip()
            if txt:
                paragraphs.append(txt)
        elif tag == f"{w_ns}tbl":
            rows: list[list[str]] = []
            for tr in child.findall("./w:tr", ns):
                r_vals = []
                for tc in tr.findall("./w:tc", ns):
                    c_txt = " ".join(
                        "".join(t.text or "" for t in p.findall(".//w:t", ns)).strip()
                        for p in tc.findall(".//w:p", ns)
                    ).strip()
                    r_vals.append(_WHITESPACE_RE.sub(" ", c_txt))
                if any(r_vals):
                    rows.append(r_vals)
            if rows:
                t_idx += 1
                banners, hdr, data_rows = detect_table_header(rows)
                chunks.extend(
                    chunk_table_rows(
                        title="",
                        header=hdr,
                        data_rows=data_rows,
                        page_no=t_idx + 1,
                        section_prefix=f"表格 {t_idx}",
                        chunk_size=chunk_size,
                        banner_lines=banners,
                        header_tag=f"[表格 {t_idx}]",
                    )
                )

    # Textboxes not already captured in top-level paragraphs
    txbx_texts: list[str] = []
    para_set = set(paragraphs)
    for txbx in root.findall(".//w:txbxContent", ns):
        box_lines = [
            "".join(t.text or "" for t in p.findall(".//w:t", ns)).strip()
            for p in txbx.findall(".//w:p", ns)
        ]
        joined = "\n".join(line for line in box_lines if line)
        if joined and joined not in para_set:
            txbx_texts.append(joined)

    full_text = "\n\n".join(paragraphs + txbx_texts).strip()
    text_chunks, _ = chunk_structured_text(
        full_text,
        page_no=1,
        section_prefix="Word 正文",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    all_chunks = text_chunks + chunks
    non_ws_len = len(re.sub(r"\s+", "", "".join(c["content"] for c in all_chunks)))
    return all_chunks, non_ws_len


def _parse_word_file(
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr_enabled: bool,
    warnings: list[str],
) -> list[dict]:
    chunks: list[dict] = []
    non_ws_len = 0

    used_xml_fallback = False
    try:
        import docx  # type: ignore

        document = docx.Document(str(path))
        paras = [p.text.strip() for p in document.paragraphs if p.text.strip()]

        # Extract any textbox text from word/document.xml
        try:
            with zipfile.ZipFile(path) as zf:
                if "word/document.xml" in zf.namelist():
                    root = ET.fromstring(zf.read("word/document.xml"))
                    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                    para_set = set(paras)
                    for txbx in root.findall(f".//{{{w_ns}}}txbxContent"):
                        box_lines = [
                            "".join(t.text or "" for t in p.findall(f".//{{{w_ns}}}t")).strip()
                            for p in txbx.findall(f".//{{{w_ns}}}p")
                        ]
                        joined = "\n".join(l for l in box_lines if l)
                        if joined and joined not in para_set:
                            paras.append(joined)
        except Exception:
            pass

        full_text = "\n\n".join(paras).strip()
        if full_text:
            t_chunks, _ = chunk_structured_text(
                full_text,
                page_no=1,
                section_prefix="Word 正文",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks.extend(t_chunks)

        for t_idx, table in enumerate(document.tables, 1):
            t_rows = []
            for row in table.rows:
                vals: list[str] = []
                seen_tc = set()
                for cell in row.cells:
                    if cell._tc in seen_tc:
                        vals.append("")
                    else:
                        seen_tc.add(cell._tc)
                        vals.append(_WHITESPACE_RE.sub(" ", cell.text.strip().replace("\n", " ")))
                if any(vals):
                    t_rows.append(vals)
            if t_rows:
                banners, hdr, data_rows = detect_table_header(t_rows)
                chunks.extend(
                    chunk_table_rows(
                        title="",
                        header=hdr,
                        data_rows=data_rows,
                        page_no=t_idx + 1,
                        section_prefix=f"表格 {t_idx}",
                        chunk_size=chunk_size,
                        banner_lines=banners,
                        header_tag=f"[表格 {t_idx}]",
                    )
                )
        non_ws_len = len(re.sub(r"\s+", "", "".join(c["content"] for c in chunks)))
    except ValueError:
        used_xml_fallback = True
    except Exception as exc:
        warnings.append(f"Word 解析中断（{type(exc).__name__}），尝试 XML 解析。")
        used_xml_fallback = True

    if used_xml_fallback:
        try:
            chunks, non_ws_len = _extract_word_xml_chunks(path, chunk_size, chunk_overlap)
        except Exception as exc:
            warnings.append(f"Word 解析失败（{type(exc).__name__}），请核对原件。")
            return []

    # If the Word doc is mainly embedded images (scanned pages pasted into Word), run offline OCR
    if non_ws_len < 30 and ocr_enabled:
        try:
            import local_ocr

            if local_ocr.is_ocr_available():
                with zipfile.ZipFile(path) as zf:
                    media_names = sorted(
                        n for n in zf.namelist()
                        if n.startswith("word/media/") and Path(n).suffix.lower() in IMAGE_SUFFIXES
                    )[:5]
                    if media_names:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            for idx, m_name in enumerate(media_names, 1):
                                img_path = Path(tmp_dir) / f"media_{idx}{Path(m_name).suffix.lower()}"
                                img_path.write_bytes(zf.read(m_name))
                                pages_res = local_ocr.recognize_image(img_path)
                                for pg in pages_res:
                                    ocr_txt = group_ocr_lines(pg.get("lines") or [])
                                    if ocr_txt.strip():
                                        ocr_chunks, _ = chunk_structured_text(
                                            "[Word插图OCR，请核对原件]\n" + ocr_txt,
                                            page_no=idx,
                                            section_prefix=f"Word 插图OCR {idx}",
                                            chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap,
                                            chunk_type="ocr",
                                        )
                                        chunks.extend(ocr_chunks)
                        if any(c.get("chunk_type") == "ocr" for c in chunks):
                            warnings.append("Word 文档文字较少，已对内嵌图片执行本地 OCR，请核对原件。")
        except Exception:
            pass

    return chunks


def _parse_excel_file(
    path: Path,
    chunk_size: int,
    warnings: list[str],
) -> list[dict]:
    import openpyxl  # type: ignore

    chunks: list[dict] = []
    # Check sheet sizes in read_only mode first to keep memory bounded
    use_read_only = False
    with closing(openpyxl.load_workbook(path, data_only=True, read_only=True)) as ro_wb:
        total_cells = sum((ws.max_row or 0) * (ws.max_column or 0) for ws in ro_wb.worksheets)
        if total_cells > 200_000:
            use_read_only = True
            for s_idx, sheet_name in enumerate(ro_wb.sheetnames, 1):
                matrix = [
                    ["" if cell is None else str(cell).strip() for cell in row]
                    for row in ro_wb[sheet_name].iter_rows(values_only=True)
                ]
                banners, hdr, data_rows = detect_table_header(matrix)
                if not hdr:
                    continue
                chunks.extend(
                    chunk_table_rows(
                        title=f"工作表 {sheet_name}",
                        header=hdr,
                        data_rows=data_rows,
                        page_no=s_idx,
                        section_prefix=f"工作表 {sheet_name}",
                        chunk_size=chunk_size,
                        banner_lines=banners,
                        header_tag="[表头]",
                    )
                )

    if not use_read_only:
        with closing(openpyxl.load_workbook(path, data_only=True, read_only=False)) as wb:
            for s_idx, sheet_name in enumerate(wb.sheetnames, 1):
                ws = wb[sheet_name]
                matrix = [
                    ["" if cell is None else str(cell).strip() for cell in row]
                    for row in ws.iter_rows(values_only=True)
                ]
                if not matrix:
                    continue
                # Propagate merged cell anchors in header rows (rows 1..5) or leading category columns (cols 1..3)
                for rng in ws.merged_cells.ranges:
                    val = matrix[rng.min_row - 1][rng.min_col - 1] if (rng.min_row - 1 < len(matrix) and rng.min_col - 1 < len(matrix[rng.min_row - 1])) else ""
                    if not val:
                        continue
                    is_header_merge = rng.max_row <= 5
                    is_category_merge = rng.max_col <= 3 and (rng.max_row - rng.min_row) <= 25
                    if is_header_merge or is_category_merge:
                        for r_i in range(rng.min_row - 1, min(rng.max_row, len(matrix))):
                            for c_i in range(rng.min_col - 1, min(rng.max_col, len(matrix[r_i]))):
                                if not matrix[r_i][c_i]:
                                    matrix[r_i][c_i] = val
                banners, hdr, data_rows = detect_table_header(matrix)
                if not hdr:
                    continue
                chunks.extend(
                    chunk_table_rows(
                        title=f"工作表 {sheet_name}",
                        header=hdr,
                        data_rows=data_rows,
                        page_no=s_idx,
                        section_prefix=f"工作表 {sheet_name}",
                        chunk_size=chunk_size,
                        banner_lines=banners,
                        header_tag="[表头]",
                    )
                )
    return chunks


def _parse_pdf_file(
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr_enabled: bool,
    warnings: list[str],
) -> tuple[list[dict], int]:
    chunks: list[dict] = []
    pages = 0

    try:
        import fitz  # type: ignore
        import local_ocr

        with fitz.open(path) as doc:
            if not doc.is_pdf or doc.needs_pass:
                raise ValueError("PDF 格式无效或已加密。")
            pages = len(doc)
            boilerplate = _detect_pdf_boilerplate(doc)
            active_heading = ""
            ocr_missing_pages: list[int] = []
            embedded_ocr_calls = 0

            for idx, page in enumerate(doc, 1):
                page_chunks: list[dict] = []
                table_rects: list[tuple[float, float, float, float]] = []

                try:
                    found_tables = page.find_tables().tables
                except Exception as exc:
                    found_tables = []
                    warnings.append(f"第 {idx} 页表格解析失败（{type(exc).__name__}），请核对原件。")

                for t_idx, table in enumerate(found_tables, 1):
                    try:
                        raw_rows = table.extract() or []
                    except Exception:
                        continue
                    banners, hdr, data_rows = detect_table_header(raw_rows)
                    if not hdr or (not data_rows and len([c for c in hdr if c]) < 2):
                        continue
                    bbox = getattr(table, "bbox", None)
                    if bbox and len(bbox) == 4:
                        table_rects.append(tuple(float(v) for v in bbox))
                    t_chunks = chunk_table_rows(
                        title="",
                        header=hdr,
                        data_rows=data_rows,
                        page_no=idx,
                        section_prefix=f"第 {idx} 页 · 表格 {t_idx}",
                        chunk_size=chunk_size,
                        banner_lines=banners,
                        header_tag=f"[表格 {t_idx}]",
                    )
                    page_chunks.extend(t_chunks)

                # Extract reading-order text blocks excluding cells already emitted as structured tables
                raw_blocks = page.get_text("blocks", sort=True) or []
                body_parts: list[str] = []
                for block in raw_blocks:
                    if len(block) > 6 and block[6] == 0 and str(block[4]).strip():
                        if table_rects and _rect_intersection_ratio(block[:4], table_rects) >= 0.70:
                            continue
                        cleaned_block = _WHITESPACE_RE.sub(" ", str(block[4]).strip())
                        cleaned_block = _strip_boilerplate(cleaned_block, boilerplate)
                        if cleaned_block:
                            body_parts.append(cleaned_block)

                body_text = "\n\n".join(body_parts).strip()
                if not body_text and not page_chunks:
                    fallback_txt = _strip_boilerplate((page.get_text("text") or "").strip(), boilerplate)
                    body_text = fallback_txt

                non_ws_len = len(re.sub(r"\s+", "", body_text))

                # Inspect raster image coverage to detect scanned pages (even with noisy page numbers)
                page_area = max(1.0, float(page.rect.width * page.rect.height))
                max_img_ratio = 0.0
                large_img_rects: list[tuple[float, float, float, float]] = []
                try:
                    for info in page.get_image_info(xrefs=False) or []:
                        bbox = info.get("bbox")
                        w_px = int(info.get("width") or 0)
                        h_px = int(info.get("height") or 0)
                        if not bbox or len(bbox) != 4:
                            continue
                        bw = max(0.0, float(bbox[2]) - float(bbox[0]))
                        bh = max(0.0, float(bbox[3]) - float(bbox[1]))
                        ratio = (bw * bh) / page_area
                        if ratio > max_img_ratio:
                            max_img_ratio = ratio
                        if ratio >= 0.25 and w_px >= 180 and h_px >= 180:
                            large_img_rects.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                except Exception:
                    pass

                should_full_ocr = ocr_enabled and (
                    (not body_text and not page_chunks)
                    or (non_ws_len < 30 and not page_chunks and max_img_ratio >= 0.15)
                )

                if should_full_ocr:
                    try:
                        lines = local_ocr.recognize_page(path, idx)
                        ocr_text = group_ocr_lines(lines)
                        if ocr_text.strip():
                            ocr_chunks, active_heading = chunk_structured_text(
                                "[OCR识别，请核对原页]\n" + ocr_text,
                                page_no=idx,
                                section_prefix=f"第 {idx} 页 · OCR",
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                active_heading=active_heading,
                                chunk_type="ocr",
                            )
                            page_chunks.extend(ocr_chunks)
                            warnings.append(f"第 {idx} 页使用本地 OCR，引用需核对原页。")
                    except ValueError as exc:
                        if "本地OCR环境未安装" in str(exc):
                            ocr_missing_pages.append(idx)
                        else:
                            warnings.append(f"第 {idx} 页 OCR 失败（ValueError）：{exc}")
                    except OSError as exc:
                        warnings.append(f"第 {idx} 页 OCR 失败（{type(exc).__name__}）：{exc}")

                if body_text:
                    txt_chunks, active_heading = chunk_structured_text(
                        body_text,
                        page_no=idx,
                        section_prefix=f"第 {idx} 页",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        active_heading=active_heading,
                        chunk_type="text",
                    )
                    page_chunks.extend(txt_chunks)

                # Extract embedded charts/tables as images inside text-rich pages when local OCR is ready
                if (
                    ocr_enabled
                    and not should_full_ocr
                    and non_ws_len >= 30
                    and large_img_rects
                    and embedded_ocr_calls < 12
                    and local_ocr.is_ocr_available()
                ):
                    norm_body = re.sub(r"\s+", "", body_text)
                    for img_i, rect in enumerate(large_img_rects[:2], 1):
                        if embedded_ocr_calls >= 12:
                            break
                        embedded_ocr_calls += 1
                        try:
                            img_lines = local_ocr.recognize_pdf_region(path, idx, rect)
                            img_txt = group_ocr_lines(img_lines)
                            norm_img = re.sub(r"\s+", "", img_txt)
                            if len(norm_img) >= 10 and norm_img not in norm_body:
                                region_chunks, _ = chunk_structured_text(
                                    "[图片OCR，请核对原页]\n" + img_txt,
                                    page_no=idx,
                                    section_prefix=f"第 {idx} 页 · 插图OCR {img_i}",
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    chunk_type="ocr",
                                )
                                page_chunks.extend(region_chunks)
                        except (ValueError, OSError):
                            pass

                if not page_chunks and idx not in ocr_missing_pages:
                    warnings.append(f"第 {idx} 页没有可索引文本。")
                chunks.extend(page_chunks)

            if ocr_missing_pages:
                sample = "、".join(str(p) for p in ocr_missing_pages[:10])
                more = f" 等 {len(ocr_missing_pages)} 页" if len(ocr_missing_pages) > 10 else ""
                warnings.append(f"第 {sample}{more} 为扫描/图片页，但本地 OCR 环境未安装；请先安装本地 OCR 环境。")

        return chunks, pages

    except (ImportError, RuntimeError, ValueError, OSError):
        chunks.clear()
        warnings.clear()
        try:
            import pypdf
            import local_ocr

            reader = pypdf.PdfReader(path)
            if reader.is_encrypted:
                raise ValueError("PDF 已加密。")
            pages = len(reader.pages)
            warnings.append("PDF 使用备用文本解析，未提取结构化表格。")
            active_heading = ""
            for idx, page in enumerate(reader.pages, 1):
                try:
                    txt = (page.extract_text() or "").strip()
                    if not txt and ocr_enabled:
                        try:
                            lines = local_ocr.recognize_page(path, idx)
                            ocr_txt = group_ocr_lines(lines)
                            if ocr_txt.strip():
                                txt = "[OCR识别，请核对原页]\n" + ocr_txt
                                warnings.append(f"第 {idx} 页使用本地 OCR，引用需核对原页。")
                        except (ValueError, OSError) as exc:
                            warnings.append(f"第 {idx} 页 OCR 失败（{type(exc).__name__}）：{exc}")
                    if txt:
                        t_chunks, active_heading = chunk_structured_text(
                            txt,
                            page_no=idx,
                            section_prefix=f"第 {idx} 页",
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            active_heading=active_heading,
                        )
                        chunks.extend(t_chunks)
                    else:
                        warnings.append(f"第 {idx} 页没有可索引文本。")
                except (pypdf.errors.PyPdfError, ValueError, OSError) as exc:
                    warnings.append(f"第 {idx} 页解析失败（{type(exc).__name__}），请核对原件。")
        except Exception:
            warnings.append("PDF 无法读取或已加密，请核对原件。")
            pages = 0
        return chunks, pages


def parse_document(
    path: Path,
    chunk_size: int = 480,
    chunk_overlap: int = 64,
    ocr_enabled: bool = True,
) -> ParsedDocument:
    """Parse engineering documents using specialized per-format extractors & chunkers."""
    suffix = path.suffix.lower()
    chunks: list[dict] = []
    warnings: list[str] = []
    pages = 0

    if suffix == ".csv":
        try:
            raw_txt = _decode_text_bytes(path.read_bytes())
            rows = list(csv.reader(raw_txt.splitlines()))
            banners, hdr, data_rows = detect_table_header(rows)
            if len([c for c in hdr if c]) >= 2 and data_rows:
                chunks.extend(
                    chunk_table_rows(
                        title=path.stem,
                        header=hdr,
                        data_rows=data_rows,
                        page_no=1,
                        section_prefix="CSV 表格",
                        chunk_size=chunk_size,
                        banner_lines=banners,
                        header_tag="[表头]",
                    )
                )
            else:
                t_chunks, _ = chunk_structured_text(raw_txt, 1, "正文", chunk_size, chunk_overlap)
                chunks.extend(t_chunks)
        except Exception as exc:
            warnings.append(f"CSV 读取失败（{type(exc).__name__}）。")

    elif suffix in {".txt", ".md", ".json", ".log"}:
        try:
            text = _decode_text_bytes(path.read_bytes())
            t_chunks, _ = chunk_structured_text(text, 1, "正文", chunk_size, chunk_overlap)
            chunks.extend(t_chunks)
        except (OSError, UnicodeError) as exc:
            warnings.append(f"文本读取失败（{type(exc).__name__}）。")

    elif suffix == ".pdf":
        chunks, pages = _parse_pdf_file(path, chunk_size, chunk_overlap, ocr_enabled, warnings)

    elif suffix in IMAGE_SUFFIXES:
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as img:
                pages = int(getattr(img, "n_frames", 1) or 1)
        except Exception:
            pages = 1
        if not ocr_enabled:
            warnings.append("当前项目已关闭 OCR，图片未进行文字识别。")
        else:
            try:
                import local_ocr

                page_results = local_ocr.recognize_image(path)
                for pg in page_results:
                    p_no = int(pg.get("page") or 1)
                    ocr_text = group_ocr_lines(pg.get("lines") or [])
                    if ocr_text.strip():
                        sec = "图片 OCR" if pages <= 1 else f"图片第 {p_no} 页 · OCR"
                        o_chunks, _ = chunk_structured_text(
                            "[图片OCR识别，请核对原图]\n" + ocr_text,
                            page_no=p_no,
                            section_prefix=sec,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            chunk_type="ocr",
                        )
                        chunks.extend(o_chunks)
                if chunks:
                    warnings.append("图片使用本地 OCR 识别，数值与标注请核对原图。")
                else:
                    warnings.append("图片未识别出可用文字，请核对原图或提高清晰度。")
            except (ValueError, OSError) as exc:
                warnings.append(f"图片 OCR 失败（{type(exc).__name__}）：{exc}")

    elif suffix in {".docx", ".docm"}:
        chunks.extend(_parse_word_file(path, chunk_size, chunk_overlap, ocr_enabled, warnings))

    elif suffix in {".xlsx", ".xlsm"}:
        try:
            chunks.extend(_parse_excel_file(path, chunk_size, warnings))
        except Exception as exc:
            warnings.append(f"Excel 解析中断（{type(exc).__name__}），请核对原件。")

    elif suffix in {".html", ".htm"}:
        try:
            html_text = _decode_text_bytes(path.read_bytes())
            chunks.extend(_parse_html_content(html_text, chunk_size, chunk_overlap, warnings))
        except Exception as exc:
            warnings.append(f"HTML 解析中断（{type(exc).__name__}），请核对原件。")

    elif suffix in {".doc", ".xls", ".wps"}:
        head = b""
        try:
            with path.open("rb") as f:
                head = f.read(256).lstrip()
        except OSError as exc:
            warnings.append(f"旧格式文件读取失败（{type(exc).__name__}）。")
            head = b""

        if head.startswith(b"PK\x03\x04"):
            if suffix == ".xls":
                try:
                    chunks.extend(_parse_excel_file(path, chunk_size, warnings))
                except Exception as exc:
                    warnings.append(f"Excel 解析中断（{type(exc).__name__}），请核对原件。")
            else:
                chunks.extend(_parse_word_file(path, chunk_size, chunk_overlap, ocr_enabled, warnings))
        elif head.lower().startswith((b"<html", b"<!doctype", b"<?xml", b"<meta", b"<div", b"\xef\xbb\xbf<")):
            try:
                html_text = _decode_text_bytes(path.read_bytes())
                chunks.extend(_parse_html_content(html_text, chunk_size, chunk_overlap, warnings))
            except Exception as exc:
                warnings.append(f"网页格式文档解析中断（{type(exc).__name__}）。")
        elif head:
            try:
                from office_convert import convert_snapshot

                with tempfile.TemporaryDirectory() as tmp_dir:
                    snapshot = Path(tmp_dir) / f"src{suffix}"
                    shutil.copy2(path, snapshot)
                    converted, _ = convert_snapshot(snapshot)
                    inner = parse_document(
                        converted,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        ocr_enabled=ocr_enabled,
                    )
                    chunks.extend(inner.chunks)
                    warnings.append("已通过本机 Office 转换临时副本解析；位置基于转换副本，请核对原件。")
                    warnings.extend(inner.warnings)
            except Exception as exc:
                warnings.append(f"旧格式转换失败（{exc}），建议另存为 DOCX/XLSX/PDF 后上传。")
    else:
        warnings.append("此文件格式暂无本地解析器。")

    if not chunks and not warnings:
        warnings.append("文件没有可索引文本。")
    return ParsedDocument(chunks=chunks, pages=pages, warnings=list(dict.fromkeys(warnings)))
