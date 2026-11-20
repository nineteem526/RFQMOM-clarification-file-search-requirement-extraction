"""Local readers with explicit coverage limits and source-native locations."""

from pathlib import Path

from docx import Document as WordDocument
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from pypdf import PdfReader
import fitz

from local_embedding import local_engine


def parse_local(path, use_ocr=False):
    path = Path(path)
    blocks, warnings = [], []

    def add(text, location):
        if isinstance(text, str) and text.strip():
            blocks.append({"text": text, "location": location})

    suffix = path.suffix.lower()
    if suffix == ".txt":
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            add(line, f"文本第{number}行")
    elif suffix == ".docx":
        word = WordDocument(path)
        for number, paragraph in enumerate(word.paragraphs, 1):
            add(paragraph.text, f"正文第{number}段（不提供页码）")

        def table_blocks(table, label):
            context = []
            for header_number, header_row in enumerate(table.rows[:3], 1):
                context.append(
                    f"顶部行{header_number}：" +
                    "；".join(f"列{col}={cell.text}" for col, cell in enumerate(header_row.cells, 1))
                )
            for row_number, row in enumerate(table.rows, 1):
                values, visited = [], set()
                for column, cell in enumerate(row.cells, 1):
                    if cell._tc in visited:
                        continue
                    visited.add(cell._tc)
                    values.append(f"列{column}：{cell.text}")
                    for nested_number, nested in enumerate(cell.tables, 1):
                        table_blocks(
                            nested, f"{label}/行{row_number}列{column}/嵌套表{nested_number}"
                        )
                blocks.append({
                    "text": "；".join(values), "location": f"{label}第{row_number}行",
                    "context": "表格顶部原文（不自动推断表头层级）：\n" + "\n".join(context),
                })

        for number, table in enumerate(word.tables, 1):
            table_blocks(table, f"正文表{number}，")
        warnings.append(
            "DOCX按正文段落和表格行定位；不含文本框、图片文字、页眉页脚、"
            "批注和修订差异。表格跨行表头、合并关系和单位需核对，不能视为完整解析。"
        )
    elif suffix == ".pdf":
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("加密PDF未解析；请提供公司允许的可读取副本。")
        with fitz.open(path) as pdf:
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if not text.strip():
                    if use_ocr:
                        from local_ocr import recognize_page

                        try:
                            lines = recognize_page(path, number)
                            for line in lines:
                                box = line["box"]
                                bbox = (min(point[0] for point in box), min(point[1] for point in box),
                                        max(point[0] for point in box), max(point[1] for point in box))
                                add(
                                    line["text"],
                                    f"PDF物理第{number}页；本地OCR（非文本层，需核对）"
                                    f"；置信度{line['confidence']:.3f}；OCR坐标{bbox}",
                                )
                            if not lines:
                                warnings.append(f"PDF物理第{number}页OCR未识别到文字，未覆盖。")
                            elif any(line["confidence"] < 0.8 for line in lines):
                                warnings.append(f"PDF物理第{number}页存在低置信度OCR文字，数字、型号须核对原页。")
                        except (ValueError, OSError) as error:
                            warnings.append(f"PDF物理第{number}页OCR失败，未覆盖：{error}")
                    else:
                        warnings.append(f"PDF物理第{number}页无可提取文本，未启用本地OCR。")
                else:
                    add(text, f"PDF物理第{number}页（非印刷页码）")
                try:
                    tables = pdf[number - 1].find_tables().tables
                    for table_number, table in enumerate(tables, 1):
                        rows = table.extract()
                        context = "\n".join(
                            f"顶部行{index}：" + "；".join(
                                f"列{col}={value or ''}" for col, value in enumerate(row, 1)
                            ) for index, row in enumerate(rows[:3], 1)
                        )
                        for row_number, row in enumerate(rows, 1):
                            blocks.append({
                                "text": "；".join(
                                    f"列{col}={value or ''}" for col, value in enumerate(row, 1)
                                ),
                                "context": "PDF自动识别表格顶部原文（需核对）：\n" + context,
                                "location": f"PDF物理第{number}页；表{table_number}"
                                f"第{row_number}行；页面坐标{tuple(round(v, 1) for v in table.bbox)}",
                            })
                except (ValueError, RuntimeError, TypeError) as error:
                    warnings.append(f"PDF第{number}页表格识别失败：{type(error).__name__}: {error}")
        warnings.append(
            "PDF提取文本层及可识别的表格行；自动表格、合并表头及多栏顺序需核对。"
            + ("无文本页已尝试本地OCR；图片表格不保证行列关联，识别字词及数字须对照原页。"
               if use_ocr else "图片文字未启用OCR。")
        )
    elif suffix == ".xlsx":
        book = load_workbook(path, read_only=False, data_only=False, keep_links=False)
        cached_book = None
        try:
            cached_book = load_workbook(path, read_only=True, data_only=True, keep_links=False)
            for sheet in book:
                if sheet.max_row * sheet.max_column > 2_000_000:
                    raise ValueError(f"工作表{sheet.title}范围过大，拒绝不受控全表读取。")
                cached_sheet = cached_book[sheet.title]
                cached_formulas = 0
                missing_formula_caches = []
                error_formula_caches = []
                headers = []
                merged = {}
                for region in sheet.merged_cells.ranges:
                    anchor = sheet.cell(region.min_row, region.min_col)
                    if anchor.value is not None:
                        for row_number in range(region.min_row, region.max_row + 1):
                            merged.setdefault(row_number, []).append(
                                f"合并区域{region}，锚点{anchor.coordinate}={anchor.value}"
                            )
                for row, cached_row in zip(sheet.iter_rows(), cached_sheet.iter_rows()):
                    cells = []
                    for cell in row:
                        if cell.value is None:
                            continue
                        if cell.data_type == "f":
                            cached_cell = cached_row[cell.column - 1]
                            if cached_cell.data_type == "e":
                                error_formula_caches.append(cell.coordinate)
                                cached_value = f"缓存错误:{cached_cell.value}"
                            elif cached_cell.value is None:
                                missing_formula_caches.append(cell.coordinate)
                                cached_value = "没有已保存缓存值"
                            else:
                                cached_formulas += 1
                                cached_value = f"已保存缓存值={cached_cell.value}"
                            formatted = (
                                f"（数字格式：{cell.number_format}）"
                                if cell.number_format not in {"General", "0", "@"} else ""
                            )
                            cells.append(
                                f"{cell.coordinate}=[公式（{cell.value}）；{cached_value}"
                                f"{formatted}；未重算，可能陈旧]"
                            )
                        elif cell.data_type == "e":
                            warnings.append(f"{sheet.title}!{cell.coordinate}为错误单元格。")
                            cells.append(f"{cell.coordinate}=[错误值:{cell.value}]")
                        else:
                            formatted = (
                                f"（数字格式：{cell.number_format}）"
                                if cell.number_format not in {"General", "0", "@"} else ""
                            )
                            cells.append(f"{cell.coordinate}={cell.value}{formatted}")
                    if not cells:
                        continue
                    row_text = "；".join(cells)
                    if len(headers) < 5:
                        headers.append(row_text)
                    context = "\n".join(headers)
                    blocks.append({
                        "text": f"当前行原文：{row_text}",
                        "context": f"工作表：{sheet.title}\n"
                        f"顶部非空行原文（不推断层级）：\n{context}\n"
                        + "\n".join(merged.get(row[0].row, [])),
                        "location": f"工作表 {sheet.title}!A{row[0].row}:"
                        f"{get_column_letter(len(row))}{row[0].row}",
                    })
                if cached_formulas:
                    warnings.append(
                        f"工作表{sheet.title}有{cached_formulas}个公式单元格读取到已保存缓存值；"
                        "保留公式和缓存并标记未重算、可能陈旧，不代表已验证。"
                    )
                if missing_formula_caches:
                    samples = "、".join(missing_formula_caches[:10])
                    suffix_note = "等" if len(missing_formula_caches) > 10 else ""
                    warnings.append(
                        f"工作表{sheet.title}有{len(missing_formula_caches)}个公式单元格"
                        f"没有已保存缓存值（{samples}{suffix_note}）。"
                    )
                if error_formula_caches:
                    samples = "、".join(error_formula_caches[:10])
                    suffix_note = "等" if len(error_formula_caches) > 10 else ""
                    warnings.append(
                        f"工作表{sheet.title}有{len(error_formula_caches)}个公式单元格"
                        f"缓存为错误值（{samples}{suffix_note}）。"
                    )
                if sheet.merged_cells.ranges:
                    warnings.append(
                        f"工作表{sheet.title}存在合并单元格："
                        + ", ".join(str(item) for item in sheet.merged_cells.ranges)
                        + "；相关行附锚点与范围，不自动判断合并语义。"
                    )
                if sheet.sheet_state != "visible":
                    warnings.append(f"工作表{sheet.title}为隐藏表，已包含但需核对适用范围。")
        finally:
            if cached_book is not None:
                cached_book.close()
            book.close()
        warnings.append(
            "Excel保留坐标、顶部五条非空行、数字格式、公式及其已保存缓存值和合并范围；"
            "缓存未重算且可能陈旧，不代表已验证；不自动推断单位或多级表头语义，"
            "不读取图表及图片。长行拆分须核对相邻片段。"
        )
    elif suffix in {".doc", ".xls", ".wps"}:
        from office_convert import convert_snapshot

        converted, record = convert_snapshot(path)
        chunks, warnings = parse_local(converted, use_ocr=use_ocr)
        for chunk in chunks:
            chunk["location"] = f"本地转换副本 {converted.name} / " + chunk["location"]
        warnings.insert(0, f"本地转换关系：{record}；位置属于转换副本，不冒充源文件页码。")
        return chunks, warnings
    else:
        raise ValueError(f"未支持的本地解析格式：{suffix}")
    if not blocks:
        detail = " 解析说明：" + "；".join(warnings) if warnings else ""
        raise ValueError("没有提取到可用文本；没有创建成功索引。" + detail)
    return chunk_blocks(blocks), list(dict.fromkeys(warnings))


def chunk_blocks(blocks):
    tokenizer = local_engine().tokenizer
    chunks = []
    for block in blocks:
        text = block["text"]
        context = block.get("context", "")
        if context:
            # Keep context repeated on every row fragment, never silently truncate it.
            if len(tokenizer.encode(context).ids) > 220:
                chunks.extend(chunk_blocks([{
                    "text": context, "location": block["location"] + "；表头及合并范围原文",
                }]))
                context = "表头上下文较长，见同一位置的表头及合并范围原文片段。\n"
            context += "\n"
        start = 0
        while start < len(text):
            low, high = start + 1, min(start + 1600, len(text))
            end = low
            while low <= high:
                middle = (low + high) // 2
                if len(tokenizer.encode(context + text[start:middle]).ids) <= 480:
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            piece = text[start:end]
            if piece.strip():
                chunks.append({
                    "text": context + piece,
                    "location": block["location"] + (
                        f"；字符{start + 1}-{end}（分块，须核对相邻片段）"
                        if start or end < len(text) else ""
                    ),
                })
            start = end
    return chunks
