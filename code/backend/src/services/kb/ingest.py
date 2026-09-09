"""文档解析与分块（入库链路第一步）。

支持格式：Markdown / 纯文本 / PDF（含扫描件，走 OCR）/ Word(docx) / 图片（OCR）
后续可扩展：HTML、Excel

分块策略（P1 简单版）：按字符固定大小 + 重叠窗口，保留元数据。
注意：分块质量直接影响检索质量——这是 RAG 的"GIGO"关卡。
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jfif"}
# 图片类型：直接走 OCR 识别图中文字（jfif 是 JPEG 的容器格式，常见于浏览器保存）
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jfif"}

_FINANCIAL_TABLE_BASE_NAMES = (
    "所有者权益变动表",
    "股东权益变动表",
    "资产负债表",
    "现金流量表",
    "利润表",
    # 营收/净利润等核心指标常在这两张「主要会计数据/财务指标」表中，
    # 不识别它们会导致财务事实的期间/口径全部落空。
    "主要会计数据",
    "主要财务指标",
)
_FINANCIAL_SCOPE_PREFIXES = ("合并", "母公司", "子公司")
_UNIT_NAMES = ("百万元", "亿元", "万元", "千元", "万美元", "港元", "元")
_REPORT_PERIOD_SUFFIXES = (
    "半年度",
    "上半年",
    "下半年",
    "第一季度",
    "第二季度",
    "第三季度",
    "第四季度",
    "一季度",
    "二季度",
    "三季度",
    "四季度",
    "年度",
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_FINANCIAL_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "营业总收入": ("营业总收入",),
    "营业收入": ("营业收入",),
    "归属于上市公司股东的净利润": (
        "归属于上市公司股东的净利润",
        "归属于母公司股东的净利润",
        "归属于母公司所有者的净利润",
        "归属于上市公司普通股股东的净利润",
        "归母净利润",
    ),
    "研发投入合计": ("研发投入合计", "研发投入总额"),
    "研发费用": ("研发费用",),
    "经营活动产生的现金流量净额": (
        "经营活动产生的现金流量净额",
        "经营活动现金流量净额",
        "经营活动产生现金流量净额",
    ),
}
_FINANCIAL_METRIC_ORDER = tuple(_FINANCIAL_METRIC_ALIASES)
_FINANCIAL_METRIC_ALIASES_SORTED = tuple(
    sorted(
        (
            (alias, metric)
            for metric, aliases in _FINANCIAL_METRIC_ALIASES.items()
            for alias in aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)
_FINANCIAL_UNIT_SCALE: dict[str, Decimal] = {
    "": Decimal("1"),
    "元": Decimal("1"),
    "港元": Decimal("1"),
    "美元": Decimal("1"),
    "千元": Decimal("1000"),
    "万元": Decimal("10000"),
    "万美元": Decimal("10000"),
    "百万元": Decimal("1000000"),
    "亿元": Decimal("100000000"),
}
_FINANCIAL_INLINE_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(_FINANCIAL_UNIT_SCALE, key=len, reverse=True) if unit
)
_FINANCIAL_VALUE_RE = re.compile(
    rf"(?<!\d)(?P<raw>\(?[+-]?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?\)?)(?P<unit>\s*(?:{_FINANCIAL_INLINE_UNIT_PATTERN}|%|％))?(?!\d)"
)
_PERIOD_TOKEN_RE = re.compile(
    rf"20\d{{2}}\s*年\s*(?:"
    rf"\d{{1,2}}\s*[-—至到]\s*\d{{1,2}}\s*月|"
    rf"{'|'.join(map(re.escape, _REPORT_PERIOD_SUFFIXES))})?"
)
_FINANCIAL_CHANGE_HEADER_RE = re.compile(
    r"(?:同比|增减|变动|增长|下降|较上年|比上年|本期比|[%％])"
)
_FINANCIAL_NON_VALUE_HEADER_RE = re.compile(
    r"^(?:附注|注释|注|行次|编号|序号|项目|科目|指标)(?:编号)?$"
)
_FINANCIAL_CURRENT_PERIOD_HEADER_RE = re.compile(
    r"(?:本期(?:数|金额|发生额|累计|期末)?|本年(?:数|金额|发生额|累计)?)(?!增加|减少|余额)"
)
_FINANCIAL_PRIOR_PERIOD_HEADER_RE = re.compile(
    r"(?:上年同期(?:数|金额)?|上期(?:数|金额|发生额|累计)?|上年(?:数|金额|发生额)?)(?!增加|减少|余额)"
)
_FINANCIAL_PERIOD_SUBHEADER_RE = re.compile(r"^(?:调整后|调整前|经调整|未调整)$")


@dataclass(kw_only=True)
class DocumentChunk:
    """一个分块：检索的最小单元，携带定位元数据。"""

    text: str
    doc_id: str            # 所属文档 ID（UUID）
    doc_title: str         # 文档标题
    chunk_index: int       # 第几个分块（0 起）
    source_type: str       # pdf / markdown / text / docx
    kb_id: str = "default"   # 所属知识库（多库隔离，P3）
    page: int | None = None   # 页码（PDF 有，其余 None）
    metadata: dict[str, Any] = field(default_factory=dict)
    # RAG V2 定位字段。保留为可选字段，旧调用只填写上面的字段即可。
    chunk_type: str = "child"
    parent_id: str = ""
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    previous_chunk_id: str = ""
    next_chunk_id: str = ""


@dataclass(frozen=True)
class StructuredBlock:
    """解析后的可定位文本块；页码使用 PDF 的 1 起始编号。"""

    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str = ""
    table_name: str = ""
    statement_scope: str = "unknown"
    report_period: str = ""
    unit: str = ""
    table_id: str = ""
    is_table: bool = False
    # find_tables 产出的 Markdown 表格原文（可能含空列/拆行等噪声）。
    # 仅作为 financial_facts 的提取原料，不进入 chunk.text，
    # 避免污染 embedding 与 BM25（见 KB_SEPARATE_TABLE_MD）。
    table_markdown: str = ""


def parse_document(path: Path, *, ocr_mode: str = "local") -> str:
    """按扩展名解析文档 → 纯文本。解析失败抛异常（由上层降级处理）。

    ocr_mode：图片/扫描 PDF 的 OCR 模式（local/baidu/auto，见 services.kb.ocr）。
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}，支持 {sorted(SUPPORTED_EXTENSIONS)}")

    if ext == ".md" or ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        return _parse_pdf(path, ocr_mode=ocr_mode)
    if ext == ".docx":
        return _parse_docx(path)
    if ext in IMAGE_EXTENSIONS:
        return _parse_image(path, ocr_mode=ocr_mode)
    raise ValueError(f"未实现的解析器: {ext}")


def parse_document_structured(
    path: Path, *, ocr_mode: str = "local", document_title: str | None = None
) -> list[StructuredBlock]:
    """解析文档并保留页码、章节路径。

    这是新入口；``parse_document`` 仍只返回纯文本，供现有调用继续使用。
    Markdown/TXT/DOCX 没有可靠的物理页概念，因此页码为空；PDF 页码从 1 开始。
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}，支持 {sorted(SUPPORTED_EXTENSIONS)}")
    # 统一为 (页码, 正文, 表格 Markdown) 三元组；只有 PDF 会产生表格 Markdown。
    if ext in {".md", ".txt"}:
        pages = [(None, path.read_text(encoding="utf-8", errors="ignore"), "")]
    elif ext == ".pdf":
        pages = _parse_pdf_pages(path, ocr_mode=ocr_mode)
    elif ext == ".docx":
        pages = [(None, _parse_docx(path), "")]
    elif ext in IMAGE_EXTENSIONS:
        pages = [(None, _parse_image(path, ocr_mode=ocr_mode), "")]
    else:
        raise ValueError(f"未实现的解析器: {ext}")
    return _structured_blocks_from_pages(
        pages,
        doc_period=_extract_doc_period(
            pages, document_title=document_title or path.stem
        ),
    )


def _is_page_number(text: str) -> bool:
    """只匹配孤立页码，绝不匹配正文中的金额/编号。"""
    compact = re.sub(r"\s+", "", text)
    return bool(
        re.fullmatch(
            r"(?:第)?\d{1,4}(?:页)?|\d{1,4}/\d{1,4}|第\d{1,4}页(?:共\d{1,4}页)?",
            compact,
        )
    )


def _normalize_checkbox_text(text: str) -> str:
    """把常见勾选项变为可检索的明确结论。"""
    compact = re.sub(r"\s+", "", text)
    if re.search(r"[√☑]适用.*[□☐]不适用", compact):
        return "适用：是"
    if re.search(r"[□☐]适用.*[√☑]不适用", compact):
        return "适用：否"
    if re.search(r"[√☑]是.*[□☐]否", compact):
        return "结论：是"
    if re.search(r"[□☐]是.*[√☑]否", compact):
        return "结论：否"
    return text.strip()


def _clean_layout_lines(lines: list[tuple[float, float, str]], page_height: float) -> list[str]:
    """移除底部孤立页码，并在保留版面阅读顺序后归一化勾选项。"""
    cleaned: list[str] = []
    for _x0, y1, text in sorted(lines, key=lambda item: (item[1], item[0])):
        if y1 >= page_height * 0.88 and _is_page_number(text):
            continue
        normalized = _normalize_checkbox_text(text)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _render_ocr_layout(lines: list[Any], page_width: float, page_height: float) -> str:
    """按 OCR 文字框坐标恢复简单表格行；普通同行仍按自然空格拼接。"""
    visible = [line for line in lines if not (line.y1 >= page_height * 0.88 and _is_page_number(line.text))]
    rows: list[list[Any]] = []
    for line in sorted(visible, key=lambda item: (item.y0, item.x0)):
        if rows and abs(line.y0 - rows[-1][0].y0) <= max(12.0, (line.y1 - line.y0) * 0.75):
            rows[-1].append(line)
        else:
            rows.append([line])
    rendered: list[str] = []
    for row in rows:
        row.sort(key=lambda item: item.x0)
        cells = [_normalize_checkbox_text(item.text) for item in row]
        is_table_row = len(row) > 1 and any(
            row[index + 1].x0 - row[index].x1 >= page_width * 0.12
            for index in range(len(row) - 1)
        )
        rendered.append(" | ".join(cells) if is_table_row else " ".join(cells))
    return "\n".join(item for item in rendered if item.strip())

def _extract_pdf_tables(page: Any) -> list[str]:
    """把文字版 PDF 表格补充为 Markdown，保留列关系供检索使用。"""
    try:
        tables = page.find_tables().tables
    except Exception:
        return []
    result: list[str] = []
    for table in tables:
        rows = table.extract() or []
        rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in rows]
        rows = [row for row in rows if any(row)]
        if len(rows) < 2 or len(rows[0]) < 2:
            continue
        # 文字层通常也含表格单元格，但不保留稳定列关系。这里仍补充 Markdown
        # 结构；后续候选去重负责压制近重复，不能为去重牺牲表头和列语义。
        header = rows[0]
        result.append("| " + " | ".join(header) + " |")
        result.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[1:]:
            padded = row + [""] * (len(header) - len(row))
            result.append("| " + " | ".join(_normalize_checkbox_text(cell) for cell in padded[:len(header)]) + " |")
    return ["\n".join(result)] if result else []


def _separate_table_markdown() -> bool:
    """分离模式：表格 Markdown 只作 financial_facts 原料，不进入 chunk.text。

    默认关闭，保持既有入库行为不变；由 rebuild_full_v3_sep.py 显式开启。
    """
    return os.getenv("KB_SEPARATE_TABLE_MD", "0") == "1"


def _parse_pdf_pages(path: Path, *, ocr_mode: str = "local") -> list[tuple[int, str, str]]:
    """逐页按坐标提取文字版 PDF，扫描页走带坐标 OCR；页脚页码不进入正文。

    返回 (页码, 正文, 表格 Markdown) 三元组。分离模式下正文只含 get_text
    结果，表格 Markdown 单独返回；否则沿用旧行为拼进正文、第三项为空串。
    """
    import pymupdf

    separate = _separate_table_markdown()
    pages: list[tuple[int, str, str]] = []
    with pymupdf.open(path) as document:
        for index, page in enumerate(document):
            blocks = page.get_text("blocks")
            lines = [(float(block[0]), float(block[3]), str(block[4]).strip()) for block in blocks if str(block[4]).strip()]
            if lines:
                body = "\n".join(_clean_layout_lines(lines, float(page.rect.height)))
                # find_tables 会产出空列/拆行等垃圾 Markdown，而 get_text 正文已含表格内容。
                # 实验开关：KB_USE_FIND_TABLES=0 时跳过 find_tables，正文更干净（T1-lite 新策略）。
                tables = _extract_pdf_tables(page) if os.getenv("KB_USE_FIND_TABLES", "1") != "0" else []
                table_md = "\n\n".join(part for part in tables if part)
                # 分离模式：表格 Markdown 不进正文，只作事实提取原料，避免污染检索。
                text = body if separate else "\n\n".join(part for part in [body, *tables] if part)
                pages.append((index + 1, text, table_md if separate else ""))
            else:
                text = _ocr_pdf_page(str(path), index, ocr_mode=ocr_mode)
                pages.append((index + 1, text, ""))
    return pages


def _parse_pdf(path: Path, *, ocr_mode: str = "local") -> str:
    """兼容入口：逐页解析后仍只返回纯文本（丢弃独立的表格 Markdown）。"""
    return "\n\n".join(
        page[1].strip() for page in _parse_pdf_pages(path, ocr_mode=ocr_mode) if page[1].strip()
    )


def _ocr_pdf_page(pdf_path: str, page_index: int, *, ocr_mode: str = "local") -> str:
    """渲染扫描 PDF 页后做带坐标 OCR，过滤底部孤立页码。"""
    import pymupdf

    from services.kb.ocr import ocr_image, ocr_image_with_layout

    with pymupdf.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
    if ocr_mode == "local":
        lines = ocr_image_with_layout(png_bytes)
        return _render_ocr_layout(lines, float(pix.width), float(pix.height))
    return ocr_image(png_bytes, mode=ocr_mode)

def _parse_image(path: Path, *, ocr_mode: str = "local") -> str:
    """图片解析：直接 OCR 识别图中文字。"""
    from services.kb.ocr import ocr_image

    return ocr_image(path, mode=ocr_mode)


def _parse_docx(path: Path) -> str:
    """Word 解析：提取段落 + 表格。"""
    import docx

    doc = docx.Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            continue
        if len(rows) >= 2 and len(rows[0]) >= 2:
            header = rows[0]
            table_lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * len(header)) + " |",
            ]
            for row in rows[1:]:
                padded = row + [""] * (len(header) - len(row))
                table_lines.append("| " + " | ".join(padded[:len(header)]) + " |")
            parts.append("\n".join(table_lines))
        else:
            parts.extend(" | ".join(row) for row in rows)
    return "\n\n".join(parts)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(?:第\s*[一二三四五六七八九十百千万\d]+\s*[章节条])"
    r"|(?:[一二三四五六七八九十百千万]+[.、）)])"
    r"|(?:\d+[.、）)])"
    r"|(?:\d+(?:\.\d+)+[、）)]?))\s*[^。！？]{1,80}$"
)


def _heading_info(line: str) -> tuple[int, str] | None:
    """识别 Markdown 标题及常见编号章节，避免把普通长句当标题。"""
    stripped = line.strip()
    match = _HEADING_RE.match(stripped)
    if match:
        return len(match.group(1)), match.group(2).strip()
    if _NUMBERED_HEADING_RE.match(stripped) and len(stripped) <= 100:
        # 年报表格里大量金额、百分比形似 ``1.2`` 章节编号；它们必须继续
        # 留在表格正文中，不能一格变成一个 section。
        if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?%?", re.sub(r"\s+", "", stripped)):
            return None
        prefix = re.match(
            r"(?:第\s*[^\s]+\s*[章节条]"
            r"|[一二三四五六七八九十百千万]+[.、）)]"
            r"|\d+[.、）)]"
            r"|\d+(?:\.\d+)+[、）)]?)",
            stripped,
        )
        # PDF 抽取常把带编号的正文在标点前换行。过长的“1、……”更像
        # 列表正文而非标题；仅“第X章/节/条”允许较长名称。
        if prefix and not prefix.group(0).startswith("第") and len(stripped) > 50:
            return None
        depth = max(1, (prefix.group(0).count(".") + 1) if prefix else 1)
        return depth, stripped
    return None


def _structured_blocks_from_pages(
    pages: list[tuple[Any, ...]], *, doc_period: str | None = None
) -> list[StructuredBlock]:
    """按章节和页边界生成结构块，跨页章节保留相同 section_path。

    pages 元素为 (页码, 正文) 或 (页码, 正文, 表格 Markdown)，兼容旧调用方。

    每页的表格 Markdown 挂到该页产生的块上，只用于 financial_facts 提取，
    不进入块正文（chunk.text）。
    """
    blocks: list[StructuredBlock] = []
    stack: list[str] = []
    current: list[str] = []
    current_tables: list[str] = []
    current_start: int | None = None
    current_end: int | None = None
    current_path = ""

    def flush() -> None:
        nonlocal current, current_tables, current_start, current_end
        text = "\n".join(current).strip()
        if text:
            blocks.append(
                StructuredBlock(
                    text,
                    current_start,
                    current_end,
                    current_path,
                    table_markdown="\n\n".join(part for part in current_tables if part),
                )
            )
        current = []
        current_tables = []
        current_start = None
        current_end = None

    for parts in pages:
        page = parts[0]
        page_text = parts[1] if len(parts) > 1 else ""
        table_md = parts[2] if len(parts) > 2 else ""
        if not page_text or not page_text.strip():
            continue
        if table_md and table_md.strip():
            current_tables.append(table_md.strip())
        for line in page_text.splitlines():
            heading = _heading_info(line)
            if heading:
                flush()
                level, title = heading
                stack[:] = stack[: level - 1]
                stack.append(title)
                current_path = " > ".join(stack)
                current = [line.strip()]
                current_start = page
                current_end = page
                continue
            if not current:
                current_path = " > ".join(stack)
                current_start = page
            current.append(line.rstrip())
            current_end = page
        # PDF 页是硬边界，便于每个子块至少携带准确的起始页；无页的文本
        # 作为一个整体处理，仍可跨段落按自然边界聚合。
        if page is not None:
            flush()
    flush()
    if doc_period is None:
        doc_period = _extract_doc_period(pages)
    return _annotate_financial_blocks(blocks, doc_period)


def _find_financial_table_title(text: str) -> tuple[str, str] | None:
    """提取高置信度财务表名，并返回规范名和原始标题行。"""
    candidates = [
        prefix + base
        for prefix in _FINANCIAL_SCOPE_PREFIXES
        for base in _FINANCIAL_TABLE_BASE_NAMES
    ] + list(_FINANCIAL_TABLE_BASE_NAMES)
    candidates.sort(key=len, reverse=True)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        compact = re.sub(r"\s+", "", line)
        if not compact or len(compact) > 100:
            continue
        # 带句号/逗号的长句中提到“利润表”不算表头，避免把正文口径污染到后续页。
        if re.search(r"[。！？!?；;，,]", compact):
            continue
        for candidate in candidates:
            if candidate in compact:
                return candidate, compact
    return None


def _is_markdown_table(text: str) -> bool:
    """只把具有多列分隔符的 Markdown 文本视为表格。"""
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_rows = [line for line in rows if line.count("|") >= 2]
    if len(pipe_rows) < 2:
        return False
    return any(re.search(r"\|?\s*:?-{3,}\s*\|", line) for line in pipe_rows) or len(pipe_rows) >= 3


def _extract_unit(text: str) -> str:
    """仅从明确的“单位/金额单位”标记提取单位，避免把正文金额当单位。"""
    unit_pattern = "|".join(re.escape(unit) for unit in _UNIT_NAMES)
    marker = re.compile(
        rf"(?:单位|金额单位|货币单位|币种及单位)[^\n]*?({unit_pattern})"
    )
    found: set[str] = set()
    for line in text.splitlines():
        compact = re.sub(r"\s+", "", line)
        match = marker.search(compact)
        if match:
            value = match.group(1)
            found.add("元" if value == "人民币元" else value)
            continue
        if re.search(r"[（(][^）)]*(?:单位|金额)[^）)]*[）)]", compact):
            match = re.search(rf"({unit_pattern})", compact)
            if match:
                found.add(match.group(1))
    return next(iter(found)) if len(found) == 1 else ""


def _canonical_report_period(year: str, suffix: str = "") -> str:
    if suffix == "年度":
        return f"{year}年度"
    return f"{year}年{suffix}" if suffix else f"{year}年"


def _extract_report_period(text: str) -> str:
    """提取表头附近的报告期；没有明确期间时只回退到首个表头年份。"""
    lines = [re.sub(r"\s+", "", line) for line in text.splitlines() if line.strip()]
    period_pattern = re.compile(
        rf"(20\d{{2}})年?({'|'.join(_REPORT_PERIOD_SUFFIXES)})"
    )
    range_pattern = re.compile(
        r"20\d{2}年\d{1,2}月\d{1,2}日(?:至|到|—|-)"
        r"20\d{2}年\d{1,2}月\d{1,2}日"
    )
    month_range_pattern = re.compile(
        r"(20\d{2})年\d{1,2}月(?:-|—|至|到)\d{1,2}月"
    )
    date_pattern = re.compile(r"20\d{2}年\d{1,2}月\d{1,2}日")
    for line in lines[:12]:
        range_match = range_pattern.search(line)
        if range_match and re.search(r"报告期|截至|期间|本期", line):
            return range_match.group(0)
        period_match = period_pattern.search(line)
        if period_match and re.search(r"报告|期间|截至|本期|年度|半年度", line):
            return _canonical_report_period(period_match.group(1), period_match.group(2))
        date_match = date_pattern.search(line)
        if date_match and re.search(r"报告期|截至|期间|本期", line):
            return date_match.group(0)
        month_range_match = month_range_pattern.search(line)
        if month_range_match and re.search(r"报告|期间|截至|本期|年度|半年度", line):
            return re.sub(r"\s+", "", month_range_match.group(0))
    for line in lines[:12]:
        period_match = period_pattern.search(line)
        if period_match:
            return _canonical_report_period(period_match.group(1), period_match.group(2))
        month_range_match = month_range_pattern.search(line)
        if month_range_match:
            return re.sub(r"\s+", "", month_range_match.group(0))
    for line in lines[:12]:
        year_match = re.search(r"(20\d{2})年", line)
        if year_match:
            return _canonical_report_period(year_match.group(1))
    return ""


def _extract_doc_period(
    pages: list[tuple[Any, ...]], *, document_title: str = ""
) -> str:
    """从页眉标题（「XX公司2025年半年度报告」）提取报告期，作为表格期间的回退。

    年报每页页眉都带公司名+报告期，而财务表格块本身往往只有表名和数据行，
    没有期间字样；这里优先从原文件标题、再从页眉捞出文档级期间，回填给没
    提取到期间的表格块。文件封面有时只有“半年度报告”而没有年份，不能只看
    封面；但文件名通常包含完整报告期，是更可靠的回退证据。
    """
    period_pattern = re.compile(
        rf"(20\d{{2}})年?({'|'.join(_REPORT_PERIOD_SUFFIXES)})"
    )
    range_pattern = re.compile(
        r"20\d{2}年\d{1,2}月\d{1,2}日(?:至|到|—|-)"
        r"20\d{2}年\d{1,2}月\d{1,2}日"
    )
    month_range_pattern = re.compile(
        r"(20\d{2})年\d{1,2}月(?:-|—|至|到)\d{1,2}月"
    )

    sources: list[tuple[str, bool]] = []
    if document_title:
        sources.append((document_title, True))
    for parts in pages:
        page_text = parts[1]
        if not page_text:
            continue
        # 页眉一般位于前几行；放宽到 20 行以覆盖目录/中期报告页的版式，
        # 仍只采信带“报告/年报”标记的行，避免把正文年份当文档期。
        for line in page_text.splitlines()[:20]:
            sources.append((line, False))

    for source, from_title in sources:
        compact = re.sub(r"\s+", "", source)
        if not compact or (not from_title and not re.search(r"报告|年报", compact)):
            continue
        range_match = range_pattern.search(compact)
        if range_match:
            return range_match.group(0)
        period_match = period_pattern.search(compact)
        if period_match:
            return _canonical_report_period(period_match.group(1), period_match.group(2))
        month_range_match = month_range_pattern.search(compact)
        if month_range_match:
            return month_range_match.group(0)
        year_match = re.search(r"(20\d{2})年", compact)
        if year_match:
            return _canonical_report_period(year_match.group(1))
    return ""


def _scope_mentions(text: str) -> set[str]:
    """提取明确的口径标记；普通“合并日”等会计术语不算报表口径。"""
    scopes: set[str] = set()
    for raw_line in text.splitlines()[:40]:
        line = re.sub(r"\s+", "", raw_line)
        if not line:
            continue
        has_context = any(marker in line for marker in ("表", "报表", "口径", "主体", "范围"))
        has_parenthetical_scope = bool(re.search(r"[（(](?:合并|母公司|子公司)[）)]", line))
        if not has_context and not has_parenthetical_scope:
            continue
        if "合并" in line:
            scopes.add("consolidated")
        if "母公司" in line:
            scopes.add("parent")
        if "子公司" in line:
            scopes.add("subsidiary")
    return scopes


def _explicit_scope_from_text(text: str) -> str:
    """只从明确的表头/口径上下文读取报表口径。"""
    scopes: set[str] = set()
    for raw_line in text.splitlines()[:12]:
        line = re.sub(r"\s+", "", raw_line)
        if not line:
            continue
        if not any(marker in line for marker in ("表", "报表", "口径", "主体", "范围")):
            continue
        if "合并" in line:
            scopes.add("consolidated")
        if "母公司" in line:
            scopes.add("parent")
        if "子公司" in line:
            scopes.add("subsidiary")
    return next(iter(scopes)) if len(scopes) == 1 else "unknown"


def _statement_scope(table_name: str, title_line: str, context_text: str = "") -> str:
    """按表名、标题或明确上下文判断口径；无标记时保持 unknown。"""
    explicit = _explicit_scope_from_text("\n".join((table_name, title_line, context_text)))
    if explicit != "unknown":
        return explicit

    markers = {
        "consolidated": "合并" in table_name or "合并" in title_line,
        "parent": "母公司" in table_name or "母公司" in title_line,
        "subsidiary": "子公司" in table_name or "子公司" in title_line,
    }
    found = [scope for scope, matched in markers.items() if matched]
    if found:
        return found[0] if len(found) == 1 else "unknown"
    return "unknown"


def _local_table_context(text: str) -> dict[str, Any]:
    title = _find_financial_table_title(text)
    markdown_table = _is_markdown_table(text)
    if not title and not markdown_table:
        explicit_scope = _statement_scope("", "", text)
        return {
            "table_name": "",
            "statement_scope": explicit_scope,
            "report_period": _extract_report_period(text),
            "unit": _extract_unit(text),
            "table_id": "",
            "is_table": False,
        }
    table_name, title_line = title or ("", "")
    return {
        "table_name": table_name,
        "statement_scope": _statement_scope(table_name, title_line, text) if title else _statement_scope("", "", text),
        "report_period": _extract_report_period(text),
        "unit": _extract_unit(text),
        "table_id": "",
        "is_table": True,
    }


def _normalize_period_token(raw: str, default_period: str) -> str:
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return default_period
    year_match = re.match(r"(20\d{2})年", compact)
    if year_match and default_period.startswith(f"{year_match.group(1)}年"):
        # 表头常写“2024年”，上下文写“2024年度”时保留已有规范形式。
        if default_period.endswith("年度") and compact == f"{year_match.group(1)}年":
            return default_period
    return compact


def _previous_report_period(default_period: str) -> str:
    match = re.match(r"(20\d{2})(.*)", str(default_period or ""))
    if not match:
        return ""
    return f"{int(match.group(1)) - 1}{match.group(2)}"


def _extract_period_headers(text: str, default_period: str) -> list[str]:
    """提取表头中的列期间；没有明确列期间时返回空列表。"""
    candidates: list[str] = []
    for raw_line in text.splitlines()[:30]:
        line = raw_line.strip()
        if not line:
            continue
        tokens = [_normalize_period_token(match.group(0), default_period) for match in _PERIOD_TOKEN_RE.finditer(line)]
        if not tokens:
            continue
        compact = re.sub(r"\s+", "", line)
        if len(tokens) >= 2:
            return tokens
        if any(marker in compact for marker in ("项目", "科目", "指标", "本期", "上期")):
            candidates = tokens
    return candidates


def _markdown_table_groups(text: str) -> list[list[list[str]]]:
    """按连续 pipe 行拆出 Markdown 表，保留空单元格以维持列位置。"""
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            if current:
                groups.append(current)
                current = []
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            if current:
                groups.append(current)
                current = []
            continue
        current.append(cells)
    if current:
        groups.append(current)
    return groups


def _is_markdown_separator_row(cells: list[str]) -> bool:
    nonempty = [cell for cell in cells if cell]
    return bool(nonempty) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in nonempty)


def _financial_table_column_periods(
    rows: list[list[str]], default_period: str
) -> tuple[list[str], set[int], int]:
    """返回每列期间、应跳过的非金额列，以及第一条财务数据行位置。"""
    first_data = len(rows)
    for index, cells in enumerate(rows):
        if cells and _metric_from_line(cells[0]):
            first_data = index
            break
    if first_data == len(rows):
        return [], set(), first_data

    width = max((len(cells) for cells in rows[:first_data]), default=0)
    periods = ["" for _ in range(width)]
    header_text = ["" for _ in range(width)]
    for cells in rows[:first_data]:
        if _is_markdown_separator_row(cells):
            continue
        for column, cell in enumerate(cells):
            if column >= width:
                break
            compact = re.sub(r"\s+", "", cell)
            header_text[column] += compact
            match = _PERIOD_TOKEN_RE.search(cell)
            if match:
                periods[column] = _normalize_period_token(match.group(0), default_period)

    skip_columns = {0}
    for column, text_value in enumerate(header_text):
        if _FINANCIAL_CHANGE_HEADER_RE.search(text_value) or _FINANCIAL_NON_VALUE_HEADER_RE.fullmatch(text_value):
            skip_columns.add(column)

    # “调整后/调整前”等二级表头通常不重复年份，继承左侧最近的明确期间；
    # 增减率、附注等列被明确跳过，不能参与期间传播。
    active_period = ""
    for column in range(1, width):
        if column in skip_columns:
            continue
        column_header = header_text[column]
        if not periods[column] and _FINANCIAL_PRIOR_PERIOD_HEADER_RE.search(column_header):
            periods[column] = _previous_report_period(default_period)
        elif not periods[column] and _FINANCIAL_CURRENT_PERIOD_HEADER_RE.search(column_header):
            periods[column] = default_period
        if periods[column]:
            active_period = periods[column]
        elif active_period and _FINANCIAL_PERIOD_SUBHEADER_RE.fullmatch(column_header):
            periods[column] = active_period
    return periods, skip_columns, first_data


def _parse_financial_decimal(raw: str) -> Decimal | None:
    number_text = re.sub(r"[(),\s]", "", raw)
    negative = raw.strip().startswith("(") and raw.strip().endswith(")")
    try:
        value = Decimal(number_text)
    except InvalidOperation:
        return None
    return -value if negative else value


def _metric_from_line(line: str) -> tuple[str, str] | None:
    compact = re.sub(r"\s+", "", line)
    for alias, metric in _FINANCIAL_METRIC_ALIASES_SORTED:
        match = re.search(re.escape(alias), compact)
        if match:
            return metric, compact[match.end() :]
    return None


def _financial_row_values(rest: str, table_unit: str) -> list[tuple[str, Decimal, str]]:
    values: list[tuple[str, Decimal, str]] = []
    for match in _FINANCIAL_VALUE_RE.finditer(rest):
        raw = match.group("raw")
        inline_unit = (match.group("unit") or "").strip()
        end_char = rest[match.end() : match.end() + 1]
        if inline_unit in {"%", "％"} or end_char in {"年", "%", "％"}:
            continue
        number = _parse_financial_decimal(raw)
        if number is None:
            continue
        # 四位数年份即使没有“年”后缀，也不应成为财务金额事实。
        if not inline_unit and len(re.sub(r"\D", "", raw)) == 4 and Decimal("1900") <= abs(number) <= Decimal("2100"):
            continue
        unit = inline_unit or table_unit
        scale = _FINANCIAL_UNIT_SCALE.get(unit)
        if scale is None:
            # 未知单位不做换算，但保留原始数值和单位，避免臆测。
            scale = Decimal("1")
        values.append((raw, number * scale, unit))
    return values


def _financial_fact(
    *,
    metric: str,
    raw_value: str,
    canonical_value: Decimal,
    unit: str,
    statement_scope: str,
    report_period: str,
) -> dict[str, str]:
    return {
        "metric": metric,
        "raw_value": raw_value,
        "canonical_value": format(canonical_value, "f"),
        "unit": unit,
        "statement_scope": statement_scope or "unknown",
        "report_period": report_period or "",
    }


def _extract_markdown_financial_facts(
    text: str,
    *,
    statement_scope: str,
    report_period: str,
    unit: str,
) -> list[dict[str, str]]:
    """按 Markdown 表的真实列位置绑定指标、期间与数值。"""
    facts: list[dict[str, str]] = []
    for rows in _markdown_table_groups(text):
        periods, skip_columns, first_data = _financial_table_column_periods(
            rows, report_period
        )
        if first_data >= len(rows):
            continue
        for cells in rows[first_data:]:
            if not cells or _is_markdown_separator_row(cells):
                continue
            metric_info = _metric_from_line(cells[0])
            if not metric_info:
                continue
            metric, _rest = metric_info
            for column, cell in enumerate(cells[1:], start=1):
                if column in skip_columns or not cell:
                    continue
                values = _financial_row_values(cell, unit)
                # 一个表格单元格应只对应一个事实；多个数字往往意味着附注编号
                # 与金额粘连，宁可不建事实也不能猜哪一个是金额。
                if len(values) != 1:
                    continue
                raw_value, canonical_value, value_unit = values[0]
                period = periods[column] if column < len(periods) else ""
                if not period:
                    # 只有一个金额列且没有列期间时，文档/表头报告期仍可可靠
                    # 绑定；多金额列的期初/期末、调整列等无法恢复关系时宁缺毋滥。
                    if len(cells) == 2 and report_period:
                        period = report_period
                    else:
                        continue
                facts.append(
                    _financial_fact(
                        metric=metric,
                        raw_value=raw_value,
                        canonical_value=canonical_value,
                        unit=value_unit,
                        statement_scope=statement_scope,
                        report_period=period,
                    )
                )
    return facts


def _extract_financial_facts(
    text: str,
    *,
    is_table: bool,
    statement_scope: str,
    report_period: str,
    unit: str,
) -> list[dict[str, str]]:
    """从当前表格块的明确行提取可追溯财务事实。"""
    if not is_table:
        return []
    markdown_facts = _extract_markdown_financial_facts(
        text,
        statement_scope=statement_scope,
        report_period=report_period,
        unit=unit,
    )
    markdown_has_financial_rows = any(
        _metric_from_line(cells[0])
        for rows in _markdown_table_groups(text)
        for cells in rows
        if cells
    )
    if markdown_facts or markdown_has_financial_rows:
        # PDF 解析结果常同时包含一份打散的原始文本和一份 Markdown 表。
        # Markdown 表保留列位置，优先使用它，避免同一数字被错绑到其他年度。
        return _stable_financial_facts(markdown_facts)

    periods = _extract_period_headers(text, report_period)
    facts: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or any(mark in line for mark in ("。", "！", "？", "；")):
            continue
        metric_info = _metric_from_line(line)
        if not metric_info:
            continue
        metric, rest = metric_info
        values = _financial_row_values(rest, unit)
        if not values:
            continue
        if periods and len(values) != len(periods):
            # 纯文本表若数值列与期间列数量不同，通常夹有增减率、附注或调整列，
            # 已无法可靠恢复列关系，不能按序硬绑。
            continue
        for index, (raw_value, canonical_value, value_unit) in enumerate(values):
            period = periods[index] if index < len(periods) else report_period
            facts.append(
                _financial_fact(
                    metric=metric,
                    raw_value=raw_value,
                    canonical_value=canonical_value,
                    unit=value_unit,
                    statement_scope=statement_scope,
                    report_period=period,
                )
            )
    return _stable_financial_facts(facts)


def _stable_financial_facts(facts: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """去重并稳定排序，保证索引重建后的元数据字节级可复现。"""
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    for fact in facts or []:
        normalized = {
            "metric": str(fact.get("metric") or ""),
            "raw_value": str(fact.get("raw_value") or ""),
            "canonical_value": str(fact.get("canonical_value") or ""),
            "unit": str(fact.get("unit") or ""),
            "statement_scope": str(fact.get("statement_scope") or "unknown"),
            "report_period": str(fact.get("report_period") or ""),
        }
        key = tuple(normalized[field] for field in (
            "metric", "raw_value", "canonical_value", "unit", "statement_scope", "report_period"
        ))
        unique[key] = normalized
    order = {metric: index for index, metric in enumerate(_FINANCIAL_METRIC_ORDER)}
    return sorted(
        unique.values(),
        key=lambda fact: (
            order.get(fact["metric"], len(order)),
            fact["metric"],
            fact["report_period"],
            fact["raw_value"],
            fact["unit"],
            fact["statement_scope"],
            fact["canonical_value"],
        ),
    )


def _financial_metadata_fields(facts: list[dict[str, str]] | None) -> dict[str, str]:
    stable = _stable_financial_facts(facts)
    if not stable:
        return {}
    order = {metric: index for index, metric in enumerate(_FINANCIAL_METRIC_ORDER)}
    metrics = sorted(
        {fact["metric"] for fact in stable},
        key=lambda metric: (order.get(metric, len(order)), metric),
    )
    return {
        "financial_facts_json": json.dumps(
            stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "financial_metrics": "|".join(metrics),
    }


def _merge_financial_facts(
    left: list[dict[str, str]] | None, right: list[dict[str, str]] | None
) -> list[dict[str, str]]:
    return _stable_financial_facts([*(left or []), *(right or [])])


def _is_weak_report_period(period: str) -> bool:
    """裸年份只能作为弱提示，不能覆盖文档标题中的年度/半年度。"""
    return bool(re.fullmatch(r"20\d{2}年", str(period or "")))


def _has_strong_report_period(text: str) -> bool:
    """判断文本是否明确说明了本表/本报告的报告期。"""
    strong_period = re.compile(
        rf"20\d{{2}}年?(?:{'|'.join(map(re.escape, _REPORT_PERIOD_SUFFIXES))})"
    )
    date_or_range = re.compile(r"20\d{2}年\d{1,2}月\d{1,2}日")
    for raw_line in text.splitlines()[:30]:
        line = re.sub(r"\s+", "", raw_line)
        if not line or not re.search(r"报告期|截至|报告|期间|本期", line):
            continue
        if strong_period.search(line) or date_or_range.search(line):
            return True
    return False


def _is_unit_marker_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    return bool(re.match(r"^(?:单位|金额单位|货币单位|币种及单位)[:：]", compact))


def _is_financial_header_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    has_period = bool(
        _PERIOD_TOKEN_RE.search(compact)
        or re.search(r"本期|上期|期末|期初|调整后|调整前|同比|增减|变动", compact)
    )
    has_label = bool(
        "|" in line
        or re.search(r"项目|科目|指标|金额|发生额|余额", compact)
    )
    return has_period and has_label


def _looks_like_new_table_heading(text: str) -> bool:
    heading_re = re.compile(
        r"^(?:合并|母公司|子公司)?(?:资产负债表|利润表|现金流量表|"
        r"所有者权益变动表|股东权益变动表|主要会计数据|主要财务指标|"
        r"利润构成|现金流量状况|营业收入和营业成本)$"
    )
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", "", raw_line.strip())
        normalized = re.sub(
            r"^(?:第[一二三四五六七八九十百千万\d]+[章节条]|"
            r"[一二三四五六七八九十百千万]+[、.．）)]|\d+[、.．）)])",
            "",
            line,
        )
        if line and len(line) <= 80 and (
            heading_re.fullmatch(line) or heading_re.fullmatch(normalized)
        ):
            return True
    return False


def _is_financial_table_subheading(text: str) -> bool:
    """允许报表内部“一、营业总收入”这类行标题继续沿用表上下文。"""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first or not _heading_info(first):
        return False
    compact = re.sub(r"\s+", "", first)
    return bool(
        any(alias in compact for alias, _metric in _FINANCIAL_METRIC_ALIASES_SORTED)
        or re.search(r"营业总成本|营业利润|利润总额|净利润|综合收益总额", compact)
    )


def _has_metric_with_nearby_values(lines: list[str]) -> bool:
    for index, line in enumerate(lines):
        metric_info = _metric_from_line(line)
        if not metric_info:
            continue
        if _financial_row_values(metric_info[1], ""):
            return True
        for value_line in lines[index + 1 : index + 5]:
            if _financial_row_values(value_line, ""):
                return True
    return False


def _has_financial_table_evidence(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or _looks_like_new_table_heading(text):
        return False
    if _extract_unit(text):
        return True
    if any(_is_financial_header_line(line) for line in lines):
        return True
    return _has_metric_with_nearby_values(lines)


def _is_financial_context_fragment(text: str) -> bool:
    """只接受高置信度的单位/列头/续表/数据行作为跨块上下文。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or _looks_like_new_table_heading(text):
        return False
    if _is_markdown_table(text) and _has_financial_table_evidence(text):
        return True
    if any("续表" in re.sub(r"\s+", "", line) for line in lines):
        return any(len(_NUMBER_RE.findall(line)) >= 2 for line in lines)
    if any(_is_unit_marker_line(line) for line in lines):
        # 单独的单位行是可靠上下文；若混入完整正文则不向后传播。
        return all(
            _is_unit_marker_line(line)
            or _is_financial_header_line(line)
            or not re.search(r"[。！？!?；;]", line)
            for line in lines
        )
    if any(_is_financial_header_line(line) for line in lines):
        return True
    if _has_metric_with_nearby_values(lines) and not any(
        re.search(r"[。！？!?；;]", line) for line in lines
    ):
        return True
    return _is_tabular_continuation(text)


def _scopes_compatible(active_scope: str, local_scope: str) -> bool:
    """未知口径可被显式口径补全，但两个显式不同口径必须断开。"""
    explicit = {"consolidated", "parent", "subsidiary"}
    return not (active_scope in explicit and local_scope in explicit and active_scope != local_scope)


def _contexts_compatible(
    active: StructuredBlock,
    local: dict[str, Any],
    text: str,
    *,
    local_period_is_strong: bool,
) -> bool:
    if active.unit and local["unit"] and active.unit != local["unit"]:
        return False
    if (
        local_period_is_strong
        and active.report_period
        and local["report_period"]
        and active.report_period != local["report_period"]
    ):
        return False
    if not _scopes_compatible(active.statement_scope, local["statement_scope"]):
        return False
    mentions = _scope_mentions(text)
    if len(mentions) > 1:
        return False
    if active.statement_scope in mentions and local["statement_scope"] in {"", "unknown"}:
        return True
    if mentions and active.statement_scope in {"consolidated", "parent", "subsidiary"}:
        return active.statement_scope in mentions
    return True


def _is_tabular_continuation(text: str) -> bool:
    """识别跨页续表的行形态，不以单个数字或普通正文触发继承。"""
    if _is_markdown_table(text):
        return _has_financial_table_evidence(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if re.search(r"续表|续|接上页|下转", "".join(lines)):
        return any(len(_NUMBER_RE.findall(line)) >= 2 for line in lines)
    tabular_lines = [
        line
        for line in lines
        if "\t" in line or re.search(r"\s{2,}", line)
    ]
    numeric_rows = sum(len(_NUMBER_RE.findall(line)) >= 2 for line in tabular_lines)
    return len(tabular_lines) >= 2 and numeric_rows >= 2


def _pages_are_adjacent(left: StructuredBlock, right: StructuredBlock) -> bool:
    return (
        left.page_end is not None
        and right.page_start is not None
        and right.page_start == left.page_end + 1
    )


def _blocks_are_adjacent(left: StructuredBlock, right: StructuredBlock) -> bool:
    if (
        left.page_start is not None
        and right.page_start is not None
        and left.page_start == right.page_start
    ):
        return True
    if left.page_end is None and right.page_start is None:
        return True
    return _pages_are_adjacent(left, right)


def _backfill_table_context(
    annotated: list[StructuredBlock], current: StructuredBlock
) -> None:
    """同一连续表已出现更可靠字段后，补回此前同表的空字段。"""
    if not current.table_id:
        return
    for index, previous in enumerate(annotated):
        if previous.table_id != current.table_id:
            continue
        updates: dict[str, Any] = {}
        if not previous.table_name and current.table_name:
            updates["table_name"] = current.table_name
        if previous.statement_scope == "unknown" and current.statement_scope != "unknown":
            updates["statement_scope"] = current.statement_scope
        if not previous.report_period and current.report_period:
            updates["report_period"] = current.report_period
        if not previous.unit and current.unit:
            updates["unit"] = current.unit
        if updates:
            annotated[index] = replace(previous, **updates)


def _annotate_financial_blocks(
    blocks: list[StructuredBlock], doc_period: str = ""
) -> list[StructuredBlock]:
    """给结构块补财务上下文；正文会立即清除活动表状态。

    doc_period：文档级报告期（页眉提取），作为表格块 report_period 的回退。
    """
    annotated: list[StructuredBlock] = []
    active: StructuredBlock | None = None
    table_number = 0

    for block in blocks:
        # 结构判定（是否表格、表名、期间、单位）与事实提取使用同一输入：
        # 「正文 + 表格 Markdown」。只有最终入库的 chunk.text 用干净正文。
        context_text = _facts_source_text(block.text, block.table_markdown)
        local = _local_table_context(context_text)
        local_period_is_strong = _has_strong_report_period(context_text)
        # 裸年份常来自当前表的列头，不一定是本报告期；明确的文件标题/页眉
        # 是更可靠的文档级回退，尤其能修复“2025 年半年度报告”封面不带年份。
        if doc_period and (
            not local["report_period"]
            or _is_weak_report_period(local["report_period"])
        ):
            local["report_period"] = doc_period
        same_table = False
        if local["table_name"]:
            same_table = bool(
                active
                and active.table_name == local["table_name"]
                and active.section_path == block.section_path
                and _blocks_are_adjacent(active, block)
                and _contexts_compatible(
                    active,
                    local,
                    context_text,
                    local_period_is_strong=local_period_is_strong,
                )
            )
        elif active and active.table_name:
            same_table = bool(
                _blocks_are_adjacent(active, block)
                and _is_financial_context_fragment(context_text)
                and (
                    active.section_path == block.section_path
                    or not _heading_info(
                        next(
                            (line.strip() for line in context_text.splitlines() if line.strip()),
                            "",
                        )
                    )
                    or _is_financial_table_subheading(context_text)
                )
                and _contexts_compatible(
                    active,
                    local,
                    context_text,
                    local_period_is_strong=local_period_is_strong,
                )
            )

        if same_table and active:
            context = {
                "table_name": active.table_name,
                "statement_scope": local["statement_scope"]
                if local["statement_scope"] != "unknown"
                else active.statement_scope,
                "report_period": local["report_period"] or active.report_period,
                "unit": local["unit"] or active.unit,
                "table_id": active.table_id,
                "is_table": True,
            }
        elif local["is_table"]:
            context = local
            context["table_id"] = f"table-{table_number}"
            table_number += 1
        else:
            context = {
                "table_name": "",
                "statement_scope": "unknown",
                "report_period": "",
                "unit": "",
                "table_id": "",
                "is_table": False,
            }

        current = replace(block, **context)
        annotated.append(current)
        if same_table:
            _backfill_table_context(annotated, current)
        active = current if current.is_table else None
    return annotated


def _split_units(text: str) -> list[str]:
    """按自然边界切成原子单元：段落 / 标题 / 表格 / 代码块（空行分隔）。

    目的：分块绝不拦腰切断一个段落或一行表格——召回质量的第一道关卡。
    """
    units = re.split(r"\n\s*\n", text)
    return [u.strip() for u in units if u.strip()]


def _hard_split_unit(unit: str, *, chunk_size: int, overlap: int) -> list[str]:
    """切超长原子单元；Markdown 表格的每片都重复表头，避免丢列含义。"""
    lines = unit.splitlines()
    is_table = len(lines) >= 2 and all(not line.strip() or line.lstrip().startswith("|") for line in lines)
    if is_table:
        header = "\n".join(lines[:2]).strip()
        body_lines = lines[2:]
        pieces: list[str] = []
        current = header
        for line in body_lines:
            candidate = f"{current}\n{line}" if current else line
            if current and len(candidate) > chunk_size:
                pieces.append(current)
                current = f"{header}\n{line}"
            else:
                current = candidate
        if current:
            pieces.append(current)
        if pieces:
            return pieces
    step = max(chunk_size - overlap, 1)
    return [unit[i : i + chunk_size] for i in range(0, len(unit), step) if unit[i : i + chunk_size]]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """结构感知分块：先按段落/标题/表格边界切成原子单元，再聚合到 chunk_size。

    相比纯字符固定切分（会拦腰断句/断表格），这里：
    - 空行分隔的段落、Markdown 标题、表格块、代码块都是"不可再分的单元"，
      只在单元边界处分块，语义完整。
    - 仅当单个单元本身超长（如超长段落/代码块）时才退化为字符硬切（带重叠）。
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    units = _split_units(text)
    chunks: list[str] = []
    current = ""
    for unit in units:
        # 单个单元超长：硬切（带重叠），保留语义完整性之外的大块
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split_unit(unit, chunk_size=chunk_size, overlap=overlap))
            continue
        # 聚合：当前块 + 单元不超过 chunk_size 就合并，否则开新块
        if current and len(current) + len(unit) + 2 > chunk_size:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def _page_span(blocks: list[StructuredBlock]) -> tuple[int | None, int | None]:
    pages = [page for block in blocks for page in (block.page_start, block.page_end) if page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def _common_section_path(left: str, right: str) -> str:
    """返回两个章节路径的公共祖先，供跨小节合并后的元数据使用。"""
    if left == right:
        return left
    left_parts = [part.strip() for part in left.split(" > ") if part.strip()]
    right_parts = [part.strip() for part in right.split(" > ") if part.strip()]
    common: list[str] = []
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        common.append(left_part)
    return " > ".join(common)


def _merge_page_start(left: int | None, right: int | None) -> int | None:
    pages = [page for page in (left, right) if page is not None]
    return min(pages) if pages else None


def _merge_page_end(left: int | None, right: int | None) -> int | None:
    pages = [page for page in (left, right) if page is not None]
    return max(pages) if pages else None


def _facts_source_text(text: str, table_markdown: str) -> str:
    """financial_facts 的提取输入：正文 + 表格 Markdown。

    表格 Markdown 保留了「指标与数字同行」的列结构，是事实提取唯一可靠的
    原料；但它含空列与拆行噪声，只用于提取，不写入 chunk.text。
    """
    if not table_markdown or not table_markdown.strip():
        return text
    return f"{text}\n\n{table_markdown}"


def _financial_spec_fields(block: StructuredBlock) -> dict[str, Any]:
    return {
        "table_name": block.table_name,
        "statement_scope": block.statement_scope,
        "report_period": block.report_period,
        "unit": block.unit,
        "table_id": block.table_id,
        "is_table": block.is_table,
        "financial_facts": _extract_financial_facts(
            _facts_source_text(block.text, block.table_markdown),
            is_table=block.is_table,
            statement_scope=block.statement_scope,
            report_period=block.report_period,
            unit=block.unit,
        ),
    }


def _can_merge_specs(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """表格上下文变化时切断合并，避免把表口径挂到普通正文。"""
    left_table_id = left.get("table_id", "")
    right_table_id = right.get("table_id", "")
    return not (left_table_id or right_table_id) or left_table_id == right_table_id


def _build_chunk_specs(
    blocks: list[StructuredBlock],
    *,
    chunk_size: int,
    overlap: int,
    parent_chunk_size: int | None,
    child_chunk_size: int | None,
    include_parent_chunks: bool,
) -> list[dict[str, Any]]:
    """先生成不带最终 id 的规格，最后统一补 parent/previous/next 引用。"""
    specs: list[dict[str, Any]] = []
    parent_mode = parent_chunk_size is not None or child_chunk_size is not None
    if not parent_mode:
        pending: dict[str, Any] | None = None
        for block in blocks:
            for text in chunk_text(block.text, chunk_size=chunk_size, overlap=overlap):
                current = {
                    "text": text,
                    "chunk_type": "child",
                    "parent_key": "",
                    "section_path": block.section_path,
                    "page_start": block.page_start,
                    "page_end": block.page_end,
                    **_financial_spec_fields(block),
                }
                if (
                    pending
                    and len(pending["text"]) + len(text) + 2 <= chunk_size
                    and _can_merge_specs(pending, current)
                ):
                    pending["text"] = f"{pending['text']}\n\n{text}"
                    pending["section_path"] = _common_section_path(
                        pending["section_path"], block.section_path
                    )
                    pending["page_start"] = _merge_page_start(
                        pending["page_start"], block.page_start
                    )
                    pending["page_end"] = _merge_page_end(
                        pending["page_end"], block.page_end
                    )
                    pending["financial_facts"] = _merge_financial_facts(
                        pending.get("financial_facts"), current.get("financial_facts")
                    )
                else:
                    if pending:
                        specs.append(pending)
                    pending = current
        if pending:
            specs.append(pending)
        return specs

    parent_size = parent_chunk_size or max(chunk_size * 2, chunk_size + 1)
    child_size = child_chunk_size or chunk_size
    # 相邻且章节相同的页块合为一个父章节；没有标题的 PDF 会形成一个
    # 大父上下文，但子块仍按页切出并携带文档的页范围。
    groups: list[list[StructuredBlock]] = []
    for block in blocks:
        if groups and (
            groups[-1][0].section_path == block.section_path
            and groups[-1][0].table_id == block.table_id
        ):
            groups[-1].append(block)
        else:
            groups.append([block])

    parent_number = 0
    for group in groups:
        combined = "\n\n".join(block.text for block in group)
        # 父/子模式不在当前生产路径（parent_chunk_size 默认 None），此处只做
        # 一致性处理：表格 Markdown 同样只作提取原料，不进正文。
        group_table_md = "\n\n".join(
            block.table_markdown for block in group if block.table_markdown
        )
        parent_parts = chunk_text(combined, chunk_size=parent_size, overlap=min(overlap, max(parent_size - 1, 0)))
        start_page, end_page = _page_span(group)
        for parent_text in parent_parts:
            parent_key = f"parent-{parent_number}"
            parent_number += 1
            if include_parent_chunks:
                specs.append(
                    {
                        "text": parent_text,
                        "chunk_type": "parent",
                        "parent_key": parent_key,
                        "section_path": group[0].section_path,
                        "page_start": start_page,
                        "page_end": end_page,
                        **_financial_spec_fields(group[0]),
                        "financial_facts": _extract_financial_facts(
                            _facts_source_text(parent_text, group_table_md),
                            is_table=group[0].is_table,
                            statement_scope=group[0].statement_scope,
                            report_period=group[0].report_period,
                            unit=group[0].unit,
                        ),
                    }
                )
            for child_text in chunk_text(parent_text, chunk_size=child_size, overlap=overlap):
                specs.append(
                    {
                        "text": child_text,
                        "chunk_type": "child",
                        "parent_key": parent_key,
                        "section_path": group[0].section_path,
                        "page_start": start_page,
                        "page_end": end_page,
                        **_financial_spec_fields(group[0]),
                        "financial_facts": _extract_financial_facts(
                            _facts_source_text(child_text, group_table_md),
                            is_table=group[0].is_table,
                            statement_scope=group[0].statement_scope,
                            report_period=group[0].report_period,
                            unit=group[0].unit,
                        ),
                    }
                )
    return specs


def build_chunks(
    path: Path,
    *,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 100,
    kb_id: str = "default",
    ocr_mode: str = "local",
    document_title: str | None = None,
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    include_parent_chunks: bool = True,
) -> list[DocumentChunk]:
    """完整分块管线：解析 → 分块 → 挂元数据。kb_id 标记所属知识库（P3）。

    ocr_mode：图片/扫描 PDF 的 OCR 模式（local/baidu/auto），透传给 parse_document。
    """
    blocks = parse_document_structured(
        path, ocr_mode=ocr_mode, document_title=document_title
    )
    specs = _build_chunk_specs(
        blocks,
        chunk_size=chunk_size,
        overlap=overlap,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        include_parent_chunks=include_parent_chunks,
    )
    ids = [f"{doc_id}-{index}" for index in range(len(specs))]
    parent_ids = {
        spec["parent_key"]: ids[index]
        for index, spec in enumerate(specs)
        if spec["chunk_type"] == "parent"
    }
    chains: dict[str, list[int]] = {"parent": [], "child": []}
    for index, spec in enumerate(specs):
        chains[spec["chunk_type"]].append(index)
    positions = {
        index: position
        for chain in chains.values()
        for position, index in enumerate(chain)
    }

    result: list[DocumentChunk] = []
    for index, spec in enumerate(specs):
        chunk_id = ids[index]
        chain = chains[spec["chunk_type"]]
        position = positions[index]
        previous_id = ids[chain[position - 1]] if position else ""
        next_id = ids[chain[position + 1]] if position + 1 < len(chain) else ""
        parent_id = parent_ids.get(spec["parent_key"], "") if spec["chunk_type"] == "child" else ""
        page_start = spec["page_start"]
        page_end = spec["page_end"]
        metadata: dict[str, Any] = {
            "chunk_id": chunk_id,
            "chunk_type": spec["chunk_type"],
            "parent_id": parent_id,
            "section_path": spec["section_path"],
            "previous_chunk_id": previous_id,
            "next_chunk_id": next_id,
            "table_name": spec["table_name"],
            "statement_scope": spec["statement_scope"],
            "report_period": spec["report_period"],
            "unit": spec["unit"],
            "table_id": spec["table_id"],
            "is_table": spec["is_table"],
        }
        metadata.update(_financial_metadata_fields(spec.get("financial_facts")))
        if page_start is not None:
            metadata["page"] = page_start
            metadata["page_start"] = page_start
        if page_end is not None:
            metadata["page_end"] = page_end
        result.append(
            DocumentChunk(
                text=spec["text"],
                doc_id=doc_id,
                doc_title=path.stem,
                chunk_index=index,
                source_type=path.suffix.lstrip("."),
                kb_id=kb_id,
                page=page_start,
                metadata=metadata,
                chunk_type=spec["chunk_type"],
                parent_id=parent_id,
                section_path=spec["section_path"],
                page_start=page_start,
                page_end=page_end,
                previous_chunk_id=previous_id,
                next_chunk_id=next_id,
            )
        )
    return result


def build_legacy_chunks(
    path: Path,
    *,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 100,
    kb_id: str = "default",
    ocr_mode: str = "local",
) -> list[DocumentChunk]:
    """旧版回滚切片：整篇纯文本切分，不生成页码、章节或邻接元数据。"""
    texts = chunk_text(
        parse_document(path, ocr_mode=ocr_mode),
        chunk_size=chunk_size,
        overlap=overlap,
    )
    source_type = path.suffix.lstrip(".").lower()
    return [
        DocumentChunk(
            text=text,
            doc_id=doc_id,
            doc_title=path.stem,
            chunk_index=index,
            source_type=source_type,
            kb_id=kb_id,
        )
        for index, text in enumerate(texts)
    ]
