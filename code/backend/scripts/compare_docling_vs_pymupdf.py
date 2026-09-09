"""Docling vs 现有 pymupdf 解析对比：格力电器 2023 年报「主要会计数据」表。

只读验证：不改任何代码，不写任何向量库。输出两边对第 7 页表格的还原结果。
"""
import sys
from pathlib import Path

# Pass the local evaluation PDF as the first argument.  Keep the fallback
# relative so this diagnostic never leaks a developer machine path.
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "data_kb_test/cninfo_annual/格力电器-2023年年度报告.pdf"
)
TARGET_PAGE = 7  # 测试集 expected_pages


def current_pymupdf(path: Path, page_no: int) -> str:
    import pymupdf
    with pymupdf.open(path) as doc:
        page = doc[page_no - 1]
        body = page.get_text("blocks")
        lines = [(float(b[0]), float(b[3]), str(b[4]).strip()) for b in body if str(b[4]).strip()]
        text = "\n".join(l[2] for l in sorted(lines, key=lambda x: (round(x[1], 1), x[0])))
        tables = []
        try:
            for t in page.find_tables().tables:
                rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in (t.extract() or [])]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2:
                    continue
                hdr = rows[0]
                md = "| " + " | ".join(hdr) + " |\n| " + " | ".join(["---"] * len(hdr)) + " |\n"
                md += "\n".join("| " + " | ".join(r + [""] * (len(hdr) - len(r)))[:len(hdr)] + " |" for r in rows[1:])
                tables.append(md)
        except Exception:
            pass
        return text + ("\n\n" + "\n\n".join(tables) if tables else "")


def docling_parse(path: Path, page_no: int) -> str:
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    result = conv.convert(str(path))
    doc = result.document
    out = []
    # 表格
    for t in doc.tables:
        # 通过表格所在页过滤
        try:
            if (t.prov or []) and t.prov[0].page_no == page_no:
                out.append(t.export_to_markdown())
        except Exception:
            pass
    return "\n\n".join(out)


if __name__ == "__main__":
    print("=" * 30, "当前 pymupdf 输出（第 %d 页）" % TARGET_PAGE, "=" * 30)
    try:
        cur = current_pymupdf(PDF, TARGET_PAGE)
        print(cur[:1500])
    except Exception as e:
        print("pymupdf 失败:", e)

    print()
    print("=" * 30, "Docling 输出（第 %d 页表格）" % TARGET_PAGE, "=" * 30)
    try:
        doc = docling_parse(PDF, TARGET_PAGE)
        print(doc[:1500] if doc else "（该页无表格或未解析到）")
    except Exception as e:
        print("Docling 失败:", e)
