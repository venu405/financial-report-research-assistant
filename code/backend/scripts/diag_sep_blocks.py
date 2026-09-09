"""诊断：对比旧行为与分离模式下「块级」事实提取的差异。

重点回答：分离模式为什么带 metrics 的块比旧行为少？
逐项统计：块数、is_table 块数、带 table_markdown 的块数、提取到 facts 的块数。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

TARGET_PDF = BACKEND_DIR / "data_kb_test" / "cninfo" / "伊泰Ｂ股-内蒙古伊泰煤炭股份有限公司2025年半年度报告.pdf"

MODES = [
    ("A 旧行为", {"KB_USE_FIND_TABLES": "1", "KB_SEPARATE_TABLE_MD": "0"}),
    ("C 分离  ", {"KB_USE_FIND_TABLES": "1", "KB_SEPARATE_TABLE_MD": "1"}),
]


def run(mode_env: dict[str, str]):
    for key, value in mode_env.items():
        os.environ[key] = value
    for name in [m for m in list(sys.modules) if m.startswith("services.kb.ingest")]:
        del sys.modules[name]
    from services.kb import ingest  # noqa: PLC0415

    blocks = ingest.parse_document_structured(TARGET_PDF)
    rows = []
    for block in blocks:
        facts = ingest._extract_financial_facts(
            ingest._facts_source_text(block.text, block.table_markdown),
            is_table=block.is_table,
            statement_scope=block.statement_scope,
            report_period=block.report_period,
            unit=block.unit,
        )
        rows.append(
            {
                "page": block.page_start,
                "is_table": block.is_table,
                "table_name": block.table_name,
                "unit": block.unit,
                "period": block.report_period,
                "len_text": len(block.text),
                "len_md": len(block.table_markdown),
                "n_facts": len(facts),
                "facts": facts,
            }
        )
    return rows


def main() -> int:
    out = {}
    for label, env in MODES:
        rows = run(env)
        out[label.strip()] = rows
        n_table = sum(1 for r in rows if r["is_table"])
        n_md = sum(1 for r in rows if r["len_md"] > 0)
        n_facts = sum(1 for r in rows if r["n_facts"] > 0)
        total_facts = sum(r["n_facts"] for r in rows)
        print(
            f"{label}: 块数={len(rows)}  is_table块={n_table}  带md块={n_md}  "
            f"有facts块={n_facts}  facts总数={total_facts}"
        )

    print("\n--- 旧行为有 facts 的块（前 12）---")
    for r in out["A 旧行为"]:
        if r["n_facts"]:
            print(
                f"  p{r['page']} is_table={r['is_table']} 表='{r['table_name'][:14]}' "
                f"len_text={r['len_text']} n_facts={r['n_facts']}"
            )
    print("--- 分离模式有 facts 的块（前 12）---")
    for r in out["C 分离"]:
        if r["n_facts"]:
            print(
                f"  p{r['page']} is_table={r['is_table']} 表='{r['table_name'][:14]}' "
                f"len_text={r['len_text']} len_md={r['len_md']} n_facts={r['n_facts']}"
            )

    print("\n--- 第 6 页（目标页）块对比 ---")
    for label in ["A 旧行为", "C 分离"]:
        for r in out[label]:
            if r["page"] == 6:
                print(f"  {label}: len_text={r['len_text']} len_md={r['len_md']} is_table={r['is_table']} n_facts={r['n_facts']}")

    dump = {k: [{kk: vv for kk, vv in r.items() if kk != "facts"} for r in v] for k, v in out.items()}
    Path(BACKEND_DIR / "migration-record" / "sep-block-diag.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n明细已写入 migration-record/sep-block-diag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
