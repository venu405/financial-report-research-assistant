"""关卡 0：分离方案单文档验证（不入库，不写任何 collection）。

对同一份 PDF 跑三种解析模式，对比：
  A 旧行为   KB_USE_FIND_TABLES=1, KB_SEPARATE_TABLE_MD=0  → 表格 Markdown 进正文
  B 去表格   KB_USE_FIND_TABLES=0, KB_SEPARATE_TABLE_MD=0  → 正文干净但事实提取失效
  C 分离     KB_USE_FIND_TABLES=1, KB_SEPARATE_TABLE_MD=1  → 正文干净 + 事实从正文+Markdown 提取

判据：
  1. C 的 chunk.text 里不应再出现 Markdown 表格噪声（" | " / "|---"）。
  2. C 的 financial_facts 应恢复到 A 的水平（数量和命中目标数字）。
  3. C 的 financial_metrics 元数据应非空（窗口三过滤依赖它）。
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
TARGET_METRIC = "经营活动产生的现金流量净额"
TARGET_VALUE = "3,898,863,782.71"
TARGET_VALUE_COMPACT = "3898863782.71"
TARGET_PAGE = 6

MODES = [
    ("A 旧行为", {"KB_USE_FIND_TABLES": "1", "KB_SEPARATE_TABLE_MD": "0"}),
    ("B 去表格", {"KB_USE_FIND_TABLES": "0", "KB_SEPARATE_TABLE_MD": "0"}),
    ("C 分离  ", {"KB_USE_FIND_TABLES": "1", "KB_SEPARATE_TABLE_MD": "1"}),
]


def has_markdown_noise(text: str) -> bool:
    return " | " in text or "|---" in text


def run_mode(mode_env: dict[str, str]):
    for key, value in mode_env.items():
        os.environ[key] = value
    # 每次重新导入，确保模块级缓存不串味
    for name in [m for m in list(sys.modules) if m.startswith("services.kb.ingest")]:
        del sys.modules[name]
    from services.kb.ingest import build_chunks  # noqa: PLC0415

    chunks = build_chunks(TARGET_PDF, doc_id="verify-yitai", kb_id="cninfo_report")
    total = len(chunks)
    noisy = sum(1 for c in chunks if has_markdown_noise(c.text))
    # financial_facts 以 JSON 字符串落在 metadata.financial_facts_json
    fact_lists = []
    for c in chunks:
        raw = c.metadata.get("financial_facts_json")
        if not raw:
            continue
        try:
            fact_lists.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    with_facts = len(fact_lists)
    with_metrics = sum(1 for c in chunks if c.metadata.get("financial_metrics"))
    fact_count = sum(len(f) for f in fact_lists)
    # 同一批 facts 会被复制到同一块的每个 chunk 上，去重后才是真实事实量
    distinct = {
        (
            f.get("metric"),
            f.get("raw_value"),
            f.get("unit"),
            f.get("report_period"),
            f.get("statement_scope"),
        )
        for facts in fact_lists
        for f in facts
    }

    hit_facts = 0
    hit_text = 0
    for facts in fact_lists:
        for fact in facts:
            if TARGET_METRIC in str(fact.get("metric", "")) and (
                TARGET_VALUE_COMPACT in str(fact.get("canonical_value", "")).replace(",", "")
                or TARGET_VALUE in str(fact.get("raw_value", ""))
            ):
                hit_facts += 1
    for chunk in chunks:
        if chunk.page == TARGET_PAGE and (
            TARGET_VALUE in chunk.text or TARGET_VALUE_COMPACT in chunk.text.replace(" ", "")
        ):
            hit_text += 1
    return {
        "chunks": total,
        "noisy_chunks": noisy,
        "chunks_with_facts": with_facts,
        "chunks_with_metrics": with_metrics,
        "facts_total": fact_count,
        "facts_distinct": len(distinct),
        "target_in_facts": hit_facts,
        "target_in_text_p6": hit_text,
    }


def main() -> int:
    if not TARGET_PDF.is_file():
        print(f"找不到目标 PDF: {TARGET_PDF}")
        return 1
    print(f"目标文件: {TARGET_PDF.name}")
    print(f"判据指标: {TARGET_METRIC} = {TARGET_VALUE}（第 {TARGET_PAGE} 页）\n")

    results = {}
    for label, env in MODES:
        results[label.strip()] = run_mode(env)

    header = f"{'模式':<8}{'chunks':>8}{'噪声块':>8}{'有facts块':>10}{'有metrics块':>12}{'facts去重':>10}{'目标数字(facts)':>16}{'目标数字(正文p6)':>18}"
    print(header)
    print("-" * len(header))
    for label in [m[0].strip() for m in MODES]:
        r = results[label]
        print(
            f"{label:<8}{r['chunks']:>8}{r['noisy_chunks']:>8}{r['chunks_with_facts']:>10}"
            f"{r['chunks_with_metrics']:>12}{r['facts_distinct']:>10}{r['target_in_facts']:>16}{r['target_in_text_p6']:>18}"
        )

    print("\n判定：")
    a, b, c = results["A 旧行为"], results["B 去表格"], results["C 分离"]
    ok1 = c["noisy_chunks"] == 0
    ok2 = c["facts_distinct"] >= a["facts_distinct"]
    ok3 = c["target_in_facts"] > 0
    ok4 = c["chunks_with_metrics"] > 0
    print(f"  1. 分离模式 chunk.text 无 Markdown 噪声: {'通过' if ok1 else '未通过'} (噪声块={c['noisy_chunks']})")
    print(
        f"  2. 去重事实数恢复到旧行为水平: {'通过' if ok2 else '未通过'} "
        f"(C={c['facts_distinct']} vs A={a['facts_distinct']} vs B={b['facts_distinct']})"
    )
    print(f"  3. 目标数字被 facts 绑定: {'通过' if ok3 else '未通过'} (C={c['target_in_facts']}, A={a['target_in_facts']}, B={b['target_in_facts']})")
    print(f"  4. financial_metrics 元数据非空: {'通过' if ok4 else '未通过'} (C={c['chunks_with_metrics']})")
    return 0 if (ok1 and ok2 and ok3 and ok4) else 2


if __name__ == "__main__":
    raise SystemExit(main())
