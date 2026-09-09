#!/usr/bin/env python3
"""验证「结构化财务路由期间过滤」是否把正确块排除掉。

假设（来自 Chroma 全库 report_period 分布）：ingest 写入的期间格式不统一，
同一年度报告既有 '2024年度' 也有 '2024年'；而 `_financial_route_period()`
只产出一种格式（'2024年度' / '2025年半年度'）。路由用
`{"report_period": period}` 做**等值硬过滤**，格式不一致的块会被整批排除，
结构化通道因此对这类题静默失效（退化为纯向量召回 → 块级定位失败 → 拒答）。

本脚本在 Chroma 隔离副本上**只读**复算：
  1. 真实调用 retriever._financial_route_period / _financial_route_doc_ids
  2. 按当前等值过滤条件筛出候选池，检查目标块是否在内
  3. 改用「期间归一化」比较（2024年度 ≡ 2024年），再检查一次
对比两者的候选池差异，即可确认假设。

用法：
    .venv/Scripts/python.exe scripts/diag_route_period_filter.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))

# 必须在导入 retriever 之前打开开关，因为 _structured_fin_route 是模块级常量。
os.environ["KB_STRUCTURED_FIN_ROUTE"] = "1"

CHROMA_DIR = ".rag_eval/chroma_repro_26_current_20260831/chroma_data"
COLLECTION = "enterprise_kb"

# （题号, 问句, doc_title 关键字, 目标数字串, 期望页码）
CASES = [
    (
        "real_furun_2024_adjusted_revenue_synonym",
        "ST富润2024年扣除与主营业务无关及不具备商业实质的收入后，营业收入是多少？",
        "富润",
        "110,682,912.05",
        5,
    ),
    (
        "real_shenlian_2025_h1_rnd_ratio_synonym",
        "申联生物2025年上半年研发投入占营业收入的比例是多少？",
        "申联",
        "22.32",
        7,
    ),
    (
        "real_yutai_2025_h1_revenue",
        "裕太微电子2025年上半年营业收入具体是多少？",
        "裕太微",
        "221,828,689.43",
        8,
    ),
]

# 与 retriever 保持一致
_FIN_KEY_SECTIONS = ("主要会计数据", "主要财务指标")


def normalize_period(value: str) -> tuple[str, bool]:
    """把 '2024年度' / '2024年' / '2025年半年度' 归一到 (年份, 是否半年度)。

    返回 ('', False) 表示无法识别。
    """
    text = str(value or "").strip()
    match = re.search(r"(20\d{2})\s*年", text)
    if not match:
        return "", False
    year = match.group(1)
    half = bool(re.search(r"半年|中期|半年度|1-6", text))
    return year, half


def main() -> int:
    from services.kb import retriever as R

    print(f"KB_STRUCTURED_FIN_ROUTE 生效状态: {R._structured_fin_route}")
    print(f"模块常量 _structured_fin_route = {R._structured_fin_route!r}\n")

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(BACKEND_DIR / CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION)
    total = collection.count()
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    print(f"集合 {COLLECTION}：{total} 条，已读取 {len(documents)} 条（只读）\n")

    # 构造与 retriever 期望一致的 corpus
    corpus: list[dict] = []
    for text, meta in zip(documents, metadatas):
        meta = meta or {}
        corpus.append({"text": text, "metadata": meta, "page_content": text})

    for case_id, question, company_key, needle, expect_page in CASES:
        print("=" * 92)
        print(f"【{case_id}】")
        print(f"  问句: {question}")
        period = R._financial_route_period(question)
        doc_ids = list(R._financial_route_doc_ids(question, corpus))
        print(f"  _financial_route_period() -> {period!r}")
        print(f"  公司路由 doc_id 命中 {len(doc_ids)} 个")

        # 该公司名下所有块的期间分布
        company_blocks = [
            (t, m)
            for t, m in zip(documents, metadatas)
            if company_key in str((m or {}).get("doc_title") or "")
        ]
        periods: dict[str, int] = {}
        for _t, m in company_blocks:
            key = str((m or {}).get("report_period") or "")
            periods[key] = periods.get(key, 0) + 1
        print(f"  「{company_key}」块数 {len(company_blocks)}，period 分布: {periods}")

        # 目标块
        targets = [
            (t, m)
            for t, m in company_blocks
            if needle in str(t or "")
        ]
        print(f"  含目标数字 {needle!r} 的块: {len(targets)} 个")
        for t, m in targets:
            m = m or {}
            print(
                f"    page={m.get('page')} period={m.get('report_period')!r} "
                f"is_table={m.get('is_table')} section={str(m.get('section_path'))[:38]!r}"
            )

        # ---- 当前实现：等值硬过滤 ----
        sections = sorted({
            str((m or {}).get("section_path"))
            for _t, m in zip(documents, metadatas)
            if any(k in str((m or {}).get("section_path") or "") for k in _FIN_KEY_SECTIONS)
        })
        strict_pool = [
            (t, m)
            for t, m in company_blocks
            if str((m or {}).get("report_period") or "") == period
            and bool((m or {}).get("is_table"))
            and any(k in str((m or {}).get("section_path") or "") for k in _FIN_KEY_SECTIONS)
            and str((m or {}).get("doc_id") or "") in doc_ids
        ]
        strict_hit = any(needle in str(t or "") for t, _m in strict_pool)
        print(f"\n  [当前实现] report_period == {period!r} 等值过滤")
        print(f"    候选池 {len(strict_pool)} 块；目标数字在内: {strict_hit}")
        for t, m in strict_pool[:8]:
            m = m or {}
            mark = "★目标★" if needle in str(t or "") else ""
            print(
                f"      page={m.get('page')} period={m.get('report_period')!r} "
                f"section={str(m.get('section_path'))[:34]!r} {mark}"
            )

        # ---- 归一化过滤：年 + 半年度 ----
        want_year, want_half = normalize_period(period)
        norm_pool = [
            (t, m)
            for t, m in company_blocks
            if normalize_period(str((m or {}).get("report_period") or "")) == (want_year, want_half)
            and bool((m or {}).get("is_table"))
            and any(k in str((m or {}).get("section_path") or "") for k in _FIN_KEY_SECTIONS)
            and str((m or {}).get("doc_id") or "") in doc_ids
        ]
        norm_hit = any(needle in str(t or "") for t, _m in norm_pool)
        print(f"\n  [归一化后] period 归一到 ({want_year!r}, 半年度={want_half})")
        print(f"    候选池 {len(norm_pool)} 块；目标数字在内: {norm_hit}")
        for t, m in norm_pool[:8]:
            m = m or {}
            mark = "★目标★" if needle in str(t or "") else ""
            print(
                f"      page={m.get('page')} period={m.get('report_period')!r} "
                f"section={str(m.get('section_path'))[:34]!r} {mark}"
            )

        verdict = "假设成立（等值过滤漏掉目标块，归一化后找回）" if (
            not strict_hit and norm_hit
        ) else "假设不成立或另有原因"
        print(f"\n  >>> 结论: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
