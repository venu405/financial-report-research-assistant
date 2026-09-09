#!/usr/bin/env python3
"""探针：用真实富润 2024 年报主要会计数据表，验证期间/列绑定是否成立。

用途：设计「调整前/调整后 / 本报告期 / 上年同期」相关单测前，先确认
真实财报的列布局与核验层 `_table_row_period_for_claim` 的假设是否一致，
避免写出与真实结构不符的测试。

用法：
    .venv/Scripts/python.exe scripts/diag_furun_period_binding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))
sys.path.insert(0, str(BACKEND_DIR))

from services.kb.qa_graph import _verify_numeric_claims  # noqa: E402

# 真实 ST富润 2024 年年度报告 page 5「七、近三年主要会计数据和财务指标」
# 列布局：项目 | 2024年 | 2023年 | 本期比上年同期增减(%) | 2022年
REAL_FURUN_TABLE = (
    "主要会计数据\n"
    "| 主要会计数据 | 2024年 | 2023年 | 本期比上年同期增减(%) | 2022年 |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| 营业收入 | 134,241,812.75 | 93,231,677.42 | 43.99 | 194,681,518.35 |\n"
    "| 扣除与主营业务无关的业务收入和不具备商业实质的收入后的营业收入 "
    "| 110,682,912.05 | 72,897,668.51 | 51.83 | 175,609,918.35 |\n"
)

EVIDENCE = [
    {
        "text": REAL_FURUN_TABLE,
        "metadata": {
            "is_table": True,
            "unit": "元",
            "report_period": "2024年度",
            "statement_scope": "consolidated",
        },
    }
]

CASES = [
    # (说明, 答案, 问题, 期望)
    ("2024 取本期值", "营业收入为134,241,812.75元[1]", "浙江富润2024年营业收入是多少？", True),
    ("2024 误取上年同期", "营业收入为93,231,677.42元[1]", "浙江富润2024年营业收入是多少？", False),
    ("2024 误取 2022 值", "营业收入为194,681,518.35元[1]", "浙江富润2024年营业收入是多少？", False),
    ("2023 取上年同期", "营业收入为93,231,677.42元[1]", "浙江富润2023年营业收入是多少？", True),
    ("2023 误取本期值", "营业收入为134,241,812.75元[1]", "浙江富润2023年营业收入是多少？", False),
    ("2022 取第三年", "营业收入为194,681,518.35元[1]", "浙江富润2022年营业收入是多少？", True),
    ("扣除后营业收入(2024)", "扣除后的营业收入为110,682,912.05元[1]",
     "浙江富润2024年扣除与主营业务无关及不具备商业实质的收入后，营业收入是多少？", True),
    ("扣除后误取上年同期", "扣除后的营业收入为72,897,668.51元[1]",
     "浙江富润2024年扣除与主营业务无关及不具备商业实质的收入后，营业收入是多少？", False),
]


# A 股发生前期差错更正/追溯调整时的真实双层表头：
# 调整前/调整后是「上年同期(2023)」的子列，而不是 2023 之后的独立列。
ADJUSTMENT_TABLE = (
    "主要会计数据\n"
    "| 主要会计数据 | 2024年 | 2023年 | | 本期比上年同期增减(%) | 2022年 |\n"
    "| --- | --- | 调整前 | 调整后 | --- | --- |\n"
    "| 营业收入 | 120.00万元 | 110.00万元 | 115.00万元 | 8.26 | 125.00万元 |\n"
)

ADJUSTMENT_EVIDENCE = [
    {
        "text": ADJUSTMENT_TABLE,
        "metadata": {
            "is_table": True,
            "unit": "万元",
            "report_period": "2024年度",
            "statement_scope": "consolidated",
        },
    }
]

ADJUSTMENT_CASES = [
    ("2024 取本期", "营业收入为120.00万元[1]", "虚构企业2024年营业收入是多少？", True),
    ("2024 误取调整前", "营业收入为110.00万元[1]", "虚构企业2024年营业收入是多少？", False),
    ("2023 取调整前", "营业收入为110.00万元[1]", "虚构企业2023年调整前营业收入是多少？", True),
    ("2023 取调整后", "营业收入为115.00万元[1]", "虚构企业2023年调整后营业收入是多少？", True),
    ("2023调整前 取错列", "营业收入为115.00万元[1]", "虚构企业2023年调整前营业收入是多少？", False),
    ("2023调整后 取错列", "营业收入为110.00万元[1]", "虚构企业2023年调整后营业收入是多少？", False),
    ("2022 取第三年", "营业收入为125.00万元[1]", "虚构企业2022年营业收入是多少？", True),
    ("2024 误取 2022 值", "营业收入为125.00万元[1]", "虚构企业2024年营业收入是多少？", False),
]


def main() -> int:
    ok = 0
    total = 0

    print("【表 1】真实富润 page5 布局：项目 | 2024年 | 2023年 | 增减(%) | 2022年")
    print(REAL_FURUN_TABLE)
    print("=" * 78)
    for note, answer, question, expected in CASES:
        result = _verify_numeric_claims(answer, EVIDENCE, question=question)[0]
        passed = result == expected
        ok += passed
        total += 1
        print(f"[{'OK  ' if passed else 'FAIL'}] {note:22s} "
              f"期望={expected!s:5s} 实际={result!s:5s}")

    print()
    print("【表 2】真实双层表头：调整前/调整后挂在 2023 年之下")
    print(ADJUSTMENT_TABLE)
    print("=" * 78)
    for note, answer, question, expected in ADJUSTMENT_CASES:
        result = _verify_numeric_claims(
            answer, ADJUSTMENT_EVIDENCE, question=question
        )[0]
        passed = result == expected
        ok += passed
        total += 1
        print(f"[{'OK  ' if passed else 'FAIL'}] {note:22s} "
              f"期望={expected!s:5s} 实际={result!s:5s}")

    print("=" * 78)
    print(f"通过 {ok}/{total}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
