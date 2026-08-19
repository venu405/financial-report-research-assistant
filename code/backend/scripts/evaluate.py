#!/usr/bin/env python3
"""客服问答评估脚本（客服改造第 16 项，轻量版）。

跑 YAML 测试集，调 /kb/ask 完整链路，输出：
  护栏分类准确率 / 检索命中率（可回答率）/ FAQ 命中率 / 平均忠实度 / 关键词命中率。

用法（服务需已启动）：
  ./.venv/Scripts/python.exe scripts/evaluate.py --testset testsets/customer_service.yaml

测试集 YAML 格式：
  kb_id: default
  tests:
    - question: "采购超过多少要招投标"
      expect_intent: kb_question      # 可选：期望护栏分类
      expect_keyword: "5万元"          # 可选：期望答案含关键词
      expect_answerable: true         # 可选：期望可回答（未转人工）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _call_ask(base_url: str, kb_id: str, question: str) -> dict[str, Any]:
    import urllib.request

    payload = json.dumps({"question": question, "kb_id": kb_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/kb/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="客服问答评估")
    parser.add_argument("--testset", required=True, help="YAML 测试集路径")
    parser.add_argument("--base-url", default="http://localhost:8000", help="服务地址")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.testset).read_text(encoding="utf-8"))
    kb_id = cfg.get("kb_id", "default")
    tests = cfg.get("tests", [])

    intent_hits = faq_hits = answerable_hits = keyword_hits = 0
    intent_total = faq_total = answerable_total = keyword_total = 0
    scores: list[int] = []
    failures: list[dict[str, Any]] = []

    for t in tests:
        question = t["question"]
        try:
            r = _call_ask(args.base_url, kb_id, question)
        except Exception as exc:
            failures.append({"question": question, "error": str(exc)})
            continue
        meta = r.get("search_meta", {})
        answer = r.get("answer", "")
        score = r.get("score", 0)
        scores.append(score)

        if "expect_intent" in t:
            intent_total += 1
            if meta.get("intent") == t["expect_intent"]:
                intent_hits += 1
        if "expect_answerable" in t:
            answerable_total += 1
            answerable = not meta.get("escalate", False)
            if answerable == t["expect_answerable"]:
                answerable_hits += 1
        if "expect_faq" in t:
            faq_total += 1
            if meta.get("faq_hit", False) == t["expect_faq"]:
                faq_hits += 1
        if "expect_keyword" in t:
            keyword_total += 1
            if t["expect_keyword"] in answer:
                keyword_hits += 1

        time.sleep(0.2)  # 轻量限速，避免打爆

    def pct(hit: int, total: int) -> str:
        return f"{hit}/{total} ({hit / total * 100:.1f}%)" if total else "n/a"

    print("=" * 50)
    print("客服问答评估报告")
    print("=" * 50)
    print(f"测试集: {args.testset}  共 {len(tests)} 条")
    print(f"护栏分类准确率: {pct(intent_hits, intent_total)}")
    print(f"检索命中率(可回答): {pct(answerable_hits, answerable_total)}")
    print(f"FAQ 命中率: {pct(faq_hits, faq_total)}")
    print(f"关键词命中率: {pct(keyword_hits, keyword_total)}")
    if scores:
        print(f"平均忠实度: {sum(scores) / len(scores):.1f}/10")
    if failures:
        print(f"\n失败 {len(failures)} 条：")
        for f in failures:
            print(f"  - {f['question'][:40]}: {f.get('error')}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
