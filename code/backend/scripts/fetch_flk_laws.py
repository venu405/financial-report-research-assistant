#!/usr/bin/env python3
"""国家法律法规数据库（flk.npc.gov.cn）批量下载——企业知识库测试数据来源之一。

flk 的新版搜索接口（/law-search/*）对裸 HTTP 请求返回 500，这里复用
cnlaw-cli（PyPI 上维护中的工具，已实测可用）做搜索+下载，产出 docx
（项目 ingest 原生支持 .docx/.pdf/.txt/.md）。

流程：按关键词列表 → cnlaw search npc <kw> → 取第一个"现行有效"命中 →
      cnlaw download npc <id> --output data_kb_test/flk/<标题>.docx

依赖：cnlaw-cli（pip install cnlaw-cli，已装）
用法：
  ./.venv/Scripts/python.exe scripts/fetch_flk_laws.py
  # 或只下指定关键词：  scripts/fetch_flk_laws.py 政府采购 招标投标
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# 企业知识库测试常用的核心法律关键词（标题精确匹配优先）
DEFAULT_KEYWORDS = [
    "政府采购法",
    "招标投标法",
    "公司法",
    "劳动合同法",
    "消费者权益保护法",
    "证券法",
    "企业破产法",
    "民法典",
    "劳动法",
    "商标法",
]
_SLUG_RE = re.compile(r'[\\/:*?"<>|\s]+')


def slug(title: str) -> str:
    return _SLUG_RE.sub("_", title).strip("_")[:80]


def search_law(keyword: str) -> dict | None:
    """搜 flk 标题，返回第一个有 id 的命中（JSONL 逐行解析）。"""
    cmd = ["cnlaw", "search", "npc", keyword, "--scope", "title", "--status", "effective", "--limit", "3", "--format", "jsonl"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60, env=env)
    except Exception as exc:
        print(f"  !! 搜索失败 {keyword}: {exc}")
        return None
    if out.returncode != 0:
        print(f"  !! 搜索失败 {keyword}: {out.stderr.strip()[:200]}")
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("source_document_id"):
            return item
    return None


def download_law(doc_id: str, dest: Path) -> bool:
    cmd = ["cnlaw", "download", "npc", doc_id, "--output", str(dest), "--format", "docx"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=180, env=env)
    except Exception as exc:
        print(f"  !! 下载失败 {doc_id}: {exc}")
        return False
    return out.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="flk 法规批量下载（经 cnlaw）")
    parser.add_argument("keywords", nargs="*", help="关键词，缺省用内置企业法律清单")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data_kb_test", "flk"),
        help="docx 输出目录",
    )
    args = parser.parse_args(argv)
    keywords = args.keywords or DEFAULT_KEYWORDS
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = skip = 0
    for kw in keywords:
        item = search_law(kw)
        if not item:
            print(f"[跳过] 无命中：{kw}")
            fail += 1
            continue
        title = item.get("title") or kw
        dest = out_dir / f"{slug(title)}.docx"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[跳过] 已存在：{title}")
            skip += 1
            continue
        if download_law(item["source_document_id"], dest):
            print(f"[成功] {title} -> {dest.name}")
            ok += 1
        else:
            print(f"[失败] {title}")
            fail += 1

    print("=" * 50)
    print(f"完成：成功 {ok}，失败 {fail}，跳过 {skip}，目录 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())