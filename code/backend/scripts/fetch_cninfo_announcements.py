#!/usr/bin/env python3
"""巨潮资讯（cninfo.com.cn）公告/年报批量下载——企业知识库测试数据来源之二。

上市公司定期报告（年报/半年报/季报）PDF，用于测试本项目的 PDF 解析 + OCR 链路。

接口（实测可用，2026-08）：
  POST http://www.cninfo.com.cn/new/hisAnnouncement/query
  表单参数：pageNum / pageSize / column / tabName=fulltext / category / seDate ...
  返回 JSON：{ announcements: [{announcementTitle, secName, adjunctUrl, ...}], totalAnnouncement }
  PDF 下载地址 = http://static.cninfo.com.cn/ + adjunctUrl

用法：
  ./.venv/Scripts/python.exe scripts/fetch_cninfo_announcements.py \
      --category category_ndbg_szsh --limit 20 --se-date 2025-01-01~2025-12-31

分类码（category）：
  category_ndbg_szsh   年度报告（PDF 大，10-30MB/份）
  category_bndbg_szsh   半年度报告
  category_yjdbg_szsh   第一季度/第三季度报告
  留空则抓全量公告（混合类型）

column：szse（深市）/ sse（沪市）/ bj（北交所），默认 szse。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
DOWNLOAD_BASE = "http://static.cninfo.com.cn/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
    "Content-Type": "application/x-www-form-urlencoded",
}
_SLUG_RE = re.compile(r'[\\/:*?"<>|\s]+')


def query(page_num: int, page_size: int, *, category: str, column: str, se_date: str) -> dict:
    form = {
        "pageNum": page_num,
        "pageSize": page_size,
        "column": column,
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": "",
        "secid": "",
        "category": category,
        "trade": "",
        "seDate": se_date,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    data = urllib.parse.urlencode(form).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(QUERY_URL, data=data, headers=HEADERS, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return __import__("json").loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))  # 退避重试（防 502/限流）
    raise last_exc or RuntimeError("query failed")


def slug(title: str) -> str:
    return _SLUG_RE.sub("_", title).strip("_")[:80]


def download(adjunct_url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False  # 已下载过，跳过
    url = DOWNLOAD_BASE + adjunct_url
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise last_exc or RuntimeError("download failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="巨潮公告/年报 PDF 批量下载")
    parser.add_argument("--category", default="category_ndbg_szsh", help="分类码，默认年报")
    parser.add_argument("--column", default="szse", choices=["szse", "sse", "bj"], help="交易所")
    parser.add_argument("--se-date", default="2025-01-01~2025-12-31", help="公告日期区间")
    parser.add_argument("--limit", type=int, default=20, help="最多下载份数")
    parser.add_argument("--page-size", type=int, default=10, help="每页查询条数")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data_kb_test", "cninfo"),
        help="PDF 输出目录",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()

    page = 1
    downloaded = skipped = 0
    while downloaded < args.limit:
        resp = query(page, args.page_size, category=args.category, column=args.column, se_date=args.se_date)
        items = resp.get("announcements") or []
        if not items:
            break
        for it in items:
            if downloaded >= args.limit:
                break
            title = it.get("announcementTitle", "")
            sec_name = it.get("secName", "")
            adjunct = it.get("adjunctUrl", "")
            if not adjunct:
                continue
            name = f"{slug(sec_name)}-{slug(title)}.pdf"
            dest = out_dir / name
            try:
                new = download(adjunct, dest)
            except Exception as exc:
                print(f"  !! {title}: {exc}")
                skipped += 1
                continue
            if new:
                downloaded += 1
                print(f"  [{downloaded}] {sec_name} | {title} -> {name}")
            else:
                skipped += 1
            time.sleep(0.5)  # 礼貌限速
        page += 1
        if page > 50:  # 安全上限
            break

    print("=" * 50)
    print(f"完成：新下载 {downloaded} 份，跳过 {skipped} 份，目录 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())