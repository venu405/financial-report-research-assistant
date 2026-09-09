"""把阶段 0 的 2024 财务指标样本幂等导入指定 SQLite 指标库。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = BACKEND_ROOT / "data" / "financial_metrics_seed_2024.json"
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from services.kb.financial_metric_store import FinancialMetricStore  # noqa: E402


def seed_metrics(seed_path: str | Path, db_path: str | Path, kb_id: str = "default") -> dict[str, int]:
    """导入样本；重复键已存在时跳过，不覆盖已有指标或修订。"""
    with Path(seed_path).open("r", encoding="utf-8") as handle:
        records: list[dict[str, Any]] = json.load(handle)
    store = FinancialMetricStore(db_path)
    inserted = 0
    skipped = 0
    try:
        for source in records:
            data = dict(source)
            data["kb_id"] = kb_id
            existing, _ = store.list(
                kb_id=kb_id,
                company_name=data["company_name"],
                report_period=data["report_period"],
                metric_code=data["metric_code"],
                limit=200,
            )
            if existing:
                skipped += 1
                continue
            store.create(data)
            inserted += 1
    finally:
        store.close()
    return {"inserted": inserted, "skipped": skipped, "total": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description="幂等导入 2024 财务指标样本")
    parser.add_argument("--db-path", type=Path, required=True, help="目标指标 SQLite 路径；必须显式指定，避免误写业务库")
    parser.add_argument("--kb-id", default="default", help="写入的知识库 ID，默认 default")
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH, help="种子 JSON 路径")
    args = parser.parse_args()
    result = seed_metrics(args.seed_path, args.db_path, args.kb_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
