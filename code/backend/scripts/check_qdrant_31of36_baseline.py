"""Compatibility entry point for the historical 31/36 Qdrant gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from scripts.check_qdrant_baseline import BaselineGateError
    from scripts.check_qdrant_baseline import check_candidate as _check_candidate
    from scripts.check_qdrant_baseline import main as _generic_main
except ModuleNotFoundError as exc:  # pragma: no cover - direct script execution
    if exc.name != "scripts":
        raise
    from check_qdrant_baseline import BaselineGateError
    from check_qdrant_baseline import check_candidate as _check_candidate
    from check_qdrant_baseline import main as _generic_main

__all__ = ["BaselineGateError", "check_candidate", "main"]
SUCCESS_MARKER = "QDRANT_31OF36_BASELINE_GATE: PASS"


def check_candidate(
    baseline_manifest: str | Path,
    candidate_report: str | Path,
    candidate_lineage: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    result = _check_candidate(
        baseline_manifest,
        candidate_report,
        candidate_lineage,
        project_root=project_root,
    )
    return {**result, "marker": SUCCESS_MARKER}


def main(argv: list[str] | None = None) -> int:
    return _generic_main(
        argv,
        success_marker=SUCCESS_MARKER,
        failure_marker="QDRANT_31OF36_BASELINE_GATE",
        description="只读检查Qdrant 31/36防退基线（兼容入口）",
    )


if __name__ == "__main__":
    raise SystemExit(main())
