"""A/B 质量报告比较器的离线测试。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "compare_quality_reports.py"
SPEC = importlib.util.spec_from_file_location("compare_quality_reports", SCRIPT)
assert SPEC and SPEC.loader
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def _report(*, hit: float, latency: float, ids: list[str] | None = None) -> dict:
    ids = ids or ["case-1", "case-2"]
    return {
        "reports": [
            {
                "kind": "rag",
                "schema_version": 2,
                "label": "test",
                "testset": "same.yaml",
                "results": [{"id": item} for item in ids],
                "summary": {
                    "total": len(ids),
                    "pass_rate": 0.8,
                    "error_count": 0,
                    "answerable": {"accuracy": 0.9, "denominator": len(ids)},
                    "escalate_rate": 0.1,
                    "avg_citations": 2.0,
                    "latency": {
                        "p50_s": latency / 2,
                        "p95_s": latency,
                        "max_s": latency,
                        "avg_s": latency / 2,
                    },
                    "retrieval": {
                        "contexts": {
                            "hit@5": hit,
                            "hit@5_denominator": len(ids),
                            "recall@5": hit,
                            "recall@5_denominator": len(ids),
                            "evidence_rate": hit,
                            "evidence_rate_denominator": len(ids),
                            "page": {"hit_rate": hit, "denominator": len(ids)},
                        }
                    },
                },
            }
        ]
    }


def _row(diff: dict, name: str) -> dict:
    return next(item for item in diff["metrics"] if item["name"] == name)


def test_compare_uses_actual_summary_paths_and_directions():
    diff = compare.compare_reports(_report(hit=0.5, latency=10), _report(hit=0.8, latency=12))

    assert diff["comparable"] is True
    assert _row(diff, "retrieval.contexts.hit@5")["abs_delta"] == 0.3
    assert _row(diff, "retrieval.contexts.hit@5")["verdict"] == "改善"
    assert _row(diff, "latency.p95_s")["direction"] == "low"
    assert _row(diff, "latency.p95_s")["verdict"] == "恶化"


def test_compare_warns_when_case_sets_and_denominators_differ():
    baseline = _report(hit=0.5, latency=10, ids=["a", "b"])
    candidate = _report(hit=0.8, latency=9, ids=["a", "c", "d"])

    diff = compare.compare_reports(baseline, candidate)

    assert any("案例集合不同" in warning for warning in diff["warnings"])
    assert any("分母不同" in warning for warning in diff["warnings"])


def test_gates_prevent_recall_gain_from_hiding_latency_regression():
    diff = compare.compare_reports(_report(hit=0.5, latency=10), _report(hit=0.8, latency=12))

    gates = compare.evaluate_gates(
        diff,
        ["retrieval.contexts.hit@5=0.20"],
        ["latency.p95_s=1.0"],
    )

    assert gates["configured"] is True
    assert gates["passed"] is False
    assert any("latency.p95_s" in failure for failure in gates["failures"])


def test_missing_or_legacy_summary_is_not_comparable():
    diff = compare.compare_reports({"reports": [{"kind": "rag", "results": []}]}, _report(hit=0.8, latency=9))

    assert diff["comparable"] is False
    assert diff["metrics"]
    assert all(row["status"] != "ok" for row in diff["metrics"])
    assert diff["warnings"]


def test_console_encoding_is_forced_to_utf8(monkeypatch):
    calls: list[dict] = []

    class Stream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(compare.sys, "stdout", Stream())
    monkeypatch.setattr(compare.sys, "stderr", Stream())

    compare.configure_console_encoding()

    assert calls == [
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "utf-8", "errors": "replace"},
    ]
