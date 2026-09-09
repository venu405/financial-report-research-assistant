"""Validate a Qdrant RAG report against a versioned pass baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

BASELINE_SCHEMA_VERSION = 1
SUCCESS_MARKER = "QDRANT_BASELINE_GATE: PASS"
FAILURE_MARKER = "QDRANT_BASELINE_GATE"


class BaselineGateError(ValueError):
    """A report, lineage, testset, or baseline failed the gate contract."""


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineGateError(f"{label}不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise BaselineGateError(f"{label}无法读取: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise BaselineGateError(f"{label} JSON损坏: 第{exc.lineno}行第{exc.colno}列") from exc
    if not isinstance(payload, dict):
        raise BaselineGateError(f"{label}顶层不是JSON对象: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BaselineGateError(f"无法计算文件SHA256: {path} ({exc})") from exc
    return digest.hexdigest()


def _resolve(project_root: Path, path_value: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise BaselineGateError(f"{label}路径为空")
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def _normal_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("./")


def _path_matches(value: Any, expected: str) -> bool:
    actual = _normal_path(value)
    target = _normal_path(expected)
    return actual == target or actual.endswith("/" + target)


def _report_entry(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    reports = payload.get("reports")
    if reports is None:
        return dict(payload)
    if not isinstance(reports, list) or len(reports) != 1:
        raise BaselineGateError(f"{label}必须包含且只能包含一个RAG报告")
    report = reports[0]
    if not isinstance(report, dict):
        raise BaselineGateError(f"{label}.reports项不是JSON对象")
    return report


def _load_testset_ids(path: Path, expected_case_count: int) -> tuple[str, ...]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project runtime supplies PyYAML
        raise BaselineGateError("读取题集需要PyYAML依赖") from exc
    if not path.is_file():
        raise BaselineGateError(f"题集不存在: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BaselineGateError(f"题集无法读取或解析: {path} ({exc})") from exc
    tests = data.get("tests") if isinstance(data, dict) else None
    if not isinstance(tests, list):
        raise BaselineGateError("题集缺少tests数组")
    ids: list[str] = []
    for index, item in enumerate(tests):
        case_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(case_id, str) or not case_id.strip():
            raise BaselineGateError(f"题集第{index + 1}项缺少有效id")
        ids.append(case_id)
    if len(ids) != expected_case_count:
        raise BaselineGateError(
            f"题集预期{expected_case_count}题，实际为{len(ids)}题"
        )
    if len(set(ids)) != len(ids):
        raise BaselineGateError("题集存在重复case ID")
    return tuple(sorted(ids))


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(config),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BaselineGateError(f"检索配置无法生成指纹: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BaselineGateError(f"{label}不一致: 实际={actual!r}，基线={expected!r}")


def _validate_report(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_ids: tuple[str, ...],
    expected_report_path: str,
    expected_identity: Mapping[str, Any],
    expected_case_count: int,
    min_passed: int,
) -> tuple[dict[str, Any], set[str], set[str]]:
    report = _report_entry(payload, label)
    _require_equal(report.get("kind"), expected_identity["kind"], f"{label} kind")
    _require_equal(report.get("label"), expected_identity["label"], f"{label} label")
    if not _path_matches(report.get("testset"), expected_report_path):
        raise BaselineGateError(
            f"{label}题集路径不一致: 实际={report.get('testset')!r}，基线={expected_report_path!r}"
        )
    _require_equal(report.get("run_status"), "completed", f"{label} run_status")
    for field in ("total", "planned_total", "completed_total"):
        _require_equal(report.get(field), expected_case_count, f"{label} {field}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise BaselineGateError(f"{label}缺少summary对象")
    _require_equal(summary.get("error_count"), 0, f"{label} error_count")

    results = report.get("results")
    if not isinstance(results, list) or len(results) != expected_case_count:
        raise BaselineGateError(
            f"{label}必须包含{expected_case_count}条results，实际为"
            f"{len(results) if isinstance(results, list) else '非数组'}"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(results):
        case_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(case_id, str) or not case_id.strip():
            raise BaselineGateError(f"{label}第{index + 1}条result缺少有效id")
        if case_id in by_id:
            raise BaselineGateError(f"{label}存在重复case ID: {case_id}")
        by_id[case_id] = item

    actual_ids = set(by_id)
    expected_set = set(expected_ids)
    missing = sorted(expected_set - actual_ids)
    unexpected = sorted(actual_ids - expected_set)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"缺失={','.join(missing)}")
        if unexpected:
            detail.append(f"多出={','.join(unexpected)}")
        raise BaselineGateError(f"{label} case ID集合不一致: {'; '.join(detail)}")

    passed_ids = {case_id for case_id, item in by_id.items() if item.get("passed") is True}
    _require_equal(report.get("passed"), len(passed_ids), f"{label} passed")
    if len(passed_ids) < min_passed:
        raise BaselineGateError(
            f"{label} passed低于最低门槛: 实际={len(passed_ids)}，最低={min_passed}"
        )
    return report, actual_ids, passed_ids


def _validate_lineage(
    payload: Mapping[str, Any],
    *,
    label: str,
    manifest: Mapping[str, Any],
    expected_report_path: str | None = None,
) -> None:
    _require_equal(payload.get("collection"), manifest["collection"], f"{label} collection")
    embedding = manifest["embedding"]
    _require_equal(
        payload.get("embedding_model"), embedding["model"], f"{label} embedding_model"
    )
    _require_equal(payload.get("vector_size"), embedding["vector_size"], f"{label} vector_size")
    _require_equal(payload.get("qdrant_url"), manifest["qdrant_url"], f"{label} qdrant_url")
    if not _path_matches(payload.get("testset"), manifest["testset"]["path"]):
        raise BaselineGateError(f"{label}题集路径与基线不一致: {payload.get('testset')!r}")
    if expected_report_path is not None and not _path_matches(
        payload.get("report_path"), expected_report_path
    ):
        raise BaselineGateError(f"{label}报告路径与基线不一致: {payload.get('report_path')!r}")

    backend = payload.get("backend")
    environment = backend.get("environment") if isinstance(backend, dict) else None
    if not isinstance(environment, dict):
        raise BaselineGateError(f"{label}缺少backend.environment")
    config = manifest["retrieval_config"]
    for key, expected in config.items():
        _require_equal(environment.get(key), expected, f"{label} 检索配置 {key}")
    _require_equal(
        _config_fingerprint({key: environment.get(key) for key in config}),
        manifest["retrieval_config_sha256"],
        f"{label} 检索配置指纹",
    )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[str, Any], tuple[str, ...], set[str]]:
    _require_equal(manifest.get("schema_version"), BASELINE_SCHEMA_VERSION, "基线 schema_version")
    _require_equal(manifest.get("kind"), "qdrant_rag_baseline", "基线 kind")

    expected_case_count = manifest.get("expected_case_count")
    if (
        not isinstance(expected_case_count, int)
        or isinstance(expected_case_count, bool)
        or expected_case_count < 1
    ):
        raise BaselineGateError(f"基线 expected_case_count必须为正整数: {expected_case_count!r}")
    min_passed = manifest.get("min_passed")
    if (
        not isinstance(min_passed, int)
        or isinstance(min_passed, bool)
        or not 1 <= min_passed <= expected_case_count
    ):
        raise BaselineGateError(
            f"基线 min_passed必须在1..{expected_case_count}内: {min_passed!r}"
        )

    testset_info = manifest.get("testset")
    if not isinstance(testset_info, dict):
        raise BaselineGateError("基线缺少testset对象")
    testset_path = _resolve(project_root, testset_info.get("path"), "基线题集")
    expected_ids = _load_testset_ids(testset_path, expected_case_count)
    if manifest.get("expected_case_ids") != list(expected_ids):
        raise BaselineGateError("基线expected_case_ids与题集不一致")
    _require_equal(testset_info.get("sha256"), _sha256(testset_path), "题集SHA256")

    baseline_ids = manifest.get("baseline_passed_case_ids")
    if not isinstance(baseline_ids, list) or baseline_ids != sorted(set(baseline_ids)):
        raise BaselineGateError("基线通过ID必须是排序且无重复的数组")
    if len(baseline_ids) != min_passed or not set(baseline_ids).issubset(set(expected_ids)):
        raise BaselineGateError("基线通过ID数量或范围不符合min_passed契约")

    identity = manifest.get("report_identity")
    if not isinstance(identity, dict) or not identity.get("kind") or not identity.get("label"):
        raise BaselineGateError("基线缺少report_identity")
    config = manifest.get("retrieval_config")
    if not isinstance(config, dict) or not config:
        raise BaselineGateError("基线缺少retrieval_config")
    _require_equal(
        manifest.get("retrieval_config_sha256"),
        _config_fingerprint(config),
        "基线检索配置指纹",
    )

    collection = manifest.get("collection")
    qdrant_url = manifest.get("qdrant_url")
    embedding = manifest.get("embedding")
    if not isinstance(collection, str) or not collection:
        raise BaselineGateError("基线缺少collection")
    if not isinstance(qdrant_url, str) or not qdrant_url:
        raise BaselineGateError("基线缺少qdrant_url")
    if (
        not isinstance(embedding, dict)
        or not isinstance(embedding.get("model"), str)
        or not isinstance(embedding.get("mode"), str)
        or not isinstance(embedding.get("vector_size"), int)
    ):
        raise BaselineGateError("基线embedding配置不完整")

    source_report = manifest.get("source_report")
    if not isinstance(source_report, dict):
        raise BaselineGateError("基线缺少source_report")
    source_report_path = _resolve(project_root, source_report.get("path"), "源报告")
    _require_equal(source_report.get("sha256"), _sha256(source_report_path), "源报告SHA256")
    source_payload = _read_json(source_report_path, "源报告")
    _, _, source_passed = _validate_report(
        source_payload,
        label="源报告",
        expected_ids=expected_ids,
        expected_report_path=testset_info.get("report_path"),
        expected_identity=identity,
        expected_case_count=expected_case_count,
        min_passed=min_passed,
    )
    _require_equal(set(baseline_ids), source_passed, "基线通过ID集合")

    source_lineage = manifest.get("source_lineage")
    if not isinstance(source_lineage, dict):
        raise BaselineGateError("基线缺少source_lineage")
    source_lineage_path = _resolve(project_root, source_lineage.get("path"), "源lineage")
    _require_equal(source_lineage.get("sha256"), _sha256(source_lineage_path), "源lineage SHA256")
    _validate_lineage(
        _read_json(source_lineage_path, "源lineage"),
        label="源lineage",
        manifest=manifest,
        expected_report_path=source_report.get("path"),
    )
    return dict(manifest), expected_ids, set(baseline_ids)


def check_candidate(
    baseline_manifest: str | Path,
    candidate_report: str | Path,
    candidate_lineage: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one completed candidate report without running any service."""
    root = Path(project_root) if project_root is not None else _default_project_root()
    manifest = _read_json(Path(baseline_manifest), "基线清单")
    manifest, expected_ids, baseline_passed_ids = _validate_manifest(
        manifest,
        project_root=root,
    )
    report_payload = _read_json(Path(candidate_report), "候选报告")
    _, candidate_ids, candidate_passed_ids = _validate_report(
        report_payload,
        label="候选报告",
        expected_ids=expected_ids,
        expected_report_path=manifest["testset"]["report_path"],
        expected_identity=manifest["report_identity"],
        expected_case_count=manifest["expected_case_count"],
        min_passed=manifest["min_passed"],
    )
    _validate_lineage(
        _read_json(Path(candidate_lineage), "候选lineage"),
        label="候选lineage",
        manifest=manifest,
    )
    lost = sorted(baseline_passed_ids - candidate_passed_ids)
    if lost:
        raise BaselineGateError(f"候选报告回退了基线通过ID: {','.join(lost)}")
    return {
        "marker": SUCCESS_MARKER,
        "baseline_passed": len(baseline_passed_ids),
        "candidate_passed": len(candidate_passed_ids),
        "candidate_total": len(candidate_ids),
        "newly_passed": sorted(candidate_passed_ids - baseline_passed_ids),
        "report": str(candidate_report),
    }


def main(
    argv: list[str] | None = None,
    *,
    success_marker: str = SUCCESS_MARKER,
    failure_marker: str = FAILURE_MARKER,
    description: str = "只读检查Qdrant防退基线",
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-lineage", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = check_candidate(
            args.baseline_manifest,
            args.candidate_report,
            args.candidate_lineage,
            project_root=args.project_root,
        )
    except BaselineGateError as exc:
        print(f"{failure_marker}: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"{success_marker} baseline_passed={result['baseline_passed']} "
        f"candidate_passed={result['candidate_passed']} "
        f"candidate_total={result['candidate_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
