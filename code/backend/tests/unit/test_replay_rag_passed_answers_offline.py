from __future__ import annotations

from copy import deepcopy

from scripts.replay_rag_passed_answers_offline import (
    HISTORICAL_REPLAY_FIXTURES,
    replay_fixture,
)


def test_replay_manifest_has_four_explicit_audited_fixtures():
    assert len(HISTORICAL_REPLAY_FIXTURES) == 4
    assert {
        fixture["case_id"] for fixture in HISTORICAL_REPLAY_FIXTURES
    } == {
        "real_huawei_2024_revenue",
        "real_jinzhou_port_2025_plan_not_promise",
        "real_longyu_2024_revenue",
        "real_shenlian_2025_h1_operating_cash",
    }
    for fixture in HISTORICAL_REPLAY_FIXTURES:
        assert fixture["source_basis"]
        assert fixture["answer"]
        assert fixture["evidence"]


def test_all_audited_fixtures_pass_current_verification():
    results = [replay_fixture(fixture) for fixture in HISTORICAL_REPLAY_FIXTURES]
    assert all(result["ok"] for result in results), results


def test_replay_reports_case_specific_numeric_regression():
    fixture = deepcopy(
        next(
            item
            for item in HISTORICAL_REPLAY_FIXTURES
            if item["case_id"] == "real_huawei_2024_revenue"
        )
    )
    fixture["answer"] = "2024年营业收入为2,057,608,183.79元。[1]"
    result = replay_fixture(fixture)
    assert result["ok"] is False
    assert result["case_id"] == "real_huawei_2024_revenue"
    assert result["category"] == "numeric_verification"
