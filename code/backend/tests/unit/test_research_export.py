from __future__ import annotations

from services.kb.research_export import render_company_analysis, render_peer_comparison


def test_company_export_is_deterministic_and_keeps_missing_and_negative_values():
    result = {
        "company": {"name": "华微测试公司", "code": None},
        "report_period": "2024年度",
        "comparison_period": None,
        "metrics": [
            {
                "metric_code": "revenue",
                "metric_name": "营业收入",
                "current": {"normalized_value": "-100", "source_title": "unused"},
                "comparison": None,
                "change_amount": None,
                "change_rate_percent": None,
                "comparable": False,
                "comparison_note": "比较期指标缺失",
            }
        ],
        "profit_cash_flow": {
            "net_profit_value": None,
            "operating_cash_flow_value": "-20",
            "difference": None,
            "relation": "unavailable",
            "comparable": False,
            "note": "数据不足",
        },
        "disclosures": [
            {
                "metric_code": "revenue",
                "source_title": "2024年度报告",
                "source_page": 8,
                "source_page_end": 9,
                "source_text": "营业收入为-100元",
            },
            {
                "metric_code": "total_assets",
                "source_title": "无原文报告",
                "source_page": None,
                "source_page_end": None,
                "source_text": "",
            },
        ],
        "pending_items": [{"code": "missing_comparison_metric", "metric_code": "revenue", "message": "比较期缺失"}],
    }

    markdown = render_company_analysis(
        result,
        kb_id="default",
        generated_at="2026-09-07T00:00:00Z",
    )

    assert "# 单公司财报研究底稿" in markdown
    assert "UTC生成时间：2026-09-07T00:00:00Z" in markdown
    assert "华微测试公司" in markdown
    assert "-100" in markdown
    assert "未提供/不可计算" in markdown
    assert "原文：营业收入为-100元" in markdown
    assert "## 来源文档与页码" in markdown
    assert "## 待核实事项" in markdown
    assert "不构成投资建议" in markdown
    assert markdown.count("原文：") == 1


def test_peer_export_contains_sample_boundary_and_source_pages():
    result = {
        "report_period": "2024",
        "companies": [
            {"name": "甲公司", "code": "A", "found": True},
            {"name": "乙公司", "code": None, "found": False},
        ],
        "metrics": [
            {
                "metric_code": "revenue",
                "metric_name": "营业收入",
                "unit": "元",
                "comparable": False,
                "comparison_note": "存在缺失指标，无法比较",
                "rows": [
                    {
                        "company_name": "甲公司",
                        "company_code": "A",
                        "value": "-100",
                        "statement_scope": "consolidated",
                        "period_type": "annual",
                        "extraction_status": "verified",
                        "source_title": "甲公司年报",
                        "source_page": 10,
                        "source_page_end": 11,
                        "bar_percent": None,
                        "comparable": False,
                        "note": "公司乙缺失",
                    },
                    {
                        "company_name": "乙公司",
                        "company_code": None,
                        "value": None,
                        "statement_scope": None,
                        "period_type": None,
                        "extraction_status": None,
                        "source_title": "",
                        "source_page": None,
                        "source_page_end": None,
                        "bar_percent": None,
                        "comparable": False,
                        "note": "指标缺失",
                    },
                ],
            }
        ],
        "pending_items": [{"code": "missing_metric", "company_name": "乙公司", "metric_code": "revenue", "message": "缺少营业收入"}],
        "selection_note": "用户选择的公司样本仅用于本次并列展示，不代表整个行业。",
    }

    markdown = render_peer_comparison(
        result,
        kb_id="default",
        generated_at="2026-09-07T00:00:00Z",
    )

    assert "# 用户选定公司同期间指标对比研究底稿" in markdown
    assert "甲公司、乙公司" in markdown
    assert "不代表整个行业" in markdown
    assert "-100" in markdown
    assert "未提供/不可计算" in markdown
    assert "甲公司年报" in markdown
    assert "页码 10-11" in markdown
    assert "## 待核实事项" in markdown
    assert "不构成投资建议" in markdown
    assert "排名" not in markdown
