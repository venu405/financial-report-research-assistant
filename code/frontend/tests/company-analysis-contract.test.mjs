import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  FINANCIAL_METRIC_OPTIONS,
  formatAnalysisChange,
  formatAnalysisRate,
  profitCashFlowRelationLabel,
} from "../src/services/financial-metrics-utils.js";

const apiSource = await readFile(new URL("../src/services/financial-metrics-api.ts", import.meta.url), "utf8");
const viewSource = await readFile(new URL("../src/CompanyAnalysisView.vue", import.meta.url), "utf8");

test("company analysis API keeps the fixed route and response fields", () => {
  assert.match(apiSource, /\/kb\/company-analysis/);
  for (const field of [
    "company",
    "report_period",
    "comparison_period",
    "metrics",
    "profit_cash_flow",
    "disclosures",
    "pending_items",
    "change_amount",
    "change_rate_percent",
    "comparison_note",
  ]) {
    assert.match(apiSource, new RegExp(`\\b${field}\\b`));
  }
  assert.match(apiSource, /comparison_period/);
  assert.match(apiSource, /X-Api-Token/);
});

test("analysis page requests summary and current-period details together", () => {
  assert.match(viewSource, /Promise\.all\(\[/);
  assert.match(viewSource, /getCompanyAnalysis\(/);
  assert.match(viewSource, /listFinancialMetrics\(\{ companyName, reportPeriod/);
  assert.match(viewSource, /analysisMetrics/);
  assert.match(viewSource, /报告披露/);
  assert.match(viewSource, /待核实事项/);
  assert.match(viewSource, /formatAnalysisValue\(metric\.normalized_value, "元"\)/);
  assert.match(viewSource, /formatAnalysisChange\(metric\.change_amount, "元"\)/);
  assert.match(viewSource, /formatAnalysisValue\(analysis\.profit_cash_flow\.net_profit_value, "元"\)/);
  assert.match(viewSource, /pendingIndex/);
  assert.match(viewSource, /pending\.metric_code \|\| ['"]none['"]/);
  assert.match(viewSource, /\.metric-card-grid \{[^}]*grid-template-columns: repeat\(3,/s);
  assert.doesNotMatch(viewSource, /\.metric-values strong[^}]*overflow-wrap:\s*anywhere/s);
});

test("analysis contract preserves five metric slots and null change values", () => {
  assert.deepEqual(
    FINANCIAL_METRIC_OPTIONS.map((option) => option.code),
    ["revenue", "net_profit_parent", "operating_cash_flow", "total_assets", "total_liabilities"],
  );
  assert.equal(formatAnalysisChange(null), "不可计算");
  assert.equal(formatAnalysisRate(null), "不可计算");
  assert.equal(formatAnalysisChange("-123.4", "万元"), "-123.4 万元");
  assert.equal(formatAnalysisRate("-8.5"), "-8.5%");
  assert.equal(profitCashFlowRelationLabel("higher"), "经营现金流高于归母净利润");
  assert.equal(profitCashFlowRelationLabel("lower"), "经营现金流低于归母净利润");
  assert.equal(profitCashFlowRelationLabel("equal"), "经营现金流与归母净利润相等");
  assert.equal(profitCashFlowRelationLabel("unavailable"), "无法比较");
});
