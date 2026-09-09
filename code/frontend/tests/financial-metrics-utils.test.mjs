import assert from "node:assert/strict";
import test from "node:test";

import {
  FINANCIAL_METRIC_OPTIONS,
  FINANCIAL_METRIC_STATUS_OPTIONS,
  formatAnalysisChange,
  formatAnalysisRate,
  formatAnalysisValue,
  formatMetricValue,
  formatPageRange,
  isMissingValue,
  metricStatusLabel,
  profitCashFlowRelationLabel,
  revisionActorLabel,
  revisionReasonLabel,
  revisionTransitionText,
  statementScopeLabel,
} from "../src/services/financial-metrics-utils.js";

test("financial metric codes match the backend contract", () => {
  const codes = FINANCIAL_METRIC_OPTIONS.map((option) => option.code);
  assert.ok(codes.includes("net_profit_parent"));
  assert.equal(codes.includes("net_profit_attributable"), false);
  assert.deepEqual(
    FINANCIAL_METRIC_STATUS_OPTIONS.map((option) => option.value),
    ["verified", "missing", "conflict", "failed", "corrected"],
  );
});

test("formatMetricValue keeps units, grouping, and negative signs", () => {
  assert.equal(formatMetricValue("1234567", "元"), "1,234,567 元");
  assert.equal(formatMetricValue("-1234.5", "万元"), "-1,234.5 万元");
  assert.equal(formatMetricValue(12.3, "亿元"), "12.3 亿元");
});

test("missing values remain missing and never become zero", () => {
  assert.equal(isMissingValue(null), true);
  assert.equal(formatMetricValue(null, "元"), "缺失");
  assert.equal(formatMetricValue("", "元"), "缺失");
  assert.equal(formatMetricValue(0, "元"), "0 元");
});

test("analysis values and changes keep units, signs, and null semantics", () => {
  assert.equal(formatAnalysisValue("1234567", "元"), "1,234,567 元");
  assert.equal(formatAnalysisValue("-12.5", "万元"), "-12.5 万元");
  assert.equal(formatAnalysisValue(null, "元"), "缺失");
  assert.equal(formatAnalysisChange(null, "元"), "不可计算");
  assert.equal(formatAnalysisChange("-100", "万元"), "-100 万元");
  assert.equal(formatAnalysisRate(null), "不可计算");
  assert.equal(formatAnalysisRate("-12.5"), "-12.5%");
  assert.equal(formatAnalysisRate("8%"), "8%");
  assert.equal(profitCashFlowRelationLabel("higher"), "经营现金流高于归母净利润");
  assert.equal(profitCashFlowRelationLabel("lower"), "经营现金流低于归母净利润");
  assert.equal(profitCashFlowRelationLabel("equal"), "经营现金流与归母净利润相等");
  assert.equal(profitCashFlowRelationLabel("unavailable"), "无法比较");
});

test("page ranges and labels remain explicit", () => {
  assert.equal(formatPageRange(6, 8), "第 6-8 页");
  assert.equal(formatPageRange(6, 6), "第 6 页");
  assert.equal(formatPageRange(null, null), "未提供");
  assert.equal(metricStatusLabel("verified"), "已核验");
  assert.equal(metricStatusLabel("missing"), "缺失");
  assert.equal(metricStatusLabel("conflict"), "有冲突");
  assert.equal(metricStatusLabel("failed"), "提取失败");
  assert.equal(statementScopeLabel("consolidated"), "合并口径");
  assert.equal(statementScopeLabel("parent"), "母公司口径");
  assert.equal(statementScopeLabel("unknown"), "未标注");
});

test("revision mapping uses actor, reason, and old/new snapshots", () => {
  const revision = {
    actor: "reviewer-1",
    reason: "按年报第 6 页修正单位",
    created_at: "2026-09-08T10:00:00Z",
    old_snapshot: { raw_value: "10", raw_unit: "万元" },
    new_snapshot: { raw_value: "100000", raw_unit: "元" },
  };
  assert.equal(revisionActorLabel(revision), "reviewer-1");
  assert.equal(revisionReasonLabel(revision), "按年报第 6 页修正单位");
  assert.deepEqual(revisionTransitionText(revision), {
    oldValue: '{\n  "raw_value": "10",\n  "raw_unit": "万元"\n}',
    newValue: '{\n  "raw_value": "100000",\n  "raw_unit": "元"\n}',
  });
});
