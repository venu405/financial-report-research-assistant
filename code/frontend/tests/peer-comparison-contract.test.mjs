import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  FINANCIAL_METRIC_OPTIONS,
  formatMetricValue,
} from "../src/services/financial-metrics-utils.js";
import {
  formatPeerBarPercent,
  formatPeerValue,
  isNegativePeerValue,
  numericPeerBarPercent,
  peerBarWidth,
} from "../src/services/peer-comparison-utils.js";

const apiSource = await readFile(new URL("../src/services/peer-comparison-api.ts", import.meta.url), "utf8");
const viewSource = await readFile(new URL("../src/ResearchPlaceholder.vue", import.meta.url), "utf8");

test("peer comparison API keeps repeated sample and metric query parameters", () => {
  assert.match(apiSource, /\/kb\/peer-comparison/);
  assert.match(apiSource, /query\.append\("company_names"/);
  assert.match(apiSource, /query\.append\("metric_codes"/);
  assert.match(apiSource, /kb_id: currentKb\.value/);
  assert.match(apiSource, /X-Api-Token/);
  for (const field of ["report_period", "companies", "metrics", "pending_items", "selection_note", "bar_percent", "source_page_end"]) {
    assert.match(apiSource, new RegExp(`\\b${field}\\b`));
  }
});

test("peer comparison preserves five metric choices and validates samples", () => {
  assert.deepEqual(
    FINANCIAL_METRIC_OPTIONS.map((option) => option.code),
    ["revenue", "net_profit_parent", "operating_cash_flow", "total_assets", "total_liabilities"],
  );
  assert.match(viewSource, /companyNames\.length < 2 \|\| companyNames\.length > 3/);
  assert.match(viewSource, /new Set\(normalizedNames\)\.size/);
  assert.match(viewSource, /selectedMetricCodes\.value\.length/);
  assert.match(viewSource, /公司名称不能重复/);
});

test("peer comparison values keep yuan, missing values, and negative visual scales honest", () => {
  assert.equal(formatPeerValue("1234567"), "1,234,567 元");
  assert.equal(formatPeerValue("-123.5"), "-123.5 元");
  assert.equal(formatPeerValue(null), "缺失");
  assert.equal(formatMetricValue(null, "元"), "缺失");
  assert.equal(numericPeerBarPercent("-25.5"), -25.5);
  const negativeRow = { value: -100, bar_percent: 100 };
  assert.equal(isNegativePeerValue(negativeRow.value), true);
  assert.equal(isNegativePeerValue("100"), false);
  assert.equal(isNegativePeerValue(null), false);
  assert.equal(isNegativePeerValue("not-a-number"), false);
  assert.equal(peerBarWidth("-25.5"), 25.5);
  assert.equal(peerBarWidth("150"), 100);
  assert.equal(formatPeerBarPercent("-25.5"), "-25.5%");
  assert.equal(formatPeerBarPercent(null), "未提供");
});

test("peer comparison displays traceability, comparability, and pending items without ranking", () => {
  assert.match(viewSource, /来源/);
  assert.match(viewSource, /formatPageRange\(row\.source_page, row\.source_page_end\)/);
  assert.match(viewSource, /metric\.comparable && row\.comparable/);
  assert.match(viewSource, /isNegativePeerValue\(row\.value\)/);
  assert.match(viewSource, /peerBarWidth\(row\.bar_percent\)/);
  assert.match(viewSource, /未绘制视觉尺度/);
  assert.match(viewSource, /待核实事项/);
  assert.doesNotMatch(viewSource, /最佳|最差|行业代表|投资建议/);
});
