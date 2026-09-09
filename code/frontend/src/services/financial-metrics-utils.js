export const FINANCIAL_METRIC_OPTIONS = [
  { code: "revenue", label: "营业收入" },
  { code: "net_profit_parent", label: "归属于上市公司股东的净利润" },
  { code: "operating_cash_flow", label: "经营活动产生的现金流量净额" },
  { code: "total_assets", label: "资产总额" },
  { code: "total_liabilities", label: "负债总额" },
];

export const FINANCIAL_METRIC_STATUS_OPTIONS = [
  { value: "verified", label: "已核验" },
  { value: "missing", label: "缺失" },
  { value: "conflict", label: "有冲突" },
  { value: "failed", label: "提取失败" },
  { value: "corrected", label: "已修订" },
];

const STATUS_LABELS = {
  verified: "已核验",
  missing: "缺失",
  conflict: "有冲突",
  failed: "提取失败",
  corrected: "已修订",
};

const SCOPE_LABELS = {
  consolidated: "合并口径",
  merged: "合并口径",
  parent: "母公司口径",
  parent_company: "母公司口径",
  unknown: "未标注",
};

export function isMissingValue(value) {
  return value === null || value === undefined || String(value).trim() === "";
}

function numericValue(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const compact = String(value).replace(/,/g, "").trim();
  if (!compact || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(compact)) return null;
  const parsed = Number(compact);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatMetricValue(value, unit = "") {
  if (isMissingValue(value)) return "缺失";
  const parsed = numericValue(value);
  const display = parsed === null
    ? String(value).trim()
    : new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 20 }).format(parsed);
  const suffix = isMissingValue(unit) ? "" : ` ${String(unit).trim()}`;
  return `${display}${suffix}`;
}

export function formatAnalysisValue(value, unit = "") {
  return formatMetricValue(value, unit);
}

export function formatAnalysisChange(value, unit = "", missingLabel = "不可计算") {
  return isMissingValue(value) ? missingLabel : formatMetricValue(value, unit);
}

export function formatAnalysisRate(value, missingLabel = "不可计算") {
  if (isMissingValue(value)) return missingLabel;
  const text = String(value).trim();
  if (text.endsWith("%")) return text;
  const parsed = numericValue(value);
  return parsed === null ? text : `${new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 20 }).format(parsed)}%`;
}

export function profitCashFlowRelationLabel(relation) {
  const labels = {
    higher: "经营现金流高于归母净利润",
    lower: "经营现金流低于归母净利润",
    equal: "经营现金流与归母净利润相等",
    unavailable: "无法比较",
  };
  return labels[String(relation || "unavailable").trim().toLowerCase()] || "无法比较";
}

function pageNumber(value) {
  if (isMissingValue(value)) return null;
  const parsed = Number(String(value).trim());
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

export function formatPageRange(start, end) {
  const first = pageNumber(start);
  const last = pageNumber(end);
  if (first !== null && last !== null && last !== first) return `第 ${first}-${last} 页`;
  if (first !== null) return `第 ${first} 页`;
  if (last !== null) return `第 ${last} 页`;
  return "未提供";
}

export function metricStatusLabel(status) {
  if (isMissingValue(status)) return "未标注";
  return STATUS_LABELS[String(status).trim().toLowerCase()] || String(status);
}

export function statementScopeLabel(scope) {
  if (isMissingValue(scope)) return "未标注";
  return SCOPE_LABELS[String(scope).trim().toLowerCase()] || String(scope);
}

export function revisionActorLabel(revision) {
  return revision && revision.actor ? String(revision.actor) : "未提供操作人";
}

export function revisionReasonLabel(revision) {
  return revision && revision.reason ? String(revision.reason) : "未提供修订原因";
}

export function revisionTransitionText(revision) {
  const oldValue = revision && revision.old_snapshot ? revision.old_snapshot : null;
  const newValue = revision && revision.new_snapshot ? revision.new_snapshot : null;
  return {
    oldValue: oldValue ? JSON.stringify(oldValue, null, 2) : "旧值未提供",
    newValue: newValue ? JSON.stringify(newValue, null, 2) : "新值未提供",
  };
}
