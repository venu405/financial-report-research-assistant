import { formatMetricValue, isMissingValue } from "./financial-metrics-utils.js";

export function formatPeerValue(value) {
  return formatMetricValue(value, "元");
}

export function numericPeerBarPercent(value) {
  if (isMissingValue(value)) return null;
  const parsed = Number(String(value).replace(/,/g, "").trim());
  return Number.isFinite(parsed) ? parsed : null;
}

export function isNegativePeerValue(value) {
  if (isMissingValue(value)) return false;
  const parsed = Number(String(value).replace(/,/g, "").trim());
  return Number.isFinite(parsed) && parsed < 0;
}

export function peerBarWidth(value) {
  const parsed = numericPeerBarPercent(value);
  if (parsed === null) return 0;
  return Math.min(100, Math.abs(parsed));
}

export function formatPeerBarPercent(value) {
  const parsed = numericPeerBarPercent(value);
  if (parsed === null) return "未提供";
  const display = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 }).format(parsed);
  return `${display}%`;
}
