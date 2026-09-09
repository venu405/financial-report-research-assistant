export const TICKET_STATUS_LABELS = Object.freeze({
  pending: "待处理",
  processing: "处理中",
  resolved: "已解决",
  closed: "已关闭",
});

export function ticketStatusLabel(status) {
  return TICKET_STATUS_LABELS[status] || status || "未知";
}
export function readMetric(source, paths, fallback = 0) {
  for (const path of paths) {
    const value = path.split(".").reduce((current, key) => current?.[key], source);
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return fallback;
}

export function normalizeAlerts(payload) {
  if (Array.isArray(payload)) return payload;
  return Array.isArray(payload?.alerts) ? payload.alerts : [];
}
