export declare const TICKET_STATUS_LABELS: Readonly<Record<string, string>>;
export declare function ticketStatusLabel(status: string): string;
export declare function readMetric(
  source: unknown,
  paths: string[],
  fallback?: unknown
): unknown;
export declare function normalizeAlerts(payload: unknown): unknown[];
