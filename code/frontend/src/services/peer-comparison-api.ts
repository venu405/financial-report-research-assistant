import { currentKb, userToken } from "../useKbState";
import { filenameFromContentDisposition } from "./download-utils.js";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface PeerComparisonCompany {
  name: string;
  code: string | null;
  found: boolean;
}

export interface PeerComparisonRow {
  company_name: string;
  company_code: string | null;
  value: string | number | null;
  statement_scope: string | null;
  period_type: string | null;
  extraction_status: string | null;
  source_title: string | null;
  source_page: number | null;
  source_page_end: number | null;
  bar_percent: string | number | null;
  comparable: boolean;
  note: string | null;
}

export interface PeerComparisonMetric {
  metric_code: string;
  metric_name: string;
  unit: string;
  comparable: boolean;
  comparison_note: string | null;
  rows: PeerComparisonRow[];
}

export interface PeerComparisonPendingItem {
  code: string;
  company_name: string | null;
  metric_code: string | null;
  message: string;
}

export interface PeerComparisonResponse {
  report_period: string | null;
  companies: PeerComparisonCompany[];
  metrics: PeerComparisonMetric[];
  pending_items: PeerComparisonPendingItem[];
  selection_note: string | null;
}

export interface PeerComparisonDownload {
  blob: Blob;
  filename: string;
}

function headers(): Record<string, string> {
  const result: Record<string, string> = {};
  if (userToken.value.trim()) result["X-Api-Token"] = userToken.value.trim();
  return result;
}

async function responseError(response: Response): Promise<Error> {
  const data = await response.json().catch(() => ({}));
  const detail = typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`;
  return new Error(detail);
}

export async function getPeerComparison(params: {
  companyNames: string[];
  reportPeriod: string;
  metricCodes?: string[];
}): Promise<PeerComparisonResponse> {
  const query = peerComparisonQuery(params);
  const response = await fetch(`${baseURL}/kb/peer-comparison?${query.toString()}`, { headers: headers() });
  if (!response.ok) throw await responseError(response);
  const data = await response.json();
  return {
    report_period: data.report_period ?? null,
    companies: Array.isArray(data.companies) ? data.companies as PeerComparisonCompany[] : [],
    metrics: Array.isArray(data.metrics) ? data.metrics as PeerComparisonMetric[] : [],
    pending_items: Array.isArray(data.pending_items) ? data.pending_items as PeerComparisonPendingItem[] : [],
    selection_note: data.selection_note ?? null,
  };
}

function peerComparisonQuery(params: {
  companyNames: string[];
  reportPeriod: string;
  metricCodes?: string[];
}) {
  const query = new URLSearchParams({
    kb_id: currentKb.value,
    report_period: params.reportPeriod.trim(),
  });
  params.companyNames.forEach((companyName) => query.append("company_names", companyName.trim()));
  (params.metricCodes || []).forEach((metricCode) => query.append("metric_codes", metricCode.trim()));
  return query;
}

export async function exportPeerComparison(params: {
  companyNames: string[];
  reportPeriod: string;
  metricCodes?: string[];
}): Promise<PeerComparisonDownload> {
  const query = peerComparisonQuery(params);
  const response = await fetch(`${baseURL}/kb/peer-comparison/export?${query.toString()}`, { headers: headers() });
  if (!response.ok) throw await responseError(response);
  return {
    blob: await response.blob(),
    filename: filenameFromContentDisposition(response.headers.get("Content-Disposition"), "peer-comparison-research.md"),
  };
}
