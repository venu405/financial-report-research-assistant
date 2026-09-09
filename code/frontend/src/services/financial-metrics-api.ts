import { currentKb, userToken } from "../useKbState";
import { filenameFromContentDisposition } from "./download-utils.js";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface FinancialMetric {
  id: number;
  kb_id: string;
  company_name: string;
  company_code?: string | null;
  report_period?: string | null;
  period_type?: string | null;
  metric_code: string;
  metric_name: string;
  raw_value?: string | number | null;
  raw_unit?: string | null;
  normalized_value?: string | number | null;
  normalized_unit?: string | null;
  statement_scope?: string | null;
  source_doc_id?: string | null;
  source_title?: string | null;
  source_page?: number | null;
  source_page_end?: number | null;
  source_chunk_id?: string | null;
  source_text?: string | null;
  extraction_status?: string | null;
  created_by?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface FinancialMetricRevision {
  id?: number | string;
  actor?: string | null;
  old_snapshot?: Record<string, unknown> | null;
  new_snapshot?: Record<string, unknown> | null;
  reason?: string | null;
  created_at?: string | null;
}

export interface FinancialMetricFilters {
  companyName?: string;
  reportPeriod?: string;
  metricCode?: string;
  limit?: number;
  offset?: number;
}

export interface FinancialMetricList {
  items: FinancialMetric[];
  total: number;
  limit: number;
  offset: number;
}

export interface CompanyAnalysisMetric {
  metric_code: string;
  metric_name: string;
  current: FinancialMetric | null;
  comparison: FinancialMetric | null;
  change_amount: string | null;
  change_rate_percent: string | null;
  comparable: boolean;
  comparison_note: string;
}

export type ProfitCashFlowRelation = "higher" | "lower" | "equal" | "unavailable";

export interface CompanyAnalysisProfitCashFlow {
  net_profit_value: string | null;
  operating_cash_flow_value: string | null;
  difference: string | null;
  relation: ProfitCashFlowRelation;
  comparable: boolean;
  note: string;
}

export interface CompanyAnalysisDisclosure {
  metric_code: string;
  source_title: string | null;
  source_page: number | null;
  source_page_end: number | null;
  source_text: string | null;
}

export interface CompanyAnalysisPendingItem {
  code: string;
  metric_code: string | null;
  message: string;
}

export interface CompanyAnalysisResponse {
  company: { name: string | null; code: string | null } | null;
  report_period: string | null;
  comparison_period: string | null;
  metrics: CompanyAnalysisMetric[];
  profit_cash_flow: CompanyAnalysisProfitCashFlow;
  disclosures: CompanyAnalysisDisclosure[];
  pending_items: CompanyAnalysisPendingItem[];
}

export interface MarkdownDownload {
  blob: Blob;
  filename: string;
}

export interface FinancialMetricPatch {
  raw_value?: string | null;
  raw_unit?: string | null;
  statement_scope?: string | null;
  source_title?: string | null;
  source_page?: number | null;
  source_page_end?: number | null;
  source_text?: string | null;
  extraction_status?: string | null;
  reason: string;
}

function headers(includeJson = false): Record<string, string> {
  const result: Record<string, string> = {};
  if (includeJson) result["Content-Type"] = "application/json";
  if (userToken.value.trim()) result["X-Api-Token"] = userToken.value.trim();
  return result;
}

async function responseError(response: Response): Promise<Error> {
  const data = await response.json().catch(() => ({}));
  const detail = typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`;
  return new Error(detail);
}

export async function listFinancialMetrics(filters: FinancialMetricFilters = {}): Promise<FinancialMetricList> {
  const query = new URLSearchParams({
    kb_id: currentKb.value,
    limit: String(filters.limit ?? 20),
    offset: String(filters.offset ?? 0),
  });
  if (filters.companyName?.trim()) query.set("company_name", filters.companyName.trim());
  if (filters.reportPeriod?.trim()) query.set("report_period", filters.reportPeriod.trim());
  if (filters.metricCode?.trim()) query.set("metric_code", filters.metricCode.trim());
  const response = await fetch(`${baseURL}/kb/financial-metrics?${query.toString()}`, { headers: headers() });
  if (!response.ok) throw await responseError(response);
  const data = await response.json();
  return {
    items: Array.isArray(data.items) ? data.items as FinancialMetric[] : [],
    total: Number.isFinite(Number(data.total)) ? Number(data.total) : 0,
    limit: Number.isFinite(Number(data.limit)) ? Number(data.limit) : Number(filters.limit ?? 20),
    offset: Number.isFinite(Number(data.offset)) ? Number(data.offset) : Number(filters.offset ?? 0),
  };
}

export async function getCompanyAnalysis(params: {
  companyName: string;
  reportPeriod: string;
  comparisonPeriod?: string;
}): Promise<CompanyAnalysisResponse> {
  const query = companyAnalysisQuery(params);
  const response = await fetch(`${baseURL}/kb/company-analysis?${query.toString()}`, { headers: headers() });
  if (!response.ok) throw await responseError(response);
  const data = await response.json();
  return {
    company: data.company ? {
      name: data.company.name ?? null,
      code: data.company.code ?? null,
    } : null,
    report_period: data.report_period ?? null,
    comparison_period: data.comparison_period ?? null,
    metrics: Array.isArray(data.metrics) ? data.metrics as CompanyAnalysisMetric[] : [],
    profit_cash_flow: data.profit_cash_flow || {
      net_profit_value: null,
      operating_cash_flow_value: null,
      difference: null,
      relation: "unavailable",
      comparable: false,
      note: "未提供利润与现金流对照数据。",
    },
    disclosures: Array.isArray(data.disclosures) ? data.disclosures as CompanyAnalysisDisclosure[] : [],
    pending_items: Array.isArray(data.pending_items) ? data.pending_items as CompanyAnalysisPendingItem[] : [],
  };
}

function companyAnalysisQuery(params: {
  companyName: string;
  reportPeriod: string;
  comparisonPeriod?: string;
}) {
  const query = new URLSearchParams({
    kb_id: currentKb.value,
    company_name: params.companyName.trim(),
    report_period: params.reportPeriod.trim(),
  });
  if (params.comparisonPeriod?.trim()) query.set("comparison_period", params.comparisonPeriod.trim());
  return query;
}

export async function exportCompanyAnalysis(params: {
  companyName: string;
  reportPeriod: string;
  comparisonPeriod?: string;
}): Promise<MarkdownDownload> {
  const query = companyAnalysisQuery(params);
  const response = await fetch(`${baseURL}/kb/company-analysis/export?${query.toString()}`, { headers: headers() });
  if (!response.ok) throw await responseError(response);
  return {
    blob: await response.blob(),
    filename: filenameFromContentDisposition(response.headers.get("Content-Disposition"), "company-analysis-research.md"),
  };
}

export async function updateFinancialMetric(id: number, patch: FinancialMetricPatch): Promise<FinancialMetric> {
  if (!patch.reason.trim()) throw new Error("修订原因不能为空");
  const response = await fetch(`${baseURL}/kb/financial-metrics/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: headers(true),
    body: JSON.stringify(patch),
  });
  if (!response.ok) throw await responseError(response);
  const data = await response.json();
  return data.item as FinancialMetric;
}

export async function listFinancialMetricRevisions(id: number): Promise<FinancialMetricRevision[]> {
  const response = await fetch(`${baseURL}/kb/financial-metrics/${encodeURIComponent(id)}/revisions`, {
    headers: headers(),
  });
  if (!response.ok) throw await responseError(response);
  const data = await response.json();
  return Array.isArray(data.revisions) ? data.revisions as FinancialMetricRevision[] : [];
}
