export interface FinancialMetricOption {
  code: string;
  label: string;
}

export interface FinancialMetricStatusOption {
  value: string;
  label: string;
}

export const FINANCIAL_METRIC_OPTIONS: readonly FinancialMetricOption[];
export const FINANCIAL_METRIC_STATUS_OPTIONS: readonly FinancialMetricStatusOption[];
export function isMissingValue(value: unknown): boolean;
export function formatMetricValue(value: unknown, unit?: unknown): string;
export function formatAnalysisValue(value: unknown, unit?: unknown): string;
export function formatAnalysisChange(value: unknown, unit?: unknown, missingLabel?: string): string;
export function formatAnalysisRate(value: unknown, missingLabel?: string): string;
export function profitCashFlowRelationLabel(relation: unknown): string;
export function formatPageRange(start: unknown, end?: unknown): string;
export function metricStatusLabel(status: unknown): string;
export function statementScopeLabel(scope: unknown): string;
export function revisionActorLabel(revision: { actor?: unknown } | null | undefined): string;
export function revisionReasonLabel(revision: { reason?: unknown } | null | undefined): string;
export function revisionTransitionText(revision: { old_snapshot?: unknown; new_snapshot?: unknown } | null | undefined): { oldValue: string; newValue: string };
