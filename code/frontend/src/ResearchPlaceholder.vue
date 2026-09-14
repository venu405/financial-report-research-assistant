<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { currentKb } from "./useKbState";
import {
  type PeerComparisonMetric,
  type PeerComparisonResponse,
  type PeerComparisonRow,
  exportPeerComparison as requestPeerComparisonExport,
  generatePeerComparisonBrief,
  getPeerComparison,
} from "./services/peer-comparison-api";
import {
  FINANCIAL_METRIC_OPTIONS,
  formatPageRange,
  isMissingValue,
  metricStatusLabel,
  statementScopeLabel,
} from "./services/financial-metrics-utils.js";
import {
  formatPeerBarPercent,
  formatPeerValue,
  isNegativePeerValue,
  numericPeerBarPercent,
  peerBarWidth,
} from "./services/peer-comparison-utils.js";
import { downloadBlob } from "./services/download-utils.js";

const metricOptions = FINANCIAL_METRIC_OPTIONS;
const form = reactive({ companyOne: "", companyTwo: "", companyThree: "", reportPeriod: "" });
const selectedMetricCodes = ref<string[]>(metricOptions.map((option) => option.code));
const result = ref<PeerComparisonResponse | null>(null);
const loading = ref(false);
const error = ref("");
const notice = ref("");
const peerExportParams = ref<{ companyNames: string[]; reportPeriod: string; metricCodes: string[] } | null>(null);
const peerExportLoading = ref(false);
const peerExportError = ref("");
const peerExportMessage = ref("");
const peerBrief = ref<{ brief: string; brief_source: "llm" | "template" } | null>(null);
const peerBriefLoading = ref(false);
const peerBriefError = ref("");
const peerBriefCopyMessage = ref("");

const selectedMetricCount = computed(() => selectedMetricCodes.value.length);
const hasResult = computed(() => result.value !== null);

function companyNamesFromForm() {
  return [form.companyOne, form.companyTwo, form.companyThree].map((name) => name.trim()).filter(Boolean);
}

function clearComparison() {
  form.companyOne = "";
  form.companyTwo = "";
  form.companyThree = "";
  form.reportPeriod = "";
  selectedMetricCodes.value = metricOptions.map((option) => option.code);
  result.value = null;
  loading.value = false;
  error.value = "";
  notice.value = "";
  peerExportParams.value = null;
  peerExportError.value = "";
  peerExportMessage.value = "";
  peerBrief.value = null;
  peerBriefLoading.value = false;
  peerBriefError.value = "";
  peerBriefCopyMessage.value = "";
}

async function generateComparison() {
  const companyNames = companyNamesFromForm();
  const normalizedNames = companyNames.map((name) => name.toLocaleLowerCase());
  const reportPeriod = form.reportPeriod.trim();
  error.value = "";
  notice.value = "";
  peerExportParams.value = null;
  peerExportError.value = "";
  peerExportMessage.value = "";
  peerBrief.value = null;
  peerBriefError.value = "";
  peerBriefCopyMessage.value = "";

  if (!reportPeriod) {
    result.value = null;
    notice.value = "请填写报告期后再生成同业对比。";
    return;
  }
  if (companyNames.length < 2 || companyNames.length > 3) {
    result.value = null;
    notice.value = "请填写 2 至 3 家公司名称。";
    return;
  }
  if (new Set(normalizedNames).size !== companyNames.length) {
    result.value = null;
    notice.value = "公司名称不能重复，请选择不同的对比样本。";
    return;
  }
  if (!selectedMetricCodes.value.length) {
    result.value = null;
    notice.value = "请至少选择一项财务指标。";
    return;
  }

  loading.value = true;
  result.value = null;
  try {
    const metricCodes = [...selectedMetricCodes.value];
    result.value = await getPeerComparison({
      companyNames,
      reportPeriod,
      metricCodes,
    });
    peerExportParams.value = { companyNames: [...companyNames], reportPeriod, metricCodes: [...metricCodes] };
    if (!result.value.metrics.length && !result.value.companies.length) {
      notice.value = "接口未返回同业对比记录，页面不会用示例数据填充。";
    }
  } catch (e) {
    error.value = `加载同业对比失败：${(e as Error).message}`;
  } finally {
    loading.value = false;
  }
}

async function exportComparisonDraft() {
  if (!peerExportParams.value || peerExportLoading.value) return;
  peerExportLoading.value = true;
  peerExportError.value = "";
  peerExportMessage.value = "";
  try {
    const download = await requestPeerComparisonExport(peerExportParams.value);
    downloadBlob(download.blob, download.filename);
    peerExportMessage.value = "研究底稿已下载。";
  } catch (e) {
    peerExportError.value = `导出研究底稿失败：${(e as Error).message}`;
  } finally {
    peerExportLoading.value = false;
  }
}

async function createComparisonBrief() {
  if (!peerExportParams.value || peerBriefLoading.value) return;
  peerBriefLoading.value = true;
  peerBriefError.value = "";
  peerBriefCopyMessage.value = "";
  try {
    peerBrief.value = await generatePeerComparisonBrief(peerExportParams.value);
  } catch (e) {
    peerBriefError.value = `生成对比简报失败：${(e as Error).message}`;
  } finally {
    peerBriefLoading.value = false;
  }
}

async function copyComparisonBrief() {
  if (!peerBrief.value?.brief) return;
  peerBriefCopyMessage.value = "";
  try {
    await navigator.clipboard.writeText(peerBrief.value.brief);
    peerBriefCopyMessage.value = "已复制到剪贴板。";
  } catch {
    peerBriefCopyMessage.value = "复制失败，请手动选择简报内容复制。";
  }
}

function metricNote(metric: PeerComparisonMetric) {
  return metric.comparison_note || "当前没有可比口径说明。";
}

function showBar(metric: PeerComparisonMetric, row: PeerComparisonRow) {
  return metric.comparable && row.comparable && !isMissingValue(row.value) && numericPeerBarPercent(row.bar_percent) !== null;
}

function barIsNegative(row: PeerComparisonRow) {
  return isNegativePeerValue(row.value);
}

function barStyle(row: PeerComparisonRow) {
  return { width: `${peerBarWidth(row.bar_percent)}%` };
}

function noBarLabel(metric: PeerComparisonMetric, row: PeerComparisonRow) {
  if (!metric.comparable) return "该指标不可比，未绘制视觉尺度";
  if (!row.comparable) return row.note || "该公司记录不可比，未绘制视觉尺度";
  return "视觉尺度未提供";
}
</script>

<template>
  <section class="peer-comparison-view">
    <div class="peer-command card">
      <div class="peer-command-heading">
        <div>
          <span class="eyebrow">横向研究 · 手动样本</span>
          <h1>同业对比</h1>
          <p>选择 2 至 3 家公司和同一报告期，读取接口返回的财务事实与来源；页面仅呈现可追溯记录，不生成推断结论。</p>
        </div>
        <span class="dataset-chip">资料库：{{ currentKb }}</span>
      </div>

      <form class="peer-form" @submit.prevent="generateComparison">
        <div class="company-fields">
          <label>公司一 <span class="required-mark">必填</span><input v-model="form.companyOne" placeholder="公司名称或简称" /></label>
          <label>公司二 <span class="required-mark">必填</span><input v-model="form.companyTwo" placeholder="公司名称或简称" /></label>
          <label>公司三 <span class="optional-mark">可选</span><input v-model="form.companyThree" placeholder="可留空" /></label>
        </div>
        <div class="period-field"><label>报告期 <span class="required-mark">必填</span><input v-model="form.reportPeriod" placeholder="如：2024年度" /></label></div>
        <fieldset class="metric-selection">
          <legend>对比指标 <span>{{ selectedMetricCount }} / {{ metricOptions.length }} 项</span></legend>
          <label v-for="option in metricOptions" :key="option.code" class="metric-option"><input v-model="selectedMetricCodes" type="checkbox" :value="option.code" /><span>{{ option.label }}</span></label>
        </fieldset>
        <div class="peer-actions"><button class="primary" type="submit" :disabled="loading">{{ loading ? "生成中…" : "生成对比" }}</button><button class="secondary" type="button" :disabled="loading" @click="clearComparison">清空</button></div>
      </form>
    </div>

    <div v-if="error" class="state-banner error">{{ error }}</div>
    <div v-else-if="notice" class="state-banner">{{ notice }}</div>
    <div v-if="loading" class="loading-state"><span class="loader-dot"></span>正在读取同业对比与财报来源…</div>

    <section v-else-if="hasResult && result" class="comparison-result">
      <div class="boundary-banner card"><div><span class="panel-kicker">样本边界</span><strong>{{ result.selection_note || "以下内容仅基于本次手动选择与接口返回记录。" }}</strong></div><div class="boundary-actions"><span class="period-chip">报告期：{{ result.report_period || form.reportPeriod || "未提供" }}</span><button class="export-button" type="button" :disabled="peerBriefLoading || !peerExportParams" @click="createComparisonBrief">{{ peerBriefLoading ? "简报生成中…" : "生成对比简报" }}</button><button class="export-button" type="button" :disabled="peerExportLoading || !peerExportParams" @click="exportComparisonDraft">{{ peerExportLoading ? "导出中…" : "导出研究底稿" }}</button></div></div>
      <p v-if="peerExportError" class="export-feedback error">{{ peerExportError }}</p>
      <p v-else-if="peerExportMessage" class="export-feedback success">{{ peerExportMessage }}</p>
      <p v-if="peerBriefError" class="export-feedback error">{{ peerBriefError }}</p>

      <article v-if="peerBrief" class="brief-panel card">
        <div class="section-heading"><div><span class="panel-kicker">对比简报</span><h2>基于结构化对比结果生成</h2></div><div class="brief-actions"><span class="status-tag">{{ peerBrief.brief_source === "llm" ? "LLM 简报" : "规则模板简报" }}</span><button class="export-button" type="button" @click="copyComparisonBrief">一键复制</button></div></div>
        <p v-if="peerBriefCopyMessage" class="export-feedback" :class="{ error: peerBriefCopyMessage.includes('失败'), success: !peerBriefCopyMessage.includes('失败') }">{{ peerBriefCopyMessage }}</p>
        <pre class="brief-content">{{ peerBrief.brief }}</pre>
      </article>

      <article class="company-status card">
        <div class="section-heading"><div><span class="panel-kicker">样本核对</span><h2>公司返回状态</h2></div><span>{{ result.companies.length }} 家</span></div>
        <div v-if="result.companies.length" class="company-status-grid">
          <div v-for="company in result.companies" :key="company.name" class="company-status-item"><div><strong>{{ company.name || "公司名称未提供" }}</strong><span>{{ company.code || "代码未提供" }}</span></div><span class="found-chip" :class="{ found: company.found }">{{ company.found ? "已找到记录" : "未找到记录" }}</span></div>
        </div>
        <p v-else class="muted">接口未返回公司状态。</p>
      </article>

      <div v-if="!result.metrics.length" class="empty-result card"><strong>接口未返回指标记录</strong><span>当前样本和报告期没有可展示的对比指标，页面不会使用示例数据填充。</span></div>

      <article v-for="metric in result.metrics" :key="metric.metric_code" class="metric-comparison card">
        <div class="section-heading metric-heading"><div><span class="panel-kicker">财务指标</span><h2>{{ metric.metric_name || metric.metric_code }}</h2><code>{{ metric.metric_code }}</code></div><span class="unit-chip">单位：元</span></div>
        <p v-if="!metric.comparable" class="comparability-note">{{ metricNote(metric) }}</p>
        <div class="table-scroll">
          <table class="peer-table">
            <thead><tr><th>公司</th><th>数值</th><th>报表口径</th><th>期间类型</th><th>提取状态</th><th>来源</th><th>横向尺度</th></tr></thead>
            <tbody>
              <tr v-for="(row, rowIndex) in metric.rows || []" :key="`${metric.metric_code}-${row.company_name}-${rowIndex}`">
                <td><strong>{{ row.company_name || "公司名称未提供" }}</strong><small>{{ row.company_code || "代码未提供" }}</small></td>
                <td class="value-cell">{{ formatPeerValue(row.value) }}</td>
                <td><span class="tag">{{ statementScopeLabel(row.statement_scope) }}</span></td>
                <td>{{ row.period_type || "未提供" }}</td>
                <td><span class="status-tag">{{ metricStatusLabel(row.extraction_status) }}</span></td>
                <td><span class="source-title">{{ row.source_title || "未提供" }}</span><small>{{ formatPageRange(row.source_page, row.source_page_end) }}</small></td>
                <td class="bar-cell"><div v-if="showBar(metric, row)" class="bar-readout" :class="{ negative: barIsNegative(row) }"><div class="bar-track"><span class="bar-fill" :style="barStyle(row)"></span></div><small>{{ formatPeerBarPercent(row.bar_percent) }}</small></div><span v-else class="no-bar">{{ noBarLabel(metric, row) }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-if="!(metric.rows || []).length" class="metric-empty">该指标没有返回公司记录。</p>
      </article>

      <article class="pending-panel card">
        <div class="section-heading"><div><span class="panel-kicker">待核实</span><h2>待核实事项</h2></div><span>{{ result.pending_items.length }} 条</span></div>
        <div v-if="result.pending_items.length" class="pending-list"><div v-for="(pending, pendingIndex) in result.pending_items" :key="`${pending.code}-${pending.company_name || 'none'}-${pending.metric_code || 'none'}-${pendingIndex}`" class="pending-item"><strong>{{ pending.code }}</strong><span>{{ pending.company_name || "公司未提供" }}{{ pending.metric_code ? ` · ${pending.metric_code}` : "" }}</span><p>{{ pending.message || "未提供说明" }}</p></div></div>
        <p v-else class="muted">当前响应未返回待核实事项。</p>
      </article>
    </section>

    <div v-else-if="!loading && !error && !notice" class="initial-state card"><strong>请先选择 2 至 3 家公司和报告期</strong><span>生成后这里会显示接口返回的指标、来源页码、可比性说明与待核实事项。</span></div>
  </section>
</template>

<style scoped>
.peer-comparison-view { padding: 16px 24px 28px; min-width: 0; }
.card { border: 1px solid var(--color-border); border-radius: var(--radius-lg); background: var(--color-bg); }
.peer-command { padding: 18px; box-shadow: var(--shadow-sm); }
.peer-command-heading, .section-heading, .company-status-item, .peer-actions { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.eyebrow, .panel-kicker { color: var(--color-accent); font-size: var(--text-xs); font-weight: var(--weight-semibold); letter-spacing: .08em; }
.peer-command-heading h1 { margin: 5px 0 4px; font-size: var(--text-xl); }
.peer-command-heading p { max-width: 780px; margin: 0; color: var(--color-charcoal); font-size: var(--text-sm); line-height: 1.5; }
.dataset-chip, .period-chip, .unit-chip, .found-chip, .tag, .status-tag { display: inline-flex; align-items: center; white-space: nowrap; border: 1px solid var(--color-border); border-radius: var(--radius-pill); padding: 4px 9px; color: var(--color-charcoal); background: var(--color-bg-quiet); font-size: var(--text-xs); }
.peer-form { margin-top: 18px; }
.company-fields { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.period-field { max-width: 360px; margin-top: 10px; }
.peer-form label { display: flex; flex-direction: column; gap: 5px; color: var(--color-charcoal); font-size: var(--text-xs); }
.peer-form input { width: 100%; min-width: 0; padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); color: var(--color-ink); background: var(--color-bg); font: inherit; font-size: var(--text-sm); }
.required-mark { color: var(--color-accent); }
.optional-mark { color: var(--color-slate); }
.metric-selection { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 16px 0 0; padding: 12px; border: 1px solid var(--color-border-soft); border-radius: var(--radius-md); }
.metric-selection legend { width: 100%; padding: 0; color: var(--color-ink-soft); font-size: var(--text-sm); font-weight: var(--weight-semibold); }
.metric-selection legend span { margin-left: 7px; color: var(--color-slate); font-size: var(--text-xs); font-weight: var(--weight-regular); }
.metric-option { flex-direction: row !important; align-items: center; gap: 6px !important; }
.metric-option input { width: auto; accent-color: var(--color-accent); }
.peer-actions { align-items: center; justify-content: flex-start; margin-top: 14px; }
button { cursor: pointer; font: inherit; }
.primary, .secondary { border-radius: var(--radius-md); padding: 8px 14px; font-size: var(--text-sm); }
.primary { border: 1px solid var(--color-accent); color: var(--color-bg); background: var(--color-accent); }
.secondary { border: 1px solid var(--color-border); color: var(--color-charcoal); background: var(--color-bg); }
.primary:disabled, .secondary:disabled { opacity: .5; cursor: not-allowed; }
.state-banner { margin-top: 12px; padding: 10px 12px; border-radius: var(--radius-md); color: var(--color-charcoal); background: var(--color-bg-quiet); font-size: var(--text-sm); }
.state-banner.error { color: var(--color-error); }
.loading-state, .initial-state { display: flex; align-items: center; gap: 8px; margin-top: 12px; padding: 20px; color: var(--color-charcoal); font-size: var(--text-sm); }
.initial-state { flex-direction: column; align-items: flex-start; }
.initial-state strong { color: var(--color-ink-soft); }
.empty-result { display: flex; flex-direction: column; gap: 5px; padding: 16px; color: var(--color-charcoal); font-size: var(--text-sm); }
.empty-result strong { color: var(--color-ink-soft); }
.loader-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--color-accent); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .35; transform: scale(.7); } }
.comparison-result { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
.boundary-banner, .company-status, .metric-comparison, .pending-panel { padding: 16px; }
.boundary-banner { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-left: 3px solid var(--color-accent); }
.boundary-banner div { display: flex; flex-direction: column; gap: 5px; }
.boundary-banner strong { color: var(--color-ink-soft); font-size: var(--text-sm); font-weight: var(--weight-medium); }
.boundary-actions, .brief-actions { align-items: center; flex-direction: row !important; }
.export-button { border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); padding: 7px 11px; color: var(--color-charcoal); background: var(--color-bg); font: inherit; font-size: var(--text-xs); cursor: pointer; }
.export-button:disabled { opacity: .55; cursor: not-allowed; }
.export-feedback { margin: -4px 0 0; font-size: var(--text-xs); }
.export-feedback.error { color: var(--color-error); }
.export-feedback.success { color: var(--color-success); }
.brief-panel { padding: 16px; }
.brief-content { margin: 0; padding: 12px; overflow: auto; border: 1px solid var(--color-border-soft); border-radius: var(--radius-md); color: var(--color-charcoal); background: var(--color-bg-quiet); font: 12px/1.65 var(--font-mono); white-space: pre-wrap; word-break: break-word; }
.section-heading { align-items: flex-start; margin-bottom: 12px; color: var(--color-slate); font-size: var(--text-xs); }
.section-heading h2 { margin: 3px 0 0; color: var(--color-ink); font-size: var(--text-base); }
.metric-heading { align-items: center; }
.metric-heading code { display: block; margin-top: 5px; color: var(--color-slate); font: 10px var(--font-mono); }
.company-status-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.company-status-item { align-items: center; padding: 10px; border: 1px solid var(--color-border-soft); border-radius: var(--radius-md); background: var(--color-bg-quiet); }
.company-status-item div { min-width: 0; }
.company-status-item strong, .company-status-item span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.company-status-item strong { color: var(--color-ink-soft); font-size: var(--text-sm); }
.company-status-item div span { margin-top: 3px; color: var(--color-slate); font-size: var(--text-xs); }
.found-chip.found { color: var(--color-success); border-color: rgba(15, 123, 108, .25); background: var(--color-success-bg); }
.comparability-note { margin: 0 0 12px; padding: 9px 10px; border-left: 3px solid var(--color-warning); color: var(--color-charcoal); background: var(--color-warning-bg); font-size: var(--text-sm); line-height: 1.5; }
.table-scroll { overflow-x: auto; }
.peer-table { width: 100%; min-width: 1000px; border-collapse: collapse; font-size: var(--text-sm); }
.peer-table th { padding: 10px 11px; border-bottom: 1px solid var(--color-border); color: var(--color-slate); font-size: var(--text-xs); font-weight: var(--weight-medium); text-align: left; white-space: nowrap; background: var(--color-bg-quiet); }
.peer-table td { padding: 11px; border-bottom: 1px solid var(--color-border-soft); vertical-align: top; }
.peer-table tbody tr:hover { background: var(--color-bg-warm); }
.peer-table td strong { display: block; color: var(--color-ink-soft); font-weight: var(--weight-medium); }
.peer-table td small, .source-title + small { display: block; margin-top: 3px; color: var(--color-slate); font-size: var(--text-xs); }
.value-cell { color: var(--color-ink-soft); font-variant-numeric: tabular-nums; white-space: nowrap; }
.source-title { display: block; max-width: 190px; overflow: hidden; color: var(--color-charcoal); text-overflow: ellipsis; white-space: nowrap; }
.bar-cell { min-width: 175px; }
.bar-readout { display: flex; align-items: center; gap: 8px; min-width: 160px; }
.bar-track { width: 106px; height: 7px; overflow: hidden; border-radius: var(--radius-pill); background: var(--color-border-soft); }
.bar-fill { display: block; height: 100%; border-radius: inherit; background: var(--color-accent); }
.bar-readout.negative .bar-fill { background: var(--color-warning); }
.bar-readout small { margin: 0 !important; color: var(--color-charcoal) !important; font-variant-numeric: tabular-nums; white-space: nowrap; }
.no-bar { color: var(--color-slate); font-size: var(--text-xs); line-height: 1.4; }
.metric-empty, .muted { margin: 0; color: var(--color-slate); font-size: var(--text-xs); }
.pending-list { display: flex; flex-direction: column; gap: 7px; }
.pending-item { display: grid; grid-template-columns: auto auto 1fr; align-items: baseline; gap: 8px; padding: 8px 10px; border-left: 3px solid var(--color-warning); background: var(--color-warning-bg); }
.pending-item strong { color: var(--color-warning); font: 11px var(--font-mono); }
.pending-item span { color: var(--color-slate); font-size: var(--text-xs); }
.pending-item p { margin: 0; color: var(--color-charcoal); font-size: var(--text-sm); }
@media (max-width: 900px) {
  .company-fields, .company-status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .company-fields label:last-child { grid-column: 1 / -1; }
  .period-field { max-width: none; }
}
@media (max-width: 640px) {
  .peer-comparison-view { padding: 12px; }
  .peer-command-heading, .boundary-banner, .metric-heading { align-items: flex-start; flex-direction: column; }
  .boundary-actions { align-items: flex-start; flex-wrap: wrap; }
  .company-fields, .company-status-grid { grid-template-columns: 1fr; }
  .company-fields label:last-child { grid-column: auto; }
  .pending-item { grid-template-columns: 1fr; gap: 4px; }
}
</style>
