<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { currentKb, userToken } from "./useKbState";
import {
  exportCompanyAnalysis,
  getCompanyAnalysis,
  generateCompanyAnalysisBrief,
  listFinancialMetricRevisions,
  listFinancialMetrics,
  updateFinancialMetric,
  type CompanyAnalysisMetric,
  type CompanyAnalysisResponse,
  type FinancialMetric,
  type FinancialMetricPatch,
  type FinancialMetricRevision,
} from "./services/financial-metrics-api";
import {
  FINANCIAL_METRIC_OPTIONS,
  FINANCIAL_METRIC_STATUS_OPTIONS,
  formatAnalysisChange,
  formatAnalysisRate,
  formatAnalysisValue,
  formatMetricValue,
  formatPageRange,
  metricStatusLabel,
  profitCashFlowRelationLabel,
  revisionActorLabel,
  revisionReasonLabel,
  revisionTransitionText,
  statementScopeLabel,
} from "./services/financial-metrics-utils.js";
import { downloadBlob } from "./services/download-utils.js";

type MetricDraft = {
  rawValue: string;
  rawUnit: string;
  statementScope: string;
  extractionStatus: string;
  sourceTitle: string;
  sourcePage: string;
  sourcePageEnd: string;
  sourceText: string;
  reason: string;
};

const filters = reactive({ companyName: "", reportPeriod: "", metricCode: "" });
const analysisForm = reactive({ companyName: "", reportPeriod: "" });
const analysis = ref<CompanyAnalysisResponse | null>(null);
const analysisGenerated = ref(false);
const analysisLoading = ref(false);
const analysisError = ref("");
const analysisNotice = ref("");
const analysisExportParams = ref<{ companyName: string; reportPeriod: string; comparisonPeriod?: string } | null>(null);
const analysisExportLoading = ref(false);
const analysisExportError = ref("");
const analysisExportMessage = ref("");
const analysisBrief = ref<{ brief: string; brief_source: "llm" | "template" } | null>(null);
const analysisBriefLoading = ref(false);
const analysisBriefError = ref("");
const analysisBriefCopyMessage = ref("");
const items = ref<FinancialMetric[]>([]);
const total = ref(0);
const limit = ref(20);
const offset = ref(0);
const loading = ref(false);
const error = ref("");
const notice = ref("");
const expandedId = ref("");
const revisions = reactive<Record<string, FinancialMetricRevision[]>>({});
const revisionLoading = reactive<Record<string, boolean>>({});
const revisionError = reactive<Record<string, string>>({});
const drafts = reactive<Record<string, MetricDraft>>({});
const savingId = ref("");
const saveMessage = reactive<Record<string, string>>({});
const saveError = reactive<Record<string, string>>({});

const hasToken = computed(() => Boolean(userToken.value.trim()));
const currentPage = computed(() => (total.value ? Math.floor(offset.value / limit.value) + 1 : 1));
const pageCount = computed(() => Math.max(1, Math.ceil(total.value / limit.value)));
const pageSummary = computed(() => {
  if (!total.value) return "暂无记录";
  return `第 ${currentPage.value} / ${pageCount.value} 页 · 共 ${total.value} 条`;
});

const analysisMetrics = computed<CompanyAnalysisMetric[]>(() => {
  const returned = new Map((analysis.value?.metrics || []).map((metric) => [metric.metric_code, metric]));
  return FINANCIAL_METRIC_OPTIONS.map((option) => returned.get(option.code) || {
    metric_code: option.code,
    metric_name: option.label,
    current: null,
    comparison: null,
    change_amount: null,
    change_rate_percent: null,
    comparable: false,
    comparison_note: "接口未返回该指标记录。",
  });
});

const scopeOptions = [
  { value: "consolidated", label: "合并口径" },
  { value: "parent", label: "母公司口径" },
  { value: "unknown", label: "未标注" },
];

function analysisMetricValue(metric: FinancialMetric | null | undefined) {
  if (!metric) return "缺失";
  return formatAnalysisValue(metric.normalized_value, "元");
}

function analysisMetricSource(metric: FinancialMetric | null | undefined) {
  if (!metric) return "未提供指标记录";
  return `${statementScopeLabel(metric.statement_scope)} · ${metricStatusLabel(metric.extraction_status)}`;
}

function disclosurePage(disclosure: CompanyAnalysisResponse["disclosures"][number]) {
  return formatPageRange(disclosure.source_page, disclosure.source_page_end);
}

function resetMetricList() {
  items.value = [];
  total.value = 0;
  offset.value = 0;
  expandedId.value = "";
}

async function generateAnalysis() {
  const companyName = analysisForm.companyName.trim();
  const reportPeriod = analysisForm.reportPeriod.trim();
  analysisError.value = "";
  analysisNotice.value = "";
  analysisExportParams.value = null;
  analysisExportError.value = "";
  analysisExportMessage.value = "";
  analysisBrief.value = null;
  analysisBriefError.value = "";
  analysisBriefCopyMessage.value = "";
  if (!companyName || !reportPeriod) {
    analysisGenerated.value = false;
    analysis.value = null;
    resetMetricList();
    analysisNotice.value = "请填写公司和报告期后生成分析。";
    return;
  }

  analysisLoading.value = true;
  analysisGenerated.value = false;
  analysis.value = null;
  resetMetricList();
  filters.companyName = companyName;
  filters.reportPeriod = reportPeriod;
  filters.metricCode = "";
  try {
    const [summary, metricList] = await Promise.all([
      getCompanyAnalysis({ companyName, reportPeriod, withComparison: false }),
      listFinancialMetrics({ companyName, reportPeriod, limit: limit.value, offset: 0 }),
    ]);
    analysisExportParams.value = { companyName, reportPeriod };
    analysis.value = summary;
    analysisGenerated.value = true;
    items.value = metricList.items;
    total.value = metricList.total;
    limit.value = metricList.limit || limit.value;
    offset.value = metricList.offset;
    if (!summary.metrics.length && !metricList.items.length) {
      analysisNotice.value = "接口未返回当前条件下的指标数据，页面不会用示例数据填充。";
    }
  } catch (e) {
    analysisError.value = `生成公司分析失败：${(e as Error).message}`;
  } finally {
    analysisLoading.value = false;
  }
}

async function exportAnalysisDraft() {
  if (!analysisExportParams.value || analysisExportLoading.value) return;
  analysisExportLoading.value = true;
  analysisExportError.value = "";
  analysisExportMessage.value = "";
  try {
    const download = await exportCompanyAnalysis(analysisExportParams.value);
    downloadBlob(download.blob, download.filename);
    analysisExportMessage.value = "研究底稿已下载。";
  } catch (e) {
    analysisExportError.value = `导出研究底稿失败：${(e as Error).message}`;
  } finally {
    analysisExportLoading.value = false;
  }
}

async function createAnalysisBrief() {
  if (!analysisExportParams.value || analysisBriefLoading.value) return;
  analysisBriefLoading.value = true;
  analysisBriefError.value = "";
  analysisBriefCopyMessage.value = "";
  try {
    analysisBrief.value = await generateCompanyAnalysisBrief(analysisExportParams.value);
  } catch (e) {
    analysisBriefError.value = `生成分析简报失败：${(e as Error).message}`;
  } finally {
    analysisBriefLoading.value = false;
  }
}

async function copyAnalysisBrief() {
  if (!analysisBrief.value?.brief) return;
  analysisBriefCopyMessage.value = "";
  try {
    await navigator.clipboard.writeText(analysisBrief.value.brief);
    analysisBriefCopyMessage.value = "已复制到剪贴板。";
  } catch {
    analysisBriefCopyMessage.value = "复制失败，请手动选择简报内容复制。";
  }
}

function clearAnalysis() {
  analysisForm.companyName = "";
  analysisForm.reportPeriod = "";
  analysis.value = null;
  analysisGenerated.value = false;
  analysisError.value = "";
  analysisNotice.value = "";
  analysisExportParams.value = null;
  analysisExportError.value = "";
  analysisExportMessage.value = "";
  analysisBrief.value = null;
  analysisBriefError.value = "";
  analysisBriefCopyMessage.value = "";
  filters.companyName = "";
  filters.reportPeriod = "";
  filters.metricCode = "";
  resetMetricList();
}

function rowKey(item: FinancialMetric) { return String(item.id); }

function createDraft(item: FinancialMetric): MetricDraft {
  return {
    rawValue: item.raw_value == null ? "" : String(item.raw_value),
    rawUnit: item.raw_unit || "",
    statementScope: item.statement_scope || "unknown",
    extractionStatus: item.extraction_status || "missing",
    sourceTitle: item.source_title || "",
    sourcePage: item.source_page == null ? "" : String(item.source_page),
    sourcePageEnd: item.source_page_end == null ? "" : String(item.source_page_end),
    sourceText: item.source_text || "",
    reason: "",
  };
}

function ensureDraft(item: FinancialMetric) {
  const key = rowKey(item);
  if (!drafts[key]) drafts[key] = createDraft(item);
  return drafts[key];
}

async function queryMetrics(nextOffset = 0) {
  loading.value = true;
  error.value = "";
  notice.value = "";
  try {
    const result = await listFinancialMetrics({
      companyName: filters.companyName,
      reportPeriod: filters.reportPeriod,
      metricCode: filters.metricCode,
      limit: limit.value,
      offset: nextOffset,
    });
    items.value = result.items;
    total.value = result.total;
    limit.value = result.limit || limit.value;
    offset.value = result.offset;
    expandedId.value = "";
    if (!items.value.length) notice.value = "当前筛选条件下暂无财报指标记录。";
  } catch (e) {
    error.value = `加载指标失败：${(e as Error).message}`;
    items.value = [];
    total.value = 0;
  } finally {
    loading.value = false;
  }
}

async function clearFilters() {
  filters.companyName = "";
  filters.reportPeriod = "";
  filters.metricCode = "";
  await queryMetrics(0);
}

async function changePage(nextPage: number) {
  if (loading.value || nextPage < 1 || nextPage > pageCount.value) return;
  await queryMetrics((nextPage - 1) * limit.value);
}

async function loadRevisions(item: FinancialMetric) {
  const key = rowKey(item);
  if (revisions[key] || revisionLoading[key]) return;
  revisionLoading[key] = true;
  revisionError[key] = "";
  try {
    revisions[key] = await listFinancialMetricRevisions(item.id);
  } catch (e) {
    revisionError[key] = `加载修订历史失败：${(e as Error).message}`;
  } finally {
    revisionLoading[key] = false;
  }
}

async function toggleDetails(item: FinancialMetric) {
  const key = rowKey(item);
  if (expandedId.value === key) {
    expandedId.value = "";
    return;
  }
  ensureDraft(item);
  expandedId.value = key;
  await loadRevisions(item);
}

function optionalPage(value: string, label: string): number | null {
  const text = value.trim();
  if (!text) return null;
  const parsed = Number(text);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${label}必须是正整数`);
  return parsed;
}

function revisionMeta(revision: FinancialMetricRevision) {
  return `${revisionActorLabel(revision)} · ${revision.created_at || "时间未提供"}`;
}

function revisionChanges(revision: FinancialMetricRevision) {
  const transition = revisionTransitionText(revision);
  return `旧值\n${transition.oldValue}\n\n新值\n${transition.newValue}`;
}

async function saveRevision(item: FinancialMetric) {
  const key = rowKey(item);
  const draft = drafts[key];
  saveError[key] = "";
  saveMessage[key] = "";
  if (!hasToken.value) {
    saveError[key] = "需登录或填写 token 后才能提交修订。";
    return;
  }
  if (!draft.reason.trim()) {
    saveError[key] = "请填写本次修订原因。";
    return;
  }
  try {
    const patch: FinancialMetricPatch = {
      raw_value: draft.rawValue.trim() || null,
      raw_unit: draft.rawUnit.trim() || null,
      statement_scope: draft.statementScope || null,
      source_title: draft.sourceTitle.trim() || null,
      source_page: optionalPage(draft.sourcePage, "来源页码"),
      source_page_end: optionalPage(draft.sourcePageEnd, "来源结束页码"),
      source_text: draft.sourceText.trim() || null,
      extraction_status: draft.extractionStatus || null,
      reason: draft.reason.trim(),
    };
    savingId.value = key;
    const updated = await updateFinancialMetric(item.id, patch);
    const index = items.value.findIndex((candidate) => candidate.id === item.id);
    if (index >= 0) items.value[index] = updated;
    drafts[key] = createDraft(updated);
    saveMessage[key] = "修订已保存，原因已随本次变更记录。";
    revisions[key] = await listFinancialMetricRevisions(item.id);
  } catch (e) {
    saveError[key] = `保存修订失败：${(e as Error).message}`;
  } finally {
    savingId.value = "";
  }
}

</script>

<template>
  <section class="analysis-view">
    <div class="analysis-command card">
      <div class="command-heading">
        <div>
          <span class="eyebrow">阶段 3 · 单公司研究</span>
          <h1>公司分析</h1>
          <p>输入公司（支持简称）和报告期（支持「2024年」简写），读取接口返回的五项指标事实与证据；页面不生成预测、评分或投资建议。</p>
        </div>
        <span class="auth-chip" :class="{ ready: hasToken }">{{ hasToken ? "已具备修订凭据" : "未登录：修订提交已禁用" }}</span>
      </div>
      <form class="analysis-form" @submit.prevent="generateAnalysis">
        <label>公司 <span class="required-mark">必填</span><input v-model="analysisForm.companyName" placeholder="公司名称或简称" /></label>
        <label>报告期 <span class="required-mark">必填</span><input v-model="analysisForm.reportPeriod" placeholder="如：2024年" /></label>
        <div class="analysis-actions"><button class="primary" type="submit" :disabled="analysisLoading">{{ analysisLoading ? "生成中…" : "生成分析" }}</button><button class="secondary" type="button" :disabled="analysisLoading" @click="clearAnalysis">清空</button></div>
      </form>
    </div>

    <div v-if="analysisError" class="state-banner error">{{ analysisError }}</div>
    <div v-else-if="analysisNotice" class="state-banner">{{ analysisNotice }}</div>
    <div v-if="analysisLoading" class="loading-state"><span class="loader-dot"></span>正在读取公司分析与当前期指标…</div>

    <section v-else-if="analysisGenerated && analysis" class="summary-stack">
      <div class="summary-header card">
        <div>
          <span class="eyebrow">研究摘要 · 已返回接口事实</span>
          <h2>{{ analysis.company?.name || analysisForm.companyName || "公司名称未提供" }}</h2>
          <p>{{ analysis.company?.code || "公司代码未提供" }} · 报告期 {{ analysis.report_period || analysisForm.reportPeriod || "未提供" }}</p>
        </div>
        <div class="summary-actions"><span class="summary-state">{{ analysis.metrics.length ? "已返回指标" : "未返回指标" }}</span><button class="export-button" type="button" :disabled="analysisBriefLoading || !analysisExportParams" @click="createAnalysisBrief">{{ analysisBriefLoading ? "简报生成中…" : "生成分析简报" }}</button><button class="export-button" type="button" :disabled="analysisExportLoading || !analysisExportParams" @click="exportAnalysisDraft">{{ analysisExportLoading ? "导出中…" : "导出研究底稿" }}</button></div>
      </div>
      <p v-if="analysisExportError" class="export-feedback error">{{ analysisExportError }}</p>
      <p v-else-if="analysisExportMessage" class="export-feedback success">{{ analysisExportMessage }}</p>
      <p v-if="analysisBriefError" class="export-feedback error">{{ analysisBriefError }}</p>

      <article v-if="analysisBrief" class="summary-panel card">
        <div class="panel-heading"><div><span class="panel-kicker">分析简报</span><h3>基于结构化分析结果生成</h3></div><div class="brief-actions"><span class="summary-state">{{ analysisBrief.brief_source === "llm" ? "LLM 简报" : "规则模板简报" }}</span><button class="export-button" type="button" @click="copyAnalysisBrief">一键复制</button></div></div>
        <pre class="brief-content">{{ analysisBrief.brief }}</pre>
        <p v-if="analysisBriefCopyMessage" class="export-feedback success">{{ analysisBriefCopyMessage }}</p>
      </article>

      <div class="metric-card-grid">
        <article v-for="metric in analysisMetrics" :key="metric.metric_code" class="metric-card card">
          <div class="metric-card-heading"><span>{{ metric.metric_name || metric.metric_code }}</span><code>{{ metric.metric_code }}</code></div>
          <div class="metric-values">
            <div><small>当前值</small><strong>{{ analysisMetricValue(metric.current) }}</strong><em>{{ analysisMetricSource(metric.current) }}</em></div>
          </div>
          <p class="comparison-note">{{ metric.comparison_note || "当前没有可比口径" }}</p>
        </article>
      </div>

      <div class="summary-columns">
        <article class="summary-panel card">
          <div class="panel-heading"><div><span class="panel-kicker">计算结果</span><h3>利润与现金流对照</h3></div><span class="summary-state" :class="{ pending: !analysis.profit_cash_flow.comparable }">{{ profitCashFlowRelationLabel(analysis.profit_cash_flow.relation) }}</span></div>
          <div class="cash-flow-grid">
            <div><small>归母净利润</small><strong>{{ formatAnalysisValue(analysis.profit_cash_flow.net_profit_value, "元") }}</strong></div>
            <div><small>经营现金流</small><strong>{{ formatAnalysisValue(analysis.profit_cash_flow.operating_cash_flow_value, "元") }}</strong></div>
            <div><small>差额</small><strong>{{ formatAnalysisValue(analysis.profit_cash_flow.difference, "元") }}</strong></div>
          </div>
          <p class="summary-note">{{ analysis.profit_cash_flow.note || "未提供对照说明。" }}</p>
        </article>

        <article class="summary-panel card">
          <div class="panel-heading"><div><span class="panel-kicker">证据追溯</span><h3>报告披露</h3></div><span>{{ analysis.disclosures.length }} 条</span></div>
          <div v-if="analysis.disclosures.length" class="disclosure-list">
            <div v-for="disclosure in analysis.disclosures" :key="`${disclosure.metric_code}-${disclosure.source_title}-${disclosure.source_page}`" class="disclosure-item">
              <div><strong>{{ disclosure.source_title || "来源文档未提供" }}</strong><span>{{ disclosurePage(disclosure) }}</span></div>
              <small>{{ disclosure.metric_code || "指标未提供" }}</small>
              <p>{{ disclosure.source_text || "未提供证据原文" }}</p>
            </div>
          </div>
          <p v-else class="muted summary-empty">暂无报告披露证据。</p>
        </article>
      </div>

      <article class="pending-panel card">
        <div class="panel-heading"><div><span class="panel-kicker">待核实</span><h3>待核实事项</h3></div><span>{{ analysis.pending_items.length }} 条</span></div>
        <div v-if="analysis.pending_items.length" class="pending-list">
          <div v-for="(pending, pendingIndex) in analysis.pending_items" :key="`${pending.code}-${pending.metric_code || 'none'}-${pendingIndex}`" class="pending-item"><strong>{{ pending.code }}</strong><span v-if="pending.metric_code">{{ pending.metric_code }}</span><p>{{ pending.message || "未提供说明" }}</p></div>
        </div>
        <p v-else class="muted summary-empty">当前响应未返回待核实事项。</p>
      </article>
    </section>

    <div v-else-if="!analysisGenerated && !analysisNotice" class="analysis-empty card"><strong>请先输入公司和报告期</strong><span>生成后这里会显示五项指标、现金流对照、报告披露和待核实事项。</span></div>

    <div class="analysis-toolbar card">
      <div class="toolbar-heading">
        <div>
          <span class="eyebrow">证据明细 · 人工核验</span>
          <h2>指标明细与人工修订</h2>
          <p>当前资料库：{{ currentKb }} · 只展示接口返回的财报记录，不生成未核验结论。</p>
        </div>
        <span class="auth-chip" :class="{ ready: hasToken }">
          {{ hasToken ? "已具备修订凭据" : "未登录：修订提交已禁用" }}
        </span>
      </div>
      <div class="filter-grid">
        <label>公司<input v-model="filters.companyName" placeholder="公司名称或简称" @keyup.enter="queryMetrics(0)" /></label>
        <label>报告期<input v-model="filters.reportPeriod" placeholder="如：2024年度" @keyup.enter="queryMetrics(0)" /></label>
        <label>指标<select v-model="filters.metricCode"><option value="">全部五项指标</option><option v-for="option in FINANCIAL_METRIC_OPTIONS" :key="option.code" :value="option.code">{{ option.label }}</option></select></label>
        <div class="filter-actions"><button class="primary" :disabled="loading" @click="queryMetrics(0)">{{ loading ? "查询中…" : "查询" }}</button><button class="secondary" :disabled="loading" @click="clearFilters">清空</button></div>
      </div>
    </div>

    <div v-if="error" class="state-banner error">{{ error }}</div>
    <div v-else-if="notice" class="state-banner">{{ notice }}</div>
    <div v-if="loading" class="loading-state"><span class="loader-dot"></span>正在读取指标记录…</div>

    <div v-else-if="items.length" class="table-shell">
      <table class="metrics-table">
        <thead><tr><th>公司</th><th>报告期</th><th>指标</th><th>原始值</th><th>归一值</th><th>口径</th><th>状态</th><th>来源</th><th></th></tr></thead>
        <tbody>
          <template v-for="item in items" :key="item.id">
            <tr>
              <td><strong>{{ item.company_name || "未提供" }}</strong><small>{{ item.company_code || "代码未提供" }}</small></td>
              <td>{{ item.report_period || "未提供" }}<small>{{ item.period_type || "期间类型未提供" }}</small></td>
              <td><strong>{{ item.metric_name || item.metric_code }}</strong><small>{{ item.metric_code }}</small></td>
              <td class="value-cell">{{ formatMetricValue(item.raw_value, item.raw_unit) }}</td>
              <td class="value-cell normalized">{{ formatMetricValue(item.normalized_value, item.normalized_unit) }}</td>
              <td><span class="tag">{{ statementScopeLabel(item.statement_scope) }}</span></td>
              <td><span class="status-tag" :class="`status-${item.extraction_status || 'unknown'}`">{{ metricStatusLabel(item.extraction_status) }}</span></td>
              <td><span class="source-title">{{ item.source_title || "未提供" }}</span><small>{{ formatPageRange(item.source_page, item.source_page_end) }}</small></td>
              <td><button class="detail-button" @click="toggleDetails(item)">{{ expandedId === rowKey(item) ? "收起" : "证据与历史" }}</button></td>
            </tr>
            <tr v-if="expandedId === rowKey(item)" class="details-row">
              <td colspan="9">
                <div class="detail-layout">
                  <article class="evidence-panel">
                    <div class="panel-heading"><div><span class="panel-kicker">证据追溯</span><h3>来源与原文</h3></div><span>{{ item.source_doc_id || "文档编号未提供" }}</span></div>
                    <dl class="source-meta"><div><dt>来源文档</dt><dd>{{ item.source_title || "未提供" }}</dd></div><div><dt>页码</dt><dd>{{ formatPageRange(item.source_page, item.source_page_end) }}</dd></div><div><dt>来源块</dt><dd>{{ item.source_chunk_id || "未提供" }}</dd></div></dl>
                    <blockquote>{{ item.source_text || "未提供证据原文" }}</blockquote>
                  </article>
                  <article class="revision-panel">
                    <div class="panel-heading"><div><span class="panel-kicker">变更追溯</span><h3>修订历史</h3></div><span v-if="revisionLoading[rowKey(item)]">读取中…</span></div>
                    <p v-if="revisionError[rowKey(item)]" class="inline-error">{{ revisionError[rowKey(item)] }}</p>
                    <p v-else-if="!revisionLoading[rowKey(item)] && !(revisions[rowKey(item)] || []).length" class="muted">暂无修订记录。</p>
                    <div v-else class="revision-list"><div v-for="revision in revisions[rowKey(item)] || []" :key="String(revision.id)" class="revision-item"><div class="revision-meta">{{ revisionMeta(revision) }}</div><strong>{{ revisionReasonLabel(revision) }}</strong><pre>{{ revisionChanges(revision) }}</pre></div></div>
                  </article>
                </div>
                <form class="edit-panel" @submit.prevent="saveRevision(item)">
                  <div class="panel-heading"><div><span class="panel-kicker">人工修订</span><h3>修订当前指标记录</h3></div><span class="muted">保存时必须填写原因</span></div>
                  <div class="edit-grid">
                    <label>原始值<input v-model="drafts[rowKey(item)].rawValue" placeholder="留空表示缺失" /></label>
                    <label>原始单位<input v-model="drafts[rowKey(item)].rawUnit" placeholder="如：元、万元、亿元" /></label>
                    <label>报表口径<select v-model="drafts[rowKey(item)].statementScope"><option v-for="option in scopeOptions" :key="option.value" :value="option.value">{{ option.label }}</option></select></label>
                    <label>提取状态<select v-model="drafts[rowKey(item)].extractionStatus"><option v-for="option in FINANCIAL_METRIC_STATUS_OPTIONS" :key="option.value" :value="option.value">{{ option.label }}</option></select></label>
                    <label>来源文档<input v-model="drafts[rowKey(item)].sourceTitle" placeholder="未提供时留空" /></label>
                    <label>来源页码<input v-model="drafts[rowKey(item)].sourcePage" inputmode="numeric" placeholder="起始页" /></label>
                    <label>结束页码<input v-model="drafts[rowKey(item)].sourcePageEnd" inputmode="numeric" placeholder="结束页，可留空" /></label>
                    <label class="wide-field">证据原文<textarea v-model="drafts[rowKey(item)].sourceText" rows="3" placeholder="未提供时留空"></textarea></label>
                    <label class="wide-field">修订原因（必填）<textarea v-model="drafts[rowKey(item)].reason" rows="2" placeholder="说明为什么修改，以及依据哪份报告或页码"></textarea></label>
                  </div>
                  <div class="edit-actions"><span v-if="saveError[rowKey(item)]" class="inline-error">{{ saveError[rowKey(item)] }}</span><span v-else-if="saveMessage[rowKey(item)]" class="inline-success">{{ saveMessage[rowKey(item)] }}</span><span v-else-if="!hasToken" class="muted">需登录或填写 token 后才能提交修订。</span><button class="primary" type="submit" :disabled="!hasToken || savingId === rowKey(item) || !drafts[rowKey(item)].reason.trim()">{{ savingId === rowKey(item) ? "保存中…" : "保存修订" }}</button></div>
                </form>
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>

    <div v-else-if="!error && !loading" class="empty-state"><strong>{{ analysisGenerated ? "暂无财报指标记录" : "尚未读取指标明细" }}</strong><span>{{ analysisGenerated ? "请调整公司、报告期或指标筛选条件；页面不会用示例数据填充空结果。" : "生成分析后会同步读取当前期指标明细；页面不会自动请求或填充示例数据。" }}</span></div>
    <div class="pagination"><span>{{ pageSummary }}</span><div><button class="secondary" :disabled="loading || currentPage <= 1" @click="changePage(currentPage - 1)">上一页</button><button class="secondary" :disabled="loading || currentPage >= pageCount" @click="changePage(currentPage + 1)">下一页</button></div></div>
  </section>
</template>

<style scoped>
.analysis-view { padding: 16px 24px 28px; min-width: 0; }
.card, .table-shell, .empty-state, .loading-state { border: 1px solid var(--color-border); border-radius: var(--radius-lg); background: var(--color-bg); }
.analysis-command { padding: 18px; box-shadow: var(--shadow-sm); }
.command-heading, .summary-header, .metric-card-heading, .cash-flow-grid, .disclosure-item > div, .pending-item { display: flex; gap: 12px; }
.command-heading, .summary-header { align-items: flex-start; justify-content: space-between; }
.command-heading h1, .summary-header h2 { margin: 5px 0 4px; font-size: var(--text-xl); }
.command-heading p, .summary-header p { margin: 0; color: var(--color-charcoal); font-size: var(--text-sm); line-height: 1.5; }
.analysis-form { display: grid; grid-template-columns: 1.2fr 1fr 1fr auto; gap: 10px; align-items: end; margin-top: 18px; }
.analysis-form label { display: flex; flex-direction: column; gap: 5px; color: var(--color-charcoal); font-size: var(--text-xs); }
.analysis-form input { width: 100%; min-width: 0; padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); color: var(--color-ink); background: var(--color-bg); font: inherit; font-size: var(--text-sm); }
.required-mark { color: var(--color-accent); }
.optional-mark { color: var(--color-slate); }
.analysis-actions { display: flex; gap: 8px; }
.summary-stack { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
.summary-header, .summary-panel, .pending-panel { padding: 16px; }
.summary-actions { display: flex; align-items: center; gap: 8px; }
.export-button { border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); padding: 7px 11px; color: var(--color-charcoal); background: var(--color-bg); font: inherit; font-size: var(--text-xs); cursor: pointer; }
.export-button:disabled { opacity: .55; cursor: not-allowed; }
.export-feedback { margin: -4px 0 0; font-size: var(--text-xs); }
.export-feedback.error { color: var(--color-error); }
.export-feedback.success { color: var(--color-success); }
.summary-state { display: inline-flex; align-items: center; white-space: nowrap; border: 1px solid rgba(15, 123, 108, .25); border-radius: var(--radius-pill); padding: 4px 9px; color: var(--color-success); background: var(--color-success-bg); font-size: var(--text-xs); }
.summary-state.pending { border-color: rgba(180, 107, 27, .25); color: var(--color-warning); background: var(--color-warning-bg); }
.metric-card-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.metric-card { min-width: 0; padding: 14px; }
.metric-card-heading { align-items: baseline; justify-content: space-between; min-height: 34px; color: var(--color-ink-soft); font-size: var(--text-sm); font-weight: var(--weight-semibold); }
.metric-card-heading code { color: var(--color-slate); font: 10px var(--font-mono); }
.metric-values { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.metric-values div { min-width: 0; padding: 8px; border-radius: var(--radius-sm); background: var(--color-bg-quiet); }
.metric-values small, .cash-flow-grid small { display: block; color: var(--color-slate); font-size: var(--text-xs); }
.metric-values strong, .cash-flow-grid strong { display: block; max-width: 100%; margin-top: 4px; overflow-x: auto; color: var(--color-ink-soft); font-size: clamp(.75rem, 1.1vw, var(--text-sm)); font-variant-numeric: tabular-nums; white-space: nowrap; }
.metric-values em { display: block; margin-top: 4px; overflow: hidden; color: var(--color-slate); font-size: 10px; font-style: normal; text-overflow: ellipsis; white-space: nowrap; }
.change-line { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; color: var(--color-success); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
.change-line.pending { color: var(--color-warning); }
.comparison-note, .summary-note { margin: 9px 0 0; color: var(--color-slate); font-size: var(--text-xs); line-height: 1.5; }
.brief-actions { display: flex; align-items: center; gap: 8px; }
.brief-content { margin: 0; padding: 12px; overflow: auto; border: 1px solid var(--color-border-soft); border-radius: var(--radius-md); color: var(--color-charcoal); background: var(--color-bg-quiet); font: 12px/1.65 var(--font-mono); white-space: pre-wrap; word-break: break-word; }
.summary-columns { display: grid; grid-template-columns: minmax(0, .9fr) minmax(0, 1.1fr); gap: 12px; }
.cash-flow-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 12px; }
.cash-flow-grid > div { padding: 10px; border-left: 2px solid var(--color-accent); background: var(--color-bg-quiet); }
.disclosure-list { display: flex; flex-direction: column; gap: 8px; max-height: 220px; overflow: auto; }
.disclosure-item { padding: 9px; border: 1px solid var(--color-border-soft); border-radius: var(--radius-sm); }
.disclosure-item > div { align-items: baseline; justify-content: space-between; }
.disclosure-item strong { overflow: hidden; color: var(--color-ink-soft); font-size: var(--text-sm); text-overflow: ellipsis; white-space: nowrap; }
.disclosure-item span, .disclosure-item small { color: var(--color-slate); font-size: var(--text-xs); }
.disclosure-item p { margin: 6px 0 0; color: var(--color-charcoal); font-size: var(--text-xs); line-height: 1.5; white-space: pre-wrap; }
.summary-empty { margin: 18px 0 4px; }
.pending-list { display: flex; flex-direction: column; gap: 7px; }
.pending-item { align-items: baseline; padding: 8px 10px; border-left: 3px solid var(--color-warning); background: var(--color-warning-bg); }
.pending-item strong { color: var(--color-warning); font: 11px var(--font-mono); }
.pending-item span { color: var(--color-slate); font-size: var(--text-xs); }
.pending-item p { flex: 1; margin: 0; color: var(--color-charcoal); font-size: var(--text-sm); }
.analysis-empty { display: flex; flex-direction: column; gap: 5px; margin-top: 12px; padding: 20px; color: var(--color-charcoal); font-size: var(--text-sm); }
.analysis-empty strong { color: var(--color-ink-soft); }
.analysis-toolbar { padding: 18px; box-shadow: var(--shadow-sm); }
.toolbar-heading, .filter-actions, .edit-actions, .panel-heading, .pagination { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.eyebrow, .panel-kicker { color: var(--color-accent); font-size: var(--text-xs); font-weight: var(--weight-semibold); letter-spacing: .08em; }
.toolbar-heading h2 { margin: 5px 0 4px; font-size: var(--text-xl); }
.toolbar-heading p { margin: 0; color: var(--color-charcoal); font-size: var(--text-sm); }
.auth-chip, .tag, .status-tag { display: inline-flex; align-items: center; white-space: nowrap; border: 1px solid var(--color-border); border-radius: var(--radius-pill); padding: 4px 9px; color: var(--color-charcoal); background: var(--color-bg-quiet); font-size: var(--text-xs); }
.auth-chip.ready { color: var(--color-success); border-color: rgba(15, 123, 108, .25); background: var(--color-success-bg); }
.filter-grid { display: grid; grid-template-columns: 1.2fr 1fr 1.2fr auto; gap: 10px; align-items: end; margin-top: 18px; }
.filter-grid label, .edit-grid label { display: flex; flex-direction: column; gap: 5px; color: var(--color-charcoal); font-size: var(--text-xs); }
.filter-grid input, .filter-grid select, .edit-grid input, .edit-grid select, .edit-grid textarea { width: 100%; min-width: 0; padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); color: var(--color-ink); background: var(--color-bg); font: inherit; font-size: var(--text-sm); }
.edit-grid textarea { resize: vertical; line-height: 1.5; }
button { cursor: pointer; font: inherit; }
.primary, .secondary, .detail-button { border-radius: var(--radius-md); padding: 8px 14px; font-size: var(--text-sm); }
.primary { border: 1px solid var(--color-accent); color: var(--color-bg); background: var(--color-accent); }
.secondary, .detail-button { border: 1px solid var(--color-border); color: var(--color-charcoal); background: var(--color-bg); }
.primary:disabled, .secondary:disabled { opacity: .5; cursor: not-allowed; }
.filter-actions { justify-content: flex-start; }
.state-banner { margin-top: 12px; padding: 10px 12px; border-radius: var(--radius-md); color: var(--color-charcoal); background: var(--color-bg-quiet); font-size: var(--text-sm); }
.state-banner.error, .inline-error { color: var(--color-error); }
.loading-state, .empty-state { display: flex; align-items: center; gap: 8px; margin-top: 12px; padding: 22px; color: var(--color-charcoal); font-size: var(--text-sm); }
.empty-state { flex-direction: column; align-items: flex-start; }
.empty-state strong { color: var(--color-ink-soft); }
.loader-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--color-accent); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .35; transform: scale(.7); } }
.table-shell { overflow: auto; margin-top: 12px; }
.metrics-table { width: 100%; min-width: 1120px; border-collapse: collapse; font-size: var(--text-sm); }
.metrics-table th { padding: 11px 12px; border-bottom: 1px solid var(--color-border); color: var(--color-slate); font-size: var(--text-xs); font-weight: var(--weight-medium); text-align: left; white-space: nowrap; background: var(--color-bg-quiet); }
.metrics-table td { padding: 12px; border-bottom: 1px solid var(--color-border-soft); vertical-align: top; }
.metrics-table tbody tr:hover { background: var(--color-bg-warm); }
.metrics-table td strong { display: block; color: var(--color-ink-soft); font-weight: var(--weight-medium); }
.metrics-table td small, .source-title + small { display: block; margin-top: 3px; color: var(--color-slate); font-size: var(--text-xs); }
.value-cell { color: var(--color-ink-soft); font-variant-numeric: tabular-nums; white-space: nowrap; }
.value-cell.normalized { color: var(--color-accent); }
.status-verified, .status-corrected { color: var(--color-success); }
.status-conflict { color: var(--color-warning); }
.status-failed { color: var(--color-error); }
.status-missing, .status-unknown { color: var(--color-charcoal); }
.details-row td { padding: 0; background: var(--color-bg-quiet); }
.detail-layout { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(300px, .8fr); gap: 12px; padding: 16px; }
.evidence-panel, .revision-panel, .edit-panel { padding: 14px; border: 1px solid var(--color-border); border-radius: var(--radius-md); background: var(--color-bg); }
.panel-heading { align-items: flex-start; margin-bottom: 12px; color: var(--color-slate); font-size: var(--text-xs); }
.panel-heading h3 { margin: 3px 0 0; color: var(--color-ink); font-size: var(--text-base); }
.source-meta { display: grid; grid-template-columns: 1.4fr .8fr 1.1fr; gap: 8px; margin: 0; }
.source-meta div { min-width: 0; padding: 8px; border-radius: var(--radius-sm); background: var(--color-bg-quiet); }
.source-meta dt { color: var(--color-slate); font-size: var(--text-xs); }
.source-meta dd { margin: 3px 0 0; overflow: hidden; color: var(--color-charcoal); font-size: var(--text-xs); text-overflow: ellipsis; white-space: nowrap; }
blockquote { max-height: 190px; overflow: auto; margin: 10px 0 0; padding: 12px; border-left: 3px solid var(--color-accent); color: var(--color-ink-soft); background: var(--color-bg-quiet); font-size: var(--text-sm); line-height: 1.6; white-space: pre-wrap; }
.muted { color: var(--color-slate); font-size: var(--text-xs); }
.revision-list { display: flex; flex-direction: column; gap: 8px; max-height: 240px; overflow: auto; }
.revision-item { padding: 9px; border: 1px solid var(--color-border-soft); border-radius: var(--radius-sm); }
.revision-meta { margin-bottom: 4px; color: var(--color-slate); font-size: var(--text-xs); }
.revision-item strong { color: var(--color-ink-soft); font-size: var(--text-sm); font-weight: var(--weight-medium); }
.revision-item pre { max-height: 100px; overflow: auto; margin: 7px 0 0; color: var(--color-charcoal); font: 11px/1.5 var(--font-mono); white-space: pre-wrap; }
.edit-panel { margin: 0 16px 16px; }
.edit-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.wide-field { grid-column: span 2; }
.edit-actions { justify-content: flex-end; margin-top: 12px; }
.inline-success { color: var(--color-success); font-size: var(--text-xs); }
.pagination { margin-top: 12px; color: var(--color-slate); font-size: var(--text-xs); }
.pagination div { display: flex; gap: 8px; }
@media (max-width: 1024px) {
  .analysis-form { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .analysis-actions { grid-column: 1 / -1; }
  .metric-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .summary-columns { grid-template-columns: 1fr; }
  .filter-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .filter-actions { grid-column: 1 / -1; }
  .detail-layout { grid-template-columns: 1fr; }
  .edit-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .analysis-view { padding: 12px; }
  .command-heading, .toolbar-heading, .summary-header { align-items: flex-start; flex-direction: column; }
  .summary-actions { flex-wrap: wrap; }
  .analysis-form, .metric-card-grid { grid-template-columns: 1fr; }
  .analysis-actions { grid-column: auto; }
  .cash-flow-grid { grid-template-columns: 1fr; }
  .filter-grid, .edit-grid { grid-template-columns: 1fr; }
  .filter-actions, .wide-field { grid-column: auto; }
  .source-meta { grid-template-columns: 1fr; }
  .edit-actions { align-items: flex-start; flex-direction: column; }
}
</style>
