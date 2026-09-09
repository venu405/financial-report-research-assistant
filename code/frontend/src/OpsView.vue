<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  type OperationsAlert,
  type OperationsDashboard,
  acknowledgeOperationsAlert,
  fetchOperationsAlerts,
  fetchOperationsDashboard,
} from "./services/api";
import { readMetric } from "./services/operations-utils.js";

const apiKey = ref(sessionStorage.getItem("kb_admin_key") || "");
const adminToken = ref(sessionStorage.getItem("kb_api_token") || sessionStorage.getItem("kb_user_id") || "");
const dashboard = ref<OperationsDashboard>({});
const alerts = ref<OperationsAlert[]>([]);
const loading = ref(false);
const error = ref("");
const updatedAt = ref("");
const acknowledging = ref<number | string | null>(null);

function saveCredentials() {
  if (apiKey.value) sessionStorage.setItem("kb_admin_key", apiKey.value);
  else sessionStorage.removeItem("kb_admin_key");
  if (adminToken.value) sessionStorage.setItem("kb_api_token", adminToken.value);
  else sessionStorage.removeItem("kb_api_token");
}

function metric(paths: string[], fallback = "0"): string {
  const value = readMetric(dashboard.value, paths, fallback);
  if (typeof value === "number") return value.toLocaleString();
  return String(value ?? fallback);
}

function percent(paths: string[]): string {
  const value = readMetric(dashboard.value, paths, 0);
  if (typeof value === "number") return `${value <= 1 ? Math.round(value * 100) : Math.round(value)}%`;
  return String(value || "0%");
}

const sessionCards = computed(() => [
  { label: "总会话", value: metric(["sessions.total", "conversations.total", "session.total"]) },
  { label: "待接入", value: metric(["sessions.waiting", "conversations.waiting", "session.waiting"]) },
  { label: "人工处理中", value: metric(["sessions.human", "conversations.human", "session.human"]) },
  { label: "已关闭", value: metric(["sessions.closed", "conversations.closed", "session.closed"]) },
]);

const ticketCards = computed(() => [
  { label: "工单总数", value: metric(["tickets.total", "ticket.total"]) },
  { label: "待处理", value: metric(["tickets.pending", "ticket.pending"]) },
  { label: "处理中", value: metric(["tickets.processing", "ticket.processing"]) },
  { label: "已解决", value: metric(["tickets.resolved", "ticket.resolved"]) },
]);

const unacknowledgedAlerts = computed(() => alerts.value.filter((alert) =>
  alert.status ? alert.status === "open" : !alert.acknowledged && !alert.acked,
));

async function refresh() {
  error.value = "";
  loading.value = true;
  try {
    const [summary, alertData] = await Promise.all([
      fetchOperationsDashboard(apiKey.value, adminToken.value),
      fetchOperationsAlerts(apiKey.value, adminToken.value),
    ]);
    dashboard.value = summary || {};
    alerts.value = alertData.alerts || [];
    updatedAt.value = dashboard.value.generated_at || new Date().toLocaleString();
  } catch (e) {
    error.value = `加载运营数据失败：${(e as Error).message}`;
  } finally {
    loading.value = false;
  }
}

async function acknowledge(alert: OperationsAlert) {
  acknowledging.value = alert.id;
  error.value = "";
  try {
    await acknowledgeOperationsAlert(alert.id, apiKey.value, adminToken.value);
    alert.acknowledged = true;
    alert.acked = true;
    alert.status = "acknowledged";
  } catch (e) {
    error.value = `确认告警失败：${(e as Error).message}`;
  } finally {
    acknowledging.value = null;
  }
}

onMounted(refresh);
</script>

<template>
  <section class="ops-page">
    <div class="ops-toolbar card">
      <div>
        <span class="eyebrow">OPERATIONS CONTROL ROOM</span>
        <h2>运营看板</h2>
        <p class="muted">会话、工单、满意度和 RAG 质量集中监测</p>
      </div>
      <div class="toolbar-actions">
        <input v-model="adminToken" type="password" placeholder="运营/管理员 Token" @change="saveCredentials" />
        <input v-model="apiKey" type="password" placeholder="X-API-Key（可选）" @change="saveCredentials" />
        <button class="primary" :disabled="loading" @click="refresh">{{ loading ? "刷新中…" : "刷新数据" }}</button>
      </div>
    </div>
    <p v-if="error" class="feedback error">{{ error }}</p>
    <p v-if="updatedAt" class="updated">最近更新：{{ updatedAt }}</p>

    <section class="metric-section">
      <div class="section-title"><h3>会话概览</h3><span class="muted">实时汇总</span></div>
      <div class="metric-grid">
        <article v-for="card in sessionCards" :key="card.label" class="metric-card">
          <span>{{ card.label }}</span><strong>{{ card.value }}</strong>
        </article>
      </div>
    </section>

    <section class="metric-section">
      <div class="section-title"><h3>工单概览</h3><span class="muted">当前状态分布</span></div>
      <div class="metric-grid">
        <article v-for="card in ticketCards" :key="card.label" class="metric-card">
          <span>{{ card.label }}</span><strong>{{ card.value }}</strong>
        </article>
      </div>
    </section>

    <div class="detail-grid">
      <section class="card detail-card">
        <div class="section-title"><h3>满意度</h3><span class="score">{{ percent(["feedback.satisfaction_rate"]) }}</span></div>
        <div class="quality-list">
          <div><span>评价数</span><strong>{{ metric(["feedback.total"]) }}</strong></div>
          <div><span>好评数</span><strong>{{ metric(["feedback.positive"]) }}</strong></div>
          <div><span>差评数</span><strong>{{ metric(["feedback.negative"]) }}</strong></div>
        </div>
      </section>
      <section class="card detail-card">
        <div class="section-title"><h3>RAG 质量</h3><span class="score">{{ metric(["rag.avg_faithfulness"]) }}</span></div>
        <div class="quality-list">
          <div><span>问答样本</span><strong>{{ metric(["rag.total"]) }}</strong></div>
          <div><span>转人工率</span><strong>{{ percent(["rag.escalation_rate"]) }}</strong></div>
          <div><span>平均证据分</span><strong>{{ metric(["rag.avg_evidence_score"]) }}</strong></div>
        </div>
      </section>
    </div>

    <section class="card alerts-card">
      <div class="section-title"><h3>运营告警</h3><span class="status-chip">{{ unacknowledgedAlerts.length }} 条待确认</span></div>
      <div v-if="alerts.length" class="alert-list">
        <div v-for="alert in alerts" :key="alert.id" class="alert-row" :class="`alert-${alert.level || alert.severity || 'info'}`">
          <div class="alert-copy">
            <strong>{{ alert.title || alert.message || "运营告警" }}</strong>
            <p>{{ alert.message || alert.detail || "请查看告警详情" }}</p>
            <small>{{ alert.created_at || "" }}</small>
          </div>
          <button v-if="alert.status ? alert.status === 'open' : !alert.acknowledged && !alert.acked" class="secondary" :disabled="acknowledging === alert.id" @click="acknowledge(alert)">
            {{ acknowledging === alert.id ? "确认中…" : "确认告警" }}
          </button>
          <span v-else class="ack">已确认</span>
        </div>
      </div>
      <p v-else class="empty">当前没有运营告警</p>
    </section>
  </section>
</template>

<style scoped>
.ops-page { padding: 16px; display: flex; flex-direction: column; gap: 16px; }
.card { background: var(--color-bg); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 16px; }
.ops-toolbar { display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; }
.ops-toolbar h2 { margin: 4px 0; }
.eyebrow { color: var(--color-slate); font-size: var(--text-xs); letter-spacing: .08em; }
.muted, .updated { color: var(--color-slate); font-size: var(--text-xs); }
.toolbar-actions { display: grid; grid-template-columns: repeat(2, minmax(130px, 1fr)) auto; gap: 8px; min-width: min(100%, 620px); }
input { width: 100%; padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); background: var(--color-bg); color: var(--color-ink); font: inherit; }
button { font: inherit; cursor: pointer; }
button:disabled { opacity: .55; cursor: not-allowed; }
.primary, .secondary { border-radius: var(--radius-md); padding: 8px 14px; white-space: nowrap; }
.primary { border: 0; background: var(--color-accent); color: white; }
.secondary { background: var(--color-bg-warm); border: 1px solid var(--color-border); color: var(--color-ink); }
.feedback { margin: 0; padding: 8px 12px; border-radius: var(--radius-md); font-size: var(--text-sm); }
.feedback.error { color: var(--color-error); background: var(--color-error-bg); }
.metric-section { display: flex; flex-direction: column; gap: 8px; }
.section-title { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.section-title h3 { margin: 0; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.metric-card { padding: 14px 16px; background: var(--color-bg); border: 1px solid var(--color-border); border-radius: var(--radius-lg); }
.metric-card span { display: block; color: var(--color-charcoal); font-size: var(--text-sm); }
.metric-card strong { display: block; margin-top: 8px; font-size: var(--text-2xl); letter-spacing: var(--tracking-heading); }
.detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.detail-card { min-height: 170px; }
.score { color: var(--color-accent); font-size: var(--text-xl); font-weight: var(--weight-semibold); }
.quality-list { display: grid; gap: 9px; margin-top: 16px; }
.quality-list div { display: flex; justify-content: space-between; gap: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--color-border-soft); }
.quality-list span { color: var(--color-charcoal); }
.alerts-card { min-height: 150px; }
.status-chip { color: var(--color-warning); background: var(--color-warning-bg); border-radius: var(--radius-pill); padding: 3px 9px; font-size: var(--text-xs); }
.alert-list { display: grid; gap: 8px; margin-top: 12px; }
.alert-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 12px; border-left: 3px solid var(--color-warning); background: var(--color-bg-quiet); border-radius: var(--radius-md); }
.alert-critical, .alert-error { border-left-color: var(--color-error); }
.alert-info { border-left-color: var(--color-accent); }
.alert-copy { min-width: 0; }
.alert-copy p { margin: 4px 0; color: var(--color-charcoal); white-space: pre-wrap; }
.alert-copy small { color: var(--color-slate); }
.ack { color: var(--color-success); font-size: var(--text-sm); white-space: nowrap; }
.empty { color: var(--color-slate); text-align: center; padding: 20px; font-size: var(--text-sm); }
@media (max-width: 900px) { .ops-toolbar { align-items: stretch; flex-direction: column; } .toolbar-actions { min-width: 0; } .metric-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .toolbar-actions, .detail-grid { grid-template-columns: 1fr; } .metric-grid { grid-template-columns: 1fr 1fr; } .alert-row { align-items: stretch; flex-direction: column; } }
</style>
