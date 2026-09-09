<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  type Ticket,
  createTicket,
  getTicket,
  listTickets,
  updateTicketStatus,
  assignTicket,
} from "./services/api";
import { ticketStatusLabel, TICKET_STATUS_LABELS } from "./services/operations-utils.js";

const apiKey = ref(sessionStorage.getItem("kb_admin_key") || "");
const adminToken = ref(sessionStorage.getItem("kb_api_token") || sessionStorage.getItem("kb_user_id") || "");
const tickets = ref<Ticket[]>([]);
const selected = ref<Ticket | null>(null);
const filter = ref("all");
const search = ref("");
const statusDraft = ref("");
const note = ref("");
const assignee = ref("");
const loading = ref(false);
const detailLoading = ref(false);
const saving = ref(false);
const error = ref("");
const message = ref("");
const newTitle = ref("");
const newDescription = ref("");
const newPriority = ref("normal");
const newDueHours = ref(24);

const statusOptions = Object.keys(TICKET_STATUS_LABELS);

const filteredTickets = computed(() => {
  const query = search.value.trim().toLowerCase();
  return tickets.value.filter((ticket) => {
    const matchesStatus = filter.value === "all" || ticket.status === filter.value;
    const haystack = `${ticket.ticket_no} ${ticket.title} ${ticket.assignee || ""}`.toLowerCase();
    return matchesStatus && (!query || haystack.includes(query));
  });
});

function saveCredentials() {
  if (apiKey.value) sessionStorage.setItem("kb_admin_key", apiKey.value);
  else sessionStorage.removeItem("kb_admin_key");
  if (adminToken.value) sessionStorage.setItem("kb_api_token", adminToken.value);
  else sessionStorage.removeItem("kb_api_token");
}

function clearFeedback() {
  error.value = "";
  message.value = "";
}

async function loadTickets() {
  clearFeedback();
  loading.value = true;
  try {
    const data = await listTickets(apiKey.value, adminToken.value);
    tickets.value = data.tickets || [];
    if (selected.value) {
      const fresh = tickets.value.find((ticket) => ticket.id === selected.value?.id);
      if (fresh) await selectTicket(fresh);
    }
  } catch (e) {
    error.value = `加载工单失败：${(e as Error).message}`;
  } finally {
    loading.value = false;
  }
}

async function selectTicket(ticket: Ticket) {
  clearFeedback();
  selected.value = ticket;
  statusDraft.value = ticket.status;
  assignee.value = ticket.assignee || "";
  note.value = "";
  detailLoading.value = true;
  try {
    selected.value = await getTicket(ticket.id, apiKey.value, adminToken.value);
    statusDraft.value = selected.value.status;
    assignee.value = selected.value.assignee || "";
  } catch (e) {
    error.value = `加载工单详情失败：${(e as Error).message}`;
  } finally {
    detailLoading.value = false;
  }
}

async function changeStatus() {
  if (!selected.value || !statusDraft.value || statusDraft.value === selected.value.status) return;
  clearFeedback();
  saving.value = true;
  try {
    await updateTicketStatus(selected.value.id, statusDraft.value, note.value.trim(), apiKey.value, adminToken.value);
    message.value = "工单状态已更新";
    await loadTickets();
    if (selected.value) await selectTicket(selected.value);
  } catch (e) {
    error.value = `更新状态失败：${(e as Error).message}`;
  } finally {
    saving.value = false;
  }
}

async function saveAssignment() {
  if (!selected.value || !assignee.value.trim()) return;
  clearFeedback();
  saving.value = true;
  try {
    await assignTicket(selected.value.id, assignee.value.trim(), apiKey.value, adminToken.value);
    message.value = "工单已指派，状态已进入处理中";
    await loadTickets();
    if (selected.value) await selectTicket(selected.value);
  } catch (e) {
    error.value = `指派失败：${(e as Error).message}`;
  } finally {
    saving.value = false;
  }
}

async function addTicket() {
  if (!newTitle.value.trim()) { error.value = "请填写工单标题"; return; }
  clearFeedback();
  saving.value = true;
  try {
    await createTicket({
      title: newTitle.value.trim(), description: newDescription.value.trim(),
      kb_id: localStorage.getItem("kb_id") || "default",
      priority: newPriority.value, due_hours: Number(newDueHours.value),
    }, apiKey.value, adminToken.value);
    newTitle.value = "";
    newDescription.value = "";
    message.value = "工单已创建";
    await loadTickets();
  } catch (e) {
    error.value = `创建工单失败：${(e as Error).message}`;
  } finally {
    saving.value = false;
  }
}

onMounted(loadTickets);
</script>

<template>
  <section class="ticket-page">
    <div class="ticket-toolbar card">
      <div class="toolbar-title">
        <div>
          <span class="eyebrow">SERVICE OPERATIONS</span>
        <h2>核验任务中心</h2>
        </div>
        <button class="primary" :disabled="loading" @click="loadTickets">{{ loading ? "加载中…" : "刷新" }}</button>
      </div>
      <div class="credentials">
        <input v-model="adminToken" type="password" placeholder="运营/管理员 Token" @change="saveCredentials" />
        <input v-model="apiKey" type="password" placeholder="X-API-Key（可选）" @change="saveCredentials" />
      </div>
      <div class="filters">
        <input v-model="search" placeholder="搜索编号、标题或坐席" />
        <select v-model="filter" aria-label="工单状态筛选">
          <option value="all">全部状态</option>
          <option v-for="status in statusOptions" :key="status" :value="status">{{ ticketStatusLabel(status) }}</option>
        </select>
      </div>
      <div class="create-ticket">
        <input v-model="newTitle" placeholder="新工单标题" />
        <input v-model="newDescription" placeholder="问题描述（可选）" @keyup.enter="addTicket" />
        <select v-model="newPriority" aria-label="新工单优先级"><option value="low">低</option><option value="normal">普通</option><option value="high">高</option><option value="urgent">紧急</option></select>
        <input v-model.number="newDueHours" type="number" min="1" max="8760" aria-label="处理时限（小时）" />
        <button class="primary" :disabled="saving" @click="addTicket">新建工单</button>
      </div>
    </div>

    <p v-if="error" class="feedback error">{{ error }}</p>
    <p v-if="message" class="feedback success">{{ message }}</p>

    <div class="ticket-layout">
      <section class="card ticket-list">
        <div class="section-head">
          <h3>工单列表</h3>
          <span class="muted">{{ filteredTickets.length }} 条</span>
        </div>
        <button
          v-for="ticket in filteredTickets"
          :key="ticket.id"
          class="ticket-row"
          :class="{ active: selected?.id === ticket.id }"
          @click="selectTicket(ticket)"
        >
          <span class="ticket-row-main">
            <strong>{{ ticket.ticket_no }}</strong>
            <span>{{ ticket.title || "无标题工单" }}</span>
          </span>
          <span class="ticket-row-meta">
            <span class="status-chip" :class="`status-${ticket.status}`">{{ ticketStatusLabel(ticket.status) }}</span>
            <small>{{ ticket.priority || "normal" }} 优先级</small>
            <small>{{ ticket.assignee || "未指派" }}</small>
          </span>
        </button>
        <p v-if="!filteredTickets.length" class="empty">暂无符合条件的工单</p>
      </section>

      <section class="card ticket-detail">
        <template v-if="selected">
          <div class="section-head detail-head">
            <div>
              <span class="eyebrow">{{ selected.ticket_no }}</span>
              <h3>{{ selected.title || "无标题工单" }}</h3>
            </div>
            <span class="status-chip" :class="`status-${selected.status}`">{{ ticketStatusLabel(selected.status) }}</span>
          </div>
          <p v-if="selected.description" class="description">{{ selected.description }}</p>
          <p v-if="selected.conversation_id" class="muted">关联会话：{{ selected.conversation_id }}</p>
          <p class="muted">知识库：{{ selected.kb_id || "default" }} · 优先级：{{ selected.priority || "normal" }} · 截止时间：{{ selected.due_at || "未设置" }}</p>

          <div v-if="detailLoading" class="empty">正在加载详情…</div>
          <template v-else>
            <div class="detail-grid">
              <label>
                <span>状态流转</span>
                <select v-model="statusDraft">
                  <option v-for="status in statusOptions" :key="status" :value="status">{{ ticketStatusLabel(status) }}</option>
                </select>
              </label>
              <label>
                <span>流转备注</span>
                <input v-model="note" placeholder="可选，记录处理说明" />
              </label>
              <button class="primary" :disabled="saving || statusDraft === selected.status" @click="changeStatus">保存状态</button>
            </div>
            <div class="assignment-row">
              <label>
                <span>主管指派</span>
                <input v-model="assignee" placeholder="坐席 user_id" />
              </label>
              <button class="secondary" :disabled="saving || !assignee.trim()" @click="saveAssignment">指派并进入处理中</button>
            </div>

            <div class="timeline">
              <div class="section-head"><h4>处理时间线</h4></div>
              <div v-for="(item, index) in selected.progress || []" :key="`${item.created_at}-${index}`" class="timeline-item">
                <span class="timeline-dot" />
                <div>
                  <strong>{{ ticketStatusLabel(item.action) }}</strong>
                  <p>{{ item.note || "状态已更新" }}</p>
                  <small>{{ item.created_at }}</small>
                </div>
              </div>
              <p v-if="!selected.progress?.length" class="empty">暂无时间线记录</p>
            </div>
          </template>
        </template>
        <p v-else class="empty detail-empty">选择左侧工单查看详情</p>
      </section>
    </div>
  </section>
</template>

<style scoped>
.ticket-page { padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.card { background: var(--color-bg); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 16px; }
.ticket-toolbar { display: flex; flex-direction: column; gap: 12px; }
.toolbar-title, .section-head, .detail-head, .assignment-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.toolbar-title h2, h3, h4 { margin: 3px 0 0; }
.eyebrow { color: var(--color-slate); font-size: var(--text-xs); letter-spacing: .08em; }
.credentials, .filters, .detail-grid, .create-ticket { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.filters { grid-template-columns: 1fr 180px; }
.create-ticket { grid-template-columns: 1fr 1fr 110px 110px auto; }
input, select { width: 100%; padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); background: var(--color-bg); color: var(--color-ink); font: inherit; }
button { font: inherit; cursor: pointer; }
button:disabled { opacity: .55; cursor: not-allowed; }
.primary, .secondary { border: 0; border-radius: var(--radius-md); padding: 8px 14px; white-space: nowrap; }
.primary { background: var(--color-accent); color: white; }
.secondary { background: var(--color-bg-warm); color: var(--color-ink); border: 1px solid var(--color-border); }
.feedback { margin: 0; padding: 8px 12px; border-radius: var(--radius-md); font-size: var(--text-sm); }
.feedback.error { color: var(--color-error); background: var(--color-error-bg); }
.feedback.success { color: var(--color-success); background: var(--color-success-bg); }
.ticket-layout { display: grid; grid-template-columns: minmax(300px, 380px) minmax(0, 1fr); gap: 12px; min-height: 560px; }
.ticket-list, .ticket-detail { min-width: 0; }
.ticket-row { display: flex; width: 100%; justify-content: space-between; gap: 12px; margin-top: 8px; padding: 11px; text-align: left; border: 1px solid var(--color-border); border-radius: var(--radius-md); background: var(--color-bg); color: inherit; }
.ticket-row:hover, .ticket-row.active { background: var(--color-accent-soft); border-color: var(--color-accent); }
.ticket-row-main, .ticket-row-meta { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.ticket-row-main span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ticket-row-meta { align-items: flex-end; }
.ticket-row-meta small, .muted, small { color: var(--color-slate); font-size: var(--text-xs); }
.status-chip { display: inline-flex; align-items: center; border-radius: var(--radius-pill); padding: 2px 8px; font-size: var(--text-xs); white-space: nowrap; background: var(--color-bg-quiet); }
.status-pending { color: var(--color-warning); background: var(--color-warning-bg); }
.status-processing { color: var(--color-accent); background: var(--color-accent-soft); }
.status-resolved { color: var(--color-success); background: var(--color-success-bg); }
.status-closed { color: var(--color-slate); }
.description { white-space: pre-wrap; background: var(--color-bg-quiet); border-radius: var(--radius-md); padding: 10px; }
.detail-grid { grid-template-columns: 160px 1fr auto; align-items: end; margin-top: 18px; }
.detail-grid label, .assignment-row label { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.detail-grid label span, .assignment-row label span { font-size: var(--text-xs); color: var(--color-charcoal); }
.assignment-row { justify-content: flex-start; margin-top: 10px; }
.assignment-row label { flex: 1; }
.timeline { margin-top: 24px; border-top: 1px solid var(--color-border-soft); padding-top: 14px; }
.timeline-item { display: flex; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--color-border-soft); }
.timeline-dot { flex: 0 0 8px; width: 8px; height: 8px; margin-top: 7px; border-radius: 50%; background: var(--color-accent); }
.timeline-item p { margin: 4px 0; }
.empty { color: var(--color-slate); text-align: center; padding: 24px 8px; font-size: var(--text-sm); }
.detail-empty { min-height: 300px; display: grid; place-items: center; }
@media (max-width: 900px) { .ticket-layout { grid-template-columns: 1fr; } .detail-grid { grid-template-columns: 1fr 1fr; } .detail-grid .primary { grid-column: span 2; } .create-ticket { grid-template-columns: 1fr 1fr; } }
@media (max-width: 600px) { .credentials, .filters, .detail-grid, .create-ticket { grid-template-columns: 1fr; } .filters { gap: 8px; } .detail-grid .primary { grid-column: auto; } .assignment-row { align-items: stretch; flex-direction: column; } }
</style>
