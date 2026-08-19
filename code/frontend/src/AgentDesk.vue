<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import {
  type AgentConversation,
  type AgentMessage,
  type QuickReply,
  fetchAgentQueue,
  claimConversation,
  fetchMessages,
  replyConversation,
  closeConversation,
  listQuickReplies,
  createTicket,
  agentHeaders,
} from "./services/api";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// 鉴权：X-API-Key（ADMIN_API_KEY）或 admin token（kb_ 开头），存 localStorage
const apiKey = ref(localStorage.getItem("kb_admin_key") || "");
const adminToken = ref(localStorage.getItem("kb_api_token") || "");

const queue = ref<AgentConversation[]>([]);
const active = ref<AgentConversation | null>(null);
const msgs = ref<AgentMessage[]>([]);
const replyText = ref("");
const quickReplies = ref<QuickReply[]>([]);
const err = ref("");
const info = ref("");

let pollTimer: number | undefined;

function hdrs(): Record<string, string> {
  return agentHeaders(apiKey.value, adminToken.value);
}

function saveKey() {
  localStorage.setItem("kb_admin_key", apiKey.value);
}

async function loadQueue() {
  try {
    const data = await fetchAgentQueue(apiKey.value, adminToken.value);
    queue.value = data.conversations || [];
    // 已激活的会话状态同步
    if (active.value) {
      const fresh = queue.value.find((c) => c.id === active.value!.id);
      if (fresh) active.value = fresh;
    }
  } catch (e) {
    err.value = `加载待接入队列失败: ${(e as Error).message}`;
  }
}

async function loadQuickReplies() {
  try {
    const data = await listQuickReplies(apiKey.value, adminToken.value);
    quickReplies.value = data.quick_replies || [];
  } catch {
    // 快捷回复非关键，失败静默
  }
}

async function openConversation(conv: AgentConversation) {
  try {
    const claimed = await claimConversation(conv.id, apiKey.value, adminToken.value);
    if (!claimed.claimed) {
      info.value = "该会话已被其他坐席领取";
      loadQueue();
      return;
    }
    active.value = { ...conv, status: "human", agent_id: "me" };
    await loadMessages(conv.id);
  } catch (e) {
    err.value = `领取失败: ${(e as Error).message}`;
  }
}

async function loadMessages(convId: number) {
  try {
    const data = await fetchMessages(convId, apiKey.value, adminToken.value);
    msgs.value = data.messages || [];
  } catch (e) {
    err.value = `加载消息失败: ${(e as Error).message}`;
  }
}

async function sendReply() {
  if (!active.value || !replyText.value.trim()) return;
  try {
    await replyConversation(active.value.id, replyText.value.trim(), apiKey.value, adminToken.value);
    msgs.value.push({ role: "agent", content: replyText.value.trim(), created_at: "" });
    replyText.value = "";
    loadMessages(active.value.id);
  } catch (e) {
    err.value = `回复失败: ${(e as Error).message}`;
  }
}

async function endConversation() {
  if (!active.value) return;
  if (!window.confirm("结束本次会话？")) return;
  try {
    await closeConversation(active.value.id, apiKey.value, adminToken.value);
    info.value = "会话已结束";
    active.value = null;
    msgs.value = [];
    loadQueue();
  } catch (e) {
    err.value = `结束失败: ${(e as Error).message}`;
  }
}

async function makeTicket() {
  if (!active.value) return;
  const title = window.prompt("工单标题（默认取最近用户问题）", active.value.transfer_reason || "客服工单");
  if (!title) return;
  try {
    const t = await createTicket(
      { conversation_id: active.value.id, title, description: "转人工会话自动建单" },
      apiKey.value,
      adminToken.value
    );
    info.value = `已建工单 ${t.ticket_no}`;
  } catch (e) {
    err.value = `建单失败: ${(e as Error).message}`;
  }
}

function insertQuickReply(qr: QuickReply) {
  replyText.value = qr.content;
}

onMounted(() => {
  loadQueue();
  loadQuickReplies();
  pollTimer = window.setInterval(loadQueue, 5000); // 5s 轮询待接入池
});
onUnmounted(() => {
  if (pollTimer) window.clearInterval(pollTimer);
});
</script>

<template>
  <div class="agent-desk">
    <!-- 顶部鉴权 -->
    <div class="desk-head">
      <span class="desk-title">💬 客服工作台</span>
      <input v-model="apiKey" placeholder="X-API-Key（管理员）" @change="saveKey" />
      <button class="ghost" @click="loadQueue">刷新</button>
    </div>
    <p v-if="err" class="err">{{ err }}</p>
    <p v-if="info" class="info">{{ info }}</p>

    <div class="desk-body">
      <!-- 左栏：待接入 / 服务中 -->
      <aside class="queue-panel">
        <div class="queue-head">待接入（{{ queue.length }}）</div>
        <div v-if="!queue.length" class="empty">暂无待接入会话</div>
        <div v-for="c in queue" :key="c.id" class="queue-item" :class="{ active: active?.id === c.id }" @click="openConversation(c)">
          <div class="q-visitor">👤 {{ c.visitor_id || "匿名访客" }}</div>
          <div class="q-meta">
            <span class="q-kb">{{ c.kb_id }}</span>
            <span class="q-reason">{{ c.transfer_reason || "用户转人工" }}</span>
          </div>
        </div>
      </aside>

      <!-- 右栏：对话窗 -->
      <section class="chat-panel">
        <template v-if="active">
          <div class="chat-head">
            <span>与 {{ active.visitor_id || "访客" }} 对话</span>
            <button class="ghost" @click="makeTicket">＋ 建工单</button>
            <button class="ghost danger" @click="endConversation">结束会话</button>
          </div>
          <div class="chat-body">
            <div v-for="(m, i) in msgs" :key="i" class="msg" :class="m.role">
              <div class="bubble">{{ m.content }}</div>
            </div>
          </div>
          <div v-if="quickReplies.length" class="quick-replies">
            <span class="qr-label">快捷回复：</span>
            <button v-for="qr in quickReplies" :key="qr.id" class="qr-chip" @click="insertQuickReply(qr)">
              {{ qr.title }}
            </button>
          </div>
          <div class="reply-row">
            <input v-model="replyText" placeholder="输入回复…" @keyup.enter="sendReply" />
            <button class="send" @click="sendReply">发送</button>
          </div>
        </template>
        <div v-else class="empty">← 点击左侧会话开始接待</div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.agent-desk { padding: 16px; }
.desk-head { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
.desk-title { font-weight: var(--weight-semibold); font-size: var(--text-lg); }
.desk-head input { flex: 1; padding: 8px 12px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); font-size: var(--text-sm); }
.ghost { background: var(--color-bg); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: 6px 12px; cursor: pointer; font-size: var(--text-sm); color: var(--color-ink); }
.ghost:hover { background: var(--color-bg-warm); }
.ghost.danger { color: var(--color-error); }
.err { color: var(--color-error); font-size: var(--text-sm); }
.info { color: var(--color-success); font-size: var(--text-sm); }

.desk-body { display: grid; grid-template-columns: 260px 1fr; gap: 16px; min-height: 60vh; }
.queue-panel { border: 1px solid var(--color-border); border-radius: var(--radius-lg); overflow: hidden; background: var(--color-bg); }
.queue-head { padding: 12px 16px; font-weight: var(--weight-semibold); border-bottom: 1px solid var(--color-border); }
.queue-item { padding: 12px 16px; border-bottom: 1px solid var(--color-border-soft); cursor: pointer; transition: background var(--duration-micro) var(--ease); }
.queue-item:hover { background: var(--color-bg-warm); }
.queue-item.active { background: var(--color-accent-soft); }
.q-visitor { font-size: var(--text-sm); font-weight: var(--weight-medium); }
.q-meta { font-size: var(--text-xs); color: var(--color-charcoal); margin-top: 4px; display: flex; gap: 8px; }
.q-kb { background: var(--color-bg-quiet); border-radius: var(--radius-sm); padding: 1px 6px; }
.empty { color: var(--color-slate); padding: 24px; text-align: center; font-size: var(--text-sm); }

.chat-panel { border: 1px solid var(--color-border); border-radius: var(--radius-lg); display: flex; flex-direction: column; background: var(--color-bg); overflow: hidden; }
.chat-head { padding: 12px 16px; border-bottom: 1px solid var(--color-border); display: flex; gap: 8px; align-items: center; }
.chat-head span { flex: 1; font-weight: var(--weight-semibold); }
.chat-body { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 8px; min-height: 300px; background: var(--color-bg-quiet); }
.msg { display: flex; }
.msg.user { justify-content: flex-start; }
.msg.agent { justify-content: flex-end; }
.msg.assistant { justify-content: flex-start; }
.bubble { max-width: 75%; padding: 8px 12px; border-radius: var(--radius-md); font-size: var(--text-sm); line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.msg.user .bubble { background: var(--color-bg); border: 1px solid var(--color-border); }
.msg.agent .bubble { background: var(--color-accent); color: var(--color-bg); }
.msg.assistant .bubble { background: var(--color-bg); border: 1px solid var(--color-border); }
.quick-replies { padding: 8px 16px; border-top: 1px solid var(--color-border-soft); display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.qr-label { font-size: var(--text-xs); color: var(--color-charcoal); }
.qr-chip { background: var(--color-bg-warm); border: 1px solid var(--color-border); border-radius: var(--radius-pill); padding: 4px 10px; font-size: var(--text-xs); cursor: pointer; }
.qr-chip:hover { background: var(--color-accent-soft); }
.reply-row { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--color-border); }
.reply-row input { flex: 1; padding: 8px 12px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); font-size: var(--text-sm); }
.send { background: var(--color-accent); color: var(--color-bg); border: none; border-radius: var(--radius-md); padding: 0 16px; cursor: pointer; }
</style>
