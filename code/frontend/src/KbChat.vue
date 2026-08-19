<script setup lang="ts">
import { ref } from "vue";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// ---------- 状态 ----------
interface Citation {
  index: number; chunk_id: string; text: string;
  metadata: { doc_title?: string; kb_id?: string; doc_id?: string; chunk_index?: number };
}
interface Msg {
  role: "user" | "assistant" | "system";
  content: string;
  score?: number;
  citations?: Citation[];
  rating?: number;        // 满意度：1=👍 / 0=👎，未评 undefined
  system?: boolean;       // 系统状态条（转人工提示）
  conversationId?: number;
}
const messages = ref<Msg[]>([]);
const input = ref("");
const expandedCites = ref<Set<number>>(new Set());
function toggleCites(i: number) {
  const s = new Set(expandedCites.value);
  s.has(i) ? s.delete(i) : s.add(i);
  expandedCites.value = s;
}
const loading = ref(false);

// 多库 + 用户（RBAC）：kb 列表 / 当前库 / API Token，localStorage 持久化
// 🟠4 token 鉴权：值以 "kb_" 开头 → 走 X-Api-Token 头；
// 旧值（明文 user_id）仍按 user_id 参数发送，过渡期兼容
const kbs = ref<string[]>([]);
const currentKb = ref(localStorage.getItem("kb_id") || "default");
const userToken = ref(localStorage.getItem("kb_api_token") || localStorage.getItem("kb_user_id") || "");

function isTokenAuth(): boolean {
  return userToken.value.startsWith("kb_");
}
function authHeaders(): Record<string, string> {
  return isTokenAuth() ? { "X-Api-Token": userToken.value } : {};
}

// 会话线程 ID：按库隔离（切换库换新 thread，避免跨库串历史）
const threadId = ref(localStorage.getItem(`kb_thread_${currentKb.value}`) || `kb-${Date.now()}`);

interface Doc { doc_id: string; title: string; chunks: number; kb_id?: string; source_type?: string }
const docs = ref<Doc[]>([]);
const uploadMsg = ref("");
const uploadErr = ref("");

// 更新文档：记录待更新的 doc_id，复用 PUT 接口（保持 doc_id 稳定）
const updatingDocId = ref("");

// ---------- 知识库 ----------
async function loadKbs() {
  try {
    const q = !isTokenAuth() && userToken.value ? `?user_id=${encodeURIComponent(userToken.value)}` : "";
    const resp = await fetch(`${baseURL}/kb/kbs${q}`, { headers: authHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    kbs.value = data.kbs || [];
    if (kbs.value.length && !kbs.value.includes(currentKb.value)) {
      currentKb.value = kbs.value[0];
    }
  } catch (e) {
    uploadErr.value = `加载知识库列表失败: ${(e as Error).message}`;
  }
}

// ---------- 文档管理 ----------
async function loadDocs() {
  try {
    const q = new URLSearchParams();
    q.set("kb_id", currentKb.value);
    if (!isTokenAuth() && userToken.value) q.set("user_id", userToken.value);
    const resp = await fetch(`${baseURL}/kb/docs?${q.toString()}`, { headers: authHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    docs.value = data.docs || [];
  } catch (e) {
    uploadErr.value = `加载文档列表失败: ${(e as Error).message}`;
  }
}

async function onUpload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  // P2-3：前端预检文件大小（后端 413 之前先拦一道，避免上传了才发现超限）
  const maxMB = 50;
  if (file.size > maxMB * 1024 * 1024) {
    uploadErr.value = `文件超过 ${maxMB}MB 限制（当前 ${(file.size / 1024 / 1024).toFixed(1)}MB）`;
    uploadMsg.value = "";
    (event.target as HTMLInputElement).value = "";
    return;
  }
  uploadMsg.value = `正在上传 ${file.name}...`;
  uploadErr.value = "";
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("kb_id", currentKb.value);
    if (!isTokenAuth() && userToken.value) form.append("user_id", userToken.value);
    const resp = await fetch(`${baseURL}/kb/ingest`, { method: "POST", body: form, headers: authHeaders() });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    uploadMsg.value = `入库成功：${data.title}，共 ${data.chunks} 个分块`;
    (event.target as HTMLInputElement).value = "";
    loadDocs();
  } catch (e) {
    uploadErr.value = `入库失败: ${(e as Error).message}`;
    uploadMsg.value = "";
  }
}

async function onDelete(docId: string) {
  // P2-3：删除二次确认，防误删不可恢复的数据
  if (!window.confirm(`确定删除文档 ${docId}？此操作不可恢复。`)) return;
  try {
    const q = !isTokenAuth() && userToken.value ? `?user_id=${encodeURIComponent(userToken.value)}` : "";
    const resp = await fetch(`${baseURL}/kb/docs/${docId}${q}`, { method: "DELETE", headers: authHeaders() });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    loadDocs();
  } catch (e) {
    uploadErr.value = `删除失败: ${(e as Error).message}`;
  }
}

// P2-3：登出——清除本地 token，返回未登录态
function logout() {
  userToken.value = "";
  localStorage.removeItem("kb_api_token");
  localStorage.removeItem("kb_user_id");
  messages.value = [];
  loadKbs();
  loadDocs();
}

// 更新文档：选中文件后走 PUT（先删旧分块，再用原 doc_id 重新入库）
function pickUpdate(docId: string) {
  updatingDocId.value = docId;
  updateInput.value?.click();
}
const updateInput = ref<HTMLInputElement | null>(null);
async function onUpdate(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  const docId = updatingDocId.value;
  if (!file || !docId) return;
  uploadMsg.value = `正在更新 ${file.name}...`;
  uploadErr.value = "";
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("kb_id", currentKb.value);
    if (!isTokenAuth() && userToken.value) form.append("user_id", userToken.value);
    const resp = await fetch(`${baseURL}/kb/docs/${docId}`, { method: "PUT", body: form, headers: authHeaders() });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    uploadMsg.value = `更新成功：${data.title}，共 ${data.chunks} 个分块`;
    (event.target as HTMLInputElement).value = "";
    updatingDocId.value = "";
    loadDocs();
  } catch (e) {
    uploadErr.value = `更新失败: ${(e as Error).message}`;
    uploadMsg.value = "";
  }
}

// 切换知识库 / 用户：重新加载库与文档，并重置对话（换 thread）
function onSwitchKb() {
  localStorage.setItem("kb_id", currentKb.value);
  localStorage.setItem("kb_api_token", userToken.value);
  threadId.value = `kb-${Date.now()}`;
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);
  messages.value = [];
  loadKbs();
  loadDocs();
}

// 多会话管理（第14项）：开启新会话——换 threadId + 清空对话
function newSession() {
  threadId.value = `kb-${Date.now()}`;
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);
  messages.value = [];
}

// ---------- 问答 ----------
async function onSend() {
  const question = input.value.trim();
  if (!question || loading.value) return;

  messages.value.push({ role: "user", content: question });
  input.value = "";
  loading.value = true;
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);

  // 传最近 6 轮历史（供 LangGraph 多轮改写）
  const history = messages.value
    .slice(-6, -1)
    .map((m) => ({ role: m.role, content: m.content }));

  try {
    const resp = await fetch(`${baseURL}/kb/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({
        question,
        history,
        kb_id: currentKb.value,
        thread_id: threadId.value,
        // token 鉴权走头；明文 user_id 仅在旧值（非 kb_ 开头）时兼容发送
        user_id: isTokenAuth() ? undefined : userToken.value || undefined,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    messages.value.push({
      role: "assistant",
      content: data.answer || "（无回答）",
      score: data.score,
      citations: data.citations || [],
    });
    // 2.2 转人工：escalate 时插入系统状态条，并轮询坐席接入状态
    if (data.escalate && data.conversation_id) {
      messages.value.push({
        role: "system",
        system: true,
        content: "已为您转接人工客服，等待接入…",
        conversationId: data.conversation_id,
      });
      pollAgentStatus(data.conversation_id);
    }
  } catch (e) {
    messages.value.push({ role: "assistant", content: `❌ 问答失败: ${(e as Error).message}` });
  } finally {
    loading.value = false;
  }
}

// ---------- 满意度（2.1） ----------
async function sendFeedback(i: number, rating: number) {
  const m = messages.value[i];
  if (!m || m.role !== "assistant" || m.rating !== undefined) return;
  // 找最近的用户问题（作为 feedback 的 question 字段）
  let question = "";
  for (let j = i - 1; j >= 0; j--) {
    if (messages.value[j].role === "user") { question = messages.value[j].content; break; }
  }
  let comment = "";
  if (rating === 0) {
    comment = window.prompt("哪里回答得不好？（可选，帮助改进）") || "";
  }
  try {
    const resp = await fetch(`${baseURL}/kb/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kb_id: currentKb.value,
        question,
        answer: m.content,
        rating,
        comment,
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    m.rating = rating;
  } catch (e) {
    uploadErr.value = `反馈提交失败: ${(e as Error).message}`;
  }
}

// ---------- 转人工轮询（2.2） ----------
async function pollAgentStatus(convId: number) {
  // 每 5s 轮询一次会话状态，waiting -> human 时更新状态条，最多 24 次（2 分钟）
  for (let n = 0; n < 24; n++) {
    await new Promise((r) => setTimeout(r, 5000));
    try {
      const resp = await fetch(`${baseURL}/kb/conversation/${convId}/status`);
      if (!resp.ok) continue;
      const data = await resp.json();
      if (data.status === "human") {
        const sys = messages.value.find((m) => m.system && m.conversationId === convId);
        if (sys) sys.content = "坐席已接入，请继续对话。";
        break;
      }
      if (data.status === "closed") {
        const sys = messages.value.find((m) => m.system && m.conversationId === convId);
        if (sys) sys.content = "本次服务已结束，感谢咨询。";
        break;
      }
    } catch (e) {
      // 轮询失败静默继续
    }
  }
}

// ---------- 管理页（P2：用户/角色/审计界面化） ----------
const showAdmin = ref(false);
const adminKey = ref(localStorage.getItem("kb_admin_key") || "");
const users = ref<{ user_id: string; name: string; role: string; allowed_kbs: string[]; created_at?: string }[]>([]);
const auditLogs = ref<{ ts: string; user_id: string; action: string; target: string; detail: string }[]>([]);
const newUserName = ref("");
const newUserRole = ref("member");
const adminMsg = ref("");
const adminErr = ref("");

// 管理接口鉴权：X-API-Key（ADMIN_API_KEY）或已登录的 admin token，二选一
function adminHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  if (adminKey.value) h["X-API-Key"] = adminKey.value;
  if (isTokenAuth()) h["X-Api-Token"] = userToken.value;
  return h;
}

async function loadUsers() {
  try {
    const resp = await fetch(`${baseURL}/kb/users`, { headers: adminHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    users.value = data.users || [];
  } catch (e) {
    adminErr.value = `加载用户失败: ${(e as Error).message}`;
  }
}

async function createUser() {
  if (!newUserName.value) return;
  adminErr.value = ""; adminMsg.value = "";
  try {
    const form = new FormData();
    form.append("name", newUserName.value);
    form.append("role", newUserRole.value);
    const resp = await fetch(`${baseURL}/kb/users`, { method: "POST", body: form, headers: adminHeaders() });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    adminMsg.value = `已创建用户「${data.name}」，API Token（仅显示一次，请复制）: ${data.api_token}`;
    newUserName.value = "";
    loadUsers();
  } catch (e) {
    adminErr.value = `建用户失败: ${(e as Error).message}`;
  }
}

async function setRole(userId: string, role: string) {
  adminErr.value = ""; adminMsg.value = "";
  try {
    const form = new FormData();
    form.append("role", role);
    const resp = await fetch(`${baseURL}/kb/users/${userId}/role`, { method: "POST", body: form, headers: adminHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    adminMsg.value = `已更新角色为 ${role}`;
    loadUsers();
  } catch (e) {
    adminErr.value = `设置角色失败: ${(e as Error).message}`;
  }
}

async function loadAudit() {
  adminErr.value = "";
  try {
    const resp = await fetch(`${baseURL}/admin/audit?limit=100`, { headers: adminHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    auditLogs.value = data.entries || [];
  } catch (e) {
    adminErr.value = `加载审计日志失败: ${(e as Error).message}`;
  }
}

function openAdmin() {
  showAdmin.value = true;
  adminKey.value = localStorage.getItem("kb_admin_key") || adminKey.value;
  loadUsers();
  loadAudit();
}
function closeAdmin() {
  showAdmin.value = false;
  localStorage.setItem("kb_admin_key", adminKey.value);
}

loadKbs();
loadDocs();
</script>

<template>
  <div class="kb-page">
    <header class="kb-header">
      <h1>📚 企业知识库管理</h1>
      <p class="sub">上传文档 → 向量化入库 → LangGraph 智能问答（带引用）</p>
      <button class="admin-toggle" @click="showAdmin ? closeAdmin() : openAdmin()">
        {{ showAdmin ? '返回问答' : '⚙️ 管理' }}
      </button>
    </header>

    <!-- 管理页（P2：用户/角色/审计，需 admin 鉴权） -->
    <div v-if="showAdmin" class="admin-page">
      <section class="upload-card">
        <h3>🔐 管理员鉴权</h3>
        <label class="field">
          <span>Admin API Key（X-API-Key，配了 ADMIN_API_KEY 时用；或用已登录的 admin token）</span>
          <input v-model="adminKey" placeholder="留空则用上方 API Token 的 admin 身份" />
        </label>
      </section>

      <section class="upload-card">
        <h3>👤 新建用户</h3>
        <div class="admin-row">
          <input v-model="newUserName" placeholder="用户名" />
          <select v-model="newUserRole">
            <option value="member">member（读写）</option>
            <option value="readonly">readonly（只读）</option>
            <option value="admin">admin（全通）</option>
          </select>
          <button class="file-btn" @click="createUser">创建</button>
        </div>
        <p v-if="adminMsg" class="admin-msg">{{ adminMsg }}</p>
        <p v-if="adminErr" class="admin-err">{{ adminErr }}</p>
      </section>

      <section class="upload-card">
        <h3>👥 用户列表</h3>
        <div v-if="users.length" class="admin-user-list">
          <div v-for="u in users" :key="u.user_id" class="admin-user">
            <span class="admin-name">{{ u.name }}</span>
            <span class="doc-kb">{{ u.role }}</span>
            <span class="doc-chunks">可访问: {{ (u.allowed_kbs || []).join(', ') || '（无）' }}</span>
            <select :value="u.role" @change="setRole(u.user_id, ($event.target as HTMLSelectElement).value)">
              <option value="member">member</option>
              <option value="readonly">readonly</option>
              <option value="admin">admin</option>
            </select>
          </div>
        </div>
        <p v-else class="empty">暂无用户（可先在上方创建，或设 KB_BOOTSTRAP_ADMIN_TOKEN 引导）</p>
      </section>

      <section class="upload-card">
        <h3>📋 审计日志（最近 100 条）</h3>
        <button class="upd-btn" @click="loadAudit">刷新</button>
        <div v-if="auditLogs.length" class="admin-audit">
          <div v-for="(a, i) in auditLogs" :key="i" class="admin-audit-item">
            <span class="doc-chunks">{{ (a.ts || '').slice(0, 19) }}</span>
            <span class="doc-kb">{{ a.action }}</span>
            <span class="admin-name">{{ a.user_id }}</span>
            <span class="doc-chunks">→ {{ a.target }}</span>
          </div>
        </div>
        <p v-else class="empty">暂无审计记录</p>
      </section>
    </div>

    <!-- 知识库 + 用户选择（RBAC） -->
    <div v-if="!showAdmin">
    <section class="kb-selector">
      <label class="field">
        <span>知识库</span>
        <select v-model="currentKb" @change="onSwitchKb">
          <option v-for="k in kbs" :key="k" :value="k">{{ k }}</option>
          <option v-if="!kbs.length" value="default">default</option>
        </select>
      </label>
      <label class="field">
        <span>API Token（问答可留空；上传/删除/更新需填。创建用户时下发，kb_ 开头）</span>
        <input v-model="userToken" placeholder="如 kb_3af2...（写操作需此身份且有库权限）" @change="onSwitchKb" />
      </label>
      <button v-if="userToken" class="logout-btn" @click="logout">登出</button>
    </section>

    <!-- 文档管理区 -->
    <section class="upload-card">
      <div class="upload-row">
        <label class="file-btn">
          📄 上传文档
          <input type="file" accept=".md,.txt,.pdf,.docx" hidden @change="onUpload" />
        </label>
        <input ref="updateInput" type="file" accept=".md,.txt,.pdf,.docx" hidden @change="onUpdate" />
        <span class="upload-msg" :class="{ err: uploadErr }">{{ uploadErr || uploadMsg }}</span>
      </div>
      <div v-if="docs.length" class="doc-list">
        <div v-for="d in docs" :key="d.doc_id" class="doc-item">
          <span class="doc-title">📁 {{ d.title }}</span>
          <span class="doc-kb">{{ d.kb_id }}</span>
          <span class="doc-chunks">{{ d.chunks }} 分块</span>
          <button class="upd-btn" @click="pickUpdate(d.doc_id)">更新</button>
          <button class="del-btn" @click="onDelete(d.doc_id)">删除</button>
        </div>
      </div>
      <p v-else class="empty">暂无文档，先上传一份 .md / .pdf 试试</p>
    </section>

    <!-- 对话区 -->
    <section class="chat-card">
      <div class="chat-body">
        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
          <div v-if="m.system" class="system-bar">{{ m.content }}</div>
          <template v-else>
            <div class="bubble">{{ m.content }}</div>
            <div v-if="m.score !== undefined && m.role === 'assistant'" class="score-badge">
              忠实度 {{ m.score }}/10
            </div>
            <!-- 2.1 满意度评价 -->
            <div v-if="m.role === 'assistant' && !m.system" class="feedback-bar">
              <button
                v-if="m.rating === undefined"
                class="fb-btn"
                title="有帮助"
                @click="sendFeedback(i, 1)"
              >👍</button>
              <button
                v-if="m.rating === undefined"
                class="fb-btn"
                title="没帮助"
                @click="sendFeedback(i, 0)"
              >👎</button>
              <span v-if="m.rating === 1" class="fb-thanks">已收到评价 👍</span>
              <span v-if="m.rating === 0" class="fb-thanks">已收到评价，我们会改进 👎</span>
            </div>
            <div v-if="m.citations && m.citations.length" class="cite-area">
              <button class="cite-toggle" @click="toggleCites(i)">
                📎 引用 {{ m.citations.length }} 条来源 · {{ expandedCites.has(i) ? '收起' : '展开' }}
              </button>
              <div v-if="expandedCites.has(i)" class="cite-list">
                <div v-for="c in m.citations" :key="c.index" class="cite-card">
                  <div class="cite-head">
                    [{{ c.index }}] 📄 {{ c.metadata.doc_title || '未命名' }}
                    <span class="cite-kb">· kb={{ c.metadata.kb_id }}</span>
                  </div>
                  <div class="cite-text">{{ c.text }}</div>
                </div>
              </div>
            </div>
          </template>
        </div>
        <div v-if="loading" class="msg assistant">
          <div class="bubble typing">思考中<span class="dot">...</span></div>
        </div>
      </div>
      <div class="chat-input">
        <button class="new-session-btn" title="开启新会话" @click="newSession">＋</button>
        <input
          v-model="input"
          placeholder="问知识库：例如「采购超过多少要招投标？」"
          @keyup.enter="onSend"
        />
        <button :disabled="loading" @click="onSend">发送</button>
      </div>
    </section>
    </div>
  </div>
</template>

<style scoped>
.kb-page {
  max-width: 860px;
  margin: 0 auto;
  padding: 24px 16px;
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}
.kb-header h1 { font-size: 24px; margin: 0 0 4px; }
.sub { color: var(--color-slate); font-size: 13px; margin: 0 0 16px; }
.kb-selector {
  display: flex; gap: 16px; align-items: flex-end; margin-bottom: 16px;
  background: var(--color-bg); border: 1px solid var(--color-border); border-radius: 12px; padding: 12px 16px;
}
.kb-selector .field { display: flex; flex-direction: column; gap: 4px; flex: 1; }
.kb-selector .field span { font-size: 12px; color: var(--color-charcoal); }
.kb-selector select, .kb-selector input {
  padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: 8px; font-size: 14px; outline: none;
}
.logout-btn {
  padding: 8px 14px; border: 1px solid var(--color-error); color: var(--color-error); background: var(--color-bg);
  border-radius: 8px; cursor: pointer; font-size: 13px; white-space: nowrap;
}
.logout-btn:hover { background: var(--color-error-bg); }
.admin-toggle {
  padding: 6px 14px; border: 1px solid var(--color-accent); color: var(--color-accent); background: var(--color-bg);
  border-radius: 8px; cursor: pointer; font-size: 13px;
}
.admin-toggle:hover { background: var(--color-accent-soft); }
.admin-page h3 { margin: 0 0 10px; font-size: 15px; }
.admin-page .field { display: flex; flex-direction: column; gap: 4px; }
.admin-page .field span { font-size: 12px; color: var(--color-charcoal); }
.admin-page .field input {
  padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: 8px; font-size: 14px; outline: none;
}
.admin-row { display: flex; gap: 8px; }
.admin-row input, .admin-row select {
  padding: 8px 10px; border: 1px solid var(--color-border-strong); border-radius: 8px; font-size: 14px; outline: none;
}
.admin-row input { flex: 1; }
.admin-msg { font-size: 12px; color: var(--color-success); margin: 8px 0 0; word-break: break-all; }
.admin-err { font-size: 12px; color: var(--color-error); margin: 8px 0 0; }
.admin-user-list { display: flex; flex-direction: column; gap: 6px; }
.admin-user {
  display: flex; align-items: center; gap: 8px; padding: 6px 10px;
  background: var(--color-bg-quiet); border-radius: 6px; font-size: 13px;
}
.admin-user select { padding: 3px 6px; border: 1px solid var(--color-border-strong); border-radius: 6px; font-size: 12px; }
.admin-name { font-weight: 600; }
.admin-audit { display: flex; flex-direction: column; gap: 4px; margin-top: 8px; }
.admin-audit-item {
  display: flex; align-items: center; gap: 8px; padding: 4px 8px;
  background: var(--color-bg-quiet); border-radius: 6px; font-size: 12px;
}
.upload-card, .chat-card {
  background: var(--color-bg); border: 1px solid var(--color-border); border-radius: 12px;
  padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.upload-row { display: flex; align-items: center; gap: 12px; }
.file-btn {
  background: var(--color-accent); color: var(--color-bg); padding: 8px 16px; border-radius: 8px;
  cursor: pointer; font-size: 14px; border: none;
}
.upload-msg { font-size: 13px; color: var(--color-success); }
.upload-msg.err { color: var(--color-error); }
.doc-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
.doc-item {
  display: flex; align-items: center; gap: 8px; padding: 6px 10px;
  background: var(--color-bg-quiet); border-radius: 6px; font-size: 13px;
}
.doc-title { flex: 1; }
.doc-kb {
  background: var(--color-accent-soft); color: var(--color-accent); border-radius: 4px; padding: 1px 6px; font-size: 11px;
}
.doc-chunks { color: var(--color-slate); font-size: 12px; }
.upd-btn {
  background: none; border: 1px solid var(--color-accent); color: var(--color-accent);
  border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 12px;
}
.del-btn {
  background: none; border: 1px solid var(--color-error); color: var(--color-error);
  border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 12px;
}
.empty { color: var(--color-muted); font-size: 13px; text-align: center; padding: 12px; }
.chat-body {
  min-height: 220px; max-height: 420px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 10px; padding: 8px 4px;
}
.msg { display: flex; }
.msg.user { justify-content: flex-end; }
.msg.assistant { justify-content: flex-start; }
.msg.system { justify-content: center; }
.system-bar {
  max-width: 90%; padding: 6px 14px; border-radius: 999px;
  background: var(--color-accent-soft); color: var(--color-accent);
  font-size: 13px; text-align: center;
}
.feedback-bar { display: flex; gap: 6px; margin-top: 4px; align-items: center; }
.fb-btn {
  background: transparent; border: 1px solid var(--color-border);
  border-radius: 6px; padding: 2px 8px; cursor: pointer; font-size: 14px;
  transition: background var(--duration-micro) var(--ease);
}
.fb-btn:hover { background: var(--color-bg-warm); }
.fb-thanks { font-size: 12px; color: var(--color-charcoal); }
.bubble {
  max-width: 80%; padding: 10px 14px; border-radius: 12px;
  font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
}
.msg.user .bubble { background: var(--color-accent); color: var(--color-bg); }
.msg.assistant .bubble { background: var(--color-bg-quiet); color: var(--color-ink); }
.score-badge {
  display: inline-block; margin-top: 4px; padding: 2px 8px;
  background: var(--color-success-bg); color: var(--color-success); border: 1px solid rgba(15, 123, 108, 0.25);
  border-radius: 999px; font-size: 12px;
}
.cite-area { margin-top: 6px; max-width: 80%; }
.cite-toggle {
  background: none; border: 1px solid var(--color-border-strong); color: var(--color-charcoal);
  border-radius: 6px; padding: 3px 10px; cursor: pointer; font-size: 12px;
}
.cite-toggle:hover { background: var(--color-bg-quiet); }
.cite-list { margin-top: 6px; display: flex; flex-direction: column; gap: 6px; }
.cite-card {
  background: var(--color-bg-quiet); border-left: 3px solid var(--color-accent); border-radius: 4px;
  padding: 6px 10px; font-size: 12px;
}
.cite-head { font-weight: 500; color: var(--color-ink-soft); margin-bottom: 2px; }
.cite-kb { color: var(--color-slate); font-weight: 400; }
.cite-text { color: var(--color-charcoal); line-height: 1.5; }
.typing .dot { animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: 0; } }
.chat-input { display: flex; gap: 8px; margin-top: 10px; }
.chat-input input {
  flex: 1; padding: 10px 14px; border: 1px solid var(--color-border-strong); border-radius: 8px;
  font-size: 14px; outline: none;
}
.chat-input button {
  background: var(--color-accent); color: var(--color-bg); border: none; border-radius: 8px;
  padding: 0 20px; cursor: pointer;
}
.chat-input button:disabled { opacity: .5; cursor: not-allowed; }
.chat-input .new-session-btn {
  background: var(--color-bg); color: var(--color-accent); border: 1px solid var(--color-accent); border-radius: 8px;
  padding: 0 14px; font-size: 18px; cursor: pointer;
}
.chat-input .new-session-btn:hover { background: var(--color-accent-soft); }
</style>
