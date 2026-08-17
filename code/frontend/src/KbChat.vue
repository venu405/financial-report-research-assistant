<script setup lang="ts">
import { ref } from "vue";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// ---------- 状态 ----------
interface Citation {
  index: number; chunk_id: string; text: string;
  metadata: { doc_title?: string; kb_id?: string; doc_id?: string; chunk_index?: number };
}
const messages = ref<{ role: "user" | "assistant"; content: string; score?: number; citations?: Citation[] }[]>([]);
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
  } catch (e) {
    messages.value.push({ role: "assistant", content: `❌ 问答失败: ${(e as Error).message}` });
  } finally {
    loading.value = false;
  }
}

loadKbs();
loadDocs();
</script>

<template>
  <div class="kb-page">
    <header class="kb-header">
      <h1>📚 企业知识库管理</h1>
      <p class="sub">上传文档 → 向量化入库 → LangGraph 智能问答（带引用）</p>
    </header>

    <!-- 知识库 + 用户选择（RBAC） -->
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
          <div class="bubble">{{ m.content }}</div>
          <div v-if="m.score !== undefined && m.role === 'assistant'" class="score-badge">
            忠实度 {{ m.score }}/10
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
        </div>
        <div v-if="loading" class="msg assistant">
          <div class="bubble typing">思考中<span class="dot">...</span></div>
        </div>
      </div>
      <div class="chat-input">
        <input
          v-model="input"
          placeholder="问知识库：例如「采购超过多少要招投标？」"
          @keyup.enter="onSend"
        />
        <button :disabled="loading" @click="onSend">发送</button>
      </div>
    </section>
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
.sub { color: #888; font-size: 13px; margin: 0 0 16px; }
.kb-selector {
  display: flex; gap: 16px; align-items: flex-end; margin-bottom: 16px;
  background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 16px;
}
.kb-selector .field { display: flex; flex-direction: column; gap: 4px; flex: 1; }
.kb-selector .field span { font-size: 12px; color: #6b7280; }
.kb-selector select, .kb-selector input {
  padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 14px; outline: none;
}
.logout-btn {
  padding: 8px 14px; border: 1px solid #dc2626; color: #dc2626; background: #fff;
  border-radius: 8px; cursor: pointer; font-size: 13px; white-space: nowrap;
}
.logout-btn:hover { background: #fef2f2; }
.upload-card, .chat-card {
  background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
  padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.upload-row { display: flex; align-items: center; gap: 12px; }
.file-btn {
  background: #2563eb; color: #fff; padding: 8px 16px; border-radius: 8px;
  cursor: pointer; font-size: 14px; border: none;
}
.upload-msg { font-size: 13px; color: #16a34a; }
.upload-msg.err { color: #dc2626; }
.doc-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
.doc-item {
  display: flex; align-items: center; gap: 8px; padding: 6px 10px;
  background: #f9fafb; border-radius: 6px; font-size: 13px;
}
.doc-title { flex: 1; }
.doc-kb {
  background: #eef2ff; color: #4f46e5; border-radius: 4px; padding: 1px 6px; font-size: 11px;
}
.doc-chunks { color: #888; font-size: 12px; }
.upd-btn {
  background: none; border: 1px solid #2563eb; color: #2563eb;
  border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 12px;
}
.del-btn {
  background: none; border: 1px solid #dc2626; color: #dc2626;
  border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 12px;
}
.empty { color: #aaa; font-size: 13px; text-align: center; padding: 12px; }
.chat-body {
  min-height: 220px; max-height: 420px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 10px; padding: 8px 4px;
}
.msg { display: flex; }
.msg.user { justify-content: flex-end; }
.msg.assistant { justify-content: flex-start; }
.bubble {
  max-width: 80%; padding: 10px 14px; border-radius: 12px;
  font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
}
.msg.user .bubble { background: #2563eb; color: #fff; }
.msg.assistant .bubble { background: #f3f4f6; color: #111827; }
.score-badge {
  display: inline-block; margin-top: 4px; padding: 2px 8px;
  background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0;
  border-radius: 999px; font-size: 12px;
}
.cite-area { margin-top: 6px; max-width: 80%; }
.cite-toggle {
  background: none; border: 1px solid #d1d5db; color: #4b5563;
  border-radius: 6px; padding: 3px 10px; cursor: pointer; font-size: 12px;
}
.cite-toggle:hover { background: #f3f4f6; }
.cite-list { margin-top: 6px; display: flex; flex-direction: column; gap: 6px; }
.cite-card {
  background: #f9fafb; border-left: 3px solid #2563eb; border-radius: 4px;
  padding: 6px 10px; font-size: 12px;
}
.cite-head { font-weight: 500; color: #1f2937; margin-bottom: 2px; }
.cite-kb { color: #9ca3af; font-weight: 400; }
.cite-text { color: #4b5563; line-height: 1.5; }
.typing .dot { animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: 0; } }
.chat-input { display: flex; gap: 8px; margin-top: 10px; }
.chat-input input {
  flex: 1; padding: 10px 14px; border: 1px solid #d1d5db; border-radius: 8px;
  font-size: 14px; outline: none;
}
.chat-input button {
  background: #2563eb; color: #fff; border: none; border-radius: 8px;
  padding: 0 20px; cursor: pointer;
}
.chat-input button:disabled { opacity: .5; cursor: not-allowed; }
</style>
