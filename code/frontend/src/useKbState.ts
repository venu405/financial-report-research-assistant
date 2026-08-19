// 知识库前端共享状态（模块级单例，供 Sidebar/ChatView/DocsView/AdminView 共享）
import { ref } from "vue";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface Citation {
  index: number; chunk_id: string; text: string;
  metadata: { doc_title?: string; kb_id?: string; doc_id?: string; chunk_index?: number };
}
export interface Msg {
  role: "user" | "assistant" | "system";
  content: string;
  score?: number;
  citations?: Citation[];
  rating?: number;
  system?: boolean;
  conversationId?: number;
}
export interface Doc {
  doc_id: string; title: string; chunks: number; kb_id?: string; source_type?: string;
}

// ---------- 状态（模块级，跨组件共享） ----------
export const messages = ref<Msg[]>([]);
// 流式问答：当前正在生成的答案（打字机增量，ChatView 渲染最后一条 assistant 用）
export const currentStreamMessage = ref("");
export function setCurrentStreamMessage(text: string) {
  currentStreamMessage.value = text;
}
export const input = ref("");
export const expandedCites = ref<Set<number>>(new Set());
export const loading = ref(false);

export const kbs = ref<string[]>([]);
export const currentKb = ref(localStorage.getItem("kb_id") || "default");
export const userToken = ref(localStorage.getItem("kb_api_token") || localStorage.getItem("kb_user_id") || "");

export const docs = ref<Doc[]>([]);
export const uploadMsg = ref("");
export const uploadErr = ref("");
export const updatingDocId = ref("");
export const updateInput = ref<HTMLInputElement | null>(null);

export const threadId = ref(localStorage.getItem(`kb_thread_${currentKb.value}`) || `kb-${Date.now()}`);

// ---------- 多会话（P2-1：列表/切换/删除，接 /kb/threads） ----------
export const threads = ref<{ thread_id: string; checkpoints: number }[]>([]);

export async function loadThreads() {
  try {
    const resp = await fetch(`${baseURL}/kb/threads`, { headers: adminHeaders() });
    if (!resp.ok) return;
    const data = await resp.json();
    threads.value = data.threads || [];
  } catch {
    // 非管理员或后端不可用时静默
  }
}

export function switchThread(tid: string) {
  threadId.value = tid;
  localStorage.setItem(`kb_thread_${currentKb.value}`, tid);
  messages.value = [];
}

export async function deleteThread(tid: string) {
  if (!window.confirm(`删除会话 ${tid.slice(0, 20)}… 的历史记录？`)) return;
  try {
    const resp = await fetch(`${baseURL}/kb/threads/${encodeURIComponent(tid)}`, {
      method: "DELETE",
      headers: adminHeaders(),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    if (threadId.value === tid) {
      threadId.value = `kb-${Date.now()}`;
      messages.value = [];
    }
    loadThreads();
  } catch (e) {
    adminErr.value = `删除会话失败: ${(e as Error).message}`;
  }
}

// ---------- 管理页 ----------
export const showAdmin = ref(false);
export const adminKey = ref(localStorage.getItem("kb_admin_key") || "");
export const users = ref<{ user_id: string; name: string; role: string; allowed_kbs: string[]; created_at?: string }[]>([]);
export const auditLogs = ref<{ ts: string; user_id: string; action: string; target: string; detail: string }[]>([]);
export const newUserName = ref("");
export const newUserRole = ref("member");
export const adminMsg = ref("");
export const adminErr = ref("");

// ---------- 鉴权 ----------
export function isTokenAuth(): boolean {
  return userToken.value.startsWith("kb_");
}
export function authHeaders(): Record<string, string> {
  return isTokenAuth() ? { "X-Api-Token": userToken.value } : {};
}
function adminHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  if (adminKey.value) h["X-API-Key"] = adminKey.value;
  if (isTokenAuth()) h["X-Api-Token"] = userToken.value;
  return h;
}

// ---------- 知识库 / 文档 ----------
export async function loadKbs() {
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

export async function loadDocs() {
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

export async function onUpload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
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

export async function onDelete(docId: string) {
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

export function pickUpdate(docId: string) {
  updatingDocId.value = docId;
  updateInput.value?.click();
}

export async function onUpdate(event: Event) {
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

// ---------- 会话 / 问答 ----------
export function onSwitchKb() {
  localStorage.setItem("kb_id", currentKb.value);
  localStorage.setItem("kb_api_token", userToken.value);
  threadId.value = `kb-${Date.now()}`;
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);
  messages.value = [];
  loadKbs();
  loadDocs();
}

export function newSession() {
  threadId.value = `kb-${Date.now()}`;
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);
  messages.value = [];
}

export function logout() {
  userToken.value = "";
  localStorage.removeItem("kb_api_token");
  localStorage.removeItem("kb_user_id");
  messages.value = [];
  loadKbs();
  loadDocs();
}

export function toggleCites(i: number) {
  const s = new Set(expandedCites.value);
  s.has(i) ? s.delete(i) : s.add(i);
  expandedCites.value = s;
}

export async function onSend() {
  const question = input.value.trim();
  if (!question || loading.value) return;

  messages.value.push({ role: "user", content: question });
  input.value = "";
  loading.value = true;
  currentStreamMessage.value = "";
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);

  const history = messages.value
    .slice(-6, -1)
    .map((m) => ({ role: m.role, content: m.content }));

  try {
    // 统一走流式（聊天框交接文档坑1：主聊天也应有打字机体验）
    const resp = await fetch(`${baseURL}/kb/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({
        question,
        history,
        kb_id: currentKb.value,
        thread_id: threadId.value,
        user_id: isTokenAuth() ? undefined : userToken.value || undefined,
      }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    const reader = resp.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let final: any = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop()!;
      for (const chunk of parts) {
        const line = chunk.trim();
        if (!line.startsWith("data:")) continue;
        try {
          const ev = JSON.parse(line.slice(5));
          if (ev.type === "token") currentStreamMessage.value += ev.text;
          else if (ev.type === "final") final = ev;
          else if (ev.type === "error") throw new Error(ev.detail || "流式错误");
        } catch (e) {
          // 非 error 解析异常忽略
        }
      }
    }
    if (final) {
      messages.value.push({
        role: "assistant",
        content: final.answer || currentStreamMessage.value || "（无回答）",
        score: final.score,
        citations: final.citations || [],
      });
      if (final.escalate && final.conversation_id) {
        messages.value.push({
          role: "system",
          system: true,
          content: "已为您转接人工客服，等待接入…",
          conversationId: final.conversation_id,
        });
        pollAgentStatus(final.conversation_id);
      }
    } else {
      messages.value.push({ role: "assistant", content: currentStreamMessage.value || "（无回答）" });
    }
  } catch (e) {
    messages.value.push({ role: "assistant", content: `❌ 问答失败: ${(e as Error).message}` });
  } finally {
    loading.value = false;
    currentStreamMessage.value = "";
  }
}

export async function sendFeedback(i: number, rating: number) {
  const m = messages.value[i];
  if (!m || m.role !== "assistant" || m.rating !== undefined) return;
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
      body: JSON.stringify({ kb_id: currentKb.value, question, answer: m.content, rating, comment }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    m.rating = rating;
  } catch (e) {
    uploadErr.value = `反馈提交失败: ${(e as Error).message}`;
  }
}

export async function pollAgentStatus(convId: number) {
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
    } catch {
      // 轮询失败静默继续
    }
  }
}

// ---------- 管理 ----------
export async function loadUsers() {
  try {
    const resp = await fetch(`${baseURL}/kb/users`, { headers: adminHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    users.value = data.users || [];
  } catch (e) {
    adminErr.value = `加载用户失败: ${(e as Error).message}`;
  }
}

export async function createUser() {
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

export async function setRole(userId: string, role: string) {
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

export async function loadAudit() {
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

export function openAdmin() {
  showAdmin.value = true;
  adminKey.value = localStorage.getItem("kb_admin_key") || adminKey.value;
  loadUsers();
  loadAudit();
}

export function closeAdmin() {
  showAdmin.value = false;
  localStorage.setItem("kb_admin_key", adminKey.value);
}

export function saveAdminKey() {
  localStorage.setItem("kb_admin_key", adminKey.value);
}

export function initKb() {
  loadKbs();
  loadDocs();
}
