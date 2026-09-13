// 知识库前端共享状态（模块级单例，供 Sidebar/ChatView/DocsView/AdminView 共享）
import { ref } from "vue";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const credentialKeys = ["kb_api_token", "kb_user_id", "kb_admin_key"] as const;
const visitorTokenKey = "kb_visitor_token";

function createVisitorToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return `kbv_${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function getVisitorToken(): string {
  const saved = localStorage.getItem(visitorTokenKey);
  if (saved && /^kbv_[a-f0-9]{64}$/.test(saved)) return saved;
  const token = createVisitorToken();
  localStorage.setItem(visitorTokenKey, token);
  return token;
}

function clearLegacyLocalCredentials() {
  credentialKeys.forEach((key) => localStorage.removeItem(key));
}

function clearSessionCredentials() {
  credentialKeys.forEach((key) => sessionStorage.removeItem(key));
}

function saveUserCredential(value: string) {
  sessionStorage.removeItem("kb_api_token");
  sessionStorage.removeItem("kb_user_id");
  if (value) sessionStorage.setItem("kb_api_token", value);
}

function saveAdminCredential(value: string) {
  if (value) sessionStorage.setItem("kb_admin_key", value);
  else sessionStorage.removeItem("kb_admin_key");
}

// 清除旧版本遗留的持久化凭据，凭据仅在当前标签页会话中保留。
clearLegacyLocalCredentials();

export interface Citation {
  index: number; chunk_id: string; text: string;
  page?: number; page_start?: number; page_end?: number; section_path?: string;
  metadata: {
    doc_title?: string; kb_id?: string; doc_id?: string; chunk_index?: number;
    page?: number; page_start?: number; page_end?: number; section_path?: string;
  };
}
export interface Msg {
  role: "user" | "assistant" | "agent" | "system";
  content: string;
  score?: number | null;
  citations?: Citation[];
  rating?: number;
  system?: boolean;
  conversationId?: number;
}
export interface VisitorConversation {
  id: number;
  status: "waiting" | "human" | "closed" | "ai" | string;
  agent_id?: string;
}
export interface Doc {
  doc_id: string; title: string; chunks: number; kb_id?: string; source_type?: string;
}
export interface DocVersion {
  id: number;
  doc_id: string;
  kb_id: string;
  version_no: number;
  title: string;
  state: string;
  created_by?: string;
  created_at?: string;
  updated_at?: string;
  submitted_at?: string;
  reviewed_by?: string;
  review_note?: string;
}
export interface DataSource {
  id: number;
  name: string;
  kb_id: string;
  source_type: "file" | "http" | string;
  location: string;
  interval_minutes: number;
  enabled?: boolean;
  last_sync_at?: string;
  last_error?: string;
  next_sync_at?: string;
}
export interface MemoryItem {
  id: number;
  key: string;
  value: string;
  expires_at?: string | null;
  updated_at?: string;
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
export const activeConversation = ref<VisitorConversation | null>(null);
export const visitorToken = getVisitorToken();

export const kbs = ref<string[]>([]);
export const currentKb = ref(localStorage.getItem("kb_id") || "default");
export const userToken = ref(sessionStorage.getItem("kb_api_token") || sessionStorage.getItem("kb_user_id") || "");
export const currentUserRole = ref("");

// 全局轻提示：showNotice 设置文案，约 2.5 秒后自动消失。
export const notice = ref("");
let noticeTimer: ReturnType<typeof setTimeout> | undefined;
export function showNotice(message: string) {
  notice.value = message;
  if (noticeTimer) clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.value = ""; }, 2500);
}

export const docs = ref<Doc[]>([]);
export const uploadMsg = ref("");
export const uploadErr = ref("");
export const updatingDocId = ref("");
export const updateInput = ref<HTMLInputElement | null>(null);

export const threadId = ref(localStorage.getItem(`kb_thread_${currentKb.value}`) || `kb-${Date.now()}`);
let conversationPollTimer: number | undefined;

// ---------- 多会话：统一读取当前用户拥有的服务端会话 ----------
export const threads = ref<{ thread_id: string; status?: string; updated_at?: string }[]>([]);

export async function loadThreads(): Promise<{ thread_id: string; status?: string; updated_at?: string }[]> {
  if (!isTokenAuth()) {
    threads.value = [];
    return [];
  }
  try {
    const query = new URLSearchParams({
      kb_id: currentKb.value,
      visitor_token: visitorToken,
    });
    const resp = await fetch(`${baseURL}/kb/conversations?${query.toString()}`, {
      headers: authHeaders(),
    });
    if (!resp.ok) return [];
    const data = await resp.json();
    threads.value = data.conversations || [];
    return threads.value;
  } catch {
    // 非管理员或后端不可用时静默
  }
  return [];
}

async function loadConversationHistoryAfterThreadSync() {
  const available = await loadThreads();
  if (!available.some((thread) => thread.thread_id === threadId.value)) {
    threadId.value = `kb-${Date.now()}`;
    localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);
    activeConversation.value = null;
    messages.value = [];
    return null;
  }
  return loadConversationHistory();
}

export function switchThread(tid: string) {
  stopConversationPolling();
  activeConversation.value = null;
  threadId.value = tid;
  localStorage.setItem(`kb_thread_${currentKb.value}`, tid);
  messages.value = [];
  void loadConversationHistory();
}

export async function deleteThread(tid: string) {
  if (!window.confirm(`删除会话 ${tid.slice(0, 20)}… 的历史记录？`)) return;
  try {
    const query = new URLSearchParams({ visitor_token: visitorToken });
    const resp = await fetch(`${baseURL}/kb/conversations/${encodeURIComponent(tid)}?${query.toString()}`, {
      method: "DELETE",
      headers: authHeaders(),
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
export const adminKey = ref(sessionStorage.getItem("kb_admin_key") || "");
export const users = ref<{ user_id: string; name: string; role: string; allowed_kbs: string[]; created_at?: string }[]>([]);
export const auditLogs = ref<{ ts: string; user_id: string; action: string; target: string; detail: string }[]>([]);
export const newUserName = ref("");
export const newUserRole = ref("member");
export const adminMsg = ref("");
export const adminErr = ref("");
export const kbMeta = ref<{ kb_id: string; name: string; description: string; visibility: "internal" | "public" }[]>([]);
export const newKbId = ref("");
export const newKbName = ref("");
export const newKbDescription = ref("");
export const newKbVisibility = ref<"internal" | "public">("internal");

// ---------- 鉴权 ----------
export function isTokenAuth(): boolean {
  return userToken.value.startsWith("kb_");
}
export function authHeaders(): Record<string, string> {
  return isTokenAuth() ? { "X-Api-Token": userToken.value } : {};
}
export function adminHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  if (adminKey.value) h["X-API-Key"] = adminKey.value;
  if (isTokenAuth()) h["X-Api-Token"] = userToken.value;
  return h;
}

export async function loadCurrentUser() {
  if (!userToken.value) { currentUserRole.value = ""; return; }
  const query = !isTokenAuth() ? `?user_id=${encodeURIComponent(userToken.value)}` : "";
  const resp = await fetch(`${baseURL}/kb/me${query}`, { headers: authHeaders() });
  if (!resp.ok) { currentUserRole.value = ""; return; }
  currentUserRole.value = String((await resp.json()).role || "");
}

// ---------- 文档版本治理 ----------
export const docVersions = ref<Record<string, DocVersion[]>>({});
export const pendingDocVersions = ref<DocVersion[]>([]);

async function responseError(resp: Response): Promise<Error> {
  const data = await resp.json().catch(() => ({}));
  return new Error(data.detail || `HTTP ${resp.status}`);
}

export async function loadDocVersions(docId: string): Promise<DocVersion[]> {
  const query = !isTokenAuth() && userToken.value
    ? `?user_id=${encodeURIComponent(userToken.value)}` : "";
  const resp = await fetch(`${baseURL}/kb/docs/${encodeURIComponent(docId)}/versions${query}`, {
    headers: authHeaders(),
  });
  if (!resp.ok) throw await responseError(resp);
  const data = await resp.json();
  const versions = Array.isArray(data.versions) ? data.versions : [];
  docVersions.value = { ...docVersions.value, [docId]: versions };
  return versions;
}

export async function loadPendingDocVersions(): Promise<DocVersion[]> {
  const resp = await fetch(`${baseURL}/kb/docs/versions/pending`, { headers: adminHeaders() });
  if (!resp.ok) throw await responseError(resp);
  const data = await resp.json();
  pendingDocVersions.value = Array.isArray(data.versions) ? data.versions : [];
  return pendingDocVersions.value;
}

export async function createDocDraft(docId: string, file: File, title: string): Promise<DocVersion> {
  const form = new FormData();
  form.append("file", file);
  form.append("title", title.trim() || file.name);
  form.append("kb_id", currentKb.value);
  if (!isTokenAuth() && userToken.value) form.append("user_id", userToken.value);
  const resp = await fetch(`${baseURL}/kb/docs/${encodeURIComponent(docId)}/versions`, {
    method: "POST", body: form, headers: authHeaders(),
  });
  if (!resp.ok) throw await responseError(resp);
  const version = await resp.json() as DocVersion;
  await loadDocVersions(docId);
  return version;
}

export async function submitDocVersion(versionId: number): Promise<DocVersion> {
  const query = !isTokenAuth() && userToken.value
    ? `?user_id=${encodeURIComponent(userToken.value)}` : "";
  const resp = await fetch(`${baseURL}/kb/docs/versions/${versionId}/submit${query}`, {
    method: "POST", headers: authHeaders(),
  });
  if (!resp.ok) throw await responseError(resp);
  return await resp.json() as DocVersion;
}

async function supervisorVersionAction(versionId: number, action: "review" | "publish" | "rollback", note = "", reviewAction?: "approve" | "reject") {
  const query = adminKey.value ? "" : (userToken.value ? `?admin_id=${encodeURIComponent(userToken.value)}` : "");
  const options: RequestInit = { method: "POST", headers: adminHeaders() };
  if (action === "review") {
    const form = new FormData();
    form.append("action", reviewAction || "approve");
    form.append("note", note);
    options.body = form;
  }
  const resp = await fetch(`${baseURL}/kb/docs/versions/${versionId}/${action}${query}`, options);
  if (!resp.ok) throw await responseError(resp);
  return await resp.json() as DocVersion;
}

export async function reviewDocVersion(versionId: number, action: "approve" | "reject", note: string) {
  return supervisorVersionAction(versionId, "review", note, action);
}
export async function publishDocVersion(versionId: number) {
  return supervisorVersionAction(versionId, "publish");
}
export async function rollbackDocVersion(versionId: number) {
  return supervisorVersionAction(versionId, "rollback");
}

// ---------- 数据源与隐私治理 ----------
export const dataSources = ref<DataSource[]>([]);
export const privacyPolicy = ref<Record<string, unknown>>({});
export const cleanupResult = ref<Record<string, unknown> | null>(null);

export async function loadDataSources() {
  const resp = await fetch(`${baseURL}/kb/sources`, { headers: adminHeaders() });
  if (!resp.ok) throw await responseError(resp);
  const data = await resp.json();
  dataSources.value = Array.isArray(data.sources) ? data.sources : [];
  return dataSources.value;
}

export async function createDataSource(payload: { name: string; kb_id: string; source_type: string; location: string; interval_minutes: number }) {
  const resp = await fetch(`${baseURL}/kb/sources`, {
    method: "POST", headers: { "Content-Type": "application/json", ...adminHeaders() }, body: JSON.stringify(payload),
  });
  if (!resp.ok) throw await responseError(resp);
  const source = await resp.json() as DataSource;
  await loadDataSources();
  return source;
}
export async function runDataSource(sourceId: number) {
  const resp = await fetch(`${baseURL}/kb/sources/${sourceId}/run`, { method: "POST", headers: adminHeaders() });
  if (!resp.ok) throw await responseError(resp);
  const result = await resp.json();
  await loadDataSources();
  return result;
}
export async function deleteDataSource(sourceId: number) {
  const resp = await fetch(`${baseURL}/kb/sources/${sourceId}`, { method: "DELETE", headers: adminHeaders() });
  if (!resp.ok) throw await responseError(resp);
  await loadDataSources();
}

export async function loadPrivacyPolicy() {
  const resp = await fetch(`${baseURL}/kb/privacy/policy`, { headers: adminHeaders() });
  if (!resp.ok) throw await responseError(resp);
  privacyPolicy.value = (await resp.json()).policy || {};
  return privacyPolicy.value;
}
export async function updatePrivacyPolicy(payload: Record<string, unknown>) {
  const resp = await fetch(`${baseURL}/kb/privacy/policy`, {
    method: "PUT", headers: { "Content-Type": "application/json", ...adminHeaders() }, body: JSON.stringify(payload),
  });
  if (!resp.ok) throw await responseError(resp);
  privacyPolicy.value = (await resp.json()).policy || {};
  return privacyPolicy.value;
}
export async function runPrivacyCleanup(dryRun: boolean) {
  const resp = await fetch(`${baseURL}/kb/privacy/cleanup?dry_run=${dryRun ? "true" : "false"}`, {
    method: "POST", headers: adminHeaders(),
  });
  if (!resp.ok) throw await responseError(resp);
  cleanupResult.value = await resp.json();
  return cleanupResult.value;
}

// ---------- 用户长期记忆 ----------
export const memoryConsent = ref(false);
export const memories = ref<MemoryItem[]>([]);
export async function loadMemory() {
  const query = new URLSearchParams({ visitor_token: visitorToken });
  const resp = await fetch(`${baseURL}/kb/memory?${query.toString()}`, { headers: authHeaders() });
  if (!resp.ok) throw await responseError(resp);
  const data = await resp.json();
  memoryConsent.value = Boolean(data.consent);
  memories.value = Array.isArray(data.memories) ? data.memories : [];
  return memories.value;
}
export async function setMemoryConsent(enabled: boolean) {
  const resp = await fetch(`${baseURL}/kb/memory/consent`, {
    method: "PUT", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ enabled, visitor_token: visitorToken }),
  });
  if (!resp.ok) throw await responseError(resp);
  memoryConsent.value = Boolean((await resp.json()).consent);
  if (!memoryConsent.value) memories.value = [];
}
export async function addMemory(key: string, value: string) {
  const resp = await fetch(`${baseURL}/kb/memory`, {
    method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ key, value, visitor_token: visitorToken }),
  });
  if (!resp.ok) throw await responseError(resp);
  const item = await resp.json() as MemoryItem;
  await loadMemory();
  return item;
}
export async function deleteMemory(memoryId: number) {
  const query = new URLSearchParams({ visitor_token: visitorToken });
  const resp = await fetch(`${baseURL}/kb/memory/${memoryId}?${query.toString()}`, {
    method: "DELETE", headers: authHeaders(),
  });
  if (!resp.ok) throw await responseError(resp);
  memories.value = memories.value.filter((item) => item.id !== memoryId);
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
function stopConversationPolling() {
  if (conversationPollTimer !== undefined) {
    window.clearInterval(conversationPollTimer);
    conversationPollTimer = undefined;
  }
}

function toChatMessages(rawMessages: unknown): Msg[] {
  if (!Array.isArray(rawMessages)) return [];
  return rawMessages
    .filter((message): message is { role?: unknown; content?: unknown } => Boolean(message && typeof message === "object"))
    .map((message) => ({
      role: message.role === "agent" ? "agent" : message.role === "assistant" ? "assistant" : "user",
      content: typeof message.content === "string" ? message.content : "",
    }));
}

function setConversationHint(conversation: VisitorConversation | null) {
  messages.value = messages.value.filter(
    (message) => !message.system || Boolean(conversation && message.conversationId !== conversation.id),
  );
  if (!conversation || conversation.status === "ai") return;
  const content = conversation.status === "waiting"
    ? "已为您转接人工客服，等待接入…"
    : conversation.status === "human"
      ? "座席已接入，请继续对话。"
      : "本次人工服务已结束。";
  messages.value.push({ role: "system", system: true, content, conversationId: conversation.id });
}

async function refreshConversationHistory(expectedThreadId = threadId.value): Promise<VisitorConversation | null> {
  const query = new URLSearchParams({ visitor_token: visitorToken });
  const resp = await fetch(
    `${baseURL}/kb/conversations/${encodeURIComponent(expectedThreadId)}/messages?${query.toString()}`,
    { headers: authHeaders() },
  );
  if (resp.status === 404) {
    if (threadId.value === expectedThreadId) activeConversation.value = null;
    return null;
  }
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  const data = await resp.json();
  if (threadId.value !== expectedThreadId) return null;
  const conversation = data.conversation as VisitorConversation | null;
  activeConversation.value = conversation;
  messages.value = toChatMessages(data.messages);
  setConversationHint(conversation);
  if (!conversation || conversation.status === "closed" || conversation.status === "ai") {
    stopConversationPolling();
  }
  return conversation;
}

function startConversationPolling(expectedThreadId = threadId.value) {
  stopConversationPolling();
  conversationPollTimer = window.setInterval(() => {
    if (threadId.value !== expectedThreadId) {
      stopConversationPolling();
      return;
    }
    refreshConversationHistory(expectedThreadId).catch(() => {
      // 短暂网络故障时继续保留轮询，下次再恢复。
    });
  }, 5000);
}

export async function loadConversationHistory() {
  stopConversationPolling();
  try {
    const conversation = await refreshConversationHistory();
    if (conversation && (conversation.status === "waiting" || conversation.status === "human")) {
      startConversationPolling();
    }
  } catch {
    // 当前线程尚未转人工或服务暂不可用时，保留现有聊天界面。
  }
}

async function sendVisitorHumanMessage(content: string) {
  const expectedThreadId = threadId.value;
  const resp = await fetch(`${baseURL}/kb/conversations/${encodeURIComponent(expectedThreadId)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ content, visitor_token: visitorToken }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  await refreshConversationHistory(expectedThreadId);
}

export function onSwitchKb() {
  stopConversationPolling();
  activeConversation.value = null;
  localStorage.setItem("kb_id", currentKb.value);
  saveUserCredential(userToken.value);
  threadId.value = `kb-${Date.now()}`;
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);
  messages.value = [];
  loadKbs();
  loadDocs();
  void loadConversationHistoryAfterThreadSync();
}

export function onCredentialChange() {
  saveUserCredential(userToken.value);
  loadKbs();
  loadDocs();
  void loadConversationHistoryAfterThreadSync();
  void loadCurrentUser();
}

// 显式登入：保存凭据后用 /kb/me 验证 token，给出成功/失败提示。
export async function login() {
  if (!userToken.value) return;
  saveUserCredential(userToken.value);
  loadKbs();
  loadDocs();
  void loadConversationHistoryAfterThreadSync();
  await loadCurrentUser();
  if (currentUserRole.value) {
    showNotice(`登入成功（${currentUserRole.value}）`);
  } else {
    showNotice("登入失败：token 无效或已过期");
  }
}

export function newSession() {
  stopConversationPolling();
  activeConversation.value = null;
  threadId.value = `kb-${Date.now()}`;
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);
  messages.value = [];
}

export async function endHumanAndStartNewSession() {
  const conversation = activeConversation.value;
  if (!conversation || !["waiting", "human"].includes(conversation.status)) {
    newSession();
    return;
  }
  if (!window.confirm("结束当前人工服务并开始新的智能问答会话？")) return;
  const query = new URLSearchParams({ visitor_token: visitorToken });
  const resp = await fetch(
    `${baseURL}/kb/conversations/${encodeURIComponent(threadId.value)}/close?${query.toString()}`,
    { method: "POST", headers: authHeaders() },
  );
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  await loadThreads();
  newSession();
}

export function logout() {
  userToken.value = "";
  stopConversationPolling();
  activeConversation.value = null;
  clearLegacyLocalCredentials();
  clearSessionCredentials();
  messages.value = [];
  currentUserRole.value = "";
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

  input.value = "";
  loading.value = true;
  currentStreamMessage.value = "";
  localStorage.setItem(`kb_thread_${currentKb.value}`, threadId.value);

  if (!activeConversation.value) {
    try {
      await refreshConversationHistory();
    } catch {
      // 新会话通常还不存在；继续走智能问答即可。
    }
  }

  if (activeConversation.value && (activeConversation.value.status === "waiting" || activeConversation.value.status === "human")) {
    try {
      await sendVisitorHumanMessage(question);
      if (activeConversation.value && (activeConversation.value.status === "waiting" || activeConversation.value.status === "human")) {
        startConversationPolling();
      }
    } catch (e) {
      messages.value.push({ role: "assistant", content: `\u274c\u53d1\u9001\u4eba\u5de5\u6d88\u606f\u5931\u8d25: ${(e as Error).message}` });
    } finally {
      loading.value = false;
    }
    return;
  }

  messages.value.push({ role: "user", content: question });

  const history = messages.value
    .slice(-6, -1)
    .map((m) => ({ role: m.role === "agent" ? "assistant" : m.role, content: m.content }));

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
        visitor_token: visitorToken,
      }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 409 && String(data.detail || "").includes("会话已转人工")) {
        const conversation = await refreshConversationHistory();
        if (conversation && ["waiting", "human"].includes(conversation.status)) {
          await sendVisitorHumanMessage(question);
          startConversationPolling();
          await loadThreads();
          return;
        }
      }
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
        activeConversation.value = {
          id: final.conversation_id,
          status: final.status || "waiting",
          agent_id: final.agent_id,
        };
        messages.value.push({
          role: "system",
          system: true,
          content: "已为您转接人工客服，等待接入…",
          conversationId: final.conversation_id,
        });
        await loadConversationHistory();
      }
      await loadThreads();
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
      const query = new URLSearchParams({ visitor_token: visitorToken });
      const resp = await fetch(`${baseURL}/kb/conversation/${convId}/status?${query.toString()}`, {
        headers: authHeaders(),
      });
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

export async function loadKbMeta() {
  try {
    const resp = await fetch(`${baseURL}/kb/meta`, { headers: adminHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    kbMeta.value = (await resp.json()).kbs || [];
  } catch (e) {
    adminErr.value = `加载知识库设置失败: ${(e as Error).message}`;
  }
}

export async function createKbMeta() {
  if (!newKbId.value.trim() || !newKbName.value.trim()) return;
  const form = new FormData();
  form.append("kb_id", newKbId.value.trim());
  form.append("name", newKbName.value.trim());
  form.append("description", newKbDescription.value.trim());
  form.append("visibility", newKbVisibility.value);
  try {
    const resp = await fetch(`${baseURL}/kb/meta`, { method: "POST", body: form, headers: adminHeaders() });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    newKbId.value = ""; newKbName.value = ""; newKbDescription.value = ""; newKbVisibility.value = "internal";
    adminMsg.value = "知识库已创建";
    await loadKbMeta();
    await loadKbs();
  } catch (e) {
    adminErr.value = `创建知识库失败: ${(e as Error).message}`;
  }
}

export async function setKbVisibility(kbId: string, visibility: "internal" | "public") {
  const form = new FormData();
  form.append("visibility", visibility);
  try {
    const resp = await fetch(`${baseURL}/kb/meta/${encodeURIComponent(kbId)}`, {
      method: "PUT", body: form, headers: adminHeaders(),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    adminMsg.value = `${kbId} 已设为${visibility === "public" ? "公开" : "内部"}`;
    await loadKbMeta();
  } catch (e) {
    adminErr.value = `更新可见范围失败: ${(e as Error).message}`;
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
  adminKey.value = sessionStorage.getItem("kb_admin_key") || adminKey.value;
  loadUsers();
  loadAudit();
  loadKbMeta();
}

export function closeAdmin() {
  showAdmin.value = false;
  saveAdminCredential(adminKey.value);
}

export function saveAdminKey() {
  saveAdminCredential(adminKey.value);
}

export function initKb() {
  loadKbs();
  loadDocs();
  void loadConversationHistoryAfterThreadSync();
  void loadCurrentUser();
}
