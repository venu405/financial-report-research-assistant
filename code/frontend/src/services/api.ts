const baseURL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/* ============================================================================
   客服工作台接口封装（收尾第 2 步 · 2.3）
   后端：/kb/agent/* /kb/tickets /kb/quick-replies，均需 X-API-Key 或 admin token
   ============================================================================ */

export interface AgentConversation {
  id: number;
  thread_id: string;
  kb_id: string;
  visitor_id: string;
  status: string;        // waiting / human / closed / ai
  transfer_reason: string;
  agent_id: string;
  unread_count: number;
  tag: string;
}

export interface AgentMessage {
  role: string;
  content: string;
  created_at: string;
}

export interface QuickReply {
  id: number;
  title: string;
  content: string;
}

export interface Ticket {
  id: number;
  ticket_no: string;
  title: string;
  status: string;
  assignee: string;
  description?: string;
}

/** 管理接口鉴权头：X-API-Key（ADMIN_API_KEY）或 admin 的 X-Api-Token，二选一 */
export function agentHeaders(apiKey: string, token: string): Record<string, string> {
  const h: Record<string, string> = {};
  if (apiKey) h["X-API-Key"] = apiKey;
  if (token) h["X-Api-Token"] = token;
  return h;
}

async function agentFetch(
  url: string,
  opts: RequestInit,
  apiKey: string,
  token: string
): Promise<any> {
  const resp = await fetch(`${baseURL}${url}`, {
    ...opts,
    headers: { ...agentHeaders(apiKey, token), ...(opts.headers || {}) },
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}

export function fetchAgentQueue(apiKey: string, token: string): Promise<{ conversations: AgentConversation[] }> {
  return agentFetch(`/kb/agent/queue`, {}, apiKey, token);
}

export function claimConversation(convId: number, apiKey: string, token: string): Promise<{ claimed: boolean }> {
  return agentFetch(`/kb/agent/claim/${convId}`, { method: "POST" }, apiKey, token);
}

export function fetchMessages(convId: number, apiKey: string, token: string): Promise<{ messages: AgentMessage[] }> {
  return agentFetch(`/kb/agent/conversations/${convId}/messages`, {}, apiKey, token);
}

export function replyConversation(
  convId: number,
  content: string,
  apiKey: string,
  token: string
): Promise<{ replied: boolean }> {
  const form = new FormData();
  form.append("content", content);
  return agentFetch(`/kb/agent/reply/${convId}`, { method: "POST", body: form }, apiKey, token);
}

export function closeConversation(convId: number, apiKey: string, token: string): Promise<{ closed: boolean }> {
  return agentFetch(`/kb/agent/close/${convId}`, { method: "POST" }, apiKey, token);
}

export function listQuickReplies(apiKey: string, token: string): Promise<{ quick_replies: QuickReply[] }> {
  return agentFetch(`/kb/quick-replies`, {}, apiKey, token);
}

export function createTicket(
  payload: { conversation_id?: number; title: string; description: string },
  apiKey: string,
  token: string
): Promise<{ ticket_id: number; ticket_no: string }> {
  return agentFetch(
    `/kb/tickets`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) },
    apiKey,
    token
  );
}

export function listTickets(apiKey: string, token: string): Promise<{ tickets: Ticket[] }> {
  return agentFetch(`/kb/tickets`, {}, apiKey, token);
}
