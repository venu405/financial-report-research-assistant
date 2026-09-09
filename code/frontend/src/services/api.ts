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
  priority?: string;
  created_at?: string;
  updated_at?: string;
}

export interface AgentMessage {
  id?: number;
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
  conversation_id?: number | null;
  title: string;
  status: string;
  assignee: string;
  kb_id?: string;
  priority?: string;
  due_at?: string | null;
  description?: string;
  created_at?: string;
  updated_at?: string;
  progress?: TicketProgress[];
}

export interface TicketProgress {
  action: string;
  note: string;
  created_at: string;
}

export interface OperationsAlert {
  id: number | string;
  level?: string;
  severity?: string;
  title?: string;
  message?: string;
  detail?: string;
  created_at?: string;
  acknowledged?: boolean;
  acked?: boolean;
  status?: string;
}

export interface OperationsDashboard {
  generated_at?: string;
  sessions?: Record<string, unknown>;
  conversations?: Record<string, unknown>;
  tickets?: Record<string, unknown>;
  satisfaction?: Record<string, unknown>;
  rag_quality?: Record<string, unknown>;
  rag?: Record<string, unknown>;
  [key: string]: unknown;
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
  payload: { conversation_id?: number; title: string; description: string; kb_id?: string; priority?: string; due_hours?: number },
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

export function getTicket(ticketId: number, apiKey: string, token: string): Promise<Ticket> {
  return agentFetch(`/kb/tickets/${ticketId}`, {}, apiKey, token);
}

export function updateTicketStatus(
  ticketId: number,
  status: string,
  note: string,
  apiKey: string,
  token: string
): Promise<{ ticket_id: number; status: string }> {
  const form = new FormData();
  form.append("status", status);
  form.append("note", note);
  return agentFetch(`/kb/tickets/${ticketId}/status`, { method: "POST", body: form }, apiKey, token);
}

export function assignTicket(
  ticketId: number,
  assignee: string,
  apiKey: string,
  token: string
): Promise<{ ticket_id: number; assignee: string }> {
  const form = new FormData();
  form.append("assignee", assignee);
  return agentFetch(`/kb/tickets/${ticketId}/assign`, { method: "POST", body: form }, apiKey, token);
}

export function fetchOperationsDashboard(
  apiKey: string,
  token: string
): Promise<OperationsDashboard> {
  return agentFetch(`/kb/operations/dashboard`, {}, apiKey, token);
}

export function fetchOperationsAlerts(
  apiKey: string,
  token: string
): Promise<{ alerts: OperationsAlert[] }> {
  return agentFetch(`/kb/operations/alerts`, {}, apiKey, token).then((data) => ({
    alerts: Array.isArray(data) ? data : data.alerts || [],
  }));
}

export function acknowledgeOperationsAlert(
  alertId: number | string,
  apiKey: string,
  token: string
): Promise<{ id: number | string; acknowledged: boolean }> {
  return agentFetch(
    `/kb/operations/alerts/${encodeURIComponent(String(alertId))}/ack`,
    { method: "POST" },
    apiKey,
    token
  );
}
