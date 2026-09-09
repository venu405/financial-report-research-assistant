import type { AgentConversation } from "./api";

export function filterAndSortConversations(
  conversations: AgentConversation[],
  filters?: { kb?: string; status?: string; search?: string },
): AgentConversation[];
