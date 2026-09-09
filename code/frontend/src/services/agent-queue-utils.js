export function filterAndSortConversations(conversations, filters = {}) {
  const kb = filters.kb || "all";
  const status = filters.status || "all";
  const search = String(filters.search || "").trim().toLowerCase();
  return [...(conversations || [])]
    .filter((conversation) => kb === "all" || conversation.kb_id === kb)
    .filter((conversation) => status === "all" || conversation.status === status)
    .filter((conversation) => !search || [conversation.visitor_id, conversation.transfer_reason, conversation.thread_id]
      .some((value) => String(value || "").toLowerCase().includes(search)))
    .sort((left, right) => {
      const unread = Number(right.unread_count || 0) - Number(left.unread_count || 0);
      if (unread) return unread;
      return String(right.updated_at || right.created_at || "").localeCompare(
        String(left.updated_at || left.created_at || "")
      );
    });
}
