import assert from "node:assert/strict";
import test from "node:test";

import {
  normalizeAlerts,
  readMetric,
  ticketStatusLabel,
} from "../src/services/operations-utils.js";
import { filterAndSortConversations } from "../src/services/agent-queue-utils.js";

test("ticket status labels remain human-readable with a safe fallback", () => {
  assert.equal(ticketStatusLabel("processing"), "处理中");
  assert.equal(ticketStatusLabel("unknown"), "unknown");
  assert.equal(ticketStatusLabel(""), "未知");
});

test("readMetric resolves the first available nested dashboard value", () => {
  const dashboard = { sessions: { waiting: 7 }, conversations: { waiting: 3 } };
  assert.equal(readMetric(dashboard, ["sessions.total", "sessions.waiting"]), 7);
  assert.equal(readMetric(dashboard, ["tickets.total"], "—"), "—");
});

test("normalizeAlerts accepts both array and wrapped API payloads", () => {
  const alert = { id: "a-1", title: "延迟升高" };
  assert.deepEqual(normalizeAlerts([alert]), [alert]);
  assert.deepEqual(normalizeAlerts({ alerts: [alert] }), [alert]);
  assert.deepEqual(normalizeAlerts({ alerts: null }), []);
});

test("agent queue keeps unread and recently updated conversations visible first", () => {
  const conversations = [
    { id: 1, kb_id: "law", status: "waiting", visitor_id: "old", unread_count: 0, updated_at: "2026-08-20T00:00:00Z" },
    { id: 2, kb_id: "law", status: "human", visitor_id: "latest", unread_count: 0, updated_at: "2026-08-21T00:00:00Z" },
    { id: 3, kb_id: "report", status: "human", visitor_id: "needs-reply", unread_count: 1, updated_at: "2026-08-19T00:00:00Z" },
  ];
  assert.deepEqual(
    filterAndSortConversations(conversations).map((item) => item.id),
    [3, 2, 1],
  );
  assert.deepEqual(
    filterAndSortConversations(conversations, { kb: "law", status: "human", search: "latest" }).map((item) => item.id),
    [2],
  );
});
