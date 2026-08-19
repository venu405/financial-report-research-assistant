<script setup lang="ts">
import {
  adminKey, users, auditLogs, newUserName, newUserRole, adminMsg, adminErr,
  loadUsers, createUser, setRole, loadAudit, saveAdminKey,
} from "./useKbState";
</script>

<template>
  <div class="admin-page">
    <section class="card">
      <h3>🔐 管理员鉴权</h3>
      <label class="field">
        <span>Admin API Key（X-API-Key，配了 ADMIN_API_KEY 时用；或用已登录的 admin token）</span>
        <input v-model="adminKey" placeholder="留空则用已登录的 admin 身份" @change="saveAdminKey" />
      </label>
    </section>

    <section class="card">
      <h3>👤 新建用户</h3>
      <div class="row">
        <input v-model="newUserName" placeholder="用户名" />
        <select v-model="newUserRole">
          <option value="member">member（读写）</option>
          <option value="readonly">readonly（只读）</option>
          <option value="admin">admin（全通）</option>
        </select>
        <button class="primary" @click="createUser">创建</button>
      </div>
      <p v-if="adminMsg" class="msg ok">{{ adminMsg }}</p>
      <p v-if="adminErr" class="msg err">{{ adminErr }}</p>
    </section>

    <section class="card">
      <h3>👥 用户列表</h3>
      <div v-if="users.length" class="user-list">
        <div v-for="u in users" :key="u.user_id" class="user">
          <span class="name">{{ u.name }}</span>
          <span class="chip">{{ u.role }}</span>
          <span class="muted">可访问: {{ (u.allowed_kbs || []).join(', ') || '（无）' }}</span>
          <select :value="u.role" @change="setRole(u.user_id, ($event.target as HTMLSelectElement).value)">
            <option value="member">member</option>
            <option value="readonly">readonly</option>
            <option value="admin">admin</option>
          </select>
        </div>
      </div>
      <p v-else class="empty">暂无用户（可先在上方创建，或设 KB_BOOTSTRAP_ADMIN_TOKEN 引导）</p>
    </section>

    <section class="card">
      <h3>📋 审计日志（最近 100 条）</h3>
      <button class="primary" @click="loadAudit">刷新</button>
      <div v-if="auditLogs.length" class="audit">
        <div v-for="(a, i) in auditLogs" :key="i" class="audit-item">
          <span class="muted">{{ (a.ts || '').slice(0, 19) }}</span>
          <span class="chip">{{ a.action }}</span>
          <span class="name">{{ a.user_id }}</span>
          <span class="muted">→ {{ a.target }}</span>
        </div>
      </div>
      <p v-else class="empty">暂无审计记录</p>
    </section>
  </div>
</template>

<style scoped>
.admin-page { padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.card {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px;
}
h3 { margin: 0 0 12px; font-size: var(--text-base); }
.field { display: flex; flex-direction: column; gap: 6px; }
.field span { font-size: var(--text-xs); color: var(--color-charcoal); }
.field input, .row input, .row select, .user select {
  padding: 8px 12px; border: 1px solid var(--color-border-strong);
  border-radius: var(--radius-md); font-size: var(--text-sm);
  background: var(--color-bg); color: var(--color-ink);
}
.row { display: flex; gap: 8px; }
.primary {
  background: var(--color-accent); color: var(--color-bg);
  border: none; border-radius: var(--radius-md); padding: 8px 16px; cursor: pointer; font-size: var(--text-sm);
}
.msg { font-size: var(--text-sm); margin: 8px 0 0; }
.msg.ok { color: var(--color-success); }
.msg.err { color: var(--color-error); }
.user-list, .audit { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.user { display: flex; gap: 10px; align-items: center; padding: 8px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-md); }
.name { font-weight: var(--weight-medium); }
.chip { background: var(--color-bg-quiet); border-radius: var(--radius-sm); padding: 1px 8px; font-size: var(--text-xs); color: var(--color-charcoal); }
.muted { font-size: var(--text-xs); color: var(--color-slate); }
.audit-item { display: flex; gap: 10px; font-size: var(--text-sm); padding: 4px 0; border-bottom: 1px solid var(--color-border-soft); }
.empty { color: var(--color-slate); text-align: center; padding: 16px; font-size: var(--text-sm); }
</style>
