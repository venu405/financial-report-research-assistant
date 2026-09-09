<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import {
  adminKey, users, auditLogs, newUserName, newUserRole, adminMsg, adminErr,
  loadUsers, createUser, setRole, loadAudit, saveAdminKey,
  kbMeta, newKbId, newKbName, newKbDescription, newKbVisibility,
  loadKbMeta, createKbMeta, setKbVisibility,
  dataSources, loadDataSources, createDataSource, runDataSource, deleteDataSource,
  privacyPolicy, cleanupResult, loadPrivacyPolicy, updatePrivacyPolicy, runPrivacyCleanup,
  pendingDocVersions, loadPendingDocVersions, reviewDocVersion, publishDocVersion,
} from "./useKbState";

const sourceForm = reactive({ name: "", kb_id: "default", source_type: "file", location: "", interval_minutes: 60 });
const governanceMsg = ref("");
const governanceErr = ref("");
const policyForm = reactive({ conversation_days: 30, retrieval_days: 90, feedback_days: 365, audit_days: 365, memory_days: 365, ticket_days: 365 });
onMounted(async () => {
  try { await Promise.all([loadDataSources(), loadPrivacyPolicy(), loadPendingDocVersions()]); Object.assign(policyForm, privacyPolicy.value); }
  catch (e) { governanceErr.value = `加载治理配置失败：${(e as Error).message}`; }
});
async function addSource() {
  if (!sourceForm.name.trim() || !sourceForm.location.trim()) { governanceErr.value = "请填写数据源名称和位置"; return; }
  try { await createDataSource({ ...sourceForm, name: sourceForm.name.trim(), location: sourceForm.location.trim(), interval_minutes: Number(sourceForm.interval_minutes) }); governanceMsg.value = "数据源已创建"; sourceForm.name = ""; sourceForm.location = ""; }
  catch (e) { governanceErr.value = `创建数据源失败：${(e as Error).message}`; }
}
async function runSource(id: number) { try { const result = await runDataSource(id); governanceMsg.value = result.status === "pending_review" ? "同步草稿已生成，请到文档版本中审核发布" : "数据源已检查，内容无变化"; } catch (e) { governanceErr.value = `同步失败：${(e as Error).message}`; } }
async function removeSource(id: number) { if (!window.confirm("确定删除这个数据源吗？")) return; try { await deleteDataSource(id); governanceMsg.value = "数据源已删除"; } catch (e) { governanceErr.value = `删除失败：${(e as Error).message}`; } }
async function savePolicy() { try { await updatePrivacyPolicy({ ...policyForm }); governanceMsg.value = "保留策略已保存"; } catch (e) { governanceErr.value = `保存策略失败：${(e as Error).message}`; } }
async function cleanup(dryRun: boolean) { try { await runPrivacyCleanup(dryRun); governanceMsg.value = dryRun ? "已生成清理预览" : "清理已执行"; } catch (e) { governanceErr.value = `清理失败：${(e as Error).message}`; } }
async function reviewPending(id: number, action: "approve" | "reject") { try { await reviewDocVersion(id, action, "数据源同步审核"); await loadPendingDocVersions(); governanceMsg.value = action === "approve" ? "版本已批准，可继续发布" : "版本已拒绝"; } catch (e) { governanceErr.value = `审核失败：${(e as Error).message}`; } }
async function publishPending(id: number) { try { await publishDocVersion(id); await Promise.all([loadPendingDocVersions(), loadDataSources()]); governanceMsg.value = "版本已发布到知识库"; } catch (e) { governanceErr.value = `发布失败：${(e as Error).message}`; } }
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
          <option value="agent">agent（客服坐席）</option>
          <option value="supervisor">supervisor（客服主管）</option>
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
            <option value="agent">agent</option>
            <option value="supervisor">supervisor</option>
            <option value="admin">admin</option>
          </select>
        </div>
      </div>
      <p v-else class="empty">暂无用户（可先在上方创建，或设 KB_BOOTSTRAP_ADMIN_TOKEN 引导）</p>
    </section>

    <section class="card">
      <h3>📚 知识库可见范围</h3>
      <div class="row wrap">
        <input v-model="newKbId" placeholder="知识库 ID" />
        <input v-model="newKbName" placeholder="显示名称" />
        <input v-model="newKbDescription" placeholder="描述（可选）" />
        <select v-model="newKbVisibility">
          <option value="internal">内部</option><option value="public">公开</option>
        </select>
        <button class="primary" @click="createKbMeta">新建</button>
        <button @click="loadKbMeta">刷新</button>
      </div>
      <div v-if="kbMeta.length" class="user-list">
        <div v-for="kb in kbMeta" :key="kb.kb_id" class="user">
          <span class="name">{{ kb.name || kb.kb_id }}</span>
          <span class="muted">{{ kb.kb_id }} · {{ kb.description }}</span>
          <select :value="kb.visibility" @change="setKbVisibility(kb.kb_id, ($event.target as HTMLSelectElement).value as 'internal' | 'public')">
            <option value="internal">内部</option><option value="public">公开</option>
          </select>
        </div>
      </div>
      <p v-else class="empty">暂无知识库元数据</p>
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

    <section class="card governance-card">
      <h3>数据源管理</h3>
      <p v-if="governanceMsg" class="msg ok">{{ governanceMsg }}</p><p v-if="governanceErr" class="msg err">{{ governanceErr }}</p>
      <div class="row wrap"><input v-model="sourceForm.name" placeholder="名称" /><input v-model="sourceForm.kb_id" placeholder="知识库 ID" /><select v-model="sourceForm.source_type"><option value="file">文件</option><option value="http">HTTP</option></select><input v-model="sourceForm.location" placeholder="允许目录内的文件路径或 HTTPS URL" /><input v-model.number="sourceForm.interval_minutes" type="number" min="5" placeholder="间隔（分钟）" /><button class="primary" @click="addSource">新增</button><button @click="loadDataSources">刷新</button></div>
      <div v-if="dataSources.length" class="user-list"><div v-for="source in dataSources" :key="source.id" class="user source-item"><span class="name">{{ source.name }}</span><span class="chip">{{ source.source_type }}</span><span class="muted">{{ source.location }} · {{ source.interval_minutes }} 分钟</span><span v-if="source.last_error" class="source-error">{{ source.last_error }}</span><button @click="runSource(source.id)">立即同步</button><button class="danger" @click="removeSource(source.id)">删除</button></div></div><p v-else class="empty">暂无数据源</p>
      <h4>待审核同步版本</h4>
      <div v-if="pendingDocVersions.length" class="user-list"><div v-for="version in pendingDocVersions" :key="version.id" class="user source-item"><span class="name">{{ version.title }}</span><span class="chip">v{{ version.version_no }} · {{ version.state }}</span><span class="muted">{{ version.kb_id }} / {{ version.doc_id }}</span><button v-if="version.state === 'review'" @click="reviewPending(version.id, 'approve')">批准</button><button v-if="version.state === 'review'" class="danger" @click="reviewPending(version.id, 'reject')">拒绝</button><button v-if="version.state === 'approved'" class="primary" @click="publishPending(version.id)">发布</button></div></div><p v-else class="empty">暂无待审核版本</p>
    </section>

    <section class="card governance-card">
      <h3>保留策略与隐私清理</h3>
      <div class="policy-grid"><label v-for="(value, key) in policyForm" :key="key">{{ key }}<input v-model.number="policyForm[key as keyof typeof policyForm]" type="number" min="1" /></label></div>
      <div class="row"><button class="primary" @click="savePolicy">保存策略</button><button @click="cleanup(true)">清理预览</button><button class="danger" @click="cleanup(false)">执行清理</button></div>
      <pre v-if="cleanupResult" class="cleanup-result">{{ JSON.stringify(cleanupResult, null, 2) }}</pre>
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
.row.wrap { flex-wrap: wrap; }
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
.governance-card { overflow: hidden; }.source-item { flex-wrap: wrap; }.source-item .muted { flex: 1; }.source-error { color: var(--color-error); font-size: var(--text-xs); }.danger { color: var(--color-error); border-color: var(--color-error); background: var(--color-bg); border-radius: var(--radius-md); padding: 6px 10px; cursor: pointer; }.policy-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-bottom: 12px; }.policy-grid label { display: flex; flex-direction: column; gap: 4px; font-size: var(--text-xs); color: var(--color-charcoal); }.policy-grid input { padding: 7px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); }.cleanup-result { white-space: pre-wrap; font-size: var(--text-xs); background: var(--color-bg-quiet); padding: 8px; margin-top: 10px; border-radius: var(--radius-md); }
</style>
