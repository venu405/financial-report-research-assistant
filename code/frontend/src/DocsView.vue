<script setup lang="ts">
import { computed, ref } from "vue";
import {
  adminKey, currentKb, currentUserRole, docs, docVersions, uploadErr, uploadMsg,
  onUpload, onDelete, pickUpdate, onUpdate,
  createDocDraft, loadDocVersions, submitDocVersion, reviewDocVersion,
  publishDocVersion, rollbackDocVersion,
} from "./useKbState";

const expandedDoc = ref("");
const draftFiles = ref<Record<string, File | undefined>>({});
const draftTitles = ref<Record<string, string>>({});
const versionBusy = ref<number | null>(null);
const versionMsg = ref("");
const versionErr = ref("");
const reviewNotes = ref<Record<number, string>>({});
const canReview = computed(() => Boolean(
  adminKey.value || ["supervisor", "admin"].includes(currentUserRole.value),
));

async function toggleVersions(docId: string) {
  expandedDoc.value = expandedDoc.value === docId ? "" : docId;
  if (expandedDoc.value && !docVersions.value[docId]) {
    try { await loadDocVersions(docId); } catch (e) { versionErr.value = `加载版本失败：${(e as Error).message}`; }
  }
}
function chooseDraft(docId: string, event: Event) { draftFiles.value[docId] = (event.target as HTMLInputElement).files?.[0]; }
async function uploadDraft(docId: string) {
  const file = draftFiles.value[docId];
  if (!file) { versionErr.value = "请先选择要上传的文件"; return; }
  versionBusy.value = -1; versionErr.value = ""; versionMsg.value = "";
  try { await createDocDraft(docId, file, draftTitles.value[docId] || ""); versionMsg.value = "草稿已创建，可送审"; draftFiles.value[docId] = undefined; draftTitles.value[docId] = ""; }
  catch (e) { versionErr.value = `创建草稿失败：${(e as Error).message}`; }
  finally { versionBusy.value = null; }
}
async function submit(version: { id: number; doc_id: string }) {
  versionBusy.value = version.id; versionErr.value = "";
  try { await submitDocVersion(version.id); await loadDocVersions(version.doc_id); versionMsg.value = "版本已送审"; }
  catch (e) { versionErr.value = `送审失败：${(e as Error).message}`; }
  finally { versionBusy.value = null; }
}
async function review(version: { id: number; doc_id: string }, action: "approve" | "reject") {
  versionBusy.value = version.id; versionErr.value = "";
  try { await reviewDocVersion(version.id, action, reviewNotes.value[version.id] || ""); await loadDocVersions(version.doc_id); versionMsg.value = action === "approve" ? "已批准" : "已拒绝"; }
  catch (e) { versionErr.value = `审核失败：${(e as Error).message}`; }
  finally { versionBusy.value = null; }
}
async function publish(version: { id: number; doc_id: string }) {
  versionBusy.value = version.id; versionErr.value = "";
  try { await publishDocVersion(version.id); await loadDocVersions(version.doc_id); versionMsg.value = "版本已发布"; }
  catch (e) { versionErr.value = `发布失败：${(e as Error).message}`; }
  finally { versionBusy.value = null; }
}
async function rollback(version: { id: number; doc_id: string }) {
  if (!window.confirm("确定回滚到这个已发布版本吗？")) return;
  versionBusy.value = version.id; versionErr.value = "";
  try { await rollbackDocVersion(version.id); await loadDocVersions(version.doc_id); versionMsg.value = "已创建回滚版本"; }
  catch (e) { versionErr.value = `回滚失败：${(e as Error).message}`; }
  finally { versionBusy.value = null; }
}
function stateLabel(state: string) { return ({ draft: "草稿", review: "待审核", approved: "已批准", rejected: "已拒绝", published: "已发布" } as Record<string, string>)[state] || state; }
</script>

<template>
  <section class="docs-view">
    <div v-if="versionMsg || versionErr" class="notice" :class="{ error: versionErr }">{{ versionErr || versionMsg }}</div>
    <div class="upload-card">
      <div class="upload-row"><label class="file-btn">📄 上传新文档<input type="file" accept=".md,.txt,.pdf,.docx" hidden @change="onUpload" /></label><input ref="updateInput" type="file" accept=".md,.txt,.pdf,.docx" hidden @change="onUpdate" /><span class="upload-msg" :class="{ err: uploadErr }">{{ uploadErr || uploadMsg }}</span></div>
      <div v-if="docs.length" class="doc-list">
        <article v-for="d in docs" :key="d.doc_id" class="doc-item">
          <div class="doc-main"><span class="doc-title">📑 {{ d.title }}</span><span class="doc-kb">{{ d.kb_id || currentKb }}</span><span class="doc-chunks">{{ d.chunks }} 分块</span></div>
          <div class="doc-actions"><button class="upd-btn" @click="pickUpdate(d.doc_id)">更新</button><button class="del-btn" @click="onDelete(d.doc_id)">删除</button><button class="version-btn" @click="toggleVersions(d.doc_id)">{{ expandedDoc === d.doc_id ? '收起版本' : '版本治理' }}</button></div>
          <div v-if="expandedDoc === d.doc_id" class="version-panel">
            <div class="draft-form"><input v-model="draftTitles[d.doc_id]" placeholder="草稿标题（可选）" /><input type="file" accept=".md,.txt,.pdf,.docx" @change="chooseDraft(d.doc_id, $event)" /><button class="primary" :disabled="versionBusy !== null" @click="uploadDraft(d.doc_id)">上传草稿</button></div>
            <div v-if="docVersions[d.doc_id]?.length" class="version-list">
              <div v-for="v in docVersions[d.doc_id]" :key="v.id" class="version-item">
                <div class="version-meta"><b>v{{ v.version_no }} · {{ v.title }}</b><span class="state" :class="`state-${v.state}`">{{ stateLabel(v.state) }}</span><small>{{ (v.updated_at || v.created_at || '').slice(0, 19) }}</small></div>
                <p v-if="v.review_note" class="review-note">审核意见：{{ v.review_note }}</p>
                <div class="version-actions"><button v-if="v.state === 'draft'" @click="submit(v)">送审</button><template v-if="canReview"><input v-if="v.state === 'review'" v-model="reviewNotes[v.id]" placeholder="审核备注" /><button v-if="v.state === 'review'" @click="review(v, 'approve')">批准</button><button v-if="v.state === 'review'" class="danger" @click="review(v, 'reject')">拒绝</button><button v-if="v.state === 'approved'" @click="publish(v)">发布</button><button v-if="v.state === 'published'" @click="rollback(v)">回滚</button></template></div>
              </div>
            </div>
            <p v-else class="empty">暂无版本记录</p>
          </div>
        </article>
      </div>
      <p v-else class="empty">暂无文档，请先上传一份 .md / .pdf 文件</p>
    </div>
  </section>
</template>

<style scoped>
.docs-view { padding: 16px; }.upload-card { background: var(--color-bg); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 16px; }.upload-row, .doc-main, .doc-actions, .draft-form, .version-actions, .version-meta { display: flex; gap: 8px; align-items: center; }.file-btn, .primary { background: var(--color-accent); color: var(--color-bg); border: none; border-radius: var(--radius-md); padding: 8px 16px; cursor: pointer; font-size: var(--text-sm); }.upload-msg { font-size: var(--text-sm); color: var(--color-charcoal); }.upload-msg.err, .notice.error { color: var(--color-error); }.notice { margin-bottom: 10px; padding: 8px 12px; color: var(--color-success); background: var(--color-bg-quiet); border-radius: var(--radius-md); font-size: var(--text-sm); }.doc-list { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }.doc-item { padding: 10px 12px; border: 1px solid var(--color-border); border-radius: var(--radius-md); }.doc-main { margin-bottom: 8px; }.doc-title { font-weight: var(--weight-medium); flex: 1; }.doc-kb, .state { background: var(--color-bg-quiet); border-radius: var(--radius-sm); padding: 2px 8px; font-size: var(--text-xs); color: var(--color-charcoal); }.doc-chunks, small { font-size: var(--text-xs); color: var(--color-slate); }.doc-actions button, .version-actions button { border: 1px solid var(--color-border); background: var(--color-bg); border-radius: var(--radius-md); padding: 4px 10px; cursor: pointer; font-size: var(--text-xs); }.del-btn, .danger { color: var(--color-error); border-color: var(--color-error) !important; }.version-btn { color: var(--color-accent); }.version-panel { margin-top: 10px; padding: 10px; background: var(--color-bg-quiet); border-radius: var(--radius-md); }.draft-form { flex-wrap: wrap; margin-bottom: 10px; }.draft-form input { padding: 7px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); }.draft-form input:first-child { flex: 1; }.version-list { display: flex; flex-direction: column; gap: 6px; }.version-item { background: var(--color-bg); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: 8px; }.version-meta b { flex: 1; font-size: var(--text-sm); }.version-meta { flex-wrap: wrap; }.state-review { color: #9a6700; }.state-approved, .state-published { color: var(--color-success); }.state-rejected { color: var(--color-error); }.review-note { margin: 5px 0; font-size: var(--text-xs); color: var(--color-charcoal); }.version-actions input { min-width: 150px; padding: 5px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); }.empty { color: var(--color-slate); text-align: center; padding: 16px; font-size: var(--text-sm); }
</style>
