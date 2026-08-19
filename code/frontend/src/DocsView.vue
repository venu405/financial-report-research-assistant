<script setup lang="ts">
import {
  docs, uploadMsg, uploadErr, updateInput,
  onUpload, onDelete, pickUpdate, onUpdate,
} from "./useKbState";
</script>

<template>
  <section class="docs-view">
    <div class="upload-card">
      <div class="upload-row">
        <label class="file-btn">
          📄 上传文档
          <input type="file" accept=".md,.txt,.pdf,.docx" hidden @change="onUpload" />
        </label>
        <input ref="updateInput" type="file" accept=".md,.txt,.pdf,.docx" hidden @change="onUpdate" />
        <span class="upload-msg" :class="{ err: uploadErr }">{{ uploadErr || uploadMsg }}</span>
      </div>
      <div v-if="docs.length" class="doc-list">
        <div v-for="d in docs" :key="d.doc_id" class="doc-item">
          <span class="doc-title">📁 {{ d.title }}</span>
          <span class="doc-kb">{{ d.kb_id }}</span>
          <span class="doc-chunks">{{ d.chunks }} 分块</span>
          <button class="upd-btn" @click="pickUpdate(d.doc_id)">更新</button>
          <button class="del-btn" @click="onDelete(d.doc_id)">删除</button>
        </div>
      </div>
      <p v-else class="empty">暂无文档，先上传一份 .md / .pdf 试试</p>
    </div>
  </section>
</template>

<style scoped>
.docs-view { padding: 16px; }
.upload-card {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px;
}
.upload-row { display: flex; gap: 10px; align-items: center; }
.file-btn {
  background: var(--color-accent); color: var(--color-bg);
  border: none; border-radius: var(--radius-md);
  padding: 8px 16px; cursor: pointer; font-size: var(--text-sm);
}
.upload-msg { font-size: var(--text-sm); color: var(--color-charcoal); }
.upload-msg.err { color: var(--color-error); }
.doc-list { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }
.doc-item {
  display: flex; gap: 12px; align-items: center;
  padding: 10px 12px; border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
.doc-title { font-weight: var(--weight-medium); flex: 1; }
.doc-kb { background: var(--color-bg-quiet); border-radius: var(--radius-sm); padding: 2px 8px; font-size: var(--text-xs); color: var(--color-charcoal); }
.doc-chunks { font-size: var(--text-xs); color: var(--color-slate); }
.upd-btn, .del-btn {
  border: 1px solid var(--color-border); background: var(--color-bg);
  border-radius: var(--radius-md); padding: 4px 10px; cursor: pointer; font-size: var(--text-xs);
}
.del-btn { color: var(--color-error); border-color: var(--color-error); }
.empty { color: var(--color-slate); text-align: center; padding: 24px; font-size: var(--text-sm); }
</style>
