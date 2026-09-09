<script setup lang="ts">
import { onMounted } from "vue";
import {
  kbs, currentKb, userToken, onSwitchKb, onCredentialChange, logout,
  threads, threadId, loadThreads, switchThread, deleteThread,
} from "./useKbState";

defineProps<{ active: string }>();
const emit = defineEmits<{ (e: "select", tab: string): void }>();

const navItems = [
  { key: "chat", label: "财报问答", icon: "◈" },
  { key: "analysis", label: "公司分析", icon: "◌" },
  { key: "peers", label: "同业对比", icon: "⇄" },
  { key: "docs", label: "报告管理", icon: "▤" },
];

onMounted(loadThreads);
</script>

<template>
  <aside class="sidebar">
    <div class="workspace">◈ 上市公司财报研究</div>

    <nav class="nav">
      <button
        v-for="n in navItems"
        :key="n.key"
        class="nav-item"
        :class="{ active: active === n.key }"
        @click="emit('select', n.key)"
      >
        {{ n.icon }} {{ n.label }}
      </button>
    </nav>

    <div class="kb-picker">
      <div class="label">财报资料库</div>
      <select v-model="currentKb" @change="onSwitchKb">
        <option v-for="k in kbs" :key="k" :value="k">{{ k }}</option>
        <option v-if="!kbs.length" value="default">default</option>
      </select>
      <input
        v-model="userToken"
        class="token"
        placeholder="API Token（kb_ 开头）"
        @change="onCredentialChange"
      />
      <button v-if="userToken" class="logout" @click="logout">登出</button>
    </div>

    <!-- 当前用户自己的会话列表 -->
    <div class="threads">
      <div class="label">研究会话（{{ threads.length }}）</div>
      <div v-if="!threads.length" class="threads-empty">暂无财报研究会话</div>
      <div
        v-for="t in threads"
        :key="t.thread_id"
        class="thread-item"
        :class="{ active: t.thread_id === threadId }"
        @click="switchThread(t.thread_id)"
      >
        <span class="thread-id" :title="t.thread_id">💬 {{ t.thread_id.slice(0, 18) }}</span>
        <button class="thread-del" title="删除" @click.stop="deleteThread(t.thread_id)">✕</button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 240px;
  flex-shrink: 0;
  height: 100vh;
  background: var(--color-bg-quiet);
  border-right: 1px solid var(--color-border);
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  box-sizing: border-box;
  position: sticky;
  top: 0;
}
.workspace {
  font-weight: var(--weight-semibold);
  font-size: var(--text-base);
  padding: 0 8px;
}
.nav { display: flex; flex-direction: column; gap: 2px; }
.nav-item {
  text-align: left;
  padding: 8px 12px;
  border: none;
  background: transparent;
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: var(--text-sm);
  color: var(--color-charcoal);
  transition: background var(--duration-micro) var(--ease);
}
.nav-item:hover { background: var(--color-bg-warm); }
.nav-item.active { background: var(--color-accent-soft); color: var(--color-accent); font-weight: var(--weight-medium); }
.kb-picker { display: flex; flex-direction: column; gap: 8px; padding: 0 8px; }
.label { font-size: var(--text-xs); color: var(--color-slate); }
.kb-picker select,
.kb-picker input {
  padding: 8px 10px;
  border: 1px solid var(--color-border-strong);
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
  background: var(--color-bg);
  color: var(--color-ink);
}
.logout {
  border: 1px solid var(--color-border);
  background: var(--color-bg);
  border-radius: var(--radius-md);
  padding: 6px;
  cursor: pointer;
  font-size: var(--text-sm);
  color: var(--color-charcoal);
}
.threads {
  display: flex; flex-direction: column; gap: 4px; padding: 0 8px;
  flex: 1; overflow-y: auto; min-height: 0;
}
.threads-empty { font-size: var(--text-xs); color: var(--color-slate); padding: 4px 0; }
.thread-item {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 8px; border-radius: var(--radius-md); cursor: pointer;
  font-size: var(--text-xs); color: var(--color-charcoal);
  transition: background var(--duration-micro) var(--ease);
}
.thread-item:hover { background: var(--color-bg-warm); }
.thread-item.active { background: var(--color-accent-soft); color: var(--color-accent); }
.thread-id { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.thread-del {
  border: none; background: transparent; color: var(--color-slate);
  cursor: pointer; font-size: 12px; padding: 0 2px;
}
.thread-del:hover { color: var(--color-error); }
</style>
