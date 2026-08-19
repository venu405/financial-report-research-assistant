<template>
  <div class="app-layout">
    <!-- 左侧栏（240px）：工作区名 + 导航 + 知识库选择 -->
    <Sidebar :active="tab" @select="tab = $event" />

    <!-- 主区：页面头 + tab 内容 -->
    <main class="main-area">
      <header class="page-head">
        <h1>{{ tabTitle }}</h1>
      </header>
      <div class="page-body">
        <ChatView v-if="tab === 'chat'" />
        <DocsView v-else-if="tab === 'docs'" />
        <AgentDesk v-else-if="tab === 'agent'" />
        <AdminView v-else-if="tab === 'admin'" />
      </div>
    </main>
  </div>
</template>

<script lang="ts" setup>
import { computed, onMounted, ref } from "vue";
import Sidebar from "./Sidebar.vue";
import ChatView from "./ChatView.vue";
import DocsView from "./DocsView.vue";
import AdminView from "./AdminView.vue";
import AgentDesk from "./AgentDesk.vue";
import { initKb } from "./useKbState";

const tab = ref("chat");
const titles: Record<string, string> = {
  chat: "对话",
  docs: "文档",
  agent: "工作台",
  admin: "管理",
};
const tabTitle = computed(() => titles[tab.value] || "");

onMounted(initKb);
</script>

<style scoped>
.app-layout {
  display: flex;
  min-height: 100vh;
  background: var(--color-bg);
  color: var(--color-ink);
}
.main-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.page-head {
  padding: 16px 24px;
  border-bottom: 1px solid var(--color-border);
}
.page-head h1 {
  margin: 0;
  font-size: var(--text-xl);
  font-weight: var(--weight-semibold);
}
.page-body {
  flex: 1;
  overflow: auto;
}
</style>
