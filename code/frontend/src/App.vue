<template>
  <div class="app-layout">
    <!-- 左侧栏（240px）：工作区名 + 导航 + 知识库选择 -->
    <Sidebar :active="tab" @select="tab = $event" />

    <!-- 主区：页面头 + tab 内容 -->
    <main class="main-area">
      <header class="page-head">
        <h1>{{ tabTitle }}</h1>
        <p>{{ tabDescription }}</p>
      </header>
      <div class="page-body">
        <template v-if="tab !== 'peers'">
          <ChatView v-if="tab === 'chat'" />
          <CompanyAnalysisView v-else-if="tab === 'analysis'" />
          <DocsView v-else-if="tab === 'docs'" />
        </template>
        <!-- 同业对比用 v-show 保持挂载，切换标签后表单与结果不丢失 -->
        <ResearchPlaceholder v-show="tab === 'peers'" />
      </div>
    </main>

    <!-- 全局轻提示（登入成功/失败等） -->
    <div v-if="notice" class="toast" role="status">{{ notice }}</div>
  </div>
</template>

<script lang="ts" setup>
import { computed, onMounted, ref } from "vue";
import Sidebar from "./Sidebar.vue";
import ChatView from "./ChatView.vue";
import DocsView from "./DocsView.vue";
import CompanyAnalysisView from "./CompanyAnalysisView.vue";
import ResearchPlaceholder from "./ResearchPlaceholder.vue";
import { initKb, notice } from "./useKbState";

const tab = ref("chat");
const titles: Record<string, string> = {
  chat: "财报问答",
  analysis: "公司分析",
  peers: "同业对比",
  docs: "报告管理",
};
const tabTitle = computed(() => titles[tab.value] || "");
const descriptions: Record<string, string> = {
  chat: "围绕上市公司定期报告提问，答案附带原文来源与证据提示。",
  analysis: "按公司、报告期和指标筛选财报记录，支持证据追溯与带原因的人工修订。",
  peers: "手动选择 2 至 3 家公司，按统一报告期与报表口径对比可追溯财务指标。",
  docs: "上传、审核与发布财报研究所需的报告资料。",
};
const tabDescription = computed(() => descriptions[tab.value] || "");

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
.page-head p {
  margin: 5px 0 0;
  color: var(--color-charcoal);
  font-size: var(--text-sm);
}
.page-body {
  flex: 1;
  overflow: auto;
}

.toast {
  position: fixed;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--color-ink);
  color: var(--color-bg);
  padding: 10px 20px;
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
  z-index: 1000;
}
</style>
