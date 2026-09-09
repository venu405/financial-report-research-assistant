<script setup lang="ts">
import { onMounted, ref } from "vue";
import {
  messages, input, loading, expandedCites,
  onSend, sendFeedback, toggleCites, newSession,
  currentStreamMessage, setCurrentStreamMessage,
  memoryConsent, memories, loadMemory, setMemoryConsent, addMemory, deleteMemory,
} from "./useKbState";

const memoryKey = ref("");
const memoryValue = ref("");
const memoryMsg = ref("");
const memoryErr = ref("");
onMounted(() => { loadMemory().catch((e) => { memoryErr.value = `加载长期记忆失败：${(e as Error).message}`; }); });
async function changeMemoryConsent(event: Event) {
  const enabled = (event.target as HTMLInputElement).checked;
  memoryErr.value = "";
  try { await setMemoryConsent(enabled); memoryMsg.value = enabled ? "已同意保存长期记忆" : "已关闭并清空长期记忆"; }
  catch (e) { (event.target as HTMLInputElement).checked = !enabled; memoryErr.value = `更新同意状态失败：${(e as Error).message}`; }
}
async function saveMemory() {
  if (!memoryKey.value.trim() || !memoryValue.value.trim()) { memoryErr.value = "请填写记忆名称和内容"; return; }
  try { await addMemory(memoryKey.value.trim(), memoryValue.value.trim()); memoryKey.value = ""; memoryValue.value = ""; memoryMsg.value = "记忆已保存"; }
  catch (e) { memoryErr.value = `保存记忆失败：${(e as Error).message}`; }
}
async function removeMemory(id: number) { try { await deleteMemory(id); memoryMsg.value = "记忆已删除"; } catch (e) { memoryErr.value = `删除记忆失败：${(e as Error).message}`; } }
</script>

<template>
  <section class="chat-card">
    <div class="chat-body">
      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
        <template v-if="!m.system">
          <div class="bubble">{{ m.content }}</div>
          <div v-if="m.score != null && m.role === 'assistant'" class="score-badge">
            忠实度 {{ m.score }}/10
          </div>
          <div v-if="m.role === 'assistant' && !m.system" class="feedback-bar">
            <button v-if="m.rating === undefined" class="fb-btn" title="有帮助" @click="sendFeedback(i, 1)">👍</button>
            <button v-if="m.rating === undefined" class="fb-btn" title="没帮助" @click="sendFeedback(i, 0)">👎</button>
            <span v-if="m.rating === 1" class="fb-thanks">已收到评价 👍</span>
            <span v-if="m.rating === 0" class="fb-thanks">已收到评价，我们会改进 👎</span>
          </div>
          <div v-if="m.citations && m.citations.length" class="cite-area">
            <button class="cite-toggle" @click="toggleCites(i)">
              📎 引用 {{ m.citations.length }} 条来源 · {{ expandedCites.has(i) ? '收起' : '展开' }}
            </button>
            <div v-if="expandedCites.has(i)" class="cite-list">
              <div v-for="c in m.citations" :key="c.index" class="cite-card">
                <div class="cite-head">
                  [{{ c.index }}] 📄 {{ c.metadata.doc_title || '未命名' }}
                  <span class="cite-kb">· kb={{ c.metadata.kb_id }}</span>
                  <span v-if="c.page || c.metadata.page_start" class="cite-kb">
                    · 第 {{ c.page || c.metadata.page_start }} 页
                  </span>
                  <span v-if="c.section_path || c.metadata.section_path" class="cite-kb">
                    · {{ c.section_path || c.metadata.section_path }}
                  </span>
                </div>
                <div class="cite-text">{{ c.text }}</div>
              </div>
            </div>
          </div>
        </template>
      </div>
      <div v-if="loading" class="msg assistant">
        <div class="bubble" :class="{ typing: !currentStreamMessage }">{{ currentStreamMessage || '思考中…' }}</div>
      </div>
    </div>
    <div class="chat-input">
      <button class="new-session-btn" title="开启新会话" @click="newSession">＋</button>
      <input
        v-model="input"
        placeholder="问上市公司定期报告：例如「公司的营业收入是多少？」"
        @keyup.enter="onSend"
      />
      <button :disabled="loading" @click="onSend">发送</button>
    </div>
    <section class="memory-panel" aria-label="长期记忆管理">
      <div class="memory-head"><div><strong>长期记忆</strong><p>默认关闭。只有明确同意后，才会保存你提供的信息。</p></div><label class="consent"><input type="checkbox" :checked="memoryConsent" @change="changeMemoryConsent" /> 我同意保存</label></div>
      <p v-if="memoryMsg" class="memory-ok">{{ memoryMsg }}</p><p v-if="memoryErr" class="memory-error">{{ memoryErr }}</p>
      <template v-if="memoryConsent">
        <div class="memory-form"><input v-model="memoryKey" placeholder="名称，如：公司名称" /><input v-model="memoryValue" placeholder="要记住的内容" @keyup.enter="saveMemory" /><button @click="saveMemory">新增记忆</button></div>
        <div v-if="memories.length" class="memory-list"><div v-for="item in memories" :key="item.id" class="memory-item"><span><b>{{ item.key }}</b>：{{ item.value }}</span><button @click="removeMemory(item.id)">删除</button></div></div><p v-else class="memory-empty">暂无已保存记忆</p>
      </template>
    </section>
  </section>
</template>

<style scoped>
.chat-card {
  display: flex;
  flex-direction: column;
  height: 100%;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-bg);
  overflow: hidden;
}
.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  background: var(--color-bg-quiet);
}
.msg { display: flex; }
.msg.user { justify-content: flex-end; }
.msg.assistant { justify-content: flex-start; }
.msg.system { justify-content: center; }
.feedback-bar { display: flex; gap: 6px; margin-top: 4px; align-items: center; }
.fb-btn {
  background: transparent; border: 1px solid var(--color-border);
  border-radius: 6px; padding: 2px 8px; cursor: pointer; font-size: 14px;
  transition: background var(--duration-micro) var(--ease);
}
.fb-btn:hover { background: var(--color-bg-warm); }
.fb-thanks { font-size: 12px; color: var(--color-charcoal); }
.bubble {
  max-width: 80%; padding: 10px 14px; border-radius: 12px;
  font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
}
.msg.user .bubble { background: var(--color-accent); color: var(--color-bg); }
.msg.assistant .bubble { background: var(--color-bg); border: 1px solid var(--color-border); }
.score-badge {
  display: inline-block; margin-top: 4px; padding: 2px 8px;
  background: var(--color-success-bg); color: var(--color-success);
  border: 1px solid rgba(15, 123, 108, 0.25);
  border-radius: 999px; font-size: 11px;
}
.cite-area { margin-top: 4px; }
.cite-toggle {
  background: transparent; border: none; cursor: pointer;
  font-size: 12px; color: var(--color-accent); padding: 0;
}
.cite-list { margin-top: 6px; display: flex; flex-direction: column; gap: 6px; }
.cite-card {
  background: var(--color-bg); border: 1px solid var(--color-border);
  border-radius: var(--radius-md); padding: 8px 10px; font-size: 12px;
}
.cite-head { font-weight: var(--weight-medium); margin-bottom: 4px; }
.cite-kb { color: var(--color-slate); }
.cite-text { color: var(--color-charcoal); line-height: 1.5; }
.chat-input { display: flex; gap: 8px; padding: 10px; border-top: 1px solid var(--color-border); }
.chat-input input {
  flex: 1; padding: 10px 14px; border: 1px solid var(--color-border-strong);
  border-radius: var(--radius-md); font-size: 14px; outline: none;
}
.chat-input button {
  background: var(--color-accent); color: var(--color-bg); border: none;
  border-radius: var(--radius-md); padding: 0 20px; cursor: pointer;
}
.chat-input button:disabled { opacity: .5; cursor: not-allowed; }
.chat-input .new-session-btn {
  background: var(--color-bg); color: var(--color-accent);
  border: 1px solid var(--color-accent); border-radius: var(--radius-md);
  padding: 0 14px; font-size: 18px; cursor: pointer;
}
.typing { color: var(--color-slate); }
.memory-panel { border-top: 1px solid var(--color-border); padding: 10px 12px; background: var(--color-bg); }.memory-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; }.memory-head strong { font-size: var(--text-sm); }.memory-head p { margin: 3px 0 0; color: var(--color-slate); font-size: var(--text-xs); }.consent { white-space: nowrap; font-size: var(--text-xs); color: var(--color-charcoal); }.memory-form { display: flex; gap: 6px; margin-top: 8px; }.memory-form input { min-width: 0; flex: 1; padding: 7px 9px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); }.memory-form button, .memory-item button { border: 1px solid var(--color-border); background: var(--color-bg); color: var(--color-accent); border-radius: var(--radius-md); padding: 5px 9px; cursor: pointer; }.memory-list { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }.memory-item { display: flex; justify-content: space-between; gap: 8px; align-items: center; font-size: var(--text-xs); }.memory-ok { color: var(--color-success); font-size: var(--text-xs); margin: 6px 0 0; }.memory-error { color: var(--color-error); font-size: var(--text-xs); margin: 6px 0 0; }.memory-empty { color: var(--color-slate); font-size: var(--text-xs); margin: 8px 0 0; }
.dot { animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }
</style>
