<script setup lang="ts">
import {
  messages, input, loading, expandedCites,
  onSend, sendFeedback, toggleCites, newSession,
} from "./useKbState";
</script>

<template>
  <section class="chat-card">
    <div class="chat-body">
      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
        <div v-if="m.system" class="system-bar">{{ m.content }}</div>
        <template v-else>
          <div class="bubble">{{ m.content }}</div>
          <div v-if="m.score !== undefined && m.role === 'assistant'" class="score-badge">
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
                </div>
                <div class="cite-text">{{ c.text }}</div>
              </div>
            </div>
          </div>
        </template>
      </div>
      <div v-if="loading" class="msg assistant">
        <div class="bubble typing">思考中<span class="dot">...</span></div>
      </div>
    </div>
    <div class="chat-input">
      <button class="new-session-btn" title="开启新会话" @click="newSession">＋</button>
      <input
        v-model="input"
        placeholder="问知识库：例如「采购超过多少要招投标？」"
        @keyup.enter="onSend"
      />
      <button :disabled="loading" @click="onSend">发送</button>
    </div>
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
.system-bar {
  max-width: 90%; padding: 6px 14px; border-radius: 999px;
  background: var(--color-accent-soft); color: var(--color-accent);
  font-size: 13px; text-align: center;
}
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
.dot { animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }
</style>
