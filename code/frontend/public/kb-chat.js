/*!
 * 上市公司财报智能问答挂件 SDK（注入式）
 * 用法：一行 script 嵌入任意网页
 *   <script data-kb-base="http://localhost:8000" data-kb-id="default" data-channel="web"
 *           src="http://localhost:5174/kb-chat.js"></script>
 * data-kb-base  后端地址（默认 http://localhost:8000）
 * data-kb-id    知识库 ID（默认 default）
 * data-channel  渠道标识（默认 web）
 */
(function () {
  "use strict";

  // 读取当前 script 标签的 data 属性
  var cfg = { base: "http://localhost:8000", kbId: "default", channel: "web" };
  var scripts = document.querySelectorAll("script[data-kb-id], script[data-kb-base]");
  for (var i = 0; i < scripts.length; i++) {
    var s = scripts[i];
    if (s.getAttribute("data-kb-base")) cfg.base = s.getAttribute("data-kb-base");
    if (s.getAttribute("data-kb-id")) cfg.kbId = s.getAttribute("data-kb-id");
    if (s.getAttribute("data-channel")) cfg.channel = s.getAttribute("data-channel");
  }

  function createVisitorToken() {
    var bytes = new Uint8Array(32);
    window.crypto.getRandomValues(bytes);
    var hex = "";
    for (var i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, "0");
    return "kbv_" + hex;
  }

  var visitorToken = localStorage.getItem("kb_visitor_token");
  if (!visitorToken || !/^kbv_[a-f0-9]{64}$/.test(visitorToken)) {
    visitorToken = createVisitorToken();
    localStorage.setItem("kb_visitor_token", visitorToken);
  }

  // 访客临时身份（不要求登录）
  var visitorId = localStorage.getItem("kb_visitor_id");
  if (!visitorId) {
    visitorId = "visitor-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
    localStorage.setItem("kb_visitor_id", visitorId);
  }
  var threadStorageKey = "kb_widget_thread_" + cfg.kbId;
  var threadId = localStorage.getItem(threadStorageKey) || ("kb-" + Date.now());
  localStorage.setItem(threadStorageKey, threadId);
  var conversation = null;
  var pollTimer = null;

  // 注入样式
  var css = [
    "#kb-bubble{position:fixed;right:20px;bottom:20px;width:56px;height:56px;border-radius:50%;",
    "background:#0075de;color:#fff;border:none;cursor:pointer;font-size:24px;",
    "box-shadow:0 4px 14px rgba(0,117,222,.35);display:flex;align-items:center;",
    "justify-content:center;z-index:9999;font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}",
    "#kb-panel{position:fixed;right:20px;bottom:90px;width:340px;height:480px;background:#fff;",
    "border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.18);display:none;flex-direction:column;",
    "overflow:hidden;z-index:9998;font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}",
    "#kb-panel.open{display:flex}",
    "#kb-head{background:#0075de;color:#fff;padding:12px 16px;font-size:15px;font-weight:600}",
    "#kb-head small{display:block;font-weight:400;opacity:.85;font-size:11px}",
    "#kb-msgs{flex:1;overflow-y:auto;padding:12px;background:#f7f7f5;display:flex;flex-direction:column;gap:8px}",
    ".kb-msg{max-width:82%;padding:8px 12px;border-radius:10px;font-size:14px;line-height:1.55;",
    "word-break:break-word;white-space:pre-wrap}",
    ".kb-msg.user{align-self:flex-end;background:#0075de;color:#fff}",
    ".kb-msg.assistant{align-self:flex-start;background:#fff;color:#111;border:1px solid #e5e7eb}",
    "#kb-input{display:flex;border-top:1px solid #e5e7eb;padding:10px;gap:8px;background:#fff}",
    "#kb-input textarea{flex:1;border:1px solid #d1d5db;border-radius:8px;padding:8px;font-size:14px;",
    "resize:none;outline:none;font-family:inherit}",
    "#kb-input button{background:#0075de;color:#fff;border:none;border-radius:8px;padding:0 16px;cursor:pointer;font-size:14px}"
  ].join("");
  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // 注入 DOM
  var bubble = document.createElement("button");
  bubble.id = "kb-bubble";
  bubble.title = "在线问答";
  bubble.textContent = "💬";
  document.body.appendChild(bubble);

  var panel = document.createElement("div");
  panel.id = "kb-panel";
  panel.innerHTML =
    '<div id="kb-head">财报智能问答 <small id="kb-hint">AI 服务中</small></div>' +
    '<div id="kb-msgs"></div>' +
    '<div id="kb-input"><textarea id="kb-q" rows="1" placeholder="输入您的问题…"></textarea>' +
    '<button id="kb-send">发送</button></div>';
  document.body.appendChild(panel);

  var msgs = document.getElementById("kb-msgs");
  var q = document.getElementById("kb-q");
  var hint = document.getElementById("kb-hint");

  bubble.addEventListener("click", function () { panel.classList.toggle("open"); });

  function addMsg(role, text) {
    var d = document.createElement("div");
    d.className = "kb-msg " + role;
    d.textContent = text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  function setHint(status) {
    if (status === "waiting") hint.textContent = "已为您转接人工客服，等待接入…";
    else if (status === "human") hint.textContent = "座席已接入，请继续对话。";
    else if (status === "closed") hint.textContent = "本次服务已结束。";
    else hint.textContent = "AI 服务中";
  }

  function renderMessages(serverMessages) {
    msgs.innerHTML = "";
    (serverMessages || []).forEach(function (message) {
      addMsg(message.role === "agent" ? "assistant" : message.role, message.content || "");
    });
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function refreshConversation() {
    var url = cfg.base + "/kb/conversations/" + encodeURIComponent(threadId) + "/messages?visitor_token=" + encodeURIComponent(visitorToken);
    return fetch(url)
      .then(function (resp) {
        if (resp.status === 404) { conversation = null; stopPolling(); return null; }
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(function (data) {
        if (!data) return null;
        conversation = data.conversation || null;
        renderMessages(data.messages);
        setHint(conversation && conversation.status);
        if (!conversation || conversation.status === "closed" || conversation.status === "ai") stopPolling();
        return conversation;
      });
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(function () { refreshConversation().catch(function () {}); }, 5000);
  }

  function sendHumanMessage(content) {
    return fetch(cfg.base + "/kb/conversations/" + encodeURIComponent(threadId) + "/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: content, visitor_token: visitorToken })
    }).then(function (resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return refreshConversation();
    });
  }

  function pollStatus(convId) {
    var n = 0;
    var timer = setInterval(function () {
      n++;
      fetch(cfg.base + "/kb/conversation/" + convId + "/status?visitor_token=" + encodeURIComponent(visitorToken))
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.status === "human") { hint.textContent = "坐席已接入"; clearInterval(timer); }
          else if (d.status === "closed") { hint.textContent = "服务已结束"; clearInterval(timer); }
        })
        .catch(function () {});
      if (n >= 24) clearInterval(timer);
    }, 5000);
  }

  function ask() {
    var question = q.value.trim();
    if (!question) return;
    if (conversation && (conversation.status === "waiting" || conversation.status === "human")) {
      q.value = "";
      sendHumanMessage(question)
        .then(function (current) { if (current && current.status !== "closed") startPolling(); })
        .catch(function (e) { addMsg("assistant", "\u274c\u53d1\u9001\u4eba\u5de5\u6d88\u606f\u5931\u8d25: " + e.message); });
      return;
    }
    addMsg("user", question);
    q.value = "";
    var el = addMsg("assistant", "");
    el.style.opacity = "0.6";

    fetch(cfg.base + "/kb/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, kb_id: cfg.kbId, thread_id: threadId, user_id: visitorId, visitor_token: visitorToken }),
    }).then(function (resp) {
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buf = "", answer = "";
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) return;
          buf += decoder.decode(r.value, { stream: true });
          var parts = buf.split("\n\n");
          buf = parts.pop();
          parts.forEach(function (chunk) {
            var line = chunk.trim();
            if (!line.startsWith("data:")) return;
            try {
              var ev = JSON.parse(line.slice(5));
              if (ev.type === "token") { answer += ev.text; }
              else if (ev.type === "final") {
                answer = ev.answer;
                if (ev.escalate) {
                  conversation = { id: ev.conversation_id, status: ev.status || "waiting", agent_id: ev.agent_id };
                  refreshConversation().then(function (current) {
                    if (current && current.status !== "closed") startPolling();
                  }).catch(function () {});
                }
              } else if (ev.type === "error") { answer = "❌ " + ev.detail; }
            } catch (e) {}
          });
          el.style.opacity = "1";
          el.textContent = answer;
          msgs.scrollTop = msgs.scrollHeight;
          return pump();
        });
      }
      return pump();
    }).catch(function (e) {
      el.style.opacity = "1";
      el.textContent = "❌ 连接失败: " + e.message;
    });
  }

  document.getElementById("kb-send").addEventListener("click", ask);
  q.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
  refreshConversation().then(function (current) {
    if (current && (current.status === "waiting" || current.status === "human")) startPolling();
  }).catch(function () {});
})();
