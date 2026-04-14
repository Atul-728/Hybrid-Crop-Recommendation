/* ============================================================
   CropOracle Assistant — Shared Chatbot Component
   Features: Draggable, Hideable (with revival tab), Persistent state
   Inject this file into every page: <script src="/static/chatbot.js"></script>
   ============================================================ */
(function () {
  'use strict';

  // ── Inject CSS ──────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    /* ------ Chatbot FAB button ------ */
    #co-chat-fab {
      position: fixed;
      bottom: 24px; right: 24px;
      z-index: 9998;
      width: 58px; height: 58px;
      background: #52b788;
      border: none; border-radius: 50%;
      color: #1c1c1e;
      font-size: 22px;
      cursor: grab;
      box-shadow: 0 6px 24px rgba(82,183,136,0.45);
      display: flex; align-items: center; justify-content: center;
      transition: transform 0.2s, box-shadow 0.2s;
      user-select: none;
      -webkit-user-select: none;
    }
    #co-chat-fab:hover { transform: scale(1.08); box-shadow: 0 10px 32px rgba(82,183,136,0.5); }
    #co-chat-fab.dragging { cursor: grabbing; transform: scale(1.12); }
    #co-chat-fab.hidden { display: none !important; }

    /* ------ Revival Tab (shows when FAB is hidden) ------ */
    #co-chat-tab {
      position: fixed;
      right: 0; bottom: 120px;
      z-index: 9997;
      background: #52b788;
      color: #1c1c1e;
      border: none;
      border-radius: 12px 0 0 12px;
      padding: 10px 8px;
      font-size: 11px; font-weight: 800;
      font-family: 'Inter', sans-serif;
      letter-spacing: 0.5px;
      writing-mode: vertical-rl;
      text-orientation: mixed;
      cursor: pointer;
      box-shadow: -3px 4px 16px rgba(82,183,136,0.35);
      transition: all 0.2s;
      display: none;
    }
    #co-chat-tab:hover { padding-right: 12px; background: #95d5b2; }

    /* ------ Chat Window ------ */
    #co-chat-window {
      position: fixed;
      z-index: 9999;
      width: 360px; height: 460px;
      background: var(--bg, #1c1c1e);
      border: 1px solid var(--border, rgba(255,255,255,0.1));
      border-radius: 18px;
      box-shadow: 0 20px 56px rgba(0,0,0,0.5);
      display: none;
      flex-direction: column;
      overflow: hidden;
      bottom: 90px; right: 24px;
      transition: opacity 0.2s, transform 0.2s;
    }
    #co-chat-window.open { display: flex; animation: co-pop-in 0.22s ease; }
    @keyframes co-pop-in {
      from { opacity: 0; transform: scale(0.92) translateY(12px); }
      to   { opacity: 1; transform: scale(1) translateY(0); }
    }

    /* ------ Header ------ */
    #co-chat-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 16px;
      background: var(--surface, rgba(255,255,255,0.06));
      border-bottom: 1px solid var(--border, rgba(255,255,255,0.08));
      cursor: grab;
      flex-shrink: 0;
    }
    #co-chat-header.dragging { cursor: grabbing; }
    #co-chat-header-title {
      display: flex; align-items: center; gap: 8px;
      font-weight: 700; font-size: 14px;
      color: var(--text-primary, #fff);
      font-family: 'Inter', sans-serif;
    }
    #co-chat-header-title .dot {
      width: 8px; height: 8px; border-radius: 50%; background: #52b788;
      box-shadow: 0 0 6px #52b788;
      animation: co-blink 2s infinite;
    }
    @keyframes co-blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .co-hdr-actions { display: flex; gap: 6px; }
    .co-hdr-btn {
      background: none; border: none;
      color: var(--text-secondary, rgba(255,255,255,0.5));
      cursor: pointer; font-size: 14px;
      width: 28px; height: 28px; border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      transition: all 0.15s;
    }
    .co-hdr-btn:hover { background: rgba(255,255,255,0.08); color: var(--text-primary, #fff); }
    .co-hdr-btn.red:hover { background: rgba(229,57,53,0.15); color: #ff4f5a; }

    /* ------ Messages ------ */
    #co-chat-body {
      flex: 1; overflow-y: auto; padding: 14px 14px 6px;
      display: flex; flex-direction: column; gap: 8px;
      scrollbar-width: thin;
      scrollbar-color: rgba(82,183,136,0.3) transparent;
    }
    #co-chat-body::-webkit-scrollbar { width: 4px; }
    #co-chat-body::-webkit-scrollbar-track { background: transparent; }
    #co-chat-body::-webkit-scrollbar-thumb { background: rgba(82,183,136,0.3); border-radius: 4px; }
    .co-msg {
      max-width: 82%; padding: 9px 13px; border-radius: 14px;
      font-size: 13.5px; line-height: 1.5;
      font-family: 'Inter', sans-serif;
      word-break: break-word;
      animation: co-msg-in 0.18s ease;
    }
    @keyframes co-msg-in { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
    .co-msg.bot {
      align-self: flex-start;
      background: rgba(82,183,136,0.14);
      color: var(--text-primary, #fff);
      border-radius: 14px 14px 14px 3px;
    }
    .co-msg.user {
      align-self: flex-end;
      background: var(--surface, rgba(255,255,255,0.08));
      color: var(--text-primary, #fff);
      border-radius: 14px 14px 3px 14px;
    }

    /* ------ Input bar ------ */
    #co-chat-input-row {
      display: flex; border-top: 1px solid var(--border, rgba(255,255,255,0.08));
      height: 52px; flex-shrink: 0;
    }
    #co-chat-input {
      flex: 1; padding: 12px 14px;
      background: transparent; border: none;
      color: var(--text-primary, #fff); outline: none;
      font-family: 'Inter', sans-serif; font-size: 13.5px;
    }
    #co-chat-input::placeholder { color: var(--text-muted, rgba(255,255,255,0.3)); }
    #co-chat-send {
      background: #52b788; color: #1c1c1e; border: none;
      width: 52px; cursor: pointer; font-size: 15px;
      display: flex; align-items: center; justify-content: center;
      transition: background 0.2s;
    }
    #co-chat-send:hover { background: #95d5b2; }

    /* ------ Light mode overrides ------ */
    [data-theme="light"] #co-chat-window {
      background: #ffffff;
      border-color: rgba(0,0,0,0.09);
      box-shadow: 0 20px 56px rgba(0,0,0,0.14);
    }
    [data-theme="light"] #co-chat-header { background: #f8faf8; border-bottom-color: rgba(0,0,0,0.07); }
    [data-theme="light"] #co-chat-header-title { color: #0d1f0f; }
    [data-theme="light"] .co-hdr-btn { color: #7a9a7a; }
    [data-theme="light"] .co-hdr-btn:hover { background: rgba(45,106,79,0.08); color: #0d1f0f; }
    [data-theme="light"] #co-chat-body { background: #fdfffd; }
    [data-theme="light"] .co-msg.bot { background: rgba(45,106,79,0.09); color: #0d1f0f; }
    [data-theme="light"] .co-msg.user { background: #eef7f1; color: #0d1f0f; }
    [data-theme="light"] #co-chat-input-row { border-top-color: rgba(0,0,0,0.07); background: #f8faf8; }
    [data-theme="light"] #co-chat-input { color: #0d1f0f; background: transparent; }
    [data-theme="light"] #co-chat-input::placeholder { color: #9ab89a; }
  `;
  document.head.appendChild(style);

  // ── Build HTML ──────────────────────────────────────────────
  const fab = document.createElement('button');
  fab.id = 'co-chat-fab';
  fab.title = 'CropOracle Assistant (drag to move)';
  fab.innerHTML = '<i class="fa-solid fa-robot"></i>';

  const revivalTab = document.createElement('button');
  revivalTab.id = 'co-chat-tab';
  revivalTab.title = 'Show CropOracle Assistant';
  revivalTab.textContent = 'AI';

  const win = document.createElement('div');
  win.id = 'co-chat-window';
  win.innerHTML = `
    <div id="co-chat-header">
      <div id="co-chat-header-title">
        <div class="dot"></div>
        CropOracle Assistant
      </div>
      <div class="co-hdr-actions">
        <button class="co-hdr-btn" id="co-minimize-btn" title="Minimize"><i class="fa-solid fa-minus"></i></button>
        <button class="co-hdr-btn red" id="co-hide-btn" title="Hide chatbot (click AI tab to restore)"><i class="fa-solid fa-xmark"></i></button>
      </div>
    </div>
    <div id="co-chat-body">
      <div class="co-msg bot">👋 Hello! I'm your CropOracle AI assistant. Ask me anything about crops, soil, or farming!</div>
    </div>
    <div id="co-chat-input-row">
      <input type="text" id="co-chat-input" placeholder="Ask about crops or farming..." autocomplete="off">
      <button id="co-chat-send"><i class="fa-solid fa-paper-plane"></i></button>
    </div>
  `;

  document.body.appendChild(fab);
  document.body.appendChild(revivalTab);
  document.body.appendChild(win);

  // ── State ────────────────────────────────────────────────────
  const LS = {
    get: k => { try { return JSON.parse(localStorage.getItem('co_chat_' + k)); } catch { return null; } },
    set: (k, v) => localStorage.setItem('co_chat_' + k, JSON.stringify(v))
  };

  let fabPos = LS.get('fab_pos') || { x: window.innerWidth - 82, y: window.innerHeight - 82 };
  let winPos = LS.get('win_pos') || null;
  let isHidden = LS.get('hidden') || false;
  let isOpen = false;

  // Apply saved fab position
  function applyFabPos(x, y) {
    x = Math.max(0, Math.min(x, window.innerWidth - fab.offsetWidth));
    y = Math.max(0, Math.min(y, window.innerHeight - fab.offsetHeight));
    fab.style.left = x + 'px';
    fab.style.top = y + 'px';
    fab.style.right = 'auto';
    fab.style.bottom = 'auto';
    fabPos = { x, y };
    LS.set('fab_pos', fabPos);
  }

  // Apply saved chat window position
  function applyWinPos(x, y) {
    x = Math.max(0, Math.min(x, window.innerWidth - win.offsetWidth - 4));
    y = Math.max(0, Math.min(y, window.innerHeight - win.offsetHeight - 4));
    win.style.left = x + 'px';
    win.style.top = y + 'px';
    win.style.right = 'auto';
    win.style.bottom = 'auto';
    winPos = { x, y };
    LS.set('win_pos', winPos);
  }

  function positionWindowNearFab() {
    const fx = parseFloat(fab.style.left) || fabPos.x;
    const fy = parseFloat(fab.style.top) || fabPos.y;
    let wx = fx - win.offsetWidth - 10;
    let wy = fy - win.offsetHeight + fab.offsetHeight;
    if (wx < 0) wx = fx + fab.offsetWidth + 10;
    if (wy < 0) wy = 10;
    applyWinPos(wx, wy);
  }

  // Apply hidden state
  function applyHiddenState() {
    if (isHidden) {
      fab.classList.add('hidden');
      revivalTab.style.display = 'block';
      closeWindow();
    } else {
      fab.classList.remove('hidden');
      revivalTab.style.display = 'none';
    }
    LS.set('hidden', isHidden);
  }

  // Open/close window
  function openWindow() {
    isOpen = true;
    win.classList.add('open');
    if (!win.style.left) {
      if (winPos) applyWinPos(winPos.x, winPos.y);
      else positionWindowNearFab();
    }
    document.getElementById('co-chat-input').focus();
  }
  function closeWindow() {
    isOpen = false;
    win.classList.remove('open');
  }
  function toggleWindow() { isOpen ? closeWindow() : openWindow(); }

  // ── FAB drag ─────────────────────────────────────────────────
  let dragging = false, startX, startY, origX, origY;

  fab.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    dragging = false;
    startX = e.clientX; startY = e.clientY;
    origX = parseFloat(fab.style.left) || fabPos.x;
    origY = parseFloat(fab.style.top) || fabPos.y;

    const onMove = (ev) => {
      const dx = ev.clientX - startX, dy = ev.clientY - startY;
      if (!dragging && (Math.abs(dx) > 4 || Math.abs(dy) > 4)) {
        dragging = true;
        fab.classList.add('dragging');
        closeWindow();
      }
      if (dragging) applyFabPos(origX + dx, origY + dy);
    };
    const onUp = (ev) => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      fab.classList.remove('dragging');
      if (!dragging) toggleWindow(); // it was a click
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    e.preventDefault();
  });

  // Touch drag for FAB
  fab.addEventListener('touchstart', (e) => {
    const t = e.touches[0];
    startX = t.clientX; startY = t.clientY;
    origX = parseFloat(fab.style.left) || fabPos.x;
    origY = parseFloat(fab.style.top) || fabPos.y;
    dragging = false;

    const onMove = (ev) => {
      const tt = ev.touches[0];
      const dx = tt.clientX - startX, dy = tt.clientY - startY;
      if (!dragging && (Math.abs(dx) > 6 || Math.abs(dy) > 6)) {
        dragging = true;
        fab.classList.add('dragging');
        closeWindow();
      }
      if (dragging) { applyFabPos(origX + dx, origY + dy); ev.preventDefault(); }
    };
    const onEnd = () => {
      fab.removeEventListener('touchmove', onMove);
      fab.removeEventListener('touchend', onEnd);
      fab.classList.remove('dragging');
      if (!dragging) toggleWindow();
    };
    fab.addEventListener('touchmove', onMove, { passive: false });
    fab.addEventListener('touchend', onEnd);
  }, { passive: true });

  // ── Chat Window header drag ───────────────────────────────────
  const hdr = document.getElementById('co-chat-header');
  let wDragging = false, wStartX, wStartY, wOrigX, wOrigY;

  hdr.addEventListener('mousedown', (e) => {
    if (e.target.closest('.co-hdr-btn')) return;
    wStartX = e.clientX; wStartY = e.clientY;
    wOrigX = parseFloat(win.style.left) || winPos?.x || 0;
    wOrigY = parseFloat(win.style.top) || winPos?.y || 0;
    wDragging = true;
    hdr.classList.add('dragging');
    e.preventDefault();
    const onMove = (ev) => {
      if (!wDragging) return;
      applyWinPos(wOrigX + ev.clientX - wStartX, wOrigY + ev.clientY - wStartY);
    };
    const onUp = () => {
      wDragging = false;
      hdr.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });

  // ── Hide / Minimize / Revival ────────────────────────────────
  document.getElementById('co-minimize-btn').addEventListener('click', closeWindow);

  document.getElementById('co-hide-btn').addEventListener('click', () => {
    isHidden = true;
    applyHiddenState();
  });

  revivalTab.addEventListener('click', () => {
    isHidden = false;
    applyHiddenState();
    // Reset to default corner position
    applyFabPos(window.innerWidth - 82, window.innerHeight - 82);
    setTimeout(openWindow, 80);
  });

  // ── Send / Receive messages ───────────────────────────────────
  async function sendMsg() {
    const inp = document.getElementById('co-chat-input');
    const body = document.getElementById('co-chat-body');
    const text = inp.value.trim();
    if (!text) return;
    inp.value = '';

    const userBubble = document.createElement('div');
    userBubble.className = 'co-msg user';
    userBubble.textContent = text;
    body.appendChild(userBubble);

    const botBubble = document.createElement('div');
    botBubble.className = 'co-msg bot';
    botBubble.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
    body.appendChild(botBubble);
    body.scrollTop = body.scrollHeight;

    try {
      const r = await fetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ message: text }),
        headers: { 'Content-Type': 'application/json' }
      });
      const d = await r.json();
      botBubble.innerHTML = d.reply
        .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
        .replace(/\n/g, '<br>');
    } catch (e) {
      botBubble.textContent = 'Connection error. Please try again.';
    }
    body.scrollTop = body.scrollHeight;
  }

  document.getElementById('co-chat-send').addEventListener('click', sendMsg);
  document.getElementById('co-chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMsg();
  });

  // ── Initialize ────────────────────────────────────────────────
  // Position FAB
  if (fabPos) {
    fab.style.position = 'fixed';
    fab.style.left = Math.min(fabPos.x, window.innerWidth - 70) + 'px';
    fab.style.top = Math.min(fabPos.y, window.innerHeight - 70) + 'px';
    fab.style.bottom = 'auto';
    fab.style.right = 'auto';
  }

  // Apply hidden state from localStorage
  applyHiddenState();

  // Handle window resize
  window.addEventListener('resize', () => {
    const fx = parseFloat(fab.style.left);
    const fy = parseFloat(fab.style.top);
    if (!isNaN(fx) && !isNaN(fy)) applyFabPos(fx, fy);
  });

})();
