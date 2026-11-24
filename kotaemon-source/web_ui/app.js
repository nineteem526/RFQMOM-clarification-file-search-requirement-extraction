/* RFQ Engineering Knowledge Workbench — Frontend Application Controller */

const state = {
  projects: [],
  projectId: "rfq-demo",
  activeTab: "qa",
  docsData: { folders: [], tree: [], documents: [], total: 0, last_updated: "" },
  selectedFolder: null, // null = all or first folder
  selectedDocId: null,
  qaSearch: "",
  docsSearch: "",
  chatMessages: [],
  activeCitations: [],
  activeCitationIndex: 0,
  previewChunks: [],
  previewChunkIdx: 0,
  previewDocId: null,
  compareLeftId: null,
  compareRightId: null,
  compareKeyword: "",
  compareSections: [],
  compareData: null,
  savedKind: "all",
  savedKeyword: "",
  savedItems: [],
  selectedSavedId: null,
  settingsData: null,
  statusData: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => el.classList.add("hidden"), 2600);
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = `请求失败 (${res.status})`;
    try {
      const err = await res.json();
      if (err.detail) detail = err.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function formatSize(bytes) {
  const n = Number(bytes || 0);
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
  if (n >= 1024) return (n / 1024).toFixed(1) + " KB";
  return n + " B";
}

function highlightText(text, needle, cls = "hl-yellow") {
  const safe = esc(text);
  if (!needle || !needle.trim()) return safe;
  const escapedNeedle = esc(needle.trim()).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  try {
    return safe.replace(new RegExp(`(${escapedNeedle})`, "gi"), `<mark class="${cls}">$1</mark>`);
  } catch {
    return safe;
  }
}

function formatInlineMarkdown(text) {
  let html = esc(text);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[(\d+)\]/g, '<a class="cite-link" data-cite="$1">[$1]</a>');
  return html;
}

/* ==================== PROJECTS & SWITCHER ==================== */
async function loadProjects(preferredId = null) {
  const data = await api("/api/projects");
  state.projects = data.projects || [];
  if (preferredId && state.projects.some((p) => p.id === preferredId)) {
    state.projectId = preferredId;
  } else if (!state.projects.some((p) => p.id === state.projectId) && state.projects.length) {
    state.projectId = state.projects[0].id;
  }
  renderProjectSwitcher();
  await reloadProjectContext();
}

function renderProjectSwitcher() {
  const cur = state.projects.find((p) => p.id === state.projectId);
  $("#current-project-name").textContent = cur ? cur.name : state.projectId;
  $("#saved-project-readonly").value = cur ? cur.name : state.projectId;

  const list = $("#project-list");
  list.innerHTML = state.projects
    .map(
      (p) => `
      <button class="project-option ${p.id === state.projectId ? "active" : ""}" data-pid="${esc(p.id)}" type="button">
        <span>
          <span class="project-option-name">${esc(p.name)}</span>
          <span class="project-option-meta">ID: ${esc(p.id)}</span>
        </span>
        <span class="project-option-badge">${p.doc_count} 份资料</span>
      </button>
    `
    )
    .join("");

  list.querySelectorAll(".project-option").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const pid = btn.dataset.pid;
      $("#project-dropdown").classList.add("hidden");
      if (pid !== state.projectId) {
        state.projectId = pid;
        state.selectedFolder = null;
        state.selectedDocId = null;
        state.compareLeftId = null;
        state.compareRightId = null;
        state.compareSections = [];
        state.selectedSavedId = null;
        renderProjectSwitcher();
        await reloadProjectContext();
        const proj = state.projects.find((x) => x.id === pid);
        showToast(`已切换至项目数据库：${proj ? proj.name : pid}`);
      }
    });
  });
}

async function reloadProjectContext() {
  await Promise.all([
    loadDocuments(),
    loadChatHistory(),
    loadSavedItems(),
    loadSettings(),
  ]);
  initCompareDefaults();
}

/* ==================== DOCUMENTS TREE & TABLE ==================== */
async function loadDocuments() {
  const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/documents`);
  state.docsData = data;
  if (!state.selectedFolder && data.tree.length) {
    const firstNonEmpty = data.tree.find((g) => g.count > 0) || data.tree[0];
    state.selectedFolder = firstNonEmpty.name;
  }
  if (!state.selectedDocId && data.documents.length) {
    const inFolder = data.documents.find((d) => d.folder === state.selectedFolder) || data.documents[0];
    state.selectedDocId = inFolder.id;
  }
  renderDocTrees();
  renderDocsTable();
  if (state.selectedDocId) {
    await loadDocDetail(state.selectedDocId);
  } else {
    renderEmptyDocDetail();
  }
  populateUploadFolders();
}

function buildTreeHtml(tree, searchTerm, activeDocId, activeFolder, mode) {
  const term = searchTerm.trim().toLowerCase();
  if (!tree.length) return '<div class="empty-state-box">暂无文件夹</div>';
  return tree
    .map((group) => {
      const docs = term
        ? group.documents.filter((d) => d.name.toLowerCase().includes(term))
        : group.documents;
      if (term && !docs.length) return "";
      const isFolderActive = mode === "docs" && activeFolder === group.name;
      return `
        <div class="tree-folder">
          <button class="tree-folder-head ${isFolderActive ? "active" : ""}" data-folder="${esc(group.name)}" type="button">
            <span>⌄</span>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            <span>${esc(group.name)}</span>
            <span class="tree-folder-count">(${group.count})</span>
          </button>
          <ul class="tree-files">
            ${docs
              .map(
                (d) => `
              <li>
                <button class="tree-file-btn ${d.id === activeDocId ? "active" : ""}" data-doc-id="${esc(d.id)}" data-folder="${esc(group.name)}" type="button">
                  <span class="tree-file-left">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>
                    <span class="tree-file-name" title="${esc(d.name)}">${esc(d.name)}</span>
                  </span>
                  ${d.version ? `<span class="tree-file-ver">(${esc(d.version)})</span>` : ""}
                </button>
              </li>`
              )
              .join("")}
          </ul>
        </div>
      `;
    })
    .join("");
}

function renderDocTrees() {
  const total = state.docsData.total || 0;
  const updated = state.docsData.last_updated || "";
  $("#qa-doc-count").textContent = `${total} 份文件`;
  $("#docs-doc-count").textContent = `${total} 份文件`;
  $("#qa-rail-footer").textContent = `资料已索引 · 最近更新 ${updated}`;
  $("#docs-rail-footer").textContent = `资料已索引 · 最近更新 ${updated}`;

  const qaContainer = $("#qa-doc-tree");
  qaContainer.innerHTML = buildTreeHtml(state.docsData.tree, state.qaSearch, state.previewDocId || state.selectedDocId, state.selectedFolder, "qa");
  qaContainer.querySelectorAll(".tree-file-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const docId = btn.dataset.docId;
      state.selectedDocId = docId;
      await openDocInQaPreview(docId);
      renderDocTrees();
    });
  });

  const docsContainer = $("#docs-doc-tree");
  docsContainer.innerHTML = buildTreeHtml(state.docsData.tree, state.docsSearch, state.selectedDocId, state.selectedFolder, "docs");
  docsContainer.querySelectorAll(".tree-folder-head").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedFolder = btn.dataset.folder;
      renderDocTrees();
      renderDocsTable();
    });
  });
  docsContainer.querySelectorAll(".tree-file-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      state.selectedFolder = btn.dataset.folder;
      state.selectedDocId = btn.dataset.docId;
      renderDocTrees();
      renderDocsTable();
      await loadDocDetail(state.selectedDocId);
    });
  });
}

function renderDocsTable() {
  const folder = state.selectedFolder;
  const allDocs = state.docsData.documents || [];
  const filtered = folder ? allDocs.filter((d) => d.folder === folder) : allDocs;
  $("#docs-folder-title").textContent = folder || "全部资料";
  $("#docs-folder-count").textContent = `共 ${filtered.length} 份文件`;

  const tbody = $("#docs-table-body");
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="5"><div class="empty-state-box">当前目录下暂无文件，请点击右上角“上传资料”导入工程文档。</div></td></tr>`;
    return;
  }
  tbody.innerHTML = filtered
    .map(
      (d) => `
      <tr class="${d.id === state.selectedDocId ? "active" : ""}" data-doc-id="${esc(d.id)}">
        <td>
          <div class="file-name-cell">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>
            <span>${esc(d.name)}</span>
          </div>
        </td>
        <td>${esc(d.file_type)}</td>
        <td>${esc(d.version)}</td>
        <td>${esc(d.updated_at)}</td>
        <td>${esc(d.status)}</td>
      </tr>
    `
    )
    .join("");

  tbody.querySelectorAll("tr[data-doc-id]").forEach((tr) => {
    tr.addEventListener("click", async () => {
      state.selectedDocId = tr.dataset.docId;
      renderDocTrees();
      renderDocsTable();
      await loadDocDetail(state.selectedDocId);
    });
  });
}

async function loadDocDetail(docId) {
  try {
    const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/documents/${encodeURIComponent(docId)}/preview`);
    const doc = data.document;
    $("#detail-doc-name").textContent = doc.name;
    $("#detail-open-source").href = `/api/projects/${encodeURIComponent(state.projectId)}/source/${encodeURIComponent(doc.id)}`;
    $("#detail-kv-grid").innerHTML = `
      <div class="kv-row"><dt>版本</dt><dd>${esc(doc.version)}</dd></div>
      <div class="kv-row"><dt>页数</dt><dd>${doc.pages > 0 ? esc(doc.pages) + " 页" : "—"}</dd></div>
      <div class="kv-row"><dt>更新时间</dt><dd>${esc(doc.updated_at)}</dd></div>
      <div class="kv-row"><dt>解析状态</dt><dd>${esc(doc.status)}</dd></div>
      <div class="kv-row"><dt>所属目录</dt><dd>${esc(doc.folder)}</dd></div>
      <div class="kv-row"><dt>文件大小</dt><dd>${formatSize(doc.size_bytes)}</dd></div>
    `;
    const chunk = data.active_chunk;
    const warnings = (doc.parse_warnings || []).map((message) => '<li>' + esc(message) + '</li>').join("");
    $("#detail-preview-body").innerHTML = (warnings ? '<ul class="basis-list">' + warnings + '</ul>' : "") + (chunk
      ? highlightText(chunk.content, chunk.highlight, "hl-yellow")
      : '<div class="empty-state-box">暂无可预览文本内容</div>');
  } catch {
    renderEmptyDocDetail();
  }
}

function renderEmptyDocDetail() {
  $("#detail-doc-name").textContent = "未选择文件";
  $("#detail-open-source").href = "#";
  $("#detail-kv-grid").innerHTML = "";
  $("#detail-preview-body").innerHTML = '<div class="empty-state-box">点击左侧或中间列表中的文件查看详情与内容预览。</div>';
}

function populateUploadFolders() {
  const sel = $("#upload-folder-select");
  sel.innerHTML = (state.docsData.folders || [])
    .map((f) => `<option value="${esc(f)}" ${f === state.selectedFolder ? "selected" : ""}>${esc(f)}</option>`)
    .join("");
}

/* ==================== TAB 1: QA & PREVIEW ==================== */
async function loadChatHistory() {
  const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/chat/history`);
  state.chatMessages = data.messages || [];
  renderChatFeed();
  const lastTurn = state.chatMessages[state.chatMessages.length - 1];
  if (lastTurn && lastTurn.payload && lastTurn.payload.citations && lastTurn.payload.citations.length) {
    setQaCitations(lastTurn.payload.citations, 0);
  } else if (state.docsData.documents.length) {
    state.activeCitations = [];
    renderCitationTabs();
    await openDocInQaPreview(state.docsData.documents[0].id);
  } else {
    state.activeCitations = [];
    renderCitationTabs();
    renderEmptyQaPreview();
  }
}

function renderChatFeed() {
  const feed = $("#chat-feed");
  if (!state.chatMessages.length) {
    const proj = state.projects.find((p) => p.id === state.projectId);
    feed.innerHTML = `
      <div class="empty-state-box" style="margin-top:48px;">
        <h3 style="margin:0 0 8px;color:#111827;font-size:18px;">当前项目：${esc(proj ? proj.name : state.projectId)}</h3>
        <p style="margin:0;color:#6b7280;">在下方输入工程问题（如“电控柜的防护等级要求是什么？”），系统将在当前项目的独立数据库内检索并定位原文出处。</p>
      </div>
    `;
    return;
  }
  feed.innerHTML = state.chatMessages
    .map((turn, tIdx) => {
      const p = turn.payload || {};
      const r = p.retrieval || {};
      const st = p.answer_status || (p.llm_provider && p.llm_provider !== "本地原文摘录" && p.llm_provider !== "本地检索" ? "llm_success" : "local_only");
      const diag = p.diagnostic || {};
      const stats = p.llm_stats || {};
      const scopeText = r.scope ? r.scope + " · " + (r.evidence_count ?? (p.citations || []).length) + " 条回答依据" : "";

      const statusMeta = {
        llm_success: { cls: "st-ok", label: `模型生成 · ${p.llm_provider || "在线模型"}${stats.retried ? "（经1次自纠正）" : ""}` },
        llm_call_failed: { cls: "st-err", label: "模型调用失败 · 已回退本地原文摘录" },
        verification_failed: { cls: "st-warn", label: "未通过证据校验 · 已回退本地原文摘录" },
        no_evidence: { cls: "st-muted", label: "未检索到满足阈值的证据" },
        local_only: { cls: "st-muted", label: "本地原文摘录" },
      }[st] || { cls: "st-muted", label: p.llm_provider || "本地原文摘录" };

      const statsParts = [];
      if (stats.api_calls > 0) statsParts.push(`API ${stats.api_calls} 次`);
      if (stats.total_tokens > 0) statsParts.push(`${stats.total_tokens} tokens`);
      if (stats.latency_ms > 0) statsParts.push(`${Math.round(stats.latency_ms)} ms`);
      if (p.cached) statsParts.push("缓存命中");
      const statsHtml = statsParts.length ? `<span class="answer-stats">${esc(statsParts.join(" · "))}</span>` : "";

      const diagHtml = (st === "llm_call_failed" || st === "verification_failed") && diag.message
        ? `<div class="answer-diag-box ${statusMeta.cls}">
             <div class="diag-msg">${esc(diag.message)}</div>
             ${diag.next_step ? `<div class="diag-next">建议下一步：${esc(diag.next_step)}</div>` : ""}
           </div>`
        : "";
      const verificationWarnings = (st === "llm_call_failed" || st === "verification_failed")
        ? ""
        : ((p.verification || {}).warnings || []).map((message) => '<li>' + esc(message) + '</li>').join("");

      const basisHtml = (p.basis || []).length
        ? `<div class="answer-subhead">主要依据</div>
           <ul class="basis-list">
             ${p.basis
               .map(
                 (b) => `<li><strong>${esc(b.source)}：</strong>${formatInlineMarkdown(b.detail)}</li>`
               )
               .join("")}
           </ul>`
        : "";
      const tableHtml = (p.table || []).length
        ? `<div class="answer-subhead">相关信息汇总</div>
           <table class="summary-table">
             <thead><tr><th style="width:28%;">项目</th><th>要求 / 说明</th></tr></thead>
             <tbody>
               ${p.table
                 .map(
                   (row) => `<tr><td>${esc(row.item)}</td><td>${formatInlineMarkdown(row.value)}</td></tr>`
                 )
                 .join("")}
             </tbody>
           </table>`
        : "";
      return `
        <div class="chat-turn" data-turn-idx="${tIdx}">
          <div class="user-msg-wrap">
            <div class="user-bubble">${esc(turn.question)}</div>
            <div class="msg-time">${esc(turn.time)}</div>
          </div>
          <div class="ai-answer">
            <div class="answer-head-row">
              <div class="blue-bar-title">检索结论</div>
              <div class="answer-status-meta">
                <span class="answer-status-pill ${statusMeta.cls}">${esc(statusMeta.label)}</span>
                ${statsHtml}
              </div>
            </div>
            ${scopeText ? '<p class="rail-sub">' + esc(scopeText) + '</p>' : ""}
            ${diagHtml}
            ${verificationWarnings ? '<ul class="basis-list">' + verificationWarnings + '</ul>' : ""}
            <p class="answer-paragraph">${formatInlineMarkdown(p.conclusion || "")}</p>
            ${basisHtml}
            ${tableHtml}
            <div class="answer-actions">
              <button class="btn-secondary btn-xs btn-save-turn" data-turn-idx="${tIdx}" type="button">保存到已保存</button>
            </div>
          </div>
        </div>
      `;
    })
    .join("");

  feed.querySelectorAll(".cite-link").forEach((link) => {
    link.addEventListener("click", () => {
      const turnEl = link.closest(".chat-turn");
      const tIdx = Number(turnEl.dataset.turnIdx);
      const citeNum = Number(link.dataset.cite) - 1;
      const turn = state.chatMessages[tIdx];
      if (turn && turn.payload && turn.payload.citations) {
        setQaCitations(turn.payload.citations, Math.max(0, citeNum));
      }
    });
  });

  feed.querySelectorAll(".btn-save-turn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const turn = state.chatMessages[Number(btn.dataset.turnIdx)];
      if (!turn) return;
      const p = turn.payload || {};
      const sources = (p.citations || []).map((c) => c.file_name).filter((v, i, a) => a.indexOf(v) === i).join("; ");
      await api(`/api/projects/${encodeURIComponent(state.projectId)}/saved`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: "qa",
          title: turn.question,
          summary: (p.conclusion || "").replace(/\*\*/g, "").slice(0, 90),
          sources_text: sources,
          payload: p,
        }),
      });
      await loadSavedItems();
      showToast("已保存至当前项目的“已保存”记录");
    });
  });

  feed.scrollTop = feed.scrollHeight;
}

function setQaCitations(citations, activeIdx = 0) {
  state.activeCitations = citations || [];
  state.activeCitationIndex = Math.min(activeIdx, Math.max(0, state.activeCitations.length - 1));
  renderCitationTabs();
  const c = state.activeCitations[state.activeCitationIndex];
  if (c) {
    state.previewDocId = c.doc_id;
    state.previewChunks = [c];
    state.previewChunkIdx = 0;
    renderQaPreviewCard(c.file_name, c.location || c.section, c.page, c.total_pages, c.content, c.highlight, c.doc_id);
  } else {
    state.previewChunks = [];
    state.previewDocId = null;
    renderEmptyQaPreview();
  }
}

function renderCitationTabs() {
  const box = $("#qa-citation-list");
  if (!state.activeCitations.length) {
    box.innerHTML = "";
    return;
  }
  box.innerHTML = state.activeCitations
    .map(
      (c, i) => `
      <button class="citation-tab-btn ${i === state.activeCitationIndex ? "active" : ""}" data-idx="${i}" type="button">
        <span class="citation-tab-idx">[${c.index}]</span>
        <span class="citation-tab-file">${esc(c.file_name)}</span>
        <span class="citation-tab-loc">${esc(c.location || c.section)}</span>
      </button>
    `
    )
    .join("");
  box.querySelectorAll(".citation-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      setQaCitations(state.activeCitations, Number(btn.dataset.idx));
    });
  });
}

async function openDocInQaPreview(docId) {
  const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/documents/${encodeURIComponent(docId)}/preview`);
  state.previewDocId = docId;
  state.previewChunks = data.chunks || [];
  state.previewChunkIdx = 0;
  const doc = data.document;
  const ch = state.previewChunks[0];
  if (ch) {
    renderQaPreviewCard(doc.name, ch.section, ch.page, doc.pages, ch.content, ch.highlight, doc.id);
  } else {
    renderQaPreviewCard(doc.name, "暂无文本片段", 1, doc.pages, "该文档暂无已解析文本。", "", doc.id);
  }
}

function renderQaPreviewCard(name, loc, page, totalPages, content, highlight, docId) {
  $("#qa-preview-name").textContent = name;
  $("#qa-preview-loc").textContent = loc;
  $("#qa-page-indicator").textContent = totalPages > 0 ? `${page} / ${totalPages}` : "—";
  $("#qa-preview-body").innerHTML = highlightText(content, highlight, "hl-yellow");
  $("#qa-open-full-file").href = `/api/projects/${encodeURIComponent(state.projectId)}/source/${encodeURIComponent(docId)}`;
}

function renderEmptyQaPreview() {
  $("#qa-preview-name").textContent = "暂无原文预览";
  $("#qa-preview-loc").textContent = "上传文档或提问后在此定位原文";
  $("#qa-page-indicator").textContent = "- / -";
  $("#qa-preview-body").innerHTML = '<div class="empty-state-box">当前项目数据库暂无可预览文档。</div>';
  $("#qa-open-full-file").href = "#";
}

async function handleSendQuestion() {
  const input = $("#qa-input");
  const q = input.value.trim();
  if (!q) return;
  const mode = $("#qa-mode-select").value;
  input.value = "";
  const btn = $("#btn-send-qa");
  btn.disabled = true;
  btn.textContent = "检索中…";
  try {
    const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, mode }),
    });
    state.chatMessages.push(data.turn);
    renderChatFeed();
    setQaCitations(data.turn.payload?.citations || [], 0);
  } catch (err) {
    showToast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "发送";
  }
}

/* ==================== TAB 3: COMPARE ==================== */
function initCompareDefaults() {
  const docs = state.docsData.documents || [];
  const leftSel = $("#compare-left-select");
  const rightSel = $("#compare-right-select");
  const opts = docs.map((d) => `<option value="${esc(d.id)}">${esc(d.name)} (${esc(d.version)})</option>`).join("");
  leftSel.innerHTML = opts;
  rightSel.innerHTML = opts;

  if (docs.length >= 2) {
    const clar = docs.find((d) => d.name.includes("Clarification 02")) || docs[1];
    const rfq = docs.find((d) => d.name.includes("RFQ Rev.A")) || docs[0];
    state.compareLeftId = state.compareLeftId || rfq.id;
    state.compareRightId = state.compareRightId || (clar.id === state.compareLeftId ? docs[1].id : clar.id);
    leftSel.value = state.compareLeftId;
    rightSel.value = state.compareRightId;
    runCompare();
  } else {
    $("#compare-selected-pills").innerHTML = '<div class="rail-sub">当前项目需要至少 2 份文档才能进行对照</div>';
    $("#compare-board").innerHTML = '<div class="empty-state-box">请在“资料”页上传至少两份文档后使用原文对照功能。</div>';
    $("#compare-summary-list").innerHTML = "";
    $("#compare-diff-count").textContent = "共 0 项主要差异";
  }
}

async function runCompare() {
  if (!state.compareLeftId || !state.compareRightId) return;
  try {
    const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        left_id: state.compareLeftId,
        right_id: state.compareRightId,
        keyword: state.compareKeyword,
        sections: state.compareSections,
      }),
    });
    state.compareData = data;
    renderCompareView(data);
  } catch (err) {
    showToast(err.message);
  }
}

function renderCompareView(data) {
  const lDoc = data.left_doc;
  const rDoc = data.right_doc;
  $("#compare-selected-pills").innerHTML = `
    <div class="selected-doc-pill"><span>📄 ${esc(lDoc.name)}</span></div>
    <div class="selected-doc-pill"><span>📄 ${esc(rDoc.name)}</span></div>
  `;

  const secBox = $("#compare-section-checks");
  secBox.innerHTML = (data.available_sections || [])
    .map(
      (sec) => `
      <label class="checkbox-item">
        <input type="checkbox" value="${esc(sec)}" ${state.compareSections.includes(sec) ? "checked" : ""} />
        <span>${esc(sec)}</span>
      </label>
    `
    )
    .join("");
  secBox.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", () => {
      state.compareSections = Array.from(secBox.querySelectorAll("input:checked")).map((x) => x.value);
      runCompare();
    });
  });

  const kw = data.keyword || "";
  const groupsHtml = (data.groups || []).length
    ? data.groups
        .map(
          (g) => `
        <div class="cmp-sec-banner">${esc(g.section)}</div>
        ${g.rows
          .map((row) => {
            const lText = row.left
              ? highlightText(row.left.content, kw || row.left.highlight, "hl-left")
              : '<span style="color:#9ca3af;">（当前范围无对应条款）</span>';
            const rText = row.right
              ? highlightText(row.right.content, kw || row.right.highlight, "hl-right")
              : '<span style="color:#9ca3af;">（当前范围无对应条款）</span>';
            return `<div class="cmp-row"><div class="cmp-cell">${lText}</div><div class="cmp-cell">${rText}</div></div>`;
          })
          .join("")}
      `
        )
        .join("")
    : '<div class="empty-state-box">当前筛选条件下无匹配的对照片段，可清空关键词或取消章节勾选。</div>';

  $("#compare-board").innerHTML = `
    <div class="cmp-table">
      <div class="cmp-doc-heads">
        <div class="cmp-doc-head">📄 ${esc(lDoc.name)} <span>(${esc(lDoc.version)})</span></div>
        <div class="cmp-doc-head">📄 ${esc(rDoc.name)} <span>(${esc(rDoc.version)})</span></div>
      </div>
      ${groupsHtml}
    </div>
  `;

  const summary = data.summary || [];
  $("#compare-diff-count").textContent = `共 ${summary.length} 项主要差异`;
  $("#compare-summary-list").innerHTML = summary
    .map(
      (item, idx) => `
      <div class="diff-item">
        <span class="diff-num">${idx + 1}</span>
        <div>
          <h4>${esc(item.title)}</h4>
          <p>${esc(item.detail)}</p>
        </div>
      </div>
    `
    )
    .join("");
  $("#compare-conclusion-text").textContent = data.conclusion || "";
}

/* ==================== TAB 4: SAVED ==================== */
async function loadSavedItems() {
  const url = `/api/projects/${encodeURIComponent(state.projectId)}/saved?kind=${encodeURIComponent(state.savedKind)}&keyword=${encodeURIComponent(state.savedKeyword)}`;
  const data = await api(url);
  state.savedItems = data.items || [];
  const c = data.counts || { all: 0, qa: 0, compare: 0, export: 0 };
  $("#saved-total-sub").textContent = `共 ${c.all} 条`;
  $("#cnt-all").textContent = `(${c.all})`;
  $("#cnt-qa").textContent = `(${c.qa})`;
  $("#cnt-compare").textContent = `(${c.compare})`;
  $("#cnt-export").textContent = `(${c.export})`;
  $("#saved-list-count").textContent = `共 ${state.savedItems.length} 条结果`;

  if (!state.selectedSavedId || !state.savedItems.some((x) => x.id === state.selectedSavedId)) {
    state.selectedSavedId = state.savedItems.length ? state.savedItems[0].id : null;
  }
  renderSavedCards();
  renderSavedDetail();
}

function renderSavedCards() {
  const list = $("#saved-cards-list");
  if (!state.savedItems.length) {
    list.innerHTML = '<div class="empty-state-box">当前项目暂无已保存记录。可在“问答”页点击“保存到已保存”。</div>';
    return;
  }
  list.innerHTML = state.savedItems
    .map(
      (item) => `
      <div class="saved-card ${item.id === state.selectedSavedId ? "active" : ""}" data-id="${esc(item.id)}">
        <div class="saved-card-head">
          <div class="saved-card-title">${esc(item.title)}</div>
          <div class="badge-group">
            <span class="badge badge-gray">已保存</span>
            <span class="badge badge-blue">可导出</span>
          </div>
        </div>
        <div class="saved-card-meta">已保存：${esc(item.created_at)} &nbsp;|&nbsp; 项目：${esc(item.project_name)}</div>
        <div class="saved-card-summary">${esc(item.summary)}</div>
        <div class="saved-card-source">来源：${esc(item.sources_text || "项目资料")}</div>
      </div>
    `
    )
    .join("");
  list.querySelectorAll(".saved-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedSavedId = card.dataset.id;
      renderSavedCards();
      renderSavedDetail();
    });
  });
}

function renderSavedDetail() {
  const body = $("#saved-detail-body");
  const item = state.savedItems.find((x) => x.id === state.selectedSavedId);
  if (!item) {
    body.innerHTML = '<div class="empty-state-box">选择一条已保存记录查看详情。</div>';
    return;
  }
  const p = item.payload || {};
  const basisHtml = (p.basis || []).length
    ? `<div class="answer-subhead">主要依据</div>
       <ul class="basis-list">${p.basis.map((b) => `<li><strong>${esc(b.source)}：</strong>${esc(b.detail)}</li>`).join("")}</ul>`
    : "";
  const tableHtml = (p.table || []).length
    ? `<div class="answer-subhead">相关信息汇总</div>
       <table class="summary-table">
         <thead><tr><th style="width:30%;">项目</th><th>内容</th></tr></thead>
         <tbody>${p.table.map((r) => `<tr><td>${esc(r.item)}</td><td>${esc(r.value)}</td></tr>`).join("")}</tbody>
       </table>`
    : "";
  const citesHtml = (p.citations || []).length
    ? `<div class="blue-bar-title">引用来源 (${p.citations.length})</div>
       <div class="saved-cite-list">
         ${p.citations
           .map(
             (c) => `
           <div class="saved-cite-row">
             <span>[${c.index}]</span>
             <a class="link-primary" href="/api/projects/${encodeURIComponent(state.projectId)}/source/${encodeURIComponent(c.doc_id)}" target="_blank" rel="noopener">${esc(c.file_name)}</a>
             <span style="color:#6b7280;">第 ${c.page} 页 · ${esc(c.section)}</span>
           </div>`
           )
           .join("")}
       </div>`
    : "";

  body.innerHTML = `
    <h3 class="saved-detail-title">${esc(item.title)}</h3>
    <div class="saved-card-meta">
      <span class="badge badge-gray">已保存</span>
      <span class="badge badge-blue">可导出</span>
      &nbsp;|&nbsp; 保存时间：${esc(item.created_at)} &nbsp;|&nbsp; 项目：${esc(item.project_name)}
    </div>
    <div class="blue-bar-title">检索结论</div>
    <p class="answer-paragraph">${formatInlineMarkdown(p.conclusion || item.summary)}</p>
    ${basisHtml}
    ${tableHtml}
    ${citesHtml}
  `;
}

/* ==================== TAB 5: SETTINGS ==================== */
async function loadSettings() {
  const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/settings`);
  state.settingsData = data.settings;
  state.statusData = data.status;
  syncSettingsForm();
  renderSettingsRightRail();
}

function syncSettingsForm() {
  const s = state.settingsData;
  if (!s) return;
  $("#cfg-chat-model").value = s.chat_model || "GPT-4o（企业内网）";
  $("#cfg-embed-model").value = s.embedding_model || "BAAI/bge-small-zh-v1.5（本地离线）";
  $("#cfg-retrieval-mode").value = s.retrieval_mode || "混合检索（语义 + 关键词）";
  const thr = Number(s.similarity_threshold ?? 0.62).toFixed(2);
  $("#cfg-threshold").value = thr;
  $("#cfg-threshold-val").value = thr;
  $("#cfg-top-k").value = String(s.top_k ?? 10);
  $("#cfg-ocr").checked = Boolean(s.ocr_enabled);
  $("#cfg-online-answers").checked = s.online_answers_enabled === true;
  $("#cfg-data-path").value = s.data_path || "";
  $("#cfg-chunk-size").value = s.chunk_size ?? 480;
  $("#cfg-chunk-overlap").value = s.chunk_overlap ?? 64;
}

function readSettingsFromForm() {
  return {
    chat_model: $("#cfg-chat-model").value,
    embedding_model: $("#cfg-embed-model").value,
    retrieval_mode: $("#cfg-retrieval-mode").value,
    similarity_threshold: Number($("#cfg-threshold").value),
    top_k: Number($("#cfg-top-k").value),
    ocr_enabled: $("#cfg-ocr").checked,
    online_answers_enabled: $("#cfg-online-answers").checked,
    chunk_size: Number($("#cfg-chunk-size").value),
    chunk_overlap: Number($("#cfg-chunk-overlap").value),
  };
}

function renderSettingsRightRail() {
  const s = readSettingsFromForm();
  const st = state.statusData || {};
  $("#settings-summary-kv").innerHTML = `
    <div class="kv-row"><dt>回答模型</dt><dd>${esc(s.chat_model)}</dd></div>
    <div class="kv-row"><dt>向量模型</dt><dd>${esc(s.embedding_model)}</dd></div>
    <div class="kv-row"><dt>检索方式</dt><dd>${esc(s.retrieval_mode)}</dd></div>
    <div class="kv-row"><dt>相似度阈值</dt><dd>${Number(s.similarity_threshold).toFixed(2)}</dd></div>
    <div class="kv-row"><dt>最大候选数</dt><dd>${s.top_k}</dd></div>
    <div class="kv-row"><dt>启用 OCR</dt><dd>${s.ocr_enabled ? "已启用" : "已关闭"}</dd></div>
    <div class="kv-row"><dt>数据路径</dt><dd>${esc($("#cfg-data-path").value)}</dd></div>
  `;
  $("#settings-status-list").innerHTML = `
    <li><span class="st-label">模型接口</span><span class="st-val"><i class="st-dot"></i>${esc(st.model_api || "连接正常")}</span></li>
    <li><span class="st-label">向量服务</span><span class="st-val"><i class="st-dot"></i>${esc(st.vector_service || "连接正常")}</span></li>
    <li><span class="st-label">索引服务</span><span class="st-val"><i class="st-dot"></i>${esc(st.index_service || "运行中")}</span></li>
    <li><span class="st-label">OCR 服务</span><span class="st-val"><i class="st-dot ${s.ocr_enabled ? "" : "off"}"></i>${esc(st.ocr_service || "状态未查询")}</span></li>
  `;
  $("#settings-updated-at").textContent = `最近更新：${st.updated_at || "-"}`;
}

async function saveSettings() {
  const payload = readSettingsFromForm();
  const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.settingsData = data.settings;
  state.statusData = data.status;
  renderSettingsRightRail();
  showToast("当前项目设置已保存");
}

async function triggerBackup() {
  const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/backup`, { method: "POST" });
  const b = data.backup;
  $("#backup-result-card").innerHTML = `<p><strong>最新备份已创建：</strong>${esc(b.name)}（${esc(b.size)}）</p><p>路径：${esc(b.path)}</p>`;
  showToast(`已创建项目备份：${b.name}`);
}

/* ==================== EVENT WIRING ==================== */
function bindEvents() {
  // Top nav tabs
  $$(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      state.activeTab = tab;
      $$(".nav-tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
      $$(".tab-view").forEach((v) => v.classList.toggle("active", v.id === `view-${tab}`));
    });
  });

  // Project switcher dropdown
  $("#project-trigger").addEventListener("click", (e) => {
    e.stopPropagation();
    $("#project-dropdown").classList.toggle("hidden");
  });
  document.addEventListener("click", () => $("#project-dropdown").classList.add("hidden"));
  $("#project-dropdown").addEventListener("click", (e) => e.stopPropagation());

  // Modals open/close
  $$("[data-close]").forEach((btn) => {
    btn.addEventListener("click", () => $(`#${btn.dataset.close}`).classList.add("hidden"));
  });

  $("#open-new-project-modal").addEventListener("click", () => {
    $("#project-dropdown").classList.add("hidden");
    $("#new-proj-name").value = "";
    $("#new-proj-id").value = "";
    $("#new-proj-desc").value = "";
    $("#modal-new-project").classList.remove("hidden");
  });

  $("#btn-confirm-new-project").addEventListener("click", async () => {
    const name = $("#new-proj-name").value.trim();
    const id = $("#new-proj-id").value.trim();
    const description = $("#new-proj-desc").value.trim();
    if (!name) return showToast("请输入项目名称");
    try {
      const data = await api("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, id, description }),
      });
      $("#modal-new-project").classList.add("hidden");
      await loadProjects(data.project.id);
      showToast(`已创建并切换至独立项目库：${data.project.name}`);
    } catch (err) {
      showToast(err.message);
    }
  });

  // QA interactions
  $("#qa-file-search").addEventListener("input", (e) => {
    state.qaSearch = e.target.value;
    renderDocTrees();
  });
  $("#btn-send-qa").addEventListener("click", handleSendQuestion);
  $("#qa-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSendQuestion();
  });
  $("#btn-clear-chat").addEventListener("click", async () => {
    await api(`/api/projects/${encodeURIComponent(state.projectId)}/chat/history`, { method: "DELETE" });
    state.chatMessages = [];
    renderChatFeed();
    showToast("已清空当前对话");
  });
  $("#qa-prev-page").addEventListener("click", () => {
    if (state.previewChunks.length <= 1) return;
    state.previewChunkIdx = Math.max(0, state.previewChunkIdx - 1);
    const ch = state.previewChunks[state.previewChunkIdx];
    const doc = state.docsData.documents.find((d) => d.id === state.previewDocId);
    renderQaPreviewCard(doc ? doc.name : "", ch.section, ch.page, doc?.pages, ch.content, ch.highlight, state.previewDocId);
  });
  $("#qa-next-page").addEventListener("click", () => {
    if (state.previewChunks.length <= 1) return;
    state.previewChunkIdx = Math.min(state.previewChunks.length - 1, state.previewChunkIdx + 1);
    const ch = state.previewChunks[state.previewChunkIdx];
    const doc = state.docsData.documents.find((d) => d.id === state.previewDocId);
    renderQaPreviewCard(doc ? doc.name : "", ch.section, ch.page, doc?.pages, ch.content, ch.highlight, state.previewDocId);
  });

  // Docs interactions
  $("#docs-file-search").addEventListener("input", (e) => {
    state.docsSearch = e.target.value;
    renderDocTrees();
  });
  $("#btn-refresh-docs").addEventListener("click", async () => {
    await loadDocuments();
    showToast("资料列表已刷新");
  });
  $("#btn-open-upload").addEventListener("click", () => {
    populateUploadFolders();
    $("#upload-file-input").value = "";
    $("#modal-upload").classList.remove("hidden");
  });
  $("#btn-confirm-upload").addEventListener("click", async () => {
    const files = $("#upload-file-input").files;
    if (!files || !files.length) return showToast("请选择至少一个文件");
    const fd = new FormData();
    fd.append("folder", $("#upload-folder-select").value);
    fd.append("version", $("#upload-version-input").value.trim() || "Rev.0");
    for (const f of files) fd.append("files", f);
    try {
      const data = await api(`/api/projects/${encodeURIComponent(state.projectId)}/documents/upload`, {
        method: "POST",
        body: fd,
      });
      $("#modal-upload").classList.add("hidden");
      await loadProjects(state.projectId);
      showToast(`已上传并索引 ${data.count} 份资料`);
    } catch (err) {
      showToast(err.message);
    }
  });
  $("#btn-open-folder").addEventListener("click", () => {
    $("#new-folder-name").value = "";
    $("#modal-folder").classList.remove("hidden");
  });
  $("#btn-confirm-folder").addEventListener("click", async () => {
    const name = $("#new-folder-name").value.trim();
    if (!name) return showToast("请输入文件夹名称");
    try {
      await api(`/api/projects/${encodeURIComponent(state.projectId)}/folders`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      state.selectedFolder = name;
      $("#modal-folder").classList.add("hidden");
      await loadDocuments();
      showToast(`已新建文件夹：${name}`);
    } catch (err) {
      showToast(err.message);
    }
  });
  $("#btn-reparse-doc").addEventListener("click", async () => {
    if (!state.selectedDocId) return;
    await api(`/api/projects/${encodeURIComponent(state.projectId)}/documents/${encodeURIComponent(state.selectedDocId)}/reparse`, { method: "POST" });
    await loadDocuments();
    showToast("已完成重新解析与索引更新");
  });

  // Compare interactions
  $("#compare-left-select").addEventListener("change", (e) => {
    state.compareLeftId = e.target.value;
    runCompare();
  });
  $("#compare-right-select").addEventListener("change", (e) => {
    state.compareRightId = e.target.value;
    runCompare();
  });
  $("#btn-swap-compare").addEventListener("click", () => {
    const tmp = state.compareLeftId;
    state.compareLeftId = state.compareRightId;
    state.compareRightId = tmp;
    $("#compare-left-select").value = state.compareLeftId;
    $("#compare-right-select").value = state.compareRightId;
    runCompare();
  });
  $("#btn-run-compare").addEventListener("click", () => {
    state.compareKeyword = $("#compare-kw-top").value.trim() || $("#compare-kw-left").value.trim();
    runCompare();
  });
  $("#compare-kw-left").addEventListener("input", (e) => {
    state.compareKeyword = e.target.value;
    $("#compare-kw-top").value = e.target.value;
  });
  $("#compare-kw-left").addEventListener("keydown", (e) => { if (e.key === "Enter") runCompare(); });
  $("#compare-kw-top").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      state.compareKeyword = e.target.value;
      $("#compare-kw-left").value = e.target.value;
      runCompare();
    }
  });
  $$("#compare-kw-tags .kw-tag").forEach((tag) => {
    tag.addEventListener("click", () => {
      const kw = tag.dataset.kw;
      state.compareKeyword = kw;
      $("#compare-kw-left").value = kw;
      $("#compare-kw-top").value = kw;
      runCompare();
    });
  });

  // Saved interactions
  $$("#saved-kind-menu .side-menu-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.savedKind = btn.dataset.kind;
      $$("#saved-kind-menu .side-menu-item").forEach((b) => b.classList.toggle("active", b === btn));
      loadSavedItems();
    });
  });
  $("#saved-keyword-input").addEventListener("input", (e) => {
    state.savedKeyword = e.target.value;
    loadSavedItems();
  });
  $("#btn-export-xlsx").addEventListener("click", () => {
    if (!state.selectedSavedId) return showToast("请先选择一条记录");
    window.open(`/api/projects/${encodeURIComponent(state.projectId)}/saved/${encodeURIComponent(state.selectedSavedId)}/export`, "_blank");
  });
  $("#btn-copy-saved").addEventListener("click", async () => {
    const item = state.savedItems.find((x) => x.id === state.selectedSavedId);
    if (!item) return;
    const text = `${item.title}\n\n${(item.payload && item.payload.conclusion) || item.summary}\n来源：${item.sources_text}`;
    try {
      await navigator.clipboard.writeText(text);
      showToast("已复制到剪贴板");
    } catch {
      showToast("复制失败，请手动选中复制");
    }
  });
  $("#btn-goto-source").addEventListener("click", () => {
    const item = state.savedItems.find((x) => x.id === state.selectedSavedId);
    const firstCite = item && item.payload && item.payload.citations && item.payload.citations[0];
    if (firstCite) {
      window.open(`/api/projects/${encodeURIComponent(state.projectId)}/source/${encodeURIComponent(firstCite.doc_id)}`, "_blank");
    } else {
      showToast("该记录暂无可跳转原文");
    }
  });

  // Settings interactions
  $$("#settings-menu .side-menu-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sec = btn.dataset.sec;
      $$("#settings-menu .side-menu-item").forEach((b) => b.classList.toggle("active", b === btn));
      $$(".settings-panel").forEach((p) => p.classList.toggle("active", p.id === `settings-sec-${sec}`));
    });
  });
  $("#cfg-threshold").addEventListener("input", (e) => {
    $("#cfg-threshold-val").value = Number(e.target.value).toFixed(2);
    renderSettingsRightRail();
  });
  ["#cfg-chat-model", "#cfg-embed-model", "#cfg-retrieval-mode", "#cfg-top-k", "#cfg-ocr"].forEach((sel) => {
    $(sel).addEventListener("change", renderSettingsRightRail);
  });
  $("#btn-save-settings").addEventListener("click", saveSettings);
  $("#btn-save-settings-2").addEventListener("click", saveSettings);
  $("#btn-reset-settings").addEventListener("click", () => {
    state.settingsData = {
      ...state.settingsData,
      chat_model: "GPT-4o（企业内网）",
      embedding_model: "BAAI/bge-small-zh-v1.5（本地离线）",
      retrieval_mode: "混合检索（语义 + 关键词）",
      similarity_threshold: 0.62,
      top_k: 10,
      ocr_enabled: true,
      online_answers_enabled: false,
    };
    syncSettingsForm();
    renderSettingsRightRail();
    showToast("已恢复默认参数（点击“保存设置”生效）");
  });
  $("#btn-copy-path").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("#cfg-data-path").value);
      showToast("已复制当前项目数据库路径");
    } catch {}
  });
  $("#btn-create-backup").addEventListener("click", triggerBackup);
  $("#btn-backup-center").addEventListener("click", triggerBackup);
}

/* ==================== BOOT ==================== */
document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadProjects("rfq-demo").catch((err) => showToast(`初始化失败：${err.message}`));
});
