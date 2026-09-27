/* PersonalAngel operator console — vanilla JS, no build step, no external assets.
   Talks to the local backend over fetch + Server-Sent Events. */
(() => {
  const $ = (id) => document.getElementById(id);
  const KIND_COLOR = {
    fall: "var(--fall)", person_down: "var(--fall)", slump_unresponsive: "var(--fall)", distress_speech: "var(--speech)",
    weapon_visible: "var(--weapon)", aggressive_interaction: "var(--aggr)", threatening_speech: "var(--weapon)", infant_distress: "var(--speech)",
    hateful_speech: "var(--warn)", acoustic_alarm: "var(--sound)", normal_activity: "var(--normal)", scene_change: "var(--muted)",
  };
  const STAGES = ["INGESTED", "SCENE", "SCANNING", "CANDIDATES", "EVENTS", "INVESTIGATING", "DONE"];
  const state = { mediaId: null, mediaPath: null, runId: null, es: null, duration: 0, evidence: {}, events: [], windows: null,
                  clips: [], steps: {}, thinkBuf: "", countdownTimer: null, mediaKind: "video", originalUrl: null, health: {} };

  // ------------------------------------------------------------ bootstrap
  async function init() {
    const h = await fetch("/api/health").then(r => r.json()).catch(() => ({}));
    state.health = h;
    $("pill-profile").textContent = `profile: ${h.profile || "?"}`;
    const llm = h.llm || {}, models = h.models || {};
    const name = (models.llm || llm.backend || "?").replace("openai_compatible:", "");
    $("pill-llm").textContent = `model: ${name} ${llm.ok ? "● ready" : "○ not reachable"}`;
    $("pill-llm").className = "pill " + (llm.ok ? "ok" : "bad");
    $("pill-llm").title = llm.ok ? `served locally by ${llm.backend}` : (llm.error || "start the local model server (Ollama / vLLM)");
    $("pill-local").className = "pill " + (h.local_only ? "ok" : "warn");
    renderWarm(h.warm);
    if ((models.planner || "") === "llm") { const d = document.querySelector('input[name="depth"][value="deep"]'); if (d) d.checked = true; }
    const st = h.scene_selftest || {};
    if (st.ok === false) { const p = document.createElement("span"); p.className = "pill bad"; p.textContent = "scene model: error"; p.title = st.error || ""; $("pill-sim").before(p); }
    else if (st.backend && st.backend !== "fixture") { const p = document.createElement("span"); p.className = "pill ok"; p.textContent = `scene model ● ${st.ms ? st.ms + " ms" : "ok"}`; $("pill-sim").before(p); }
    if (h.simulate_actions === false) { $("pill-sim").textContent = "actions: LIVE"; $("pill-sim").className = "pill bad"; }
    netStatus();
    window.addEventListener("online", netStatus); window.addEventListener("offline", netStatus);
    $("backends").textContent = ["detector", "pose", "audio", "scene"].map(k => `${k}: ${models[k] || "?"}`).join(" · ");
    refreshRuns();
  }
  function renderWarm(w) {
    const el = $("pill-warm"); if (!el) return;
    const st = (w && w.state) || "cold";
    if (st.startsWith("n/a")) { el.style.display = "none"; return; }
    const n = w && w.count ? ` (${w.count})` : "";
    el.textContent = st === "loading" ? "models: loading into memory…" : st === "resident" ? `models resident in memory${n}` : st === "partial" ? `models: partly loaded${n}` : st === "failed" ? "models: warm-up failed" : "models: cold";
    el.className = "pill " + (st === "resident" ? "ok" : st === "loading" ? "warn" : st === "cold" ? "" : "bad");
    if (w && w.errors && Object.keys(w.errors).length) el.title = Object.entries(w.errors).map(([k, v]) => `${k}: ${v}`).join("\n");
    if (st === "loading" || st === "cold") setTimeout(async () => { const h = await fetch("/api/health").then(r => r.json()).catch(() => null); if (h) renderWarm(h.warm); }, 4000);
  }
  function netStatus() {
    const on = navigator.onLine;
    $("pill-net").textContent = on ? "internet: on (not used)" : "internet: off — still fully operational";
    $("pill-net").className = "pill " + (on ? "" : "ok");
  }

  // ------------------------------------------------------------ input
  const drop = $("drop"), fileInput = $("file");
  drop.onclick = () => fileInput.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); };
  fileInput.onchange = () => fileInput.files[0] && upload(fileInput.files[0]);
  async function upload(file) {
    $("drop-text").textContent = `uploading ${file.name}…`;
    const fd = new FormData(); fd.append("file", file);
    const r = await fetch("/api/upload", { method: "POST", body: fd }).then(r => r.json());
    if (r.error) { $("drop-text").textContent = "✖ " + r.error; return; }
    state.mediaId = r.media_id; state.mediaPath = null;
    drop.classList.add("loaded");
    $("drop-text").innerHTML = `✔ ${esc(file.name)} <span class="muted">(${r.kind}, ${(file.size / 1e6).toFixed(1)} MB)</span>`;
    showMedia(r.url, r.kind);
  }
  function pickLibrary(item) {
    state.mediaPath = item.path; state.mediaId = null;
    drop.classList.add("loaded");
    $("drop-text").innerHTML = `✔ ${esc(item.name)} <span class="muted">(${item.kind}${item.source ? ", " + esc(item.source) : ""})</span>`;
    showMedia(item.url, item.kind);
    $("library").hidden = true;
  }

  function showMedia(url, kind, keepTabs) {
    state.mediaKind = kind;
    if (!keepTabs) { state.originalUrl = url; renderTabs(); }
    const p = $("player"); p.innerHTML = "";
    let el;
    if (kind === "audio") { el = document.createElement("audio"); el.controls = true; }
    else if (kind === "image") { el = document.createElement("img"); }
    else { el = document.createElement("video"); el.controls = true; el.muted = kind !== "clip"; el.playsInline = true; }
    el.src = url; p.appendChild(el);
    if (el.addEventListener && kind !== "image") {
      el.addEventListener("loadedmetadata", () => { if (kind !== "clip") { state.duration = el.duration || state.duration; drawTimeline(); } });
      el.addEventListener("timeupdate", () => { if (state.duration && kind !== "clip") $("tl-cursor").style.left = (100 * el.currentTime / state.duration) + "%"; });
    }
  }
  function renderTabs() {
    const box = $("media-tabs"); box.innerHTML = "";
    const mk = (label, on, active) => { const b = document.createElement("button"); b.className = "tab" + (active ? " active" : ""); b.textContent = label;
      b.onclick = () => { [...box.children].forEach(x => x.classList.remove("active")); b.classList.add("active"); on(); }; box.appendChild(b); };
    mk("Original", () => showMedia(state.originalUrl, state.mediaKind === "clip" ? "video" : state.mediaKind, true), true);
    for (const c of state.clips) mk(`Evidence replay ${fmt(c.start_s)}–${fmt(c.end_s)}`, () => { const v = showMedia(c.url, "clip", true); }, false);
  }
  function seek(t) { const el = $("player").firstElementChild; if (el && "currentTime" in el) { el.currentTime = t; el.play && el.play().catch(() => {}); } }

  // ------------------------------------------------------------ run
  $("analyze").onclick = async () => {
    if (!state.mediaId && !state.mediaPath) { $("ingest").textContent = "Drop a recording first (or pick an example)."; return; }
    resetPanels();
    $("analyze").disabled = true; $("analyze").textContent = "Investigating…";
    const depthEl = document.querySelector('input[name="depth"]:checked');
    const body = { question: $("question").value, context: $("context").value || "", depth: depthEl ? depthEl.value : "fast" };
    if (state.mediaId) body.media_id = state.mediaId; else body.media_path = state.mediaPath;
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json());
    if (r.error) { $("ingest").textContent = r.error; $("analyze").disabled = false; $("analyze").textContent = "Investigate"; return; }
    state.runId = r.run_id;
    if (r.media_url) showMedia(r.media_url, r.kind);
    listen(r.run_id);
    refreshRuns();
  };

  function resetPanels() {
    state.evidence = {}; state.events = []; state.windows = null; state.clips = []; state.steps = {}; state.thinkBuf = "";
    $("trace").innerHTML = ""; $("events").innerHTML = ""; $("gallery").innerHTML = ""; $("transcript").innerHTML = "";
    $("scene-body").innerHTML = `<span class="muted">Inferring the environment…</span>`; $("scene-backend").textContent = "";
    $("final-card").hidden = true; $("final-card").className = "card verdict"; $("question-card").hidden = true; $("decisions").innerHTML = ""; $("chat-a").textContent = "";
    $("hyp-text").textContent = "Perceiving…"; setBelief(null); $("critic-badge").textContent = "critic: not run"; $("budget").textContent = "";
    [...$("stages").children].forEach(s => s.className = "");
    $("timeline").querySelectorAll(".tl-window,.tl-event").forEach(n => n.remove());
    renderTabs();
    for (const id of ["t-wall", "t-calls", "t-tokens", "t-tps", "t-frames", "t-skip", "t-gpu", "t-tools", "t-cloud", "t-local", "t-edge", "t-sec"]) $(id).textContent = "—";
  }

  function listen(runId) {
    if (state.es) state.es.close();
    const es = new EventSource(`/api/runs/${runId}/events`);
    state.es = es;
    es.onmessage = (m) => { if (!m.data) return; handle(JSON.parse(m.data)); };
    es.addEventListener("end", () => { es.close(); $("analyze").disabled = false; $("analyze").textContent = "Investigate"; setStage("DONE"); refreshRuns(); document.dispatchEvent(new Event("angel:run-end")); });
    es.onerror = () => { /* keepalive gaps are normal */ };
  }

  function setStage(name) {
    const idx = STAGES.indexOf(name === "AUDIO" ? "SCANNING" : name);
    if (idx < 0) return;
    [...$("stages").children].forEach((s, i) => { s.className = i < idx ? "done" : i === idx ? "active" : ""; });
  }
  function setBelief(p) {
    if (p == null) { $("meter-fill").style.width = "0%"; $("meter-label").textContent = "belief —"; return; }
    $("meter-fill").style.width = Math.round(p * 100) + "%"; $("meter-label").textContent = `belief in hypothesis: ${Math.round(p * 100)}%`;
  }

  // ------------------------------------------------------------ SSE handlers
  function handle(ev) {
    switch (ev.type) {
      case "run": addLine(`run ${ev.run_id} · ${ev.profile} · ${ev.llm}`); break;
      case "stage": setStage(ev.stage); addLine(`[${ev.stage}] ${ev.detail || ""}`); if (ev.windows) { state.windows = ev.windows; drawWindows(ev.windows); } if (ev.scene) renderScene(ev.scene); break;
      case "events": onEvents(ev); break;
      case "hypothesis": $("hyp-text").textContent = (ev.revised ? "↻ revised: " : "") + ev.hypothesis.statement; setBelief(ev.hypothesis.probability); break;
      case "thinking": onThinking(ev.delta); break;
      case "reasoning": onReasoning(ev); break;
      case "thought": onThought(ev); break;
      case "belief": onBelief(ev); break;
      case "observation": onObservation(ev); break;
      case "await_answer": onAwaitAnswer(ev.question); break;
      case "question": onQuestionAnswered(ev); break;
      case "decision": onDecision(ev.decision); break;
      case "warning": addLine("⚠ " + ev.detail, "warn-line"); break;
      case "error": addLine("✖ " + ev.detail, "err-line"); $("analyze").disabled = false; $("analyze").textContent = "Investigate"; break;
      case "final": onFinal(ev.report); break;
    }
  }
  function addLine(text, cls) { const d = document.createElement("div"); d.className = "line " + (cls || ""); d.textContent = text; $("trace").appendChild(d); scrollTrace(); }
  function scrollTrace() { const t = $("trace"); t.scrollTop = t.scrollHeight; }

  // ---- scene ------------------------------------------------------------------
  function renderScene(sc) {
    $("scene-backend").textContent = sc.backend && sc.backend !== "none" ? `${sc.backend} · ${sc.frames_used || 0} frames` : "";
    const people = (sc.people || []).map(p => `<div><span>${p.track_id != null ? "PERSON_" + String(p.track_id).padStart(2, "0") : "person"}</span><span>${p.age_group === "unknown" ? `<span class="muted" title="${esc(p.note || "figure too small to judge age")}">age not judged (too small)</span>` : `${esc(p.age_group)} <span class="muted">${Math.round(p.confidence * 100)}%</span>`}</span></div>`).join("") || `<div class="muted">no person profiled yet</div>`;
    const objs = (sc.object_checks || []).map(o => { const cls = o.verdict === "weapon" ? "weapon" : o.verdict === "uncertain" ? "uncertain" : "benign";
      return `<div><span>${esc(o.label)} @ frame ${o.frame_index}</span><span class="verdict-chip ${cls}">${esc(o.verdict)} · weapon p=${o.weapon_probability.toFixed(2)}</span></div>`; }).join("") || `<div class="muted">no weapon-like objects to second-guess</div>`;
    const alt = Object.entries(sc.scores || {}).sort((a, b) => b[1] - a[1]).slice(1, 3).map(([k, v]) => `${k.toLowerCase().replace(/_/g, " ")} ${Math.round(v * 100)}%`).join(" · ");
    const act = Object.entries(sc.activity || {}).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => `<span class="${(k === "fight" || k === "shove" || k === "weapon") && v >= 0.4 ? "hot" : ""}">${esc(k)} ${Math.round(v * 100)}%</span>`).join("");
    const extra = (sc.error ? `<div class="warn">scene model error: ${esc(sc.error)} — falling back to the vision model's description</div>` : "") + (sc.vlm_scene ? `<div class="muted small">vision model: “${esc(sc.vlm_scene)}”</div>` : "");
    $("scene-body").innerHTML = `
      <div class="kv"><b>Environment (inferred)</b><div class="big">${esc(sc.label || sc.location || "unknown")}</div>
        <div class="bar"><i style="width:${Math.round((sc.confidence || 0) * 100)}%"></i></div>
        <div class="muted small">confidence ${Math.round((sc.confidence || 0) * 100)}%${alt ? " · also considered: " + esc(alt) : ""}</div>
        ${sc.operator_note ? `<div class="muted small">operator note: “${esc(sc.operator_note)}”</div>` : ""}</div>
      <div class="kv"><b>People (age group)</b><div class="list">${people}</div></div>
      <div class="kv"><b>Activity (zero-shot, per window)</b><div class="act">${act || '<span class="muted">not computed</span>'}</div></div>
      <div class="kv"><b>Weapon candidates — second opinion before the VLM</b><div class="list">${objs}</div></div>
      ${extra ? `<div style="grid-column: 1 / -1">${extra}</div>` : ""}`;
  }

  function onEvents(ev) {
    state.evidence = ev.evidence; state.events = ev.events; state.duration = ev.duration_s || state.duration; state.clips = ev.clips || [];
    if (ev.scene) renderScene(ev.scene);
    const b = ev.backends || {};
    $("backends").textContent = Object.entries(b).map(([k, v]) => `${k}: ${v}`).join(" · ");
    $("audio-backend").textContent = b.audio ? `${b.audio}` : "";
    renderTabs();
    const box = $("events"); box.innerHTML = "";
    for (const e of ev.events) {
      const d = document.createElement("div"); d.className = "event"; d.style.borderLeftColor = KIND_COLOR[e.kind] || "var(--accent)";
      const age = e.attributes && (e.attributes.age_group || e.attributes.holder_age_group);
      d.innerHTML = `<div class="kind" style="color:${KIND_COLOR[e.kind] || "var(--accent)"}">${e.kind.replace(/_/g, " ")}</div>
        <div class="meta">${esc(e.subject)}${age ? " (" + esc(age) + ")" : ""} → ${esc(e.action)}${e.obj ? " → " + esc(e.obj) : ""} · ${fmt(e.start_s)}–${fmt(e.end_s)} · conf ${e.confidence.toFixed(2)} · severity ${e.severity.toFixed(2)}</div>
        <div class="summary">${esc(e.summary)}</div>`;
      d.onclick = () => seek(e.start_s);
      box.appendChild(d);
    }
    const g = $("gallery"); g.innerHTML = "";
    const frames = Object.values(ev.evidence).filter(x => x.kind === "frame" && x.url).sort((a, b) => a.start_s - b.start_s);
    for (const f of frames.slice(0, 18)) {
      const w = document.createElement("div");
      w.innerHTML = `<img src="${f.url}" title="${esc(f.description)}" loading="lazy"><div class="cap">${fmt(f.start_s)} · ${esc(f.description.slice(0, 48))}</div>`;
      w.querySelector("img").onclick = () => seek(f.start_s);
      g.appendChild(w);
    }
    const tr = $("transcript"); tr.innerHTML = "";
    const audio = Object.values(ev.evidence).filter(x => x.kind === "audio").sort((a, b) => a.start_s - b.start_s);
    for (const a of audio) {
      const seg = a.payload || {};
      const d = document.createElement("div"); d.className = "seg";
      d.innerHTML = `<b>${fmt(a.start_s)}–${fmt(a.end_s)} [${esc(seg.language || "?")}]</b> ${esc(seg.text || "")}` + (seg.translation_en && seg.language !== "en" ? `<div class="en">EN: ${esc(seg.translation_en)}</div>` : "");
      tr.appendChild(d);
    }
    const ac = Object.values(ev.evidence).filter(x => x.kind === "acoustic");
    if (ac.length) { const d = document.createElement("div"); d.innerHTML = ac.map(a => `<span class="acoustic">${esc(a.description.replace("Acoustic event: ", ""))} @ ${fmt(a.start_s)}</span>`).join(""); tr.appendChild(d); }
    const risk = Object.values(ev.evidence).find(x => x.kind === "text");
    if (risk) { const d = document.createElement("div"); d.className = "risk"; d.textContent = risk.description; tr.appendChild(d); }
    if (!audio.length && !ac.length) tr.innerHTML = `<span class="muted">no speech or salient sounds</span>`;
    drawTimeline();
  }

  function drawWindows(windows) {
    if (!state.duration) return;
    const tl = $("timeline"); tl.querySelectorAll(".tl-window").forEach(n => n.remove());
    for (const w of windows) { const d = document.createElement("div"); d.className = "tl-window"; d.style.left = pct(w.start_s) + "%"; d.style.width = Math.max(0.5, pct(w.end_s) - pct(w.start_s)) + "%"; d.title = w.reasons.join(", "); tl.appendChild(d); }
  }
  function drawTimeline() {
    if (!state.duration) return;
    if (state.windows) drawWindows(state.windows);
    const tl = $("timeline"); tl.querySelectorAll(".tl-event").forEach(n => n.remove());
    const legend = new Map();
    for (const e of state.events) {
      const d = document.createElement("div"); d.className = "tl-event"; d.style.left = pct(e.start_s) + "%";
      d.style.width = Math.max(0.8, pct(e.end_s) - pct(e.start_s)) + "%"; d.style.background = KIND_COLOR[e.kind] || "var(--accent)";
      d.title = `${e.kind} ${fmt(e.start_s)}–${fmt(e.end_s)}`; d.onclick = () => seek(e.start_s); tl.appendChild(d);
      legend.set(e.kind, KIND_COLOR[e.kind] || "var(--accent)");
    }
    $("legend").innerHTML = [...legend].map(([k, c]) => `<span><i style="background:${c}"></i>${k.replace(/_/g, " ")}</span>`).join("") + `<span><i style="background:rgba(76,201,240,.3)"></i>candidate window (dense inspection)</span>`;
  }
  const pct = (t) => Math.min(100, Math.max(0, 100 * t / (state.duration || 1)));

  // ---- trace ------------------------------------------------------------
  function onThinking(delta) {
    if (!$("show-think").checked) return;
    let live = $("live-think");
    if (!live) { live = document.createElement("details"); live.id = "live-think"; live.className = "think"; live.open = true; live.innerHTML = `<summary>model thinking (live)</summary><div class="body"></div>`; $("trace").appendChild(live); }
    state.thinkBuf += delta; live.querySelector(".body").textContent = state.thinkBuf.slice(-3000); scrollTrace();
  }
  function onReasoning(ev) {
    const live = $("live-think"); if (live) live.remove();
    state.thinkBuf = "";
    if (!$("show-think").checked) return;
    const d = document.createElement("details"); d.className = "think"; d.innerHTML = `<summary>model thinking · step ${ev.step}</summary><div>${esc(ev.text)}</div>`; $("trace").appendChild(d);
  }
  function onThought(ev) {
    const live = $("live-think"); if (live) live.remove(); state.thinkBuf = "";
    const cls = ev.action === "run_critic" ? " critic" : (ev.action === "propose_action" || ev.action === "ask_user") ? " policy" : ev.action === "finalize" ? " final" : "";
    const d = document.createElement("div"); d.className = "step" + cls; d.id = "step-" + ev.step;
    const args = ev.action === "finalize" ? { recommended_action: ev.action_input.recommended_action, probability: ev.action_input.probability } : ev.action_input;
    d.innerHTML = `<div class="head"><span>step ${ev.step} · master agent</span><span class="lat"></span></div>
      <div class="thought">${esc(ev.thought)}</div>
      <div class="action">${esc(ev.action)}(${esc(JSON.stringify(args)).slice(0, 220)})</div>`;
    $("trace").appendChild(d); state.steps[ev.step] = d; scrollTrace();
  }
  function onBelief(ev) {
    setBelief(ev.after);
    const d = state.steps[ev.step]; if (!d) return;
    const b = document.createElement("div"); b.className = "belief";
    b.textContent = `belief ${Math.round(ev.before * 100)}% → ${Math.round(ev.after * 100)}% (${ev.source} support ${ev.support >= 0 ? "+" : ""}${ev.support.toFixed(2)})`;
    d.appendChild(b);
    if (ev.source === "run_critic") { const v = (ev.support > 0 ? "supported" : ev.support > -1 ? "weakened" : "refuted"); $("critic-badge").textContent = "critic: " + v; }
  }
  function onObservation(ev) {
    const d = state.steps[ev.step]; if (!d) return;
    const o = document.createElement("div"); o.className = "obs"; o.textContent = ev.observation; d.appendChild(o);
    if (ev.image_urls && ev.image_urls.length) { const im = document.createElement("div"); im.className = "images"; im.innerHTML = ev.image_urls.map(u => `<img src="${u}">`).join(""); d.appendChild(im); }
    d.querySelector(".lat").textContent = `${(ev.latency_ms / 1000).toFixed(1)} s · cost ${ev.cost}`;
    $("budget").textContent = `compute ${Math.round(ev.compute_spent)} / ${Math.round(ev.compute_budget)} units`;
    scrollTrace();
  }

  // ---- ask user ---------------------------------------------------------------
  function onAwaitAnswer(q) {
    const card = $("question-card"); card.hidden = false;
    $("q-text").textContent = q.text; $("q-meta").textContent = `spoken through the cabin/room speaker · language: ${q.language} · timeout ${q.timeout_s}s · silence → "${q.default_if_silent}"`;
    $("answer").value = ""; $("answer").focus();
    const quick = $("quick-answers"); quick.innerHTML = "";
    const presets = q.language === "es" ? ["Sí, por favor", "No, estoy bien", "¡Ayuda! Me duele mucho", "(stay silent)"] : ["Yes please", "No, I'm fine", "Help, it hurts", "(stay silent)"];
    for (const p of presets) { const c = document.createElement("span"); c.className = "chip"; c.textContent = p; c.onclick = () => sendAnswer(p.startsWith("(") ? "" : p); quick.appendChild(c); }
    if (/0 to 10|0 al 10|0-10/.test(q.text)) {
      const row = document.createElement("div"); row.className = "chips pain";
      row.innerHTML = `<span class="muted small" style="align-self:center">pain:</span>`;
      for (const n of [0, 2, 4, 5, 6, 8, 10]) { const c = document.createElement("span"); c.className = "chip num" + (n >= 7 ? " hi" : n >= 4 ? " mid" : ""); c.textContent = String(n); c.onclick = () => sendAnswer(String(n)); row.appendChild(c); }
      quick.after(row);
    }
    let left = Math.round(q.timeout_s);
    clearInterval(state.countdownTimer);
    $("countdown").textContent = `waiting for an answer… ${left}s`;
    state.countdownTimer = setInterval(() => { left -= 1; $("countdown").textContent = left > 0 ? `waiting for an answer… ${left}s` : "no answer — the agent treats silence as unresponsive"; if (left <= 0) clearInterval(state.countdownTimer); }, 1000);
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  async function sendAnswer(text) {
    if (!state.runId) return;
    await fetch(`/api/runs/${state.runId}/answer`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ answer: text }) });
    clearInterval(state.countdownTimer); $("countdown").textContent = text ? `answered: "${text}"` : "silence sent";
  }
  $("answer-send").onclick = () => sendAnswer($("answer").value.trim());
  $("answer").addEventListener("keydown", (e) => { if (e.key === "Enter") sendAnswer($("answer").value.trim()); });
  function onQuestionAnswered(ev) { clearInterval(state.countdownTimer); $("countdown").textContent = `→ "${ev.question.answer}" (${ev.normalized})`; setTimeout(() => { $("question-card").hidden = true; document.querySelectorAll(".chips.pain").forEach(n => n.remove()); }, 2500); }

  // ---- decisions / final --------------------------------------------------------
  function onDecision(d) {
    const box = $("decisions");
    const el = document.createElement("div"); el.className = "decision " + (d.executed ? "exec" : "rej");
    const dispatch = d.params && d.params.dispatch ? JSON.stringify(d.params.dispatch) : "";
    el.innerHTML = `<b>${esc(d.action.replace(/_/g, " "))}</b><span class="tag">${d.executed ? (d.simulated ? "simulated dispatch" : "executed") : "rejected by policy gate"}</span>
      <div class="muted">${esc(d.rationale || d.result || "")}</div>${dispatch ? `<div class="muted small">${esc(dispatch.slice(0, 260))}</div>` : ""}
      <div class="muted small">risk if wrong ${d.risk_if_wrong} · expected benefit ${d.expected_benefit}</div>`;
    box.appendChild(el);
  }
  function onFinal(r) {
    const v = r.verdict || { level: "clear", headline: "" };
    const card = $("final-card"); card.hidden = false; card.className = "card verdict " + v.level;
    $("verdict-level").textContent = v.level === "alert" ? "ALERT" : v.level === "watch" ? "WATCH" : "CLEAR";
    $("verdict-headline").textContent = v.headline || "";
    $("final-text").innerHTML = (r.final_answer || "").split("\n").map(line => {
      const m = line.match(/^([A-Z][A-Z ]{3,}) — (.*)$/);
      return m ? `<div class="rline"><b>${esc(m[1])}</b><span>${esc(m[2])}</span></div>` : `<div>${esc(line)}</div>`;
    }).join("");
    $("final-unc").textContent = r.final_uncertainty ? "Uncertainty: " + r.final_uncertainty : "";
    setBelief(r.belief); if (r.hypothesis) { $("hyp-text").textContent = r.hypothesis.statement; if (r.hypothesis.critic_verdict) $("critic-badge").textContent = "critic: " + r.hypothesis.critic_verdict; }
    if (r.scene) renderScene(r.scene);
    if (r.clips && r.clips.length) { state.clips = r.clips; renderTabs(); }
    const t = r.telemetry || {};
    $("t-wall").textContent = t.wall_time_s + " s"; $("t-calls").textContent = t.model_calls;
    $("t-tokens").textContent = `${t.input_tokens} / ${t.output_tokens}`; $("t-tps").textContent = t.tokens_per_s ?? "n/a";
    $("t-frames").textContent = `${t.frames_processed} / ${t.frames_total}`; $("t-skip").textContent = `${t.frames_skipped} (${t.frame_skip_ratio != null ? Math.round(t.frame_skip_ratio * 100) + "%" : "—"})`;
    $("t-gpu").textContent = t.gpu ? `${t.gpu.util_mean}% / ${Math.round(t.gpu.mem_used_mb_max / 1024 * 10) / 10} GB` : "not sampled here";
    $("t-tools").textContent = `${t.tool_calls} · ${t.critic_calls}`;
    $("t-cloud").textContent = `$${t.cost.cloud_equivalent_usd}`; $("t-local").textContent = `$${t.cost.local_energy_cost_usd} (${t.cost.local_energy_kwh} kWh)`;
    const ec = r.edge_cloud || {}; $("t-edge").textContent = ec.summary ? `${ec.summary.local} local · ${ec.summary.cloud_eligible} cloud-eligible (${ec.mode})` : "—";
    const sec = r.security || {}; $("t-sec").textContent = `${Object.values(sec.tool_calls || {}).reduce((a, b) => a + b, 0)} gated calls · ${(sec.injection_flags || []).length} injection flags`;
    addLine("✔ investigation complete — report.json written");
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---- follow-up chat -------------------------------------------------------------
  $("chat-send").onclick = async () => {
    if (!state.runId) return; const q = $("chat-q").value.trim(); if (!q) return;
    const a = $("chat-a"); a.className = "chat-a wait"; const t0 = Date.now();
    a.textContent = "the local model is reading the investigation… 0 s";
    const tick = setInterval(() => { a.textContent = `the local model is reading the investigation… ${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
    $("chat-send").disabled = true;
    try {
      const res = await fetch(`/api/runs/${state.runId}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
      const r = await res.json();
      clearInterval(tick);
      if (r.error) { a.className = "chat-a err"; a.textContent = r.error; }
      else { a.className = "chat-a"; a.textContent = r.answer + (r.cache && r.cache !== "miss" ? `\n\n(answered from the semantic cache: ${r.cache}${r.similarity ? ", similarity " + r.similarity : ""})` : (r.seconds ? `\n\n(${r.seconds} s, local model)` : "")); }
    } catch (e) { clearInterval(tick); a.className = "chat-a err"; a.textContent = "request failed: " + e; }
    $("chat-send").disabled = false;
  };
  $("chat-q").addEventListener("keydown", (e) => { if (e.key === "Enter") $("chat-send").click(); });

  async function refreshRuns() {
    const r = await fetch("/api/runs").then(r => r.json()).catch(() => ({ runs: [] }));
    const ul = $("runs"); ul.innerHTML = "";
    if (!r.runs.length) { ul.innerHTML = `<li class="muted" style="cursor:default">none yet</li>`; return; }
    for (const run of r.runs.slice().reverse()) {
      const li = document.createElement("li"); const v = run.verdict || {};
      li.innerHTML = `<span>${esc(run.run_id)}</span><span class="v ${v.level || ""}">${v.level || run.status}</span>`;
      li.className = run.run_id === state.runId ? "active" : "";
      li.onclick = () => { state.runId = run.run_id; resetPanels(); listen(run.run_id); };
      ul.appendChild(li);
    }
  }

  // ---- library ----------------------------------------------------------------------
  $("open-library").onclick = async () => {
    const r = await fetch("/api/library").then(r => r.json()).catch(() => ({ items: [] }));
    const box = $("library-items"); box.innerHTML = "";
    if (!r.items.length) box.innerHTML = `<div class="muted">No example recordings yet — run <code>python scripts/fetch_real_demo_clips.py</code>.</div>`;
    for (const it of r.items) {
      const d = document.createElement("div"); d.className = "item";
      d.innerHTML = `<span class="g">${esc(it.group)}</span><b>${esc(it.name)}</b><span class="s">${esc(it.kind)}${it.source ? " · " + esc(it.source) : ""}${it.license ? " · " + esc(it.license) : ""}</span>${it.expected ? `<span class="s exp">expected: ${esc(it.expected)}</span>` : ""}`;
      d.onclick = () => pickLibrary(it);
      box.appendChild(d);
    }
    $("library").hidden = false;
  };
  $("close-library").onclick = () => { $("library").hidden = true; };
  $("library").addEventListener("click", (e) => { if (e.target === $("library")) $("library").hidden = true; });

  // ---- live capture: camera / microphone of the machine running this browser ------------------
  // Records a short window with MediaRecorder, uploads it like any file (the server converts the
  // browser's webm to mp4/wav) and starts the investigation. "Keep watching" repeats after each verdict.
  const live = { rec: null, stream: null, timer: null, tick: null, mode: null, chunks: [], stopping: false };
  const liveOk = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
  if ($("live-support")) {
    if (!liveOk) $("live-support").textContent = window.isSecureContext ? "not supported by this browser" : "open the console at http://127.0.0.1 or https:// to allow the camera";
  }
  function liveStatus(text, rec) { const s = $("live-status"); if (!s) return; s.textContent = text; s.className = "muted small" + (rec ? " rec" : ""); }
  function liveButtons(recording) {
    $("live-video").hidden = recording; $("live-audio").hidden = recording; $("live-stop").hidden = !recording; $("live-secs").disabled = recording;
    $("analyze").disabled = recording;
  }
  function liveMime(video) {
    const cands = video ? ["video/webm;codecs=vp8,opus", "video/webm;codecs=vp9,opus", "video/webm", "video/mp4"] : ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
    return cands.find(m => MediaRecorder.isTypeSupported(m)) || "";
  }
  async function liveStart(mode) {
    if (!liveOk) { liveStatus("This browser cannot record. Open the console at http://127.0.0.1:<port> (not a tunnel address) or drop a file instead."); return; }
    const video = mode === "video";
    try {
      live.stream = await navigator.mediaDevices.getUserMedia(video
        ? { video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 25 } }, audio: { echoCancellation: true, noiseSuppression: true } }
        : { audio: { echoCancellation: true, noiseSuppression: true } });
    } catch (e) {
      liveStatus("Camera / microphone permission was refused (" + (e && e.name ? e.name : "error") + "). Allow it in the address bar and try again.");
      return;
    }
    live.mode = mode; live.chunks = []; live.stopping = false;
    const pv = $("live-preview");
    if (video) { pv.srcObject = live.stream; pv.hidden = false; } else { pv.hidden = true; }
    const mime = liveMime(video);
    try { live.rec = new MediaRecorder(live.stream, mime ? { mimeType: mime, videoBitsPerSecond: 2500000 } : undefined); }
    catch (e) { liveStatus("Recorder error: " + e); liveStop(true); return; }
    live.rec.ondataavailable = (e) => { if (e.data && e.data.size) live.chunks.push(e.data); };
    live.rec.onstop = liveFinish;
    const secs = parseInt($("live-secs").value, 10) || 15;
    let left = secs;
    liveButtons(true);
    liveStatus(`● recording ${video ? "camera + mic" : "microphone"} · ${left} s left · act the scene now`, true);
    live.rec.start(500);
    live.tick = setInterval(() => { left -= 1; if (left > 0) liveStatus(`● recording ${video ? "camera + mic" : "microphone"} · ${left} s left`, true); }, 1000);
    live.timer = setTimeout(() => liveStop(false), secs * 1000);
  }
  function liveStop(abort) {
    if (live.timer) { clearTimeout(live.timer); live.timer = null; }
    if (live.tick) { clearInterval(live.tick); live.tick = null; }
    live.stopping = !abort;
    if (live.rec && live.rec.state !== "inactive") { try { live.rec.stop(); } catch (e) { /* already stopped */ } }
    else if (abort) liveRelease();
    if (abort) { liveButtons(false); liveStatus("Recording cancelled."); }
  }
  function liveRelease() {
    if (live.stream) { live.stream.getTracks().forEach(t => t.stop()); live.stream = null; }
    const pv = $("live-preview"); pv.srcObject = null; pv.hidden = true;
  }
  async function liveFinish() {
    liveRelease();
    liveButtons(false);
    if (!live.stopping || !live.chunks.length) { liveStatus("Nothing was recorded."); return; }
    const type = (live.rec && live.rec.mimeType) || (live.mode === "video" ? "video/webm" : "audio/webm");
    const ext = type.includes("mp4") ? ".mp4" : ".webm";
    const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
    const file = new File([new Blob(live.chunks, { type })], `live_${live.mode === "video" ? "camera" : "mic"}_${stamp}${ext}`, { type });
    liveStatus(`sending ${(file.size / 1e6).toFixed(1)} MB to the device and converting…`);
    state.mediaId = null;
    await upload(file);
    if (!state.mediaId) { liveStatus("Upload failed; see the recording box above."); return; }
    live.lastMediaId = state.mediaId;
    liveStatus("Investigating the live window…");
    $("analyze").click();
  }
  if ($("live-video")) {
    $("live-video").onclick = () => liveStart("video");
    $("live-audio").onclick = () => liveStart("audio");
    $("live-stop").onclick = () => liveStop(false);
    // keep watching: when a run started from a live window ends, record the next window with the same mode
    document.addEventListener("angel:run-end", () => {
      if ($("live-watch").checked && live.mode && state.mediaId && state.mediaId === live.lastMediaId) setTimeout(() => liveStart(live.mode), 800);
    });
  }

  $("metrics-help").onclick = () => { const x = $("metrics-explain"); x.hidden = !x.hidden; };
  window.__angel = { pickPath: (p) => { state.mediaPath = p; state.mediaId = null; drop.classList.add("loaded"); $("drop-text").textContent = "✔ " + p; } };  // test hook
  const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}.${String(Math.round((t % 1) * 10))}`;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  init();
})();
