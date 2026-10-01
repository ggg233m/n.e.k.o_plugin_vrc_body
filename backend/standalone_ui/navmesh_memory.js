"use strict";

// navmesh 记忆管理页。世界名/标签来自磁盘，一律用 textContent 建节点，不拼 innerHTML。
const token = new URLSearchParams(location.hash.replace(/^#/, "")).get("token") ||
  sessionStorage.getItem("nekoBackendToken") || "";
if (token) sessionStorage.setItem("nekoBackendToken", token);
const headers = token ? {"X-Neko-Backend-Token": token} : {};
const $ = (id) => document.getElementById(id);
const BASE = "/worldmodel/navmesh/memory";
const THUMB_LIMIT = 48;

let worldSel = null, sessionSel = null, thumbUrls = [];

async function api(path) {
  const r = await fetch(path, {headers, cache: "no-store"});
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function command(path, body = {}) {
  const r = await fetch(path, {method: "POST", cache: "no-store",
    headers: {...headers, "Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  if (data.ok === false) throw new Error(data.reason || "failed");
  return data;
}

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined && text !== null) e.textContent = String(text);
  if (cls) e.className = cls;
  return e;
}

function btn(text, onClick, {danger = false, disabled = false, title = ""} = {}) {
  const b = el("button", text, danger ? "danger" : "");
  b.disabled = disabled;
  if (title) b.title = title;
  b.addEventListener("click", (ev) => { ev.stopPropagation(); onClick(); });
  return b;
}

const q = (params) => new URLSearchParams(params).toString();
const when = (wall) => wall ? new Date(wall * 1000).toLocaleString() : "—";
const say = (text, cls = "muted") => { const m = $("msg"); m.textContent = text; m.className = cls; };

async function guarded(fn) {
  try { await fn(); } catch (err) { say(`失败：${err.message}`, "bad"); }
}

async function loadWorlds() {
  const s = await api(BASE);
  const cur = s.current_world || {};
  $("curWorld").textContent = cur.world_key ? `${cur.world_name || cur.world_key}（${cur.world_source}）` : "未知（不写记忆）";
  const act = s.active;
  $("summary").textContent = `${s.worlds} 个世界 · ${s.sessions} 个会话 · ${s.mb} / ${s.max_total_mb} MB` +
    (act ? ` · 正在写 ${act.session_id}（${act.keyframes} 帧）` : "") + (s.enabled ? "" : " · 记忆已关闭");
  const tb = $("worlds");
  tb.replaceChildren();
  for (const w of s.world_list) {
    const tr = el("tr", null, "pick" + (w.world_id === worldSel ? " sel" : ""));
    const name = el("td");
    name.append(el("div", w.world_name || w.world_key), el("div", `${w.world_key} · ${when(w.last_used_wall)}`, "muted"));
    tr.append(name, el("td", w.sessions), el("td", w.mb));
    const ops = el("td");
    ops.append(btn("删世界", () => deleteWorld(w), {danger: true, disabled: w.active,
      title: w.active ? "正在写入，先停止导航" : ""}));
    tr.append(ops);
    tr.addEventListener("click", () => selectWorld(w.world_id));
    tb.append(tr);
  }
  if (!s.world_list.length) {
    const tr = el("tr"), td = el("td", "还没有记忆（设置世界身份后启动导航才会记录）", "muted");
    td.colSpan = 4; tr.append(td); tb.append(tr);
  }
  if (worldSel && !s.world_list.some((w) => w.world_id === worldSel)) { worldSel = null; sessionSel = null; }
  if (worldSel) await loadSessions();
  else { $("sessions").replaceChildren(); $("detail").replaceChildren(); $("sessTitle").textContent = "会话（先选一个世界）"; }
}

async function selectWorld(id) {
  worldSel = id; sessionSel = null;
  await guarded(loadWorlds);
}

function statusCell(s) {
  const cls = {complete: "ok", recording: "warn", interrupted: "warn", error: "bad", corrupt: "bad"}[s.status] || "muted";
  const td = el("td");
  td.append(el("span", s.active ? "写入中" : s.status, cls));
  if (s.size_limited) td.append(el("div", "已达单会话上限", "warn"));
  if (s.dropped) td.append(el("div", `丢 ${s.dropped} 帧`, "warn"));
  return td;
}

async function loadSessions() {
  const r = await api(`${BASE}/sessions?${q({world: worldSel})}`);
  if (r.ok === false) throw new Error(r.reason);
  $("sessTitle").textContent = `会话（${worldSel}）`;
  const tb = $("sessions");
  tb.replaceChildren();
  for (const s of r.sessions) {
    const tr = el("tr", null, "pick" + (s.session_id === sessionSel ? " sel" : ""));
    const id = el("td");
    id.append(el("div", (s.pinned ? "📌 " : "") + s.session_id), el("div", when(s.started_wall), "muted"));
    const label = el("input");
    label.type = "text"; label.value = s.label || ""; label.maxLength = 80; label.disabled = s.active;
    label.setAttribute("aria-label", `会话 ${s.session_id} 的标签`);
    label.addEventListener("click", (ev) => ev.stopPropagation());
    label.addEventListener("change", () => guarded(async () => {
      await command(`${BASE}/update`, {world: worldSel, session: s.session_id, label: label.value});
      say("标签已保存", "ok");
    }));
    const labelTd = el("td"); labelTd.append(label);
    const ops = el("td");
    ops.append(
      btn(s.pinned ? "取消钉住" : "钉住", () => guarded(async () => {
        await command(`${BASE}/update`, {world: worldSel, session: s.session_id, pinned: !s.pinned});
        await loadSessions();
      }), {disabled: s.active, title: "钉住的会话不会被配额清理删除"}),
      btn("删除", () => deleteSession(s), {danger: true, disabled: s.active}));
    tr.append(id, statusCell(s), el("td", `${s.keyframes ?? 0} / ${s.features ?? 0}`), el("td", s.mb),
      el("td", s.baseline_m ?? "—"), labelTd, ops);
    tr.addEventListener("click", () => { sessionSel = s.session_id; guarded(loadSessions); });
    tb.append(tr);
  }
  if (sessionSel) await loadDetail();
  else $("detail").replaceChildren();
}

async function loadDetail() {
  const d = await api(`${BASE}/session?${q({world: worldSel, session: sessionSel})}`);
  if (d.ok === false) throw new Error(d.reason);
  for (const u of thumbUrls) URL.revokeObjectURL(u);
  thumbUrls = [];
  const box = $("detail");
  box.replaceChildren();
  const info = {status: d.status, frame: d.frame, world_scale: d.world_scale, baseline_m: d.baseline_m,
    odometry_m: d.odometry_m, loops: d.loops, ended: when(d.ended_wall), error: d.error};
  box.append(el("pre", JSON.stringify(info, null, 1)));
  const ks = d.thumbnails || [];
  // 大会话只抽样显示，避免一次拉几百张图。
  const step = Math.max(1, Math.ceil(ks.length / THUMB_LIMIT));
  const shown = ks.filter((_, i) => i % step === 0);
  box.append(el("div", `缩略图 ${ks.length} 张${step > 1 ? `，每 ${step} 张显示 1 张` : ""}`, "muted"));
  const grid = el("div", null, "thumbs");
  box.append(grid);
  for (const k of shown) {
    const fig = el("figure");
    const img = el("img");
    img.alt = `关键帧 ${k} 的左目缩略图`;
    fig.append(img, el("figcaption", `kf ${k}`));
    grid.append(fig);
    fetch(`${BASE}/thumb?${q({world: worldSel, session: sessionSel, k})}`, {headers, cache: "no-store"})
      .then((r) => r.ok ? r.blob() : null)
      .then((b) => { if (b) { const u = URL.createObjectURL(b); thumbUrls.push(u); img.src = u; } })
      .catch(() => {});
  }
}

async function deleteSession(s) {
  if (!confirm(`删除会话 ${s.session_id}（${s.mb} MB）？不可恢复。`)) return;
  await guarded(async () => {
    const r = await command(`${BASE}/delete`, {world: worldSel, session: s.session_id});
    if (sessionSel === s.session_id) sessionSel = null;
    say(`已删除，释放 ${r.freed_mb} MB`, "ok");
    await loadWorlds();
  });
}

async function deleteWorld(w) {
  const name = w.world_name || w.world_key;
  if (!confirm(`删除世界「${name}」的全部 ${w.sessions} 个会话（${w.mb} MB），包括钉住的？不可恢复。`)) return;
  await guarded(async () => {
    const r = await command(`${BASE}/delete`, {world: w.world_id, scope: "world"});
    if (worldSel === w.world_id) { worldSel = null; sessionSel = null; }
    say(`已删除世界，释放 ${r.freed_mb} MB`, "ok");
    await loadWorlds();
  });
}

$("refresh").addEventListener("click", () => guarded(loadWorlds));
$("prune").addEventListener("click", () => guarded(async () => {
  const r = await command(`${BASE}/prune`);
  say(r.deleted.length ? `清理了 ${r.deleted.length} 个会话，释放 ${r.freed_mb} MB` : "未超配额，没有删除", "ok");
  await loadWorlds();
}));
$("navLink").href = `/navmesh${token ? `#token=${encodeURIComponent(token)}` : ""}`;

if (!token) say("URL 里没有 token：用 /navmesh/memory#token=… 打开。", "bad");
else guarded(loadWorlds);
