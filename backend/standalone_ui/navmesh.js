"use strict";

const token = new URLSearchParams(location.hash.replace(/^#/, "")).get("token") ||
  sessionStorage.getItem("nekoBackendToken") || "";
if (token) sessionStorage.setItem("nekoBackendToken", token);
const headers = token ? {"X-Neko-Backend-Token": token} : {};
const $ = (id) => document.getElementById(id);
const num = (v, d = 2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "—";

let meta = null, view = null, pending = null, lastPng = null, pollBusy = false;
const img = new Image();
img.onload = () => draw();

async function api(path, timeoutMs = 4000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(path, {headers, cache: "no-store", signal: ctl.signal});
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  } finally { clearTimeout(timer); }
}

async function command(path, body = {}) {
  const r = await fetch(path, {method: "POST", cache: "no-store",
    headers: {...headers, "Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}

async function act(label, path, body) {
  $("msg").textContent = `${label}…`;
  try {
    const res = await command(path, body);
    const ok = res.ok !== false && res.accepted !== false;
    $("msg").textContent = `${label}：${ok ? "成功" : "被拒"}${res.reason ? " · " + res.reason : ""}`;
    return res;
  } catch (e) {
    $("msg").textContent = `${label}失败：${e.message}`;
    return null;
  } finally { poll(); }
}

// 像素（格）↔ 导航系世界米；和 NavGrid.to_world / to_cell 同一套公式。
function cellToWorld(r, c) {
  const s = meta.world_scale, res = meta.resolution_track_m, o = meta.origin_xy_track_m;
  return [(o[0] + (c + 0.5) * res) * s, (o[1] + (meta.rows - 1 - r + 0.5) * res) * s];
}
function worldToCell(x, y) {
  const s = meta.world_scale, res = meta.resolution_track_m, o = meta.origin_xy_track_m;
  return [meta.rows - 1 - Math.floor((y / s - o[1]) / res), Math.floor((x / s - o[0]) / res)];
}

function draw() {
  const cv = $("grid"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#080b10";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!meta || !img.complete || !img.naturalWidth) {
    view = null;
    ctx.fillStyle = "#8b949e"; ctx.font = "14px sans-serif";
    ctx.fillText("还没有栅格", 20, 30);
    return;
  }
  const k = Math.min(cv.width / meta.cols, cv.height / meta.rows);
  const ox = (cv.width - meta.cols * k) / 2, oy = (cv.height - meta.rows * k) / 2;
  view = {k, ox, oy};
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, ox, oy, meta.cols * k, meta.rows * k);
  // 1 m 刻度尺
  const cellM = meta.resolution_track_m * meta.world_scale, px = k / cellM;
  ctx.strokeStyle = "#8b949e"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(12, cv.height - 14); ctx.lineTo(12 + px, cv.height - 14); ctx.stroke();
  ctx.fillStyle = "#8b949e"; ctx.font = "12px sans-serif"; ctx.fillText("1 m", 16 + px, cv.height - 10);
  if (pending) {
    const [r, c] = worldToCell(pending[0], pending[1]);
    const x = ox + (c + 0.5) * k, y = oy + (r + 0.5) * k;
    ctx.strokeStyle = "#f85149"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.moveTo(x - 13, y); ctx.lineTo(x + 13, y); ctx.moveTo(x, y - 13); ctx.lineTo(x, y + 13); ctx.stroke();
  }
}

$("grid").addEventListener("click", (ev) => {
  if (!meta || !view) return;
  const cv = $("grid"), rect = cv.getBoundingClientRect();
  const px = (ev.clientX - rect.left) * cv.width / rect.width;
  const py = (ev.clientY - rect.top) * cv.height / rect.height;
  const c = Math.floor((px - view.ox) / view.k), r = Math.floor((py - view.oy) / view.k);
  if (r < 0 || c < 0 || r >= meta.rows || c >= meta.cols) return;
  pending = cellToWorld(r, c);
  $("clickInfo").textContent = `待确认目标 x ${num(pending[0])} / y ${num(pending[1])} m（后端会检查是否可走、是否连通）`;
  $("gotoConfirm").disabled = false; $("gotoClear").disabled = false;
  draw();
});
$("gotoConfirm").onclick = async () => {
  if (!pending) return;
  const [x, y] = pending;
  const res = await act("前往", "/worldmodel/navmesh/goto", {x, y});
  if (res && res.ok !== false) { pending = null; $("gotoConfirm").disabled = true; $("gotoClear").disabled = true; }
};
$("gotoClear").onclick = () => {
  pending = null; $("gotoConfirm").disabled = true; $("gotoClear").disabled = true;
  $("clickInfo").textContent = "点击栅格选择目标。"; draw();
};
$("start").onclick = () => act("启动", "/worldmodel/navmesh/start");
$("stop").onclick = () => act("停止", "/worldmodel/navmesh/stop");
$("explore").onclick = () => act("探索", "/worldmodel/navmesh/explore");
$("cancel").onclick = () => act("取消目标", "/worldmodel/navmesh/cancel");
$("arm").onclick = () => act("武装", "/autonomy/arm", {ttl_s: 300});
$("disarm").onclick = () => act("解除武装", "/autonomy/disarm", {reason: "navmesh_ui_disarm"});
$("estop").onclick = async () => {
  await act("急停：取消目标", "/worldmodel/navmesh/cancel");
  await act("急停：停止自主", "/autonomy/stop", {reason: "navmesh_ui_estop"});
};

function flag(el, text, cls) { el.textContent = text; el.className = `flag ${cls}`; }

function render(s, auto) {
  const running = !!s.running;
  $("state").textContent = running ? "运行中" : "未运行";
  $("state").className = `state ${running ? "running" : "stopped"}`;
  $("start").disabled = running; $("stop").disabled = !running; $("explore").disabled = !running;
  $("mode").textContent = s.mode || "—";
  const p = s.pose;
  $("pose").textContent = p ? `x ${num(p.xy_m[0])}  y ${num(p.xy_m[1])}  θ ${num(p.theta_rad * 180 / Math.PI, 0)}°  σ ${num(p.sigma_m)}` : "—";
  $("poseState").textContent = s.pose_state || "—";
  $("goal").textContent = s.goal_xy_m ? `${num(s.goal_xy_m[0])}, ${num(s.goal_xy_m[1])}` : "—";
  $("odo").textContent = `${num(s.odometry_distance_m, 1)} m / ${s.osc_samples ?? 0}`;
  const sen = s.sensors;
  $("sensors").textContent = sen && sen.size ? `${sen.size[0]}×${sen.size[1]} fx ${num(sen.fx, 1)} 基线 ${num(sen.baseline_m, 3)}` : "未打开";
  $("stereo").textContent = `${s.stereo_ms == null ? "—" : num(s.stereo_ms, 0) + " ms"} / ${s.stereo_age_s == null ? "—" : num(s.stereo_age_s, 1) + " s"}`;
  $("kf").textContent = `${s.keyframes ?? 0} / ${s.map_updates ?? 0}`;
  $("mapMs").textContent = s.map_update_ms == null ? "—" : `${num(s.map_update_ms, 0)} ms`;
  $("camH").textContent = s.camera_height_m == null ? "—" : `${num(s.camera_height_m)} m`;
  const lc = s.loop_closure;
  $("loops").textContent = lc ? `${lc.loops} / ${lc.loops_downweighted} / ${num(lc.correction_m)} m` : "关闭";
  const yc = (lc && lc.yaw_checks) || [];
  const jumps = s.hmd_yaw_jumps || [];
  const lastYaw = yc.slice(-4).map(c => `${c.accepted ? "" : "✗"}${num(c.yaw_err_deg, 0)}°`).join(" ");
  flag($("yawDiag"), `${lastYaw || "—"} / ${jumps.length ? jumps.slice(-2).map(j => num(j.delta_deg, 0) + "°").join(" ") : "无"}`,
       jumps.length ? "bad" : "ok");
  const near = s.near_obstacle || {};
  flag($("near"), near.stop ? `停（${near.points} 点）` : `无（${near.points ?? 0} 点）`, near.stop ? "bad" : "ok");
  if (auto) {
    const armed = !!auto.armed;
    flag($("autonomy"), armed ? `已武装 ${num(auto.remaining_seconds, 0)} s` : (auto.state || "未武装"), armed ? "ok" : "warn");
  } else flag($("autonomy"), "读取失败", "bad");
  flag($("block"), s.drive_block || "无", s.drive_block ? "warn" : "ok");
  const st = s.last_step || {};
  $("step").textContent = st.state ? `${st.state}${st.reason ? " · " + st.reason : ""} · 前 ${num(st.forward)} 转 ${num(st.turn_rate)}` : "—";
  const lp = s.last_plan || {};
  $("plan").textContent = Object.keys(lp).length
    ? `规划：${lp.accepted === true ? "已接受" : lp.accepted === false ? "被拒" : "—"}${lp.reason ? " · " + lp.reason : ""}${lp.length_m != null ? " · 路径 " + num(lp.length_m) + " m" : ""}${lp.frontiers != null ? " · frontier " + lp.frontiers : ""}${lp.grid && lp.grid.walkable_center_m2 != null ? " · 可走 " + num(lp.grid.walkable_center_m2, 1) + " m²" : ""}`
    : "规划：还没有";
  $("errors").textContent = (s.errors || []).map(e => typeof e === "string" ? e : JSON.stringify(e)).join("\n");
  meta = s.grid_meta || null;
  const png = s.grid_png_base64 || null;
  if (!png) { lastPng = null; draw(); }
  else if (png !== lastPng) { lastPng = png; img.src = `data:image/png;base64,${png}`; }
  else draw();
  if (!pending && meta) $("clickInfo").textContent = "点击栅格选择目标。";
}

async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    const [nav, auto] = await Promise.allSettled([api("/worldmodel/navmesh?grid=1"), api("/autonomy")]);
    if (nav.status === "fulfilled") render(nav.value, auto.status === "fulfilled" ? auto.value : null);
    else { $("state").textContent = "后端不可达"; $("state").className = "state lost"; $("msg").textContent = String(nav.reason); }
  } finally { pollBusy = false; }
}

if (!token) $("msg").textContent = "URL 里没有 token：用 /navmesh#token=… 打开。";
poll();
setInterval(poll, 500);
