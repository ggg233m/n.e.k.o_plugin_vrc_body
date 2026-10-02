"use strict";

// 覆盖俯视页：只读消费 /worldmodel/navmesh/coverage（伴生覆盖网格 + 洞分类 + 诚实进度指标）。
// 工具函数（token/headers/api/坐标换算/PNG-diff 重绘）与 navmesh.js 同款；本页不发生任何驾驶指令。

const token = new URLSearchParams(location.hash.replace(/^#/, "")).get("token") ||
  sessionStorage.getItem("nekoBackendToken") || "";
if (token) sessionStorage.setItem("nekoBackendToken", token);
const headers = token ? {"X-Neko-Backend-Token": token} : {};
const $ = (id) => document.getElementById(id);
const num = (v, d = 2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "—";
const pct = (v) => Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(1) + "%" : "—";

let meta = null, view = null, lastPng = null, pollBusy = false;
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

// 像素（格）↔ 导航系世界米；与 NavGrid.to_world / to_cell 同一套公式（行 0 = +y 最大）。
function draw() {
  const cv = $("grid"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#080b10";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!meta || !img.complete || !img.naturalWidth) {
    view = null;
    ctx.fillStyle = "#8b949e"; ctx.font = "14px sans-serif";
    ctx.fillText("还没有覆盖快照", 20, 30);
    return;
  }
  const k = Math.min(cv.width / meta.cols, cv.height / meta.rows);
  const ox = (cv.width - meta.cols * k) / 2, oy = (cv.height - meta.rows * k) / 2;
  view = {k, ox, oy};
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, ox, oy, meta.cols * k, meta.rows * k);
  const cellM = meta.resolution_track_m * meta.world_scale, px = k / cellM;
  ctx.strokeStyle = "#8b949e"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(12, cv.height - 14); ctx.lineTo(12 + px, cv.height - 14); ctx.stroke();
  ctx.fillStyle = "#8b949e"; ctx.font = "12px sans-serif"; ctx.fillText("1 m", 16 + px, cv.height - 10);
}

function sparkline(trend) {
  const cv = $("spark"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#080b10"; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!trend || trend.length < 2) {
    ctx.fillStyle = "#8b949e"; ctx.font = "12px sans-serif";
    ctx.fillText("趋势样本累积中…", 12, 24);
    return;
  }
  const pad = 8, W = cv.width - 2 * pad, H = cv.height - 2 * pad;
  const xs = (i) => pad + W * i / (trend.length - 1);
  // 覆盖率 0–1 固定纵轴。
  ctx.strokeStyle = "#3fb950"; ctx.lineWidth = 2; ctx.beginPath();
  trend.forEach((t, i) => {
    const y = pad + H * (1 - Math.min(1, Math.max(0, t.coverage_ratio)));
    i ? ctx.lineTo(xs(i), y) : ctx.moveTo(xs(i), y);
  });
  ctx.stroke();
  // frontier 边界格数：自归一化看趋势。
  const fmax = Math.max(1, ...trend.map((t) => t.frontier_cells || 0));
  ctx.strokeStyle = "#d29922"; ctx.lineWidth = 1.5; ctx.beginPath();
  trend.forEach((t, i) => {
    const y = pad + H * (1 - (t.frontier_cells || 0) / fmax);
    i ? ctx.lineTo(xs(i), y) : ctx.moveTo(xs(i), y);
  });
  ctx.stroke();
}

function render(c) {
  if (!c || c.available === false) {
    $("clickInfo").textContent = c && c.error ? `覆盖快照出错：${c.error}` :
      (c && c.reason) || "还没有覆盖快照：先启动导航，走几步生成关键帧。";
    meta = null; lastPng = null; draw();
    return;
  }
  const m = c.metrics || {};
  $("covRatio").textContent = pct(m.coverage_ratio);
  $("nearRatio").textContent = pct(m.near_ratio);
  $("areas").textContent = `${num(m.hull_m2, 1)} m² / ${num(m.observed_m2, 1)} m²`;
  // 信息洞·未观测 = visited 内 total==0 的粗格；只远看 = total>0 且 near==0。
  $("holes").textContent = m.hole_cells != null ? `${m.hole_cells} 格` : "—";
  $("farOnly").textContent = m.far_only_cells != null ? `${m.far_only_cells} 格` : "—";
  $("frontier").textContent = `${m.frontier_cells ?? "—"} / ${m.frontier_goals ?? "—"}`;
  $("kf").textContent = `${m.keyframes ?? "—"} / ${m.map_updates ?? "—"}`;
  $("covMs").textContent = m.cov_ms != null ? `${num(m.cov_ms)} ms` : "—";
  $("caliber").textContent = `hull 半径 ${num(m.near_m, 1)} m（近看阈值）· 粗格 ${num(m.coarse_m, 2)} m`;
  meta = c.grid || null;
  const png = (c.grid && c.grid.png_base64) || null;
  if (!png) { lastPng = null; draw(); }
  else if (png !== lastPng) { lastPng = png; img.src = `data:image/png;base64,${png}`; }
  else draw();
  sparkline(c.trend);
}

async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    const c = await api("/worldmodel/navmesh/coverage");
    render(c);
  } catch (e) {
    $("msg").textContent = `后端不可达：${e.message}`;
  } finally { pollBusy = false; }
}

if ($("navLink") && token) $("navLink").href = `/navmesh#token=${encodeURIComponent(token)}`;
if (!token) $("msg").textContent = "URL 里没有 token：用 /coverage#token=… 打开。";
poll();
setInterval(poll, 1000);
