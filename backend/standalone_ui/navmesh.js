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
  } catch (e) {
    // AbortError 的原话是 "signal is aborted without reason"，直接甩出去没人看得懂
    if (e && e.name === "AbortError") throw new Error(`超时（${Math.round(timeoutMs / 1000)} s 没响应）`);
    throw e;
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
$("start").onclick = () => act("启动", "/worldmodel/navmesh/start", {record: $("record").checked});
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
  $("record").disabled = running;
  const rec = s.recording;
  $("recInfo").textContent = rec ? `录制：${rec.keyframes} 关键帧 · ${rec.mb} MB${rec.stopped ? " · " + rec.stopped : ""} · ${rec.dir}` : "";
  $("mode").textContent = s.mode || "—";
  const p = s.pose;
  $("pose").textContent = p ? `x ${num(p.xy_m[0])}  y ${num(p.xy_m[1])}  θ ${num(p.theta_rad * 180 / Math.PI, 0)}°  σ ${num(p.sigma_m)}` : "—";
  $("poseState").textContent = s.pose_state || "—";
  $("goal").textContent = s.goal_xy_m ? `${num(s.goal_xy_m[0])}, ${num(s.goal_xy_m[1])}` : "—";
  $("odo").textContent = `${num(s.odometry_distance_m, 1)} m / ${s.osc_samples ?? 0}`;
  // 静默封顶：VelocityX/Z 无心跳，丢包丢掉"停下"那个 0 包时，最后一个速度不能被永远积分。
  // 丢弃量不为 0 = 现场真的撞上了这个洞（不是误差条，是没算进去的那段）。
  const ho = s.odometry_holdout || {};
  const hoText = ho.last_sample_age_s == null ? "—" : `${num(ho.last_sample_age_s, 1)} s`;
  $("odoHold").textContent = `${hoText} / ${num(ho.dropped_m, 1)} m · 最长合法空档 ${num(ho.max_legit_gap_s, 2)} s`;
  flag($("odoHold"), (ho.dropped_m || 0) > 0.5
    || (ho.max_legit_gap_s || 0) > (ho.zoh_max_s || 2.5) ? "bad" : "ok");
  const sen = s.sensors;
  $("sensors").textContent = sen && sen.size ? `${sen.size[0]}×${sen.size[1]} fx ${num(sen.fx, 1)} 基线 ${num(sen.baseline_m, 3)}` : "未打开";
  $("stereo").textContent = `${s.stereo_ms == null ? "—" : num(s.stereo_ms, 0) + " ms"} / ${s.stereo_age_s == null ? "—" : num(s.stereo_age_s, 1) + " s"}`;
  $("kf").textContent = `${s.keyframes ?? 0} / ${s.map_updates ?? 0}`;
  $("mapMs").textContent = s.map_update_ms == null ? "—" : `${num(s.map_update_ms, 0)} ms`;
  $("camH").textContent = s.camera_height_m == null ? "—" : `${num(s.camera_height_m)} m`;
  const lc = s.loop_closure;
  // 回环健康度：距上次回环多少关键帧 / 多少 OSC 路程。长期单涨 = 回环静默（尾部 0 回环）的现场证据。
  $("loops").textContent = lc
    ? `${lc.loops} / ${lc.loops_downweighted} / ${num(lc.correction_m)} m · 距上次回环 ${lc.kf_since_last_loop ?? "—"} 帧 / ${lc.path_since_last_loop_m ?? "—"} m`
    : "关闭";
  const yc = (lc && lc.yaw_checks) || [];
  const jumps = s.hmd_yaw_jumps || [];
  const lastYaw = yc.slice(-4).map(c => `${c.accepted ? "" : "✗"}${num(c.yaw_err_deg, 0)}°`).join(" ");
  flag($("yawDiag"), `${lastYaw || "—"} / ${jumps.length ? jumps.slice(-2).map(j => (j.neutral ? "外部中立帧 " : "") + num(j.delta_deg, 0) + "°").join(" ") : "无"}`,
       jumps.length ? "bad" : "ok");
  // 跨会话检索 + 会话末采纳（P0）：索引里几个会话 / 确认了几条约束；采纳是**会话末**才跑的，
  // 所以运行中显示"采纳中/未跑"是正常的，不是故障（别把它写成 ✗）。
  const xs = s.xsession || {};
  const mem = s.memory || {};
  const al = s.xsession_align || {};
  // 会话末做两件事：逐对采纳（本场挂到看见最多的旧会话）→ 世界树（收别的会话之间的桥）。
  // 并树的结果单独说，因为 0 采纳很常见（"已经在系里"）而 1 次采纳往往就是解开缺口的那一下。
  const wt = al.world_tree || null;
  const wtText = wt && wt.ok
    ? ` · 并树 ${wt.aligned ? "+" + wt.aligned : "无"}` +
      (wt.refused ? `（拒 ${wt.refused}）` : "")
    : (wt && wt.reason ? ` · 并树失败 ${wt.reason}` : "");
  const alText = al.state === "running" ? "采纳中"
    : al.state === "pending" ? "采纳未跑（会话末才跑）"
      : (al.ok === true ? `采纳✓${al.old_sid ? " → " + al.old_sid : ""}`
         : `采纳✗ ${al.reason || al.state || ""}`) + wtText;
  // 未启用时的 reason 是内部码（world_unknown / memory_not_configured…）。直接甩码等于没说：
  // world_unknown 是要**去设世界身份**，memory_not_configured 是**没配记忆存储**，两者动作不同。
  const XS_REASON = [["world_unknown", "未设世界身份"], ["memory_not_configured", "记忆未配置"],
                     ["disabled", "已关闭"], ["vocab_missing", "词汇树缺失"],
                     ["memory_error", "记忆出错"], ["init_error", "初始化失败"],
                     ["loading", "索引装载中"], ["no_sessions", "世界里还没有可索引的会话"]];
  const xsWhy = (why) => {
    const w = String(why || "");
    const hit = XS_REASON.find(([k]) => w.startsWith(k));
    return hit ? hit[1] : (w || "未启用");
  };
  const xsHead = xs.active === false ? xsWhy(xs.reason)
    : `${mem.world_key || "—"} · ${xs.sessions ?? 0} 会话 · 确认 ${xs.confirmed ?? 0} · 验证 ${xs.verified ?? 0}`;
  // 「确认 N」只说总量，"重识别到底认出谁了"看不出来（2026-10-06 实况：用户走了整张图、
  // 确认 44 条含 18 条对上一场，却以为没识别）。这里把最近的确认按旧会话聚合出来：
  // 认出谁 + 平均位移 —— 位移小就说明"确实认出了同一个地方"。
  const recentCons = xs.recent || [];
  const byOld = {};
  for (const c of recentCons) {
    const k = String(c.old_sid || "?").slice(-6);
    (byOld[k] = byOld[k] || []).push(Number(c.offset_m) || 0);
  }
  const recentText = Object.keys(byOld).length
    ? " · 最近认出 " + Object.entries(byOld)
        .sort((a, b) => b[1].length - a[1].length).slice(0, 3)
        .map(([k, offs]) => `${k}×${offs.length}（位移 ${num(offs.reduce((s, v) => s + v, 0) / offs.length)} m）`)
        .join(" ")
    : "";
  // world_unknown：会话根本没进任何世界分区 ⇒ 下面"世界先验"必然也是零注入，两行要一起读。
  flag($("xsession"), `${xsHead}${recentText} · ${alText}`,
    (xs.errors || 0) > 0 ? "bad" : (xs.active === false ? "warn" : "ok"));
  // 世界先验注入（P0.3b）：ok 后面那串才是"真的补进去了几格"；被拒时的 state 是**闸门**报的
  // 原因，所以数字（内点/残差/前后半差异）一律要显示出来 —— 只说一个 gauge_inlier_frac
  // 等于没说，现场没法判断是位姿漂了还是阈值太紧。
  const pr = s.prior || {};
  const ap = pr.applied || {};
  const gp = pr.gauge || {};
  const sp = pr.split || {};
  const gaugeText = gp.n == null ? "" :
    ` ｜ 内点 ${gp.n_inlier}/${gp.n} · 残差中位 ${num(gp.pos_med_m)} m / ${num(gp.rot_med_deg, 1)}°` +
    (sp.m == null ? "" : ` · 前后半 ${num(sp.m)} m / ${num(sp.yaw_deg, 1)}°`);
  flag($("prior"), pr.enabled === false ? "已关闭 · " + (pr.state || "—")
    : (pr.state === "ok"
        ? `补 free ${ap.applied_free ?? 0} / 障碍 ${ap.applied_occ ?? 0}${ap.blocked_walked ? " · 挡走廊 " + ap.blocked_walked : ""}${gaugeText}`
        : `${PRIOR_WHY[pr.state] || pr.state || "—"}${gaugeText}`),
    pr.state === "ok" ? "ok" : (pr.enabled === false || PRIOR_SOFT.includes(pr.state) ? "warn" : "bad"));
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
  paintLegend(meta);
  if (!pending && meta) $("clickInfo").textContent = "点击栅格选择目标。";
}

// 底图图例：三态用离散色块，其余用色条。色条两端读后端回的 ramp.vmin/vmax
// ——后端刻意用**固定**范围（不从数据取分位），这样帧与帧之间颜色可直接比较。
const LAYER_NOTE = {
  surface: "头顶那张面的离地高。均值系统性偏高约 0.1 m，只用于肉眼找结构，不当米制。",
  bands: "四条高度带里点数最多的那条。分不出「障碍还是楼板」——那要带内高度聚类，尚未做。",
  clearance: "格中心到最近非可走格的距离（世界米）。越大越宽裕。",
  prior: "世界先验（多场会话离线下融合）投影到本场会话帧。亮色 = 先验真补进去的格（当时 live 还是 unknown），暗色 = 先验有意见但 live 已有证据、没采纳。",
  tristate: "",
};
// 先验闸门的 state → 人话。**这些是"要不要信这张先验"的判据**，只说英文码等于没说：
// `gauge_too_few` 要再走走、`gauge_unstable` 是位姿在漂（先修定位）、`no_prior` 要先去固化。
const PRIOR_WHY = {
  idle: "待首次栅格化", not_started: "待首次栅格化", disabled: "已关闭",
  no_xsession: "无跨会话（先设世界身份）", no_prior: "世界里还没有 prior/（先跑 offline_fusion.py --write-prior）",
  gauge_too_few: "约束还不够（会话早期正常）", gauge_degenerate: "约束退化，估不出 gauge",
  gauge_few_inliers: "内点太少", gauge_inlier_frac: "位姿内部不一致（内点占比不够）",
  gauge_pos_residual: "残差太大（位姿与历史对不上）", gauge_rot_residual: "朝向残差太大",
  gauge_unstable: "会话帧在漂移（前后半估的 gauge 对不上）",
  frame_mismatch: "旧会话表与先验不同坐标系", world_scale_mismatch: "世界尺度变了（换过 avatar？）",
};
// 这些不是"出错"，是"还没到时候"：画红会让人去查一个不存在的故障。
const PRIOR_SOFT = ["idle", "no_prior", "not_started", "no_xsession", "gauge_too_few", "gauge_degenerate"];

const OVERLAY = [
  ["#0078ff", "规划路径"], ["#f00", "当前位姿"], ["#0c0", "目标"],
];
// 与后端 _prior_layer 的取色逐值对应（那里是 RGB，这里是 #rrggbb）。
const PRIOR_LEGEND = [
  ["#96e196", "先验补 free"], ["#e13c3c", "先验补 障碍"],
  ["#6ea06e", "先验 free 未采纳"], ["#823232", "先验障碍 未采纳"],
];
function paintLegend(m) {
  const layer = (m && m.layer) || "tristate";
  const bar = $("rampBar"), lg = $("legend");
  if ($("layerNote")) $("layerNote").textContent = LAYER_NOTE[layer] || "";
  if (layer === "tristate") {
    bar.style.display = "none"; lg.style.display = "";
    lg.innerHTML = [["#fff", "可走中心区"], ["#aaa", "观测 free"], ["#5a5a5a", "unknown"],
                    ["#000", "障碍"]].concat(OVERLAY)
                   .map(([c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join("");
    return;
  }
  if (layer === "prior") {
    // 先验层的底图仍是 live 三态（便于对照），再加先验那四色。
    bar.style.display = "none"; lg.style.display = "";
    lg.innerHTML = [["#fff", "可走中心区"], ["#aaa", "live free"], ["#5a5a5a", "unknown"],
                    ["#000", "live 障碍"]].concat(PRIOR_LEGEND, OVERLAY)
                   .map(([c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join("");
    return;
  }
  const r = (m && m.ramp) || {};
  lg.style.display = "none"; bar.style.display = "";
  if (r.available === false) {
    // 先验层不可用时图上只有 live 灰阶 —— 不写清楚，现场就会得出"世界先验和三态没区别"
    // 这个错误结论（2026-10-06 实况）。这里直接说为什么没注入。
    $("rampNote").textContent = layer === "prior"
      ? (PRIOR_WHY[r.reason] || r.reason || "不可用") + " ⇒ 本图只有 live 观测"
      : "不可用：" + (r.reason || "");
    $("rampLo").textContent = $("rampHi").textContent = "—";
    return;
  }
  if (r.bands) {
    $("rampNote").innerHTML = r.bands.map(b =>
      `<span style="margin-right:10px"><i style="display:inline-block;width:10px;height:10px;` +
      `margin-right:3px;background:rgb(${b.rgb.join(",")});border:1px solid #666"></i>${b.name}</span>`
    ).join("");
    $("rampLo").textContent = (r.edges_m || []).map(v => v.toFixed(1)).join(" / ");
    $("rampHi").textContent = "m（离地高上界）";
    $("rampStrip").style.background = "#333";
    return;
  }
  $("rampStrip").style.background = "";
  $("rampLo").textContent = (r.vmin ?? 0).toFixed(1) + (r.unit || "");
  $("rampHi").textContent = (r.vmax ?? 0).toFixed(1) + (r.unit || "") +
    (r.data_max != null ? `（图上最大 ${r.data_max}）` : "");
  $("rampNote").textContent = "";
}

async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    const layer = ($("layer") && $("layer").value) || "tristate";
    const [nav, auto] = await Promise.allSettled([
      api(`/worldmodel/navmesh?grid=1&layer=${encodeURIComponent(layer)}`), api("/autonomy")]);
    if (nav.status === "fulfilled") render(nav.value, auto.status === "fulfilled" ? auto.value : null);
    else { $("state").textContent = "后端不可达"; $("state").className = "state lost"; $("msg").textContent = String(nav.reason); }
  } finally { pollBusy = false; }
}

// ---- 世界身份（跨会话检索 / 采纳 / 世界先验 全挂在它上面）----
// 只在进页面、设完、点按钮时拉记忆分区；`sizes=0` 让后端**别扫目录算体积**（上万个文件、
// 秒级 ⇒ 会把这里的 8 s 超时打爆，2026-10-06 实况），也**不能跟着 2 Hz 的轮询打**。
async function loadWorldIdentity() {
  try {
    const mem = await api("/worldmodel/navmesh/memory?sizes=0", 8000);
    const cur = mem.current_world || {};
    const list = mem.world_list || [];
    const priors = mem.priors || {};
    const pick = $("worldPick");
    pick.replaceChildren();
    pick.append(new Option("— 已记录的世界 —", ""));
    for (const w of list) {
      const key = w.world_key || w.world_id;
      const pri = priors[w.world_id];
      // 标签以 **key** 打头：名字相同的两个分区在这里必须能区分（2026-10-06 实况：
      // `home` 与 `wrld_home` 是**两个分区**，只显示名字会让人以为选的是同一个世界）。
      const label = `${key}${w.world_name && w.world_name !== key ? `（${w.world_name}）` : ""}` +
        ` · ${w.sessions} 会话${pri ? " · 有先验" : ""}${w.active ? " · 正在写" : ""}`;
      pick.append(new Option(label, key));
    }
    if (cur.world_key) $("worldKeyInput").value = cur.world_key;
    const curW = list.find((w) => w.world_key === cur.world_key);
    // ⚠️ `priors` 是后加的字段：老后端进程不会给。**缺字段 ≠ 没有先验**，
    // 这里必须分开说，否则会像 2026-10-06 那次一样报出"该世界还没有 prior/"这种假话
    // （明明 02:48 就固化好了，只是跑着的进程是旧代码）。
    const priKnown = Object.prototype.hasOwnProperty.call(mem, "priors");
    const pri = priKnown && curW ? (priors[curW.world_id] || null) : null;
    // 同一个世界**按名字另存了一份分区**（world_source=manual_name）时，两份记忆永远不会互相看见，
    // 而"自动识别"只在本分区里找历史 —— 这是最容易踩、也最难看出来的坑。判据只能靠相似（猜），
    // 所以措辞是"看起来像"，并明确给出可执行的动作（切过去）。
    const curKey = String(cur.world_key || "");
    const lookalike = (curKey && !curKey.startsWith("wrld_"))
      ? list.filter((w) => w.world_id !== (curW && curW.world_id) &&
          String(w.world_key || "").toLowerCase().includes(curKey.toLowerCase()))
      : [];
    const twinText = lookalike.length
      ? `｜ ⚠️ 看起来是同一个世界的另一个分区：${lookalike.map((w) =>
          `${w.world_key}（${w.sessions} 会话${priors[w.world_id] ? " · 有先验" : ""}）`).join("、")}` +
        " —— 跨会话检索与先验**只在本分区里找**，要复用就把它设为当前世界并重开会话"
      : "";
    const priText = !curW ? "" : (!priKnown
      ? " · 先验信息未上报（后端进程比 UI 旧，重启后端）"
      : (pri ? ` · 先验 ${pri.occ_cells} 障碍格 / ${pri.free_cells} 自由格（${pri.built_at || "?"}）`
             : " · 该分区还没有 prior/（先跑 offline_fusion.py --write-prior）"));
    const nameWarn = cur.world_source === "manual_name"
      ? "｜ ⚠️ 身份用的是世界**名字**而非 wrld_ 稳定 ID：同名世界会共用记忆，且换回 id 就是另一个分区"
      : "";
    // 身份是**启动时从 world_identity.json 恢复的**（你上次显式设过的值，不是系统猜的）：
    // 说出来，用户才知道现在这个 key 是哪来的、要不要改。
    const restoredWarn = cur.restored
      ? `｜ ↻ 沿用上次设置的身份${cur.restored_wall ? `（${new Date(cur.restored_wall * 1000).toLocaleString()} 保存）` : ""}`
      : "";
    $("worldHint").textContent = cur.world_key
      ? `当前：${cur.world_key}${cur.world_name && cur.world_name !== cur.world_key ? `（${cur.world_name}）` : ""}` +
        `（${cur.world_source || "?"}${curW ? ` · 分区 ${curW.world_id}` : ""}）` +
        (curW ? ` · 该分区 ${curW.sessions} 个会话` : " · 记忆里还没有这个分区的会话") + priText +
        nameWarn + restoredWarn + twinText
      : "未设世界身份：跨会话检索 / 采纳 / 世界先验都不会启动（记忆也不会写）。从上面选一个已记录的世界，或手填 wrld_… 后点「设为当前世界」。";
  } catch (e) {
    $("worldHint").textContent = `读世界列表失败：${e.message}`;
  }
}

if ($("worldPick")) $("worldPick").addEventListener("change", () => {
  if ($("worldPick").value) $("worldKeyInput").value = $("worldPick").value;
});
if ($("worldSet")) $("worldSet").onclick = async () => {
  const key = $("worldKeyInput").value.trim() || $("worldPick").value;
  if (!key) { $("msg").textContent = "先选一个已记录的世界，或填 world_key。"; return; }
  $("msg").textContent = "设置世界身份…";
  try {
    const res = await command("/worldmodel/world", {world_key: key});
    const bad = res.accepted === false;
    $("msg").textContent = bad ? `设置世界身份被拒：${res.reason || ""}` : "世界身份已设置";
    $("worldHint").textContent = bad
      ? `被拒：${res.reason || ""}`
      : "已设置。正在跑的这场会话拿不到它（tracker 在启动时才建）：点「停止」再「启动」，之后跨会话检索与世界先验才会生效。";
  } catch (e) { $("msg").textContent = `设置世界身份失败：${e.message}`; }
  await loadWorldIdentity();
};

// 切图层：立刻重画，不必等下一拍。也清掉 lastPng，否则同一张图会不刷新。
if ($("layer")) $("layer").addEventListener("change", () => { lastPng = null; poll(); });

if ($("memoryLink") && token) $("memoryLink").href = `/navmesh/memory#token=${encodeURIComponent(token)}`;
if ($("consoleLink") && token) $("consoleLink").href = `/ui#token=${encodeURIComponent(token)}`;
if (!token) $("msg").textContent = "URL 里没有 token：用 /navmesh#token=… 打开。";
loadWorldIdentity();
poll();
setInterval(poll, 500);
