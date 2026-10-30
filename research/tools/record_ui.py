# -*- coding: utf-8 -*-
"""policy_obs 录制的本地 UI：选目录、选世界、开录、看实时进度、按世界归类历史会话。

为什么单独一个进程
------------------
录制时后端必须停（OSC 9001 被它占着，见 ``record_policy_obs.py`` / ``OscRecorder``），
所以不能复用 ``backend/process.py`` 的 UI 服务；这里用标准库 ``http.server`` 起一个小服务，
只做三件事：拉起录制子进程、读它的 stdout 进度、把会话目录按世界整理。

复用
----
* 录制本体：``research/tools/record_policy_obs.py``（子进程，``-u`` 逐行输出）
* 配色：``backend/standalone_ui/styles.css``（同一个主题变量，页面里直接 link）
* 世界清单：``navmesh_memory/`` 的目录名 + 历史会话 ``meta.json`` 里的 ``world_key``

会话目录约定
------------
::

    <root>/<YYYYMMDD_HHMMSS>/        L.mp4 R.mp4 frames.jsonl meta.json ...
    meta.json 里的 world_key / world_label 就是「世界归类」，本页可随时改写。

用法
----
::

  .venv/Scripts/python.exe research/tools/record_ui.py --root navmesh_recordings
  # 打开 http://127.0.0.1:8766/
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "research" / "tools" / "record_policy_obs.py"
THEME_CSS = ROOT / "backend" / "standalone_ui" / "styles.css"
NAVMESH_MEMORY = ROOT / "navmesh_memory"


def _json_safe(path: Path) -> str:
    return str(path)


class Rig:
    """录制子进程的唯一持有者：只有它会 spawn/kill，HTTP 线程只调它的方法。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.logs: deque[str] = deque(maxlen=400)
        self.status: dict = {}
        self.done: dict | None = None
        self.session_dir: Path | None = None
        self.stop_file: Path | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.stopping = False
        self.exit_code: int | None = None
        self.error: str | None = None

    # ---- 生命周期 ----
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, payload: dict) -> dict:
        with self._lock:
            if self.running():
                return {"ok": False, "error": "已在录制中，先停止"}
            self.logs.clear()
            self.status = {}
            self.done = None
            self.error = None
            self.exit_code = None
            self.stopping = False
            self.finished_at = None
            self.started_at = time.perf_counter()

            root = Path(payload.get("root") or self.root).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            session = root / stamp
            # 停止文件不能落在会话目录里：那是数据目录，不该混入控制文件。
            self.stop_file = root / ".record_ui_stop"
            self.stop_file.unlink(missing_ok=True)

            cmd = [
                sys.executable, "-u", str(RECORDER),
                "--out", str(session),
                "--seconds", str(float(payload.get("seconds", 480))),
                "--fps", str(float(payload.get("fps", 30))),
                "--gq", str(int(payload.get("gq", 16))),
                "--target-width", str(int(payload.get("width", 1080))),
                "--queue", str(int(payload.get("queue", 24))),
                "--world-key", str(payload.get("world_key", "")),
                "--world-label", str(payload.get("world_label", "")),
                "--stop-file", str(self.stop_file),
            ]
            if payload.get("selftest"):
                cmd.append("--selftest")      # selftest 内部本来就不监听 OSC
            elif payload.get("no_osc"):
                cmd.append("--no-osc")
            else:
                cmd += ["--osc-port", str(int(payload.get("osc_port", 9001)))]

            self.session_dir = session
            env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", env={**os.environ, **env},
                )
            except Exception as exc:  # noqa: BLE001 - 起不来就把原因原样给 UI
                self.error = f"{type(exc).__name__}: {exc}"
                return {"ok": False, "error": self.error}
            threading.Thread(target=self._reader, name="record-ui-reader", daemon=True).start()
            return {"ok": True, "session": _json_safe(session)}

    def _reader(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if line.startswith("[status] "):
                try:
                    self.status = json.loads(line[len("[status] "):])
                except json.JSONDecodeError:
                    pass
            elif line.startswith("[done] "):
                try:
                    self.done = json.loads(line[len("[done] "):])
                except json.JSONDecodeError:
                    pass
            self.logs.append(line)
        self.exit_code = proc.wait()
        self.finished_at = time.perf_counter()
        self.stopping = False
        # 收尾后清掉停止标志，否则下一场一开始就会立刻被它停掉。
        try:
            if self.stop_file is not None:
                self.stop_file.unlink(missing_ok=True)
        except OSError:
            pass

    def stop(self) -> dict:
        with self._lock:
            if not self.running():
                return {"ok": False, "error": "录制已结束（或还没开始），不用再停"}
            if self.stopping:
                return {"ok": True, "note": "已在收尾中"}
            if self.stop_file is None:
                return {"ok": False, "error": "没有停止标志路径"}
            try:
                self.stop_file.write_text("stop", encoding="ascii")
            except OSError as exc:
                return {"ok": False, "error": f"写停止标志失败：{exc}"}
            self.stopping = True
        threading.Thread(target=self._stop_watchdog, name="record-ui-stop", daemon=True).start()
        return {"ok": True, "note": "已请求收尾，等 ffmpeg 收流（通常 3–8 s）"}

    def _stop_watchdog(self, wait_s: float = 30.0) -> None:
        """停止标志已写但进程迟迟不退（编码器卡住）时兜底，避免 UI 永远停在「收尾中」。"""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < wait_s:
            if not self.running():
                return
            time.sleep(0.5)
        proc = self.proc
        if proc is not None and proc.poll() is None:
            self.error = f"收尾超时 {int(wait_s)} s，已强制结束（meta.json 可能没写全）"
            proc.terminate()

    def snapshot(self) -> dict:
        if self.started_at is None:
            elapsed = None
        elif not self.running() and self.finished_at is not None:
            # 进程结束后冻结计时：以前这里一直用 perf_counter，页面会永远在「计时」。
            elapsed = round(self.finished_at - self.started_at, 1)
        else:
            elapsed = round(time.perf_counter() - self.started_at, 1)
        return {
            "running": self.running(),
            "stopping": self.stopping,
            "finished": self.finished_at is not None and not self.running(),
            "session": None if self.session_dir is None else _json_safe(self.session_dir),
            "elapsed": elapsed, "status": self.status, "done": self.done,
            "exit_code": self.exit_code, "error": self.error,
            "log": list(self.logs)[-60:],
        }


def _scan_sessions(root: Path) -> list[dict]:
    """读 root 下每个会话目录的 meta.json。缺 meta 的（正在录）也列出来，标 recording。"""
    out: list[dict] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("."):
            continue
        meta_path = d / "meta.json"
        row = {"dir": _json_safe(d), "session": d.name, "world_key": "", "world_label": "",
               "frames": None, "rate": None, "mb": None, "status": "recording"}
        if meta_path.is_file():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                m = {}
            row.update({
                "world_key": str(m.get("world_key") or ""),
                "world_label": str(m.get("world_label") or ""),
                "frames": m.get("frames"), "rate": m.get("rate_hz"),
                "status": "ok" if m.get("frames") else "empty",
                "selftest": bool(m.get("selftest")),
            })
        mb = sum(p.stat().st_size for p in (d / "L.mp4", d / "R.mp4") if p.is_file())
        row["mb"] = round(mb / 1e6, 1) if mb else 0.0
        out.append(row)
    return out


def _known_worlds(roots: list[Path] | None = None) -> list[str]:
    """已存在的世界标识：navmesh 记忆目录名 + 历史会话 meta 里的 world_key。

    只扫记忆目录的话，UI 里改成自定义世界（例如 home）后下拉框里找不到它，
    表格会显示成「未归类」；所以会话侧的世界也要并进来。
    """
    names: set[str] = set()
    if NAVMESH_MEMORY.is_dir():
        for d in NAVMESH_MEMORY.iterdir():
            if d.is_dir():
                names.add(d.name)
    for root in roots or []:
        for row in _scan_sessions(root):
            key = str(row.get("world_key") or "")
            if key:
                names.add(key)
    return sorted(names)


def _assign_world(session_dir: Path, key: str, label: str) -> dict:
    meta_path = session_dir / "meta.json"
    if not meta_path.is_file():
        return {"ok": False, "error": f"{session_dir} 里没有 meta.json（还没录完？）"}
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["world_key"] = key
    data["world_label"] = label or key
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    run_path = session_dir / "run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["world_key"] = key
        run["world_label"] = label or key
        run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


def _pick_dir(initial: str) -> str:
    """Windows 原生选目录框。放在 HTTP 线程里跑，ThreadingHTTPServer 不会阻塞其它请求。"""
    safe = (initial or "").replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
        f"if ('{safe}') {{ $d.SelectedPath = '{safe}' }};"
        "$d.Description = '选择录制输出根目录';"
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }"
    )
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps],
                             capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return (res.stdout or "").strip()


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>policy_obs 录制</title>
<link rel="stylesheet" href="/styles.css">
<style>
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }
  .field { display:flex; flex-direction:column; gap:6px; }
  .field label { color:var(--muted); font-size:12px; letter-spacing:.04em; }
  .field input, .field select { padding:9px 11px; border-radius:10px; border:1px solid var(--line);
    background:rgba(8,10,15,.72); color:var(--text); }
  .field .row { display:flex; gap:8px; }
  .field .row input { flex:1; min-width:0; }
  .metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .metric { min-height:96px; }
  .metric strong { font-variant-numeric:tabular-nums; }
  .table { width:100%; border-collapse:collapse; font-size:13px; }
  .table th, .table td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }
  .table th { color:var(--muted); font-weight:600; font-size:12px; }
  .table select, .table input { padding:5px 7px; border-radius:8px; border:1px solid var(--line);
    background:rgba(8,10,15,.72); color:var(--text); font-size:12px; }
  .table td.num { font-variant-numeric:tabular-nums; color:var(--muted); }
  pre.log { max-height:340px; overflow:auto; background:rgba(0,0,0,.35); border:1px solid var(--line);
    border-radius:12px; padding:12px; font-size:12px; line-height:1.5; color:var(--muted); }
  button { padding:9px 15px; border-radius:10px; border:1px solid var(--line);
    background:rgba(255,255,255,.05); color:var(--text); }
  button.primary { background:var(--cyan); color:#071012; border-color:transparent; font-weight:750; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .hint { color:var(--muted); font-size:12px; margin:6px 0 0; }
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div>
      <p class="eyebrow">N.E.K.O · VRChat 具身 agent</p>
      <h1>policy_obs 录制</h1>
    </div>
    <div class="top-actions">
      <span class="badge" id="badge">空闲</span>
      <button id="btnRevealAll">打开录制根目录</button>
    </div>
  </header>

  <section class="panel">
    <div class="grid">
      <div class="field" style="grid-column:span 2">
        <label>输出根目录</label>
        <div class="row">
          <input id="root" value="__ROOT__">
          <button id="btnBrowse">浏览…</button>
        </div>
        <p class="hint">每个会话写成 <code>&lt;根目录&gt;/&lt;时间戳&gt;/</code>，世界归类写在 meta.json。</p>
      </div>
      <div class="field">
        <label>世界（已有）</label>
        <select id="worldSel"></select>
      </div>
      <div class="field">
        <label>世界标识（可新建）</label>
        <input id="worldKey" placeholder="wrld_xxxx 或自定义名">
      </div>
      <div class="field">
        <label>世界显示名（可选）</label>
        <input id="worldLabel" placeholder="例如：家 / Club">
      </div>
      <div class="field"><label>时长（秒）</label><input id="seconds" type="number" value="480" min="5"></div>
      <div class="field"><label>目标帧率</label><input id="fps" type="number" value="30" min="5" max="60"></div>
      <div class="field"><label>AV1 质量 gq</label><input id="gq" type="number" value="16" min="1" max="40"></div>
      <div class="field"><label>采集宽度下限</label><input id="width" type="number" value="1080" min="360" max="2880"></div>
      <div class="field"><label>编码队列（帧对）</label><input id="queue" type="number" value="24" min="2" max="64"></div>
      <div class="field">
        <label>模式</label>
        <select id="mode">
          <option value="real">真机（需要 SteamVR，后端须停）</option>
          <option value="selftest">编码自检（合成帧，不连 VR）</option>
        </select>
      </div>
    </div>
    <div class="button-row" style="margin-top:16px">
      <button class="primary" id="btnStart">开始录制</button>
      <button id="btnStop" disabled>停止</button>
      <button id="btnRefresh">刷新会话</button>
    </div>
    <p class="hint" id="hint"></p>
  </section>

  <section class="panel">
    <div class="panel-heading"><h2>实时</h2><span class="muted" id="sessionPath">—</span></div>
    <div class="metric-grid" style="margin-top:14px">
      <div class="metric"><span>已用时</span><strong id="mElapsed">—</strong></div>
      <div class="metric"><span>帧数</span><strong id="mFrames">—</strong></div>
      <div class="metric"><span>实际帧率</span><strong id="mRate">—</strong></div>
      <div class="metric"><span>抓帧中位</span><strong id="mGrab">—</strong></div>
      <div class="metric"><span>丢帧（空/慢）</span><strong id="mDrop">—</strong></div>
    </div>
    <pre class="log" id="log">（尚无输出）</pre>
  </section>

  <section class="panel">
    <div class="panel-heading"><h2>会话与世界归类</h2><span class="muted" id="count">—</span></div>
    <table class="table" style="margin-top:12px">
      <thead><tr><th>会话</th><th>世界</th><th>帧</th><th>帧率</th><th>体积</th><th>状态</th><th></th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
</div>

<script>
const $ = id => document.getElementById(id);
let timer = null;

async function api(path, body) {
  const opt = body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)};
  const res = await fetch(path, opt);
  return res.json();
}

async function loadWorlds() {
  const d = await api("/api/worlds");
  const sel = $("worldSel");
  sel.innerHTML = '<option value="">（不指定 / 手动输入）</option>' +
    (d.worlds || []).map(w => `<option value="${w}">${w}</option>`).join("");
  if (!$("root").value && d.root) $("root").value = d.root;
}

async function loadSessions() {
  const root = $("root").value.trim();
  const d = await api("/api/sessions?root=" + encodeURIComponent(root));
  const rows = d.sessions || [];
  $("count").textContent = `${rows.length} 个会话（${(d.bytes_mb||0).toFixed(1)} MB）`;
  $("rows").innerHTML = rows.map(r => `
    <tr>
      <td><code>${r.session}</code></td>
      <td>
        <select data-dir="${r.dir}" class="worldEdit">
          <option value="">（未归类）</option>
          ${(d.worlds||[]).map(w => `<option value="${w}" ${w===r.world_key?"selected":""}>${w}</option>`).join("")}
        </select>
      </td>
      <td class="num">${r.frames ?? "—"}</td>
      <td class="num">${r.rate ? r.rate.toFixed(2)+" Hz" : "—"}</td>
      <td class="num">${r.mb != null ? r.mb+" MB" : "—"}</td>
      <td>${r.status === "ok" ? '<span class="badge ok">完成</span>' : `<span class="badge">${r.status}</span>`}</td>
      <td><button data-reveal="${r.dir}">打开</button></td>
    </tr>`).join("");
  document.querySelectorAll(".worldEdit").forEach(el => el.onchange = async () => {
    await api("/api/assign", {dir: el.dataset.dir, world_key: el.value, world_label: el.value});
    loadSessions(); loadWorlds();
  });
  document.querySelectorAll("[data-reveal]").forEach(el => el.onclick = () => api("/api/reveal", {dir: el.dataset.reveal}));
}

async function poll() {
  const s = await api("/api/state");
  const st = s.status || {};
  const state = s.stopping ? "收尾中" : (s.running ? "录制中" : (s.finished ? "已结束" : "空闲"));
  $("badge").textContent = state;
  $("badge").className = "badge " + (s.stopping ? "starting" : (s.running ? "starting" : (s.finished ? "ok" : "")));
  $("sessionPath").textContent = s.session || "—";
  $("mElapsed").textContent = s.elapsed != null ? s.elapsed + " s" : "—";
  $("mFrames").textContent = st.frames != null ? st.frames : "—";
  $("mRate").textContent = st.rate != null ? st.rate.toFixed(2) + " Hz" : "—";
  $("mGrab").textContent = st.grab_ms != null ? st.grab_ms.toFixed(1) + " ms" : "—";
  $("mDrop").textContent = st.dropped_queue != null ? `${st.dropped_blank} / ${st.dropped_queue}` : "—";
  $("log").textContent = (s.log || []).join("\\n") || "（尚无输出）";
  $("log").scrollTop = $("log").scrollHeight;
  $("btnStart").disabled = s.running || s.stopping;
  $("btnStop").disabled = !s.running || s.stopping;
  if (s.done) $("hint").textContent = `完成：${s.done.frames} 对帧，${s.done.rate} Hz，${s.done.mb} MB`;
  if (s.error) $("hint").textContent = "问题：" + s.error;
  // 一次收尾后刷新会话表；不用每轮都扫盘。
  if (s.finished && !lastFinished) { lastFinished = true; loadSessions(); }
  if (s.running) lastFinished = false;
  clearTimeout(timer);
  timer = setTimeout(poll, 1000);
}
let lastFinished = false;

$("btnBrowse").onclick = async () => {
  const d = await api("/api/pick_dir", {initial: $("root").value.trim()});
  if (d.path) { $("root").value = d.path; loadSessions(); }
};
$("worldSel").onchange = () => { if ($("worldSel").value) $("worldKey").value = $("worldSel").value; };
$("btnStart").onclick = async () => {
  const key = $("worldKey").value.trim() || $("worldSel").value.trim();
  const d = await api("/api/start", {
    root: $("root").value.trim(), world_key: key, world_label: $("worldLabel").value.trim(),
    seconds: +$("seconds").value, fps: +$("fps").value, gq: +$("gq").value,
    width: +$("width").value, queue: +$("queue").value,
    selftest: $("mode").value === "selftest",
  });
  $("hint").textContent = d.ok ? "已启动，等待首帧…" : ("启动失败：" + d.error);
  poll();
};
$("btnStop").onclick = async () => {
  const d = await api("/api/stop", {});
  $("hint").textContent = d.ok ? (d.note || "已请求停止") : ("停止未生效：" + d.error);
  poll();
};
$("btnRefresh").onclick = () => { loadSessions(); };
$("btnRevealAll").onclick = () => api("/api/reveal", {dir: $("root").value.trim()});

loadWorlds(); loadSessions(); poll();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    rig: Rig = None  # 由 main() 注入
    default_root: Path = ROOT / "navmesh_recordings"

    def log_message(self, fmt: str, *args) -> None:  # 静音默认访问日志
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的固定名字
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            # 路径进 HTML 属性，按属性转义即可（Windows 反斜杠在 HTML 里按原样显示，不要双重转义）。
            html = PAGE.replace("__ROOT__", escape(str(self.default_root), quote=True))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if u.path == "/styles.css":
            self._send(200, THEME_CSS.read_bytes(), "text/css; charset=utf-8")
            return
        if u.path == "/api/state":
            self._json(self.rig.snapshot())
            return
        if u.path == "/api/worlds":
            self._json({"worlds": _known_worlds([self.default_root]),
                        "root": _json_safe(self.default_root)})
            return
        if u.path == "/api/sessions":
            q = parse_qs(u.query)
            root = Path(q.get("root", [str(self.default_root)])[0] or self.default_root)
            rows = _scan_sessions(root)
            self._json({"sessions": rows, "worlds": _known_worlds([root, self.default_root]),
                        "bytes_mb": round(sum(r["mb"] or 0 for r in rows), 1)})
            return
        self._json({"ok": False, "error": "unknown route"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            payload = {}
        if u.path == "/api/start":
            self._json(self.rig.start(payload))
            return
        if u.path == "/api/stop":
            self._json(self.rig.stop())
            return
        if u.path == "/api/pick_dir":
            self._json({"path": _pick_dir(str(payload.get("initial") or ""))})
            return
        if u.path == "/api/assign":
            d = Path(str(payload.get("dir") or ""))
            if not d.is_dir():
                self._json({"ok": False, "error": "目录不存在"})
                return
            self._json(_assign_world(d, str(payload.get("world_key") or ""),
                                     str(payload.get("world_label") or "")))
            return
        if u.path == "/api/reveal":
            d = Path(str(payload.get("dir") or ""))
            if d.is_dir():
                subprocess.Popen(["explorer", "/select,", str(d)])
                self._json({"ok": True})
            else:
                self._json({"ok": False, "error": "目录不存在"})
            return
        self._json({"ok": False, "error": "unknown route"}, 404)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="policy_obs 录制的本地 UI")
    ap.add_argument("--root", default=str(ROOT / "navmesh_recordings"), help="默认录制根目录")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    Handler.rig = Rig(root)
    Handler.default_root = root
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"录制 UI: {url}  （根目录 {root}）")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止 UI")
    finally:
        httpd.server_close()
        if Handler.rig.running():
            Handler.rig.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
