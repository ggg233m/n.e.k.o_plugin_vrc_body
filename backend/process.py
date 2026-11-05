"""独立回环后端进程入口。

这里刻意设计为直接执行文件而不是使用 ``python -m``：否则后端进程导入插件
包时会要求 N.E.K.O SDK。下面的轻量命名空间包引导只加载核心模块。
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import math
import os
from pathlib import Path
import secrets
import signal
import site
import socket
import sys
import threading
import time
import tomllib
from collections.abc import Mapping
from typing import Any
import types
from urllib.parse import parse_qs, urlsplit


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
_FALLBACK_PACKAGE_NAME = "neko_anyadance_body"


def _toml_string(path: Path, *keys: str) -> str:
    try:
        current: Any = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return current.strip() if isinstance(current, str) else ""


def resolve_plugin_package_name(*, package: str | None, project_dir: Path) -> str:
    """独立 ``python process.py`` 必须用合法包名，不能用带点的 git 目录名。"""
    if package not in {None, ""}:
        parent = package.rsplit(".", 1)[0]
        if parent:
            return parent
    for candidate in (
        _toml_string(project_dir / "pyproject.toml", "project", "name"),
        _toml_string(project_dir / "plugin.toml", "plugin", "id"),
        project_dir.name,
        _FALLBACK_PACKAGE_NAME,
    ):
        if candidate.isidentifier():
            return candidate
    return _FALLBACK_PACKAGE_NAME


PACKAGE_NAME = resolve_plugin_package_name(package=__package__, project_dir=PROJECT_DIR)
# 后端是独立子进程，不继承宿主为插件添加的 sys.path。
# 保留宿主已有依赖的优先级，仅把安装包内的依赖作为补充。
#
# onnxruntime 刻意不进 vendor：宿主自带一份，两份不同版本的 onnxruntime.dll
# 落进同一进程时，后加载的那份 DllMain 会返回 1114（"动态链接库初始化例程失
# 败"），检测器整条降级到 OpenCV DNN。统一用宿主那份，唯一一份 DLL。
VENDOR_DIR = PROJECT_DIR / "vendor"
if VENDOR_DIR.is_dir():
    site.addsitedir(str(VENDOR_DIR))
if __package__ in {None, ""}:
    parent = str(PROJECT_DIR.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    namespace = types.ModuleType(PACKAGE_NAME)
    namespace.__path__ = [str(PROJECT_DIR)]
    sys.modules[PACKAGE_NAME] = namespace

BackendService = importlib.import_module(f"{PACKAGE_NAME}.backend.service").BackendService  # noqa: E402
webui_module = importlib.import_module(f"{PACKAGE_NAME}.backend.webui")  # noqa: E402
StandaloneConfigStore = webui_module.StandaloneConfigStore
UI_DIRECTORY = webui_module.UI_DIRECTORY
deep_merge = webui_module.deep_merge
load_settings_file = webui_module.load_settings_file
# 图层白名单必须**模块级**取：脚本模式（python backend/process.py）下本文件是 __main__、
# 没有父包，函数里的 ``from .nav_online import ...`` 会 ImportError 且只炸那一个端点
# （2026-10-05 实况：/worldmodel/navmesh 空回复 ⇒ navmesh 页面全空）。这里沿用文件顶部
# 既有的 importlib + PACKAGE_NAME 模式，两种启动方式都成立。
nav_online_module = importlib.import_module(f"{PACKAGE_NAME}.backend.nav_online")  # noqa: E402
_GRID_LAYERS = nav_online_module._GRID_LAYERS

# 空闲 keep-alive 连接的读超时。控制面板会留下几条不再发请求的长连接，没有这个
# 上界，请求线程就一直阻塞在 rfile.readline() 上——见 BackendHttpServer 里
# block_on_close 的注释：它们曾经把整条退出路径挂死。
_KEEPALIVE_TIMEOUT_S = 10.0
# 从「收到停机请求」到「进程必须消失」的硬上限。拆卸链路里有第三方库的 join、
# 子进程 wait 和若干把锁，任何一个失手都不该让用户只能去任务管理器杀进程。
_SHUTDOWN_HARD_LIMIT_S = 20.0
# 信号编号 → 日志里的短名。Windows 上 SIGBREAK 是 Ctrl+Break / 关控制台窗口，
# 语义与 Ctrl+C 不同，但在这里都只表示同一件事：请求一次优雅停机。
_SIGNAL_NAMES = {
    getattr(signal, name): name
    for name in ("SIGINT", "SIGTERM", "SIGBREAK")
    if hasattr(signal, name)
}


def _json_safe(value: Any) -> Any:
    """递归把非有限浮点数转换为 JSON 标准允许的 null。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _log_event(event: str, **fields: Any) -> None:
    """退出路径上的结构化事件日志。

    刻意直接写 stderr 而不是走 logging：拆卸途中 handler 可能已经被关掉、配置也
    可能被改坏；「进程为什么没退」这类证据不能依赖任何会失效的机制。stdout 留给
    将来可能的管道协议，这里一律走 stderr，与 ready 那一行同一个通道。
    """
    try:
        print(
            json.dumps({"event": event, **fields}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
    except (OSError, ValueError, TypeError):
        # 连 stderr 都写不进去时没有第二选择：退出路径不能因为日志抛异常。
        pass


def _force_exit(code: int, *, reason: str) -> None:
    """不再等拆卸，直接结束进程（先尽力带走 LAS2 子进程）。"""
    _log_event("forced_exit", reason=reason)
    try:
        stereo = importlib.import_module(f"{PACKAGE_NAME}.backend.stereo_backend")
        closer = getattr(stereo, "close_las2", None)
        if callable(closer):
            worker = threading.Thread(target=closer, name="backend-las2-close", daemon=True)
            worker.start()
            # 硬退路径只给它很短的时间：超时就带着子进程一起走。stereo_worker 自己
            # 有心跳看门狗（父进程消失后自行退出），所以这里不值得再拖。
            worker.join(timeout=2.0)
    except Exception:  # noqa: BLE001 - 退出路径不能因为清理失败卡住
        pass
    os._exit(code)


def _exit_process(code: int) -> None:
    """拆卸完成后强制结束解释器。

    为什么不能靠 return / sys.exit：解释器关闭时会 join 所有 **非 daemon** 线程，
    而 concurrent.futures 的 ThreadPoolExecutor 正是这一类（双目、推理、捕获链路里
    的第三方库常用它）。2026-11-05 实测：拆卸已经全部完成、backend_exited 都打出来
    了，进程还要再挂 14 s，最后是被看门狗带走的 —— 用户看到的就是「点了关停，
    进程还在」。os._exit 跳过这段等待：该刷的在上面都用 flush=True 打完了，
    LAS2 子进程由 job 句柄（KILL_ON_JOB_CLOSE）在内核里一并带走。
    """
    alive = [thread.name for thread in threading.enumerate() if thread is not threading.main_thread()]
    _log_event("exiting", code=code, threads=alive[:12], thread_count=len(alive))
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)


def _arm_shutdown_watchdog(limit_s: float) -> None:
    """起一条看门狗：limit_s 秒后进程还在，就硬退。

    为什么必须是 os._exit 而不是等正常返回：用户按下「关停后端」之后进程还赖在
    那儿，比少一行日志难受得多。宁可以 0 退出码硬退并留一条 forced_exit 记录，
    也不能让「关不掉」重新变成一个需要去任务管理器的事件。
    """
    def run() -> None:
        time.sleep(max(0.5, float(limit_s)))
        _force_exit(0, reason="shutdown_watchdog")

    threading.Thread(target=run, name="backend-shutdown-watchdog", daemon=True).start()


def _install_shutdown_signals(server: "BackendHttpServer") -> None:
    """把 Ctrl+C / Ctrl+Break / SIGTERM 统一收成一次优雅停机。

    Windows 上 SIGINT 默认会在主线程抛 KeyboardInterrupt，而 serve_forever 的
    finally 里正等着拆卸——那条路 2026-11-05 实测会卡死，所以这里自己接管：
    信号只负责「请求停机」，真正的拆卸顺序仍然留在主线程的 finally 里。
    第二次按 Ctrl+C 不再等，直接硬退：连按两次时用户想表达的就是「立刻」。
    """
    def handler(signum: int, _frame: Any) -> None:
        name = _SIGNAL_NAMES.get(signum, str(signum))
        if server.shutting_down.is_set():
            _force_exit(0, reason=f"second_signal:{name}")
            return
        server.request_stop(f"signal:{name}")

    for signum in _SIGNAL_NAMES:
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError, RuntimeError):
            # 不在主线程 / 平台不支持：跳过，KeyboardInterrupt 分支仍是退路。
            continue


_UI_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/ui": ("index.html", "text/html; charset=utf-8"),
    "/ui/": ("index.html", "text/html; charset=utf-8"),
    "/ui/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/ui/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/ui/navmesh.js": ("navmesh.js", "text/javascript; charset=utf-8"),
    "/ui/navmesh.css": ("navmesh.css", "text/css; charset=utf-8"),
    # 在线增量 navmesh：镜像双目 + 航位推算 + 回环；goto 需点击后确认，驾驶另需自主武装。
    "/navmesh": ("navmesh.html", "text/html; charset=utf-8"),
    # navmesh 记忆管理：按世界列会话、看缩略图、钉住/标签/删除/配额清理。
    "/navmesh/memory": ("navmesh_memory.html", "text/html; charset=utf-8"),
    "/ui/navmesh_memory.js": ("navmesh_memory.js", "text/javascript; charset=utf-8"),
    # 覆盖俯视页：伴生覆盖网格（观测计数/近看质量）+ 洞分类 + 诚实进度指标；只读，驾驶仍去 /navmesh。
    "/coverage": ("coverage.html", "text/html; charset=utf-8"),
    "/ui/coverage.js": ("coverage.js", "text/javascript; charset=utf-8"),
    "/ui/coverage.css": ("coverage.css", "text/css; charset=utf-8"),
}


class BackendRequestHandler(BaseHTTPRequestHandler):
    server: "BackendHttpServer"
    # 保持回环控制连接，避免重复的 OSC/动作调用每次都进行 TCP 握手。
    protocol_version = "HTTP/1.1"
    # 空闲 keep-alive 的读超时（StreamRequestHandler.setup 会把它设到 socket 上）。
    # 这是「请求线程为什么能自己结束」的前提：没有它，readline() 会一直等下一行
    # 请求头，面板留下的一条空连接就能把线程钉到进程退出为止。
    timeout = _KEEPALIVE_TIMEOUT_S

    def setup(self) -> None:
        super().setup()
        # 登记到 server：停机时要主动掐断，不能指望每条连接都自己读到 EOF 再走。
        self.server.register_connection(self.connection)

    def finish(self) -> None:
        try:
            super().finish()
        except OSError:
            # 客户端先走一步是常态；这里再抛只会让拆卸路径多一个失败源。
            pass
        finally:
            self.server.unregister_connection(self.connection)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        if self.headers.get("X-Neko-Backend-Token") == self.server.token:
            return True
        # <img> 无法附加自定义 header；仅对回环 MJPEG 预览允许查询参数令牌，
        # 其他接口仍必须使用 header，避免扩大控制 API 的认证面。
        if urlsplit(self.path).path == "/vision/mjpeg":
            return parse_qs(urlsplit(self.path).query).get("token", [None])[0] == self.server.token
        return False

    def _mjpeg(self) -> None:
        """以最新帧单槽输出 30 FPS MJPEG，避免 JSON/base64 轮询卡住页面。"""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            # 这条流是按 FPS 无限推送的，只有客户端断连才会抛 OSError 退出；停机时
            # 客户端往往还连着，所以必须自查停止位，否则请求线程会一直被
            # server_close() 等待（见 BackendHttpServer.block_on_close 的注释）。
            while not self.server.shutting_down.is_set():
                result = self.server.service.vision.latest_frame(max_age_ms=1000, overlay=False)
                data = result.get("data") if isinstance(result, dict) else None
                if isinstance(result, dict) and result.get("available") and isinstance(data, bytes):
                    packet = (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
                        + data + b"\r\n"
                    )
                    self.wfile.write(packet)
                    self.wfile.flush()
                time.sleep(1.0 / 30.0)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return
        # multipart 响应已经在头里声明了 Connection: close，这里补上同一个意图，
        # 让 handle_one_request 的循环不再进入下一条请求。
        self.close_connection = True

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        # JSON 标准不允许 NaN/Infinity；深度模型在“没有非地面点”时可能产生
        # nearest_nonfloor_m=NaN，统一在 HTTP 边界降级为 null，不能让整个页面
        # 因 JSON.parse 失败而不可用。
        body = json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _serve_ui(self, path: str) -> bool:
        asset = _UI_ASSETS.get(path)
        if asset is None:
            return False
        filename, content_type = asset
        try:
            body = (UI_DIRECTORY / filename).read_bytes()
        except OSError:
            self._json(503, {"error": "standalone UI assets are unavailable"})
            return True
        self._bytes(200, body, content_type)
        return True

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 2 * 1024 * 1024:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        # 静态外壳不包含凭据，可以先加载；所有状态和控制 API 仍要求 token。
        if self._serve_ui(path):
            return
        if path == "/favicon.ico":
            self._bytes(204, b"", "image/x-icon")
            return
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        if path == "/vision/mjpeg":
            self._mjpeg()
            return
        if path == "/health":
            # stopping 让探测方（插件侧 BackendClient、测试、运维）能分清
            # 「还没起来」和「正在关闭」——这两者的重试策略正好相反。
            self._json(200, {
                "ok": True,
                "pid": os.getpid(),
                "stopping": self.server.shutting_down.is_set(),
            })
            return
        if path == "/worldmodel/status":
            self._json(200, self.server.service.world_model_status())
            return
        if path == "/worldmodel/navmesh":
            q = parse_qs(urlsplit(self.path).query)
            grid = q.get("grid", ["0"])[0]
            # 图层名来自查询串，**未白名单则回落到默认**而不是让后端抛 ValueError：
            # 一个手输的 ?layer=xxx 不该把整个状态轮询打断。真正的校验在
            # OnlineNavigator.grid_view 里，它对未知图层仍会抛。
            layer = str(q.get("layer", ["tristate"])[0])
            if layer not in _GRID_LAYERS:
                layer = "tristate"
            self._json(200, self.server.service.navmesh_status(
                include_grid=str(grid).lower() in {"1", "true", "yes"}, layer=layer))
            return
        if path == "/worldmodel/navmesh/coverage":
            self._json(200, self.server.service.navmesh_coverage())
            return
        if path.startswith("/worldmodel/navmesh/memory"):
            q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
            svc = self.server.service
            if path == "/worldmodel/navmesh/memory":
                # sizes=0：不递归扫目录算体积（上万个文件、秒级）。管理页要那一列体积才用默认档。
                self._json(200, svc.navmesh_memory_summary(sizes=q.get("sizes", "1") != "0"))
            elif path == "/worldmodel/navmesh/memory/sessions":
                self._json(200, svc.navmesh_memory_sessions(q.get("world")))
            elif path == "/worldmodel/navmesh/memory/session":
                self._json(200, svc.navmesh_memory_session(q.get("world"), q.get("session")))
            elif path == "/worldmodel/navmesh/memory/thumb":
                try:
                    body = svc.navmesh_memory_thumbnail(q.get("world"), q.get("session"), q.get("k"))
                except (KeyError, ValueError, TypeError) as exc:
                    self._json(404, {"error": str(exc)})
                    return
                self._bytes(200, body, "image/jpeg")
            else:
                self._json(404, {"error": "unknown endpoint"})
            return
        if path == "/config":
            self._json(200, self.server.config_store.snapshot())
            return
        if path in {"/snapshot", "/state"}:
            self._json(200, self.server.service.snapshot())
            return
        if path == "/awareness":
            self._json(200, self.server.service.awareness())
            return
        if path == "/cognition":
            self._json(200, self.server.service.cognition.snapshot())
            return
        if path == "/perception":
            self._json(200, self.server.service.perception())
            return
        if path == "/autonomy":
            self._json(200, self.server.service.autonomy_snapshot())
            return
        if path == "/policy/state":
            self._json(200, self.server.service.policy_state())
            return
        if path == "/world/delta":
            query = parse_qs(urlsplit(self.path).query)
            def query_int(name: str, default: int) -> int:
                try:
                    return int(query.get(name, [default])[0])
                except (TypeError, ValueError, OverflowError):
                    return default
            self._json(
                200,
                self.server.service.world_delta(
                    query_int("after_revision", 0),
                    wait_ms=query_int("wait_ms", 250),
                    limit=query_int("limit", 16),
                ),
            )
            return
        if path == "/vision/frame":
            frame_query = parse_qs(urlsplit(self.path).query)
            try:
                max_age_ms = int(frame_query.get("max_age_ms", [3000])[0])
            except (TypeError, ValueError, OverflowError):
                max_age_ms = 3000
            overlay = str(frame_query.get("overlay", ["0"])[0]).lower() in {"1", "true", "yes"}
            self._json(
                200,
                self.server.service.vision_frame(max_age_ms=max_age_ms, overlay=overlay),
            )
            return
        if path == "/semantic/request":
            semantic_query = parse_qs(urlsplit(self.path).query)
            after_request_id = semantic_query.get("after_request_id", [None])[0]
            self._json(
                200,
                self.server.service.main_llm_semantic_request(after_request_id),
            )
            return
        self._json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        started_at = time.perf_counter()
        # 请求期间记录客户端地址，供 turn 账本归因到具体连接。
        # 放 finally 里清掉：ThreadingHTTPServer 每请求一线程，串扰窗口极小，
        # 但导航器路径读到陈旧地址会污染账本，宁可清。
        self.server.service._http_client_addr = "%s:%s" % self.client_address[:2]
        # 停机请求只在这里登记，等响应写完之后才动手：request_stop() 会主动掐断
        # 所有在途连接，先关的话正在回话的这一条也一起断，客户端只会看到
        # RemoteDisconnected——分不清「已经停了」和「崩了」。
        shutdown_reason: str | None = None
        try:
            value = self._read_json()
            if self.path == "/config":
                result = self.server.config_store.save(value.get("config"))
            elif self.path == "/action":
                result = self.server.service.submit(
                    str(value.get("kind") or ""),
                    value.get("params") or {},
                    preconditions=value.get("preconditions"),
                )
            elif self.path == "/osc/parameter":
                result = self.server.service.send_avatar_parameter(str(value.get("name") or ""), value.get("value"))
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/input":
                result = self.server.service.pulse_input(
                    str(value.get("action") or ""),
                    str(value.get("side") or ""),
                    value.get("hold_ms", 100),
                )
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/locomotion":
                result = self.server.service.set_locomotion(
                    value.get("vertical", 0),
                    value.get("horizontal", 0),
                    value.get("duration_ms", 1000),
                )
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/turn":
                result = self.server.service.set_turn(
                    value.get("horizontal"),
                    value.get("duration_ms", 500),
                    source="http:/osc/turn",
                )
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/stop_movement":
                result = self.server.service.stop_movement()
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/chatbox":
                result = self.server.service.send_chatbox(
                    value.get("text", ""),
                    value.get("immediate", True),
                )
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/batch":
                result = self.server.service.send_osc_batch(value.get("commands"))
            elif self.path == "/osc/cancel":
                self.server.service.cancel_inputs()
                result = {"accepted": True}
            elif self.path == "/input/axes":
                result = self.server.service.set_controller_axes(
                    value.get("side"),
                    value.get("x", 0.0),
                    value.get("y", 0.0),
                    value.get("duration_ms", 1000),
                )
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/input/button":
                result = self.server.service.set_controller_button(
                    value.get("side"),
                    value.get("button"),
                    value.get("pressed", True),
                    value.get("hold_ms", 100),
                    value.get("value", 1.0),
                )
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/input/release":
                result = self.server.service.release_controller_inputs(value.get("side", "all"))
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/autonomy/arm":
                result = self.server.service.autonomy_arm(value.get("ttl_s"))
            elif self.path == "/autonomy/disarm":
                result = self.server.service.autonomy_disarm(value.get("reason"))
            elif self.path == "/autonomy/goal":
                result = self.server.service.autonomy_goal(
                    value.get("text"),
                    value.get("kind", "explore"),
                    value.get("target_id"),
                    value.get("selector"),
                    value.get("constraints"),
                    value.get("based_on_revision"),
                    value.get("target_ref"),
                    value.get("frame_revision"),
                )
            elif self.path == "/autonomy/wander-step":
                result = self.server.service.autonomy_wander_step(
                    value.get("direction"),
                    value.get("route_request_id"),
                )
            elif self.path == "/autonomy/intent":
                result = self.server.service.autonomy_intent(
                    value.get("action"),
                    value.get("text"),
                    value.get("target_id"),
                    value.get("target_type", "npc"),
                    value.get("target_label"),
                    value.get("min_confidence", 0.25),
                    value.get("constraints"),
                )
            elif self.path == "/autonomy/stop":
                result = self.server.service.autonomy_stop(value.get("reason"))
            elif self.path == "/vision/start":
                result = self.server.service.vision_start()
            elif self.path == "/vision/stop":
                result = self.server.service.vision_stop(value.get("reason"))
            elif self.path == "/worldmodel/start":
                result = self.server.service.world_model_start()
            elif self.path == "/worldmodel/stop":
                result = self.server.service.world_model_stop(value.get("reason"))
            elif self.path == "/worldmodel/world":
                result = self.server.service.world_model_set_world(
                    value.get("world_key"), value.get("world_name")
                )
            elif self.path == "/worldmodel/navmesh/start":
                result = self.server.service.navmesh_start(record=value.get("record") is True,
                                                           record_images=value.get("record_images") is True)
            elif self.path == "/worldmodel/navmesh/stop":
                result = self.server.service.navmesh_stop()
            elif self.path == "/worldmodel/navmesh/goto":
                result = self.server.service.navmesh_goto(value.get("x"), value.get("y"))
            elif self.path == "/worldmodel/navmesh/explore":
                result = self.server.service.navmesh_explore()
            elif self.path == "/worldmodel/navmesh/cancel":
                result = self.server.service.navmesh_cancel()
            elif self.path == "/worldmodel/navmesh/memory/update":
                result = self.server.service.navmesh_memory_update(
                    value.get("world"), value.get("session"), value.get("label"), value.get("pinned"))
            elif self.path == "/worldmodel/navmesh/memory/delete":
                result = self.server.service.navmesh_memory_delete(
                    value.get("world"), value.get("session"), value.get("scope"))
            elif self.path == "/worldmodel/navmesh/memory/prune":
                result = self.server.service.navmesh_memory_prune()
            elif self.path == "/vmc/recalibrate":
                result = self.server.service.vmc_recalibrate(
                    value.get("reason"),
                    value.get("accept_current_pose") is True,
                )
            elif self.path == "/clips/list":
                result = self.server.service.list_clips()
            elif self.path == "/semantic_express":
                result = self.server.service.semantic_express(value)
            elif self.path == "/semantic/commit":
                result = self.server.service.main_llm_semantic_commit(
                    value.get("request_id"),
                    value.get("frame_revision"),
                    value.get("entities"),
                )
            elif self.path == "/world/ingest":
                result = self.server.service.ingest_world(
                    value,
                    ack_only=bool(value.get("ack_only")),
                )
            elif self.path == "/cognition/plan":
                result = self.server.service.plan(value)
            elif self.path == "/cognition/feedback":
                result = self.server.service.cognition_feedback(value)
            elif self.path == "/policy/prompt":
                result = self.server.service.policy_prompt(value.get("text"))
            elif self.path == "/shutdown":
                result = {"accepted": True, "stopping": True}
                shutdown_reason = "http:/shutdown"
            else:
                self._json(404, {"error": "unknown endpoint"})
                return
            if (
                self.path.startswith("/osc/")
                or self.path.startswith("/input/")
                or self.path.startswith("/autonomy/")
                or self.path.startswith("/vision/")
                or self.path.startswith("/worldmodel/")
            ):
                dispatch_latency_ms = self.server.service.record_control_dispatch(self.path, started_at)
                if isinstance(result, dict):
                    result = dict(result)
                    result["dispatch_latency_ms"] = dispatch_latency_ms
            self._json(200, result)
            if shutdown_reason is not None:
                # 响应已落到内核缓冲区，这条连接让它按 finish() 正常关闭，
                # 停机逻辑自己不碰它。
                self.close_connection = True
                self.server.request_stop(shutdown_reason, exclude=self.connection)
        except Exception as exc:
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"[:500]})
        finally:
            self.server.service._http_client_addr = None


class BackendHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    # 请求线程全是 daemon，进程退出不靠它们。
    daemon_threads = True
    # Python 3.12 的 ThreadingMixIn 默认 block_on_close=True：server_close() 末尾会
    # self._threads.join()，等待【每一个】请求线程结束。而面板留下的空闲 keep-alive
    # 连接（阻塞在 rfile.readline）和永不结束的 /vision/mjpeg 流都不会自己结束 ——
    # 2026-11-05 实测：POST /shutdown 之后监听端口已经关闭、进程却永远活着，
    # Ctrl+C 走同一条 finally 也一样卡死。所以这里明确不 join：礼貌关连接由
    # request_stop() 主动掐 socket 负责，进程退出由看门狗兜底。
    block_on_close = False

    def __init__(
        self,
        address: tuple[str, int],
        token: str,
        service: BackendService,
        *,
        config_store: StandaloneConfigStore,
    ) -> None:
        super().__init__(address, BackendRequestHandler)
        self.token = token
        self.service = service
        self.config_store = config_store
        # 停机意图的唯一真值：handler（MJPEG / 健康检查）、信号处理、看门狗都看它。
        self.shutting_down = threading.Event()
        self._connections: set[Any] = set()
        self._connections_lock = threading.Lock()
        self._stop_lock = threading.Lock()

    @property
    def bound_port(self) -> int:
        """实际绑定端口。配合 --port=0 让测试和并发实例不必抢固定端口。"""
        return int(self.server_address[1])

    def register_connection(self, connection: Any) -> None:
        with self._connections_lock:
            self._connections.add(connection)

    def unregister_connection(self, connection: Any) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # 停机时我们主动掐断连接，请求线程里的 BrokenPipe / OSError 全是预期噪声；
        # 打到 stderr 只会掩盖真正的拆卸错误。
        if self.shutting_down.is_set():
            return
        super().handle_error(request, client_address)

    def request_stop(self, reason: str = "unspecified", *, exclude: Any = None) -> bool:
        """幂等地请求停机；返回 True 表示这次调用真正发起了停机。

        只负责「叫停」：掐断在途连接 + 让 serve_forever 返回。真正的拆卸顺序留在
        main() 的 finally 里，那里才看得到 service、也才能保证只拆一次。
        """
        with self._stop_lock:
            if self.shutting_down.is_set():
                return False
            self.shutting_down.set()
            with self._connections_lock:
                connections = list(self._connections)
                self._connections.clear()
            for connection in connections:
                if connection is exclude:
                    # 正在回话的那条（比如 /shutdown 自己）要让它把响应写完，
                    # 由 finish() 正常关闭。
                    continue
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            _log_event("shutdown_requested", reason=reason)
            threading.Thread(
                target=self._stop_serving,
                name="backend-http-stop",
                daemon=True,
            ).start()
            return True

    def _stop_serving(self) -> None:
        try:
            # shutdown() 会等 serve_forever 的循环退出（poll_interval=0.1），
            # 它必须在另一条线程里调用，否则会自己等自己。
            self.shutdown()
        except Exception as exc:  # noqa: BLE001 - 停机失败只留痕，不能挡住退出
            _log_event("shutdown_serve_failed", error=f"{type(exc).__name__}: {exc}"[:240])


def _read_config_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = (
        tomllib.loads(raw.decode("utf-8"))
        if path.suffix.lower() == ".toml"
        else json.loads(raw.decode("utf-8"))
    )
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON/TOML object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48912)
    parser.add_argument("--token", default=None)
    parser.add_argument("--config-dir", default=str(PROJECT_DIR))
    # 只读的包内容（模型、片段）来自 --config-dir；一切运行时写盘走这里。缺省
    # 回落到 --config-dir 是为了独立运行/测试，插件路径下宿主会显式给出数据目录。
    parser.add_argument("--state-dir", default=None)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config-json", default=None)
    source.add_argument("--config-file", type=Path, default=None, help="JSON or TOML config file")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="load plugin.toml plus backend.settings.json and enable the standalone config UI",
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=None,
        help="standalone JSON override file (default: <config-dir>/backend.settings.json)",
    )
    parser.add_argument("--open-ui", action="store_true", help="open the standalone UI in the default browser")
    parser.add_argument(
        "--offline",
        "--dry-run",
        action="store_true",
        help="disable VMC, VRChat OSC, and driver telemetry for local development",
    )
    # 诊断用：Windows 上 GIL 交接被定时器量化，多线程同跑时 LAS2 每帧 ~730 次交接会被拖慢。
    # 缺省不动解释器默认值（5 ms）；对照看 navmesh status 的 stereo_stages。
    parser.add_argument("--switchinterval-ms", type=float, default=None,
                        help="diagnostic: sys.setswitchinterval in milliseconds")
    args = parser.parse_args()
    if args.switchinterval_ms is not None:
        sys.setswitchinterval(max(1e-6, args.switchinterval_ms / 1000.0))
    try:
        config_dir = Path(args.config_dir).resolve()
        state_dir = Path(args.state_dir).resolve() if args.state_dir else config_dir
        settings_path: Path | None = None
        config_source = "defaults"
        if args.config_json:
            config_data = json.loads(base64.urlsafe_b64decode(args.config_json.encode("ascii")).decode("utf-8"))
            config_source = "config-json"
        elif args.config_file:
            config_data = _read_config_file(args.config_file)
            config_source = str(args.config_file.resolve())
        elif args.standalone:
            default_config = config_dir / "plugin.toml"
            config_data = _read_config_file(default_config)
            config_source = str(default_config)
        else:
            config_data = {}
        if not isinstance(config_data, dict):
            raise ValueError("config must be a JSON object")
        if args.standalone:
            settings_path = args.settings_file or (state_dir / "backend.settings.json")
            if not settings_path.is_absolute():
                settings_path = state_dir / settings_path
            settings_path = settings_path.resolve()
            if settings_path.suffix.lower() != ".json":
                raise ValueError("standalone settings file must use the .json suffix")
            config_data = deep_merge(config_data, load_settings_file(settings_path))
        offline = bool(
            args.offline
            or (
                not args.standalone
                and args.config_json is None
                and args.config_file is None
            )
        )
        if offline:
            config_data = copy.deepcopy(config_data)
            for section in ("vmc_idle", "vrchat_osc", "driver_log"):
                config_data.setdefault(section, {})["enabled"] = False
            config_data.setdefault("vmc_idle", {})["manage_host_output"] = False
            config_data.setdefault("vision", {})["enabled"] = False
            config_data.setdefault("vision", {})["source"] = "none"
        token = str(args.token or secrets.token_urlsafe(24))
        config_store = StandaloneConfigStore(
            config_data,
            settings_path=settings_path,
            editable=bool(args.standalone),
            mode="standalone" if args.standalone else "managed",
            source=config_source,
            offline=offline,
        )
        service = BackendService(config_data, config_dir, state_dir=state_dir, dry_run=offline)
        server: BackendHttpServer | None = None
        try:
            # 先完成端口绑定再启动 VMC/OSC 资源，并把两者放在同一个 finally 块中；
            # 这样绑定或启动失败时，总能恢复已部分初始化的宿主状态。
            server = BackendHttpServer(
                (args.host, args.port),
                token,
                service,
                config_store=config_store,
            )
            service.start()
            # --port=0 时实际端口由内核分配；报出去的必须是真端口，否则控制台打不开、
            # 测试也找不到它。
            port = server.bound_port
            ui_url = f"http://127.0.0.1:{port}/#token={token}"
            print(
                json.dumps(
                    {
                        "ready": True,
                        "host": args.host,
                        "port": port,
                        "token": token,
                        "pid": os.getpid(),
                        "offline": offline,
                        "mode": "standalone" if args.standalone else "managed",
                        "ui": ui_url,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            if args.open_ui:
                import webbrowser
                threading.Thread(target=webbrowser.open, args=(ui_url,), daemon=True).start()

            _install_shutdown_signals(server)

            try:
                server.serve_forever(poll_interval=0.1)
            except KeyboardInterrupt:
                # 信号处理器没装上（或装好之前就按了 Ctrl+C）时的退路：仍然走同一条
                # 优雅停机路径，而不是让 KeyboardInterrupt 冲掉 finally。
                server.request_stop("keyboard_interrupt")
            _log_event("serving_stopped")
        finally:
            if server is not None:
                # 无论怎么走到这里（正常停机、启动失败、异常），先武装看门狗再拆：
                # 拆卸链路卡住时，它是唯一还保证进程会消失的东西。
                server.request_stop("teardown")
                _arm_shutdown_watchdog(_SHUTDOWN_HARD_LIMIT_S)
                try:
                    server.server_close()
                except Exception as exc:  # noqa: BLE001 - 关监听套接字失败不该挡住退出
                    _log_event("server_close_failed", error=f"{type(exc).__name__}: {exc}"[:240])
            started_at = time.perf_counter()
            try:
                service.stop()
            except Exception as exc:  # noqa: BLE001 - 拆卸失败只留痕，进程必须退出
                _log_event(
                    "service_stop_failed",
                    error=f"{type(exc).__name__}: {exc}"[:240],
                    elapsed_s=round(time.perf_counter() - started_at, 2),
                )
            else:
                _log_event("service_stopped", elapsed_s=round(time.perf_counter() - started_at, 2))
        _log_event("backend_exited")
        # 正常收工也必须走硬退出：解释器自己的线程 join 没有上界（见 _exit_process）。
        _exit_process(0)
        return 0
    except Exception as exc:
        print(f"backend startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        _exit_process(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
