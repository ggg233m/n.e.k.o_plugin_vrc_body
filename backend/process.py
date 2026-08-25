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
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
import time
import tomllib
from typing import Any
import types
from urllib.parse import urlsplit


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
PACKAGE_NAME = PROJECT_DIR.name
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


_UI_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/ui": ("index.html", "text/html; charset=utf-8"),
    "/ui/": ("index.html", "text/html; charset=utf-8"),
    "/ui/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/ui/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class BackendRequestHandler(BaseHTTPRequestHandler):
    server: "BackendHttpServer"
    # 保持回环控制连接，避免重复的 OSC/动作调用每次都进行 TCP 握手。
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        return self.headers.get("X-Neko-Backend-Token") == self.server.token

    def _json(self, status: int, payload: MappingLike) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
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
        if path == "/health":
            self._json(200, {"ok": True, "pid": os.getpid()})
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
        if path == "/world/delta":
            from urllib.parse import parse_qs
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
            from urllib.parse import parse_qs
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
            from urllib.parse import parse_qs
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
            elif self.path == "/shutdown":
                result = {"accepted": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._json(404, {"error": "unknown endpoint"})
                return
            if (
                self.path.startswith("/osc/")
                or self.path.startswith("/input/")
                or self.path.startswith("/autonomy/")
                or self.path.startswith("/vision/")
            ):
                dispatch_latency_ms = self.server.service.record_control_dispatch(self.path, started_at)
                if isinstance(result, dict):
                    result = dict(result)
                    result["dispatch_latency_ms"] = dispatch_latency_ms
            self._json(200, result)
        except Exception as exc:
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"[:500]})


MappingLike = dict[str, Any]


class BackendHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True

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
    args = parser.parse_args()
    try:
        config_dir = Path(args.config_dir).resolve()
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
            settings_path = args.settings_file or (config_dir / "backend.settings.json")
            if not settings_path.is_absolute():
                settings_path = config_dir / settings_path
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
        service = BackendService(config_data, config_dir, dry_run=offline)
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
            ui_url = f"http://127.0.0.1:{args.port}/#token={token}"
            print(
                json.dumps(
                    {
                        "ready": True,
                        "host": args.host,
                        "port": args.port,
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

            def stop(*_: Any) -> None:
                if server is not None:
                    threading.Thread(target=server.shutdown, daemon=True).start()

            signal.signal(signal.SIGTERM, stop)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, stop)
            server.serve_forever(poll_interval=0.1)
        finally:
            if server is not None:
                server.server_close()
            service.stop()
        return 0
    except Exception as exc:
        print(f"backend startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
