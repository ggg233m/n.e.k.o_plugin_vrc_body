"""独立回环后端进程入口。

这里刻意设计为直接执行文件而不是使用 ``python -m``：否则后端进程导入插件
包时会要求 N.E.K.O SDK。下面的轻量命名空间包引导只加载核心模块。
"""

from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Any
import types


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


class BackendRequestHandler(BaseHTTPRequestHandler):
    server: "BackendHttpServer"

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
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path == "/health":
            self._json(200, {"ok": True, "pid": os.getpid()})
            return
        if self.path in {"/snapshot", "/state"}:
            self._json(200, self.server.service.snapshot())
            return
        if self.path == "/awareness":
            self._json(200, self.server.service.awareness())
            return
        self._json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        try:
            value = self._read_json()
            if self.path == "/action":
                result = self.server.service.submit(str(value.get("kind") or ""), value.get("params") or {})
            elif self.path == "/osc/parameter":
                result = self.server.service.send_avatar_parameter(str(value.get("name") or ""), value.get("value"))
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/input":
                result = self.server.service.pulse_input(
                    str(value.get("action") or ""),
                    str(value.get("side") or ""),
                    int(value.get("hold_ms") or 100),
                )
                result = {"accepted": result[0], "reason": result[1]}
            elif self.path == "/osc/cancel":
                self.server.service.cancel_inputs()
                result = {"accepted": True}
            elif self.path == "/clips/list":
                result = self.server.service.list_clips()
            elif self.path == "/semantic_express":
                result = self.server.service.semantic_express(value)
            elif self.path == "/world/ingest":
                result = self.server.service.ingest_world(value)
            elif self.path == "/shutdown":
                result = {"accepted": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._json(404, {"error": "unknown endpoint"})
                return
            self._json(200, result)
        except Exception as exc:
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"[:500]})


MappingLike = dict[str, Any]


class BackendHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], token: str, service: BackendService) -> None:
        super().__init__(address, BackendRequestHandler)
        self.token = token
        self.service = service


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    try:
        config_data = json.loads(base64.urlsafe_b64decode(args.config_json.encode("ascii")).decode("utf-8"))
        if not isinstance(config_data, dict):
            raise ValueError("config must be a JSON object")
        service = BackendService(config_data, args.config_dir)
        service.start()
        server = BackendHttpServer((args.host, args.port), args.token, service)

        def stop(*_: Any) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, stop)
        try:
            server.serve_forever(poll_interval=0.1)
        finally:
            server.server_close()
            service.stop()
        return 0
    except Exception as exc:
        print(f"backend startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
