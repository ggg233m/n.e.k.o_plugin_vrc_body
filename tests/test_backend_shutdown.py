"""后端优雅退出的回归测试：POST /shutdown 与 Ctrl+C 都必须真的让进程消失。

2026-11-05 的故障（用户报「起来后关不掉」）：POST /shutdown 回 200、监听端口也
关了，进程却永远活着。根因是 socketserver.ThreadingMixIn 默认 block_on_close=True
⇒ ThreadingMixIn.server_close() 末尾会 join 每一个请求线程，而控制面板留下的
空闲 HTTP/1.1 keep-alive 连接（阻塞在 rfile.readline()）和永不结束的
/vision/mjpeg 流都不会自己还。Ctrl+C 走同一条 finally，卡法一样。

这里从三层钉住它：
  1) 进程内：停机位、request_stop() 幂等、空闲 keep-alive 会被有界关掉；
  2) 端到端：同时挂着 keep-alive 与 MJPEG 两条连接时 POST /shutdown，进程必须在
     窗口内退出且退出码 0；
  3) 信号：Ctrl+Break（POSIX 用 SIGINT）和 /shutdown 走同一条优雅路径。

⚠️ 端到端用例真的起后端进程：设置 NEKO_LAS_DISABLE=1 关掉 LAS2 预热，否则
service.start() 会顺手拉起一个 3.14 子进程开始最长 5 min 的编译
（见 backend/stereo_backend.py 的闸门注释），测试套件不该有这种副作用。
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from urllib.request import Request, urlopen

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.process import (
    BackendHttpServer,
    BackendRequestHandler,
)
from neko_anyadance_body.backend.webui import StandaloneConfigStore


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESS_SCRIPT = PROJECT_DIR / "backend" / "process.py"
TOKEN = "test-shutdown-token"
# 端到端等待窗口。关掉 LAS2 后实测拆卸 ~0.5 s，窗口给足余量，避免慢盘/CI 误报。
EXIT_WINDOW_S = 30.0


def _read_response(stream: socket.socket) -> tuple[str, dict[str, str], bytes]:
    """读一条完整的 HTTP/1.1 响应（按 Content-Length 取正文）。"""
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = stream.recv(4096)
        if not chunk:
            raise AssertionError(f"连接在响应头之前关闭：{buffer!r}")
        buffer += chunk
    head, _, body = buffer.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = lines[0]
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0") or 0)
    while len(body) < length:
        chunk = stream.recv(4096)
        if not chunk:
            break
        body += chunk
    return status, headers, body[:length]


def _request(port: int, method: str, path: str, *, token: str = TOKEN) -> tuple[int, dict]:
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        method=method,
        data=b"{}" if method == "POST" else None,
        headers={
            "Content-Type": "application/json",
            "X-Neko-Backend-Token": token,
        },
    )
    with urlopen(request, timeout=10.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _pid_alive(pid: int) -> bool:
    """pid 是否还活着（Windows 上不能用 os.kill(pid, 0)：那会直接杀掉进程）。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information | synchronize, False, pid
        )
        if not handle:
            return False
        try:
            # 句柄有效 ⇒ 进程对象还在；WAIT_TIMEOUT(258) 表示还没退出。
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class _BackendProcess:
    """起一个真后端进程，并持续收集它的 stderr（ready 与退出事件都在里面）。"""

    def __init__(self, state_dir: Path) -> None:
        self.lines: list[str] = []
        env = dict(os.environ)
        env["NEKO_LAS_DISABLE"] = "1"
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(PROCESS_SCRIPT),
                "--standalone",
                "--offline",
                "--host=127.0.0.1",
                "--port=0",
                f"--token={TOKEN}",
                f"--state-dir={state_dir}",
            ],
            cwd=str(PROJECT_DIR),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        self._reader = threading.Thread(target=self._drain, name="backend-stderr", daemon=True)
        self._reader.start()
        self.info = self._wait_ready(90.0)
        self.real_pid = int(self.info["pid"])
        self.port = int(self.info["port"])

    def _drain(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        for line in stream:
            self.lines.append(line.rstrip("\n"))

    def _wait_ready(self, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for line in list(self.lines):
                if not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if payload.get("ready"):
                    return payload
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        raise AssertionError(
            "后端没有在窗口内就绪：\n" + "\n".join(self.lines[-20:])
        )

    def events(self, name: str) -> list[dict]:
        found = []
        for line in list(self.lines):
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if payload.get("event") == name:
                found.append(payload)
        return found

    def wait_exit(self, timeout_s: float) -> int:
        try:
            return self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"后端在 {timeout_s:.0f}s 内没有退出（这就是「关不掉」）。"
                f" 端口 {self.port}，pid {self.real_pid}；stderr:\n"
                + "\n".join(self.lines[-20:])
            ) from None

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                pass
        if self.process.stderr is not None:
            try:
                self.process.stderr.close()
            except OSError:
                pass
        # 子进程（venv 启动器 + 真解释器）都要收干净，不能给下一个用例留残留。
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and _pid_alive(self.real_pid):
            time.sleep(0.1)


class BackendShutdownEndToEndTests(unittest.TestCase):
    """真进程级：面板留下的连接不该再把退出路径挂住。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.backend = _BackendProcess(Path(self.temporary.name))
        self.sockets: list[socket.socket] = []

    def tearDown(self) -> None:
        for stream in self.sockets:
            try:
                stream.close()
            except OSError:
                pass
        self.backend.cleanup()
        self.temporary.cleanup()

    def _keepalive(self) -> socket.socket:
        stream = socket.create_connection(("127.0.0.1", self.backend.port), timeout=5.0)
        self.sockets.append(stream)
        stream.sendall(
            (
                f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{self.backend.port}\r\n"
                f"X-Neko-Backend-Token: {TOKEN}\r\nConnection: keep-alive\r\n\r\n"
            ).encode("ascii")
        )
        status, _, body = _read_response(stream)
        self.assertIn("200", status)
        self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
        return stream

    def _mjpeg(self) -> socket.socket:
        stream = socket.create_connection(("127.0.0.1", self.backend.port), timeout=5.0)
        self.sockets.append(stream)
        stream.sendall(
            (
                f"GET /vision/mjpeg?token={TOKEN} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{self.backend.port}\r\n\r\n"
            ).encode("ascii")
        )
        return stream

    def test_shutdown_exits_with_idle_keepalive_and_mjpeg_open(self) -> None:
        self._keepalive()
        self._mjpeg()
        time.sleep(0.3)

        status, payload = _request(self.backend.port, "POST", "/shutdown")
        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])
        self.assertTrue(payload["stopping"])

        code = self.backend.wait_exit(EXIT_WINDOW_S)
        self.assertEqual(code, 0, "优雅退出的退出码必须是 0")
        self.assertEqual(len(self.backend.events("backend_exited")), 1)
        self.assertFalse(_pid_alive(self.backend.real_pid), "进程必须真的消失")

    @unittest.skipUnless(os.name == "nt" or hasattr(signal, "SIGINT"), "需要控制台信号")
    def test_console_signal_takes_the_same_graceful_path(self) -> None:
        self._keepalive()
        if os.name == "nt":
            # CTRL_C_EVENT 只能对整个控制台生效，所以用 CTRL_BREAK（= SIGBREAK）。
            self.backend.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.backend.process.send_signal(signal.SIGINT)

        code = self.backend.wait_exit(EXIT_WINDOW_S)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.backend.events("backend_exited")), 1)
        self.assertFalse(_pid_alive(self.backend.real_pid), "进程必须真的消失")


class BackendHttpServerShutdownUnitTests(unittest.TestCase):
    """进程内：停机位、幂等与空闲连接的上界。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self._servers: list[tuple[BackendHttpServer, threading.Thread]] = []

    def tearDown(self) -> None:
        for server, thread in self._servers:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
            thread.join(timeout=5.0)
        self.temporary.cleanup()

    def _server(self) -> tuple[BackendHttpServer, int]:
        store = StandaloneConfigStore(
            {},
            settings_path=Path(self.temporary.name) / "settings.json",
            editable=True,
            mode="standalone",
            source="defaults",
            offline=True,
        )
        # 假 service：/health 不碰它，停机位也不碰它（do_POST 才写 _http_client_addr）。
        server = BackendHttpServer(("127.0.0.1", 0), TOKEN, types.SimpleNamespace(), config_store=store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._servers.append((server, thread))
        return server, server.bound_port

    def test_health_reports_the_stopping_flag(self) -> None:
        server, port = self._server()
        status, payload = _request(port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["stopping"])

        # 只置停机位、不真的关服务：/health 必须如实说出来，探测方才能分清
        # 「还没起来」和「正在关闭」。
        server.shutting_down.set()
        status, payload = _request(port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["stopping"])

    def test_request_stop_is_idempotent(self) -> None:
        server, _port = self._server()
        self.assertTrue(server.request_stop("test-first"))
        self.assertTrue(server.shutting_down.is_set())
        # 第二次必须只返回 False，不能重复起停机线程、更不能抛。
        self.assertFalse(server.request_stop("test-second"))
        self.assertFalse(server.request_stop("test-third"))

    def test_idle_keepalive_is_closed_by_the_server(self) -> None:
        # 真的等 10 s 太慢：这里把读超时压到 1 s，验证的是「连接会被服务端关掉」
        # 这条契约，而不是那个常数本身。
        original = BackendRequestHandler.timeout
        BackendRequestHandler.timeout = 1.0
        try:
            _server, port = self._server()
            stream = socket.create_connection(("127.0.0.1", port), timeout=10.0)
            try:
                stream.sendall(
                    (
                        f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                        f"X-Neko-Backend-Token: {TOKEN}\r\nConnection: keep-alive\r\n\r\n"
                    ).encode("ascii")
                )
                status, _, _body = _read_response(stream)
                self.assertIn("200", status)
                started = time.monotonic()
                # 空闲超过读超时后，服务端必须主动关连接（读到 EOF），而不是让
                # 这个线程永远挂在 readline() 上。
                self.assertEqual(stream.recv(4096), b"")
                self.assertLess(time.monotonic() - started, 8.0)
            finally:
                stream.close()
        finally:
            BackendRequestHandler.timeout = original


if __name__ == "__main__":
    unittest.main()
