"""为轻量插件外壳提供本机回环 IPC 客户端和兼容代理。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BackendUnavailable(RuntimeError):
    """独立后端当前不可访问。"""


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class RemoteOscConfig:
    enabled: bool = True
    input_pulse_ms: int = 100


class BackendClient:
    """启动一个后端子进程并与其通信。"""

    def __init__(self, config_data: Mapping[str, Any], config_dir: str | Path, *, logger: Any = None) -> None:
        self.config_data = dict(config_data)
        self.config_dir = str(config_dir)
        self.logger = logger
        self.port = 0
        self.token = ""
        self.process: subprocess.Popen[Any] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=64)
        self._stderr_thread: threading.Thread | None = None
        osc = self.config_data.get("vrchat_osc")
        osc_mapping = osc if isinstance(osc, Mapping) else {}
        raw_enabled = osc_mapping.get("enabled", True)
        enabled = raw_enabled if isinstance(raw_enabled, bool) else True
        try:
            input_pulse_ms = int(osc_mapping.get("input_pulse_ms", 100) or 100)
        except (TypeError, ValueError, OverflowError):
            input_pulse_ms = 100
        self.osc_config = RemoteOscConfig(
            enabled=enabled,
            input_pulse_ms=min(1000, max(20, input_pulse_ms)),
        )
        self.scheduler = RemoteScheduler(self)
        self.osc = RemoteOsc(self)
        self.driver_log = RemoteDriverLog(self)
        self.vmc_idle = RemoteVmcIdle(self)
        self.host_vmc = RemoteHostVmc(self)
        self.vision = RemoteVision(self)
        self._request_lock = threading.RLock()

    def start(self, *, timeout_s: float = 10.0, _retry: int = 0) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.port = _free_loopback_port()
        self.token = base64.urlsafe_b64encode(os.urandom(24)).decode("ascii")
        encoded_config = base64.urlsafe_b64encode(
            json.dumps(self.config_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        backend_dir = Path(__file__).resolve().parent
        project_dir = backend_dir.parent
        env = os.environ.copy()
        package_parent = str(project_dir.parent)
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = package_parent if not old_pythonpath else package_parent + os.pathsep + old_pythonpath
        command = [
            sys.executable,
            str(backend_dir / "process.py"),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--token", self.token,
            "--config-dir", self.config_dir,
            "--config-json", encoded_config,
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command,
            cwd=project_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        stderr = self.process.stderr
        if stderr is not None:
            self._stderr_lines.clear()
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(stderr,),
                name="neko-backend-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
        deadline = time.monotonic() + max(1.0, timeout_s)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            try:
                response = self.request("GET", "/health", timeout_s=0.5)
                if response.get("ok") is True:
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(0.1)
        process = self.process
        self.process = None
        # 读取错误管道前先终止仍在运行的子进程；否则启动超时时读取活动管道
        # 可能永久阻塞。
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        stderr_thread = self._stderr_thread
        if stderr_thread is not None:
            stderr_thread.join(timeout=1.0)
        self._stderr_thread = None
        child_error = "".join(self._stderr_lines).strip()
        detail = child_error or str(last_error or "process exited")
        # Windows 可能在探测和子进程绑定之间暂时占用临时端口；使用新端口重试，
        # 避免瞬时竞争导致插件启动失败。
        if _retry < 2 and ("10013" in detail or "10048" in detail or "address" in detail.lower()):
            self.port = 0
            self.token = ""
            return self.start(timeout_s=timeout_s, _retry=_retry + 1)
        raise BackendUnavailable(f"backend did not become ready: {detail[:500]}")

    def _drain_stderr(self, stream: Any) -> None:
        try:
            for raw_line in iter(stream.readline, b""):
                self._stderr_lines.append(raw_line.decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            return

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self.request("POST", "/shutdown", {}, timeout_s=1.0)
                except Exception:
                    process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
            stderr_thread = self._stderr_thread
            if stderr_thread is not None:
                stderr_thread.join(timeout=1.0)
            self._stderr_thread = None
        finally:
            self.port = 0
            self.token = ""

    def semantic_express(self, params: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return self.request("POST", "/semantic_express", params)
        except BackendUnavailable as exc:
            return {
                "accepted": False,
                "action_id": None,
                "state": "backend_unavailable",
                "normalized_params": dict(params),
                "reason": str(exc),
                "safety_state": "fault",
            }

    def list_clips(self) -> dict[str, Any]:
        try:
            return self.request("POST", "/clips/list", {})
        except BackendUnavailable as exc:
            return {"clips": [], "invalid_clips": [], "reason": str(exc)}

    def catalog(self) -> dict[str, Any]:
        """兼容 Hosted UI 目录接口的名称。"""
        return self.list_clips()

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_s: float = 3.0,
    ) -> dict[str, Any]:
        if not self.port or not self.token:
            raise BackendUnavailable("backend is not started")
        body = None if payload is None else json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Neko-Backend-Token": self.token,
            },
        )
        with self._request_lock:
            try:
                with urlopen(request, timeout=timeout_s) as response:
                    value = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                try:
                    detail = json.loads(exc.read().decode("utf-8"))
                    message = detail.get("error") if isinstance(detail, dict) else None
                except (OSError, ValueError, UnicodeError):
                    message = None
                raise BackendUnavailable(message or str(exc)) from exc
            except (OSError, URLError, TimeoutError) as exc:
                raise BackendUnavailable(str(exc)) from exc
        if not isinstance(value, dict):
            raise BackendUnavailable("backend returned a non-object response")
        if "error" in value and len(value) == 1:
            raise BackendUnavailable(str(value["error"]))
        return value


class RemoteScheduler:
    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def submit(self, kind: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        normalized = {
            str(key): value
            for key, value in dict(params or {}).items()
            if key != "_clip"
        }
        try:
            return self.client.request("POST", "/action", {"kind": kind, "params": normalized})
        except BackendUnavailable as exc:
            return {
                "accepted": False,
                "action_id": None,
                "state": "backend_unavailable",
                "normalized_params": normalized,
                "reason": str(exc),
                "safety_state": "fault",
            }

    def snapshot(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/snapshot").get("body", {})
        except BackendUnavailable:
            return {
                "state": "backend_unavailable",
                "safety_state": "fault",
                "output_enabled": False,
                "current_action": None,
                "queue_length": 0,
                "awareness": {
                    "motion": None,
                    "previous_action": None,
                    "transition": None,
                    "pose": {},
                    "summary": "独立后端不可用，身体状态未知。",
                },
            }

    def shutdown(self) -> None:
        self.client.stop()


class RemoteOsc:
    def __init__(self, client: BackendClient) -> None:
        self.client = client
        self.config = client.osc_config

    @property
    def thread_alive(self) -> bool:
        return self.client.process is not None and self.client.process.poll() is None

    def snapshot(self, include_parameters: bool = True) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/snapshot").get("vrchat_osc", {})
        except BackendUnavailable as exc:
            return {"enabled": self.config.enabled, "connection": "unknown", "last_error": str(exc)}

    def awareness(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/awareness").get("vrchat_osc", {})
        except BackendUnavailable as exc:
            return {
                "enabled": self.config.enabled,
                "connection": "unknown",
                "summary": str(exc),
                "parameters": {},
                "pose_feedback_available": False,
                "pickup_confirmation_available": False,
            }

    def send_parameter(self, name: str, value: Any) -> tuple[bool, str | None]:
        try:
            result = self.client.request("POST", "/osc/parameter", {"name": name, "value": value})
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def pulse_input(self, action: str, side: str, hold_ms: int) -> tuple[bool, str | None]:
        try:
            result = self.client.request("POST", "/osc/input", {"action": action, "side": side, "hold_ms": hold_ms})
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def cancel_scheduled_inputs(self, *, release: bool = True) -> None:
        try:
            self.client.request("POST", "/osc/cancel", {})
        except BackendUnavailable:
            return


class RemoteDriverLog:
    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def snapshot(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/snapshot").get("driver_log", {})
        except BackendUnavailable as exc:
            return {"enabled": False, "connection": "unknown", "last_error": str(exc)}


class RemoteVmcIdle:
    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def snapshot(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/snapshot").get("idle_relay", {})
        except BackendUnavailable as exc:
            return {"enabled": False, "connection": "unknown", "last_error": str(exc)}


class RemoteHostVmc:
    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def snapshot(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/snapshot").get("host_vmc", {})
        except BackendUnavailable as exc:
            return {"managed": False, "active": False, "last_error": str(exc)}


class RemoteVision:
    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def snapshot(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/snapshot").get("world", {})
        except BackendUnavailable as exc:
            return {"available": False, "uncertainties": ["backend_unavailable"], "error": str(exc)}


__all__ = ["BackendClient", "BackendUnavailable"]
