"""为轻量插件外壳提供本机回环 IPC 客户端和兼容代理。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import base64
import http.client
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

    status_code: int | None = None


class BackendRejected(BackendUnavailable):
    """后端已响应，但拒绝了请求（通常是 4xx schema/lifecycle 错误）。"""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


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
        # The Hosted plugin sends frequent control commands.  Keep one local
        # HTTP/1.1 connection for that fast path instead of paying a TCP
        # handshake for every OSC/action request.
        self._fast_connection: http.client.HTTPConnection | None = None
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
        self.controller_input = RemoteControllerInput(self)
        self.autonomy = RemoteAutonomy(self)
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
            "--host=127.0.0.1",
            f"--port={self.port}",
            # ``urlsafe_b64encode`` can legally start with ``-``.  Passing
            # values with ``--name=value`` prevents argparse from treating
            # such a token/config blob as another option.
            f"--token={self.token}",
            f"--config-dir={self.config_dir}",
            f"--config-json={encoded_config}",
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
            self._close_fast_connection()
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
            self._close_fast_connection()
            self.port = 0
            self.token = ""

    def _close_fast_connection(self) -> None:
        connection = self._fast_connection
        self._fast_connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def fast_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_s: float = 1.0,
    ) -> dict[str, Any]:
        """Send a latency-sensitive local request over a reused connection.

        The regular ``request`` method intentionally keeps its urllib-based
        behavior for compatibility and diagnostics.  Control-plane callers
        use this method so repeated OSC/action commands avoid reconnecting to
        the loopback HTTP server.  A stale keep-alive socket is retried once.
        """
        if not self.port or not self.token:
            raise BackendUnavailable("backend is not started")
        body = None if payload is None else json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "X-Neko-Backend-Token": self.token,
        }
        with self._request_lock:
            for attempt in range(2):
                try:
                    connection = self._fast_connection
                    if connection is None:
                        connection = http.client.HTTPConnection(
                            "127.0.0.1",
                            self.port,
                            timeout=timeout_s,
                        )
                        self._fast_connection = connection
                    else:
                        connection.timeout = timeout_s
                    connection.request(method, path, body=body, headers=headers)
                    response = connection.getresponse()
                    raw = response.read()
                    try:
                        value = json.loads(raw.decode("utf-8"))
                    except (UnicodeError, ValueError) as exc:
                        raise BackendUnavailable("backend returned invalid JSON") from exc
                    if not isinstance(value, dict):
                        raise BackendUnavailable("backend returned a non-object response")
                    if response.status >= 400:
                        message = value.get("error") or f"HTTP {response.status}"
                        if response.status in {400, 422}:
                            raise BackendRejected(str(message), status_code=response.status)
                        raise BackendUnavailable(str(message))
                    if "error" in value and len(value) == 1:
                        raise BackendUnavailable(str(value["error"]))
                    return value
                except BackendUnavailable:
                    raise
                except BackendRejected:
                    raise
                except (
                    OSError,
                    TimeoutError,
                    http.client.HTTPException,
                ) as exc:
                    self._close_fast_connection()
                    if attempt == 0:
                        continue
                    raise BackendUnavailable(str(exc)) from exc
        raise BackendUnavailable("fast backend request failed")

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
                status_code: int | None
                try:
                    status_code = int(exc.code)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    status_code = None
                # 400/422 are the backend's structured schema/lifecycle errors.
                # Keep auth, routing, and rate-limit failures on the unavailable
                # path so callers do not mislabel them as bad observations.
                if status_code in {400, 422}:
                    raise BackendRejected(
                        message or str(exc),
                        status_code=status_code,
                    ) from exc
                raise BackendUnavailable(message or str(exc)) from exc
            except (OSError, URLError, TimeoutError) as exc:
                raise BackendUnavailable(str(exc)) from exc
        if not isinstance(value, dict):
            raise BackendUnavailable("backend returned a non-object response")
        if "error" in value and len(value) == 1:
            raise BackendUnavailable(str(value["error"]))
        return value


def _control_request(
    client: Any,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the persistent control channel when the client provides it."""
    fast_request = getattr(client, "fast_request", None)
    if callable(fast_request):
        return fast_request(method, path, payload)
    return client.request(method, path, payload)


class RemoteScheduler:
    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def submit(
        self,
        kind: str,
        params: Mapping[str, Any] | None = None,
        *,
        preconditions: Any = None,
    ) -> dict[str, Any]:
        normalized = {
            str(key): value
            for key, value in dict(params or {}).items()
            if key != "_clip"
        }
        try:
            payload: dict[str, Any] = {"kind": kind, "params": normalized}
            if preconditions is not None:
                # 保留原 JSON 类型，让后端严格 schema 统一生成结构化拒绝；
                # 在客户端 list(...) 会让标量异常逃出动作结果协议。
                payload["preconditions"] = preconditions
            return _control_request(self.client, "POST", "/action", payload)
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
            response = self.client.request("GET", "/snapshot")
            body = dict(response.get("body", {}))
            if "control_latency" in response:
                body["control_latency"] = response["control_latency"]
            return body
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
                "control_latency": {
                    "count": 0,
                    "last_operation": None,
                    "last_latency_ms": None,
                    "max_latency_ms": 0.0,
                    "by_operation": {},
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
            result = _control_request(self.client, "POST", "/osc/parameter", {"name": name, "value": value})
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def pulse_input(self, action: str, side: str, hold_ms: int) -> tuple[bool, str | None]:
        try:
            result = _control_request(
                self.client,
                "POST",
                "/osc/input",
                {"action": action, "side": side, "hold_ms": hold_ms},
            )
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def set_locomotion(self, vertical: float, horizontal: float, duration_ms: int) -> tuple[bool, str | None]:
        try:
            result = _control_request(
                self.client,
                "POST",
                "/osc/locomotion",
                {
                    "vertical": vertical,
                    "horizontal": horizontal,
                    "duration_ms": duration_ms,
                },
            )
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def set_turn(self, horizontal: float, duration_ms: int) -> tuple[bool, str | None]:
        try:
            result = _control_request(
                self.client,
                "POST",
                "/osc/turn",
                {"horizontal": horizontal, "duration_ms": duration_ms},
            )
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def stop_movement(self) -> tuple[bool, str | None]:
        try:
            result = _control_request(self.client, "POST", "/osc/stop_movement", {})
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def batch(self, commands: list[Mapping[str, Any]]) -> dict[str, Any]:
        try:
            return _control_request(self.client, "POST", "/osc/batch", {"commands": commands})
        except BackendUnavailable as exc:
            return {"accepted": False, "results": [], "reason": str(exc)}

    def send_chatbox(self, text: str, immediate: bool = True) -> tuple[bool, str | None]:
        try:
            result = _control_request(
                self.client,
                "POST",
                "/osc/chatbox",
                {"text": text, "immediate": immediate},
            )
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def cancel_scheduled_inputs(self, *, release: bool = True) -> None:
        try:
            _control_request(self.client, "POST", "/osc/cancel", {})
        except BackendUnavailable:
            return


class RemoteControllerInput:
    """Fast IPC proxy for the AnyaDance virtual Index controller inputs."""

    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def snapshot(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/snapshot").get("body", {}).get("controller_input", {})
        except BackendUnavailable as exc:
            return {"mode": "unknown", "axes": {}, "buttons": {}, "error": str(exc)}

    def set_axes(self, side: str, x: float, y: float, duration_ms: int) -> tuple[bool, str | None]:
        try:
            result = _control_request(
                self.client,
                "POST",
                "/input/axes",
                {"side": side, "x": x, "y": y, "duration_ms": duration_ms},
            )
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def set_button(
        self,
        side: str,
        button: str,
        pressed: bool = True,
        hold_ms: int = 100,
        value: float = 1.0,
    ) -> tuple[bool, str | None]:
        try:
            result = _control_request(
                self.client,
                "POST",
                "/input/button",
                {
                    "side": side,
                    "button": button,
                    "pressed": pressed,
                    "hold_ms": hold_ms,
                    "value": value,
                },
            )
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)

    def release(self, side: str = "all") -> tuple[bool, str | None]:
        try:
            result = _control_request(self.client, "POST", "/input/release", {"side": side})
            return bool(result.get("accepted")), result.get("reason")
        except BackendUnavailable as exc:
            return False, str(exc)


class RemoteAutonomy:
    """Authorization and goal proxy; arming is intentionally explicit."""

    def __init__(self, client: BackendClient) -> None:
        self.client = client

    def snapshot(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/autonomy")
        except BackendUnavailable as exc:
            return {"state": "disarmed", "armed": False, "reason": str(exc)}

    def arm(self, ttl_s: float | None = None) -> dict[str, Any]:
        try:
            payload = {} if ttl_s is None else {"ttl_s": ttl_s}
            return _control_request(self.client, "POST", "/autonomy/arm", payload)
        except BackendUnavailable as exc:
            return {"accepted": False, "reason": str(exc), **self.snapshot()}

    def disarm(self, reason: str = "manual_disarm") -> dict[str, Any]:
        try:
            return _control_request(self.client, "POST", "/autonomy/disarm", {"reason": reason})
        except BackendUnavailable as exc:
            return {"accepted": False, "reason": str(exc), **self.snapshot()}

    def goal(self, text: str, kind: str = "explore") -> dict[str, Any]:
        try:
            return _control_request(self.client, "POST", "/autonomy/goal", {"text": text, "kind": kind})
        except BackendUnavailable as exc:
            return {"accepted": False, "reason": str(exc), **self.snapshot()}

    def stop(self, reason: str = "autonomy_stop") -> dict[str, Any]:
        try:
            return _control_request(self.client, "POST", "/autonomy/stop", {"reason": reason})
        except BackendUnavailable as exc:
            return {"accepted": False, "reason": str(exc), **self.snapshot()}


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

    def perception(self) -> dict[str, Any]:
        try:
            return self.client.request("GET", "/perception")
        except BackendUnavailable as exc:
            return {"world": self.snapshot(), "worker": {"enabled": False, "error": str(exc)}}

    def delta(self, after_revision: int = 0, *, wait_ms: int = 250, limit: int = 16) -> dict[str, Any]:
        try:
            return self.client.request(
                "GET",
                f"/world/delta?after_revision={int(after_revision)}&wait_ms={int(wait_ms)}&limit={int(limit)}",
                timeout_s=max(1.0, min(5.0, wait_ms / 1000.0 + 1.0)),
            )
        except BackendUnavailable as exc:
            return {
                "revision": int(after_revision),
                "after_revision": int(after_revision),
                "changed": False,
                "coalesced": False,
                "world": {"available": False, "uncertainties": ["backend_unavailable"]},
                "navigation": {"status": "unknown", "safe_navigation": False},
                "social": {"status": "unknown", "players_persisted": False, "chat_persisted": False},
                "uncertainty": ["backend_unavailable"],
                "changes": {"entities": [], "events": [], "removed_entity_ids": [], "removed_entity_count": 0},
                "error": str(exc),
            }

    def ingest(
        self,
        observation: Mapping[str, Any],
        *,
        ack_only: bool = False,
    ) -> dict[str, Any]:
        """向后端发布一批观测；生命周期删除与事件可在同一批次提交。

        这是传输接缝，不在插件侧运行 detector，也不对观测做宽松重写；
        后端会统一执行实体 ID、来源和时间戳校验。
        """
        if not isinstance(observation, Mapping):
            return {
                "accepted": False,
                "reason_code": "invalid_world_observation",
                "reason": "observation must be an object",
            }
        payload = dict(observation)
        if ack_only:
            payload["ack_only"] = True
        try:
            return self.client.request("POST", "/world/ingest", payload)
        except BackendRejected as exc:
            if exc.status_code not in {400, 422}:
                return {
                    "accepted": False,
                    "reason_code": "backend_unavailable",
                    "reason": str(exc),
                    "status_code": exc.status_code,
                }
            return {
                "accepted": False,
                "reason_code": "invalid_world_observation",
                "reason": str(exc),
                "status_code": exc.status_code,
            }
        except BackendUnavailable as exc:
            return {
                "accepted": False,
                "reason_code": "backend_unavailable",
                "reason": str(exc),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            return {
                "accepted": False,
                "reason_code": "invalid_world_observation",
                "reason": f"invalid observation: {exc}"[:500],
            }


__all__ = ["BackendClient", "BackendRejected", "BackendUnavailable"]
