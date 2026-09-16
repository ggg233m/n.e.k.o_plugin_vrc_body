"""Lifecycle control for N.E.K.O's documented VMC output API."""

from __future__ import annotations

import json
import math
import threading
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import VmcIdleConfig


JsonRequest = Callable[[str, str, dict[str, Any] | None, str | None], dict[str, Any]]


def _normalize_base_url(value: str) -> str:
    candidate = str(value or "").strip().rstrip("/")
    parsed = urlsplit(candidate)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not parsed.port
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("vmc_idle.host_api_url must be a loopback HTTP origin with an explicit port")
    return candidate


class HostVmcController:
    """Enable host VMC while the relay is alive, then restore its prior state."""

    def __init__(self, config: VmcIdleConfig, *, logger: Any = None, requester: JsonRequest | None = None) -> None:
        self.config = config
        self.logger = logger
        self._requester = requester or self._request_json
        self._lock = threading.Lock()
        self._prior_status: dict[str, Any] | None = None
        self._changed = False
        self._active = False
        self._last_error: str | None = None
        self._status: dict[str, Any] = {}
        self._calibration_state = "idle"
        self._calibration_error: str | None = None

    @property
    def _base_url(self) -> str:
        return _normalize_base_url(self.config.host_api_url)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None, csrf_token: str | None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json", "Origin": self._base_url}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if csrf_token:
            headers["X-CSRF-Token"] = csrf_token
        request = Request(f"{self._base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.config.host_api_timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"N.E.K.O VMC API returned HTTP {exc.code}: {detail}") from exc
        except (OSError, URLError) as exc:
            raise RuntimeError(f"N.E.K.O VMC API is unavailable: {exc}") from exc
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("N.E.K.O VMC API returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("success") is False:
            reason = result.get("error") if isinstance(result, dict) else "invalid response"
            raise RuntimeError(f"N.E.K.O VMC API rejected the request: {reason}")
        return result

    def _csrf_token(self) -> str:
        page_config = self._requester("GET", "/api/config/page_config", None, None)
        token = page_config.get("autostart_csrf_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("N.E.K.O page_config did not expose a CSRF token")
        return token

    def _enable(self, *, host: str, port: int, send_rate_hz: int, token: str) -> dict[str, Any]:
        return self._requester("POST", "/api/vmc/enable", {"host": host, "port": port, "send_rate_hz": send_rate_hz}, token)

    def _disable(self, *, token: str) -> dict[str, Any]:
        return self._requester("POST", "/api/vmc/disable", {}, token)

    def _request_t_pose(self, *, duration_sec: float, token: str) -> dict[str, Any]:
        return self._requester("POST", "/api/vmc/t_pose", {"duration_sec": duration_sec}, token)

    def calibrate_rest_pose(
        self,
        on_t_pose_started: Callable[[], None],
        *,
        duration_sec: float = 2.0,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """请求宿主播一次权威静止姿势，受理后立刻让中转交出基准。

        受理凭据就是 ``POST /api/vmc/t_pose`` 的成功返回：宿主在 ``request_t_pose()``
        里自增 generation 并置 ``t_pose_requested=True``，返回体即本次请求。

        宿主的 T Pose 时序（``main_logic/vmc_sender.py`` + ``static/vrm/vrm-vmc-sender.js``）：

        * t≈0ms：POST 受理，浏览器下一次 status 轮询看到 requested 后把
          ``tPoseDeadline`` 设为 now + duration_sec；
        * t≈16ms：第一帧 ``t_pose=true`` 发出，骨骼取 ``vrm.humanoid.rawRestPose``。
          宿主 sender 正是在**收到这一帧时**把 ``t_pose_requested`` 复位，所以复位
          标记的是 T Pose 窗口的**开始**，不是结束；
        * t≈16ms..duration：持续输出 rawRestPose 帧，全部是合格 T Pose；
        * t=duration：deadline 到期，恢复普通动画帧。

        两种受理信号（POST 返回、requested 复位）只差一帧，都落在 T Pose 窗口内。
        这里取 POST 返回，因为它不依赖 status 轮询，握手是一次调用而不是一段等待。
        窗口早期可能混入过渡帧，中转用 ``_require_t_pose_frame`` 的解剖学校验挡掉，
        不合格的帧不会被锁成基准。
        """
        if not self.config.enabled or not self.config.manage_host_output:
            return False
        if not math.isfinite(duration_sec) or not 0.1 <= duration_sec <= 10.0:
            raise ValueError("invalid VMC T-pose calibration timing")
        if stop_event is not None and stop_event.is_set():
            with self._lock:
                self._calibration_state = "cancelled"
            return False
        try:
            with self._lock:
                if not self._active:
                    self._calibration_state = "unavailable"
                    self._calibration_error = "N.E.K.O VMC output is not active"
                    return False
                self._calibration_state = "requesting"
                self._calibration_error = None

            token = self._csrf_token()
            status = self._request_t_pose(duration_sec=duration_sec, token=token)
            with self._lock:
                self._status = dict(status)

            # POST 返回即受理，状态从这一刻起就是「已受理」。回调可能抛异常，那时宿主
            # 那边的 T Pose 已经在播了，把它记成 failed 会让状态与实际不符、并触发上层的
            # 失败退避（5 秒后才重试），白白错过这个窗口。所以先落状态，再回调；回调失败
            # 只记日志，不影响受理结果。
            with self._lock:
                self._calibration_state = "calibrated"
                self._calibration_error = None
            try:
                on_t_pose_started()
            except Exception as exc:
                with self._lock:
                    self._calibration_error = f"on_t_pose_started failed: {exc}"
                if self.logger:
                    self.logger.warning("VMC T-pose callback failed after the host accepted the request: %s", exc)
            return True
        except Exception as exc:
            with self._lock:
                self._calibration_state = "failed"
                self._calibration_error = str(exc)
            if self.logger:
                self.logger.warning("Could not calibrate N.E.K.O VMC rest pose: %s", exc)
            return False

    def start(self) -> bool:
        if not self.config.enabled or not self.config.manage_host_output:
            return False
        try:
            current = self._requester("GET", "/api/vmc/status", None, None)
            prior = {
                "enabled": bool(current.get("enabled")),
                "host": str(current.get("host") or "127.0.0.1"),
                "port": int(current.get("port") or 39539),
                "send_rate_hz": int(current.get("send_rate_hz") or 60),
            }
            target_matches = (
                prior["enabled"]
                and prior["host"] == self.config.host_output_host
                and prior["port"] == self.config.listen_port
                and prior["send_rate_hz"] == self.config.host_send_rate_hz
            )
            status = current
            changed = not target_matches
            if changed:
                token = self._csrf_token()
                status = self._enable(host=self.config.host_output_host, port=self.config.listen_port, send_rate_hz=self.config.host_send_rate_hz, token=token)
            with self._lock:
                self._prior_status = prior
                self._changed = changed
                self._active = bool(status.get("enabled"))
                self._status = dict(status)
                self._last_error = None
            return self._active
        except Exception as exc:
            with self._lock:
                self._active = False
                self._last_error = str(exc)
            if self.logger:
                self.logger.warning("Could not enable N.E.K.O VMC output: %s", exc)
            return False

    def stop(self) -> bool:
        with self._lock:
            prior = dict(self._prior_status) if self._prior_status else None
            changed = self._changed
        if not prior or not changed:
            with self._lock:
                self._active = False
            return True
        try:
            token = self._csrf_token()
            if prior["enabled"]:
                status = self._enable(host=prior["host"], port=prior["port"], send_rate_hz=prior["send_rate_hz"], token=token)
            else:
                # The API persists destination settings, so restore those too.
                self._enable(host=prior["host"], port=prior["port"], send_rate_hz=prior["send_rate_hz"], token=token)
                status = self._disable(token=token)
            with self._lock:
                self._active = False
                self._status = dict(status)
                self._last_error = None
                self._prior_status = None
                self._changed = False
            return True
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            if self.logger:
                self.logger.warning("Could not restore N.E.K.O VMC output state: %s", exc)
            return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "managed": self.config.manage_host_output,
                "active": self._active,
                "changed_by_plugin": self._changed,
                "api_url": self.config.host_api_url,
                "target": f"{self.config.host_output_host}:{self.config.listen_port}",
                "send_rate_hz": self.config.host_send_rate_hz,
                "last_error": self._last_error,
                "calibration": {
                    "state": self._calibration_state,
                    "last_error": self._calibration_error,
                },
                "status": dict(self._status),
            }


__all__ = ["HostVmcController"]
