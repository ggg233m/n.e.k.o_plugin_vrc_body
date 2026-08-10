"""Lifecycle control for N.E.K.O's documented VMC output API."""

from __future__ import annotations

import json
import math
import threading
import time
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
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.1,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Request the documented host T-pose and reset the relay once it starts."""
        if not self.config.enabled or not self.config.manage_host_output:
            return False
        if (
            not math.isfinite(duration_sec)
            or not 0.1 <= duration_sec <= 10.0
            or (
                timeout_seconds is not None
                and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0)
            )
            or not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds < 0.0
        ):
            raise ValueError("invalid VMC T-pose calibration timing")
        try:
            with self._lock:
                if not self._active:
                    self._calibration_state = "unavailable"
                    self._calibration_error = "N.E.K.O VMC output is not active"
                    return False
                self._calibration_state = "requesting"
                self._calibration_error = None

            token = self._csrf_token()
            requested = self._request_t_pose(duration_sec=duration_sec, token=token)
            with self._lock:
                self._status = dict(requested)
                self._calibration_state = "waiting_for_t_pose"

            # A freshly restarted N.E.K.O page may need one 5-second status
            # poll plus WebSocket reconnect/backoff before it can publish the
            # requested rest pose.  This wait runs on a daemon worker and does
            # not block plugin startup.
            deadline = (
                time.monotonic() + timeout_seconds
                if timeout_seconds is not None
                else None
            )
            while deadline is None or time.monotonic() < deadline:
                if stop_event is not None and stop_event.is_set():
                    with self._lock:
                        self._calibration_state = "cancelled"
                    return False
                status = self._requester("GET", "/api/vmc/status", None, None)
                with self._lock:
                    self._status = dict(status)
                # The browser clears t_pose_requested immediately before it
                # emits the first raw-rest frame.  The requested duration then
                # leaves ample time for the relay's next complete frame to be
                # captured as its calibration basis.
                if not bool(status.get("t_pose_requested")):
                    on_t_pose_started()
                    with self._lock:
                        self._calibration_state = "calibrated"
                        self._calibration_error = None
                    return True
                if stop_event is not None:
                    stop_event.wait(poll_interval_seconds)
                elif poll_interval_seconds > 0.0:
                    time.sleep(poll_interval_seconds)

            with self._lock:
                self._calibration_state = "timeout"
                self._calibration_error = "N.E.K.O did not start the requested T-pose before timeout"
            return False
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
