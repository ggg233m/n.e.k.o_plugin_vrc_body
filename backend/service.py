"""由独立 AnyaDance 后端进程持有的运行时。"""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from .adapters import (
    BodyCommand,
    BodyScheduler,
    ClipLibrary,
    DriverLogListener,
    HostVmcController,
    PluginConfig,
    VmcIdleRelay,
    VrchatOscBridge,
    resolve_expression,
)
from .cognition import CognitionRuntime
from .autonomy import AutonomyRuntime
from .navigator import LocalNavigator
from .vision import (
    DesktopMirrorFrameSource,
    DxcamFrameSource,
    FrameDetector,
    FrameSource,
    OpenVinoLocalDetector,
    OpenAICompatibleSemanticBackend,
    MssFrameSource,
    VisionObservation,
    VisionRuntime,
    VisionWorker,
)
from .world_state import WorldStateStore


_VMC_CALIBRATION_TIMEOUT_SECONDS = 8.0
_VMC_CALIBRATION_RETRY_SECONDS = 5.0
_WORLD_GATE_BYPASS_ACTIONS = frozenset({"stop", "disable", "reset", "cancel"})
_OSC_AXIS_MIN = -1.0
_OSC_AXIS_MAX = 1.0
_OSC_DURATION_MIN_MS = 100
_OSC_DURATION_MAX_MS = 10000
_OSC_HOLD_MIN_MS = 20
_OSC_HOLD_MAX_MS = 1000


def _osc_axis_value(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric) or not _OSC_AXIS_MIN <= numeric <= _OSC_AXIS_MAX:
        raise ValueError(f"{name} must be between {_OSC_AXIS_MIN:g} and {_OSC_AXIS_MAX:g}")
    return numeric


def _controller_value(value: Any, name: str) -> float:
    """Normalize trigger/grip values (AnyaDance protocol uses 0..1)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return numeric


def _osc_duration_ms(value: Any, default: int, name: str = "duration_ms") -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or not _OSC_DURATION_MIN_MS <= numeric <= _OSC_DURATION_MAX_MS
    ):
        raise ValueError(
            f"{name} must be an integer between {_OSC_DURATION_MIN_MS} and {_OSC_DURATION_MAX_MS}"
        )
    return int(numeric)


def _osc_hold_ms(value: Any, default: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError("hold_ms must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hold_ms must be an integer") from exc
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or not _OSC_HOLD_MIN_MS <= numeric <= _OSC_HOLD_MAX_MS
    ):
        raise ValueError(
            f"hold_ms must be an integer between {_OSC_HOLD_MIN_MS} and {_OSC_HOLD_MAX_MS}"
        )
    return int(numeric)


def _controller_hold_ms(value: Any, default: int, maximum: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError("hold_ms must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hold_ms must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or not 20 <= numeric <= maximum:
        raise ValueError(f"hold_ms must be an integer between 20 and {maximum}")
    return int(numeric)


class _DryRunDatagramTransport:
    """只记录帧数、不触碰网络的调度器传输层。"""

    local_port = None

    def __init__(self) -> None:
        self.sent_packets = 0

    def send(self, _payload: bytes, _target: tuple[str, int]) -> None:
        self.sent_packets += 1

    def close(self) -> None:
        return


class BackendService:
    """在插件进程之外持有所有长期运行的身体与视觉资源。"""

    def __init__(
        self,
        config_data: Mapping[str, Any],
        config_dir: str | Path,
        *,
        logger: Any = None,
        dry_run: bool = False,
        vision_source: FrameSource | None = None,
        vision_detector: FrameDetector | None = None,
    ) -> None:
        self.config = PluginConfig.from_mapping(config_data)
        self.config_dir = Path(config_dir)
        self.logger = logger
        self.dry_run = bool(dry_run)
        if vision_source is None and self.config.vision.enabled and self.config.vision.source != "external" and self.config.vision.capture in {
            "desktop_mirror", "dxcam", "mss"
        }:
            # Capture/model dependencies remain optional and are instantiated
            # only when vision is explicitly enabled.
            if self.config.vision.capture == "mss" or self.config.vision.source == "mss":
                vision_source = MssFrameSource(
                    monitor_index=self.config.vision.monitor_index,
                )
            elif self.config.vision.capture == "dxcam":
                vision_source = DxcamFrameSource(
                    device_idx=self.config.vision.dxcam_device_idx,
                    output_idx=self.config.vision.dxcam_output_idx,
                    backend=self.config.vision.dxcam_backend,
                )
            else:
                vision_source = DesktopMirrorFrameSource(
                    monitor_index=self.config.vision.monitor_index,
                    dxcam_device_idx=self.config.vision.dxcam_device_idx,
                    dxcam_output_idx=self.config.vision.dxcam_output_idx,
                    dxcam_backend=self.config.vision.dxcam_backend,
                )
        self._vision_source = vision_source
        if vision_detector is None and self.config.vision.enabled and self.config.vision.local_backend == "openvino":
            vision_detector = OpenVinoLocalDetector(model_path=os.getenv("VRC_OPENVINO_MODEL"))
        vision_semantic: Any | None = None
        if self.config.vision.enabled and self.config.vision.semantic_backend == "openai_compatible":
            endpoint = os.getenv("VRC_VLM_ENDPOINT") or os.getenv("OPENAI_BASE_URL")
            model = os.getenv("VRC_VLM_MODEL") or os.getenv("OPENAI_VLM_MODEL") or "gpt-4o-mini"
            if endpoint:
                vision_semantic = OpenAICompatibleSemanticBackend(
                    endpoint=endpoint,
                    model=model,
                    max_per_minute=self.config.vision.semantic_max_per_minute,
                )
        self.clip_library = ClipLibrary(self.config_dir / self.config.clip_directory, self.config)
        self.world_state = WorldStateStore(
            lifecycle_watermark_limit=self.config.vision.lifecycle_watermark_limit,
            persistence_path=self.config_dir / "world_memory.json",
            # Keep library/test callers isolated unless the section is
            # explicitly present; the shipped plugin.toml enables it.
            persist_world=self.config.world_memory.persist_world and "world_memory" in config_data,
            persist_players=self.config.world_memory.persist_players and "world_memory" in config_data,
        )
        self.vision = VisionRuntime(
            self.world_state,
            detector=vision_detector,
            semantic=vision_semantic,
            observation_callback=self._on_vision_observation,
        )
        self.vision_worker: VisionWorker | None = None
        if self.config.vision.enabled and vision_source is not None:
            self.vision_worker = self._new_vision_worker(vision_source)
        self.scheduler: BodyScheduler | None = None
        self.osc: VrchatOscBridge | None = None
        self.driver_log: DriverLogListener | None = None
        self.vmc_idle: VmcIdleRelay | None = None
        self.host_vmc: HostVmcController | None = None
        self._vmc_calibration_stop = threading.Event()
        self._vmc_calibration_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._started = False
        self._last_error: str | None = None
        self._expression_side_count = 0
        self._motion_intent_counts: dict[str, int] = {}
        self.autonomy = AutonomyRuntime(
            world_provider=lambda: self.world_state.snapshot(),
            release_inputs=self._release_all_inputs,
            session_ttl_s=self.config.autonomy.session_ttl_minutes * 60.0,
        )
        # The navigator is a local, bounded control loop.  It is deliberately
        # separate from the LLM and vision workers; it only emits short
        # AnyaDance axis updates after a fresh, visible target is observed.
        self.navigator = LocalNavigator(
            world_provider=lambda: self.world_state.snapshot(),
            goal_provider=lambda: self.autonomy.snapshot(),
            send_axes=self._navigator_send_axes,
            release_inputs=self._navigator_release_inputs,
        )
        self._control_metrics_lock = threading.Lock()
        self._control_metrics = {
            "count": 0,
            "last_operation": None,
            "last_latency_ms": None,
            "max_latency_ms": 0.0,
            "by_operation": {},
        }
        self.cognition = CognitionRuntime(
            self._cognition_sources,
            world_provider=lambda: self.world_state.snapshot(),
        )

    def _new_vision_worker(self, source: FrameSource) -> VisionWorker:
        return VisionWorker(
            self.vision,
            source,
            interval_s=self.config.vision.interval_ms / 1000.0,
            queue_size=self.config.vision.queue_size,
            capture_only=self.vision.detector is None and self.vision.semantic is None,
        )

    def _cognition_sources(self) -> dict[str, dict[str, Any]]:
        """暴露各数据源健康状况，且不会递归调用 ``snapshot``。"""
        body = self.scheduler.snapshot() if self.scheduler else {"state": "shutdown"}
        osc = self.osc.snapshot() if self.osc else {"enabled": False}
        driver = self.driver_log.snapshot() if self.driver_log else {"enabled": False}
        vmc = self.vmc_idle.snapshot() if self.vmc_idle else {"enabled": False}
        world = self.vision.snapshot()
        worker = self.vision_worker.status() if self.vision_worker else {
            "enabled": False,
            "running": False,
            "reason": "not_configured",
        }
        return {
            "body": {
                "state": body.get("state"),
                "safety_state": body.get("safety_state"),
                "output_enabled": body.get("output_enabled"),
                "current_action": body.get("current_action"),
                "sent_packets": (body.get("udp") or {}).get("sent_packets", 0),
                "send_failures": (body.get("udp") or {}).get("send_failures", 0),
                "actual_hz": (body.get("metrics") or {}).get("actual_hz", 0.0),
            },
            "vrchat_osc": {
                "enabled": osc.get("enabled"),
                "connection": osc.get("connection"),
                "received_packets": osc.get("received_packets", 0),
                "parameter_count": osc.get("parameter_count", 0),
                "send_failures": osc.get("send_failures", 0),
            },
            "driver_log": {
                "enabled": driver.get("enabled"),
                "connection": driver.get("connection"),
                "received_packets": driver.get("received_packets", 0),
                "accepted_commands": driver.get("accepted_commands", 0),
                "rejected_commands": driver.get("rejected_commands", 0),
                "last_command_age_ms": driver.get("last_command_age_ms"),
            },
            "vmc_idle": {
                "enabled": vmc.get("enabled"),
                "connection": vmc.get("connection"),
                "source_available": vmc.get("source_available"),
                "received_packets": vmc.get("received_packets", 0),
                "last_frame_age_ms": vmc.get("last_frame_age_ms"),
            },
            "world": {
                "available": world.get("available"),
                "entity_count": (world.get("status") or {}).get("entity_count", 0),
                "event_count": (world.get("status") or {}).get("event_count", 0),
                "last_observation_age_ms": (world.get("status") or {}).get("last_observation_age_ms"),
                "vision_enabled": (world.get("vision") or {}).get("enabled", False),
            },
            "vision_worker": worker,
        }

    def _apply_driver_sender_conflict(self, body: dict[str, Any], driver: Mapping[str, Any]) -> None:
        """Use driver telemetry to detect another writer of UDP latest-state."""
        senders = [str(item) for item in (driver.get("senders") or ()) if item]
        local_port = (body.get("udp") or {}).get("local_port")
        others: list[str] = []
        for sender in senders:
            try:
                sender_port = int(sender.rsplit(":", 1)[1])
            except (ValueError, IndexError):
                sender_port = None
            if local_port is None or sender_port != int(local_port):
                others.append(sender)
        udp = body.setdefault("udp", {})
        udp["other_senders"] = others
        if others:
            body["concurrent_sender_detection"] = "concurrent"
            state = self.autonomy.snapshot()
            if state.get("armed"):
                self.autonomy.disarm("another AnyaDance UDP sender detected")
        else:
            body["concurrent_sender_detection"] = "none" if senders else "unsupported"

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            try:
                self.vmc_idle = VmcIdleRelay(self.config.vmc_idle, self.config.profile, logger=self.logger)
                self.vmc_idle.start()
                self.host_vmc = HostVmcController(self.config.vmc_idle, logger=self.logger)
                self.host_vmc.start()
                self._start_vmc_calibration()
                self.scheduler = BodyScheduler(
                    self.config,
                    logger=self.logger,
                    transport=_DryRunDatagramTransport() if self.dry_run else None,
                    idle_frame_source=self.vmc_idle,
                    motion_started_callback=self._on_motion_started,
                )
                self.scheduler.start()
                self.osc = VrchatOscBridge(self.config.vrchat_osc, logger=self.logger)
                self.osc.start()
                self.driver_log = DriverLogListener(self.config.driver_log, logger=self.logger)
                self.driver_log.start()
                if self.vision_worker is not None:
                    self.vision_worker.start()
                self.navigator.start()
                self._started = True
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                self.stop()
                raise

    def _start_vmc_calibration(self) -> None:
        host_vmc = self.host_vmc
        relay = self.vmc_idle
        if host_vmc is None or relay is None:
            return
        if not self.config.vmc_idle.enabled or not self.config.vmc_idle.manage_host_output:
            return
        stop_event = threading.Event()
        self._vmc_calibration_stop = stop_event

        def calibrate() -> None:
            while not stop_event.is_set():
                if not host_vmc.snapshot()["active"]:
                    if not host_vmc.start():
                        stop_event.wait(_VMC_CALIBRATION_RETRY_SECONDS)
                        continue
                    if stop_event.is_set():
                        return
                relay.hold_calibration(reason="waiting_for_host_t_pose")
                calibrated = host_vmc.calibrate_rest_pose(
                    lambda: relay.reset_calibration(reason="host_t_pose"),
                    timeout_seconds=_VMC_CALIBRATION_TIMEOUT_SECONDS,
                    stop_event=stop_event,
                )
                if calibrated or stop_event.is_set():
                    return
                relay.reset_calibration(reason="t_pose_unavailable")
                stop_event.wait(_VMC_CALIBRATION_RETRY_SECONDS)

        self._vmc_calibration_thread = threading.Thread(
            target=calibrate,
            name="neko-vmc-rest-calibration",
            daemon=True,
        )
        self._vmc_calibration_thread.start()

    def _stop_vmc_calibration(self) -> None:
        self._vmc_calibration_stop.set()
        thread = self._vmc_calibration_thread
        if thread and thread.is_alive():
            thread.join(timeout=min(4.0, self.config.vmc_idle.host_api_timeout_seconds + 0.5))
        self._vmc_calibration_thread = None

    def _on_motion_started(self, command: BodyCommand, duration_s: float) -> None:
        if command.kind != "reach_and_grab" or self.osc is None:
            return
        action_id = command.action_id
        side = str(command.params["side"])

        def action_is_current() -> bool:
            if self.scheduler is None:
                return False
            snapshot = self.scheduler.snapshot()
            current = snapshot.get("current_action") or {}
            return (
                snapshot.get("output_enabled") is True
                and current.get("id") == action_id
                and current.get("name") == "reach_and_grab"
            )

        self.osc.schedule_input_pulse(
            "grab",
            side,
            delay_s=max(0.0, duration_s * 0.85),
            guard=action_is_current,
        )

    def stop(self) -> None:
        with self._lock:
            # Disarming first releases both the virtual controller overlay and
            # OSC fallback before worker/socket teardown.
            try:
                self.autonomy.disarm("backend_stopped")
            except Exception:
                pass
            self.navigator.stop()
            self._stop_vmc_calibration()
            if self.vision_worker:
                self.vision_worker.stop()
            self.vision_worker = None
            if self.driver_log:
                self.driver_log.stop()
            if self.osc:
                self.osc.stop()
            if self.scheduler:
                self.scheduler.shutdown()
            if self.host_vmc:
                self.host_vmc.stop()
            if self.vmc_idle:
                self.vmc_idle.stop()
            self.driver_log = None
            self.osc = None
            self.scheduler = None
            self.host_vmc = None
            self.vmc_idle = None
            self._started = False

    def _release_all_inputs(self) -> None:
        """Best-effort emergency release used by autonomy and shutdown paths."""
        scheduler = self.scheduler
        if scheduler is not None:
            try:
                scheduler.submit("input_release", {"side": "all"})
            except Exception:
                pass
        osc = self.osc
        if osc is not None:
            try:
                osc.cancel_scheduled_inputs()
            except Exception:
                pass
            try:
                osc.stop_all_axes()
            except Exception:
                pass

    def autonomy_snapshot(self) -> dict[str, Any]:
        result = self.autonomy.snapshot()
        result["navigation"] = self.navigator.snapshot()
        return result

    def autonomy_arm(self, ttl_s: Any = None) -> dict[str, Any]:
        try:
            normalized = None if ttl_s is None else float(ttl_s)
        except (TypeError, ValueError, OverflowError):
            return {"accepted": False, "reason": "ttl_s must be a number", **self.autonomy.snapshot()}
        return {"accepted": True, **self.autonomy.arm(ttl_s=normalized)}

    def autonomy_disarm(self, reason: Any = "manual_disarm") -> dict[str, Any]:
        return {"accepted": True, **self.autonomy.disarm(str(reason or "manual_disarm"))}

    def autonomy_goal(self, text: Any, kind: Any = "explore") -> dict[str, Any]:
        normalized_kind = str(kind or "explore").strip().lower()
        if normalized_kind in {"approach", "follow", "interact", "socialize"}:
            vision = self.vision.snapshot().get("vision") or {}
            semantic = vision.get("semantic") if isinstance(vision, Mapping) else {}
            if isinstance(semantic, Mapping) and semantic.get("last_error"):
                return {
                    "accepted": False,
                    "reason": "semantic vision is degraded; only safe exploration is allowed",
                    **self.autonomy.snapshot(),
                }
        return self.autonomy.submit_goal(text, kind)

    def autonomy_stop(self, reason: Any = "autonomy_stop") -> dict[str, Any]:
        return {"accepted": True, **self.autonomy.stop(str(reason or "autonomy_stop"))}

    def attach_vision(self, source: FrameSource, detector: FrameDetector) -> dict[str, Any]:
        """注入外部 detector；模型包和采集实现由调用方负责。"""
        with self._lock:
            if self.vision_worker:
                self.vision_worker.stop()
            self._vision_source = source
            self.vision.set_backends(detector=detector)
            self.vision_worker = self._new_vision_worker(source)
            started = self._started and self.vision_worker.start()
        return {
            "attached": True,
            "started": bool(started),
            "worker": self.vision_worker.status(),
        }

    def submit(
        self,
        kind: str,
        params: Mapping[str, Any] | None = None,
        *,
        preconditions: Any = None,
    ) -> dict[str, Any]:
        def finish(result: dict[str, Any]) -> dict[str, Any]:
            self.cognition.record_action(kind, result)
            return result

        scheduler = self.scheduler
        if scheduler is None:
            return finish({
                "accepted": False,
                "action_id": None,
                "state": "shutdown",
                "normalized_params": {},
                "reason": "backend scheduler is not initialized",
                "safety_state": "fault",
            })
        normalized = dict(params or {})
        embedded_preconditions = normalized.pop("_world_preconditions", None)
        precondition_alias_conflict = (
            preconditions is not None and embedded_preconditions is not None
        )
        if preconditions is None:
            preconditions = embedded_preconditions
        preconditions_declared = preconditions is not None
        if kind in _WORLD_GATE_BYPASS_ACTIONS:
            precondition_check = {
                "passed": True,
                "checked": 0,
                "preconditions": [],
                "failures": [],
                "bypassed": preconditions_declared,
                "bypass_reason": "safety_control_action" if preconditions_declared else None,
            }
        elif precondition_alias_conflict:
            precondition_check = {
                "passed": False,
                "checked": 0,
                "preconditions": [],
                "failures": [{
                    "index": None,
                    "code": "invalid_world_precondition",
                    "message": (
                        "preconditions must not be combined with "
                        "params._world_preconditions"
                    ),
                }],
            }
        else:
            precondition_check = self.cognition.check_preconditions(preconditions)
        if not precondition_check["passed"]:
            state = scheduler.snapshot()
            failures = precondition_check.get("failures") or []
            first = failures[0] if failures else {}
            detail = str(first.get("message") or "world preconditions are not satisfied")
            return finish({
                "accepted": False,
                "action_id": None,
                "state": state["state"],
                "normalized_params": normalized,
                "reason": f"world precondition failed: {detail}"[:500],
                "reason_code": "world_precondition_failed",
                "replan_required": True,
                "replan_reason": "world_precondition_failed",
                "precondition_check": precondition_check,
                "safety_state": state["safety_state"],
            })
        if kind in {"play_clip", "semantic_clip"} and "_clip" not in normalized:
            clip_name = str(normalized.get("clip_name") or "").strip()
            if not clip_name:
                return finish({
                    "accepted": False,
                    "action_id": None,
                    "state": scheduler.snapshot()["state"],
                    "normalized_params": normalized,
                    "reason": "clip_name is required",
                    "safety_state": scheduler.snapshot()["safety_state"],
                })
            try:
                normalized["_clip"] = self.clip_library.load(clip_name)
            except (KeyError, OSError, ValueError) as exc:
                state = scheduler.snapshot()
                return finish({
                    "accepted": False,
                    "action_id": None,
                    "state": state["state"],
                    "normalized_params": {
                        key: value for key, value in normalized.items() if key != "_clip"
                    },
                    "reason": str(exc),
                    "safety_state": state["safety_state"],
                })
        if kind in {"play_clip", "semantic_clip"}:
            clip = normalized.get("_clip")
            try:
                speed = float(normalized.get("speed", 1.0))
                loops = int(normalized.get("loop_count", 1))
                if not 0.25 <= speed <= 3.0 or not 1 <= loops <= 10:
                    raise ValueError("clip playback parameters are out of range")
                playback_seconds = 0.0 if clip.is_pose else (clip.duration_s / speed) * loops
            except (AttributeError, TypeError, ValueError, ZeroDivisionError, OverflowError):
                state = scheduler.snapshot()
                return finish({
                    "accepted": False,
                    "action_id": None,
                    "state": state["state"],
                    "normalized_params": {
                        key: value for key, value in normalized.items() if key != "_clip"
                    },
                    "reason": "invalid clip playback parameters",
                    "safety_state": state["safety_state"],
                })
            if not math.isfinite(playback_seconds) or playback_seconds > self.config.clip_max_duration_seconds:
                state = scheduler.snapshot()
                return finish({
                    "accepted": False,
                    "action_id": None,
                    "state": state["state"],
                    "normalized_params": {
                        key: value for key, value in normalized.items() if key != "_clip"
                    },
                    "reason": (
                        f"expanded clip playback must not exceed "
                        f"{self.config.clip_max_duration_seconds:g} seconds"
                    ),
                    "safety_state": state["safety_state"],
                })
        result = scheduler.submit(kind, normalized)
        if preconditions_declared:
            result["precondition_check"] = precondition_check
        return finish(result)

    def snapshot(self) -> dict[str, Any]:
        scheduler = self.scheduler
        body = scheduler.snapshot() if scheduler else {
            "state": "shutdown",
            "safety_state": "fault",
            "output_enabled": False,
            "current_action": None,
            "queue_length": 0,
        }
        driver_log = self.driver_log.snapshot() if self.driver_log else {"enabled": False}
        self._apply_driver_sender_conflict(body, driver_log)
        return {
            "body": body,
            "vrchat_osc": self.osc.snapshot() if self.osc else {"enabled": False},
            "driver_log": driver_log,
            "idle_relay": self.vmc_idle.snapshot() if self.vmc_idle else {"enabled": False},
            "host_vmc": self.host_vmc.snapshot() if self.host_vmc else {"managed": False, "active": False},
            "world": self.vision.snapshot(),
            "vision_worker": self.vision_worker.status() if self.vision_worker else {
                "enabled": False,
                "running": False,
                "reason": "not_configured",
            },
            "cognition": self.cognition.snapshot(),
            "autonomy": self.autonomy.snapshot(),
            "navigation": self.navigator.snapshot(),
            "control_latency": self.control_metrics_snapshot(),
            "backend": {
                "started": self._started,
                "dry_run": self.dry_run,
                "last_error": self._last_error,
                "pid": __import__("os").getpid(),
            },
        }

    def record_control_dispatch(self, operation: str, started_at: float) -> float:
        """Record server-side dispatch latency for a control-plane request."""
        elapsed_ms = max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)
        name = str(operation or "unknown")
        with self._control_metrics_lock:
            by_operation = self._control_metrics["by_operation"]
            record = by_operation.setdefault(name, {"count": 0, "last_latency_ms": None, "max_latency_ms": 0.0})
            record["count"] += 1
            record["last_latency_ms"] = round(elapsed_ms, 3)
            record["max_latency_ms"] = round(max(record["max_latency_ms"], elapsed_ms), 3)
            self._control_metrics["count"] += 1
            self._control_metrics["last_operation"] = name
            self._control_metrics["last_latency_ms"] = round(elapsed_ms, 3)
            self._control_metrics["max_latency_ms"] = round(
                max(self._control_metrics["max_latency_ms"], elapsed_ms),
                3,
            )
        return round(elapsed_ms, 3)

    def control_metrics_snapshot(self) -> dict[str, Any]:
        with self._control_metrics_lock:
            result = dict(self._control_metrics)
            result["by_operation"] = {
                name: dict(record)
                for name, record in self._control_metrics["by_operation"].items()
            }
            return result

    def awareness(self) -> dict[str, Any]:
        body = self.snapshot()["body"]
        awareness = dict(body.get("awareness") or {})
        awareness["vrchat_osc"] = self.osc.awareness() if self.osc else {"enabled": False}
        return awareness

    def world_delta(self, after_revision: Any = 0, *, wait_ms: Any = 250, limit: Any = 16) -> dict[str, Any]:
        """Long-poll the world store without entering the body control path."""
        result = self.world_state.delta(after_revision, wait_ms=wait_ms, limit=limit)
        self.autonomy.update_world(result.get("world"))
        return result

    def perception(self) -> dict[str, Any]:
        return {
            "world": self.vision.snapshot(),
            "navigation": self.navigator.snapshot(),
            "worker": self.vision_worker.status() if self.vision_worker else {
                "enabled": False,
                "running": False,
                "reason": "not_configured",
            },
        }

    def _navigator_send_axes(self, side: str, x: float, y: float, duration_ms: int) -> bool:
        """Send only the primary AnyaDance command; never fall back to OSC."""
        scheduler = self.scheduler
        if scheduler is None or self.config.input.primary != "anyadance":
            return False
        result = scheduler.submit(
            "input_axes",
            {"side": side, "x": x, "y": y, "duration_ms": duration_ms},
        )
        return bool(result.get("accepted"))

    def _navigator_release_inputs(self, side: str = "all") -> None:
        scheduler = self.scheduler
        if scheduler is not None:
            scheduler.submit("input_release", {"side": side})

    def send_avatar_parameter(self, name: str, value: Any) -> tuple[bool, str | None]:
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        return self.osc.send_parameter(name, value)

    def pulse_input(self, action: str, side: str, hold_ms: Any) -> tuple[bool, str | None]:
        try:
            normalized_hold = _osc_hold_ms(hold_ms, self.config.vrchat_osc.input_pulse_ms)
        except ValueError as exc:
            return False, str(exc)
        normalized_action = str(action or "").strip().lower()
        normalized_side = str(side or "").strip().lower()
        if normalized_action not in {"grab", "use", "drop"} or normalized_side not in {"left", "right"}:
            return False, "action/side must identify grab, use, or drop for left or right"

        if self.config.input.primary == "anyadance" and self.scheduler is not None:
            if normalized_action == "drop":
                result = self.scheduler.submit(
                    "input_button",
                    {"side": normalized_side, "button": "grip", "pressed": False},
                )
            else:
                result = self.scheduler.submit(
                    "input_button",
                    {
                        "side": normalized_side,
                        "button": "grip" if normalized_action == "grab" else "trigger",
                        "pressed": True,
                        "value": 1.0,
                        "hold_ms": normalized_hold,
                    },
                )
            if result.get("accepted") or not self.config.input.osc_fallback:
                return bool(result.get("accepted")), result.get("reason")

        if self.osc is None:
            return False, "AnyaDance controller input and VRChat OSC bridge are unavailable"
        return self.osc.pulse_input(normalized_action, normalized_side, normalized_hold)

    def set_locomotion(self, vertical: Any, horizontal: Any, duration_ms: Any) -> tuple[bool, str | None]:
        try:
            normalized_vertical = _osc_axis_value(vertical, "vertical")
            normalized_horizontal = _osc_axis_value(horizontal, "horizontal")
            normalized_duration = _osc_duration_ms(duration_ms, 1000)
        except ValueError as exc:
            return False, str(exc)
        if self.config.input.primary == "anyadance" and self.scheduler is not None:
            result = self.scheduler.submit(
                "input_axes",
                {
                    "side": "left",
                    "x": normalized_horizontal,
                    "y": normalized_vertical,
                    "duration_ms": min(normalized_duration, self.config.input.max_hold_ms),
                },
            )
            if result.get("accepted") or not self.config.input.osc_fallback:
                return bool(result.get("accepted")), result.get("reason")
        if self.osc is None:
            return False, "AnyaDance controller input and VRChat OSC bridge are unavailable"
        duration_s = normalized_duration / 1000.0
        set_axes = getattr(self.osc, "set_axes", None)
        if callable(set_axes):
            return set_axes(
                {
                    "move_vertical": normalized_vertical,
                    "move_horizontal": normalized_horizontal,
                },
                duration_s,
            )
        vertical_result = self.osc.set_axis("move_vertical", normalized_vertical, duration_s)
        if not vertical_result[0]:
            rollback = getattr(self.osc, "stop_axes", None)
            if callable(rollback):
                rollback(("move_vertical", "move_horizontal"))
            else:
                self.osc.stop_all_axes()
            return False, vertical_result[1] or "VRChat OSC locomotion send failed"
        horizontal_result = self.osc.set_axis("move_horizontal", normalized_horizontal, duration_s)
        if horizontal_result[0]:
            return True, None
        # A partial locomotion command is unsafe: release both movement axes
        # so a caller never gets an apparently successful half-command.
        rollback = getattr(self.osc, "stop_axes", None)
        if callable(rollback):
            rollback(("move_vertical", "move_horizontal"))
        else:
            self.osc.stop_all_axes()
        return False, horizontal_result[1] or "VRChat OSC locomotion send failed"

    def set_turn(self, horizontal: Any, duration_ms: Any) -> tuple[bool, str | None]:
        try:
            normalized_horizontal = _osc_axis_value(horizontal, "horizontal")
            normalized_duration = _osc_duration_ms(duration_ms, 500)
        except ValueError as exc:
            return False, str(exc)
        if self.config.input.primary == "anyadance" and self.scheduler is not None:
            result = self.scheduler.submit(
                "input_axes",
                {
                    "side": "right",
                    "x": normalized_horizontal,
                    "y": 0.0,
                    "duration_ms": min(normalized_duration, self.config.input.max_hold_ms),
                },
            )
            if result.get("accepted") or not self.config.input.osc_fallback:
                return bool(result.get("accepted")), result.get("reason")
        if self.osc is None:
            return False, "AnyaDance controller input and VRChat OSC bridge are unavailable"
        return self.osc.set_axis(
            "look_horizontal",
            normalized_horizontal,
            normalized_duration / 1000.0,
        )

    def stop_movement(self) -> tuple[bool, str | None]:
        direct_ok = False
        direct_reason: str | None = None
        if self.config.input.primary == "anyadance" and self.scheduler is not None:
            result = self.scheduler.submit("input_release", {"side": "all"})
            direct_ok = bool(result.get("accepted"))
            direct_reason = result.get("reason")
        osc_ok = True
        osc_reason: str | None = None
        if self.osc is not None:
            osc_ok, osc_reason = self.osc.stop_all_axes()
        if direct_ok or osc_ok:
            return True, None
        return False, direct_reason or osc_reason or "movement release failed"

    def set_controller_axes(self, side: Any, x: Any, y: Any, duration_ms: Any) -> tuple[bool, str | None]:
        normalized_side = str(side or "").strip().lower()
        if normalized_side not in {"left", "right"}:
            return False, "side must be left or right"
        try:
            normalized_x = _osc_axis_value(x, "x")
            normalized_y = _osc_axis_value(y, "y")
            normalized_duration = _osc_duration_ms(
                duration_ms,
                1000,
                name="duration_ms",
            )
        except ValueError as exc:
            return False, str(exc)
        if self.config.input.primary == "anyadance" and self.scheduler is not None:
            result = self.scheduler.submit(
                "input_axes",
                {
                    "side": normalized_side,
                    "x": normalized_x,
                    "y": normalized_y,
                    "duration_ms": min(normalized_duration, self.config.input.max_hold_ms),
                },
            )
            if result.get("accepted") or not self.config.input.osc_fallback:
                return bool(result.get("accepted")), result.get("reason")
        if not self.config.input.osc_fallback or self.osc is None:
            return False, "AnyaDance scheduler is not initialized"
        if normalized_side == "left":
            return self.set_locomotion(normalized_y, normalized_x, normalized_duration)
        return self.set_turn(normalized_x, normalized_duration)

    def set_controller_button(
        self,
        side: Any,
        button: Any,
        pressed: Any,
        hold_ms: Any,
        value: Any = 1.0,
    ) -> tuple[bool, str | None]:
        normalized_side = str(side or "").strip().lower()
        normalized_button = str(button or "").strip().lower()
        if normalized_side not in {"left", "right"}:
            return False, "side must be left or right"
        if normalized_button not in {"trigger", "grip", "menu", "a", "b"}:
            return False, "unsupported controller button"
        if not isinstance(pressed, bool):
            return False, "pressed must be a boolean"
        try:
            normalized_value = _controller_value(value, "value")
            normalized_hold = _controller_hold_ms(
                hold_ms,
                self.config.vrchat_osc.input_pulse_ms,
                self.config.input.max_hold_ms,
            )
        except ValueError as exc:
            return False, str(exc)
        if self.config.input.primary == "anyadance" and self.scheduler is not None:
            result = self.scheduler.submit(
                "input_button",
                {
                    "side": normalized_side,
                    "button": normalized_button,
                    "pressed": pressed,
                    "value": normalized_value if pressed else 0.0,
                    "hold_ms": min(normalized_hold, self.config.input.max_hold_ms),
                },
            )
            if result.get("accepted") or not self.config.input.osc_fallback:
                return bool(result.get("accepted")), result.get("reason")
        if not self.config.input.osc_fallback or self.osc is None:
            return False, "AnyaDance scheduler is not initialized"
        osc_action = {"grip": "grab", "trigger": "use"}.get(normalized_button)
        if osc_action is None:
            return False, "VRChat OSC fallback does not expose this Index button"
        return self.osc.pulse_input(
            osc_action if pressed else "drop",
            normalized_side,
            min(normalized_hold, _OSC_HOLD_MAX_MS),
        )

    def release_controller_inputs(self, side: Any = "all") -> tuple[bool, str | None]:
        normalized_side = str(side or "all").strip().lower()
        if normalized_side not in {"left", "right", "all"}:
            return False, "side must be left, right, or all"
        direct_ok = False
        direct_reason: str | None = None
        if self.scheduler is not None:
            result = self.scheduler.submit("input_release", {"side": normalized_side})
            direct_ok = bool(result.get("accepted"))
            direct_reason = result.get("reason")
        if direct_ok or not self.config.input.osc_fallback:
            return direct_ok, direct_reason
        if self.osc is not None:
            osc_ok, osc_reason = self.osc.stop_all_axes()
            return osc_ok, osc_reason
        return False, direct_reason or "AnyaDance scheduler is not initialized"

    def send_osc_batch(self, commands: Any) -> dict[str, Any]:
        """Apply a bounded batch of low-latency OSC commands.

        Axis commands are latest-wins inside the batch.  If any command fails
        after an axis was touched, all movement axes are released before the
        error is returned.
        """
        if not isinstance(commands, list) or not commands:
            return {"accepted": False, "results": [], "reason": "commands must be a non-empty array"}
        if len(commands) > 8:
            return {"accepted": False, "results": [], "reason": "commands must contain at most 8 items"}
        results: list[dict[str, Any]] = []
        axis_touched = False
        for index, command in enumerate(commands):
            if not isinstance(command, Mapping):
                reason = f"commands[{index}] must be an object"
                if axis_touched:
                    self.stop_movement()
                return {"accepted": False, "results": results, "reason": reason}
            kind = str(command.get("kind") or "").strip().lower()
            if kind == "locomotion":
                result = self.set_locomotion(
                    command.get("vertical", 0),
                    command.get("horizontal", 0),
                    command.get("duration_ms", 1000),
                )
                axis_touched = True
            elif kind == "turn":
                result = self.set_turn(
                    command.get("horizontal"),
                    command.get("duration_ms", 500),
                )
                axis_touched = True
            elif kind == "stop_movement":
                result = self.stop_movement()
                axis_touched = True
            elif kind == "parameter":
                result = self.send_avatar_parameter(command.get("name", ""), command.get("value"))
            elif kind == "input":
                result = self.pulse_input(
                    command.get("action", ""),
                    command.get("side", ""),
                    command.get("hold_ms", 100),
                )
            elif kind == "chatbox":
                result = self.send_chatbox(
                    command.get("text", ""),
                    command.get("immediate", True),
                )
            else:
                result = (False, f"unknown batch command kind: {kind or '<empty>'}")
            accepted, reason = result
            results.append({"index": index, "kind": kind, "accepted": accepted, "reason": reason})
            if not accepted:
                if axis_touched:
                    self.stop_movement()
                return {
                    "accepted": False,
                    "results": results,
                    "failed_index": index,
                    "reason": reason or "batch command failed",
                }
        return {"accepted": True, "results": results, "reason": None}

    def send_chatbox(self, text: Any, immediate: Any) -> tuple[bool, str | None]:
        if not isinstance(text, str):
            return False, "text must be a string"
        if not isinstance(immediate, bool):
            return False, "immediate must be a boolean"
        message = text.replace("\x00", "").strip()
        if not message or len(message) > 144:
            return False, "text must be between 1 and 144 characters"
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        return self.osc.send_chatbox(message, immediate=immediate)

    def cancel_inputs(self) -> None:
        if self.osc:
            self.osc.cancel_scheduled_inputs(release=True)

    def list_clips(self) -> dict[str, Any]:
        return self.clip_library.list()

    def semantic_express(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """在后端选择 VMD，然后提交生成的动作。"""
        intent = str(params.get("intent") or "")
        side = str(params.get("side") or "auto")
        intensity = params.get("intensity")
        duration_ms = params.get("duration_ms")
        # 将左右交替状态和动作选择器放在同一处；插件无需复制后端策略或动作状态。
        alternate_side = str(
            params.get("alternate_side")
            or ("left" if self._expression_side_count % 2 else "right")
        )
        selection_index = self._motion_intent_counts.get(intent, 0)
        if self.config.behavior.prefer_vmd_expressions:
            metadata = self.clip_library.select_for_intent(
                intent,
                side=side,
                intensity=float(intensity) if intensity is not None else None,
                sequence_index=selection_index,
            )
            if metadata is not None:
                try:
                    clip = self.clip_library.load(metadata["name"])
                    speed = float(metadata["recommended_speed"])
                    if duration_ms is not None and not clip.is_pose:
                        speed = min(3.0, max(0.25, clip.duration_s / (float(duration_ms) / 1000.0)))
                    result = self.submit("semantic_clip", {
                        "clip_name": clip.name,
                        "speed": speed,
                        "loop_count": int(metadata["loop_count"]),
                        "transition_ms": int(metadata["transition_ms"]),
                        "anchor": True,
                        "restore_after": bool(metadata["restore_after"]),
                        "semantic_intent": intent,
                        "motion_source": str(metadata["source_kind"]),
                        "motion_label": str(metadata["label"]),
                        "source_name": str(metadata["source_name"]),
                        "requested_intensity": intensity,
                        "requested_duration_ms": duration_ms,
                        "_clip": clip,
                    })
                except (KeyError, OSError, ValueError):
                    result = None
                if result is not None:
                    if result.get("accepted"):
                        self._motion_intent_counts[intent] = selection_index + 1
                        if side == "auto":
                            self._expression_side_count += 1
                    return result

        resolved = resolve_expression(
            intent,
            side=side,
            intensity=float(intensity) if intensity is not None else None,
            duration_ms=int(duration_ms) if duration_ms is not None else None,
            alternate_side=alternate_side,
        )
        result = self.submit("express", resolved)
        if result.get("accepted") and side == "auto":
            self._expression_side_count += 1
        return result

    def _on_vision_observation(
        self,
        observation: VisionObservation,
        _result: Mapping[str, Any],
    ) -> None:
        def confidence(item: Any) -> float:
            raw = item.get("confidence") if isinstance(item, Mapping) else getattr(item, "confidence", 0.0)
            try:
                numeric = float(raw)
            except (TypeError, ValueError, OverflowError):
                return 0.0
            return min(1.0, max(0.0, numeric)) if math.isfinite(numeric) else 0.0

        normalized_entities = [
            item for item in (_result.get("entities") or ()) if isinstance(item, Mapping)
        ]
        normalized_events = [
            item for item in (_result.get("events") or ()) if isinstance(item, Mapping)
        ]
        candidates = [confidence(item) for item in normalized_entities]
        candidates.extend(confidence(item) for item in normalized_events)
        self.cognition.observe(
            observation.source,
            "world_observation",
            {
                "entity_count": len(normalized_entities),
                "event_count": len(normalized_events),
                "available": bool(_result.get("available")),
            },
            confidence=max(candidates, default=0.0),
            observed_at=observation.observed_at,
            frame_id=observation.frame_id,
        )
        self.autonomy.update_world(self.vision.snapshot())

    def ingest_world(
        self,
        observation: Mapping[str, Any],
        *,
        ack_only: bool = False,
    ) -> dict[str, Any]:
        result = self.vision.ingest(observation)
        self.autonomy.update_world(result)
        if not ack_only:
            return result
        changes = result.get("changes") if isinstance(result.get("changes"), Mapping) else {}
        return {
            "accepted": True,
            "source": str(observation.get("source") or "vision"),
            "frame_id": str(observation.get("frame_id")) if observation.get("frame_id") is not None else None,
            "available": bool(result.get("available")),
            "entity_count": len(result.get("entities") or ()),
            "event_count": len(result.get("events") or ()),
            "observation_count": (result.get("status") or {}).get("observation_count", 0),
            "revision": (result.get("status") or {}).get("revision", 0),
            # WorldStateStore bounds this list with max_removals; do not apply a
            # smaller transport-side slice that makes an otherwise atomic ack
            # impossible to reconcile.
            "removed_entity_ids": list(changes.get("removed_entity_ids") or ()),
            "removed_entity_count": int(changes.get("removed_entity_count", 0) or 0),
        }

    def plan(self, goal: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.cognition.plan(goal)

    def cognition_feedback(self, feedback: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.cognition.feedback(feedback)


__all__ = ["BackendService"]
