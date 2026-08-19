"""由独立 AnyaDance 后端进程持有的运行时。"""

from __future__ import annotations

from collections.abc import Mapping
import math
import threading
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
from .vision import FrameDetector, FrameSource, VisionObservation, VisionRuntime, VisionWorker
from .world_state import WorldStateStore


_VMC_CALIBRATION_TIMEOUT_SECONDS = 8.0
_VMC_CALIBRATION_RETRY_SECONDS = 5.0
_WORLD_GATE_BYPASS_ACTIONS = frozenset({"stop", "disable", "reset", "cancel"})


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
        self._vision_source = vision_source
        self.clip_library = ClipLibrary(self.config_dir / self.config.clip_directory, self.config)
        self.world_state = WorldStateStore()
        self.vision = VisionRuntime(
            self.world_state,
            detector=vision_detector,
            observation_callback=self._on_vision_observation,
        )
        self.vision_worker: VisionWorker | None = None
        if self.config.vision.enabled and vision_source is not None and vision_detector is not None:
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
        return {
            "body": body,
            "vrchat_osc": self.osc.snapshot() if self.osc else {"enabled": False},
            "driver_log": self.driver_log.snapshot() if self.driver_log else {"enabled": False},
            "idle_relay": self.vmc_idle.snapshot() if self.vmc_idle else {"enabled": False},
            "host_vmc": self.host_vmc.snapshot() if self.host_vmc else {"managed": False, "active": False},
            "world": self.vision.snapshot(),
            "vision_worker": self.vision_worker.status() if self.vision_worker else {
                "enabled": False,
                "running": False,
                "reason": "not_configured",
            },
            "cognition": self.cognition.snapshot(),
            "backend": {
                "started": self._started,
                "dry_run": self.dry_run,
                "last_error": self._last_error,
                "pid": __import__("os").getpid(),
            },
        }

    def awareness(self) -> dict[str, Any]:
        body = self.snapshot()["body"]
        awareness = dict(body.get("awareness") or {})
        awareness["vrchat_osc"] = self.osc.awareness() if self.osc else {"enabled": False}
        return awareness

    def send_avatar_parameter(self, name: str, value: Any) -> tuple[bool, str | None]:
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        return self.osc.send_parameter(name, value)

    def pulse_input(self, action: str, side: str, hold_ms: int) -> tuple[bool, str | None]:
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        return self.osc.pulse_input(action, side, hold_ms)

    def set_locomotion(self, vertical: float, horizontal: float, duration_ms: int) -> tuple[bool, str | None]:
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        duration_s = duration_ms / 1000.0
        self.osc.set_axis("move_vertical", vertical, duration_s)
        self.osc.set_axis("move_horizontal", horizontal, duration_s)
        return True, None

    def set_turn(self, horizontal: float, duration_ms: int) -> tuple[bool, str | None]:
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        self.osc.set_axis("look_horizontal", horizontal, duration_ms / 1000.0)
        return True, None

    def stop_movement(self) -> tuple[bool, str | None]:
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        self.osc.stop_all_axes()
        return True, None

    def send_chatbox(self, text: str, immediate: bool) -> tuple[bool, str | None]:
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        return self.osc.send_chatbox(text, immediate=immediate)

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

    def ingest_world(
        self,
        observation: Mapping[str, Any],
        *,
        ack_only: bool = False,
    ) -> dict[str, Any]:
        result = self.vision.ingest(observation)
        if not ack_only:
            return result
        return {
            "accepted": True,
            "source": str(observation.get("source") or "vision"),
            "frame_id": str(observation.get("frame_id")) if observation.get("frame_id") is not None else None,
            "available": bool(result.get("available")),
            "entity_count": len(result.get("entities") or ()),
            "event_count": len(result.get("events") or ()),
            "observation_count": (result.get("status") or {}).get("observation_count", 0),
        }

    def plan(self, goal: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.cognition.plan(goal)

    def cognition_feedback(self, feedback: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.cognition.feedback(feedback)


__all__ = ["BackendService"]
