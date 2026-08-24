"""由独立 AnyaDance 后端进程持有的运行时。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    cap_openmp_threads,
    DesktopMirrorFrameSource,
    DxcamFrameSource,
    find_window_region,
    FrameDetector,
    FrameSource,
    OpenVinoLocalDetector,
    OpenAICompatibleSemanticBackend,
    MssFrameSource,
    VisionObservation,
    VisionRuntime,
    VisionWorker,
    WindowTrackedFrameSource,
)
from .world_state import WorldStateStore


_VMC_CALIBRATION_TIMEOUT_SECONDS = 8.0
_VMC_CALIBRATION_RETRY_SECONDS = 5.0
_WORLD_GATE_BYPASS_ACTIONS = frozenset({"stop", "disable", "reset", "cancel"})
# 手掌朝向由 motion.palm_rotation 解析，非法值在调度线程上抛 ValueError。
# 那时 submit() 早已返回 accepted=true，所以要在入队前挡下来。
_PALM_ORIENTATIONS = frozenset({"neutral", "forward", "down", "inward"})
_PALM_ACTIONS = frozenset({"arm_pose", "move_hand"})
_OSC_AXIS_MIN = -1.0
_OSC_AXIS_MAX = 1.0
_OSC_DURATION_MIN_MS = 100
_OSC_DURATION_MAX_MS = 10000
_OSC_HOLD_MIN_MS = 20
_OSC_HOLD_MAX_MS = 1000
# 满舵转向速度。把 (horizontal, duration_ms) 这套摇杆语义换算成角度：
# horizontal=1.0 保持 500ms 就是 90°。调度器还会按 safety.max_angular_speed_dps
# 再限一次速，这个常数只决定「一条指令要转多少」。
TURN_SPEED_DPS = 180.0


def _effective_detector_interval_ms(config: Any, detector: Any | None) -> int:
    """按实际推理设备选择检测间隔，未知状态一律走 CPU 安全值。"""
    fallback = max(0, int(config.detector_interval_ms))
    if detector is None:
        return fallback
    try:
        status = detector.status()
    except Exception:
        return fallback
    if not isinstance(status, Mapping):
        return fallback
    runtime = str(status.get("runtime") or "").strip().lower()
    resolved = str(status.get("resolved_device") or "").split(".", 1)[0].upper()
    accelerated = (
        runtime == "openvino" and resolved in {"GPU", "NPU"}
    ) or (
        runtime == "onnxruntime_cuda" and resolved == "CUDA"
    )
    if accelerated:
        return max(0, int(config.detector_accelerator_interval_ms))
    return fallback


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
    """归一化扳机/握把数值（AnyaDance 协议范围为 0..1）。"""
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
        vision_source_factory: Callable[[], FrameSource] | None = None,
    ) -> None:
        self.config = PluginConfig.from_mapping(config_data)
        self.config_dir = Path(config_dir)
        self.logger = logger
        self.dry_run = bool(dry_run)
        # 必须在这里，而且必须在建采集源／检测器之前：OpenMP 只在初始化时读一次
        # 环境变量，而本仓的 numpy 全是函数内惰性导入，所以此刻还没人导入过它。
        # 往后挪一行到 _build_configured_vision_source 之后就晚了——dxcam 会先把
        # numpy 拉进来，环境变量随之失效，只剩 ctypes 那条降级路。
        # 收的是 numpy/BLAS 的池，不是 YOLO 的：实测采集路径空转自旋就吃 7.23 核。
        self._openmp = cap_openmp_threads(self.config.vision.detector_threads)
        # FrameSource 持有操作系统句柄，调用 ``close()`` 后刻意不再复用。
        # 为已配置的来源保留工厂，使视觉停止/启动接口可以销毁并重新创建句柄。
        # 注入的来源只保证一个生命周期；测试或旁路进程可显式提供工厂来重复创建。
        self._vision_source_factory = vision_source_factory
        self._vision_source_external = vision_source is not None and vision_source_factory is None
        self._vision_stop_reason = "not_configured"
        if vision_source is None:
            vision_source = self._build_configured_vision_source()
        self._vision_source = vision_source
        if vision_detector is None and self.config.vision.enabled and self.config.vision.local_backend == "openvino":
            configured_model = self.config.vision.model_path or os.getenv("VRC_OPENVINO_MODEL")
            configured_labels = self.config.vision.labels_path or os.getenv("VRC_OPENVINO_LABELS")
            # plugin.toml 中的路径相对于后端配置目录，而不是进程工作目录。
            # 未配置路径时，环境变量覆盖仍保持历史上的绝对/相对路径行为。
            if configured_model and not Path(configured_model).is_absolute() and self.config.vision.model_path:
                configured_model = str(self.config_dir / configured_model)
            if configured_labels and not Path(configured_labels).is_absolute() and self.config.vision.labels_path:
                configured_labels = str(self.config_dir / configured_labels)
            vision_detector = OpenVinoLocalDetector(
                model_path=configured_model,
                labels_path=configured_labels,
                device=self.config.vision.device,
                onnxruntime_cuda=self.config.vision.onnxruntime_cuda,
                onnxruntime_cuda_device_id=self.config.vision.onnxruntime_cuda_device_id,
                confidence_threshold=self.config.vision.confidence_threshold,
                input_width=self.config.vision.input_width,
                input_height=self.config.vision.input_height,
                horizontal_fov_deg=self.config.vision.horizontal_fov_deg,
                max_detections=self.config.vision.max_detections,
                min_box_ratio=self.config.vision.min_box_ratio,
                min_box_width_ratio=self.config.vision.min_box_width_ratio,
                min_box_height_ratio=self.config.vision.min_box_height_ratio,
                identity_reid_enabled=self.config.vision.identity_reid_enabled,
                identity_reid_similarity=self.config.vision.identity_reid_similarity,
                identity_reid_margin=self.config.vision.identity_reid_margin,
                identity_reid_retention_s=self.config.vision.identity_reid_retention_s,
                identity_reid_max_identities=self.config.vision.identity_reid_max_identities,
                fallback_backend=self.config.vision.fallback_backend,
                intra_op_threads=self.config.vision.detector_threads,
            )
        vision_semantic: Any | None = None
        if self.config.vision.enabled and self.config.vision.semantic_backend == "openai_compatible":
            endpoint = (
                os.getenv("VRC_VLM_ENDPOINT")
                or os.getenv("OPENAI_BASE_URL")
                or self.config.vision.semantic_endpoint
            )
            model = (
                os.getenv("VRC_VLM_MODEL")
                or os.getenv("OPENAI_VLM_MODEL")
                or self.config.vision.semantic_model
                or "gpt-4o-mini"
            )
            api_key = os.getenv("VRC_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if endpoint:
                vision_semantic = OpenAICompatibleSemanticBackend(
                    endpoint=endpoint,
                    model=model,
                    api_key=api_key,
                    max_per_minute=self.config.vision.semantic_max_per_minute,
                )
        self.clip_library = ClipLibrary(self.config_dir / self.config.clip_directory, self.config)
        self.world_state = WorldStateStore(
            lifecycle_watermark_limit=self.config.vision.lifecycle_watermark_limit,
            persistence_path=self.config_dir / "world_memory.json",
            # 除非调用方明确提供该配置段，否则隔离库/测试调用方；随插件发布的
            # plugin.toml 会启用持久化配置段。
            persist_world=self.config.world_memory.persist_world and "world_memory" in config_data,
            persist_players=self.config.world_memory.persist_players and "world_memory" in config_data,
        )
        effective_detector_interval_ms = _effective_detector_interval_ms(
            self.config.vision, vision_detector
        )
        self.vision = VisionRuntime(
            self.world_state,
            detector=vision_detector,
            semantic=vision_semantic,
            main_llm_semantic=(self.config.vision.semantic_backend == "main_llm"),
            main_llm_min_interval_s=self.config.vision.semantic_main_llm_min_interval_s,
            observation_callback=self._on_vision_observation,
            detect_interval_s=effective_detector_interval_ms / 1000.0,
            frame_cache_interval_s=self.config.vision.frame_cache_interval_s,
            frame_cache_max_width=self.config.vision.frame_max_width,
            frame_cache_quality=self.config.vision.frame_jpeg_quality,
        )
        self.vision_worker: VisionWorker | None = None
        if self.config.vision.enabled and vision_source is not None:
            self.vision_worker = self._new_vision_worker(vision_source)
            self._vision_stop_reason = "configured"
            # 后端进程尚未启动 worker，不要把持久化或最近的世界数据当成新帧暴露。
            self.vision.set_capture_state(False, "not_started")
        else:
            self.vision.set_capture_state(
                False,
                "disabled_in_config" if not self.config.vision.enabled else "not_configured",
            )
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
            world_provider=lambda: self.vision.snapshot(),
            release_inputs=self._release_all_inputs,
            session_ttl_s=self.config.autonomy.session_ttl_minutes * 60.0,
        )
        # 导航器是本地有界控制环，刻意与 LLM 和视觉 worker 分离；只有观察到
        # 新鲜且可见的目标后，才发送短时 AnyaDance 轴更新。
        self.navigator = LocalNavigator(
            world_provider=lambda: self.vision.snapshot(),
            goal_provider=lambda: self.autonomy.snapshot(),
            send_axes=self._navigator_send_axes,
            send_turn=self._navigator_send_turn,
            release_inputs=self._navigator_release_inputs,
            motion_provider=self._navigator_motion_feedback,
            turn_state_provider=self._navigator_turn_state,
            complete_goal=self._navigator_complete_goal,
            turn_retarget_supported=True,
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
            world_provider=lambda: self.vision.snapshot(),
        )

    def _new_vision_worker(self, source: FrameSource) -> VisionWorker:
        # 已配置的检测器对象可能存在，但可选模型不可用（例如未部署 OpenVINO
        # IR 模型包）。此时采集路径仍应可用：帧可以计数和诊断，但不能发布猜测的实体。
        candidates = [item for item in (self.vision.detector, self.vision.semantic) if item is not None]
        backend_available = False
        for candidate in candidates:
            try:
                status = candidate.status()
                if not isinstance(status, Mapping) or status.get("available") is not False:
                    backend_available = True
                    break
            except Exception:
                # 遵循 VisionWorker 的保守策略：探测后端时若抛出异常，视为可能可用，
                # 交给处理循环报告，而不是直接禁用采集。
                backend_available = True
                break
        return VisionWorker(
            self.vision,
            source,
            interval_s=self.config.vision.interval_ms / 1000.0,
            queue_size=self.config.vision.queue_size,
            capture_only=not backend_available,
        )

    def _build_configured_vision_source(self) -> FrameSource | None:
        """创建一个新的已配置采集源。

        本函数只负责构造来源。初始化和每次启动视觉 worker 时都会调用它，
        因此关闭后的 DXcam/MSS/WinRT 对象不会被交给新的 worker。

        若配置了 ``window_title``，采集区域取该窗口的屏幕坐标。窗口矩形不是
        一次性的：窗口被拖动或改分辨率后旧坐标就会一直抓错位置，因此默认包一层
        ``WindowTrackedFrameSource`` 按 ``window_track_interval_ms`` 重新解析。
        窗口未找到时回落到无裁剪模式，以免采集完全停止。
        """
        vision = self.config.vision
        if not vision.enabled or vision.source == "external" or vision.capture == "external":
            return None

        if vision.capture == "mss" or vision.source == "mss":
            def build(region: Mapping[str, int] | None) -> FrameSource:
                return MssFrameSource(monitor_index=vision.monitor_index, region=region)
        elif vision.capture == "dxcam":
            def build(region: Mapping[str, int] | None) -> FrameSource:
                return DxcamFrameSource(
                    device_idx=vision.dxcam_device_idx,
                    output_idx=vision.dxcam_output_idx,
                    backend=vision.dxcam_backend,
                    region=region,
                )
        elif vision.capture == "desktop_mirror":
            def build(region: Mapping[str, int] | None) -> FrameSource:
                return DesktopMirrorFrameSource(
                    monitor_index=vision.monitor_index,
                    dxcam_device_idx=vision.dxcam_device_idx,
                    dxcam_output_idx=vision.dxcam_output_idx,
                    dxcam_backend=vision.dxcam_backend,
                    region=region,
                )
        else:
            return None

        if not vision.window_title:
            return build(None)
        if vision.window_track_interval_ms > 0:
            return WindowTrackedFrameSource(
                title=vision.window_title,
                factory=build,
                interval_s=vision.window_track_interval_ms / 1000.0,
            )
        # 显式关闭跟踪：保持历史行为，只在启动时解析一次窗口坐标。
        return build(find_window_region(vision.window_title))

    def _fresh_vision_source(self) -> FrameSource | None:
        """返回新分配的来源；无法重启时返回 ``None``。"""
        factory = self._vision_source_factory
        if factory is not None:
            source = factory()
            if source is None:
                raise RuntimeError("vision source factory returned no source")
            return source
        if self._vision_source_external:
            return None
        return self._build_configured_vision_source()

    @staticmethod
    def _vision_worker_not_configured(reason: str = "not_configured") -> dict[str, Any]:
        return {
            "enabled": False,
            "running": False,
            "capture_only": False,
            "reason": reason,
        }

    def _vision_worker_status(self) -> dict[str, Any]:
        worker = self.vision_worker
        if worker is None:
            return self._vision_worker_not_configured(self._vision_stop_reason)
        return worker.status()

    def _stop_vision_worker_locked(self, *, reason: str = "stopped") -> None:
        """在持有 ``_lock`` 时停止并丢弃 worker 及其来源。"""
        worker = self.vision_worker
        if worker is not None:
            try:
                worker.stop()
            finally:
                # ``VisionWorker.stop`` 会关闭来源，不能把已关闭的对象留给后续启动。
                self.vision_worker = None
        # 即使没有 worker 对象（例如重复停止或来源创建失败），也要让运行时门控
        # 保持权威。VisionWorker 的直接调用使用通用停止原因，这里恢复控制面原因，
        # 让 `/perception` 能解释采集为何结束。
        self.vision.set_capture_state(False, str(reason or "stopped"))
        self._vision_source = None
        self._vision_stop_reason = str(reason or "stopped")

    def _cognition_sources(self) -> dict[str, dict[str, Any]]:
        """暴露各数据源健康状况，且不会递归调用 ``snapshot``。"""
        body = self.scheduler.snapshot() if self.scheduler else {"state": "shutdown"}
        osc = self.osc.snapshot() if self.osc else {"enabled": False}
        driver = self.driver_log.snapshot() if self.driver_log else {"enabled": False}
        vmc = self.vmc_idle.snapshot() if self.vmc_idle else {"enabled": False}
        world = self.vision.snapshot()
        worker = self._vision_worker_status()
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
        """利用驱动遥测检测是否有其他进程写入 UDP 最新状态。"""
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
                if self.vision_worker is None and self.config.vision.enabled:
                    source = self._fresh_vision_source()
                    if source is not None:
                        self._vision_source = source
                        self.vision_worker = self._new_vision_worker(source)
                if self.vision_worker is not None:
                    if self.vision_worker.start():
                        self._vision_stop_reason = "running"
                self.navigator.start()
                self._started = True
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                self.stop()
                raise

    def _navigator_complete_goal(self, reason: str) -> None:
        """导航总时限到期时同时撤销语义单槽，避免后台继续刷新图片。"""
        self.vision.clear_main_llm_semantic_request(reason)
        self.autonomy.complete_goal(reason)

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
            # 先解除授权，释放虚拟控制器叠加层和 OSC 回退，再拆除 worker/套接字。
            try:
                self.autonomy.disarm("backend_stopped")
            except Exception:
                pass
            self.navigator.stop()
            self._stop_vmc_calibration()
            self._stop_vision_worker_locked(reason="backend_stopped")
            self.vision.close()
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
        """用于自主控制和关闭路径的尽力急停释放。"""
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
        normalized_reason = str(reason or "manual_disarm")
        self.vision.clear_main_llm_semantic_request(normalized_reason)
        return {"accepted": True, **self.autonomy.disarm(normalized_reason)}

    def autonomy_goal(
        self,
        text: Any,
        kind: Any = "explore",
        target_id: Any = None,
        selector: Any = None,
        constraints: Any = None,
        based_on_revision: Any = None,
    ) -> dict[str, Any]:
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
        result = self.autonomy.submit_goal(
            text,
            kind,
            target_id,
            selector,
            constraints,
            based_on_revision,
        )
        if result.get("accepted"):
            goal = result.get("goal") if isinstance(result.get("goal"), Mapping) else {}
            normalized_selector = goal.get("selector") if isinstance(goal.get("selector"), Mapping) else None
            if normalized_selector is not None:
                # 建立被动任务，不在这里调用模型。插件会把这张图并入当前/下一次
                # 主 LLM 对话，LocalNavigator 在此期间继续自己的高频循环。
                result["semantic_request"] = self.vision.request_main_llm_semantics(
                    normalized_selector,
                    reason="autonomy_selector_submitted",
                    force=True,
                )
            else:
                self.vision.clear_main_llm_semantic_request("goal_replaced_without_selector")
        return result

    def autonomy_intent(
        self,
        action: Any,
        text: Any = None,
        target_id: Any = None,
        target_type: Any = "npc",
        target_label: Any = None,
        min_confidence: Any = 0.25,
        constraints: Any = None,
    ) -> dict[str, Any]:
        """把 Agent 的单步自然语言意图收紧为安全的本地导航目标。

        普通插件执行器一次通常只调用一个 entry，无法可靠地先观察、再复制 ID、
        最后提交目标。这里仅在当前画面恰好有一个经语义确认的匹配实体时替它完成
        这段机械绑定；多个候选仍交还主 LLM 选择，绝不按置信度偷偷挑人。
        """
        normalized_action = str(action or "").replace("\x00", "").strip().lower()
        if normalized_action == "status":
            return {"accepted": True, "action": "status", **self.autonomy_snapshot()}
        if normalized_action == "stop":
            return {"action": "stop", **self.autonomy_stop("agent_navigation_stop")}
        if normalized_action == "inspect_occluded_area":
            return {
                **self.autonomy.snapshot(),
                "accepted": False,
                "action": normalized_action,
                "reason_code": "unsupported_spatial_navigation",
                "reason": (
                    "current perception has no depth, collision map or SLAM; "
                    "it cannot route to or verify an occluded area"
                ),
                "instruction": "请让用户手动带路到可见位置，再观察新鲜画面。",
            }
        if normalized_action not in {"find", "approach", "follow"}:
            return {
                **self.autonomy.snapshot(),
                "accepted": False,
                "reason_code": "invalid_action",
                "reason": "action must be find, approach, follow, stop or status",
            }

        autonomy = self.autonomy.snapshot()
        if not autonomy.get("armed"):
            return {
                **autonomy,
                "accepted": False,
                "reason_code": "manual_arm_required",
                "reason": "VRChat autonomy is not manually armed",
                "instruction": "请先在 AnyaDance 身体调试台点击“启用自主控制”，再重试移动命令。",
            }

        normalized_type = str(target_type or "").replace("\x00", "").strip().lower()[:32]
        normalized_label = str(target_label or "").replace("\x00", "").strip()[:64]
        try:
            normalized_confidence = float(min_confidence)
        except (TypeError, ValueError, OverflowError):
            normalized_confidence = math.nan
        if not math.isfinite(normalized_confidence) or not 0.0 <= normalized_confidence <= 1.0:
            return {
                **autonomy,
                "accepted": False,
                "reason_code": "invalid_min_confidence",
                "reason": "min_confidence must be between 0 and 1",
            }
        selector: dict[str, Any] = {"min_confidence": normalized_confidence}
        if normalized_type:
            selector["semantic_type"] = normalized_type
        if normalized_label:
            selector["label"] = normalized_label
        if len(selector) == 1:
            return {
                **autonomy,
                "accepted": False,
                "reason_code": "selector_required",
                "reason": "target_type or target_label is required",
            }

        world = self.vision.snapshot()
        # 普通 Agent entry 可能正好落在 detector 回调之间；用本次精确快照同步
        # revision，避免把自己刚读到的 revision 误判成“来自未来”。
        self.autonomy.update_world(world)
        status = world.get("status") if isinstance(world.get("status"), Mapping) else {}
        revision = int(status.get("revision", 0) or 0)
        normalized_text = str(text or "").replace("\x00", "").strip()[:256]
        if normalized_action == "find":
            result = self.autonomy_goal(
                normalized_text or f"寻找 {normalized_label or normalized_type}",
                "explore",
                None,
                selector,
                constraints,
                revision,
            )
            return {"action": "find", "resolved_by": "semantic_selector", **result}

        normalized_target_id = str(target_id or "").replace("\x00", "").strip()[:96]
        candidates = self._semantic_navigation_candidates(world, selector)
        if not normalized_target_id:
            if len(candidates) != 1:
                return {
                    **autonomy,
                    "accepted": False,
                    "action": normalized_action,
                    "reason_code": (
                        "semantic_target_not_found" if not candidates else "target_choice_required"
                    ),
                    "reason": (
                        "no currently visible semantic target matches the selector"
                        if not candidates else
                        "multiple semantic targets match; the main LLM must choose an exact target_id"
                    ),
                    "candidates": candidates,
                }
            normalized_target_id = str(candidates[0]["target_id"])
            resolved_by = "unique_semantic_target"
        else:
            resolved_by = "exact_target_id"
        result = self.autonomy_goal(
            normalized_text or f"{normalized_action} {normalized_label or normalized_type}",
            normalized_action,
            normalized_target_id,
            None,
            constraints,
            revision,
        )
        return {
            "action": normalized_action,
            "resolved_by": resolved_by,
            "resolved_target_id": normalized_target_id,
            **result,
        }

    @staticmethod
    def _semantic_navigation_candidates(
        world: Mapping[str, Any],
        selector: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """返回当前可见且经语义确认的候选；负类别永远不能进入导航。"""
        expected_type = str(selector.get("semantic_type") or "").strip().lower()
        expected_label = str(selector.get("label") or "").strip().casefold()
        minimum = float(selector.get("min_confidence", 0.0) or 0.0)
        candidates: list[dict[str, Any]] = []
        entities = world.get("entities") if isinstance(world.get("entities"), list) else []
        for entity in entities:
            if not isinstance(entity, Mapping) or not entity.get("visible"):
                continue
            attributes = entity.get("attributes") if isinstance(entity.get("attributes"), Mapping) else {}
            semantic_type = str(attributes.get("semantic_type") or "").strip().lower()
            if not attributes.get("semantic_verified") or semantic_type in {
                "", "poster", "screen", "mirror", "unknown",
            }:
                continue
            label = str(entity.get("label") or "")
            confidence = float(entity.get("confidence", 0.0) or 0.0)
            if expected_type and semantic_type != expected_type:
                continue
            if expected_label and expected_label not in label.casefold():
                continue
            if confidence < minimum:
                continue
            candidates.append({
                "target_id": str(entity.get("id") or "")[:96],
                "label": label[:64],
                "semantic_type": semantic_type,
                "confidence": round(confidence, 4),
                "bearing_deg": attributes.get("bearing_deg"),
                "apparent_height": attributes.get("apparent_height"),
            })
        return candidates

    def autonomy_stop(self, reason: Any = "autonomy_stop") -> dict[str, Any]:
        normalized_reason = str(reason or "autonomy_stop")
        self.vision.clear_main_llm_semantic_request(normalized_reason)
        return {"accepted": True, **self.autonomy.stop(normalized_reason)}

    def attach_vision(self, source: FrameSource, detector: FrameDetector) -> dict[str, Any]:
        """注入外部 detector；模型包和采集实现由调用方负责。"""
        with self._lock:
            self._stop_vision_worker_locked(reason="replaced")
            self._vision_source_factory = None
            self._vision_source_external = True
            self._vision_source = source
            self.vision.set_backends(detector=detector)
            worker: VisionWorker | None = None
            try:
                worker = self._new_vision_worker(source)
                self.vision_worker = worker
                started = self._started and worker.start()
                if self._started and not started:
                    # 后端已运行时，启动失败代表这次注入没有接管成功；关闭
                    # source 并清掉 worker，避免后续 start() 复用坏对象。
                    failure_status = worker.status()
                    worker.stop()
                    self.vision_worker = None
                    self._vision_source = None
                    self._vision_stop_reason = "attach_start_failed"
                    self.vision.set_capture_state(False, "attach_start_failed")
                    return {
                        "attached": False,
                        "started": False,
                        "reason": failure_status.get("last_error") or "vision worker failed to start",
                        "worker": failure_status,
                    }
                if started:
                    self._vision_stop_reason = "running"
            except Exception as exc:
                # 注入失败时立即释放外部 source，避免句柄泄漏或半初始化 worker
                # 在下一次 start/stop 中继续被复用。
                if worker is not None:
                    try:
                        worker.stop()
                    except Exception:
                        pass
                else:
                    try:
                        source.close()
                    except Exception:
                        pass
                self.vision_worker = None
                self._vision_source = None
                self._vision_stop_reason = "attach_failed"
                self.vision.set_capture_state(False, "attach_failed")
                return {
                    "attached": False,
                    "started": False,
                    "reason": f"{type(exc).__name__}: {exc}"[:500],
                    "worker": self._vision_worker_status(),
                }
        return {
            "attached": True,
            "started": bool(started),
            "worker": worker.status() if worker is not None else self._vision_worker_status(),
        }

    def vision_start(self) -> dict[str, Any]:
        """使用新来源启动采集。

        来源是资源所有权对象：停止 worker 会关闭其 DXcam、WinRT 或 MSS 句柄。
        重启时本方法总是分配新来源，不会尝试复用已经关闭的实例。
        """
        with self._lock:
            if not self.config.vision.enabled:
                self._vision_stop_reason = "disabled_in_config"
                return {
                    "accepted": False,
                    "started": False,
                    "running": False,
                    "reason": "vision is disabled in configuration",
                    "worker": self._vision_worker_status(),
                }
            if self.vision_worker is not None:
                current = self.vision_worker.status()
                if current.get("running"):
                    return {
                        "accepted": True,
                        "started": True,
                        "running": True,
                        "changed": False,
                        "reason": "already_running",
                        "worker": current,
                    }
                # 失败或过期的 worker 可能持有已关闭或不可用的来源，重建前先拆除。
                self._stop_vision_worker_locked(reason="restarting")
            try:
                source = self._fresh_vision_source()
            except Exception as exc:
                self._vision_stop_reason = "source_create_failed"
                return {
                    "accepted": False,
                    "started": False,
                    "running": False,
                    "reason": f"{type(exc).__name__}: {exc}"[:500],
                    "worker": self._vision_worker_status(),
                }
            if source is None:
                self._vision_stop_reason = "external_source_requires_attach"
                return {
                    "accepted": False,
                    "started": False,
                    "running": False,
                    "reason": "vision source is external; attach a fresh source before starting",
                    "worker": self._vision_worker_status(),
                }
            self._vision_source = source
            worker = self._new_vision_worker(source)
            self.vision_worker = worker
            if not self._started:
                self._vision_stop_reason = "prepared"
                return {
                    "accepted": True,
                    "started": False,
                    "running": False,
                    "changed": True,
                    "reason": "backend_not_started",
                    "worker": worker.status(),
                }
            try:
                started = worker.start()
            except Exception as exc:
                failure_status = worker.status()
                self._stop_vision_worker_locked(reason="start_failed")
                self._vision_stop_reason = "start_failed"
                return {
                    "accepted": False,
                    "started": False,
                    "running": False,
                    "changed": True,
                    "reason": f"{type(exc).__name__}: {exc}"[:500],
                    "worker": failure_status,
                }
            if not started:
                failure_status = worker.status()
                # 启动失败时可能已经打开采集句柄。现在关闭并丢弃它，让调用方重试
                # 时获得新来源，避免积累过期的操作系统资源。
                self._stop_vision_worker_locked(reason="start_failed")
                self._vision_stop_reason = "start_failed"
                return {
                    "accepted": False,
                    "started": False,
                    "running": False,
                    "changed": True,
                    "reason": failure_status.get("last_error") or "vision worker failed to start",
                    "worker": failure_status,
                }
            self._vision_stop_reason = "running"
            return {
                "accepted": True,
                "started": True,
                "running": True,
                "changed": True,
                "reason": "started",
                "worker": worker.status(),
            }

    def vision_stop(self, reason: Any = "manual_stop") -> dict[str, Any]:
        """停止采集并释放操作系统资源，但不停止后端。"""
        with self._lock:
            was_running = bool(self.vision_worker and self.vision_worker.status().get("running"))
            # 没有新鲜视觉来源时绝不能继续自主导航。解除授权还会在可能阻塞于
            # 采集/VLM 线程之前释放任何活动中的控制器叠加层。
            try:
                self.autonomy.disarm("vision_stopped")
            except Exception:
                pass
            self._stop_vision_worker_locked(reason=str(reason or "manual_stop"))
            return {
                "accepted": True,
                "stopped": True,
                "running": False,
                "changed": was_running,
                "reason": str(reason or "manual_stop"),
                "worker": self._vision_worker_status(),
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
        if kind in _PALM_ACTIONS and "palm" in normalized:
            palm = normalized["palm"]
            if palm not in _PALM_ORIENTATIONS:
                state = scheduler.snapshot()
                return finish({
                    "accepted": False,
                    "action_id": None,
                    "state": state["state"],
                    "normalized_params": normalized,
                    "reason": (
                        f"palm must be one of "
                        f"{', '.join(sorted(_PALM_ORIENTATIONS))}; got {palm!r}"
                    ),
                    "safety_state": state["safety_state"],
                })
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
            "vision_worker": self._vision_worker_status(),
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
        """记录控制面请求在服务端的分发延迟。"""
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
        """长轮询世界存储，不进入身体控制路径。"""
        result = self.vision.delta(after_revision, wait_ms=wait_ms, limit=limit)
        self.autonomy.update_world(result.get("world"))
        return result

    def perception(self) -> dict[str, Any]:
        return {
            "world": self.vision.snapshot(),
            "navigation": self.navigator.snapshot(),
            "worker": self._vision_worker_status(),
        }

    def vision_frame(self, *, max_age_ms: Any = 3000, overlay: Any = False) -> dict[str, Any]:
        """返回最近一帧的 base64 JPEG，仅供 agent 理解画面使用。

        这条路径与 ``world_state`` 完全分离，也不经过 ``navigator``：帧不产生
        实体、不产生事件，更不能拿去满足 ``body_reach_and_grab`` 的
        ``preconditions``。由画面得出的一切结论都是低置信视觉猜测。

        ``overlay=True`` 叠加检测框用于对照「检测器看到的」与「画面里实际有的」。
        框来自世界快照，但叠框图本身仍然只是像素——它不因为画了框就变成观测。
        """
        try:
            limit_ms = int(max_age_ms)
        except (TypeError, ValueError, OverflowError):
            limit_ms = 3000
        # 下限 250 ms 而不是 0：``latest_frame`` 把 0 当作「不限龄」，于是把
        # 「我要最新的」写成 0 会拿到最旧的一张——正好反了。
        limit_ms = min(30000, max(250, limit_ms))
        result = dict(self.vision.latest_frame(max_age_ms=limit_ms, overlay=bool(overlay)))
        data = result.pop("data", None)
        if isinstance(data, bytes):
            import base64
            result["data_base64"] = base64.b64encode(data).decode("ascii")
        return result

    def main_llm_semantic_request(self, after_request_id: Any = None) -> dict[str, Any]:
        """取一次被动语义任务；JPEG 仍只来自运行时的内存单槽。"""
        normalized_after = str(after_request_id or "").replace("\x00", "").strip()[:128] or None
        result = dict(
            self.vision.main_llm_semantic_request(after_request_id=normalized_after)
        )
        data = result.pop("data", None)
        if isinstance(data, bytes):
            import base64
            result["data_base64"] = base64.b64encode(data).decode("ascii")
        return result

    def main_llm_semantic_commit(
        self,
        request_id: Any,
        frame_revision: Any,
        entities: Any,
    ) -> dict[str, Any]:
        return self.vision.commit_main_llm_semantics(
            request_id,
            frame_revision,
            entities,
        )

    def _navigator_send_axes(self, side: str, x: float, y: float, duration_ms: int) -> bool:
        """只发送主 AnyaDance 命令，绝不回退到 OSC。"""
        scheduler = self.scheduler
        if scheduler is None or self.config.input.primary != "anyadance":
            return False
        result = scheduler.submit(
            "input_axes",
            {"side": side, "x": x, "y": y, "duration_ms": duration_ms},
        )
        return bool(result.get("accepted"))

    def _navigator_send_turn(self, delta_deg: float) -> bool:
        """转向直接进调度器，不经 set_turn。

        set_turn 是摇杆语义（horizontal + duration_ms），duration 有 100ms 下限，
        换算过去等于最小 18°——而导航器的死区只有 8°，小幅修正会被整条拒掉。
        这里要的就是「转这么多度」，中间那层换算只会丢精度。
        """
        scheduler = self.scheduler
        if scheduler is None:
            return False
        # scheduler 的 delta_deg 是相对上一个 target 累加，上一段尚未结束时重发会
        # 超调。correction_deg 则由 scheduler 基于未归一化的当前实际 yaw 计算目标，
        # 既能连续重定向，也不会在 0/360° 边界误转一整圈。
        result = scheduler.submit("turn", {"correction_deg": float(delta_deg)})
        return bool(result.get("accepted"))

    def _navigator_release_inputs(self, side: str = "all") -> None:
        scheduler = self.scheduler
        if scheduler is not None:
            scheduler.submit("input_release", {"side": side})

    def _navigator_motion_feedback(self) -> dict[str, Any]:
        """VRChat 内置 Velocity 参数，供导航器判断是否顶着墙推摇杆。

        OSC 桥没起来时返回 available=false：导航器会退回到没有卡墙判据的旧行为，
        而不是把「读不到」当成「速度为零」然后立刻停车。
        """
        osc = self.osc
        if osc is None:
            return {"available": False, "reason": "osc_unavailable"}
        return osc.motion_feedback()

    def _navigator_turn_state(self) -> dict[str, Any]:
        """给导航器返回虚拟 HMD 转向是否仍在执行或收尾。"""
        scheduler = self.scheduler
        if scheduler is None:
            return {"available": False, "turning": False}
        try:
            snapshot = scheduler.snapshot()
        except Exception:
            return {"available": False, "turning": False}
        heading = snapshot.get("heading") if isinstance(snapshot, Mapping) else None
        if not isinstance(heading, Mapping):
            return {"available": False, "turning": False}
        return {"available": True, **dict(heading)}

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
        """走位固定走 VRChat OSC，不经过 ``input.primary`` 路由。

        移动是游戏输入，不是骨骼姿态。VMC 那条路把摇杆值写成 avatar 的手臂
        pose，看起来像在推摇杆，VRChat 却收不到 ``/input/Move*``——人不动，
        而调用方拿到的是 ``accepted: true``。VMC 只负责待机动作。
        """
        try:
            normalized_vertical = _osc_axis_value(vertical, "vertical")
            normalized_horizontal = _osc_axis_value(horizontal, "horizontal")
            normalized_duration = _osc_duration_ms(duration_ms, 1000)
        except ValueError as exc:
            return False, str(exc)
        if self.osc is None:
            return False, "VRChat OSC bridge is unavailable"
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
        # 部分移动命令不安全：释放两个移动轴，避免调用方收到看似成功的半条命令。
        rollback = getattr(self.osc, "stop_axes", None)
        if callable(rollback):
            rollback(("move_vertical", "move_horizontal"))
        else:
            self.osc.stop_all_axes()
        return False, horizontal_result[1] or "VRChat OSC locomotion send failed"

    def set_turn(self, horizontal: Any, duration_ms: Any) -> tuple[bool, str | None]:
        """转向走 AnyaDance，直接转虚拟 HMD，不经 VRChat 的输入层。

        VR 模式下 ``/input/LookHorizontal`` 是死路：VRChat 只在桌面模式把它当连续
        转向，实测发再干净的边沿镜头也不动，但 OSC 照样回 ``accepted: true``。
        完全虚拟模式下虚拟 HMD 就是相机，转它才是唯一可靠的转向手段，而且实测移动
        方向跟着头走——转向 + 前进因此能组成完整的二维导航。

        这里**不**回落到 OSC。已知在本机配置下那条路径不产生任何效果，回落只会把
        「没转」重新包装成成功，正是之前查了很久的那个坑。
        """
        try:
            normalized_horizontal = _osc_axis_value(horizontal, "horizontal")
            normalized_duration = _osc_duration_ms(duration_ms, 500)
        except ValueError as exc:
            return False, str(exc)
        if self.scheduler is None:
            return False, "AnyaDance scheduler is not initialized"
        delta_deg = normalized_horizontal * TURN_SPEED_DPS * (normalized_duration / 1000.0)
        result = self.scheduler.submit("turn", {"delta_deg": delta_deg})
        return bool(result.get("accepted")), result.get("reason")

    def stop_movement(self) -> tuple[bool, str | None]:
        """释放移动轴。OSC 存在时，它的结果决定成败。

        走位与转向现在只经由 OSC，因此只有 ``stop_all_axes`` 能真正让人停下。
        VMC 覆盖层照常清掉（它持有待机动作的轴），但它成功不能掩盖 OSC 失败：
        那会把"还在走"报成已停住，而调用方不会重试。
        """
        direct_reason: str | None = None
        if self.config.input.primary == "anyadance" and self.scheduler is not None:
            result = self.scheduler.submit("input_release", {"side": "all"})
            direct_reason = result.get("reason")
            direct_ok = bool(result.get("accepted"))
        else:
            direct_ok = False
        if self.scheduler is not None:
            # 转向不在 OSC 轴里，stop_all_axes 碰不到它。不停就会继续转到目标角度。
            self.scheduler.submit("turn", {"halt": True})
        if self.osc is not None:
            osc_ok, osc_reason = self.osc.stop_all_axes()
            if osc_ok:
                return True, None
            return False, osc_reason or "VRChat OSC movement release failed"
        if direct_ok:
            return True, None
        return False, direct_reason or "movement release failed"

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
        """应用一批有界的低延迟 OSC 命令。

        批次内轴命令采用最新值优先。如果某条命令失败且此前已经修改过轴，
        则在返回错误前释放所有移动轴。
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
        if self.config.vision.semantic_backend == "main_llm":
            autonomy = self.autonomy.snapshot()
            goal = autonomy.get("goal") if isinstance(autonomy.get("goal"), Mapping) else {}
            selector = goal.get("selector") if isinstance(goal.get("selector"), Mapping) else None
            if selector is not None:
                if self._semantic_selector_is_satisfied(selector, normalized_entities):
                    # 结果已在当前本地帧重新绑定，尚未被宿主消费的旧任务没有价值。
                    self.vision.clear_main_llm_semantic_request()
                else:
                    # 本地 detector 每帧都可到这里，但运行时有 pending 单槽与最小间隔；
                    # 因而这里只表达“仍未解决”，不会按帧生成主 LLM 图片。
                    self.vision.request_main_llm_semantics(
                        selector,
                        reason="autonomy_selector_unresolved",
                    )

    @staticmethod
    def _semantic_selector_is_satisfied(
        selector: Mapping[str, Any],
        entities: list[Mapping[str, Any]],
    ) -> bool:
        expected_type = str(selector.get("semantic_type") or "").strip().lower()
        expected_label = str(selector.get("label") or "").strip().casefold()
        try:
            minimum = float(selector.get("min_confidence", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            minimum = 0.0
        for entity in entities:
            attributes = entity.get("attributes") if isinstance(entity.get("attributes"), Mapping) else {}
            if not bool(attributes.get("semantic_verified")):
                continue
            semantic_type = str(attributes.get("semantic_type") or "").strip().lower()
            label = str(entity.get("label") or "").casefold()
            try:
                confidence = float(entity.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                confidence = 0.0
            if expected_type and semantic_type != expected_type:
                continue
            if expected_label and expected_label not in label:
                continue
            if confidence < minimum:
                continue
            return True
        return False

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
            # WorldStateStore 已用 max_removals 限制此列表；传输层不要再使用更小
            # 的切片，否则原本原子的确认结果将无法对账。
            "removed_entity_ids": list(changes.get("removed_entity_ids") or ()),
            "removed_entity_count": int(changes.get("removed_entity_count", 0) or 0),
        }

    def plan(self, goal: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.cognition.plan(goal)

    def cognition_feedback(self, feedback: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.cognition.feedback(feedback)


__all__ = ["BackendService"]
