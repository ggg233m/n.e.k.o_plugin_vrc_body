"""AnyaDance 身体插件的配置解析与校验。"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
from typing import Any, Mapping


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any, default: float, *, minimum: float, maximum: float, name: str) -> float:
    if value is None:
        # 某些边界由其他配置段派生，因此内置默认值可能落在有效范围之外。
        return min(maximum, max(minimum, default))
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _axis_box_ratio(value: Any, *, legacy: float | None, default: float, name: str) -> float:
    """解析单轴最小框比例。

    优先级：显式的每轴键 > 已设置的共用 ``min_box_ratio`` > 内置默认值。
    只写了 ``min_box_ratio`` 的旧配置因此保持原来的两轴同值行为，不会因为
    默认值变成非对称而被静默改掉判定。
    """
    if value is not None:
        return _finite_float(value, default, minimum=0.0, maximum=0.5, name=name)
    if legacy is not None:
        return legacy
    return default


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int, name: str) -> int:
    if value is None:
        return min(maximum, max(minimum, default))
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    parsed = int(numeric)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _boolean(value: Any, default: bool, *, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class BodyProfile:
    height_m: float = 1.50
    shoulder_width_m: float = 0.36
    shoulder_drop_m: float = 0.18
    arm_length_m: float = 0.58


@dataclass(frozen=True)
class SafetyConfig:
    max_position_abs_m: float = 3.0
    max_y_m: float = 2.0
    max_linear_speed_mps: float = 2.0
    max_angular_speed_dps: float = 360.0
    max_action_duration_ms: int = 5000


@dataclass(frozen=True)
class BehaviorConfig:
    default_crossfade_ms: int = 400
    protect_full_body_motion: bool = True
    prefer_vmd_expressions: bool = True
    transition_history_size: int = 16


@dataclass(frozen=True)
class VmcIdleConfig:
    enabled: bool = True
    listen_host: str = "127.0.0.1"
    listen_port: int = 39539
    allowed_sender: str = "127.0.0.1"
    stale_after_ms: int = 500
    manage_host_output: bool = True
    host_api_url: str = "http://127.0.0.1:48911"
    host_api_timeout_seconds: float = 3.0
    host_output_host: str = "127.0.0.1"
    host_send_rate_hz: int = 60


@dataclass(frozen=True)
class VrchatOscConfig:
    enabled: bool = True
    send_host: str = "127.0.0.1"
    send_port: int = 9000
    listen_host: str = "127.0.0.1"
    listen_port: int = 9001
    allowed_sender: str = "127.0.0.1"
    input_pulse_ms: int = 100
    parameter_cache_size: int = 256
    awareness_parameters: tuple[str, ...] = (
        "NEKO_Action",
        "NEKO_ActionActive",
        "NEKO_ActionPhase",
        "NEKO_Holding",
        # VRChat 内置 Avatar 参数。它们一直在被缓存，只是过去被这张白名单挡在
        # awareness 之外——于是「我是不是卡墙了」在整个仓库里没有任何数据源。
        # 名称已实机验证（2026-08-23）；换一个没配这些参数的 avatar 时
        # motion_feedback 返回 available=false，不会退化成「速度为零」。
        "VelocityX",
        "VelocityY",
        "VelocityZ",
        "AngularY",
        "Upright",
        "Grounded",
    )


@dataclass(frozen=True)
class DriverLogConfig:
    enabled: bool = True
    multicast_group: str = "239.255.39.71"
    listen_port: int = 39571
    interface_host: str = "127.0.0.1"
    stale_after_ms: int = 3000
    history_size: int = 64


@dataclass(frozen=True)
class ControllerInputConfig:
    """虚拟 AnyaDance 控制器输入的路由与安全限制。"""

    primary: str = "anyadance"
    rate_hz: int = 120
    osc_fallback: bool = True
    max_hold_ms: int = 10000
    emergency_release_ms: int = 50


@dataclass(frozen=True)
class AutonomyConfig:
    """会话授权默认值；绝不隐式启用。"""

    manual_arm: bool = True
    session_ttl_minutes: int = 30


@dataclass(frozen=True)
class WorldMemoryConfig:
    """世界存储只允许持久化可复用的世界事实。"""

    persist_world: bool = True
    persist_players: bool = False


@dataclass(frozen=True)
class VisionConfig:
    """模型无关的感知 worker 配置；具体 detector 由后端注入。"""

    enabled: bool = False
    source: str = "none"
    capture: str = "desktop_mirror"
    local_backend: str = "openvino"
    # 本地检测器必须显式部署：缺少模型时保持感知不可用，不发布猜测实体。
    # ``model_path`` 可以指向 OpenVINO IR XML、ONNX 文件或包含其中之一的目录。
    model_path: str | None = None
    labels_path: str | None = None
    device: str = "AUTO"
    # ONNX 模型的可选 NVIDIA 路径。auto 在 OpenVINO NPU/GPU 之后探测，prefer
    # 把 CUDA 放在所有 OpenVINO 设备之前，disabled 完全跳过。依赖保持可选。
    onnxruntime_cuda: str = "auto"
    onnxruntime_cuda_device_id: int = 0
    fallback_backend: str = "none"
    confidence_threshold: float = 0.35
    input_width: int = 640
    input_height: int = 640
    horizontal_fov_deg: float = 90.0
    max_detections: int = 64
    # 检测框宽/高占画面的最小比例，用于滤掉几十像素级别的高分假阳性。
    # ``min_box_ratio`` 是两轴的共同默认值；站立的人是高而窄的，单一阈值下
    # 总是宽度先卡，所以宽阈值默认更松，让高度成为主判据。0 表示关闭对应轴。
    min_box_ratio: float = 0.02
    min_box_width_ratio: float | None = 0.008
    min_box_height_ratio: float | None = 0.02
    semantic_backend: str = "openai_compatible"
    semantic_max_per_minute: int = 30
    # 给 agent 看的单槽帧缓存。这条路径与 world_state 完全无关：帧只喂理解，
    # 不产生实体也不产生事件。编码按间隔做一次，之后所有拉取都命中缓存。
    frame_cache_interval_s: float = 1.0
    frame_max_width: int = 960
    frame_jpeg_quality: int = 70
    # agent 主动拉图的每分钟上限。滑动窗口，0 表示禁止拉图（不是不限量）。
    # 主动唤醒配的图不走这个预算，它自己有 12 s 的最小间隔。
    frame_max_per_minute: int = 10
    # -1 表示自动探测。优先使用物理监视器而不是 MSS 虚拟桌面，DXcam 会探测
    # 所有可见适配器/输出。
    monitor_index: int = -1
    dxcam_device_idx: int = -1
    dxcam_output_idx: int = -1
    dxcam_backend: str = "auto"
    interval_ms: int = 100
    queue_size: int = 1
    # 检测器的 CPU 上限。``detector_threads`` 同时拧两个池子：ONNX Runtime 自己的
    # ``SessionOptions``，以及 numpy/BLAS 的进程级 OpenMP 池（见
    # ``backend.local_perception.cap_openmp_threads``）。实测后者才是大头——仅采集
    # 路径（检测器换成空实现、零次推理）就空转自旋掉 7.23 核，收口后 0.11 核，而
    # 吞吐一模一样。``detector_interval_ms`` 管的是另一个轴：一秒里推几次。
    # 整条流水线实测 7.04 核 → 1.22 核，单次延迟 469ms → 328ms（少了抢核的线程）。
    # ``detector_threads`` 取 0 表示两个池子都不设上限，只留给基准测量。
    # ``detector_interval_ms`` 是 CPU/未知设备的安全间隔；实际解析到 OpenVINO
    # GPU/NPU 或 ORT CUDA 时改用 ``detector_accelerator_interval_ms``。两者取 0
    # 都表示对应设备不限速；它们只限推理，不限给 agent 看的帧缓存。
    detector_threads: int = 2
    detector_interval_ms: int = 500
    detector_accelerator_interval_ms: int = 500
    lifecycle_watermark_limit: int = 4096
    # 按标题定位采集窗口，非空时将窗口屏幕坐标作为采集区域传给帧源。
    # 仅限 Windows；其他平台静默忽略。窗口未找到时回落到全屏采集。
    window_title: str = ""
    # 重新解析窗口矩形的间隔。窗口被拖动、改分辨率或全屏切换后，启动时解析的
    # 那份坐标就过期了，采集会一直抓错位置。0 表示只在启动时解析一次。
    window_track_interval_ms: int = 5000


@dataclass(frozen=True)
class PluginConfig:
    host: str = "127.0.0.1"
    port: int = 39570
    rate_hz: int = 60
    default_duration_ms: int = 600
    max_queue_size: int = 8
    clip_directory: str = "motions"
    clip_max_file_bytes: int = 64 * 1024 * 1024
    clip_max_frames: int = 18000
    clip_max_duration_seconds: float = 300.0
    behavior: BehaviorConfig = BehaviorConfig()
    vmc_idle: VmcIdleConfig = VmcIdleConfig()
    vrchat_osc: VrchatOscConfig = VrchatOscConfig()
    driver_log: DriverLogConfig = DriverLogConfig()
    input: ControllerInputConfig = ControllerInputConfig()
    autonomy: AutonomyConfig = AutonomyConfig()
    world_memory: WorldMemoryConfig = WorldMemoryConfig()
    vision: VisionConfig = VisionConfig()
    profile: BodyProfile = BodyProfile()
    safety: SafetyConfig = SafetyConfig()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PluginConfig":
        root = data or {}
        anyadance = _section(root, "anyadance")
        profile = _section(root, "body_profile")
        motion = _section(root, "motion")
        clips = _section(root, "clips")
        behavior = _section(root, "behavior")
        vmc_idle = _section(root, "vmc_idle")
        vrchat_osc = _section(root, "vrchat_osc")
        driver_log = _section(root, "driver_log")
        input_config = _section(root, "input")
        autonomy = _section(root, "autonomy")
        world_memory = _section(root, "world_memory")
        vision = _section(root, "vision")
        safety = _section(root, "safety")

        host = str(anyadance.get("host", "127.0.0.1")).strip()
        if not host:
            raise ValueError("anyadance.host must not be empty")

        body_profile = BodyProfile(
            height_m=_finite_float(profile.get("height_m"), 1.50, minimum=0.8, maximum=2.0, name="body_profile.height_m"),
            shoulder_width_m=_finite_float(profile.get("shoulder_width_m"), 0.36, minimum=0.15, maximum=0.8, name="body_profile.shoulder_width_m"),
            shoulder_drop_m=_finite_float(profile.get("shoulder_drop_m"), 0.18, minimum=0.05, maximum=0.5, name="body_profile.shoulder_drop_m"),
            arm_length_m=_finite_float(profile.get("arm_length_m"), 0.58, minimum=0.25, maximum=1.0, name="body_profile.arm_length_m"),
        )
        safety_config = SafetyConfig(
            max_position_abs_m=_finite_float(safety.get("max_position_abs_m"), 3.0, minimum=0.5, maximum=30.0, name="safety.max_position_abs_m"),
            max_y_m=_finite_float(safety.get("max_y_m"), 2.0, minimum=1.0, maximum=25.0, name="safety.max_y_m"),
            max_linear_speed_mps=_finite_float(safety.get("max_linear_speed_mps"), 2.0, minimum=0.1, maximum=10.0, name="safety.max_linear_speed_mps"),
            max_angular_speed_dps=_finite_float(safety.get("max_angular_speed_dps"), 360.0, minimum=30.0, maximum=1440.0, name="safety.max_angular_speed_dps"),
            max_action_duration_ms=_bounded_int(safety.get("max_action_duration_ms"), 5000, minimum=100, maximum=30000, name="safety.max_action_duration_ms"),
        )
        if safety_config.max_y_m > safety_config.max_position_abs_m:
            # 每个轴都会检查位置边界；如果 Y 轴上限高于此值，就会拒绝 .nya
            # 加载器已经限制到该边界的帧。
            raise ValueError(
                "safety.max_y_m must not exceed safety.max_position_abs_m"
            )
        clip_directory = str(clips.get("directory", "motions")).strip()
        if not clip_directory or any(char in clip_directory for char in ("/", "\\", ":")) or clip_directory in {".", ".."}:
            raise ValueError("clips.directory must be a single relative directory name")

        def osc_host(key: str, default: str) -> str:
            value = str(vrchat_osc.get(key, default)).strip()
            if not value or "\x00" in value:
                raise ValueError(f"vrchat_osc.{key} must not be empty")
            return value

        raw_awareness = vrchat_osc.get("awareness_parameters", VrchatOscConfig.awareness_parameters)
        if not isinstance(raw_awareness, (list, tuple)) or len(raw_awareness) > 32:
            raise ValueError("vrchat_osc.awareness_parameters must be an array with at most 32 names")
        awareness_parameters: list[str] = []
        for raw_name in raw_awareness:
            name = str(raw_name).strip()
            if not name or len(name) > 128 or "/" in name or "\x00" in name:
                raise ValueError("vrchat_osc.awareness_parameters contains an invalid parameter name")
            if name not in awareness_parameters:
                awareness_parameters.append(name)
        osc_config = VrchatOscConfig(
            enabled=_boolean(vrchat_osc.get("enabled"), True, name="vrchat_osc.enabled"),
            send_host=osc_host("send_host", "127.0.0.1"),
            send_port=_bounded_int(vrchat_osc.get("send_port"), 9000, minimum=1, maximum=65535, name="vrchat_osc.send_port"),
            listen_host=osc_host("listen_host", "127.0.0.1"),
            listen_port=_bounded_int(vrchat_osc.get("listen_port"), 9001, minimum=1, maximum=65535, name="vrchat_osc.listen_port"),
            allowed_sender=osc_host("allowed_sender", "127.0.0.1"),
            input_pulse_ms=_bounded_int(vrchat_osc.get("input_pulse_ms"), 100, minimum=20, maximum=1000, name="vrchat_osc.input_pulse_ms"),
            parameter_cache_size=_bounded_int(vrchat_osc.get("parameter_cache_size"), 256, minimum=16, maximum=2048, name="vrchat_osc.parameter_cache_size"),
            awareness_parameters=tuple(awareness_parameters),
        )

        multicast_group = str(driver_log.get("multicast_group", "239.255.39.71")).strip()
        try:
            parsed_group = ipaddress.ip_address(multicast_group)
        except ValueError as exc:
            raise ValueError("driver_log.multicast_group must be an IPv4 address") from exc
        if parsed_group.version != 4 or not parsed_group.is_multicast:
            raise ValueError("driver_log.multicast_group must be an IPv4 multicast address")
        interface_host = str(driver_log.get("interface_host", "127.0.0.1")).strip()
        if not interface_host or "\x00" in interface_host:
            raise ValueError("driver_log.interface_host must not be empty")
        driver_log_config = DriverLogConfig(
            enabled=_boolean(driver_log.get("enabled"), True, name="driver_log.enabled"),
            multicast_group=multicast_group,
            listen_port=_bounded_int(driver_log.get("listen_port"), 39571, minimum=1, maximum=65535, name="driver_log.listen_port"),
            interface_host=interface_host,
            stale_after_ms=_bounded_int(driver_log.get("stale_after_ms"), 3000, minimum=100, maximum=60000, name="driver_log.stale_after_ms"),
            history_size=_bounded_int(driver_log.get("history_size"), 64, minimum=8, maximum=512, name="driver_log.history_size"),
        )
        primary_input = str(input_config.get("primary", "anyadance")).strip().lower() or "anyadance"
        if primary_input not in {"anyadance", "osc"}:
            raise ValueError("input.primary must be anyadance or osc")
        controller_input_config = ControllerInputConfig(
            primary=primary_input,
            rate_hz=_bounded_int(
                input_config.get("rate_hz"),
                _bounded_int(anyadance.get("rate_hz"), 60, minimum=10, maximum=120, name="anyadance.rate_hz"),
                minimum=10,
                maximum=120,
                name="input.rate_hz",
            ),
            osc_fallback=_boolean(input_config.get("osc_fallback"), True, name="input.osc_fallback"),
            max_hold_ms=_bounded_int(
                input_config.get("max_hold_ms"),
                10000,
                minimum=100,
                maximum=30000,
                name="input.max_hold_ms",
            ),
            emergency_release_ms=_bounded_int(
                input_config.get("emergency_release_ms"),
                50,
                minimum=20,
                maximum=500,
                name="input.emergency_release_ms",
            ),
        )
        autonomy_config = AutonomyConfig(
            manual_arm=_boolean(autonomy.get("manual_arm"), True, name="autonomy.manual_arm"),
            session_ttl_minutes=_bounded_int(
                autonomy.get("session_ttl_minutes"),
                30,
                minimum=1,
                maximum=240,
                name="autonomy.session_ttl_minutes",
            ),
        )
        world_memory_config = WorldMemoryConfig(
            persist_world=_boolean(world_memory.get("persist_world"), True, name="world_memory.persist_world"),
            persist_players=_boolean(world_memory.get("persist_players"), False, name="world_memory.persist_players"),
        )
        vision_source = str(vision.get("source", "none")).strip().lower() or "none"
        if vision_source not in {"none", "mss", "dxcam", "desktop_mirror", "external"}:
            raise ValueError("vision.source must be none, mss, dxcam, desktop_mirror, or external")
        capture_backend = str(vision.get("capture", "desktop_mirror")).strip().lower() or "desktop_mirror"
        if capture_backend not in {"desktop_mirror", "mss", "dxcam", "external"}:
            raise ValueError("vision.capture must be desktop_mirror, mss, dxcam, or external")
        local_backend = str(vision.get("local_backend", "openvino")).strip().lower() or "openvino"
        if local_backend not in {"openvino", "none", "external"}:
            raise ValueError("vision.local_backend must be openvino, none, or external")
        def optional_path(name: str) -> str | None:
            raw = vision.get(name)
            if raw is None:
                return None
            value = str(raw).strip()
            if not value:
                return None
            if len(value) > 1024 or "\x00" in value:
                raise ValueError(f"vision.{name} is too long or contains NUL")
            return value
        model_path = optional_path("model_path")
        labels_path = optional_path("labels_path")
        device = str(vision.get("device", "AUTO")).strip() or "AUTO"
        if len(device) > 32 or "\x00" in device:
            raise ValueError("vision.device must be a short non-empty string")
        onnxruntime_cuda = str(
            vision.get("onnxruntime_cuda", "auto")
        ).strip().lower() or "auto"
        if onnxruntime_cuda not in {"auto", "prefer", "disabled"}:
            raise ValueError("vision.onnxruntime_cuda must be auto, prefer, or disabled")
        fallback_backend = str(vision.get("fallback_backend", "none")).strip().lower() or "none"
        if fallback_backend not in {"none", "opencv_hog"}:
            raise ValueError("vision.fallback_backend must be none or opencv_hog")
        semantic_backend = str(vision.get("semantic_backend", "openai_compatible")).strip().lower() or "openai_compatible"
        if semantic_backend not in {"openai_compatible", "none", "external"}:
            raise ValueError("vision.semantic_backend must be openai_compatible, none, or external")
        dxcam_backend = str(vision.get("dxcam_backend", "auto")).strip().lower() or "auto"
        if dxcam_backend not in {"auto", "dxgi", "winrt"}:
            raise ValueError("vision.dxcam_backend must be auto, dxgi, or winrt")
        shared_box_ratio = _finite_float(
            vision.get("min_box_ratio"),
            0.02,
            minimum=0.0,
            maximum=0.5,
            name="vision.min_box_ratio",
        )
        detector_interval_ms = _bounded_int(
            vision.get("detector_interval_ms"),
            500,
            minimum=0,
            maximum=10000,
            name="vision.detector_interval_ms",
        )
        vision_config = VisionConfig(
            enabled=_boolean(vision.get("enabled"), False, name="vision.enabled"),
            source=vision_source,
            capture=capture_backend,
            local_backend=local_backend,
            model_path=model_path,
            labels_path=labels_path,
            device=device,
            onnxruntime_cuda=onnxruntime_cuda,
            onnxruntime_cuda_device_id=_bounded_int(
                vision.get("onnxruntime_cuda_device_id"),
                0,
                minimum=0,
                maximum=31,
                name="vision.onnxruntime_cuda_device_id",
            ),
            fallback_backend=fallback_backend,
            confidence_threshold=_finite_float(
                vision.get("confidence_threshold"),
                0.35,
                minimum=0.0,
                maximum=1.0,
                name="vision.confidence_threshold",
            ),
            input_width=_bounded_int(
                vision.get("input_width"),
                640,
                minimum=32,
                maximum=4096,
                name="vision.input_width",
            ),
            input_height=_bounded_int(
                vision.get("input_height"),
                640,
                minimum=32,
                maximum=4096,
                name="vision.input_height",
            ),
            horizontal_fov_deg=_finite_float(
                vision.get("horizontal_fov_deg"),
                90.0,
                minimum=1.0,
                maximum=180.0,
                name="vision.horizontal_fov_deg",
            ),
            max_detections=_bounded_int(
                vision.get("max_detections"),
                64,
                minimum=1,
                maximum=512,
                name="vision.max_detections",
            ),
            min_box_ratio=shared_box_ratio,
            min_box_width_ratio=_axis_box_ratio(
                vision.get("min_box_width_ratio"),
                legacy=shared_box_ratio if "min_box_ratio" in vision else None,
                default=0.008,
                name="vision.min_box_width_ratio",
            ),
            min_box_height_ratio=_axis_box_ratio(
                vision.get("min_box_height_ratio"),
                legacy=shared_box_ratio if "min_box_ratio" in vision else None,
                default=0.02,
                name="vision.min_box_height_ratio",
            ),
            semantic_backend=semantic_backend,
            semantic_max_per_minute=_bounded_int(
                vision.get("semantic_max_per_minute"),
                30,
                minimum=1,
                maximum=30,
                name="vision.semantic_max_per_minute",
            ),
            frame_cache_interval_s=_finite_float(
                vision.get("frame_cache_interval_s"),
                1.0,
                minimum=0.0,
                maximum=30.0,
                name="vision.frame_cache_interval_s",
            ),
            frame_max_width=_bounded_int(
                vision.get("frame_max_width"),
                960,
                minimum=0,
                maximum=3840,
                name="vision.frame_max_width",
            ),
            frame_jpeg_quality=_bounded_int(
                vision.get("frame_jpeg_quality"),
                70,
                minimum=30,
                maximum=95,
                name="vision.frame_jpeg_quality",
            ),
            frame_max_per_minute=_bounded_int(
                vision.get("frame_max_per_minute"),
                10,
                minimum=0,
                maximum=60,
                name="vision.frame_max_per_minute",
            ),
            monitor_index=_bounded_int(
                vision.get("monitor_index"),
                -1,
                minimum=-1,
                maximum=32,
                name="vision.monitor_index",
            ),
            dxcam_device_idx=_bounded_int(
                vision.get("dxcam_device_idx"),
                -1,
                minimum=-1,
                maximum=32,
                name="vision.dxcam_device_idx",
            ),
            dxcam_output_idx=_bounded_int(
                vision.get("dxcam_output_idx"),
                -1,
                minimum=-1,
                maximum=32,
                name="vision.dxcam_output_idx",
            ),
            dxcam_backend=dxcam_backend,
            interval_ms=_bounded_int(
                vision.get("interval_ms"),
                100,
                minimum=10,
                maximum=2000,
                name="vision.interval_ms",
            ),
            queue_size=_bounded_int(
                vision.get("queue_size"),
                1,
                minimum=1,
                maximum=4,
                name="vision.queue_size",
            ),
            detector_threads=_bounded_int(
                vision.get("detector_threads"),
                2,
                minimum=0,
                maximum=32,
                name="vision.detector_threads",
            ),
            detector_interval_ms=detector_interval_ms,
            detector_accelerator_interval_ms=_bounded_int(
                vision.get("detector_accelerator_interval_ms"),
                detector_interval_ms,
                minimum=0,
                maximum=10000,
                name="vision.detector_accelerator_interval_ms",
            ),
            lifecycle_watermark_limit=_bounded_int(
                vision.get("lifecycle_watermark_limit"),
                4096,
                minimum=256,
                maximum=65536,
                name="vision.lifecycle_watermark_limit",
            ),
            window_title=str(vision.get("window_title", "")).strip()[:256],
            window_track_interval_ms=_bounded_int(
                vision.get("window_track_interval_ms"),
                5000,
                minimum=0,
                maximum=60000,
                name="vision.window_track_interval_ms",
            ),
        )
        behavior_config = BehaviorConfig(
            default_crossfade_ms=_bounded_int(
                behavior.get("default_crossfade_ms"),
                400,
                minimum=0,
                maximum=5000,
                name="behavior.default_crossfade_ms",
            ),
            protect_full_body_motion=_boolean(
                behavior.get("protect_full_body_motion"),
                True,
                name="behavior.protect_full_body_motion",
            ),
            prefer_vmd_expressions=_boolean(
                behavior.get("prefer_vmd_expressions"),
                True,
                name="behavior.prefer_vmd_expressions",
            ),
            transition_history_size=_bounded_int(
                behavior.get("transition_history_size"),
                16,
                minimum=4,
                maximum=64,
                name="behavior.transition_history_size",
            ),
        )
        vmc_idle_config = VmcIdleConfig(
            enabled=_boolean(vmc_idle.get("enabled"), True, name="vmc_idle.enabled"),
            listen_host=str(vmc_idle.get("listen_host", "127.0.0.1")).strip(),
            listen_port=_bounded_int(
                vmc_idle.get("listen_port"),
                39539,
                minimum=1,
                maximum=65535,
                name="vmc_idle.listen_port",
            ),
            allowed_sender=str(vmc_idle.get("allowed_sender", "127.0.0.1")).strip(),
            stale_after_ms=_bounded_int(
                vmc_idle.get("stale_after_ms"),
                500,
                minimum=100,
                maximum=5000,
                name="vmc_idle.stale_after_ms",
            ),
            manage_host_output=_boolean(
                vmc_idle.get("manage_host_output"),
                True,
                name="vmc_idle.manage_host_output",
            ),
            host_api_url=str(
                vmc_idle.get("host_api_url", "http://127.0.0.1:48911")
            ).strip(),
            host_api_timeout_seconds=_finite_float(
                vmc_idle.get("host_api_timeout_seconds"),
                3.0,
                minimum=0.2,
                maximum=30.0,
                name="vmc_idle.host_api_timeout_seconds",
            ),
            host_output_host=str(
                vmc_idle.get("host_output_host", "127.0.0.1")
            ).strip(),
            host_send_rate_hz=_bounded_int(
                vmc_idle.get("host_send_rate_hz"),
                60,
                minimum=1,
                maximum=120,
                name="vmc_idle.host_send_rate_hz",
            ),
        )
        if not vmc_idle_config.listen_host or "\x00" in vmc_idle_config.listen_host:
            raise ValueError("vmc_idle.listen_host must not be empty")
        if not vmc_idle_config.allowed_sender or "\x00" in vmc_idle_config.allowed_sender:
            raise ValueError("vmc_idle.allowed_sender must not be empty")
        if not vmc_idle_config.host_api_url or "\x00" in vmc_idle_config.host_api_url:
            raise ValueError("vmc_idle.host_api_url must not be empty")
        if not vmc_idle_config.host_output_host or "\x00" in vmc_idle_config.host_output_host:
            raise ValueError("vmc_idle.host_output_host must not be empty")

        return cls(
            host=host,
            port=_bounded_int(anyadance.get("port"), 39570, minimum=1, maximum=65535, name="anyadance.port"),
            rate_hz=controller_input_config.rate_hz,
            default_duration_ms=_bounded_int(motion.get("default_duration_ms"), 600, minimum=100, maximum=safety_config.max_action_duration_ms, name="motion.default_duration_ms"),
            max_queue_size=_bounded_int(motion.get("max_queue_size"), 8, minimum=1, maximum=64, name="motion.max_queue_size"),
            clip_directory=clip_directory,
            clip_max_file_bytes=_bounded_int(clips.get("max_file_bytes"), 64 * 1024 * 1024, minimum=1024, maximum=256 * 1024 * 1024, name="clips.max_file_bytes"),
            clip_max_frames=_bounded_int(clips.get("max_frames"), 18000, minimum=1, maximum=100000, name="clips.max_frames"),
            clip_max_duration_seconds=_finite_float(clips.get("max_duration_seconds"), 300.0, minimum=0.1, maximum=3600.0, name="clips.max_duration_seconds"),
            behavior=behavior_config,
            vmc_idle=vmc_idle_config,
            vrchat_osc=osc_config,
            driver_log=driver_log_config,
            input=controller_input_config,
            autonomy=autonomy_config,
            world_memory=world_memory_config,
            vision=vision_config,
            profile=body_profile,
            safety=safety_config,
        )
