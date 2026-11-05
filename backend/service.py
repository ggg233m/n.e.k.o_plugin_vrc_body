"""由独立 AnyaDance 后端进程持有的运行时。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from collections import deque
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import sys as _sys_for_policy
_RESEARCH_TOOLS = str(Path(__file__).resolve().parents[1] / "research" / "tools")
if _RESEARCH_TOOLS not in _sys_for_policy.path:
    _sys_for_policy.path.insert(0, _RESEARCH_TOOLS)
from policy_body import body_feedback  # noqa: E402  （纯函数：OSC 静默时的 body 策略）
from policy_histframe import HISTORY_FRAME_S  # noqa: E402  （视觉历史帧间隔唯一真源）

from .adapters import (
    ActionTimeline,
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
    preload_inference_runtime,
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
    WgcWindowFrameSource,
    WindowTrackedFrameSource,
)
from .world_state import WorldStateStore
from .world_model import WorldModel
from .time_alignment import TimeAlignmentBuffer
from .nav_memory import NavMemoryStore, config_from_plugin as nav_memory_config
from .nav_online import OnlineNavConfig, OnlineNavigator
from .nav_prior import prior_meta


_VMC_CALIBRATION_RETRY_SECONDS = 5.0
# 宿主输出 rawRestPose 帧的时长，与 host_vmc 的 duration_sec 默认值一致。握手返回即
# 受理，但基准要等合格的 T Pose 帧到达才锁上（vmc_idle 要过解剖学校验），本线程得等过
# 这个窗口才能判断基准有没有真的锁上；否则 needs_recalibration() 在帧到达前恒为真，
# 会退化成紧循环。
_VMC_T_POSE_WINDOW_SECONDS = 2.0
# 基准已就绪时的空转间隔。宿主暂停或 VMC 输出重启会清空基准，本线程要在那之后
# 主动补一次 T Pose，所以它必须比首次校准活得久。
_VMC_CALIBRATION_IDLE_POLL_SECONDS = 1.0
_WORLD_GATE_BYPASS_ACTIONS = frozenset({"stop", "disable", "reset", "cancel"})
# 手掌朝向由 motion.palm_rotation 解析，非法值在调度线程上抛 ValueError。
# 那时 submit() 早已返回 accepted=true，所以要在入队前挡下来。
_PALM_ORIENTATIONS = frozenset({"neutral", "forward", "down", "inward"})
_PALM_ACTIONS = frozenset({"arm_pose", "move_hand"})
_OSC_AXIS_MIN = -1.0
_OSC_AXIS_MAX = 1.0
_OSC_DURATION_MIN_MS = 100
_OSC_DURATION_MAX_MS = 10000
# VLA 头 A 回喂的本体速度：OSC VelocityX/Z 是变化驱动参数，静默只在短时间内等于
# 「速度没变」；超过这个时长按 0，避免把一分钟前的最后速度当当前状态喂给模型
# （与 DeadReckoner 的 osc_zoh_max_s 同一口径）。
_POLICY_BODY_ZOH_S = 2.5
# navmesh 左目关键帧实测 7.9–10 Hz；超过这个年龄就当没有帧（会话没开或双目断了）。
_POLICY_FRAME_STALE_S = 0.5


def _env_flag(name: str, default: bool = False) -> bool:
    """环境变量开关：1/true/yes/on 为真；未设置或空串取 default。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _emit_event(event: str, **fields: Any) -> None:
    """拆卸/退出路径的结构化事件日志（stderr，JSON 一行）。

    为什么不走 self.logger：standalone 模式（backend/process.py）根本不注入 logger，
    那里 self.logger 恒为 None —— 2880 行那条注释记的正是同一类事故（警告彻底消失）。
    而「关机为什么慢、卡在哪一步」只有这条路径能说清楚，不能依赖一个可能为 None 的
    对象，也不能依赖还没配置的 logging。
    """
    try:
        print(
            json.dumps({"event": event, **fields}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
    except (OSError, ValueError, TypeError):
        pass

# OSC 前进轴满轴对应的实际移动速度（m/s）。方案 v3 §9.1 的速度平台实测：
# 前进/后退众数 4.0-4.1、p99 4.00，所以满轴推杆就是 4.0 m/s。
# ⚠️ 这是**世界米/秒**，而 nav_follow 的 forward 是 0..1 的轴值 —— 两者差一个系数，
# 混用会把「策略比跟随器慢 16 倍」看成「慢 4 倍」（剖析 §15.17 就踩过这个）。
_OSC_FULL_AXIS_MPS = 4.0

_OSC_HOLD_MIN_MS = 20
_OSC_HOLD_MAX_MS = 1000
# 满舵转向速度。把 (horizontal, duration_ms) 这套摇杆语义换算成角度：
# horizontal=1.0 保持 500ms 就是 90°。调度器还会按 safety.max_angular_speed_dps
# 再限一次速，这个常数只决定「一条指令要转多少」。
TURN_SPEED_DPS = 180.0
# 叠框编号只供人和多模态模型在单帧内选择。映射短期留在内存，让主 LLM
# 思考数秒后仍能提交 T1/T2；真正执行前还会重新确认稳定 ID 当前可见。
_TARGET_REF_CACHE_TTL_S = 60.0
_TARGET_REF_CACHE_LIMIT = 16
# 这只是把主 LLM 的离散空间选择映射为一条短转向；它不是人物选择器，不能
# 携带或推导 target_id。
_WANDER_DIRECTION_TURN_DEG = {
    "left": 25.0,
    "forward": 0.0,
    "right": -25.0,
}
# 本地检测器只产出人形轨迹，world.entities 里永远不会出现海报、屏幕、家具这类
# 静态物体。主 LLM 可以在画面里看见它们并提交分类，但 SemanticCandidateCache
# 的分类只能作为属性附着到本地检测已有的框上（见 vision.enrich_observation），
# 造不出新实体——所以对这类目标建语义任务必然以 semantic_target_not_found 收场，
# 白烧一张图还让模型误以为在等确认。这里提前识别出来，改走方位角闲逛。
_HUMANOID_SELECTOR_TYPES = frozenset(
    {"npc", "player", "avatar", "person", "humanoid"}
)


def _selector_is_locally_detectable(selector: Mapping[str, Any] | None) -> bool:
    """selector 描述的目标本地检测器能否产出实体框。

    只有人形类别能被确认。``object`` 和空类别都不行：前者明确指向非人物体，
    后者只有标签没有类别，无法保证落在人形轨迹上。
    """
    if not isinstance(selector, Mapping):
        return False
    semantic_type = str(selector.get("semantic_type") or "").strip().lower()
    return semantic_type in _HUMANOID_SELECTOR_TYPES



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
        state_dir: str | Path | None = None,
        logger: Any = None,
        dry_run: bool = False,
        vision_source: FrameSource | None = None,
        vision_detector: FrameDetector | None = None,
        vision_source_factory: Callable[[], FrameSource] | None = None,
    ) -> None:
        self.config = PluginConfig.from_mapping(config_data)
        self.config_dir = Path(config_dir)
        # 运行时写盘一律走 state_dir，绝不落回 config_dir。宿主把安装目录当成包
        # 内容的指纹：start_plugin 先比对 plugin.meta.json 里的 source_files，一
        # 致就直接用打包期算好的元数据，不一致就回落到起一个 worker 子进程去真
        # import 一遍插件。我们往安装目录写下一个 world_memory.json，文件集就对
        # 不上了——于是装完的第一次启动走快路径正常，插件自己跑一轮把目录写脏之
        # 后，后续每次启动都改走 worker，而那条路在本机必崩（见下）。
        #
        # worker 崩的是宿主的锅，但我们躲得开：父进程按 UTF-8 写请求，子进程的
        # sys.stdin 在 cp936 机器上是 gbk/surrogateescape，于是 plugin.toml 里
        # [plugin.entries] 的中文被解成孤立代理对；worker 回写结果时那句
        # .encode("utf-8") 正好落在它 except BaseException 的外面，UnicodeEncodeError
        # 直接掀掉进程，stderr 又已重定向到 devnull——宿主只看到 "worker exited
        # with code 1"，一路压扁成 "start_plugin failed"。不写脏目录，就永远走
        # 不到那条路径。
        self.state_dir = Path(state_dir) if state_dir is not None else self.config_dir
        self.logger = logger
        self.dry_run = bool(dry_run)
        # 必须在这里，而且必须在建采集源／检测器之前：OpenMP 只在初始化时读一次
        # 环境变量，而本仓的 numpy 全是函数内惰性导入，所以此刻还没人导入过它。
        # 往后挪一行到 _build_configured_vision_source 之后就晚了——dxcam 会先把
        # numpy 拉进来，环境变量随之失效，只剩 ctypes 那条降级路。
        # 收的是 numpy/BLAS 的池，不是 YOLO 的：实测采集路径空转自旋就吃 7.23 核。
        self._openmp = cap_openmp_threads(self.config.vision.detector_threads)
        # 同样必须早于建采集源：WinRT/WGC 会话的 is_border_required 只在
        # duplicator 的 __post_init__ 里读一次环境变量，相机建完再设就没用了。
        # 不设时 WGC 会沿屏幕外沿画一圈黄色捕获指示框（Windows 的采集提示），
        # 而且这条路径本身就比 DXGI 重。dxcam 仅在环境变量显式给值时才去设这个
        # 属性，所以默认值得由我们补上；用户已显式设置时不覆盖，Windows 拒绝
        # 该请求时 dxcam 只记一条警告，不影响采集。
        os.environ.setdefault("DXCAM_WINRT_BORDER_REQUIRED", "0")
        # 也必须早于建采集源，而且理由更硬：dxcam[winrt] 会把
        # winrt.windows.graphics.* 的 C++/WinRT 扩展拉进进程，之后
        # ``import onnxruntime`` 就会卡 4.3 秒再报 DLL 初始化失败(1114)。
        # 顺序是单向的——ORT 先进来则两者共存无碍。详见
        # ``preload_inference_runtime``。
        self._ort_preload = preload_inference_runtime()
        if not self._ort_preload["ok"] and self.logger is not None:
            self.logger.warning(
                "ONNX Runtime 预热失败，检测器将不可用：%s", self._ort_preload["error"]
            )
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
            configured_reid_model = self.config.vision.identity_reid_model_path
            if configured_reid_model and not Path(configured_reid_model).is_absolute():
                configured_reid_model = str(self.config_dir / configured_reid_model)
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
                identity_reid_model_path=configured_reid_model,
                identity_reid_model_similarity=self.config.vision.identity_reid_model_similarity,
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
                    # 显式传：限流窗口与 status().last_call_age_ms 必须和 VisionRuntime 同域。
                    clock=time.perf_counter,
                )
        self.clip_library = ClipLibrary(self.config_dir / self.config.clip_directory, self.config)
        # W1 世界身份子系统。身份会落盘（`<state_dir>/world_identity.json`）：重启后自动沿用
        # **用户显式设过**的值并标 `restored`；系统仍不猜世界（见 world_model 模块 docstring）。
        # 具体启停由 /worldmodel/start|stop 接口驱动，不随后端 service.start() 自动拉起。
        self.world_model = WorldModel(
            enabled=self.config.world_model.enabled,
            persist=self.config.world_model.persist,
            state_dir=self.state_dir,
        )
        self.world_state = WorldStateStore(
            lifecycle_watermark_limit=self.config.vision.lifecycle_watermark_limit,
            # 显式传：store 的 observed_at 要和 vision 侧同域（否则 ingest 的 clamp 会吞掉偏差）。
            clock=time.perf_counter,
            persistence_path=self.state_dir / "world_memory.json",
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
        # VLA 头 A（M0 policy）。默认 None：只有 VRC_POLICY_HEAD=1 才构造。
        self.policy_head = None
        # UI 保存的 [policy] 段优先于旧环境变量；没有该段时保持历史启动方式。
        self._policy_configured = isinstance(config_data.get("policy"), Mapping)
        # 策略头装配失败的原因。为 None 且 policy_head 也是 None 时，说明开关没打开。
        # 由 /policy/state 外显 —— 静默失败查一次要 5 分钟（2026-11-02）。
        self._policy_start_error: str | None = None
        # 当前 VLA 的场次级测试提示。它只改变推理输入，不改变导航器或动作安全门；
        # 空字符串表示回到旧 checkpoint 的原始输入分布。
        self._policy_prompt_text = ""
        self._policy_body_last: tuple[float, float] | None = None
        self._policy_body_at = 0.0
        self.driver_log: DriverLogListener | None = None
        # 动作时间轴（路线历史通道）。默认不启用：config.driver_log.action_timeline_enabled
        # 为 False 时它就是 None，所有接线点退化成 no-op，行为与它出现之前完全一致。
        self.action_timeline: ActionTimeline | None = None
        self.vmc_idle: VmcIdleRelay | None = None
        self.host_vmc: HostVmcController | None = None
        self._vmc_calibration_stop = threading.Event()
        self._vmc_calibration_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._target_ref_cache: dict[int, dict[str, Any]] = {}
        # 主 LLM 只负责一次语义选择；选择完成后由后端自动接续本地导航，避免
        # 让模型再复制稳定 ID、再发一次“走过去”，从而把行为做成遥控器。
        self._pending_semantic_intent: dict[str, Any] | None = None
        self._started = False
        self._last_error: str | None = None
        self._expression_side_count = 0
        self._motion_intent_counts: dict[str, int] = {}
        # turn 来源账本：play-space yaw 的每一次转身都必须能归因。只看
        # turn_commands 总数无法回答"这 282 条是谁发的"——导航器、HTTP
        # set_turn、调试工具共用一个调度器通道，不记账就只剩猜测。
        self._turn_trace_lock = threading.Lock()
        self._turn_source_counts: dict[str, int] = {}
        self._turn_trace: deque[dict[str, Any]] = deque(maxlen=64)
        # HTTP 处理线程在分发请求前写入的客户端地址（ip:port）。turn 账本
        # 取它归因到具体客户端连接；非 HTTP 路径（导航器）读到 None 属预期。
        self._http_client_addr: str | None = None
        self.autonomy = AutonomyRuntime(
            world_provider=lambda: self.vision.snapshot(),
            release_inputs=self._release_all_inputs,
            session_ttl_s=self.config.autonomy.session_ttl_minutes * 60.0,
            # 延迟解析 self.navigator：它在下面才构造，这个回调只会在运行期触发。
            on_world_changed=lambda: self.navigator.reset_direction_memory(
                "world_changed"
            ),
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
        # 光流按 WGC 帧时刻取对齐后的 OSC 速度；只保留有界样本。
        self.time_alignment = TimeAlignmentBuffer()
        self._last_osc_alignment_timestamp: float | None = None
        # navmesh 持久记忆：按世界分区；上次没正常结束的会话在这里标成 interrupted。
        self.navmesh_memory = NavMemoryStore(
            Path(os.getenv("VRC_NAVMESH_MEMORY_DIR") or (self.state_dir / "navmesh_memory")),
            nav_memory_config(self.config.navmesh.memory),
        )
        try:
            self.navmesh_memory.recover()
        except OSError:
            pass
        # 在线增量 navmesh：显式 start 才开 OpenVR 会话，默认不占 SteamVR。
        self.navmesh = OnlineNavigator(
            motion_history=self._navmesh_motion_history,
            vertical_history=self._navmesh_vertical_history,
            param_history=self._navmesh_param_history,
            avatar_snapshot=self._navmesh_avatar_snapshot,
            send_move=lambda forward, ms: self.set_locomotion(forward, 0.0, ms),
            send_turn=self._navigator_send_turn,
            stop_motion=self.stop_movement,
            drive_block_reason=self._navmesh_drive_block,
            # 录制默认不开；开了落到 state_dir（或 VRC_NAVMESH_RECORD_DIR），不写安装目录。
            record_root=Path(os.getenv("VRC_NAVMESH_RECORD_DIR") or (self.state_dir / "navmesh_recordings")),
            cfg=OnlineNavConfig.from_plugin(self.config.navmesh),
            memory=self.navmesh_memory,
            world_identity=self.world_model.identity,
        )
        # policy 头要左目 RGB 才让感知循环多采一路彩色。必须在 navmesh 会话 start 之前
        # 设置（need_rgb 在感知循环入口算一次）。
        self.navmesh.policy_frames = self._policy_enabled()
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
            # 显式传：StateEstimator 拿 observed_at 与 now 比较，必须和 vision 侧同域。
            # 跨域的后果是 min(now, observed_at) 把新鲜的观测钳成 ~0 age，且 sources.stale 乱翻。
            clock=time.perf_counter,
        )

    def _sync_osc_alignment_samples(self) -> None:
        """把 OSC 接收线程产生的新速度样本复制到视觉对齐缓冲。"""
        if self.osc is None:
            return
        try:
            samples = self.osc.motion_history()
        except Exception:
            return
        for sample in samples:
            try:
                timestamp = float(sample.get("timestamp"))
            except (TypeError, ValueError, OverflowError):
                continue
            if (self._last_osc_alignment_timestamp is not None
                    and timestamp <= self._last_osc_alignment_timestamp):
                continue
            # VelocityX/Z 是变化驱动参数，不能用相隔很久的两条轴拼成一个“当前”
            # 速度。两轴没有在同一小窗口内到达时，宁可让当前 WGC 帧得到 unknown。
            try:
                x_at = float(sample.get("velocity_x_timestamp", timestamp))
                z_at = float(sample.get("velocity_z_timestamp", timestamp))
            except (TypeError, ValueError, OverflowError):
                continue
            if abs(x_at - z_at) > self.time_alignment.max_gap_s:
                continue
            if self.time_alignment.add_motion(
                timestamp, sample.get("velocity_x"), sample.get("velocity_z")
            ):
                self._last_osc_alignment_timestamp = timestamp

    def _vision_motion_feedback(self, captured_at: float | None = None) -> Mapping[str, Any]:
        """给光流使用与当前 WGC 帧配对的 OSC 速度；不给时刻时读最新值。"""
        self._sync_osc_alignment_samples()
        if captured_at is None:
            if self.osc is None:
                return {"available": False, "reason": "osc_unavailable"}
            try:
                return self.osc.motion_feedback()
            except Exception:
                return {"available": False, "reason": "osc_feedback_error"}
        result = self.time_alignment.sample_motion(captured_at)
        # 保留可读的 OSC 缺失原因，同时明确这不是静止速度。
        if not result.get("available") and self.osc is None:
            result["reason"] = "osc_unavailable"
        return result

    # ---- 在线增量 navmesh（航位推算 + 镜像双目）----
    def _navmesh_motion_history(self) -> list[dict[str, Any]]:
        osc = self.osc
        return [] if osc is None else osc.motion_history(limit=64)

    def _navmesh_vertical_history(self) -> list[dict[str, Any]]:
        osc = self.osc
        fn = None if osc is None else getattr(osc, "vertical_history", None)
        return [] if fn is None else fn(limit=64)

    def _navmesh_param_history(self) -> list[dict[str, Any]]:
        osc = self.osc
        fn = None if osc is None else getattr(osc, "param_history", None)
        return [] if fn is None else fn(limit=256)

    def _navmesh_avatar_snapshot(self) -> dict[str, Any]:
        osc = self.osc
        fn = None if osc is None else getattr(osc, "avatar_scale_snapshot", None)
        # query=True：缓存没收到眼高时现问 VRChat OSCQuery —— world_scale 要按它换算。
        return {} if fn is None else fn(query=True)

    def _navmesh_drive_block(self) -> str | None:
        """None = 可以驾驶；否则是不能动的原因。与自主目标互斥。"""
        if not self._started or self.osc is None or self.scheduler is None:
            return "backend_not_started"
        autonomy = self.autonomy.snapshot()
        if not autonomy.get("armed"):
            return "autonomy_not_armed"
        if autonomy.get("goal"):
            # 本地导航器正在执行自主目标，两个控制环不能同时发轴。
            return "autonomy_goal_active"
        return None

    def navmesh_status(self, include_grid: bool = False,
                       layer: str = "tristate") -> dict[str, Any]:
        status = self.navmesh.status()
        if include_grid:
            view = self.navmesh.grid_view(layer)
            status["grid_png_base64"] = None if view is None else view.pop("png_base64")
            status["grid_meta"] = view
        return status

    def navmesh_coverage(self) -> dict[str, Any]:
        """覆盖俯视快照：伴生覆盖网格（观测计数/近看质量）+ 洞分类 + 诚实进度指标。只读。"""
        view = self.navmesh.coverage_view()
        return view if view is not None else {"available": False, "reason": "no_map_yet"}

    def navmesh_start(self, record: bool = False, record_images: bool = False) -> dict[str, Any]:
        if not self._started:
            return {**self.navmesh.status(), "ok": False, "reason": "backend_not_started"}
        return {**self.navmesh.start(record=record, record_images=record_images), "ok": True}

    def navmesh_stop(self) -> dict[str, Any]:
        return {**self.navmesh.stop(), "ok": True}

    def navmesh_goto(self, x: Any, y: Any) -> dict[str, Any]:
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "x_y_must_be_numbers"}
        return self.navmesh.goto(fx, fy)

    def navmesh_explore(self) -> dict[str, Any]:
        return self.navmesh.explore()

    def navmesh_cancel(self) -> dict[str, Any]:
        return self.navmesh.cancel()

    # ---- navmesh 记忆管理（列/看/改标签/钉住/删/清理）----
    def navmesh_memory_summary(self, sizes: bool = True) -> dict[str, Any]:
        """``sizes=False``：不扫目录算体积（见 ``NavMemoryStore.list_worlds``）。
        栅格页/主页面只要"有哪些世界、几场会话、有没有先验"，用轻的那档，别把请求超时打爆。"""
        base = {**self.navmesh_memory.summary(sizes=sizes),
                "world_list": self.navmesh_memory.list_worlds(sizes=sizes),
                "current_world": self.world_model.identity()}
        # 每个世界的**世界先验**在不在（只读 prior.json 摘要，不加载 npz）——"没设世界身份"
        # 与"设了但那个世界还没固化先验"是两件事，UI 要把它们分开说。
        base["priors"] = {w["world_id"]: prior_meta(self.navmesh_memory.root / w["world_id"])
                          for w in base["world_list"]}
        return base

    def navmesh_memory_sessions(self, world_id: Any) -> dict[str, Any]:
        return self._memory_call(lambda: {"world_id": world_id,
                                          "sessions": self.navmesh_memory.list_sessions(world_id)})

    def navmesh_memory_session(self, world_id: Any, session_id: Any) -> dict[str, Any]:
        return self._memory_call(lambda: self.navmesh_memory.session(world_id, session_id))

    def navmesh_memory_thumbnail(self, world_id: Any, session_id: Any, k: Any) -> bytes:
        return self.navmesh_memory.thumbnail(world_id, session_id, int(k))

    def navmesh_memory_update(self, world_id: Any, session_id: Any, label: Any = None,
                              pinned: Any = None) -> dict[str, Any]:
        return self._memory_call(lambda: self.navmesh_memory.update_session(
            world_id, session_id, label=label, pinned=pinned))

    def navmesh_memory_delete(self, world_id: Any, session_id: Any = None, scope: Any = None) -> dict[str, Any]:
        # 整个世界删除必须显式 scope="world"：漏传 session 不能变成删整个世界。
        if scope == "world":
            return self._memory_call(lambda: self.navmesh_memory.delete_world(world_id))
        if session_id is None:
            return {"ok": False, "reason": "session_required"}
        return self._memory_call(lambda: self.navmesh_memory.delete_session(world_id, session_id))

    def navmesh_memory_prune(self) -> dict[str, Any]:
        return self._memory_call(self.navmesh_memory.prune)

    @staticmethod
    def _memory_call(fn: Any) -> dict[str, Any]:
        # 管理错误是可预期的（id 不对、会话正在写）：给明确 reason，不让 HTTP 层当 400 异常。
        try:
            return {"ok": True, **fn()}
        except KeyError as exc:
            return {"ok": False, "reason": str(exc.args[0]) if exc.args else "not_found"}
        except (ValueError, PermissionError) as exc:
            return {"ok": False, "reason": str(exc)}
        except OSError as exc:
            return {"ok": False, "reason": f"io_error: {exc}"[:200]}

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
            # 光流只需要读取现有 OSC/scheduler 状态，不向控制线程反向发送命令。
            # provider 在 worker 真正启动后才调用，此时 scheduler/OSC 已完成初始化。
            motion_provider=self._vision_motion_feedback,
            turning_provider=self._navigator_turn_state,
            # 显式传，不依赖默认值：captured_at 要与 OSC 样本时间戳相减，必须同域。
            clock=self.vision._clock,
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

        if vision.capture == "wgc" or vision.source == "wgc":
            # 按窗口捕获：不经桌面合成，因此既不需要窗口矩形，也不需要
            # WindowTrackedFrameSource 那层 TTL 重解析——窗口被拖动或改分辨率时
            # 捕获项自己跟着走。没有 window_title 就无从捕获，直接放弃。
            if not vision.window_title:
                return None
            return WgcWindowFrameSource(title=vision.window_title)

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
                # 动作时间轴（路线历史通道）：必须在监听器之前构造，因为它同时是
                # 监听器的 on_action sink。关闭时保持 None，监听器只做统计。
                if self.config.driver_log.action_timeline_enabled:
                    self.action_timeline = ActionTimeline(
                        self.config.driver_log.action_timeline_path,
                        self.config.driver_log.action_timeline_fps,
                        enabled=True,
                    )
                    # 服务启动即视为动作时间轴原点。若录像由外部（OBS）启动，录制侧
                    # 应在真正开录时调用 action_timeline_begin() 重新锚定——只允许在
                    # 还没有任何动作行落盘前重锚（见该方法），否则帧号会整体错位。
                    self.action_timeline.begin()
                self.driver_log = DriverLogListener(
                    self.config.driver_log,
                    logger=self.logger,
                    on_action=(self.action_timeline.ingest_driver_event
                               if self.action_timeline is not None else None),
                )
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
                # LAS2 预热放后台：首帧编译 ~80 s（热缓存）起，等到开 navmesh 会话才编就会卡住首帧。
                # 子进程模式下这里拉起的是 .venv_las 的 3.14 进程，主后端不必自带 torch，仍是非阻塞。
                self.navmesh.prewarm()
                self._start_policy_head()
                self._started = True
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                self.stop()
                raise

    def _navigator_complete_goal(self, reason: str) -> None:
        """导航总时限到期时同时撤销语义单槽，避免后台继续刷新图片。"""
        self._clear_pending_semantic_intent()
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
                # 首次启动要给基准，宿主暂停或 VMC 输出重启后丢掉的基准也要补。
                # 后者如果放任普通帧自锁，恢复后的第一个动画姿势就会成为手腕
                # 朝向与手指弯曲的零点，整套骨骼会一直错位。
                if not relay.needs_recalibration():
                    stop_event.wait(_VMC_CALIBRATION_IDLE_POLL_SECONDS)
                    continue
                relay.hold_calibration(reason="waiting_for_host_t_pose")
                calibrated = host_vmc.calibrate_rest_pose(
                    lambda: relay.reset_calibration(reason="host_t_pose"),
                    stop_event=stop_event,
                )
                if stop_event.is_set():
                    return
                if calibrated:
                    # 握手只是一次 POST，立刻回到循环顶部不会经过任何等待。而基准要等
                    # 宿主的 rawRestPose 帧真正到达、并通过 _require_t_pose_frame 的
                    # 解剖学校验才会锁上（vmc_idle._finalize_pending_locked 里置
                    # _had_baseline）。在那之前 needs_recalibration() 恒为真，不歇一下
                    # 就会变成一个打满宿主 API 的紧循环。等过 T Pose 的时长，让帧有机会
                    # 到达，再判断基准是否真的锁上。
                    stop_event.wait(_VMC_T_POSE_WINDOW_SECONDS)
                    continue
                if relay.has_baseline():
                    # 曾经有过基准：保持拒绝普通帧并继续重试，角色停在最后一帧，
                    # 而不是拿动画帧重新自锁出一个错误的静止姿势。
                    relay.require_rest_baseline(reason="t_pose_unavailable")
                else:
                    # 首次校准就失败：宿主可能没有可用的 T Pose 接口，退化为
                    # 由普通帧自锁，至少让待机中转可用。
                    relay.reset_calibration(reason="t_pose_unavailable")
                stop_event.wait(_VMC_CALIBRATION_RETRY_SECONDS)

        self._vmc_calibration_thread = threading.Thread(
            target=calibrate,
            name="neko-vmc-rest-calibration",
            daemon=True,
        )
        self._vmc_calibration_thread.start()

    def vmc_recalibrate(
        self,
        reason: Any = "panel_recalibrate",
        accept_current_pose: Any = False,
    ) -> dict[str, Any]:
        """重新锁定 VMC 静止基准，用来救「动作一直是歪的」。

        手腕朝向和手指弯曲的零点一旦锁在错误的姿势上，之后每一帧都带着同样的
        偏移，而中转本身一个错误都不报——`last_error` 是空的、帧也在正常接收，
        所以除了「看起来不对」以外没有任何信号。这条入口就是给那一刻用的。

        最关键的约束是**按了不能比不按更糟**：清掉基准之后如果没有人去重建，
        中转会一直拒绝普通帧、角色停在最后一帧，从「歪着能动」变成「正着不
        动」。所以先确认真的有校准线程会跑这次握手，再动状态；确认不了就原样
        拒绝，并在 reason 里给出退路。
        """
        normalized_reason = (
            str(reason or "panel_recalibrate").replace("\x00", "").strip()[:64]
            or "panel_recalibrate"
        )
        with self._lock:
            relay = self.vmc_idle
            thread = self._vmc_calibration_thread
        if relay is None or not self.config.vmc_idle.enabled:
            return {
                "accepted": False,
                "mode": "none",
                "reason": "VMC idle relay is disabled or not initialized",
                "calibration": {},
            }
        if accept_current_pose is True:
            # 退路：不等宿主 T Pose，下一个完整帧直接成为新基准。宿主给不出
            # T Pose 时（没有托管输出、REST 接口不通、握手反复超时）这是唯一能
            # 让中转重新动起来的办法。代价是基准质量取决于按下那一刻的姿势，
            # 所以不做默认——但它必须存在，否则严格路径失败就没有出口了。
            relay.reset_calibration(reason=normalized_reason)
            return {
                "accepted": True,
                "mode": "accept_current_pose",
                "reason": normalized_reason,
                "calibration": relay.snapshot().get("calibration", {}),
            }
        if not self.config.vmc_idle.manage_host_output or thread is None or not thread.is_alive():
            # 没有校准线程就没有人去请求 T Pose。此时清掉基准等于把角色永久冻在
            # 最后一帧，比错位更糟，所以一个字段都不改。
            return {
                "accepted": False,
                "mode": "none",
                "reason": (
                    "no host T-pose handshake is available (vmc_idle.manage_host_output "
                    "is off or the calibration worker is not running); retry with "
                    "accept_current_pose=true to lock the next ordinary frame instead"
                ),
                "calibration": relay.snapshot().get("calibration", {}),
            }
        relay.require_rest_baseline(reason=normalized_reason)
        return {
            "accepted": True,
            "mode": "host_t_pose",
            "reason": normalized_reason,
            "calibration": relay.snapshot().get("calibration", {}),
        }

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

    # LAS2 收尾（按真实 pid 收整棵树 + 等子进程退出 + 释放共享内存）在子进程正忙着
    # 编译时实测要 16.7 s（2026-11-05，--offline 冷启也是这个数），空闲已就绪时 ~1 s。
    # 退出路径上不值得为「编译中」这种情况等十几秒：taskkill /T /F 与 job 的
    # KILL_ON_JOB_CLOSE 已经保证子进程活不下来，等不到就继续拆其余资源。
    _STOP_NAVMESH_SHUTDOWN_BUDGET_S = 3.0

    def stop(self) -> None:
        """按依赖顺序拆掉全部运行时资源。

        每一步都单独计时并吞异常：拆卸是「尽力做完」而不是「做对」，任何一步失败都
        不该让后面的资源泄漏，更不该让进程退不出去。最后打一行 service_stop_steps，
        把「关机为什么慢」变成可直接读到的数字（以前它只表现为「关不掉」）。
        """
        timings: dict[str, float] = {}
        # 先停 VLA 头：它 10 Hz 写摇杆，必须早于 scheduler 拆除，也不能让它在 navmesh
        # 收帧之后继续拿旧帧。
        stop_flag = getattr(self, "_policy_body_watchdog_stop", None)
        if stop_flag is not None:
            stop_flag.set()
        head = self.policy_head
        self.policy_head = None
        if head is not None:
            self._run_stop_step(timings, "policy_head", head.stop)
        self.navmesh.drive_enabled = True
        # 先停 navmesh 控制环（它会在 OSC/scheduler 拆除前发一次停车）。
        self._run_stop_step(timings, "navmesh_stop", self.navmesh.stop)
        # 会话 stop 不关 LAS2 子进程（已编译的进程要跨会话复用），只有后端收工才释放它。
        self._run_stop_step(
            timings,
            "navmesh_shutdown",
            self.navmesh.shutdown,
            budget_s=self._STOP_NAVMESH_SHUTDOWN_BUDGET_S,
        )
        with self._lock:
            # 先解除授权，释放虚拟控制器叠加层和 OSC 回退，再拆除 worker/套接字。
            self._pending_semantic_intent = None
            self._run_stop_step(
                timings,
                "autonomy_disarm",
                lambda: self.autonomy.disarm("backend_stopped"),
            )
            self._run_stop_step(timings, "navigator_stop", self.navigator.stop)
            self._run_stop_step(timings, "vmc_calibration", self._stop_vmc_calibration)
            self._run_stop_step(
                timings,
                "vision_worker",
                lambda: self._stop_vision_worker_locked(reason="backend_stopped"),
            )
            self._run_stop_step(timings, "vision_close", self.vision.close)
            if self.driver_log:
                self._run_stop_step(timings, "driver_log", self.driver_log.stop)
            if self.action_timeline:
                self._run_stop_step(timings, "action_timeline", self.action_timeline.close)
            if self.osc:
                self._run_stop_step(timings, "osc", self.osc.stop)
            if self.scheduler:
                self._run_stop_step(timings, "scheduler", self.scheduler.shutdown)
            if self.host_vmc:
                self._run_stop_step(timings, "host_vmc", self.host_vmc.stop)
            if self.vmc_idle:
                self._run_stop_step(timings, "vmc_idle", self.vmc_idle.stop)
            self.driver_log = None
            self.action_timeline = None
            self.osc = None
            self.scheduler = None
            self.host_vmc = None
            self.vmc_idle = None
            self._started = False
        _emit_event(
            "service_stop_steps",
            total_s=round(sum(timings.values()), 2),
            steps=timings,
        )

    def _run_stop_step(
        self,
        timings: dict[str, float],
        name: str,
        action: Callable[[], Any],
        *,
        budget_s: float | None = None,
    ) -> None:
        """执行一步拆卸：计时、吞异常；给了 budget_s 就限时，超时后让它在后台收尾。

        限时步骤必须放到后台线程：它多半卡在子进程 wait 或第三方库的 join 上，主线程
        不能陪着等——退出路径的总时长由调用方（后端进程的看门狗）兜底。
        """
        started = time.perf_counter()
        try:
            if budget_s is None:
                action()
            else:
                worker = threading.Thread(
                    target=action,
                    name=f"backend-stop-{name}",
                    daemon=True,
                )
                worker.start()
                worker.join(timeout=budget_s)
                if worker.is_alive():
                    _emit_event(
                        "stop_step_budget_exceeded",
                        step=name,
                        budget_s=budget_s,
                    )
        except Exception as exc:  # noqa: BLE001 - 拆卸失败只留痕，不能挡住退出
            _emit_event(
                "stop_step_failed",
                step=name,
                error=f"{type(exc).__name__}: {exc}"[:240],
            )
        finally:
            timings[name] = round(time.perf_counter() - started, 2)

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
        with self._lock:
            pending = self._pending_semantic_intent
            result["pending_semantic_intent"] = (
                {
                    "request_id": pending.get("request_id"),
                    "action": pending.get("action"),
                    "text": pending.get("text"),
                    "selector": dict(pending.get("selector") or {}),
                    "route_planning": pending.get("route_planning") is True,
                }
                if isinstance(pending, Mapping)
                else None
            )
        return result

    def _clear_pending_semantic_intent(self) -> None:
        with self._lock:
            self._pending_semantic_intent = None

    def autonomy_arm(self, ttl_s: Any = None) -> dict[str, Any]:
        try:
            normalized = None if ttl_s is None else float(ttl_s)
        except (TypeError, ValueError, OverflowError):
            return {"accepted": False, "reason": "ttl_s must be a number", **self.autonomy.snapshot()}
        return {"accepted": True, **self.autonomy.arm(ttl_s=normalized)}

    def autonomy_disarm(self, reason: Any = "manual_disarm") -> dict[str, Any]:
        normalized_reason = str(reason or "manual_disarm")
        self._clear_pending_semantic_intent()
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
        target_ref: Any = None,
        frame_revision: Any = None,
    ) -> dict[str, Any]:
        normalized_kind = str(kind or "explore").strip().lower()
        raw_target_id = str(target_id or "").replace("\x00", "").strip()
        raw_target_ref = str(target_ref or "").replace("\x00", "").strip().upper()
        if len(raw_target_id) > 96:
            return {
                **self.autonomy.snapshot(),
                "accepted": False,
                "reason_code": "invalid_target_id",
                "reason": "target_id must not exceed 96 characters",
            }
        if len(raw_target_ref) > 8:
            return {
                **self.autonomy.snapshot(),
                "accepted": False,
                "reason_code": "invalid_target_ref",
                "reason": "target_ref must not exceed 8 characters",
            }
        normalized_target_id = raw_target_id or None
        normalized_target_ref = raw_target_ref or None
        world = self.vision.snapshot()
        # 用提交瞬间的世界同步 revision；target_ref 可以来自十秒前的画面，但执行
        # 只能锁定此刻仍在视野中的同一个稳定实体。
        self.autonomy.update_world(world)
        autonomy = self.autonomy.snapshot()
        targeted_kinds = {"approach", "approach_observe", "follow", "interact", "socialize"}
        if normalized_kind in targeted_kinds and not autonomy.get("armed"):
            return {
                **autonomy,
                "accepted": False,
                "reason_code": "manual_arm_required",
                "reason": "VRChat autonomy is not manually armed",
            }
        resolved_by: str | None = None
        if normalized_target_ref:
            if normalized_kind not in targeted_kinds:
                return {
                    **autonomy,
                    "accepted": False,
                    "reason_code": "target_ref_unsupported_for_goal",
                    "reason": "target_ref is only supported for targeted autonomy goals",
                }
            if normalized_target_id:
                return {
                    **autonomy,
                    "accepted": False,
                    "reason_code": "target_selector_conflict",
                    "reason": "target_ref and target_id cannot be used together",
                }
            resolved = self._resolve_target_ref(normalized_target_ref, frame_revision)
            if not resolved.get("accepted"):
                return {**autonomy, **resolved}
            normalized_target_id = str(resolved["target_id"])
            resolved_by = "frame_target_ref"
        if normalized_kind in targeted_kinds:
            vision = world.get("vision") or {}
            semantic = vision.get("semantic") if isinstance(vision, Mapping) else {}
            if isinstance(semantic, Mapping) and semantic.get("last_error"):
                return {
                    "accepted": False,
                    "reason": "semantic vision is degraded; only safe exploration is allowed",
                    **self.autonomy.snapshot(),
                }
            if normalized_target_id is None:
                return {
                    **autonomy,
                    "accepted": False,
                    "reason_code": "target_id_required",
                    "reason": "target_ref and frame_revision are required for a visual target",
                    "instruction": "调用 vrc_vision_frame(overlay=true)，选择 T 编号后重试。",
                }
            visible_entity = self._visible_entity_by_id(world, normalized_target_id)
            if visible_entity is None:
                return {
                    **autonomy,
                    "accepted": False,
                    "reason_code": "target_id_not_visible",
                    "reason": "selected target is not currently visible in the latest world snapshot",
                    "target_ref": normalized_target_ref,
                    "instruction": "重新调用 vrc_vision_frame(overlay=true) 并选择当前画面中的 T 编号。",
                }
        # 方向记忆拒绝收口在这里：autonomy_wander_step 会先自己查一次（它必须在
        # 消费路线任务之前拒绝），但直接调 autonomy_goal(kind="wander") 的路径也
        # 必须走同一道闸，否则绕过工具层就能把刚撞过两次的方向再提一遍。
        refusal = self._refuse_blocked_bearing(normalized_kind, constraints)
        if refusal is not None:
            return refusal
        result = self.autonomy.submit_goal(
            text,
            kind,
            normalized_target_id,
            selector,
            constraints,
            based_on_revision,
        )
        if resolved_by:
            result["resolved_by"] = resolved_by
            result["resolved_target_ref"] = normalized_target_ref
            result["resolved_target_id"] = normalized_target_id
        if result.get("accepted"):
            # 任何显式新目标都会替换尚未完成的“识别后继续”意图。自动续接路径
            # 会在调用本方法前先取出并清空待决项，因此不会误删自己的数据。
            self._clear_pending_semantic_intent()
            goal = result.get("goal") if isinstance(result.get("goal"), Mapping) else {}
            if normalized_kind == "wander":
                # 方向分数来自主 LLM 当前画面，只写入短期预测记忆；导航器不能依据
                # 分数替 LLM 选择下一段路线。实测状态和撞墙历史由 DirectionMemory 保留。
                goal_constraints = (
                    goal.get("constraints")
                    if isinstance(goal.get("constraints"), Mapping)
                    else {}
                )
                direction_scores = goal_constraints.get("direction_scores")
                if isinstance(direction_scores, Mapping):
                    self.navigator.update_direction_scores(direction_scores)
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

    def _refuse_blocked_bearing(
        self,
        kind: str,
        constraints: Any,
    ) -> dict[str, Any] | None:
        """已确认封死的方向一律拒绝；返回 None 表示放行。

        只管 wander：它是唯一由主 LLM 自选方向的行为。人物绑定的 approach/follow
        有具体目标，方向由目标决定，不该被扇区记忆挡下来。

        拒绝载荷刻意只有四个键，不带 autonomy 快照：这不是一次执行结果，多余的
        字段会让主 LLM 以为动作已经发生。也刻意**不**替它换一个方向——选路权在主
        LLM，本地静默换向会让它以为自己选的方向可走。
        """
        if kind != "wander" or not isinstance(constraints, Mapping):
            return None
        raw = constraints.get("turn_deg")
        if raw is None:
            return None
        try:
            bearing = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(bearing):
            return None
        if not self.navigator.should_refuse_bearing(bearing):
            return None
        return {
            "accepted": False,
            "reason_code": "recent_direction_blocked",
            "blocked_bearing_deg": bearing,
            "route_choice_owner": "main_llm",
        }

    def autonomy_wander_step(
        self,
        direction: Any,
        route_request_id: Any,
    ) -> dict[str, Any]:
        """消费主 LLM 当前路线任务并提交一段不绑定人物的短闲逛。"""
        normalized_direction = str(direction or "").replace("\x00", "").strip().lower()
        if normalized_direction not in _WANDER_DIRECTION_TURN_DEG:
            return {
                **self.autonomy_snapshot(),
                "accepted": False,
                "movement_started": False,
                "reason_code": "invalid_wander_direction",
                "reason": "direction must be left, forward or right",
            }
        normalized_request_id = (
            str(route_request_id or "").replace("\x00", "").strip()[:128]
        )
        if not normalized_request_id:
            return {
                **self.autonomy_snapshot(),
                "accepted": False,
                "movement_started": False,
                "reason_code": "route_request_id_required",
                "reason": "no injected main-LLM wander route is bound to this call",
            }

        # 用服务锁把“核对待决意图 → 消费精确图片请求 → 提交目标”串成一个原子
        # 控制事务，避免迟到工具调用覆盖用户刚提交的新目标。
        with self._lock:
            pending = self._pending_semantic_intent
            if not isinstance(pending, Mapping) or pending.get("route_planning") is not True:
                return {
                    **self.autonomy_snapshot(),
                    "accepted": False,
                    "movement_started": False,
                    "reason_code": "no_pending_wander_route",
                    "reason": "there is no pending main-LLM wander route",
                }
            expected_request_id = str(pending.get("request_id") or "")[:128]
            if normalized_request_id != expected_request_id:
                return {
                    **self.autonomy_snapshot(),
                    "accepted": False,
                    "movement_started": False,
                    "reason_code": "wander_route_request_mismatch",
                    "reason": "the injected route request was replaced before execution",
                }
            # 方向记忆拒绝必须在消费配对任务之前：拒绝不是执行结果，路线任务要原样
            # 留着让主 LLM 看着同一张画面重选。autonomy_goal 里还有同一道闸兜住直接
            # 提交的路径，但那道闸在消费之后才生效，所以这里必须先自己查一次。
            blocked_bearing = _WANDER_DIRECTION_TURN_DEG[normalized_direction]
            refusal = self._refuse_blocked_bearing(
                "wander", {"turn_deg": blocked_bearing}
            )
            if refusal is not None:
                return refusal
            claimed = self.vision.consume_main_llm_route_request(
                normalized_request_id
            )
            if not claimed.get("accepted"):
                reason_code = str(claimed.get("reason_code") or "route_request_rejected")
                if reason_code not in {
                    "route_request_id_mismatch",
                    "request_is_not_wander_route",
                }:
                    self._pending_semantic_intent = None
                return {
                    **self.autonomy_snapshot(),
                    "accepted": False,
                    "movement_started": False,
                    "reason_code": reason_code,
                    "reason": "the paired wander-route frame is no longer executable",
                    "route_request": dict(claimed),
                }

            pending = dict(pending)
            self._pending_semantic_intent = None
            raw_constraints = (
                dict(pending.get("constraints") or {})
                if isinstance(pending.get("constraints"), Mapping) else {}
            )
            try:
                requested_duration = float(raw_constraints.get("max_duration_s", 2.0))
            except (TypeError, ValueError, OverflowError):
                requested_duration = 2.0
            if not math.isfinite(requested_duration):
                requested_duration = 2.0
            constraints: dict[str, Any] = {
                "turn_deg": _WANDER_DIRECTION_TURN_DEG[normalized_direction],
                "max_duration_s": min(3.0, max(1.0, requested_duration)),
            }
            try:
                forward_axis = float(raw_constraints.get("max_forward_axis"))
            except (TypeError, ValueError, OverflowError):
                forward_axis = math.nan
            if math.isfinite(forward_axis) and 0.05 <= forward_axis <= 1.0:
                constraints["max_forward_axis"] = forward_axis

            # 明确传入 target_id=None、selector=None。路线方向永远不能升级为人物
            # 绑定；要接近或跟随人物必须重新走 T 编号与稳定 ID 流程。
            result = self.autonomy_goal(
                pending.get("text") or "在附近随便逛逛",
                "wander",
                None,
                None,
                constraints,
                claimed.get("frame_revision"),
            )
            return {
                "action": "wander",
                "movement_started": bool(result.get("accepted")),
                "resolved_by": "main_llm_wander_step",
                "direction": normalized_direction,
                "route_binding": {
                    "request_id": normalized_request_id,
                    "frame_revision": claimed.get("frame_revision"),
                    "request_age_ms": claimed.get("request_age_ms"),
                    "frame_age_ms": claimed.get("frame_age_ms"),
                    "target_binding": "none",
                },
                **result,
            }

    @staticmethod
    def _visible_entity_by_id(
        world: Mapping[str, Any], target_id: str | None,
    ) -> Mapping[str, Any] | None:
        """只按精确稳定 ID 取当前可见实体，不按标签或置信度猜目标。"""
        if not target_id:
            return None
        entities = world.get("entities") if isinstance(world.get("entities"), list) else []
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            if str(entity.get("id") or "") == target_id and entity.get("visible") is True:
                return entity
        return None

    def _remember_target_refs(self, frame: Mapping[str, Any]) -> None:
        """缓存一次真正画进 JPEG 的 T 编号映射；不写磁盘、不保存像素。"""
        overlay = frame.get("overlay") if isinstance(frame.get("overlay"), Mapping) else {}
        if overlay.get("paired") is not True or overlay.get("drawn") is not True:
            return
        try:
            revision = int(overlay.get("revision"))
        except (TypeError, ValueError, OverflowError):
            return
        candidates = overlay.get("candidates") if isinstance(overlay.get("candidates"), list) else []
        refs: dict[str, str] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            ref = str(candidate.get("ref") or "").strip().upper()[:8]
            stable_id = str(candidate.get("target_id") or "").strip()[:96]
            if ref.startswith("T") and ref[1:].isdigit() and stable_id:
                refs[ref] = stable_id
        if not refs:
            return
        now = time.monotonic()
        with self._lock:
            self._prune_target_refs_locked(now)
            self._target_ref_cache[revision] = {
                "stored_at": now,
                "frame_id": str(overlay.get("frame_id") or "")[:128] or None,
                "refs": refs,
            }
            while len(self._target_ref_cache) > _TARGET_REF_CACHE_LIMIT:
                oldest = next(iter(self._target_ref_cache))
                self._target_ref_cache.pop(oldest, None)

    def _prune_target_refs_locked(self, now: float) -> None:
        expired = [
            revision
            for revision, entry in self._target_ref_cache.items()
            if now - float(entry.get("stored_at", 0.0) or 0.0) > _TARGET_REF_CACHE_TTL_S
        ]
        for revision in expired:
            self._target_ref_cache.pop(revision, None)

    def _resolve_target_ref(self, target_ref: str, frame_revision: Any) -> dict[str, Any]:
        """把 T 编号和帧版本原子解析为稳定 ID；任何歧义都拒绝。"""
        if not target_ref.startswith("T") or not target_ref[1:].isdigit() or int(target_ref[1:]) < 1:
            return {
                "accepted": False,
                "reason_code": "invalid_target_ref",
                "reason": "target_ref must look like T1 or T2",
            }
        if isinstance(frame_revision, bool):
            normalized_revision = None
        else:
            try:
                numeric_revision = float(frame_revision)
                normalized_revision = int(numeric_revision) if math.isfinite(numeric_revision) and numeric_revision.is_integer() else None
            except (TypeError, ValueError, OverflowError):
                normalized_revision = None
        if normalized_revision is None or normalized_revision < 0:
            return {
                "accepted": False,
                "reason_code": "frame_revision_required",
                "reason": "a non-negative frame_revision is required with target_ref",
            }
        now = time.monotonic()
        with self._lock:
            self._prune_target_refs_locked(now)
            entry = self._target_ref_cache.get(normalized_revision)
            if entry is None:
                return {
                    "accepted": False,
                    "reason_code": "target_ref_revision_unavailable",
                    "reason": "the selected overlay revision is unavailable or expired",
                    "frame_revision": normalized_revision,
                    "instruction": "重新调用 vrc_vision_frame(overlay=true) 获取当前 T 编号。",
                }
            refs = entry.get("refs") if isinstance(entry.get("refs"), Mapping) else {}
            stable_id = str(refs.get(target_ref) or "")[:96]
            if not stable_id:
                return {
                    "accepted": False,
                    "reason_code": "target_ref_not_found",
                    "reason": "target_ref was not present in the selected overlay revision",
                    "target_ref": target_ref,
                    "frame_revision": normalized_revision,
                    "available_refs": sorted(str(item) for item in refs),
                }
        return {
            "accepted": True,
            "target_id": stable_id,
            "target_ref": target_ref,
            "frame_revision": normalized_revision,
        }

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
                    "local avoidance relies on a 2.5D incremental occupancy grid "
                    "(requires stereo depth and the local navigator to be enabled); "
                    "it spans only traversed areas and has no cross-occlusion global "
                    "reasoning, so it cannot route to or verify an occluded area"
                ),
                "instruction": "请让用户手动带路到可见位置，再观察新鲜画面。",
            }
        if normalized_action not in {"find", "approach", "follow", "depart", "wander"}:
            return {
                **self.autonomy.snapshot(),
                "accepted": False,
                "reason_code": "invalid_action",
                "reason": "action must be find, approach, follow, depart, wander, stop or status",
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
        if constraints is not None and not isinstance(constraints, Mapping):
            return {
                **autonomy,
                "accepted": False,
                "reason_code": "invalid_constraints",
                "reason": "constraints must be an object",
            }

        # 离开当前位置和 LLM 规划的闲逛路段都不是“选一个人”，不能为了复用
        # 目标导航而伪造 person selector。wander 的 constraints.turn_deg 必须由
        # LLM 看当前画面后明确给出；后台 Agent 没有带上方向时，把最新配对画面
        # 交回主多模态 LLM 规划，绝不能由导航器本地猜一条路线。
        if normalized_action in {"depart", "wander"}:
            world = self.vision.snapshot()
            self.autonomy.update_world(world)
            status = world.get("status") if isinstance(world.get("status"), Mapping) else {}
            revision = int(status.get("revision", 0) or 0)
            normalized_text = str(text or "").replace("\x00", "").strip()[:256]
            has_wander_direction = (
                normalized_action != "wander"
                or (
                    isinstance(constraints, Mapping)
                    and constraints.get("turn_deg") is not None
                )
            )
            if not has_wander_direction:
                # 旧目标必须先停稳；只保留手动 arm，等待主模型基于这张图提交
                # 一条最多三秒的短路段。该请求只驻留内存，不写原图到磁盘。
                self.autonomy.complete_goal("wander_intent_waiting_for_route")
                self.vision.clear_main_llm_semantic_request(
                    "wander_intent_replaces_semantic_request"
                )
                route_request = self.vision.request_main_llm_semantics(
                    {
                        "semantic_type": "object",
                        "label": "safe visible walkable direction",
                        "min_confidence": 0.0,
                    },
                    reason="agent_wander_direction_unresolved",
                    force=True,
                )
                if route_request.get("accepted"):
                    request_id = str(route_request.get("request_id") or "")[:128]
                    with self._lock:
                        self._pending_semantic_intent = {
                            "request_id": request_id,
                            "action": "wander",
                            "text": normalized_text or "在附近随便逛逛",
                            "selector": {},
                            "constraints": (
                                dict(constraints) if isinstance(constraints, Mapping) else {}
                            ),
                            "route_planning": True,
                        }
                    return {
                        **self.autonomy.snapshot(),
                        "accepted": False,
                        "route_request_accepted": True,
                        "completed": False,
                        "action": "wander",
                        "reason_code": "wander_direction_pending",
                        "reason": "the main multimodal LLM must choose a direction from the paired frame",
                        "pending_route": True,
                        "movement_started": False,
                        "route_request": route_request,
                        "instruction": (
                            "最新画面已交给主多模态模型选择单段路线；当前尚未移动。"
                            "不要声称已经出发，也不要重复提交闲逛；主模型会根据画面提交 turn_deg。"
                        ),
                    }
                return {
                    **self.autonomy.snapshot(),
                    "accepted": False,
                    "route_request_accepted": False,
                    "completed": False,
                    "action": "wander",
                    "reason_code": "wander_direction_frame_unavailable",
                    "reason": "no fresh paired frame is available for main-LLM route planning",
                    "pending_route": False,
                    "movement_started": False,
                    "route_request": route_request,
                    "instruction": "没有可供主模型规划路线的新鲜画面，动作没有开始。",
                }
            result = self.autonomy_goal(
                normalized_text or (
                    "离开当前位置" if normalized_action == "depart" else "在附近随便逛逛"
                ),
                normalized_action,
                None,
                None,
                constraints,
                revision,
            )
            return {
                "action": normalized_action,
                "movement_started": bool(result.get("accepted")),
                "resolved_by": (
                    "llm_planned_local_step"
                    if normalized_action == "wander"
                    else "bounded_local_behavior"
                ),
                **result,
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
        if normalized_action == "approach" and not normalized_target_id and not normalized_text:
            navigation = self.navigator.snapshot()
            behavior = navigation.get("behavior") if isinstance(navigation.get("behavior"), Mapping) else {}
            outcome = behavior.get("last_outcome") if isinstance(behavior.get("last_outcome"), Mapping) else {}
            try:
                # 必须用 navigator 自己的时钟：这行原来是硬编码 time.monotonic()，跨域相减
                # 会得到负 age，被下面的 0.0 <= age 守卫静默拒绝，快路径永远走不到。
                # 必须用 navigator 自己的时钟。原先硬编码 time.monotonic()，跨域相减得负数，
                # 被下面的 0.0 <= age 守卫静默拒绝，快路径永远走不到。
                # 不设 getattr 兜底：兜底值恰好是错的域，宁可 AttributeError 炸出来。
                outcome_age_s = self.navigator._clock() - float(outcome.get("occurred_at_monotonic"))
            except (TypeError, ValueError, OverflowError):
                outcome_age_s = math.inf
            if (
                outcome.get("reason") == "approach_observe_complete"
                and 0.0 <= outcome_age_s <= 15.0
            ):
                return {
                    **self.autonomy.snapshot(),
                    "accepted": True,
                    "completed": True,
                    "action": normalized_action,
                    "movement_started": False,
                    "reason_code": "recent_approach_already_completed",
                    "reason": "the same unqualified background approach arrived after a recent completion",
                    "navigation": navigation,
                    "instruction": "最近一次接近观察已经完成；不要重复移动或创建语义任务。",
                }
        candidates = self._semantic_navigation_candidates(world, selector)
        if not normalized_target_id:
            if not candidates and not _selector_is_locally_detectable(selector):
                # 非人形目标：直接附上当前画面让主 LLM 一步估算方位并提交 wander。
                frame_result = self.vision.pull_frame(max_age_ms=2000)
                frame_data = frame_result.pop("data", None) if isinstance(frame_result, dict) else None
                return {
                    **self.autonomy.snapshot(),
                    "accepted": False,
                    "completed": False,
                    "action": normalized_action,
                    "reason_code": "target_not_locally_detectable",
                    "reason": (
                        "local detection only tracks humanoids; this target cannot be"
                        " resolved into a navigable entity"
                    ),
                    "pending_semantic": False,
                    "movement_started": False,
                    "frame": frame_result,
                    "frame_data": frame_data,
                    "instruction": (
                        "本机视觉只能锁定人物,海报/屏幕/家具无法导航。已附当前画面,立即"
                        "估算目标相对方位(正数左转、负数右转;目标在画面右侧填负值,"
                        "例如右前方填 -20)并调用 vrc_autonomy_goal(goal=\"朝目标"
                        "方向走走看\",kind=\"wander\",constraints={\"turn_deg\":估算角度,"
                        "\"max_duration_s\":3.0},based_on_revision=frame.revision)。"
                        "turn_deg 限 ±45°,更偏先 body_turn;max_duration_s 上限 3.0,"
                        "远处目标就填 3.0。不提交 target_id/target_ref/selector。"
                        "一段最多走 3~4 米,通常到不了:这是多段任务的第一段。"
                        "每段停下后你会收到新画面和 progress_m,那时再判断是继续提交下一段、"
                        "修正方向还是已经够近了。到达之前只能说朝那边走走看,不要说已经到了。"
                    ),
                }
            if not candidates:
                # 本地 person 检测只能给出稳定轨迹，无法断言它是 NPC、玩家还是
                # 海报。建立一次与当前画面配对的主多模态任务，并暂停旧目标；
                # commit 后若只得到一个匹配且仍可见的目标，后端会自动续接。
                # 这里只暂停旧目标，不撤销用户的手动 arm；语义任务完成后还要在
                # 同一授权会话里自动续接。complete_goal 会释放所有控制输入。
                self.autonomy.complete_goal("navigation_intent_waiting_for_semantics")
                self.vision.clear_main_llm_semantic_request(
                    "navigation_intent_replaces_semantic_request"
                )
                semantic_request = self.vision.request_main_llm_semantics(
                    selector,
                    reason="agent_navigation_target_unresolved",
                    force=True,
                )
                if semantic_request.get("accepted"):
                    request_id = str(semantic_request.get("request_id") or "")[:128]
                    with self._lock:
                        self._pending_semantic_intent = {
                            "request_id": request_id,
                            "action": normalized_action,
                            "text": normalized_text,
                            "selector": dict(selector),
                            "constraints": dict(constraints) if isinstance(constraints, Mapping) else constraints,
                        }
                    return {
                        **self.autonomy.snapshot(),
                        # accepted 只表示“移动目标已经进入执行态”。语义图片虽然已
                        # 成功排队，但角色尚未移动，不能复用 accepted=true 让上层
                        # 把两件事混为一谈。
                        "accepted": False,
                        "semantic_request_accepted": True,
                        "completed": False,
                        "action": normalized_action,
                        "reason_code": "semantic_target_pending",
                        "reason": "target semantics must be confirmed from the paired frame",
                        "pending_semantic": True,
                        "movement_started": False,
                        "semantic_request": semantic_request,
                        "instruction": (
                            "目标画面已交给主多模态模型确认；当前尚未移动。"
                            "不要重复提交移动，也不要声称已经到达；分类提交后后端会自动续接。"
                        ),
                    }
                return {
                    **self.autonomy.snapshot(),
                    "accepted": False,
                    "action": normalized_action,
                    "reason_code": "semantic_target_not_found",
                    "reason": "no verified target is visible and no paired semantic frame is available",
                    "pending_semantic": False,
                    "movement_started": False,
                    "semantic_request": semantic_request,
                    "instruction": "当前没有可确认的配对画面，动作没有开始。",
                }
            if len(candidates) != 1:
                return {
                    **autonomy,
                    "accepted": False,
                    "action": normalized_action,
                    "reason_code": "target_choice_required",
                    "reason": "multiple semantic targets match; the main LLM must choose an exact target_id",
                    "candidates": candidates,
                }
            normalized_target_id = str(candidates[0]["target_id"])
            resolved_by = "unique_semantic_target"
        else:
            resolved_by = "exact_target_id"
        # 高层的“走过去”默认是有限行为：到达后停稳并观察，而不是把永久目标
        # 留给主 LLM 一步步遥控。低层显式 follow 仍保持持续跟随语义。
        goal_kind = "approach_observe" if normalized_action == "approach" else normalized_action
        result = self.autonomy_goal(
            normalized_text or f"{normalized_action} {normalized_label or normalized_type}",
            goal_kind,
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
        self._clear_pending_semantic_intent()
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

    # ---- W1 世界身份子系统（world_model） ----------------------------------

    def world_model_status(self) -> dict[str, Any]:
        """GET /worldmodel/status 的契约字段。永不抛异常。"""
        return self.world_model.status()

    def world_model_start(self) -> dict[str, Any]:
        """POST /worldmodel/start：请求立即返回，真正初始化在后台线程做。"""
        return self.world_model.start()

    def world_model_stop(self, reason: Any = "manual_stop") -> dict[str, Any]:
        """POST /worldmodel/stop。"""
        return self.world_model.stop(reason=reason)

    def world_model_set_world(
        self, world_key: Any, world_name: Any = None
    ) -> dict[str, Any]:
        """POST /worldmodel/world：手动设置当前世界标识。

        世界身份不依赖子系统运行：即使未 start() 甚至 disabled 也允许设置，
        因为「知道自己在哪个世界」是记忆分区的根，比子系统是否跑起来更底层。
        """
        return self.world_model.set_world(world_key, world_name)

    # ---- 动作时间轴（路线历史通道） ----------------------------------------

    def action_timeline_status(self) -> dict[str, Any]:
        """动作时间轴：是否启用 / 是否已锚定 / 已写多少行。"""
        timeline = self.action_timeline
        return timeline.status() if timeline is not None else {
            "enabled": False, "active": False,
        }

    def action_timeline_begin(self) -> dict[str, Any]:
        """让**录像侧**在真正开录时重新锚定时间基准。

        只允许在还没有任何动作行落盘前调用：一旦写过行，帧号已按旧锚点换算，
        再改锚点会让前后两段 ``frame_index`` 落在不同基准上。
        未启用时返回 ``{"ok": False, "reason": "disabled"}``，绝不静默成功——
        调用方必须能区分"锚成功了"和"这个通道根本没开"。
        """
        timeline = self.action_timeline
        if timeline is None:
            return {"ok": False, "reason": "disabled"}
        if timeline.written > 0:
            return {"ok": False, "reason": "already_recording",
                    "records": timeline.written}
        ok = timeline.begin()
        return {"ok": ok, **timeline.status()}

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
            "action_timeline": self.action_timeline_status(),
            "idle_relay": self.vmc_idle.snapshot() if self.vmc_idle else {"enabled": False},
            "host_vmc": self.host_vmc.snapshot() if self.host_vmc else {"managed": False, "active": False},
            "world": self.vision.snapshot(),
            "vision_worker": self._vision_worker_status(),
            "cognition": self.cognition.snapshot(),
            "autonomy": self.autonomy.snapshot(),
            "navigation": self.navigator.snapshot(),
            "turn_trace": self.turn_trace_snapshot(),
            "control_latency": self.control_metrics_snapshot(),
            # 双目后端 + LAS2 预热进度：总览页要显示「预热中」，不用另开 navmesh 页去查。
            "stereo_backend": self.navmesh.stereo_backend_status(),
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
        # 导航结果是离散内存状态，不进入视觉 revision 账本、不落盘。宿主可据此
        # 只在完成/受阻/丢失时通知主 LLM，而不用注入每个控制 tick。
        result["navigation"] = self.navigator.snapshot()
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
        if overlay and result.get("available"):
            self._remember_target_refs(result)
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
        if result.get("available"):
            request_id = str(result.get("request_id") or "")[:128]
            with self._lock:
                pending = self._pending_semantic_intent
                if (
                    isinstance(pending, Mapping)
                    and pending.get("route_planning") is True
                    and str(pending.get("request_id") or "") == request_id
                ):
                    result["route_planning"] = {
                        "action": "wander",
                        "text": str(pending.get("text") or "在附近随便逛逛")[:256],
                        "constraints": dict(pending.get("constraints") or {}),
                    }
        if result.get("reason") == "request_cancelled":
            cancelled_id = str(result.get("request_id") or "")[:128]
            with self._lock:
                pending = self._pending_semantic_intent
                if (
                    isinstance(pending, Mapping)
                    and str(pending.get("request_id") or "") == cancelled_id
                ):
                    route_planning = pending.get("route_planning") is True
                    result["pending_navigation"] = {
                        "accepted": False,
                        "completed": False,
                        "action": str(pending.get("action") or "")[:32],
                        "movement_started": False,
                        "reason_code": (
                            "wander_direction_expired"
                            if route_planning else "semantic_confirmation_expired"
                        ),
                        "reason": (
                            "route planning expired before the main LLM chose a wander direction"
                            if route_planning else
                            "semantic confirmation expired before the navigation target could be resolved"
                        ),
                        "instruction": (
                            "这次闲逛没有开始；必须如实告诉用户尚未移动，不能描述已经出发。"
                            if route_planning else
                            "这次导航没有开始；必须如实告诉用户尚未移动，不能描述正在接近。"
                            "语义任务超时通常说明目标不是人：本机视觉只追踪人形，"
                            "海报、屏幕、展台这类静态物体永远确认不了。用户要去的是这类"
                            "目标时改用 vrc_autonomy_goal(kind=\"wander\") 一段段走过去——"
                            "看画面估算方位填 constraints.turn_deg（正数左转、负数右转；"
                            "目标在画面右侧填负值，例如右前方填 -20），不提交 selector。"
                        ),
                    }
                    self._pending_semantic_intent = None
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
        normalized_request_id = str(request_id or "").replace("\x00", "").strip()[:128]
        with self._lock:
            pending = self._pending_semantic_intent
            if (
                isinstance(pending, Mapping)
                and pending.get("route_planning") is True
                and str(pending.get("request_id") or "") == normalized_request_id
            ):
                return {
                    "accepted": False,
                    "reason_code": "wander_goal_commit_required",
                    "reason": "this request requires a wander goal, not a semantic classification",
                    "movement_started": False,
                    "instruction": (
                        "查看配对画面后调用 vrc_wander_step，只提交 left、forward 或 right；"
                        "请求与 revision 由插件自动绑定，不要调用 vrc_semantic_commit。"
                    ),
                }
        result = self.vision.commit_main_llm_semantics(
            request_id,
            frame_revision,
            entities,
        )
        if not result.get("accepted"):
            return result

        normalized_request_id = str(result.get("request_id") or "")[:128]
        with self._lock:
            pending = self._pending_semantic_intent
            if (
                not isinstance(pending, Mapping)
                or str(pending.get("request_id") or "") != normalized_request_id
            ):
                return result
            pending = dict(pending)
            self._pending_semantic_intent = None

        action = str(pending.get("action") or "").strip().lower()
        selector = pending.get("selector") if isinstance(pending.get("selector"), Mapping) else {}
        bindings = [
            dict(item)
            for item in (result.get("bindings") or ())
            if isinstance(item, Mapping) and self._semantic_binding_matches(item, selector)
        ]
        if len(bindings) != 1:
            result["pending_navigation"] = {
                "accepted": False,
                "action": action,
                "movement_started": False,
                "reason_code": (
                    "semantic_target_not_found" if not bindings else "target_choice_required"
                ),
                "reason": (
                    "the paired frame contains no matching semantic target"
                    if not bindings else
                    "the paired frame contains multiple matching semantic targets"
                ),
                "candidates": bindings,
            }
            return result

        target_id = str(bindings[0].get("target_id") or "")[:96]
        world = self.vision.snapshot()
        self.autonomy.update_world(world)
        if not self.autonomy.snapshot().get("armed"):
            result["pending_navigation"] = {
                "accepted": False,
                "action": action,
                "movement_started": False,
                "reason_code": "manual_arm_required",
                "reason": "VRChat autonomy was disarmed before semantic confirmation completed",
            }
            return result
        if self._visible_entity_by_id(world, target_id) is None:
            result["pending_navigation"] = {
                "accepted": False,
                "action": action,
                "movement_started": False,
                "reason_code": "target_id_not_visible",
                "reason": "the classified target is no longer visible in the latest world snapshot",
            }
            return result

        status = world.get("status") if isinstance(world.get("status"), Mapping) else {}
        revision = int(status.get("revision", 0) or 0)
        goal_kind = "approach_observe" if action == "approach" else action
        resumed = self.autonomy_goal(
            pending.get("text") or f"{action} {bindings[0].get('label') or selector.get('semantic_type') or 'target'}",
            goal_kind,
            target_id,
            None,
            pending.get("constraints"),
            revision,
        )
        result["pending_navigation"] = {
            "action": action,
            "movement_started": bool(resumed.get("accepted")),
            "resolved_by": "main_llm_semantic_commit",
            "resolved_target_id": target_id,
            **resumed,
        }
        return result

    @staticmethod
    def _semantic_binding_matches(
        binding: Mapping[str, Any], selector: Mapping[str, Any],
    ) -> bool:
        """只允许主 LLM 明确分类且满足原选择器的候选自动续接。"""
        semantic_type = str(binding.get("semantic_type") or "").strip().lower()
        if semantic_type in {"", "poster", "screen", "mirror", "unknown"}:
            return False
        expected_type = str(selector.get("semantic_type") or "").strip().lower()
        if expected_type and semantic_type != expected_type:
            return False
        expected_label = str(selector.get("label") or "").strip().casefold()
        label = str(binding.get("label") or "")
        if expected_label and expected_label not in label.casefold():
            return False
        try:
            confidence = float(binding.get("confidence", 0.0) or 0.0)
            minimum = float(selector.get("min_confidence", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            return False
        return math.isfinite(confidence) and confidence >= minimum

    def _navigator_send_axes(self, side: str, x: float, y: float, duration_ms: int) -> bool:
        """只发送主 AnyaDance 命令，绝不回退到 OSC。"""
        scheduler = self.scheduler
        if scheduler is None or self.config.input.primary != "anyadance":
            return False
        result = scheduler.submit(
            "input_axes",
            {"side": side, "x": x, "y": y, "duration_ms": duration_ms},
        )
        accepted = bool(result.get("accepted"))
        timeline = self.action_timeline
        if timeline is not None:
            # 记录"想发什么"(forward=y 前进 / strafe=x 横移) 与"本机是否真的发出"。
            # driver_ack 不由这里写：本机发送成功推不出驱动回执（见 ActionTimeline）。
            timeline.record_command(forward=y, strafe=x, sent=accepted)
        return accepted

    def _record_turn_submission(
        self, source: str, delta_deg: float, accepted: bool
    ) -> None:
        """记账一条 turn 提交：来源、请求增量、是否被调度器接受。"""
        with self._turn_trace_lock:
            self._turn_source_counts[source] = self._turn_source_counts.get(source, 0) + 1
            self._turn_trace.append(
                {
                    "t": round(time.monotonic(), 3),
                    "source": source,
                    "client": self._http_client_addr,
                    "delta_deg": round(float(delta_deg), 2),
                    "accepted": accepted,
                }
            )

    def turn_trace_snapshot(self) -> dict[str, Any]:
        with self._turn_trace_lock:
            return {
                "counts": dict(self._turn_source_counts),
                "recent": list(self._turn_trace),
            }

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
        accepted = bool(result.get("accepted"))
        recorder = getattr(self, "_record_turn_submission", None)
        if recorder is not None:
            recorder("navigator", delta_deg, accepted)
        timeline = self.action_timeline
        if timeline is not None:
            # turn_intent 记录的是**输入意图**（要求转多少度），不是实际转过的角度。
            timeline.record_turn(yaw_delta=delta_deg, sent=accepted)
        return accepted

    def _navigator_release_inputs(self, side: str = "all") -> None:
        scheduler = self.scheduler
        if scheduler is not None:
            scheduler.submit("input_release", {"side": side})

    # ------------------------------------------------------------ VLA 头 A（M0）
    def _policy_enabled(self) -> bool:
        """读取 VLA 开关；正式配置存在时不再依赖进程环境变量。"""
        if bool(getattr(self, "_policy_configured", False)):
            return bool(self.config.policy.enabled)
        return _env_flag("VRC_POLICY_HEAD")

    def _policy_value(self, field: str, env_name: str, default: Any) -> Any:
        """读取策略参数，兼容旧的环境变量部署清单。"""
        if bool(getattr(self, "_policy_configured", False)):
            return getattr(self.config.policy, field)
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            return default
        return raw

    def _start_policy_head(self) -> None:
        """按正式配置或兼容环境变量挂 M0 VLA 头 A；默认不启动。

        UI 的 ``[policy]`` 段优先；没有该段时使用 ``VRC_POLICY_*`` 环境变量。
        ``dry_run`` 默认 true：只写 ``state_dir/policy/policy_live.jsonl``，
        不发轴；设 0 才真正驱动，这时 navmesh 自带控制环让位（``drive_enabled=False``）。

        首次装配要加载权重并 torch.compile，会阻塞本进程数十秒 —— 这是显式开关，
        不接受这个代价就不要打开。依赖缺失（默认 .venv 没有 torch）只记一条警告。
        """
        if not self._policy_enabled():
            # 开关没打开是**正常默认**，不刷屏；但把「为什么没起」记下来给 /policy/state，
            # 因为「静默不启动」比「启动失败」更难查（2026-11-02 花了 20 分钟）。
            self._policy_start_error = (
                "VLA policy disabled (set policy.enabled=true or VRC_POLICY_HEAD=1)")
            return
        model_dir = str(self._policy_value("model_path", "VRC_POLICY_MODEL", "") or "").strip()
        run_dir = str(self._policy_value("run_path", "VRC_POLICY_RUN", "") or "").strip()
        if not model_dir or not run_dir:
            if self.logger:
                self.logger.warning("policy head skipped: set VRC_POLICY_MODEL and VRC_POLICY_RUN")
            return
        dry_run_raw = self._policy_value("dry_run", "VRC_POLICY_DRYRUN", True)
        dry_run = bool(dry_run_raw) if isinstance(dry_run_raw, bool) else _env_flag(
            "VRC_POLICY_DRYRUN", default=True
        )
        # 图重放：需要 Level-Zero V2 适配器（由 PolicyRuntime 内部切）。
        # 默认关闭 —— V2 的单算子派发比 V1 慢，只有确实要吃图才开。
        graph_raw = self._policy_value("graph", "VRC_POLICY_GRAPH", False)
        graph = bool(graph_raw) if isinstance(graph_raw, bool) else _env_flag(
            "VRC_POLICY_GRAPH", default=False
        )
        try:
            hz = float(self._policy_value("hz", "VRC_POLICY_HZ", 1.0))
            v_max = float(self._policy_value("v_max", "VRC_POLICY_VMAX", 1.2))
        except ValueError:
            hz, v_max = 1.0, 1.2
        # 块丢弃阈值跟着节拍走（方案 §1：2 Hz → 0.8 s，1 Hz → 1.3 s）。写死 1.3 的
        # 后果：2 Hz 配置下块在 1.3 s 才死，推理一旦慢于 1.3 s 就出现"走半秒顿半秒"
        # 的节律（2026-11-02 在线实测 halts/s ≈ sends/s）。可用 VRC_POLICY_STALE 覆盖。
        stale_s = 0.8 if hz >= 1.5 else 1.3
        try:
            stale_s = float(self._policy_value("stale_s", "VRC_POLICY_STALE", stale_s))
        except ValueError:
            pass
        # 「不许比跟随器差」的下限系数。默认 0.6（比跟随器慢也不至于慢 16 倍）；
        # 设 0 关闭，逐位回到旧行为。
        try:
            floor_frac = float(self._policy_value("floor_frac", "VRC_POLICY_FLOOR", 0.6))
        except ValueError:
            floor_frac = 0.6
        try:
            # 惰性导入：默认 .venv 没有 torch/transformers，装配失败只应退化成「没有头」。
            from .policy_head import PolicyHead

            head = PolicyHead(
                model_dir=model_dir,
                run_dir=run_dir,
                frame_source=self._policy_frame_source,
                prev_frame_source=self._policy_prev_frame,
                captured_at_source=self._policy_frame_captured_at,
                # A* 只作为软引导输入；动作出口仍完全归 VLA。
                # 当前 checkpoint 若 schema.drop_wp=true，会自动忽略航点；新训练的
                # hybrid checkpoint 可按 schema 消费引导点/路径。两者都不允许用引导
                # 触发硬停车、速度下限或 A* 转向。
                wp_provider=None,
                guidance_provider=self._policy_guidance,
                mode_provider=None,
                body_provider=self._policy_body,
                send_axes=self._policy_send_axes,
                # ⚠️ 转向必须走独立通道：VRChat 摇杆只有 x/y 两轴，
                # 早期把 turn 用关键字塞进 send_axes，_policy_send_axes 签名接不住，
                # except TypeError 兜底后 turn 被静默丢弃 —— 模型输出的 ω 从未进入闭环。
                send_turn=self._policy_send_turn,
                log_dir=str(self.state_dir / "policy"),
                dry_run=dry_run,
                hz=hz,
                v_max=v_max,
                stale_s=stale_s,
                graph=graph,
                near_obstacle=self._policy_near_obstacle,
                # A* 不代管原地转，也不给 VLA 速度下限；路径只作为模型输入先验。
                hold_provider=None,
                path_source=None,
                floor_provider=None,
                floor_frac=0.0,
                # 引导点不是停车条件；失去定位时由 guidance_provider 返回无效状态。
                wp_stop=0.0,
                max_len=int(self._policy_value("max_len", "VRC_POLICY_MAXLEN", 192)),
                prompt_provider=self._policy_prompt,
                logger=self.logger,
            )
        except Exception as exc:  # noqa: BLE001 - 可选件装不上不能拖垮 backend
            # ⚠️ standalone 模式（backend/process.py）**没有任何 logging 配置**，
            # self.logger 是 None，这条警告会彻底消失 —— 2026-11-02 就因此把
            # 「策略头没起来」当成了「策略头起得慢」，白等了 5 分钟。
            # 所以同时写 stderr：装不上必须当场可见，而不是靠 /policy/state 反推。
            msg = "policy head not started: %s: %s" % (type(exc).__name__, exc)
            if self.logger:
                self.logger.warning(msg)
            _sys_for_policy.stderr.write("[policy] " + msg + "\n")
            _sys_for_policy.stderr.flush()
            self._policy_start_error = msg
            return
        self.policy_head = head
        if not dry_run:
            # 同一组摇杆只能有一个发送方：真实驱动时 navmesh 只保留建图/引导，
            # 移动和转向全部由 VLA 动作块发送。
            self.navmesh.drive_enabled = False
            # 调度器 body 输出默认 disabled（只发转向帧、不发摇杆 overlay，见
            # scheduler.py 主发送分支的 _enabled 门）。不 enable 的话轴提交"成功"
            # 却没人发出去 —— 2026-11-02 在线"只转头不移动"的根因（剖析 §15.18）。
            try:
                snap = self.scheduler.snapshot() if self.scheduler else {}
                if snap.get("state") == "disabled":
                    result = self.submit("enable")
                    if self.logger:
                        self.logger.info("policy head auto-enabled body output: %s",
                                         result.get("state"))
            except Exception as exc:  # noqa: BLE001 - enable 失败只记日志，不拖垮启动
                if self.logger:
                    self.logger.warning("policy head body auto-enable failed: %s", exc)
            # 看门狗：stopped_latched / fault_latched / disabled 都会让摇杆静默
            # （剖析 §15.18/§15.21）。latched 的来源可能是外部急停、宿主驱动、
            # 或调度器故障 —— 有活跃任务时自动 reset+enable 恢复；无任务不动，
            # 尊重外部急停的语义。2 s 一拍，轻量读快照。
            self._policy_body_watchdog_stop = threading.Event()
            self._policy_body_watchdog = threading.Thread(
                target=self._policy_body_watchdog_loop,
                name="policy-body-watchdog", daemon=True)
            self._policy_body_watchdog.start()
        head.start()

    def _policy_frame_source(self):
        """左目 RGB（全分辨率）。没有新鲜帧返回 None，runner 会输出零块并标 no_frame。"""
        try:
            latest = self.navmesh.policy_frame()
        except Exception:
            return None
        if latest is None:
            return None
        rgb, captured_at = latest
        if time.perf_counter() - captured_at > _POLICY_FRAME_STALE_S:
            return None
        return rgb

    def _policy_follower_forward_mps(self) -> float:
        """兼容旧诊断：跟随器本拍本可以给出的前进速度（**m/s**）。

        该值不再进入 VLA 的 SafetyGate；保留它只为兼容旧状态和离线诊断。
        """
        try:
            axis = float(getattr(self.navmesh, "_follower_forward", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - 读不到就当没有下限
            return 0.0
        if axis <= 0.0:
            return 0.0
        return axis * _OSC_FULL_AXIS_MPS

    def _policy_guidance(self) -> dict[str, Any]:
        """读取 A* 软引导快照；任何失败都退化为无引导，不影响 VLA 推理。

        2026-11-06：定位安全门是**独立**字段 ``localization_ok``。这里两个兜底
        都写 ``None``（= 没有定位信息，不适用），因为「引导读取失败」不等于
        「定位坏了」—— 拿不到就停会让独立 VLA（无 navmesh）永久不动。
        """
        try:
            getter = getattr(self.navmesh, "policy_guidance", None)
            if callable(getter):
                value = getter(steps=3, step_m=0.8)
                if isinstance(value, Mapping):
                    return dict(value)
        except Exception as exc:  # noqa: BLE001 - 软引导不可用不能阻断控制
            return {
                "valid": False,
                "waypoint": (0.0, 0.0),
                "path": None,
                "mode": 0,
                "confidence": 0.0,
                "localization_ok": None,
                "localization_reason": "guidance_unavailable",
                "reason": f"guidance_error:{type(exc).__name__}",
                "source": "astar_navmesh",
            }
        return {
            "valid": False,
            "waypoint": (0.0, 0.0),
            "path": None,
            "mode": 0,
            "confidence": 0.0,
            "localization_ok": None,
            "localization_reason": "guidance_unavailable",
            "reason": "guidance_unavailable",
            "source": "astar_navmesh",
        }

    def _policy_standalone_waypoint(self) -> tuple[float, float]:
        """旧接线兼容：无导航先验时的零引导点。生产链路使用 _policy_guidance。"""
        return (0.0, 0.0)

    def _policy_standalone_mode(self) -> int:
        """旧接线兼容：无导航先验时固定 follow 模式。"""
        return 0

    def _policy_mode(self) -> int:
        """记忆卡 [mode]：0=follow 3=stop。来自 navmesh 的状态机（确定性，非模型）。"""
        try:
            return int(self.navmesh.policy_mode())
        except Exception:  # noqa: BLE001 - 读不到按 follow，不能因此停车
            return 0

    def _policy_path_ahead(self):
        """沿路径前进的 k 个机体系点，给 VLA 看路径形状。

        ``VRC_POLICY_PATH_IN=1`` 时才真的喂给模型（默认关，因为偏离训练分布）。
        拿不到路径返回 None —— runtime 会退回单点 prompt，行为与训练一致。
        """
        try:
            steps = int(os.getenv("VRC_POLICY_PATH_STEPS") or 3)
        except ValueError:
            steps = 3
        try:
            pts = self.navmesh.policy_path_ahead(steps=steps)
        except Exception:  # noqa: BLE001 - 拿不到就退回单点
            return None
        if not pts:
            return pts
        # 与 _policy_waypoint 同一口径：追踪米 → 世界米。两处必须一起改，
        # 否则「单点」和「路径形状」会差 1/0.755 倍，模型看到互相矛盾的尺度。
        # 轻量测试桩/旧注入对象可能没有这个可选换算钩子；缺失时保持原追踪米。
        scale_fn = getattr(self, "_policy_world_scale", None)
        try:
            ws = float(scale_fn()) if callable(scale_fn) else 0.0
        except Exception:  # noqa: BLE001 - 尺度拿不到不能阻断策略输入
            ws = 0.0
        if ws > 0.0:
            pts = [(float(a) * ws, float(b) * ws) for a, b in pts]
        return pts

    def _policy_frame_captured_at(self):
        """最近一帧左目的**采集时刻**（perf_counter 域）。

        给 PolicyRunner 当 obs_t 用（见 research/tools/policy_runner.py 里
        captured_at_source 的注释）：obs_t 在推理前打戳会让块的前几步永远不执行，
        而那一窗正好是模型的减速谷底。用真实采集时刻后，age 才如实反映「画面多旧」。
        """
        try:
            latest = self.navmesh.policy_frame()
        except Exception:  # noqa: BLE001
            return None
        if latest is None:
            return None
        _rgb, captured_at = latest
        return float(captured_at)

    def _policy_prev_frame(self):
        """距今 HISTORY_FRAME_S（当前 0.2 s）的左目 RGB；1 Hz 的 runner 自己采不到这个间隔。

        缺帧仍返回 None，由 PolicyRunner 回退成「当前帧复制一份」——该回退行为不变。
        """
        try:
            older = self.navmesh.policy_frame_before(HISTORY_FRAME_S)
        except Exception:
            return None
        if older is None:
            return None
        rgb, captured_at = older
        if time.perf_counter() - captured_at > _POLICY_FRAME_STALE_S:
            return None
        return rgb

    def _policy_world_scale(self) -> float:
        """追踪米 → 世界米的比例。拿不到就返回 0 = 调用方跳过换算。

        **为什么必须有这一步**（2026-11-02 实测）：训练集的 wp 来自
        `build_policy_dataset.build_dr_path`，它积分的是 **OSC VelocityX/Z**；
        用 17 场有 SLAM 位姿的录制交叉验证，「OSC 积分路程 / SLAM 位姿路程」
        在长程场次稳定在 0.73–0.75，正等于 world_scale —— 也就是
        **OSC 速度是世界米/秒，SLAM 位姿是追踪米**。
        而 `PathFollower.local_goal` 用的是网格弧长（追踪米），
        所以这里不换算的话，喂给模型的 |wp| 会比训练分布小约 25%，
        模型读到「我更近了」于是减速。
        """
        try:
            s = float(getattr(getattr(self.navmesh, "cfg", None), "world_scale", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0
        return s if 0.1 < s < 10.0 else 0.0

    def _policy_waypoint(self) -> tuple[float, float]:
        """机体系航点 (dx 右, dz 前)，**世界米**；没有路径时 (0,0) 触发停车门控。

        **wp 距离下限（2026-11-02 走停振荡的对策）**：路径侧向时投影点落在
        progress+lookahead 之前，实测 wp 恒在 0.5–0.6 m —— 训练数据里这个距离
        对应"人到航点前减速"的分布，模型于是持续输出"减速-停-再起"块
        （实测块形 [0.5, 0.1, -0.1×4, …, 0.9]，剖析 §15.20）。把喂给模型的
        距离抬到 ≥ lookahead（0.8，训练值），等价于永远喂"还在走、没到点"的
        状态；执行几何（转向对准、wp_stop 停车门）不受影响，它们用真实 wp。
        `VRC_POLICY_WP_MIN` 覆盖，0 关闭。
        """
        try:
            wp = self.navmesh.policy_waypoint()
        except Exception:
            return (0.0, 0.0)
        if wp is None:
            return (0.0, 0.0)
        # 追踪米 → 世界米（训练口径）。见 _policy_world_scale 的注释。
        # 轻量测试桩/旧注入对象可能没有这个可选换算钩子；缺失时保持原追踪米。
        scale_fn = getattr(self, "_policy_world_scale", None)
        try:
            ws = float(scale_fn()) if callable(scale_fn) else 0.0
        except Exception:  # noqa: BLE001 - 尺度拿不到不能阻断策略输入
            ws = 0.0
        if ws > 0.0:
            wp = (wp[0] * ws, wp[1] * ws)
        try:
            wp_min = float(os.getenv("VRC_POLICY_WP_MIN") or 0.8)
        except ValueError:
            wp_min = 0.8
        if wp_min > 0.0:
            d = math.hypot(wp[0], wp[1])
            if 1e-6 < d < wp_min:
                k = wp_min / d
                wp = (wp[0] * k, wp[1] * k)
        return wp

    def _policy_body(self) -> tuple[float, float]:
        """本体速度 (vx 右, vz 前) m/s：ZOH 保持最近一次回传，静默超 _POLICY_BODY_ZOH_S 按 0。

        **冷启动引导（2026-11-01 在线首跑发现的死锁）**：模型把 body 当主导特征 ——
        body=(0,0) 时即使航点 0.8 m 外也只输出 0.08 m/s（实测，静止帧 × 4 档 body 对照：
        0→0.08、0.5→0.44、0.8→1.97、1.0→0.72）。训练数据里人一直在走，body 与光流
        一致，模型学的是"先看我在不在走"。于是闭环冷启动死锁：
        body=0 → 输出≈0 → 角色不动 → body 仍=0。
        破法：OSC 静默且 navmesh 有任务（goto/explore）时，喂"wp 方向的期望速度"，
        角色一动 OSC 有回传就自动切回真实值。``VRC_POLICY_BODY_NUDGE=0`` 关闭。
        """
        osc = self.osc
        if osc is not None:
            try:
                feedback = osc.motion_feedback()
            except Exception:
                feedback = None
            if isinstance(feedback, Mapping) and feedback.get("available"):
                vx, vz = feedback.get("velocity_x"), feedback.get("velocity_z")
                if vx is not None and vz is not None:
                    self._policy_body_last = (float(vx), float(vz))
                    self._policy_body_at = time.perf_counter()
        last = self._policy_body_last
        osc_body = last if last is not None and time.perf_counter() - self._policy_body_at <= _POLICY_BODY_ZOH_S else None
        # 混合 VLA 不把 A* 航点伪造成身体速度；只喂真实 OSC 反馈，
        # A* 先验走 guidance_provider 的独立输入口。
        return body_feedback(osc_body, None, nudged=False)

    def _policy_prompt(self) -> str | None:
        """返回当前场次提示；读取失败时退化成无提示输入。"""
        with self._lock:
            value = self._policy_prompt_text
        return value or None

    def policy_prompt(self, text: Any = "") -> dict[str, Any]:
        """设置或清除 VLA 的场次级测试提示。"""
        if text is None:
            normalized = ""
        elif isinstance(text, (dict, list, tuple, set)):
            return {"accepted": False, "reason": "text must be a string", "prompt": self._policy_prompt_text}
        else:
            normalized = " ".join(str(text).replace("\x00", " ").split())
        if len(normalized) > 96:
            return {
                "accepted": False,
                "reason": "text must be at most 96 characters",
                "prompt": self._policy_prompt_text,
            }
        with self._lock:
            self._policy_prompt_text = normalized
        return {
            "accepted": True,
            "prompt": normalized,
            "active": bool(normalized),
            "scope": "persistent_until_cleared",
            "trained": False,
            "reason": "inference_only_prompt_experiment",
        }

    def policy_state(self) -> dict[str, Any]:
        """policy 头 A 的可观测性快照（调试端点 /policy/state）。

        dry-run 时代的排障全靠 policy_live.jsonl，真驱动模式下它不写盘，
        2026-11-01 在线首跑时"角色不动"查了半小时就是缺这一环。只读，不改状态。
        """
        head = self.policy_head
        configured = self._policy_configured
        policy_cfg = self.config.policy
        config_view = {
            "configured": configured,
            "enabled": self._policy_enabled(),
            "model_path": policy_cfg.model_path if configured else bool(os.getenv("VRC_POLICY_MODEL")),
            "run_path": policy_cfg.run_path if configured else bool(os.getenv("VRC_POLICY_RUN")),
            "dry_run": policy_cfg.dry_run if configured else _env_flag("VRC_POLICY_DRYRUN", default=True),
            "hz": policy_cfg.hz if configured else float(os.getenv("VRC_POLICY_HZ") or 1.0),
            "v_max": policy_cfg.v_max if configured else float(os.getenv("VRC_POLICY_VMAX") or 1.2),
            "stale_s": policy_cfg.stale_s if configured else None,
            "floor_frac": policy_cfg.floor_frac if configured else None,
        }
        if head is None:
            err = getattr(self, "_policy_start_error", None)
            with self._lock:
                prompt_text = self._policy_prompt_text
            return {"present": False,
                    "task_prompt": {
                        "text": prompt_text,
                        "active": bool(prompt_text),
                        "scope": "persistent_until_cleared",
                        "trained": False,
                    },
                    "config": config_view,
                    "reason": err or "policy head not started (VRC_POLICY_HEAD off?)",
                    "start_error": err}
        runner = getattr(head, "runner", None)
        executor = getattr(head, "executor", None)
        rt = getattr(head, "rt", None)
        with getattr(runner, "_lock", threading.Lock()):
            buf = list(getattr(runner, "buf", []) or [])
        last_frame_age = (time.perf_counter() - buf[-1][0]) if buf else None
        nav_frame = None
        try:
            latest = self.navmesh.policy_frame()
            if latest is not None:
                rgb, at = latest
                shape = getattr(rgb, "shape", None)
                nav_frame = {"age_s": round(time.perf_counter() - at, 3),
                             "shape": list(shape[:2]) if shape is not None else None}
        except Exception as exc:  # noqa: BLE001
            nav_frame = {"error": repr(exc)[:120]}
        try:
            guidance = self._policy_guidance()
        except Exception as exc:  # noqa: BLE001
            guidance = {"valid": False, "reason": repr(exc)[:120]}
        try:
            near = self._policy_near_obstacle()
        except Exception as exc:  # noqa: BLE001
            near = ("error", repr(exc)[:120])

        def _block_view(value: Any, limit: int | None = None) -> list[list[float | None]]:
            """把任意维度的动作块转换成可序列化的诊断数据。"""
            if value is None:
                return []
            try:
                rows = list(value)
            except TypeError:
                rows = [value]
            if limit is not None:
                rows = rows[:limit]

            result: list[list[float | None]] = []
            for row in rows:
                if isinstance(row, (str, bytes)):
                    values = [row]
                else:
                    try:
                        values = list(row)
                    except TypeError:
                        values = [row]
                normalized: list[float | None] = []
                for value_item in values:
                    try:
                        number = float(value_item)
                    except (TypeError, ValueError):
                        normalized.append(None)
                        continue
                    normalized.append(round(number, 3) if math.isfinite(number) else None)
                result.append(normalized)
            return result

        consumes_wp = not bool(getattr(rt, "drop_wp", False))
        consumes_path = bool(getattr(rt, "path_in", False))
        with self._lock:
            prompt_text = self._policy_prompt_text
        return {
            "present": True,
            "task_prompt": {
                "text": prompt_text,
                "active": bool(prompt_text),
                "scope": "persistent_until_cleared",
                "trained": False,
            },
            "config": config_view,
            "dry_run": bool(getattr(head, "dry_run", None)),
            "control_owner": "dry_run" if bool(getattr(head, "dry_run", None)) else "vla",
            # 兼容旧 UI：这个字段描述 A* 是否还能直接发动作；软引导不计入其中。
            "astar_control_enabled": bool(getattr(self.navmesh, "drive_enabled", True)),
            "astar_guidance": {
                "valid": bool(guidance.get("valid", False)),
                "requested": bool(guidance.get("requested", False)),
                "control_mode": guidance.get("control_mode", "unknown"),
                "t": time.perf_counter(),
                "waypoint": (list(guidance.get("waypoint"))
                              if isinstance(guidance.get("waypoint"), (tuple, list)) else [0.0, 0.0]),
                "path": (list(guidance.get("path"))
                          if isinstance(guidance.get("path"), (tuple, list)) else None),
                "confidence": guidance.get("confidence"),
                "pose_age_s": guidance.get("pose_age_s"),
                "sigma_m": guidance.get("sigma_m"),
                "reason": guidance.get("reason"),
                "source": guidance.get("source", "astar_navmesh"),
            },
            "input_contract": {
                "waypoint": consumes_wp,
                "path_shape": consumes_path,
                "path_available": bool(guidance.get("path")),
                "guidance_is_soft": True,
                "guidance_requested_only_for_memory_revisit": True,
                "guidance_confidence": True,
                "follower_speed_floor": False,
                "turn_owner": "vla",
            },
            "graph_enabled": bool(getattr(rt, "graph_enabled", None)),
            "graph_ready": bool(getattr(rt, "graph_ready", None)),
            "runner_alive": bool(getattr(getattr(runner, "_th", None), "is_alive", lambda: False)()),
            "executor_alive": bool(getattr(getattr(executor, "_th", None), "is_alive", lambda: False)()),
            "runner_hz": (1.0 / runner.period) if runner is not None and getattr(runner, "period", 0) else None,
            "frame_buffer_n": len(buf),
            "last_frame_age_s": round(last_frame_age, 3) if last_frame_age is not None else None,
            "nav_frame": nav_frame,
            "guide_waypoint_now": (
                list(guidance.get("waypoint"))
                if isinstance(guidance.get("waypoint"), (tuple, list)) else None),
            "body_now": list(self._policy_body()),
            "runner_block_last": (
                _block_view(getattr(runner, "_last_block", None), 2)
                if runner is not None else None),
            "runner_meta_last": getattr(runner, "_last_meta", None),
            "runner_block_all": (
                _block_view(getattr(runner, "_last_block", None))
                if runner is not None else None),
            "near_obstacle": {"stop": near[0], "points": near[1]} if isinstance(near, tuple) else near,
            "executor_block_last": (
                _block_view(getattr(executor, "_block", None), 2)
                if executor is not None else None),
            "executor_obs_age_s": round(time.perf_counter() - float(getattr(executor, "_obs_t", 0)), 3)
                                  if executor is not None else None,
            "executor_gated": bool(getattr(executor, "_gated", None)),
            "executor_n_submits": int(getattr(executor, "n_submits", 0)),
            "executor_n_sends": int(getattr(executor, "n_sends", 0)),
            "executor_n_halts": int(getattr(executor, "n_halts", 0)),
            "executor_last_send": (
                list(getattr(executor, "last_send", None)) if getattr(executor, "last_send", None) else None),
            "runner_n_on_block": int(getattr(runner, "n_on_block", 0)),
            "runner_dry_run": bool(getattr(runner, "dry_run", None)),
            "nav_policy_hold": bool(getattr(self.navmesh, "policy_hold", False)),
            # body 输出 disabled 时摇杆 overlay 不出网（轴"成功"却不动），必须可见：
            "body_output": {
                "state": (self.scheduler.snapshot().get("state") if self.scheduler else None),
                "output_enabled": (self.scheduler.snapshot().get("output_enabled") if self.scheduler else None),
            },
        }

    def _policy_near_obstacle(self) -> tuple[bool, int]:
        """navmesh 感知线程每帧算好的近距障碍判据 ``(stop, n_pts)``。

        见 ``nav_online.near_obstacle``：正面 ``stop_ahead_m``（默认 0.7 世界米）×
        ``2·radius_m`` 内的障碍点数 ≥ ``stop_min_pts``（默认 40）为 True。
        读不到就当没有，绝不让感知缺失变成「策略不动」的理由 —— 那是 executor
        的 stale 阈值该管的事。
        """
        try:
            return (bool(self.navmesh._near), int(self.navmesh._near_pts))
        except Exception:  # noqa: BLE001
            return (False, 0)

    def _policy_body_watchdog_loop(self) -> None:
        """body 输出闩锁的自动恢复（§15.21）。

        只在**有活跃导航任务**（navmesh mode 不在 idle/done）且无 drive_block 时
        才恢复 —— 无任务时闩锁多半是人有意的急停/停用，保持闩锁语义。恢复顺序：
        latched → reset（回 idle）；idle→disabled 过渡中的 disabled → enable。
        """
        stop = getattr(self, "_policy_body_watchdog_stop", None)
        while stop is not None and not stop.wait(2.0):
            try:
                scheduler = self.scheduler
                if scheduler is None:
                    continue
                state = scheduler.snapshot().get("state")
                if state not in {"stopped_latched", "fault_latched", "disabled"}:
                    continue
                try:
                    mode = self.navmesh.status().get("mode", "idle")
                except Exception:
                    mode = "idle"
                if mode in ("idle", "done"):
                    continue
                block = None
                try:
                    block = self._drive_block_reason()
                except Exception:
                    block = "watchdog_block_probe_failed"
                if block is not None:
                    continue
                if state in {"stopped_latched", "fault_latched"}:
                    result = self.submit("reset")
                    if self.logger:
                        self.logger.warning(
                            "policy watchdog: body was %s with active nav task (%s); "
                            "reset -> %s", state, mode, result.get("state"))
                    state = result.get("state") or "idle"
                if state == "disabled":
                    result = self.submit("enable")
                    if self.logger:
                        self.logger.info("policy watchdog: re-enabled body -> %s",
                                         result.get("state"))
            except Exception as exc:  # noqa: BLE001 - 看门狗自身绝不能死
                if self.logger:
                    self.logger.warning("policy watchdog error: %s", exc)

    def _policy_send_turn(self, delta_deg: float) -> bool:
        """policy 执行环的转向出口。单位**度**，走 scheduler 的 turn 通道。

        为什么不能走摇杆：VRChat 的 move 轴只有 Horizontal/Forward 两维，
        转向由独立的 Look/转身通道承担（见 _navigator_send_turn）。
        """
        try:
            autonomy = self.autonomy.snapshot()
            if not autonomy.get("armed"):
                return False
            if bool(autonomy.get("goal")):
                # 与 _policy_send_axes 同一条互斥：LocalNavigator 正在发转向。
                return False
            return bool(self._navigator_send_turn(float(delta_deg)))
        except Exception:  # noqa: BLE001 - 转向失败不能拖垮运动出口
            return False

    def _policy_send_axes(self, x: float, y: float, duration_ms: int) -> bool:
        """policy 执行环出口：只走 AnyaDance 主通道（左摇杆），绝不回退 OSC。"""
        try:
            autonomy = self.autonomy.snapshot()
            if not autonomy.get("armed"):
                return False
            if bool(autonomy.get("goal")):
                # LocalNavigator 正为语义目标发轴：同一组摇杆不能有两个发送方。
                return False
        except Exception:
            return False
        return self._navigator_send_axes("left", x, y, int(duration_ms))

    def _navigator_motion_feedback(self) -> dict[str, Any]:
        """VRChat 内置 Velocity 参数，供导航器判断是否顶着墙推摇杆。

        OSC 桥没起来时返回 available=false：导航器会退回到没有卡墙判据的旧行为，
        而不是把「读不到」当成「速度为零」然后立刻停车。
        """
        osc = self.osc
        if osc is None:
            return {"available": False, "reason": "osc_unavailable"}
        feedback = osc.motion_feedback()
        timeline = self.action_timeline
        if timeline is not None and isinstance(feedback, dict):
            # 只有"确实取到了样本"才记速度；读不到时保持上一份，绝不把不可观测
            # 当成零速度写进路线历史。
            if feedback.get("available"):
                timeline.note_velocity(
                    feedback.get("velocity_x") or 0.0,
                    0.0,
                    feedback.get("velocity_z") or 0.0,
                )
        return feedback

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

    def pulse_jump(self, hold_ms: Any) -> tuple[bool, str | None]:
        """跳跃走 VRChat OSC ``/input/Jump``，不经 ``input.primary`` 路由。

        之前这里接的是虚拟 Index 的 A 键，但 VRChat 的跳跃绑定是否落在 A 上完全取决于
        用户的按键配置，实机无法确认——和 ``set_turn`` 里 ``/input/LookHorizontal``
        那个坑同类。``/input/Jump`` 是语义地址，跟按键绑定无关，是唯一可靠的跳跃手段。

        按下与释放之间必须有真实间隔：VRChat 只在按下沿触发起跳，同一拍内按下又释放
        会被丢掉。这里用 ``send_button`` 的整数 1/0 通道，急停时 ``_BUTTON_ADDRESSES``
        的统一释放兜底，按钮不会卡住。
        """
        if self.osc is None:
            return False, "VRChat OSC bridge is not initialized"
        try:
            normalized_hold = _osc_hold_ms(hold_ms, self.config.vrchat_osc.input_pulse_ms)
        except ValueError as exc:
            return False, str(exc)
        pressed, reason = self.osc.send_button("jump", True)
        if not pressed:
            return False, reason or "VRChat OSC jump press failed"
        scheduled = self.osc.schedule_button_release("jump", delay_s=normalized_hold / 1000.0)
        if not scheduled:
            # 排不上释放就地同步松开：宁可跳得短，也不能留着按住的跳跃键。
            self.osc.send_button("jump", False)
            return False, "VRChat OSC jump release could not be scheduled"
        return True, None

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
        def record(sent: bool) -> None:
            if self.action_timeline is not None:
                # 路线执行和手动 OSC 都统一进入动作时间轴；失败也要留痕。
                self.action_timeline.record_command(
                    forward=normalized_vertical, strafe=normalized_horizontal, sent=sent
                )
        set_axes = getattr(self.osc, "set_axes", None)
        if callable(set_axes):
            result = set_axes(
                {
                    "move_vertical": normalized_vertical,
                    "move_horizontal": normalized_horizontal,
                },
                duration_s,
            )
            record(bool(result[0]))
            return result
        vertical_result = self.osc.set_axis("move_vertical", normalized_vertical, duration_s)
        if not vertical_result[0]:
            rollback = getattr(self.osc, "stop_axes", None)
            if callable(rollback):
                rollback(("move_vertical", "move_horizontal"))
            else:
                self.osc.stop_all_axes()
            record(False)
            return False, vertical_result[1] or "VRChat OSC locomotion send failed"
        horizontal_result = self.osc.set_axis("move_horizontal", normalized_horizontal, duration_s)
        if horizontal_result[0]:
            record(True)
            return True, None
        # 部分移动命令不安全：释放两个移动轴，避免调用方收到看似成功的半条命令。
        rollback = getattr(self.osc, "stop_axes", None)
        if callable(rollback):
            rollback(("move_vertical", "move_horizontal"))
        else:
            self.osc.stop_all_axes()
        record(False)
        return False, horizontal_result[1] or "VRChat OSC locomotion send failed"

    def set_turn(
        self,
        horizontal: Any,
        duration_ms: Any,
        source: str = "set_turn",
    ) -> tuple[bool, str | None]:
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
        recorder = getattr(self, "_record_turn_submission", None)
        if recorder is not None:
            recorder(source, delta_deg, bool(result.get("accepted")))
        if self.action_timeline is not None:
            self.action_timeline.record_turn(yaw_delta=delta_deg, sent=bool(result.get("accepted")))
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
        return self.set_turn(normalized_x, normalized_duration, source="axes:right_fallback")

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
                    source="batch:turn",
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
        """目录清单走 catalog()：只 stat 文件、复用已缓存的摘要，不解析任何片。

        这条路径唯一的消费者是每秒刷新一次的调试面板（``/clips/list`` ←
        ``RemoteClipLibrary.catalog()``）。``ClipLibrary.list()`` 会 ``load()``
        目录里的每一个片，而常驻缓存只有 2 个槽位，于是每次刷新都把整个目录
        重新解析一遍——20 个 10 秒片实测 2.5 s，面板刷新间隔才 1 s。

        返回结构与 ``list()`` 完全一致；差别只在没被解析过的片的
        ``duration_s``/``frame_count`` 为 null 且 ``indexed=false``，面板已经按
        「待首次解析」渲染这种状态。

        代价说明：损坏的片在第一次被解析之前会落在 ``clips``（``indexed=false``）
        而不是 ``invalid_clips``——判断 JSON 坏没坏必须读文件，而不读文件正是这里
        提速的全部来源。一旦有任何路径 ``load()`` 过它，错误会进摘要缓存，之后
        每次 catalog() 都照常报进 ``invalid_clips``。把它显示成「还没解析」而不是
        「有效」，是诚实的降级；为此每秒重读整个目录不值得。
        """
        return self.clip_library.catalog()

    def semantic_express(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """在后端选择 VMD，然后提交生成的动作。"""
        # 只 strip 不 lower：EXPRESSION_PROFILES 的键全是小写，lower 对行为路径没有
        # 增益，却可能改变 VMD 目录里按原样登记的 intent 名的匹配结果。
        intent = str(params.get("intent") or "").strip()
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
