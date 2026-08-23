"""VRChat 后端的可选本地视觉感知。

采集层和世界状态层不依赖具体模型格式。本模块提供一个小型、保守的适配器：
配置模型后可运行标准 OpenVINO IR/ONNX 目标检测器；未安装 OpenVINO 时，ONNX
模型可回退到 OpenCV DNN。对于尚未部署模型的机器，也支持显式启用的 OpenCV
HOG 人形检测器。HOG 路径会标记为 ``degraded``，不会声称识别了身份或除
``person`` 之外的物体类别。

没有模型时状态为 *unavailable*。加载或推理失败时，本模块不会创建占位实体。
这对导航器很重要：过期或未知的感知结果必须停止移动，不能变成虚构的障碍物
或目标。
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .vision import VisionObservation
from .avatar_identity import AvatarIdentityRegistry
from .world_state import stable_track_entity_id


# 可能被载入的 OpenMP 运行时。numpy 的 scipy-openblas 用 OpenMP 线程模型，
# 具体是哪个 DLL 取决于 wheel 怎么构建，所以按序探测而不是假定一个。
_OPENMP_RUNTIME_DLLS: tuple[str, ...] = ("vcomp140", "libiomp5md", "libomp", "vcomp")

# 第一次成功的收口结果。OpenMP 是进程级的，所以「有没有限住」也是进程级的事实：
# 之后 BackendService 之外的调用方（检测器自己、基准脚本）再调一次必然报
# too_late=True，用它覆盖记录会让 status() 反过来污蔑一次成功的收口。
_OPENMP_STATE: dict[str, Any] = {}


def cap_openmp_threads(threads: int) -> dict[str, Any]:
    """收住**进程级** OpenMP 线程池，返回实际做到了什么。

    这里限的不是 YOLO 自己的线程池——那个由 ``SessionOptions`` 管（见
    ``OpenVinoLocalDetector._onnx_session_options``）。numpy 的 BLAS
    （scipy-openblas，OpenMP 线程模型）是**另一个**池子：采集、缩放、前后处理
    全走它，它默认按逻辑核数开线程**并且自旋等待**。

    实测（20 核，1161x766 采集区，检测器换成空实现，所以下面这些开销里没有一次
    推理）：

    | 设置 | 占用 | 吞吐 |
    |---|---|---|
    | 默认 | 7.23 核 | 6.8 帧/秒 |
    | ``OMP_NUM_THREADS=2`` | 0.87 核 | 6.8 帧/秒 |
    | ``OMP_WAIT_POLICY=PASSIVE`` | **0.11 核** | 6.9 帧/秒 |

    三行吞吐一模一样——那 7 个核**全部**是空转自旋，不是计算。所以主力是
    ``OMP_WAIT_POLICY``，线程数上限只是顺带；这也是为什么限了之后单次推理延迟反而
    从 469ms 降到 312ms：少了 18 个线程抢核。

    两个都只能靠环境变量：OpenMP 运行时初始化时读一次，之后改环境变量没人看。
    所以必须在 numpy 被导入**之前**调用。本仓里 numpy 全是函数内惰性导入
    （``import numpy as np`` 都在函数体内），所以从 ``BackendService.__init__``
    开头调用仍然赶得上。

    赶不上的场景是插件跑在宿主进程里、宿主已经先导入了 numpy。那时退到 ctypes 调
    ``omp_set_num_threads``：能改线程数（→0.86 核），改不了等待策略。用
    ``setdefault`` 是为了让显式设了这两个变量的运维配置说了算。
    """
    result: dict[str, Any] = {
        "requested": int(threads),
        "env_applied": False,
        "wait_policy": None,
        "runtime_dll": None,
        "too_late": False,
        "error": None,
    }
    if threads <= 0:
        return result
    if _OPENMP_STATE.get("env_applied"):
        # 已经在 numpy 之前收住过了。再走一遍只会得到 too_late=True，把已经成立
        # 的事实记成失败，所以直接复述第一次的结果。
        return dict(_OPENMP_STATE)
    # numpy 在不在 sys.modules 里，是「OpenMP 是否已经初始化」的近似判据。宁可
    # 近似也要记下来：环境变量设晚了是静默失效的，外面必须看得出差别。
    result["too_late"] = "numpy" in sys.modules
    try:
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        # ACTIVE（很多运行时的默认）让空闲线程自旋抢核；PASSIVE 让它们睡。
        # 实测吞吐不变而占用降到 1.5%，所以这不是「省 CPU 换性能」的权衡。
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        result["env_applied"] = not result["too_late"]
        result["wait_policy"] = os.environ.get("OMP_WAIT_POLICY")
    except Exception as exc:  # pragma: no cover - environ 赋值几乎不会失败
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    if result["too_late"]:
        for name in _OPENMP_RUNTIME_DLLS:
            try:
                dll = ctypes.CDLL(name)
                dll.omp_set_num_threads(ctypes.c_int(int(threads)))
                result["runtime_dll"] = name
                break
            except Exception:
                # 探测式加载：这个名字不存在就试下一个。全都失败只意味着没有
                # OpenMP 运行时可限，不该让感知因此失败。
                continue
    _OPENMP_STATE.update(result)
    return result


def openmp_thread_count() -> int | None:
    """返回 OpenMP 当前的最大线程数，取不到就返回 ``None``。

    这是给 ``status()`` 用的**实测值**而不是配置值：配置只说明意图，这个说明
    结果。上限设晚了、或者运维显式覆盖了环境变量，只有这里看得出来。
    """
    for name in _OPENMP_RUNTIME_DLLS:
        try:
            dll = ctypes.CDLL(name)
            dll.omp_get_max_threads.restype = ctypes.c_int
            return int(dll.omp_get_max_threads())
        except Exception:
            continue
    return None


# 帧内 NMS 的候选上限。裸 YOLO 输出可能有上千个框过阈值，而贪心抑制是
# O(n^2)；先按置信度截到这个上限再抑制，把最坏情况钉死在固定开销内。取值
# 远大于 max_detections 的默认值，正常场景不会触及。
_NMS_CANDIDATE_LIMIT = 256


# COCO 名称适用于常见的 YOLOX/YOLO 导出格式。调用方可以提供标签文件替换它们；
# 未知索引会保留为明确的 ``class_<n>`` 标签，不会被静默映射成错误物体。
_COCO_LABELS: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball",
    "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
)


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, _finite(value, low)))


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    area_second = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union > 1e-9 else 0.0


@dataclass
class _Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_index: int = -1
    track_id: int | None = None


@dataclass
class _Track:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    last_seen: float


class _IoUTracker:
    """用于检测器输出的小型有界跟踪器。

    跟踪器按类别匹配，刻意不使用外观特征向量。这样可以让延迟和隐私边界
    保持可预测，同时避免实体 ID 在每一帧都变化。
    """

    def __init__(self, *, iou_threshold: float = 0.25, ttl_s: float = 1.5) -> None:
        self.iou_threshold = min(0.95, max(0.0, float(iou_threshold)))
        self.ttl_s = min(10.0, max(0.1, float(ttl_s)))
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[_Detection], now: float) -> None:
        active = {
            track_id: track
            for track_id, track in self._tracks.items()
            if now - track.last_seen <= self.ttl_s
        }
        self._tracks = active
        unmatched = set(active)
        # 先匹配置信度最高的检测结果；两个物体短暂重叠时，分配会更稳定。
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            candidates = [
                track_id for track_id in unmatched
                if active[track_id].label == detection.label
            ]
            # 如果分类器短暂改变类别（例如无法区分 Avatar 和道具），且检测框
            # 重叠很强，则保留原跟踪。相同类别的匹配仍优先，避免仅因物体相近
            # 就把附近的两个物体合并。
            candidate_threshold = self.iou_threshold
            if not candidates:
                candidates = list(unmatched)
                candidate_threshold = max(0.5, self.iou_threshold)
            best_id: int | None = None
            best_iou = candidate_threshold
            for track_id in candidates:
                overlap = _iou(active[track_id].bbox, detection.bbox)
                if overlap >= best_iou:
                    best_id, best_iou = track_id, overlap
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            else:
                unmatched.discard(best_id)
            detection.track_id = best_id
            self._tracks[best_id] = _Track(best_id, detection.label, detection.bbox, now)


class _LabelLoadError(Exception):
    """标签文件存在但格式无法识别时抛出。"""


def _load_labels(path: str | Path | None) -> tuple[str, ...]:
    """从文件加载标签列表。

    支持的格式：
    - JSON 数组：``["person", "cat", ...]``
    - JSON 对象（ultralytics yaml 风格）：``{"names": ["person", ...]}``
      或 ``{"names": {"0": "person", ...}}``
    - JSON 对象（deepghs booru_yolo 风格）：``{"labels": ["head", ...], ...}``
    - JSON 对象（整数键映射）：``{"0": "person", "1": "cat", ...}``
    - 纯文本：每行一个标签，``#`` 开头为注释

    未提供路径时返回内置 COCO 标签。路径已提供但格式无法识别时抛出
    ``_LabelLoadError``，而不是静默回落 COCO——静默回落会让模型用错误的
    标签维度解码输出，产生满置信度的垃圾检测（见 deepghs/booru_yolo 的
    meta.json 案例）。
    """
    if not path:
        return _COCO_LABELS
    candidate = Path(path)
    try:
        raw = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise _LabelLoadError(f"cannot read labels file {candidate}: {exc}") from exc
    try:
        if candidate.suffix.lower() == ".json":
            value = json.loads(raw)
            if isinstance(value, Mapping):
                # {"labels": [...]} — deepghs/booru_yolo meta.json 格式
                if "labels" in value and isinstance(value["labels"], (list, tuple)):
                    seq = value["labels"]
                    return tuple(str(item).strip() or f"class_{i}" for i, item in enumerate(seq))
                # {"names": [...]} 或 {"names": {"0": "person", ...}}
                names = value.get("names", value)
                if isinstance(names, Mapping):
                    try:
                        ordered = sorted(names.items(), key=lambda item: int(item[0]))
                    except (ValueError, TypeError) as exc:
                        raise _LabelLoadError(
                            f"labels file {candidate}: cannot sort keys as integers"
                        ) from exc
                    return tuple(str(item[1]).strip() or f"class_{item[0]}" for item in ordered)
                value = names
            if isinstance(value, (list, tuple)):
                return tuple(str(item).strip() or f"class_{i}" for i, item in enumerate(value))
            raise _LabelLoadError(
                f"labels file {candidate}: JSON root must be a list or object, got {type(value).__name__}"
            )
        # 纯文本格式
        lines = tuple(
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if lines:
            return lines
        raise _LabelLoadError(f"labels file {candidate}: file is empty or contains only comments")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _LabelLoadError(f"labels file {candidate}: {exc}") from exc


# 不接受 INFERENCE_NUM_THREADS 的设备。这个属性限的是 CPU 线程池，这些插件
# 里没有对应物，喂过去实测直接抛 RuntimeError 而不是被忽略。
_OPENVINO_ACCELERATORS = frozenset({"GPU", "NPU"})


class OpenVinoLocalDetector:
    """采用保守输出归一化策略的 OpenVINO 检测器。

    支持常见 YOLO 输出 ``[cx,cy,w,h,obj,class_scores...]``、NMS 后输出
    ``[x1,y1,x2,y2,score,class]``，以及 OpenVINO SSD 输出
    ``[image_id,class,score,xmin,ymin,xmax,ymax]``。坐标以归一化屏幕框提供。
    仍可通过 ``infer`` 注入特定模型图；这保留了原有适配器接口，也适合深度或
    OCR 模型包。
    """

    name = "openvino"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        labels_path: str | None = None,
        device: str = "AUTO",
        onnxruntime_cuda: str = "auto",
        onnxruntime_cuda_device_id: int = 0,
        confidence_threshold: float = 0.35,
        input_width: int = 640,
        input_height: int = 640,
        horizontal_fov_deg: float = 90.0,
        max_detections: int = 64,
        nms_iou_threshold: float = 0.45,
        min_box_ratio: float = 0.02,
        min_box_width_ratio: float | None = None,
        min_box_height_ratio: float | None = None,
        track_iou_threshold: float = 0.25,
        track_ttl_s: float = 1.5,
        identity_reid_enabled: bool = True,
        identity_reid_similarity: float = 0.90,
        identity_reid_margin: float = 0.04,
        identity_reid_retention_s: float = 1800.0,
        identity_reid_max_identities: int = 128,
        intra_op_threads: int = 2,
        fallback_backend: str = "none",
        infer: Callable[[Any], Mapping[str, Any] | Any] | None = None,
        openvino_core: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._model_path = str(model_path).strip() if model_path else ""
        self._labels_path = str(labels_path).strip() if labels_path else ""
        try:
            self._labels = _load_labels(self._labels_path or None)
        except _LabelLoadError as _exc:
            # 标签文件无法识别时回落 COCO，但把原因记录到状态，避免静默吞掉
            # 满置信度的垃圾检测（错误的标签维度会让解码器命中错误的分支）。
            self._labels = _COCO_LABELS
            self._label_load_error: str = str(_exc)
        else:
            self._label_load_error = ""
        self._device = str(device or "AUTO").strip() or "AUTO"
        cuda_mode = str(onnxruntime_cuda or "auto").strip().lower() or "auto"
        if cuda_mode not in {"auto", "prefer", "disabled"}:
            raise ValueError("onnxruntime_cuda must be auto, prefer, or disabled")
        self._onnx_cuda_mode = cuda_mode
        self._onnx_cuda_device_id = min(31, max(0, int(onnxruntime_cuda_device_id)))
        self._confidence_threshold = min(1.0, max(0.0, float(confidence_threshold)))
        self._input_width = min(4096, max(32, int(input_width)))
        self._input_height = min(4096, max(32, int(input_height)))
        self._horizontal_fov_deg = min(180.0, max(1.0, float(horizontal_fov_deg)))
        self._max_detections = min(512, max(1, int(max_detections)))
        # 0 会把所有同类框合成一个，1 等于完全不抑制；两端都不是有用的部署
        # 取值，因此钳到留有余量的区间内。
        self._nms_iou_threshold = min(0.95, max(0.1, float(nms_iou_threshold)))
        # 框的宽/高占画面的最小比例。实测真实帧里出现过 27×27 px 的高分假阳性
        # （1920 宽下约 1.4%），它们会被跟踪器当成真人并污染导航的距离闭环。
        #
        # 宽高分开：站立的人在画面里是高而窄的，单一阈值下总是宽度先卡。0.02
        # 的共用阈值在 1920 宽下要求最小宽 38 px，按 2:1~3:1 的人体长宽比反推，
        # 能进入世界的最小 apparent_height 已经有 7%~11%——而导航器的目标是
        # 0.55，等于把「房间对面的人」和噪点一起裁掉了。默认宽阈值更松，让
        # 高度成为主判据；两条边仍然都要过关，墙缝和 UI 边框照样被挡住。
        #
        # ``min_box_ratio`` 保留为两轴的共同默认值（旧配置与旧调用方仍然有效），
        # 显式传入的每轴阈值优先。0 表示关闭对应轴的过滤。
        shared_ratio = min(0.5, max(0.0, _finite(min_box_ratio, 0.02)))
        if min_box_width_ratio is None:
            width_ratio = shared_ratio
        else:
            width_ratio = min(0.5, max(0.0, _finite(min_box_width_ratio, shared_ratio)))
        if min_box_height_ratio is None:
            height_ratio = shared_ratio
        else:
            height_ratio = min(0.5, max(0.0, _finite(min_box_height_ratio, shared_ratio)))
        self._min_box_width_ratio = width_ratio
        self._min_box_height_ratio = height_ratio
        self._tracker = _IoUTracker(iou_threshold=track_iou_threshold, ttl_s=track_ttl_s)
        self._identity_registry = AvatarIdentityRegistry(
            enabled=identity_reid_enabled,
            similarity_threshold=identity_reid_similarity,
            similarity_margin=identity_reid_margin,
            retention_s=identity_reid_retention_s,
            max_identities=identity_reid_max_identities,
        )
        # 推理线程上限。0 表示不设置，沿用运行时自己的默认（CPU EP 会开到物理
        # 核数）——留这个出口给需要裸速度的基准测量，部署配置不该用它。
        self._intra_op_threads = min(32, max(0, int(intra_op_threads)))
        # 上限有没有真的设上去，要和「配了多少」分开记：运行时可能不认这个旋钮，
        # 此时我们宁可满线程跑也不让检测器整体失败，那就必须能从外面看出差别。
        self._thread_cap_applied = False
        self._thread_cap_error: str | None = None
        # OpenMP（numpy BLAS）那一侧的收口结果。和上面两个分开：ORT 的池和
        # numpy 的池是两个东西，实测大头在后者，混成一个字段就没法定位了。
        self._openmp: dict[str, Any] = {}
        self._infer = infer
        # 模型推理和跟踪器都不是默认线程安全的；旁路调用也必须按帧串行化，
        # 避免实体 ID 与帧计数被并发破坏。
        self._infer_lock = threading.Lock()
        self._compiled_model: Any | None = None
        self._onnx_session: Any | None = None
        self._opencv_net: Any | None = None
        self._input_name: Any | None = None
        self._input_layout = "NCHW"
        self._runtime = "injected" if infer is not None else "none"
        # 保持历史注入适配器的来源命名空间稳定，同时在 status() 中单独暴露
        # 实际运行时。
        self._source_name = "openvino" if infer is not None else "none"
        self._available = infer is not None
        self._degraded = False
        self._last_error: str | None = None
        # 设备协商的结果要能从 /perception 读到。「配了 AUTO」和「最后跑在
        # GPU 上」是两件事，混成一个字段就没法判断加速有没有真的生效——而
        # 两者的延迟差 22 倍，静默落回 CPU 会表现为整个循环变慢却无人知晓。
        self._resolved_device: str | None = None
        self._device_fallbacks: tuple[str, ...] = ()
        self._openvino_error: str | None = None
        self._core_devices: tuple[str, ...] = ()
        # ORT GPU wheel 与 CUDA/cuDNN 都是可选依赖。仅在真的轮到 CUDA 候选时
        # 才导入和探测；结果必须能从状态面区分「没探测」「wheel 不带 provider」
        # 和「provider 在但动态库/Session 初始化失败」。
        self._onnx_cuda_probed = False
        self._onnx_available_providers: tuple[str, ...] = ()
        self._onnx_session_providers: tuple[str, ...] = ()
        self._onnx_cuda_error: str | None = None
        self._uncertainties: list[str] = []
        self._frames = 0
        self._last_inference_ms: float | None = None

        if infer is None:
            self._initialize_model(openvino_core)
        if not self._available and str(fallback_backend).strip().lower() in {"opencv_hog", "hog", "hog_person"}:
            self._initialize_hog()

    def _initialize_model(self, supplied_core: Any | None) -> None:
        if not self._model_path:
            self._last_error = "OpenVINO model_path is not configured"
            return
        path = Path(self._model_path)
        if path.is_dir():
            candidates = sorted(path.glob("*.xml")) + sorted(path.glob("*.onnx"))
            path = candidates[0] if candidates else path
        if not path.exists() or not path.is_file():
            self._last_error = f"OpenVINO model does not exist: {path}"
            return
        # 曾经这里对 .onnx 无条件先走 ONNX Runtime 并直接 return，理由是
        # 「OpenVINO wheel 在冻结宿主里根本不存在」。装上 wheel 后那个前提就
        # 失效了，而短路的代价很大：本机 ORT 只有 CPU 提供者（装的是基础包，
        # providers 仅 Azure/CPU），而 CPU 是 Xeon E5-2666 v3——2014 年的
        # Haswell，无 AVX-512。实测 person_detect_v1.3_s @640：
        #
        #   ORT CPU / 2 线程        258 ms   <- 这条短路选中的路径
        #   OpenVINO CPU / 2 线程   329 ms
        #   OpenVINO GPU (Arc A770)  11.7 ms  <- 快 22 倍，整卡驻留只 +79 MB
        #
        # 所以顺序反过来。ONNX 的完整优先级是：OpenVINO NPU/GPU → 可选的
        # ORT CUDA → OpenVINO CPU → ORT CPU → OpenCV。这样 NVIDIA-only 机器
        # 不会因为 OpenVINO CPU 先成功而永远绕过 CUDA，同时也保留 Intel
        # 加速器的既有优先级。注入 core 时同样走这里，只是跳过自己 import。
        #
        # OpenMP 收口必须在这里、在**任何**运行时被导入之前做。原来它挂在
        # _initialize_onnxruntime 里，而 ORT 曾经是 .onnx 的第一站，所以顺带
        # 就收住了。现在 OpenVINO 抢在前面并直接 return，那个调用点再也到不了：
        # 实测检测器 CPU 占用从 1.2 核涨回 3~6 核，全是 numpy BLAS 的自旋等待
        # （见 plugin.toml 里 detector_threads 那段——空转 7 核而吞吐不变）。
        self._cap_openmp_before_import()
        core = supplied_core
        model: Any | None = None
        if core is None:
            try:
                from openvino import Core  # type: ignore[import-not-found]

                core = Core()
            except Exception as exc:
                # wheel 没装是预期情况（别人的机器、冻结宿主），不是故障。
                self._openvino_error = f"{type(exc).__name__}: {exc}"[:200]
                core = None
        if core is not None:
            try:
                self._core_devices = tuple(str(d) for d in core.available_devices)
            except Exception:
                # 注入的测试替身通常没有这个属性。留空表示「不知道有哪些设备」，
                # 候选列表会退化成照单全试——比假装设备不存在要好。
                self._core_devices = ()
            try:
                model = core.read_model(model=str(path))
            except Exception as exc:
                self._openvino_error = f"read_model: {type(exc).__name__}: {exc}"[:200]
                model = None

        is_onnx = path.suffix.lower() == ".onnx"
        openvino_candidates = self._openvino_device_candidates() if model is not None else []
        accelerator_candidates = [
            item for item in openvino_candidates
            if item.split(".", 1)[0].upper() in _OPENVINO_ACCELERATORS
        ]
        remaining_candidates = [
            item for item in openvino_candidates if item not in accelerator_candidates
        ]

        if is_onnx and self._onnx_cuda_mode == "prefer":
            self._initialize_onnxruntime(path, use_cuda=True)
            if self._available:
                return

        if model is not None and accelerator_candidates:
            if self._compile_openvino(core, model, accelerator_candidates):
                return

        # 只在 AUTO 模式探测可选 CUDA；显式指定 OpenVINO CPU/GPU/NPU 时尊重
        # 调用方。provider 缺失或 Session 初始化失败都只记录并继续 CPU 回落。
        if is_onnx and self._onnx_cuda_mode == "auto" and self._device.upper() == "AUTO":
            self._initialize_onnxruntime(path, use_cuda=True)
            if self._available:
                return

        if model is not None and remaining_candidates:
            if self._compile_openvino(core, model, remaining_candidates):
                return

        # OpenVINO 与 CUDA 都不通。ONNX 还有两个 CPU 后备解释器，IR XML 没有——
        # 用 cv2 去读 XML 只会得到虚假的成功状态。
        if is_onnx:
            self._initialize_onnxruntime(path, use_cuda=False)
            if self._available:
                return
            self._initialize_opencv_dnn(path)
        if not self._available and not self._last_error:
            self._last_error = (
                f"no runtime could load {path.name}"
                + (f"; openvino: {self._openvino_error}" if self._openvino_error else "")
            )

    def _openvino_device_candidates(self) -> list[str]:
        """按优先级排出要试的设备，CPU 兜底。

        配置里的 ``AUTO`` 不能直接用：实测 ``EXECUTION_DEVICES=['(CPU)']``，
        AUTO 会先把首帧放在 CPU 上、GPU 在后台慢慢编译。对一个感知循环来说
        那等于长时间拿不到 GPU 的 11.7 ms。所以 AUTO 在这里被展开成「显式挨
        个试」，而不是交给 AUTO 插件去调度。

        只在**确实知道**有哪些设备时才展开。注入的 core（测试替身、自带运行时
        的调用方）通常没有 ``available_devices``，那种情况下照配置原样传过去：
        凭空试 NPU/GPU 等于对一个不了解的运行时瞎猜设备名，而它可能来者不拒，
        于是我们会"成功"编译到一个根本不存在的加速器上。
        """
        configured = self._device.strip() or "AUTO"
        available = [d for d in self._core_devices if d]
        if not available:
            return [configured]

        def devices(root: str) -> list[str]:
            # Core 会在同类设备超过一个时返回 GPU.0/GPU.1 这类完整名字。
            # ``GPU`` 只是 GPU.0 的别名，若在这里压成根名称，后面的卡永远
            # 没机会参与回落。因此保留运行时给出的具体标识和原始顺序。
            return [d for d in available if d.split(".", 1)[0].upper() == root]

        cpu_devices = devices("CPU") or ["CPU"]
        if configured.upper() != "AUTO":
            # 显式配了具体设备时尊重它，但仍然在后面挂上 CPU：用户要求的是
            # 「用这块加速器」，不是「没有加速器就整个瞎掉」。
            order = [configured]
        else:
            # NPU 排在 GPU 前：有 NPU 的机器上它功耗低得多，而这个模型小到
            # 两者都远快于 CPU，此时省电比省几毫秒更值。
            order = devices("NPU") + devices("GPU")

        candidates = list(order)
        candidates.extend(cpu_devices)
        # 保序去重，避免 device="CPU" 时试两遍。
        seen: set[str] = set()
        return [d for d in candidates if not (d.upper() in seen or seen.add(d.upper()))]

    def _compile_openvino(
        self, core: Any, model: Any, devices: Sequence[str]
    ) -> bool:
        """逐个设备编译，成功即返回 True。全失败返回 False 交给 ORT。"""
        errors: list[str] = []
        for device in devices:
            try:
                compiled = self._compile_on_device(core, model, device)
            except Exception as exc:
                errors.append(f"{device}: {type(exc).__name__}: {exc}"[:180])
                continue
            inputs = list(getattr(model, "inputs", ()) or ())
            if not inputs:
                errors.append(f"{device}: model has no inputs")
                continue
            input_port = inputs[0]
            try:
                self._input_name = input_port.any_name
            except Exception:
                self._input_name = getattr(input_port, "name", None) or input_port
            shape = self._static_shape(getattr(input_port, "partial_shape", None))
            if shape and len(shape) == 4:
                if shape[-1] in {1, 3, 4}:
                    self._input_layout = "NHWC"
                    self._input_height, self._input_width = shape[1], shape[2]
                else:
                    self._input_layout = "NCHW"
                    self._input_height, self._input_width = shape[2], shape[3]
            self._compiled_model = compiled
            self._available = True
            self._runtime = "openvino"
            self._source_name = "openvino"
            self._resolved_device = device
            self._device_fallbacks = (*self._device_fallbacks, *errors)
            self._last_error = None
            return True
        if errors:
            previous = f"{self._openvino_error}; " if self._openvino_error else ""
            self._openvino_error = f"{previous}{'; '.join(errors)}"[:400]
            self._device_fallbacks = (*self._device_fallbacks, *errors)
        return False

    def _compile_on_device(self, core: Any, model: Any, device: str) -> Any:
        """在单个设备上编译。线程上限不喂给纯加速器设备。

        GPU 插件不认 ``INFERENCE_NUM_THREADS``，实测会抛 ``RuntimeError``
        （``Check 'it != m_options_map.end()' failed``），不是 ``TypeError``。
        原来只捕 TypeError，所以把这个属性喂给 GPU 会让整条 OpenVINO 分支
        崩掉、静默退回 258 ms 的 CPU 路径。更根本的是这个旋钮对 GPU 无意义：
        它限的是 CPU 线程池，GPU 上没有对应物。

        排除按加速器名单而不是「只给 CPU」：``AUTO`` 实测接受这个属性（它的
        候选里含 CPU），而注入的测试替身也用默认的 AUTO——按 CPU 白名单写会
        把它们的线程上限一起吞掉。
        """
        root = device.split(".", 1)[0].upper()
        if self._intra_op_threads > 0 and root not in _OPENVINO_ACCELERATORS:
            try:
                compiled = core.compile_model(
                    model, device, {"INFERENCE_NUM_THREADS": self._intra_op_threads}
                )
            except TypeError as exc:
                # 注入的 core 可能只接受两个参数（测试替身、自带运行时的调用
                # 方）。多一个位置参数不该把它们的检测器整体打掉，所以退回
                # 不带配置的调用；上限没设成会在 status() 里报出来。
                self._thread_cap_error = f"{type(exc).__name__}: {exc}"[:200]
                return core.compile_model(model, device)
            self._thread_cap_applied = True
            return compiled
        return core.compile_model(model, device)

    def _onnx_session_options(self, ort: Any) -> Any | None:
        """构造带线程上限的 ``SessionOptions``，取不到就返回 ``None``。

        上限是**优化**，不是正确性前提。所以这里刻意不让它变成加载失败的理由：
        某个运行时没有 ``SessionOptions`` 时，宁可开着满线程跑，也不能让一个
        省 CPU 的设置把整条感知通路静默关掉——那是拿故障换降级。
        真正应用成功与否记在 ``_thread_cap_applied`` 里，由 ``status()`` 报出，
        不然「限了」和「没限成」在外面看起来一模一样。
        """
        if self._intra_op_threads <= 0:
            return None
        try:
            options = ort.SessionOptions()
            # 不设的话 CPU EP 会按物理核数开线程（本机 10 个），实测把整机吃掉
            # 近一半。并行效率很差，所以砍掉的宽度几乎不换来延迟：10 线程
            # 144ms/次要 1.44 CPU·秒，2 线程 259ms/次只要 0.52。
            options.intra_op_num_threads = self._intra_op_threads
            # 单模型单输入，算子间没有可并行的分支；留着 inter-op 池只会在
            # intra-op 之外再开一组线程，把刚设的上限绕过去。
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        except Exception as exc:
            self._thread_cap_error = f"{type(exc).__name__}: {exc}"[:200]
            return None
        self._thread_cap_applied = True
        return options

    def _cap_openmp_before_import(self) -> None:
        """在导入推理运行时之前收一次进程级 OpenMP 池。

        真正管用的调用点是 ``BackendService.__init__``（那时 numpy 还没被导入）。
        这里再调一次是为了让**单独**构造检测器的调用方（基准脚本、旁路进程、
        测试）也拿到同一个上限，而不必知道要先手动设环境变量。
        ``cap_openmp_threads`` 用 ``setdefault``，所以重复调用不会互相打脸。
        """
        self._openmp = cap_openmp_threads(self._intra_op_threads)

    def _record_cuda_failure(self, message: str) -> None:
        normalized = str(message or "CUDA initialization failed")[:300]
        self._onnx_cuda_error = normalized
        entry = f"CUDA: {normalized}"[:360]
        if entry not in self._device_fallbacks:
            self._device_fallbacks = (*self._device_fallbacks, entry)

    def _initialize_onnxruntime(self, path: Path, *, use_cuda: bool) -> None:
        """用 ONNX Runtime 加载 ONNX；CUDA 探测失败时不冒充成功。

        CUDA wheel、CUDA/cuDNN 动态库和显卡驱动都是可选项。先用
        ``get_available_providers`` 判断 wheel 是否编入 provider，再创建 Session，
        最后用 ``session.get_providers`` 验证 CUDA 真正排在首位。三关任意一关失败
        都留下可观测原因，并由调用方继续 OpenVINO/ORT CPU 回落。
        """
        label = "ONNX Runtime CUDA" if use_cuda else "ONNX Runtime"
        try:
            self._cap_openmp_before_import()
            import onnxruntime as ort  # type: ignore[import-not-found]

            getter = getattr(ort, "get_available_providers", None)
            if callable(getter):
                self._onnx_available_providers = tuple(str(item) for item in getter())
            else:
                self._onnx_available_providers = ()
            if use_cuda:
                self._onnx_cuda_probed = True
                if "CUDAExecutionProvider" not in self._onnx_available_providers:
                    providers = ", ".join(self._onnx_available_providers) or "none reported"
                    raise RuntimeError(
                        f"CUDAExecutionProvider is unavailable; providers={providers}"
                    )

            options = self._onnx_session_options(ort)
            if use_cuda:
                providers: list[Any] = [
                    (
                        "CUDAExecutionProvider",
                        {"device_id": self._onnx_cuda_device_id},
                    ),
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]
            if options is None:
                session = ort.InferenceSession(str(path), providers=providers)
            else:
                session = ort.InferenceSession(
                    str(path), sess_options=options, providers=providers
                )
            session_getter = getattr(session, "get_providers", None)
            if callable(session_getter):
                self._onnx_session_providers = tuple(
                    str(item) for item in session_getter()
                )
            else:
                self._onnx_session_providers = tuple(
                    str(item[0] if isinstance(item, tuple) else item) for item in providers
                )
            if use_cuda and (
                not self._onnx_session_providers
                or self._onnx_session_providers[0] != "CUDAExecutionProvider"
            ):
                active = ", ".join(self._onnx_session_providers) or "none reported"
                raise RuntimeError(f"CUDA provider did not activate; session providers={active}")
            inputs = session.get_inputs()
            if not inputs:
                raise RuntimeError("ONNX model has no inputs")
            self._input_name = inputs[0].name
            # 动态维度是字符串（'batch'、'height'）而非整数，_static_shape 会
            # 因此返回 None，保留配置的输入尺寸——这正是动态导出想要的行为。
            shape = self._static_shape(inputs[0].shape)
            if shape and len(shape) == 4:
                if shape[-1] in {1, 3, 4}:
                    self._input_layout = "NHWC"
                    self._input_height, self._input_width = shape[1], shape[2]
                else:
                    self._input_layout = "NCHW"
                    self._input_height, self._input_width = shape[2], shape[3]
            self._onnx_session = session
            self._infer = self._run_onnxruntime
            self._available = True
            self._runtime = "onnxruntime_cuda" if use_cuda else "onnxruntime"
            self._source_name = self._runtime
            self._resolved_device = (
                f"CUDA.{self._onnx_cuda_device_id}" if use_cuda else "CPU"
            )
            if use_cuda:
                self._onnx_cuda_error = None
            self._last_error = None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if use_cuda:
                self._record_cuda_failure(message)
            else:
                self._last_error = f"{label} unavailable: {message}"[:500]

    def _run_onnxruntime(self, frame: Any) -> Any:
        session = self._onnx_session
        if session is None:
            raise RuntimeError("ONNX Runtime session is unavailable")
        tensor = self._preprocess(frame)
        return session.run(None, {self._input_name: tensor})

    def _initialize_opencv_dnn(self, path: Path) -> None:
        try:
            import cv2  # type: ignore[import-not-found]

            if self._intra_op_threads > 0:
                # cv2 的线程数是进程级全局状态，所以只在 cv2 真的成为推理引擎时
                # 才动它——放在这里而不是构造函数里，就是为了把副作用限定在这一种
                # 情况，不去影响仅用 cv2 做 resize 的那条预处理路径。
                try:
                    cv2.setNumThreads(self._intra_op_threads)
                    self._thread_cap_applied = True
                except Exception as exc:
                    self._thread_cap_error = f"{type(exc).__name__}: {exc}"[:200]
            network = cv2.dnn.readNet(str(path))
            if network is None:
                raise RuntimeError("OpenCV DNN returned no network")
            self._opencv_net = network
            self._infer = self._run_opencv_dnn
            self._available = True
            self._runtime = "opencv_dnn"
            self._source_name = "opencv_dnn"
            self._last_error = None
        except Exception as exc:
            self._last_error = f"OpenCV ONNX loader unavailable: {type(exc).__name__}: {exc}"[:500]

    def _run_opencv_dnn(self, frame: Any) -> Any:
        network = self._opencv_net
        if network is None:
            raise RuntimeError("OpenCV DNN network is unavailable")
        tensor = self._preprocess(frame)
        network.setInput(tensor)
        return network.forward()

    @staticmethod
    def _static_shape(value: Any) -> tuple[int, ...] | None:
        if value is None:
            return None
        try:
            dims = tuple(int(item) for item in value)
            if all(item > 0 for item in dims):
                return dims
        except (TypeError, ValueError, OverflowError):
            return None
        return None

    def _initialize_hog(self) -> None:
        try:
            import cv2  # type: ignore[import-not-found]

            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._infer = lambda frame: self._hog_outputs(hog, frame)
            self._available = True
            self._degraded = True
            self._runtime = "opencv_hog"
            self._source_name = "opencv_hog"
            self._last_error = None
            self._uncertainties.append("opencv_hog_person_only")
        except Exception as exc:
            self._last_error = f"OpenCV HOG unavailable: {type(exc).__name__}: {exc}"[:500]

    @staticmethod
    def _array(frame: Any) -> Any:
        try:
            import numpy as np  # type: ignore[import-not-found]

            if isinstance(frame, np.ndarray):
                array = frame
            else:
                # PIL 类帧提供 convert/数组协议。这里刻意拒绝 bytes，因为其
                # 尺寸未知。
                if isinstance(frame, (bytes, bytearray, memoryview)):
                    raise ValueError("raw bytes require a decoded image with dimensions")
                array = np.asarray(frame)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            if array.ndim != 3:
                raise ValueError("frame must be an HxWxC image")
            if array.shape[2] == 4:
                array = array[:, :, :3]
            if array.shape[2] != 3:
                raise ValueError("frame must have 3 or 4 channels")
            return array
        except ImportError as exc:
            raise RuntimeError("numpy is required for local perception") from exc

    @staticmethod
    def _resize_bilinear(np: Any, array: Any, out_width: int, out_height: int) -> Any:
        """纯 numpy 的抗混叠缩放，用于没有 cv2 的部署。

        分两步：整数倍的均值降采样先把混叠压掉，剩余的非整数倍再用双线性
        插值。只用最近邻会在大倍率降采样时丢掉细节，实测足以让检测器完全
        看不见画面里的人。
        """
        height, width = array.shape[:2]
        if height == out_height and width == out_width:
            return array
        source = array
        # 先按整数因子做块平均：每个输出像素都参与了平均，而不是被抽样丢弃。
        # 两个轴分别判断——1920x1080 -> 640x640 的因子是 (1, 3)，竖直方向无
        # 可缩减但水平方向有。要求两轴都 >1 会退化成纯双线性并重新引入混叠
        # （实测 MAE 0.98 -> 2.73）；而在非连续轴上做无意义的 1 倍归约又要
        # 白花 130ms。只对真正 >1 的轴归约，两者都避开。
        factor_y = max(1, int(height // out_height))
        factor_x = max(1, int(width // out_width))
        if factor_y > 1 or factor_x > 1:
            trimmed_h = (height // factor_y) * factor_y
            trimmed_w = (width // factor_x) * factor_x
            source = source[:trimmed_h, :trimmed_w]
            channels = source.shape[2] if source.ndim == 3 else 1
            # uint8 的块和最大 255*factor；uint16 到 257 个像素都不会溢出。
            accumulate = np.uint16 if source.dtype == np.uint8 and factor_y * factor_x < 257 else np.float32
            if factor_x > 1:
                source = source.reshape(trimmed_h, trimmed_w // factor_x, factor_x, channels).sum(
                    axis=2, dtype=accumulate
                )
            if factor_y > 1:
                source = source.reshape(
                    trimmed_h // factor_y, factor_y, source.shape[1], channels
                ).sum(axis=1, dtype=accumulate)
            source = source.astype(np.float32, copy=False) * (1.0 / (factor_y * factor_x))
            height, width = source.shape[:2]
        else:
            source = source.astype(np.float32, copy=False)
        # 再做双线性插值补齐剩余比例；半像素中心对齐，与 cv2/PIL 的约定一致。
        ys = np.clip((np.arange(out_height, dtype=np.float32) + 0.5) * height / out_height - 0.5, 0.0, height - 1.0)
        xs = np.clip((np.arange(out_width, dtype=np.float32) + 0.5) * width / out_width - 0.5, 0.0, width - 1.0)
        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = np.minimum(y0 + 1, height - 1)
        x1 = np.minimum(x0 + 1, width - 1)
        weight_y = (ys - y0).reshape(-1, 1, 1)
        weight_x = (xs - x0).reshape(1, -1, 1)
        top_left = source[y0][:, x0]
        top_right = source[y0][:, x1]
        bottom_left = source[y1][:, x0]
        bottom_right = source[y1][:, x1]
        top = top_left + (top_right - top_left) * weight_x
        bottom = bottom_left + (bottom_right - bottom_left) * weight_x
        return top + (bottom - top) * weight_y

    def _preprocess(self, frame: Any) -> Any:
        import numpy as np  # type: ignore[import-not-found]

        array = self._array(frame)
        resized = None
        try:
            import cv2  # type: ignore[import-not-found]

            resized = cv2.resize(array, (self._input_width, self._input_height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            resized = None
        if resized is None:
            # 冻结宿主没有 cv2 但有 Pillow，所以这条才是生产路径。缩放质量
            # 直接决定能否看见人：最近邻抽样会让 1161x766 的窗口降到 640x640
            # 时，一个 person@0.51 的头像彻底消失、浣熊从 dog@0.84 变成
            # fire hydrant@0.24。Pillow 的 C 实现在降采样时会按比例放大滤波
            # 支撑域，既抗混叠又比手写 numpy 快一个数量级（22ms vs 233ms）。
            try:
                from PIL import Image  # type: ignore[import-not-found]

                if array.ndim == 3 and array.shape[2] == 3 and array.dtype == np.uint8:
                    resized = np.asarray(
                        Image.fromarray(array).resize(
                            (self._input_width, self._input_height), Image.BILINEAR
                        ),
                        dtype=np.float32,
                    )
            except Exception:
                resized = None
        if resized is None:
            # 三个都没有时的最后兜底：纯 numpy 抗混叠缩放。比 Pillow 慢得多，
            # 但仍然远好过最近邻——宁可掉帧，也不能让检测器瞎掉。
            resized = self._resize_bilinear(np, array, self._input_width, self._input_height)
        tensor = resized.astype(np.float32) / 255.0
        if self._input_layout == "NCHW":
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        else:
            tensor = tensor[None, ...]
        return tensor

    def _run_openvino(self, frame: Any) -> Any:
        tensor = self._preprocess(frame)
        compiled = self._compiled_model
        if compiled is None:
            raise RuntimeError("OpenVINO compiled model is unavailable")
        try:
            result = compiled({self._input_name: tensor})
        except Exception:
            # 一些模拟/测试运行时和旧版 OpenVINO 可以直接接收张量，或要求
            # 使用输入端口作为键。
            try:
                result = compiled(tensor)
            except Exception:
                request = compiled.create_infer_request()
                result = request.infer({self._input_name: tensor})
        if isinstance(result, Mapping):
            return tuple(result.values())
        return result

    @staticmethod
    def _hog_outputs(hog: Any, frame: Any) -> Mapping[str, Any]:
        import numpy as np  # type: ignore[import-not-found]

        array = OpenVinoLocalDetector._array(frame)
        try:
            import cv2  # type: ignore[import-not-found]

            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = array
        rects, weights = hog.detectMultiScale(
            bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        height, width = array.shape[:2]
        rows = []
        for index, rect in enumerate(rects if rects is not None else ()):
            x, y, w, h = (float(item) for item in rect)
            score = float(weights[index]) if index < len(weights) else 0.5
            rows.append({
                "label": "person",
                "confidence": max(0.0, min(1.0, score)),
                "bbox": [x / width, y / height, (x + w) / width, (y + h) / height],
            })
        return {"detections": rows}

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "available": self._available,
                "degraded": self._degraded,
                "name": self.name,
                # 能力属于检测器状态，不属于逐帧不确定性：把永久性的能力边界
                # 混进每帧观测，会让每一次世界推送都带上同样两条噪声。这里只
                # 声明实际实现了的能力——本模块没有深度头，也完全没有 OCR。
                "models": ["object_detection"],
                "capabilities": {
                    "object_detection": self._available,
                    "depth": False,
                    "ocr": False,
                },
                "runtime": self._runtime,
                "device": self._device,
                # 配置意图 vs 协商结果。AUTO 被展开成挨个试，所以「配的」和
                # 「跑的」经常不同；GPU 掉到 CPU 是 22 倍的延迟差，必须能从
                # 外面直接看出来，而不是靠观察循环变慢去猜。
                "resolved_device": self._resolved_device,
                "device_fallbacks": list(self._device_fallbacks),
                "openvino_error": self._openvino_error,
                "openvino_devices": list(self._core_devices),
                "onnxruntime": {
                    "cuda_mode": self._onnx_cuda_mode,
                    "cuda_device_id": self._onnx_cuda_device_id,
                    "cuda_probed": self._onnx_cuda_probed,
                    "available_providers": list(self._onnx_available_providers),
                    "session_providers": list(self._onnx_session_providers),
                    "cuda_error": self._onnx_cuda_error,
                },
                "model_path_configured": bool(self._model_path),
                "model_path": self._model_path[-160:] if self._model_path else None,
                "labels_path_configured": bool(self._labels_path),
                "label_load_error": self._label_load_error or None,
                "confidence_threshold": self._confidence_threshold,
                "input_size": [self._input_width, self._input_height],
                "max_detections": self._max_detections,
                "nms_iou_threshold": self._nms_iou_threshold,
                "min_box_width_ratio": self._min_box_width_ratio,
                "min_box_height_ratio": self._min_box_height_ratio,
                # 身份只在当前后端进程内稳定，不落盘，也不冒充 VRChat usr_/avtr_。
                "identity_reid": dict(self._identity_registry.status()),
                # 0 表示没设上限。上限必须能从 /perception 直接读到，否则「配了
                # 但没生效」和「配对了」在外面看起来完全一样。
                "intra_op_threads": self._intra_op_threads,
                # 配置值只说明意图，这两个才说明结果：运行时不认这个旋钮时我们
                # 选择满线程降级而不是失败，那么「限住了」就必须与「没限住」可
                # 区分，否则排查吃满 CPU 时只能看见一个撒谎的 2。
                "thread_cap_applied": self._thread_cap_applied,
                "thread_cap_error": self._thread_cap_error,
                # numpy/BLAS 那个池子单独报。``threads`` 是 ctypes 现读的实测值，
                # 不是配置回显——设晚了或被运维覆盖，只有这里看得出来。实测这一项
                # 才是吃满 CPU 的主因（7.23 核 → 0.11 核），排查时先看它。
                "openmp": {
                    "threads": openmp_thread_count(),
                    "wait_policy": os.environ.get("OMP_WAIT_POLICY"),
                    **self._openmp,
                },
                "frames": self._frames,
                "last_inference_ms": self._last_inference_ms,
                "uncertainties": list(dict.fromkeys(self._uncertainties)),
                "last_error": self._last_error,
            }

    def _label(self, index: int) -> str:
        if 0 <= index < len(self._labels):
            return self._labels[index][:64]
        return f"class_{index}"[:64]

    def _row_widths(self) -> frozenset[int]:
        """检测器行的合法列宽。

        ``5`` 是单类 YOLOv8 导出的 ``cx,cy,w,h,score``（无类别向量）；
        ``6`` 是 NMS 后的 ``x1,y1,x2,y2,score,class``；``7`` 是 OpenVINO
        SSD/DetectionOutput；``len(labels)+4`` 是无 objectness 的 YOLOv8；
        ``len(labels)+5`` 是带 objectness 的 YOLOX。
        """
        count = len(self._labels)
        widths: frozenset[int] = frozenset({6, 7, count + 4, count + 5})
        # 单类模型（count==1）的 YOLOv8 ONNX 导出列宽恰好为 5；此时 count+4==5
        # 已被上面的集合包含。对多类模型也显式加入 5，以便 _as_rows 的布局判定
        # 和 _score_upper_bound 的预筛选能正确处理该列宽而不依赖 count 的值。
        return widths | frozenset({5})

    def _score_upper_bound(self, np: Any, array: Any) -> Any:
        """逐行给出与格式无关的置信度上界。

        ``_decode_rows`` 会按行宽和数值形态在四个分支里选一个解释方式，而这里
        不做那个判断——只取所有分支的上界的最大值。低于阈值的行在任何分支下
        都不可能产出检测，因此丢弃它们不会改变结果。
        """
        scores = array.astype(np.float32, copy=False)
        width = scores.shape[-1]
        with np.errstate(invalid="ignore"):
            # SSD 分支的判定条件必须一并复制。裸 YOLO 的第 2 列是像素宽度
            # （例如 90.0），不加这个门就会让每一行都"过阈值"，预筛选随即退化
            # 成按框宽排序，真正的检测反而被丢掉。
            if width >= 7:
                ssd = np.where(
                    (scores[:, 1] >= 0.0)
                    & (np.abs(scores[:, 1] - np.round(scores[:, 1])) < 1e-6)
                    & (scores[:, 2] >= 0.0)
                    & (scores[:, 2] <= 1.0)
                    & (scores[:, 3] <= scores[:, 5])
                    & (scores[:, 4] <= scores[:, 6]),
                    scores[:, 2],
                    0.0,
                )
            else:
                ssd = np.zeros(scores.shape[0], dtype=np.float32)
            if width == 5:
                # 单类 YOLOv8：[cx,cy,w,h,score]，第 4 列直接是置信度。
                other = scores[:, 4]
            elif width > 6:
                # 与 _decode_rows 相同：用标签数区分 YOLOv8（无 objectness）
                # 和 YOLOX（有 objectness）。
                count = len(self._labels)
                if width - 4 == count and width - 5 != count:
                    other = scores[:, 4:].max(axis=1)
                else:
                    other = scores[:, 4] * scores[:, 5:].max(axis=1)
            else:
                # 六列既可能是 NMS 后的 score，也可能是 objectness * class。
                other = np.maximum(scores[:, 4], scores[:, 4] * scores[:, 5])
            return np.maximum(ssd, other)

    def _as_rows(self, outputs: Any, *, prefilter: bool = False) -> list[list[float]]:
        try:
            import numpy as np  # type: ignore[import-not-found]

            values = outputs.values() if isinstance(outputs, Mapping) else outputs
            if not isinstance(values, (list, tuple)):
                if hasattr(values, "ndim"):
                    # 裸张量必须整体保留。交给 list() 会按第一轴拆成一维切片，
                    # 从而绕过下面的布局判定：``[84, 8400]`` 会被当成 84 行，
                    # 而同一张量包在 ``[arr]`` 或 ``{"out": arr}`` 里却正确解出
                    # 8400 行。两条路径必须得到相同结果。
                    values = (values,)
                else:
                    # dict_values 直接交给 numpy 会变成 0 维 object 数组，导致
                    # 命名输出被静默丢弃；先展开为稳定的值列表。
                    values = list(values) if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)) else (values,)
            widths = self._row_widths()
            rows: list[list[float]] = []
            for value in values:
                array = np.asarray(value)
                if array.size == 0:
                    continue
                if array.ndim == 1:
                    if array.size >= 5:
                        array = array.reshape(1, -1)
                    else:
                        continue
                elif array.ndim >= 2:
                    # 按已知列宽判定布局，而不是猜测形状大小。YOLOv8 的
                    # ``[batch, channels, anchors]``（例如 ``[1,84,8400]``）
                    # 需要转置，YOLOX 的 ``[batch, anchors, channels]`` 不需要。
                    # 仅当倒数第二维命中列宽、且最后一维未命中时才转置：这样
                    # 检测数恰好等于某个列宽（``(84,6)``、``(1,6,7)``）时不会
                    # 被误判，而正确布局也不会因为检测数落在某个区间被破坏。
                    if array.shape[-2] in widths and array.shape[-1] not in widths:
                        axes = list(range(array.ndim))
                        axes[-2], axes[-1] = axes[-1], axes[-2]
                        array = np.transpose(array, axes)
                    array = array.reshape(-1, array.shape[-1])
                if prefilter and array.ndim == 2 and array.shape[-1] >= 5 and array.shape[0] > _NMS_CANDIDATE_LIMIT:
                    # 裸 YOLO 输出有 8400 个 anchor，其中过阈值的通常只有几十个。
                    # 把每一行都转成 Python float 再逐行判断，光是转换就要上百
                    # 毫秒——足以让 10 Hz 的采集循环退化到 3 Hz 并触发导航器的
                    # observation_stale。阈值判断在 numpy 侧做，逐行解释仍然完全
                    # 交给 _decode_rows，避免两处对格式的理解产生分歧。
                    try:
                        bound = self._score_upper_bound(np, array)
                        keep = np.flatnonzero(bound >= self._confidence_threshold)
                        if keep.size > _NMS_CANDIDATE_LIMIT:
                            # 仍然远多于需要的数量时，只保留分数最高的一批。
                            keep = keep[np.argsort(bound[keep])[::-1][:_NMS_CANDIDATE_LIMIT]]
                        array = array[keep]
                    except (TypeError, ValueError, IndexError, FloatingPointError):
                        # 预筛选纯属优化；任何异常都退回全量逐行解码。
                        pass
                for row in array.tolist():
                    try:
                        rows.append([float(item) for item in row])
                    except (TypeError, ValueError, OverflowError):
                        continue
            return rows
        except ImportError:
            # 在刻意不安装 numpy 的轻量测试/旁路部署中，注入适配器仍然有用。
            # 保持简单的行列表协议可用，但不要假装那里能运行真正的
            # OpenVINO 张量。
            values = outputs.values() if isinstance(outputs, Mapping) else outputs
            if not isinstance(values, (list, tuple)):
                values = list(values) if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)) else (values,)
            rows: list[list[float]] = []

            def visit(value: Any) -> None:
                if isinstance(value, (list, tuple)):
                    if value and all(not isinstance(item, (list, tuple, Mapping)) for item in value):
                        try:
                            rows.append([float(item) for item in value])
                        except (TypeError, ValueError, OverflowError):
                            return
                    else:
                        for item in value:
                            visit(item)

            for value in values:
                visit(value)
            return rows
        except Exception:
            return []

    def _suppress_overlaps(self, detections: list[_Detection]) -> list[_Detection]:
        """按置信度排序并做类内非极大值抑制，返回截断到上限的检测。

        裸 YOLO 输出（YOLOv8 的 ``[1,84,8400]``、YOLOv5 的 ``[1,25200,85]``）
        对同一个目标会给出一簇高分 anchor。仅按置信度排序截断留下的正是同一
        目标的重复框：实测两个人会变成 48 个实体，进而污染跟踪器 ID、生成
        O(n^2) 条关系，并让世界状态谎报房间里的人数。

        只在同类之间抑制——不同类别的目标本就可能重叠（人拿着杯子）。对已经
        做过 NMS 的输出（后处理六列格式、SSD、``nms=True`` 导出）这一步是
        幂等的，不会改变结果。
        """
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        if len(ordered) < 2:
            return ordered[: self._max_detections]
        ordered = ordered[:_NMS_CANDIDATE_LIMIT]
        # 同时缓存分组键，避免在内层循环里对每个已保留框反复重算。
        kept: list[tuple[_Detection, Any]] = []
        for candidate in ordered:
            # class_index 缺失时（记录式适配器只给标签）退化为按标签分组。
            key: Any = candidate.class_index if candidate.class_index >= 0 else candidate.label
            duplicate = False
            for accepted, accepted_key in kept:
                if accepted_key == key and _iou(accepted.bbox, candidate.bbox) >= self._nms_iou_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append((candidate, key))
            if len(kept) >= self._max_detections:
                break
        return [item for item, _key in kept]

    def _is_too_small(self, left: float, top: float, right: float, bottom: float) -> bool:
        """归一化框的宽或高低于各自阈值时判定为噪点。

        用两条边而不是面积：细长的误检（例如墙缝、UI 边框）面积可能不小，
        但没有任何一条边像人。两条边都要过关。

        宽高分开设阈值，因为站立的人本来就是高而窄：单一阈值下总是宽度先卡，
        于是「远处的人」和「噪点」被同一个数字裁掉，而画面里的人高度还远没到
        导航器要的比例。默认宽阈值比高阈值宽松，正是为了让远处的人留下来。
        """
        if self._min_box_width_ratio > 0.0 and (right - left) < self._min_box_width_ratio:
            return True
        if self._min_box_height_ratio > 0.0 and (bottom - top) < self._min_box_height_ratio:
            return True
        return False

    def _decode_rows(self, outputs: Any) -> list[_Detection]:
        detections: list[_Detection] = []
        if isinstance(outputs, Mapping):
            # 显式记录格式由 OpenCV 回退和自定义适配器使用。这里保持严格：
            # 格式错误的记录直接跳过，不把它们猜成实体。
            records = outputs.get("detections")
            if isinstance(records, (list, tuple)):
                for record in records:
                    if not isinstance(record, Mapping):
                        continue
                    raw_bbox = record.get("bbox")
                    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                        continue
                    values = [_finite(item, math.nan) for item in raw_bbox]
                    if not all(math.isfinite(item) for item in values):
                        continue
                    if max(abs(item) for item in values) > 1.5:
                        # 记录格式适配器通常使用源帧像素坐标，但没有尺寸就无法
                        # 安全归一化；直接丢弃，不进行猜测。
                        continue
                    left, top, right, bottom = (_clip(item) for item in values)
                    confidence = _finite(record.get("confidence"), 0.0)
                    if confidence < self._confidence_threshold or right <= left or bottom <= top:
                        continue
                    if self._is_too_small(left, top, right, bottom):
                        continue
                    label = str(record.get("label") or "unknown").strip()[:64] or "unknown"
                    detections.append(_Detection(
                        label=label,
                        confidence=min(1.0, max(0.0, confidence)),
                        bbox=(left, top, right, bottom),
                    ))
                # 记录式适配器不保证已去重：OpenCV HOG 的 detectMultiScale
                # 就会对同一个人输出多个重叠框。
                return self._suppress_overlaps(detections)
        for row in self._as_rows(outputs, prefilter=True):
            if len(row) < 5 or not all(math.isfinite(item) for item in row[: min(len(row), 8)]):
                continue
            label_index = -1
            confidence = 0.0
            coords: Sequence[float]
            # 单类 YOLOv8 ONNX 导出：[cx, cy, w, h, score]。ultralytics 在只有
            # 一个类别时省略类别向量，直接输出置信度，行宽因此是 5 而不是 6。
            # 必须与下面的宽行分支共用归一化与钳制，否则模型空间的像素坐标会
            # 被整体钳到 1.0，退化成 right <= left 而丢掉每一个检测。
            if len(row) == 5:
                label_index = 0
                confidence = row[4]
                cx, cy, width, height = row[:4]
                coords = (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
            # OpenVINO DetectionOutput / SSD：image_id、class_id、score、
            # xmin、ymin、xmax、ymax。
            elif (
                len(row) >= 7
                and row[1] >= 0.0
                and abs(row[1] - round(row[1])) < 1e-6
                and 0.0 <= row[2] <= 1.0
                and row[3] <= row[5]
                and row[4] <= row[6]
            ):
                label_index = int(row[1])
                confidence = row[2]
                coords = row[3:7]
            elif len(row) > 6:
                # YOLOX 原始输出在类别分数前包含 objectness，而 YOLOv8 导出会
                # 省略该值。通过标签文件（或内置 COCO 列表）的维度可以区分
                # 两种约定；未知的自定义模型头为保持兼容，保守地按 YOLOX
                # 解释。
                has_objectness = len(row) - 5 == len(self._labels)
                if len(row) - 4 == len(self._labels) and not has_objectness:
                    class_scores = row[4:]
                    label_index = max(range(len(class_scores)), key=lambda index: class_scores[index])
                    confidence = class_scores[label_index]
                else:
                    class_scores = row[5:]
                    label_index = max(range(len(class_scores)), key=lambda index: class_scores[index])
                    confidence = row[4] * class_scores[label_index]
                cx, cy, width, height = row[:4]
                coords = (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
            else:
                # NMS 后六列输出也是常见 OpenVINO YOLO 导出中歧义最小的格式。
                # 某些导出器会输出原始 ``cx,cy,w,h,obj,class``，而不是
                # ``x1,y1,x2,y2,score,class``；下面的单调性检查无需模型元数据
                # 即可区分两者。
                # NMS 后的行在第 5 列使用整数类别 ID。非整数值通常表示六列
                # 原始 YOLO 格式 ``cx,cy,w,h,objectness,class_score``，即使宽
                # 高碰巧看起来像 xyxy 框的单调坐标。
                is_integer_class = abs(row[5] - round(row[5])) < 1e-6
                if is_integer_class and row[2] > row[0] and row[3] > row[1]:
                    coords = row[:4]
                else:
                    cx, cy, width, height = row[:4]
                    coords = (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
                if is_integer_class and row[2] > row[0] and row[3] > row[1]:
                    confidence = row[4]
                    label_index = int(row[5])
                else:
                    confidence = row[4] * row[5]
                    label_index = 0
            if not math.isfinite(confidence) or confidence < self._confidence_threshold:
                continue
            values = list(coords)
            # NMS 输出常使用像素坐标。模型空间坐标按配置的输入尺寸归一化；
            # 已归一化的坐标保持不变。
            if max(abs(item) for item in values) > 1.5:
                values[0] /= max(1.0, float(self._input_width))
                values[2] /= max(1.0, float(self._input_width))
                values[1] /= max(1.0, float(self._input_height))
                values[3] /= max(1.0, float(self._input_height))
            left, top, right, bottom = (_clip(item) for item in values)
            if right <= left or bottom <= top:
                continue
            if self._is_too_small(left, top, right, bottom):
                continue
            detections.append(_Detection(
                label=self._label(label_index),
                confidence=min(1.0, max(0.0, confidence)),
                bbox=(left, top, right, bottom),
                class_index=label_index,
            ))
        return self._suppress_overlaps(detections)

    def _entities(
        self,
        detections: list[_Detection],
        frame: Any,
        now: float,
    ) -> tuple[dict[str, Any], ...]:
        self._tracker.update(detections, now)
        identities = self._identity_registry.assign(
            detections,
            frame,
            now=now,
            source_name=self._source_name,
        )
        entities: list[dict[str, Any]] = []
        fov_half = self._horizontal_fov_deg / 2.0
        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                continue
            left, top, right, bottom = detection.bbox
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            bearing = (center_x - 0.5) * self._horizontal_fov_deg
            # 刻意不提供 ``distance_m``：二维检测器无法推断深度，而 VRChat
            # avatar 身高跨度可达十倍，按固定身高反算米制距离的误差会大于
            # 距离本身。改为发布实测的表观高度，让导航器直接闭环；深度适配器
            # 仍可通过结构化输出提供真实 ``distance_m``。
            apparent_height = bottom - top
            # bbox 已被钳制到 [0,1]，因此贴边说明目标超出画面、真实高度不可测。
            # 此时表观高度会饱和，不能再当作距离的单调函数使用。
            # 0.97/0.03: NPC 坐在地上时摄像机在站立视高，bbox 底边实测稳定
            # 在 0.991，永远够不到 0.999。放宽到 0.97 与 navigator.py 一致。
            clipped = top <= 0.03 or bottom >= 0.97
            track_entity_id = stable_track_entity_id(self._source_name, track_id)
            identity = identities.get(track_id)
            entities.append({
                # 外观可用时发布会话稳定身份；track_entity_id 始终保留，便于诊断
                # 跟踪器是否换号。外观提取失败则保持原来的轨迹 ID，安全降级。
                "id": identity.identity_id if identity is not None else track_entity_id,
                "track_id": track_id,
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": list(detection.bbox),
                "state": "visible",
                "ttl_s": self._tracker.ttl_s,
                "source": self._source_name,
                "attributes": {
                    "screen_center": [round(center_x, 5), round(center_y, 5)],
                    "screen_size": [round(right - left, 5), round(bottom - top, 5)],
                    "bearing_deg": round(max(-fov_half, min(fov_half, bearing)), 3),
                    "apparent_height": round(apparent_height, 5),
                    "apparent_height_clipped": clipped,
                    "depth": "unknown",
                    "track_entity_id": track_entity_id,
                    "identity_scope": "session" if identity is not None else "track",
                    "identity_method": (
                        identity.method if identity is not None else "appearance_unavailable"
                    ),
                    "identity_similarity": (
                        None
                        if identity is None or identity.similarity is None
                        else round(identity.similarity, 4)
                    ),
                    # 明确标出这不是已验证的 VRChat 玩家/Avatar 资源身份。
                    "identity_authoritative": False,
                },
            })
        # 只添加检测框直接支持的几何关系；不能仅凭视觉上的接近推断语义关系。
        for index, entity in enumerate(entities):
            left, top, right, bottom = entity["bbox"]
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            relations: list[dict[str, Any]] = []
            for other_index, other in enumerate(entities):
                if index == other_index:
                    continue
                o_left, o_top, o_right, o_bottom = other["bbox"]
                o_center_x = (o_left + o_right) / 2.0
                o_center_y = (o_top + o_bottom) / 2.0
                if abs(center_x - o_center_x) >= 0.08:
                    relations.append({
                        "type": "left_of" if center_x < o_center_x else "right_of",
                        "target_id": other["id"],
                        "confidence": min(entity["confidence"], other["confidence"]),
                    })
                if abs(center_y - o_center_y) >= 0.08:
                    relations.append({
                        "type": "above" if center_y < o_center_y else "below",
                        "target_id": other["id"],
                        "confidence": min(entity["confidence"], other["confidence"]),
                    })
                if _iou(entity["bbox"], other["bbox"]) >= 0.2:
                    relations.append({
                        "type": "overlapping",
                        "target_id": other["id"],
                        "confidence": min(entity["confidence"], other["confidence"]),
                    })
            # 去除相互重复或重叠关系的噪声，同时保持原顺序。
            seen: set[tuple[str, str]] = set()
            unique_relations: list[dict[str, Any]] = []
            for item in relations:
                key = (str(item["type"]), str(item["target_id"]))
                if key in seen:
                    continue
                seen.add(key)
                unique_relations.append(item)
            entity["relations"] = unique_relations
        return tuple(entities)

    def observe(self, frame: Any, *, now: float) -> VisionObservation:
        # 延迟导入，使 ``backend.local_perception`` 可以作为独立模块使用，
        # 同时由 ``backend.vision`` 重新导出此类。
        from .vision import VisionObservation

        if not self._available:
            raise RuntimeError(self._last_error or "OpenVINO detector is unavailable")
        start = self._clock()
        try:
            # 同一个 detector 可能被 worker 和旁路诊断同时调用；把推理、跟踪
            # 与帧序号更新放在同一把专用锁中，保证模型请求和 ID 分配有序。
            with self._infer_lock:
                raw = self._infer(frame) if self._infer is not None else self._run_openvino(frame)
                # 特定模型适配器可能已经返回结构化世界字段（例如包含 OCR/深度的
                # 图）。保留这些字段，同时统一写入来源和时间。
                if isinstance(raw, Mapping) and "entities" in raw:
                    entities = tuple(raw.get("entities") or ())
                    events = tuple(raw.get("events") or ())
                    uncertainties = tuple(str(item)[:160] for item in (raw.get("uncertainties") or ())[:16])
                    source = str(raw.get("source") or self._source_name)
                    remove_ids = tuple(raw.get("remove_entity_ids") or ())
                    remove_source = raw.get("remove_source")
                else:
                    entities = self._entities(self._decode_rows(raw), frame, now)
                    events = ()
                    extra_uncertainties = (
                        tuple(str(item)[:160] for item in (raw.get("uncertainties") or ())[:16])
                        if isinstance(raw, Mapping)
                        else ()
                    )
                    uncertainties = tuple(dict.fromkeys((*self._uncertainties, *extra_uncertainties)))
                    source = self._source_name
                    remove_ids = ()
                    remove_source = None
                with self._lock:
                    self._frames += 1
                    frame_number = self._frames
                    self._last_inference_ms = round(max(0.0, self._clock() - start) * 1000.0, 3)
                    self._last_error = None
            return VisionObservation(
                entities=entities,
                events=events,
                source=source,
                observed_at=now,
                frame_id=f"{self._runtime}-{frame_number}",
                remove_entity_ids=remove_ids,
                remove_source=remove_source,
                uncertainties=uncertainties,
            )
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
            raise


__all__ = ["OpenVinoLocalDetector"]
