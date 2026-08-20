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
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .vision import VisionObservation
from .world_state import stable_track_entity_id


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
        confidence_threshold: float = 0.35,
        input_width: int = 640,
        input_height: int = 640,
        horizontal_fov_deg: float = 90.0,
        max_detections: int = 64,
        nms_iou_threshold: float = 0.45,
        min_box_ratio: float = 0.02,
        track_iou_threshold: float = 0.25,
        track_ttl_s: float = 1.5,
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
        self._confidence_threshold = min(1.0, max(0.0, float(confidence_threshold)))
        self._input_width = min(4096, max(32, int(input_width)))
        self._input_height = min(4096, max(32, int(input_height)))
        self._horizontal_fov_deg = min(180.0, max(1.0, float(horizontal_fov_deg)))
        self._max_detections = min(512, max(1, int(max_detections)))
        # 0 会把所有同类框合成一个，1 等于完全不抑制；两端都不是有用的部署
        # 取值，因此钳到留有余量的区间内。
        self._nms_iou_threshold = min(0.95, max(0.1, float(nms_iou_threshold)))
        # 框最短边占画面的最小比例。实测真实帧里出现过 27×27 px 的高分假阳性
        # （1920 宽下约 1.4%），它们会被跟踪器当成真人并污染导航的距离闭环。
        # 私人房里真人 avatar 至少占画面几个百分点，2% 足以滤掉噪点又不会误伤
        # 远处的人。0 表示完全关闭该过滤。
        self._min_box_ratio = min(0.5, max(0.0, _finite(min_box_ratio, 0.02)))
        self._tracker = _IoUTracker(iou_threshold=track_iou_threshold, ttl_s=track_ttl_s)
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
        # ONNX 优先交给 ONNX Runtime。它是三条路径里唯一能正确处理动态
        # batch/宽高的：OpenCV 的 ONNX 导入器会在动态形状的 Concat 上形状
        # 推断失败（实测 yolo11n 动态导出直接抛 ConcatLayer），而 OpenVINO
        # wheel 在冻结宿主里根本不存在。显式注入 core 时跳过这里，避免旁路
        # 掉调用方指定的运行时。
        if supplied_core is None and path.suffix.lower() == ".onnx":
            self._initialize_onnxruntime(path)
            if self._available:
                return
        try:
            core = supplied_core
            if core is None:
                from openvino import Core  # type: ignore[import-not-found]

                core = Core()
            model = core.read_model(model=str(path))
            compiled = core.compile_model(model, self._device)
            inputs = list(getattr(model, "inputs", ()) or ())
            if not inputs:
                raise RuntimeError("OpenVINO model has no inputs")
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
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:500]
            # 即使未安装可选的 OpenVINO wheel，OpenCV 也通常带有可用的 ONNX
            # 导入器。此回退仅限 ONNX 文件；尝试用 cv2 解释 IR XML 会造成
            # 虚假的成功状态。
            if path.suffix.lower() == ".onnx":
                self._initialize_opencv_dnn(path)

    def _initialize_onnxruntime(self, path: Path) -> None:
        """用 ONNX Runtime 加载 ONNX 模型。

        只声明 CPU 提供者：宿主捆绑包只带了 ``onnxruntime.dll`` 和
        ``onnxruntime_providers_shared.dll``，没有 CUDA/DirectML 提供者库，
        请求它们只会得到警告和静默回退。
        """
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
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
            self._runtime = "onnxruntime"
            self._source_name = "onnxruntime"
            self._last_error = None
        except Exception as exc:
            self._last_error = f"ONNX Runtime unavailable: {type(exc).__name__}: {exc}"[:500]

    def _run_onnxruntime(self, frame: Any) -> Any:
        session = self._onnx_session
        if session is None:
            raise RuntimeError("ONNX Runtime session is unavailable")
        tensor = self._preprocess(frame)
        return session.run(None, {self._input_name: tensor})

    def _initialize_opencv_dnn(self, path: Path) -> None:
        try:
            import cv2  # type: ignore[import-not-found]

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
                "model_path_configured": bool(self._model_path),
                "model_path": self._model_path[-160:] if self._model_path else None,
                "labels_path_configured": bool(self._labels_path),
                "label_load_error": self._label_load_error or None,
                "confidence_threshold": self._confidence_threshold,
                "input_size": [self._input_width, self._input_height],
                "max_detections": self._max_detections,
                "nms_iou_threshold": self._nms_iou_threshold,
                "min_box_ratio": self._min_box_ratio,
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
        """归一化框的任一边小于 ``min_box_ratio`` 时判定为噪点。

        用最短边而不是面积：细长的误检（例如墙缝、UI 边框）面积可能不小，
        但没有任何一条边像人。两条边都要过关。
        """
        if self._min_box_ratio <= 0.0:
            return False
        return (right - left) < self._min_box_ratio or (bottom - top) < self._min_box_ratio

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

    def _entities(self, detections: list[_Detection], now: float) -> tuple[dict[str, Any], ...]:
        self._tracker.update(detections, now)
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
            clipped = top <= 0.001 or bottom >= 0.999
            entities.append({
                "id": stable_track_entity_id(self._source_name, track_id),
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
                    entities = self._entities(self._decode_rows(raw), now)
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
