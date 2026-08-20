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


def _load_labels(path: str | Path | None) -> tuple[str, ...]:
    if not path:
        return _COCO_LABELS
    candidate = Path(path)
    try:
        raw = candidate.read_text(encoding="utf-8")
        if candidate.suffix.lower() == ".json":
            value = json.loads(raw)
            if isinstance(value, Mapping):
                # 同时支持 {"0": "person"} 和 {"names": [...]} 两种格式。
                names = value.get("names", value)
                if isinstance(names, Mapping):
                    ordered = sorted(names.items(), key=lambda item: int(item[0]))
                    return tuple(str(item[1]).strip() or f"class_{item[0]}" for item in ordered)
                value = names
            if isinstance(value, (list, tuple)):
                return tuple(str(item).strip() or f"class_{index}" for index, item in enumerate(value))
        lines = tuple(line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#"))
        if lines:
            return lines
    except Exception:
        # 检测器构造时会在状态中报告问题；回退到 COCO 名称可避免格式错误的
        # 可选标签文件使控制进程崩溃。
        return _COCO_LABELS
    return _COCO_LABELS


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
        self._labels = _load_labels(self._labels_path or None)
        self._device = str(device or "AUTO").strip() or "AUTO"
        self._confidence_threshold = min(1.0, max(0.0, float(confidence_threshold)))
        self._input_width = min(4096, max(32, int(input_width)))
        self._input_height = min(4096, max(32, int(input_height)))
        self._horizontal_fov_deg = min(180.0, max(1.0, float(horizontal_fov_deg)))
        self._max_detections = min(512, max(1, int(max_detections)))
        self._tracker = _IoUTracker(iou_threshold=track_iou_threshold, ttl_s=track_ttl_s)
        self._infer = infer
        # 模型推理和跟踪器都不是默认线程安全的；旁路调用也必须按帧串行化，
        # 避免实体 ID 与帧计数被并发破坏。
        self._infer_lock = threading.Lock()
        self._compiled_model: Any | None = None
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
        if self._available:
            self._uncertainties.extend(("depth_unavailable", "ocr_unavailable"))

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

    def _preprocess(self, frame: Any) -> Any:
        import numpy as np  # type: ignore[import-not-found]

        array = self._array(frame)
        try:
            import cv2  # type: ignore[import-not-found]

            resized = cv2.resize(array, (self._input_width, self._input_height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            # 最近邻回退让适配器在只有 numpy 时仍可运行；精度属于部署问题，
            # 不能因此伪造世界观测。
            y_idx = np.linspace(0, array.shape[0] - 1, self._input_height).astype(int)
            x_idx = np.linspace(0, array.shape[1] - 1, self._input_width).astype(int)
            resized = array[y_idx][:, x_idx]
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
                # 保留旧调用方使用的公开能力清单；深度/OCR 属于可选模型头，
                # 除非注入的 infer 适配器明确提供，否则始终标记为未知。
                "models": ["yolox_tiny", "depth", "ocr"],
                "runtime": self._runtime,
                "device": self._device,
                "model_path_configured": bool(self._model_path),
                "model_path": self._model_path[-160:] if self._model_path else None,
                "labels_path_configured": bool(self._labels_path),
                "confidence_threshold": self._confidence_threshold,
                "input_size": [self._input_width, self._input_height],
                "max_detections": self._max_detections,
                "frames": self._frames,
                "last_inference_ms": self._last_inference_ms,
                "uncertainties": list(dict.fromkeys(self._uncertainties)),
                "last_error": self._last_error,
            }

    def _label(self, index: int) -> str:
        if 0 <= index < len(self._labels):
            return self._labels[index][:64]
        return f"class_{index}"[:64]

    @staticmethod
    def _as_rows(outputs: Any) -> list[list[float]]:
        try:
            import numpy as np  # type: ignore[import-not-found]

            values = outputs.values() if isinstance(outputs, Mapping) else outputs
            if not isinstance(values, (list, tuple)):
                # dict_values 直接交给 numpy 会变成 0 维 object 数组，导致
                # 命名输出被静默丢弃；先展开为稳定的值列表。
                values = list(values) if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)) else (values,)
            rows: list[list[float]] = []
            for value in values:
                array = np.asarray(value)
                if array.size == 0:
                    continue
                if array.ndim == 1:
                    if array.size >= 6:
                        array = array.reshape(1, -1)
                    else:
                        continue
                elif array.ndim >= 2:
                    # YOLOv8 风格的 ONNX 导出通常使用
                    # ``[batch, channels, anchors]``（例如 ``[1,84,8400]``），
                    # 而 YOLOX 使用 ``[batch, anchors, channels]``。将较小的
                    # 第二维视为通道，并在展平前转置。
                    if (
                        array.ndim == 3
                        and 4 <= array.shape[1] <= 256
                        and (array.shape[1] < array.shape[2] or array.shape[2] <= 4)
                    ):
                        array = np.transpose(array, (0, 2, 1))
                    elif (
                        array.ndim == 2
                        and 4 <= array.shape[0] <= 256
                        and (
                            (array.shape[0] >= 16 and array.shape[0] < array.shape[1])
                            or (
                                array.shape[0] in {6, 7, 8, 84, 85, 86}
                                and array.shape[0] > array.shape[1]
                            )
                            or array.shape[1] <= 4
                        )
                    ):
                        array = np.transpose(array, (1, 0))
                    array = array.reshape(-1, array.shape[-1])
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
                    label = str(record.get("label") or "unknown").strip()[:64] or "unknown"
                    detections.append(_Detection(
                        label=label,
                        confidence=min(1.0, max(0.0, confidence)),
                        bbox=(left, top, right, bottom),
                    ))
                detections.sort(key=lambda item: item.confidence, reverse=True)
                return detections[: self._max_detections]
        for row in self._as_rows(outputs):
            if len(row) < 6 or not all(math.isfinite(item) for item in row[: min(len(row), 8)]):
                continue
            label_index = -1
            confidence = 0.0
            coords: Sequence[float]
            # OpenVINO DetectionOutput / SSD：image_id、class_id、score、
            # xmin、ymin、xmax、ymax。
            if (
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
            detections.append(_Detection(
                label=self._label(label_index),
                confidence=min(1.0, max(0.0, confidence)),
                bbox=(left, top, right, bottom),
                class_index=label_index,
            ))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self._max_detections]

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
            # 刻意不提供 ``distance_m``：二维检测器无法推断深度。除非深度
            # 适配器提供该字段，否则导航器将其视为未知。
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
