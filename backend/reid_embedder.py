"""OSNet 外观嵌入器：给会话级 Avatar 重识别提供学习到的外观向量。

`avatar_identity.appearance_descriptor` 的颜色直方图在真实 VRChat 帧上没有可用
阈值：实测 17 张截图里，亮场景下不同 Avatar（甚至一盆吊兰）互相打到 0.89~0.99，
而同一 Avatar 换到彩色灯光下掉到 0.2——"不同人合并"与"同人打新号"同时发生。
OSNet（ICCV 2019 行人重识别网络，x0_25 变体 907KB）在同一批数据上不同人 ≤0.72、
同人同视角 ≥0.83；在 192 个同画风动漫角色的极端拼图上（18336 对负样本），
0.75 阈值的误并率 0.80%、扰动召回 96.7%。

本模块只做嵌入，不做匹配决策；匹配仍在 AvatarIdentityRegistry。模型缺失或
onnxruntime 不可用时 ``available`` 为 False，调用方应回落颜色直方图并配用
它自己的阈值——两种向量维度不同，`_similarity` 对不同长度返回 0，绝不能混库。

刻意只用 CPU provider：单人嵌入实测 6.8ms（Xeon E5-2666 v3，2 线程），且不与
检测器争抢 GPU 会话；嵌入按轨迹错峰刷新（约每人每 0.5s 一次），CPU 足够。
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .avatar_identity import _normalized_bbox

# MSMT17 训练时的 ImageNet 归一化。实测这不是可省略的细节：裸 /255 输入会把
# 极端拼图的跨身份最高相似度从 0.85 抬到 0.85+ 且整体分布糊掉（p99 0.746→0.81+），
# 分离带明显变窄。
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class OsnetReidEmbedder:
    """加载失败自记录、绝不抛出的 OSNet ONNX 嵌入器。"""

    name = "osnet_onnx"

    def __init__(
        self,
        *,
        model_path: str | Path,
        intra_op_threads: int = 2,
        input_width: int = 128,
        input_height: int = 256,
        session: Any | None = None,
    ) -> None:
        self._model_path = str(model_path).strip()
        self._intra_op_threads = min(32, max(0, int(intra_op_threads)))
        self._input_width = min(1024, max(16, int(input_width)))
        self._input_height = min(1024, max(16, int(input_height)))
        self._lock = threading.Lock()
        self._session: Any | None = session
        self._input_name: str = "input"
        self._providers: tuple[str, ...] = ()
        self._embed_count = 0
        self._last_embed_ms: float | None = None
        self._error: str | None = None
        if session is not None:
            # 注入的测试替身可能没有 get_inputs；保持默认输入名。
            try:
                inputs = session.get_inputs()
                if inputs:
                    self._input_name = inputs[0].name
            except Exception:
                pass
            self.available = True
            return
        self.available = self._initialize()

    def _initialize(self) -> bool:
        if not self._model_path:
            self._error = "reid model_path is not configured"
            return False
        path = Path(self._model_path)
        if not path.exists() or not path.is_file():
            self._error = f"reid model does not exist: {path}"
            return False
        try:
            # 函数级导入打破与 local_perception 的循环依赖，同时复用它的进程级
            # 导入缓存（含 WinRT 先于 ORT 加载必挂的排序约束与 4.8s 失败缓存）。
            from .local_perception import import_onnxruntime

            ort = import_onnxruntime()
            options = None
            if self._intra_op_threads > 0:
                try:
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = self._intra_op_threads
                    options.inter_op_num_threads = 1
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                except Exception:
                    # 线程上限是优化不是正确性前提；构造不出 options 就满线程跑。
                    options = None
            providers = ["CPUExecutionProvider"]
            if options is None:
                session = ort.InferenceSession(str(path), providers=providers)
            else:
                session = ort.InferenceSession(
                    str(path), sess_options=options, providers=providers
                )
            inputs = session.get_inputs()
            if not inputs:
                raise RuntimeError("reid model has no inputs")
            self._input_name = inputs[0].name
            shape = inputs[0].shape
            # 动态维度是字符串，保留配置尺寸；静态导出则以模型图为准。
            try:
                if len(shape) == 4 and int(shape[2]) > 0 and int(shape[3]) > 0:
                    self._input_height = int(shape[2])
                    self._input_width = int(shape[3])
            except (TypeError, ValueError):
                pass
            getter = getattr(session, "get_providers", None)
            if callable(getter):
                self._providers = tuple(str(item) for item in getter())
            self._session = session
            self._error = None
            return True
        except BaseException as exc:
            # 与检测器一致：嵌入器缺依赖是预期降级，不是感知层故障。
            self._error = f"{type(exc).__name__}: {exc}"[:300]
            return False

    def embed(self, frame: Any, bbox: Sequence[float]) -> tuple[float, ...] | None:
        """裁剪-缩放-归一化-嵌入，任何失败都返回 ``None``。

        返回 L2 归一化的向量，点积即余弦相似度，与
        ``avatar_identity._similarity`` 的约定一致。
        """
        session = self._session
        if session is None:
            return None
        normalized = _normalized_bbox(bbox)
        if normalized is None:
            return None
        started = time.perf_counter()
        try:
            import numpy as np  # type: ignore[import-not-found]

            array = np.asarray(frame)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            if array.ndim != 3 or array.shape[2] < 3:
                return None
            height, width = array.shape[:2]
            if height < 4 or width < 4:
                return None
            left, top, right, bottom = normalized
            x0 = min(width - 1, max(0, int(math.floor(left * width))))
            y0 = min(height - 1, max(0, int(math.floor(top * height))))
            x1 = min(width, max(x0 + 1, int(math.ceil(right * width))))
            y1 = min(height, max(y0 + 1, int(math.ceil(bottom * height))))
            crop = array[y0:y1, x0:x1, :3]
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                return None
            crop = np.nan_to_num(
                crop.astype(np.float32, copy=False), nan=0.0, posinf=255.0, neginf=0.0
            )
            # 与 appearance_descriptor 相同的量程判断：兼容 [0,1] 浮点帧。
            if float(np.max(crop)) <= 1.5:
                crop = crop * 255.0
            crop_u8 = np.clip(crop, 0.0, 255.0).astype(np.uint8)
            resized = self._resize(np, crop_u8)
            if resized is None:
                return None
            tensor = resized.astype(np.float32) / 255.0
            tensor = (tensor - np.asarray(_IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
                _IMAGENET_STD, dtype=np.float32
            )
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
            output = session.run(None, {self._input_name: tensor})[0]
            vector = np.asarray(output, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 1e-8:
                return None
            result = tuple(float(item) for item in vector / norm)
        except Exception:
            return None
        with self._lock:
            self._embed_count += 1
            self._last_embed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    def _resize(self, np: Any, crop: Any) -> Any | None:
        try:
            import cv2  # type: ignore[import-not-found]

            return cv2.resize(
                crop,
                (self._input_width, self._input_height),
                interpolation=cv2.INTER_LINEAR,
            )
        except Exception:
            pass
        try:
            from PIL import Image  # type: ignore[import-not-found]

            return np.asarray(
                Image.fromarray(crop).resize(
                    (self._input_width, self._input_height), Image.BILINEAR
                )
            )
        except Exception:
            return None

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "available": self.available,
                "model_path": self._model_path[-160:] if self._model_path else None,
                "input_size": [self._input_width, self._input_height],
                "providers": list(self._providers),
                "intra_op_threads": self._intra_op_threads,
                "embed_count": self._embed_count,
                "last_embed_ms": self._last_embed_ms,
                "error": self._error,
            }


__all__ = ["OsnetReidEmbedder"]
