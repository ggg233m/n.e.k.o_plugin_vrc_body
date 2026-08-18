"""可选的视觉编排层。

本模块刻意不依赖具体运行时。未来的画面采集、YOLO 或 VLM 适配器可以实现
这些小型后端协议并发布观测，而不进入 AnyaDance 控制线程。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import threading
import time
from typing import Any, Iterable, Mapping, Protocol

from .world_state import WorldEntity, WorldEvent, WorldStateStore


@dataclass(frozen=True)
class VisionObservation:
    entities: tuple[WorldEntity | Mapping[str, Any], ...] = ()
    events: tuple[WorldEvent | Mapping[str, Any], ...] = ()
    source: str = "vision"
    observed_at: float | None = None
    frame_id: str | None = None


class FrameDetector(Protocol):
    name: str

    def status(self) -> Mapping[str, Any]: ...

    def observe(self, frame: Any, *, now: float) -> VisionObservation: ...


class SemanticBackend(Protocol):
    name: str

    def status(self) -> Mapping[str, Any]: ...

    def observe(self, frame: Any, *, world: Mapping[str, Any], now: float) -> VisionObservation: ...


def optional_dependency_status() -> dict[str, bool]:
    """不导入重量级模型包，只报告可选能力是否存在。"""
    return {
        "opencv": importlib.util.find_spec("cv2") is not None,
        "ultralytics": importlib.util.find_spec("ultralytics") is not None,
        "openvr": importlib.util.find_spec("openvr") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


class VisionRuntime:
    """围绕世界状态存储协调检测器与 VLM 适配器。"""

    def __init__(
        self,
        store: WorldStateStore | None = None,
        *,
        detector: FrameDetector | None = None,
        semantic: SemanticBackend | None = None,
        semantic_cooldown_s: float = 0.75,
        clock: Any = time.monotonic,
    ) -> None:
        self.store = store or WorldStateStore(clock=clock)
        self.detector = detector
        self.semantic = semantic
        self.semantic_cooldown_s = max(0.1, float(semantic_cooldown_s))
        self._clock = clock
        self._lock = threading.Lock()
        self._last_semantic_at: float | None = None
        self._last_error: str | None = None
        self.store.set_backend_status("vision_runtime", self.status())

    def status(self) -> dict[str, Any]:
        with self._lock:
            detector = self.detector
            semantic = self.semantic
            last_error = self._last_error
        return {
            "enabled": detector is not None or semantic is not None,
            "detector": detector.status() if detector is not None else {"available": False, "reason": "not_configured"},
            "semantic": semantic.status() if semantic is not None else {"available": False, "reason": "not_configured"},
            "optional_dependencies": optional_dependency_status(),
            "last_error": last_error,
        }

    def ingest(self, observation: VisionObservation | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(observation, VisionObservation):
            value = observation
        else:
            value = VisionObservation(
                entities=tuple(observation.get("entities") or ()) if isinstance(observation, Mapping) else (),
                events=tuple(observation.get("events") or ()) if isinstance(observation, Mapping) else (),
                source=str(observation.get("source") or "vision") if isinstance(observation, Mapping) else "vision",
                observed_at=observation.get("observed_at") if isinstance(observation, Mapping) else None,
                frame_id=str(observation.get("frame_id")) if isinstance(observation, Mapping) and observation.get("frame_id") is not None else None,
            )
        result = self.store.ingest(
            value.entities,
            value.events,
            source=value.source,
            observed_at=value.observed_at,
        )
        self.store.set_backend_status("vision_runtime", self.status())
        return result

    def process_frame(self, frame: Any, *, force_semantic: bool = False) -> dict[str, Any]:
        """处理一帧画面而不阻塞身体调度器。

        采集循环由调用方负责。语义推理受到频率限制，只在明确请求或冷却时间
        到期时运行。
        """
        now = self._clock()
        detector = self.detector
        semantic = self.semantic
        try:
            if detector is not None:
                self.ingest(detector.observe(frame, now=now))
            if semantic is not None:
                with self._lock:
                    semantic_due = (
                        force_semantic
                        or self._last_semantic_at is None
                        or now - self._last_semantic_at >= self.semantic_cooldown_s
                    )
                if semantic_due:
                    with self._lock:
                        self._last_semantic_at = now
                    self.ingest(semantic.observe(frame, world=self.store.snapshot(now=now), now=now))
            with self._lock:
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
        self.store.set_backend_status("vision_runtime", self.status())
        result = self.store.snapshot(now=now)
        result["vision"] = self.status()
        return result

    def snapshot(self) -> dict[str, Any]:
        result = self.store.snapshot()
        result["vision"] = self.status()
        return result


__all__ = [
    "FrameDetector",
    "SemanticBackend",
    "VisionObservation",
    "VisionRuntime",
    "optional_dependency_status",
]
