"""可选的视觉编排层。

本模块刻意不依赖具体运行时。未来的画面采集、YOLO 或 VLM 适配器可以实现
这些小型后端协议并发布观测，而不进入 AnyaDance 控制线程。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol

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


class FrameSource(Protocol):
    """模型无关的画面源；实现可以是 mss、OpenVR 或外部采集进程。"""

    name: str

    def status(self) -> Mapping[str, Any]: ...

    def read(self) -> Any: ...

    def close(self) -> None: ...


class SemanticBackend(Protocol):
    name: str

    def status(self) -> Mapping[str, Any]: ...

    def observe(self, frame: Any, *, world: Mapping[str, Any], now: float) -> VisionObservation: ...


def optional_dependency_status() -> dict[str, bool]:
    """不导入重量级模型包，只报告可选能力是否存在。"""
    return {
        "opencv": importlib.util.find_spec("cv2") is not None,
        "mss": importlib.util.find_spec("mss") is not None,
        "PIL": importlib.util.find_spec("PIL") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "ultralytics": importlib.util.find_spec("ultralytics") is not None,
        "openvr": importlib.util.find_spec("openvr") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


class MssFrameSource:
    """可选的纯 mss 桌面采集器，不依赖插件 SDK 或模型包。"""

    name = "mss"

    def __init__(
        self,
        *,
        monitor_index: int = 1,
        region: Mapping[str, Any] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._capture: Any = None
        self._monitor: Mapping[str, int] | None = None
        self._np: Any = None
        self._image: Any = None
        self._closed = False
        self._last_error: str | None = None
        self._frames = 0
        try:
            import mss  # type: ignore[import-not-found]

            self._capture = mss.mss()
            monitors = list(getattr(self._capture, "monitors", ()))
            if not monitors:
                raise RuntimeError("mss reported no monitors")
            index = min(max(int(monitor_index), 0), len(monitors) - 1)
            base = dict(monitors[index])
            if region is not None:
                for key in ("left", "top", "width", "height"):
                    if key in region:
                        base[key] = int(region[key])
            if base.get("width", 0) <= 0 or base.get("height", 0) <= 0:
                raise ValueError("capture region must have positive width and height")
            self._monitor = {
                key: int(base[key]) for key in ("left", "top", "width", "height")
            }
            try:
                import numpy as np  # type: ignore[import-not-found]

                self._np = np
            except ImportError:
                try:
                    from PIL import Image  # type: ignore[import-not-found]

                    self._image = Image
                except ImportError:
                    self._last_error = "neither numpy nor Pillow is installed"
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:256]

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "available": self._capture is not None and self._monitor is not None and not self._closed,
                "name": self.name,
                "monitor": dict(self._monitor or {}),
                "frames": self._frames,
                "last_error": self._last_error,
            }

    def read(self) -> Any:
        with self._lock:
            capture = self._capture
            monitor = self._monitor
            closed = self._closed
        if capture is None or monitor is None or closed:
            return None
        try:
            shot = capture.grab(monitor)
            with self._lock:
                self._frames += 1
                self._last_error = None
            if self._np is not None:
                return self._np.frombuffer(shot.rgb, dtype=self._np.uint8).reshape(
                    int(shot.height), int(shot.width), 3
                ).copy()
            if self._image is not None:
                return self._image.frombytes("RGB", (int(shot.width), int(shot.height)), shot.rgb)
            return shot.rgb
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:256]
            return None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            capture = self._capture
            self._capture = None
        if capture is not None:
            try:
                capture.close()
            except Exception:
                pass


@dataclass(frozen=True)
class CapturedFrame:
    frame: Any
    captured_at: float
    sequence: int


def _release_frame(frame: Any) -> None:
    close = getattr(frame, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


class VisionWorker:
    """有界采集/推理 worker；丢帧优先于堆积，不触碰身体调度线程。"""

    def __init__(
        self,
        runtime: "VisionRuntime",
        source: FrameSource,
        *,
        interval_s: float = 0.1,
        queue_size: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.source = source
        self.interval_s = min(2.0, max(0.01, float(interval_s)))
        self.queue: Queue[CapturedFrame] = Queue(maxsize=max(1, min(4, int(queue_size))))
        self._clock = clock
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._producer: threading.Thread | None = None
        self._consumer: threading.Thread | None = None
        self._running = False
        self._sequence = 0
        self._captured = 0
        self._processed = 0
        self._dropped = 0
        self._last_capture_at: float | None = None
        self._last_processed_at: float | None = None
        self._last_error: str | None = None

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            detector = self.runtime.detector or self.runtime.semantic
            if detector is None:
                self._last_error = "no detector or semantic backend is configured"
                return False
            self._stop.clear()
            self._running = True
            self._producer = threading.Thread(
                target=self._capture_loop,
                name="neko-vision-capture",
                daemon=True,
            )
            self._consumer = threading.Thread(
                target=self._process_loop,
                name="neko-vision-process",
                daemon=True,
            )
            self._producer.start()
            self._consumer.start()
            return True

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        try:
            self.source.close()
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:256]
        for thread in (self._producer, self._consumer):
            if thread and thread.is_alive():
                thread.join(timeout=max(0.1, timeout_s))
        while True:
            try:
                item = self.queue.get_nowait()
            except Empty:
                break
            _release_frame(item.frame)
        with self._lock:
            alive = any(thread and thread.is_alive() for thread in (self._producer, self._consumer))
            self._running = alive
            if not alive:
                self._producer = None
                self._consumer = None

    def _record_error(self, exc: Exception) -> None:
        with self._lock:
            self._last_error = f"{type(exc).__name__}: {exc}"[:256]

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            captured_at = self._clock()
            try:
                frame = self.source.read()
                if frame is not None:
                    with self._lock:
                        self._sequence += 1
                        sequence = self._sequence
                        self._captured += 1
                        self._last_capture_at = captured_at
                    packet = CapturedFrame(frame, captured_at, sequence)
                    try:
                        self.queue.put_nowait(packet)
                    except Full:
                        try:
                            old = self.queue.get_nowait()
                            _release_frame(old.frame)
                        except Empty:
                            pass
                        try:
                            self.queue.put_nowait(packet)
                        except Full:
                            _release_frame(packet.frame)
                        with self._lock:
                            self._dropped += 1
            except Exception as exc:
                self._record_error(exc)
            self._stop.wait(self.interval_s)

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self.queue.get(timeout=min(0.2, self.interval_s))
            except Empty:
                continue
            try:
                self.runtime.process_frame(packet.frame, observed_at=packet.captured_at)
                with self._lock:
                    self._processed += 1
                    self._last_processed_at = self._clock()
            except Exception as exc:
                self._record_error(exc)
            finally:
                _release_frame(packet.frame)

    def status(self) -> dict[str, Any]:
        now = self._clock()
        try:
            source_status = dict(self.source.status())
        except Exception as exc:
            source_status = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}"[:256],
            }
        with self._lock:
            return {
                "enabled": True,
                "running": self._running,
                "source": source_status,
                "queue_size": self.queue.maxsize,
                "queue_depth": self.queue.qsize(),
                "interval_ms": round(self.interval_s * 1000.0, 1),
                "frames_captured": self._captured,
                "frames_processed": self._processed,
                "frames_dropped": self._dropped,
                "last_capture_age_ms": (
                    None if self._last_capture_at is None
                    else round(max(0.0, now - self._last_capture_at) * 1000.0, 1)
                ),
                "last_processed_age_ms": (
                    None if self._last_processed_at is None
                    else round(max(0.0, now - self._last_processed_at) * 1000.0, 1)
                ),
                "last_error": self._last_error,
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
        observation_callback: Callable[[VisionObservation, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.store = store or WorldStateStore(clock=clock)
        self.detector = detector
        self.semantic = semantic
        self.semantic_cooldown_s = max(0.1, float(semantic_cooldown_s))
        self._clock = clock
        self._observation_callback = observation_callback
        self._lock = threading.Lock()
        self._last_semantic_at: float | None = None
        self._last_error: str | None = None
        self.store.set_backend_status("vision_runtime", self.status())

    def status(self) -> dict[str, Any]:
        with self._lock:
            detector = self.detector
            semantic = self.semantic
            last_error = self._last_error
        def backend_status(backend: Any, *, label: str) -> Mapping[str, Any]:
            if backend is None:
                return {"available": False, "reason": "not_configured"}
            try:
                value = backend.status()
                return dict(value) if isinstance(value, Mapping) else {
                    "available": False,
                    "reason": f"{label} status must be an object",
                }
            except Exception as exc:
                return {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}"[:256],
                }
        return {
            "enabled": detector is not None or semantic is not None,
            "detector": backend_status(detector, label="detector"),
            "semantic": backend_status(semantic, label="semantic"),
            "optional_dependencies": optional_dependency_status(),
            "last_error": last_error,
        }

    def set_backends(
        self,
        *,
        detector: FrameDetector | None = None,
        semantic: SemanticBackend | None = None,
    ) -> None:
        with self._lock:
            self.detector = detector
            self.semantic = semantic
        self.store.set_backend_status("vision_runtime", self.status())

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
        callback = self._observation_callback
        if callback is not None:
            try:
                callback(value, result)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"observation callback: {type(exc).__name__}: {exc}"[:500]
        self.store.set_backend_status("vision_runtime", self.status())
        return result

    def process_frame(
        self,
        frame: Any,
        *,
        force_semantic: bool = False,
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        """处理一帧画面而不阻塞身体调度器。

        采集循环由调用方负责。语义推理受到频率限制，只在明确请求或冷却时间
        到期时运行。
        """
        processing_now = self._clock()
        try:
            observation_now = min(processing_now, float(observed_at)) if observed_at is not None else processing_now
        except (TypeError, ValueError, OverflowError):
            observation_now = processing_now
        detector = self.detector
        semantic = self.semantic
        try:
            if detector is not None:
                self.ingest(detector.observe(frame, now=observation_now))
            if semantic is not None:
                with self._lock:
                    semantic_due = (
                        force_semantic
                        or self._last_semantic_at is None
                        or processing_now - self._last_semantic_at >= self.semantic_cooldown_s
                    )
                if semantic_due:
                    with self._lock:
                        self._last_semantic_at = processing_now
                    self.ingest(semantic.observe(
                        frame,
                        world=self.store.snapshot(now=processing_now),
                        now=observation_now,
                    ))
            with self._lock:
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
        self.store.set_backend_status("vision_runtime", self.status())
        result = self.store.snapshot(now=processing_now)
        result["vision"] = self.status()
        return result

    def snapshot(self) -> dict[str, Any]:
        result = self.store.snapshot()
        result["vision"] = self.status()
        return result


__all__ = [
    "CapturedFrame",
    "FrameDetector",
    "FrameSource",
    "MssFrameSource",
    "SemanticBackend",
    "VisionObservation",
    "VisionRuntime",
    "VisionWorker",
    "optional_dependency_status",
]
