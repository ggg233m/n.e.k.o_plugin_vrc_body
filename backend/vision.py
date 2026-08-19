"""可选的视觉编排层。

本模块刻意不依赖具体运行时。未来的画面采集、YOLO 或 VLM 适配器可以实现
这些小型后端协议并发布观测，而不进入 AnyaDance 控制线程。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
import base64
import importlib.util
import json
import os
import re
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.request import Request, urlopen

from .world_state import WorldEntity, WorldEvent, WorldStateStore


@dataclass(frozen=True)
class VisionObservation:
    """一批世界观测。

    ``remove_entity_ids`` 用于生命周期事件（例如玩家离开）在同一批次中
    撤销旧实体。它不是模型检测结果的隐式清理机制；只有发布者明确提供
    ID 时才会删除实体，避免把短暂漏检误当成离场。
    """

    entities: tuple[WorldEntity | Mapping[str, Any], ...] = ()
    events: tuple[WorldEvent | Mapping[str, Any], ...] = ()
    source: str = "vision"
    observed_at: float | None = None
    frame_id: str | None = None
    # Keep lifecycle fields at the end so existing positional constructors remain
    # compatible with the original observation protocol.
    remove_entity_ids: tuple[str, ...] = ()
    remove_source: str | None = None
    uncertainties: tuple[str, ...] = ()


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
    # ``find_spec`` raises ``ModuleNotFoundError`` for a nested module when its
    # parent package is absent (for example on non-Windows hosts).  Keep this
    # probe best-effort so a missing WinRT wheel never prevents the backend
    # from starting.  The granular keys are useful to explain why DXcam's
    # WinRT candidate was or was not added to the capture probe list.
    dxcam_available = _module_available("dxcam")
    winrt_available = _module_available("winrt")
    winrt_capture_available = _module_available("winrt.windows.graphics.capture")
    return {
        "opencv": importlib.util.find_spec("cv2") is not None,
        "mss": importlib.util.find_spec("mss") is not None,
        "dxcam": dxcam_available,
        "winrt": winrt_available,
        "winrt_graphics_capture": winrt_capture_available,
        "dxcam_winrt": dxcam_available and winrt_capture_available,
        "PIL": importlib.util.find_spec("PIL") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "ultralytics": importlib.util.find_spec("ultralytics") is not None,
        "openvr": importlib.util.find_spec("openvr") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


def _module_available(module_name: str) -> bool:
    """Return whether an optional module can be resolved without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class MssFrameSource:
    """可选的纯 mss 桌面采集器，不依赖插件 SDK 或模型包。

    ``mss`` exposes monitor ``0`` as the virtual desktop and monitors
    ``1..N`` as physical outputs.  A BitBlt failure can be output-specific,
    so the source probes physical outputs and rotates to the next one after a
    failed grab instead of getting stuck on the first monitor forever.
    """

    name = "mss"

    def __init__(
        self,
        *,
        monitor_index: int = -1,
        region: Mapping[str, Any] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._capture: Any = None
        self._monitor: Mapping[str, int] | None = None
        self._monitors: list[Mapping[str, int]] = []
        self._candidate_indices: list[int] = []
        self._candidate_pos = 0
        self._requested_monitor_index = int(monitor_index)
        self._active_monitor_index: int | None = None
        self._candidate_errors: dict[str, str] = {}
        self._np: Any = None
        self._image: Any = None
        self._closed = False
        self._last_error: str | None = None
        self._frames = 0
        try:
            import mss  # type: ignore[import-not-found]

            factory = getattr(mss, "MSS", getattr(mss, "mss", None))
            if factory is None:
                raise RuntimeError("mss does not expose a capture factory")
            self._capture = factory()
            monitors = list(getattr(self._capture, "monitors", ()))
            if not monitors:
                raise RuntimeError("mss reported no monitors")
            normalized: list[Mapping[str, int]] = []
            for item in monitors:
                base = dict(item)
                if region is not None:
                    for key in ("left", "top", "width", "height"):
                        if key in region:
                            base[key] = int(region[key])
                if base.get("width", 0) <= 0 or base.get("height", 0) <= 0:
                    raise ValueError("capture region must have positive width and height")
                normalized.append({
                    key: int(base[key]) for key in ("left", "top", "width", "height")
                })
            self._monitors = normalized
            physical = list(range(1, len(normalized)))
            if monitor_index >= 0:
                preferred = [int(monitor_index)] if int(monitor_index) < len(normalized) else []
                self._candidate_indices = preferred + [
                    index for index in physical + [0] if index not in preferred
                ]
                if not preferred:
                    self._last_error = f"requested monitor index {int(monitor_index)} is unavailable; probing all outputs"
            else:
                # Prefer an actual output over MSS's virtual-desktop entry.
                self._candidate_indices = physical + ([0] if normalized else [])
            if not self._candidate_indices:
                raise RuntimeError("mss reported no usable monitor regions")
            self._select_candidate_locked(0)
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
                "available": (
                    self._capture is not None
                    and self._monitor is not None
                    and not self._closed
                    and self._last_error is None
                ),
                "name": self.name,
                "monitor": dict(self._monitor or {}),
                "monitor_index": self._active_monitor_index,
                "requested_monitor_index": self._requested_monitor_index,
                "candidate_indices": list(self._candidate_indices),
                "candidate_errors": dict(self._candidate_errors),
                "frames": self._frames,
                "last_error": self._last_error,
            }

    def _select_candidate_locked(self, position: int) -> None:
        if not self._candidate_indices or not self._monitors:
            self._monitor = None
            self._active_monitor_index = None
            return
        self._candidate_pos = int(position) % len(self._candidate_indices)
        index = self._candidate_indices[self._candidate_pos]
        self._monitor = self._monitors[index]
        self._active_monitor_index = index

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
                message = f"{type(exc).__name__}: {exc}"[:256]
                if self._active_monitor_index is not None:
                    self._candidate_errors[str(self._active_monitor_index)] = message
                if len(self._candidate_indices) > 1:
                    self._select_candidate_locked(self._candidate_pos + 1)
                    next_index = self._active_monitor_index
                    self._last_error = (
                        f"monitor capture failed ({message}); trying monitor {next_index}"
                    )[:256]
                else:
                    self._last_error = message
            return None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            capture = self._capture
            self._capture = None
            self._monitor = None
        if capture is not None:
            try:
                capture.close()
            except Exception:
                pass


class DxcamFrameSource:
    """Optional Windows desktop-mirror capture using DXcam.

    DXcam is intentionally imported lazily so the plugin remains usable on
    machines without the optional capture package. ``grab`` is latest-frame
    only; no frame queue is retained here.
    """

    name = "dxcam"

    def __init__(
        self,
        *,
        region: Mapping[str, Any] | None = None,
        output_idx: int = -1,
        device_idx: int = -1,
        backend: str = "auto",
    ) -> None:
        self._lock = threading.Lock()
        self._camera: Any = None
        self._region = None
        self._closed = False
        self._frames = 0
        self._last_error: str | None = None
        self._requested_output_idx = int(output_idx)
        self._requested_device_idx = int(device_idx)
        self._requested_backend = str(backend).strip().lower() or "auto"
        if self._requested_backend not in {"auto", "dxgi", "winrt"}:
            self._requested_backend = "auto"
        self._selected_device_idx: int | None = None
        self._selected_output_idx: int | None = None
        self._selected_backend: str | None = None
        self._candidate_specs: list[tuple[int, int | None, str]] = []
        self._candidate_pos = 0
        self._candidate_errors: dict[str, str] = {}
        self._dxcam: Any = None
        # Keep this separate from the selected backend: ``auto`` may begin on
        # DXGI and only switch to WinRT after a failed grab.  Exposing the
        # capability in status makes that fallback diagnosable before the
        # first frame arrives.
        self._winrt_available = _module_available("winrt.windows.graphics.capture")
        try:
            import dxcam  # type: ignore[import-not-found]

            self._dxcam = dxcam
            if region is not None:
                self._region = tuple(int(region.get(key, 0)) for key in ("left", "top", "right", "bottom"))
                if self._region[2] <= self._region[0] or self._region[3] <= self._region[1]:
                    raise ValueError("DXcam region must have positive width and height")
            self._candidate_specs = self._build_candidates(dxcam)
            self._activate_candidate_locked(0)
            if self._camera is None and self._last_error is None:
                self._last_error = "DXcam could not initialize any adapter/output/backend"
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:256]

    def _build_candidates(self, dxcam: Any) -> list[tuple[int, int | None, str]]:
        if self._requested_device_idx >= 0:
            devices = [self._requested_device_idx]
        else:
            try:
                devices = list(range(max(1, len(dxcam.enum_dxgi_adapters()))))
            except Exception:
                devices = [0]
        output_map: dict[int, list[int]] = {}
        if self._requested_output_idx < 0:
            try:
                info = str(dxcam.output_info())
                for device_text, output_text in re.findall(r"Device\[(\d+)\]\s+Output\[(\d+)\]", info):
                    device = int(device_text)
                    output_map.setdefault(device, []).append(int(output_text))
            except Exception:
                output_map = {}
            if output_map:
                devices = [device for device in devices if device in output_map] or [min(output_map)]
        if self._requested_output_idx >= 0:
            outputs_by_device = {
                device: [self._requested_output_idx] for device in devices
            }
        else:
            outputs_by_device = {
                # ``None`` and output 0 both normally mean the primary
                # display; avoid probing the same output twice.
                device: [None, *[
                    index for index in sorted(set(output_map.get(device, [0])))
                    if index != 0
                ]]
                for device in devices
            }
            # When output_info is unavailable, retain a small bounded probe
            # instead of silently trusting only the primary output.
            if not output_map:
                outputs_by_device = {device: [None, 0, 1, 2, 3] for device in devices}
        if self._requested_backend != "auto":
            backends = [self._requested_backend]
        else:
            backends = ["dxgi"]
            if self._winrt_available:
                backends.append("winrt")
        return [
            (int(device), output, backend_name)
            for device in devices
            for output in outputs_by_device.get(device, [None])
            for backend_name in backends
        ]

    @staticmethod
    def _release_camera(camera: Any) -> None:
        if camera is None:
            return
        for method_name in ("stop", "release"):
            try:
                method = getattr(camera, method_name, None)
                if method is not None:
                    method()
            except Exception:
                pass

    def _create_camera(self, spec: tuple[int, int | None, str]) -> Any:
        if self._dxcam is None:
            return None
        device, output, backend = spec
        try:
            return self._dxcam.create(
                device_idx=device,
                output_idx=output,
                output_color="RGB",
                backend=backend,
            )
        except TypeError:
            # Older DXcam releases do not expose the backend selector.  It is
            # safe to retry only for the historical/default DXGI path.  An
            # explicit WinRT request must not silently produce a DXGI camera
            # while reporting ``backend=winrt`` in status.
            if backend != "dxgi":
                raise
            return self._dxcam.create(
                device_idx=device,
                output_idx=output,
                output_color="RGB",
            )

    def _activate_candidate_locked(self, position: int) -> bool:
        old_camera = self._camera
        self._camera = None
        self._release_camera(old_camera)
        if not self._candidate_specs or self._closed:
            return False
        for offset in range(len(self._candidate_specs)):
            candidate_pos = (int(position) + offset) % len(self._candidate_specs)
            spec = self._candidate_specs[candidate_pos]
            try:
                camera = self._create_camera(spec)
                if camera is None:
                    raise RuntimeError("DXcam returned no camera")
                self._candidate_pos = candidate_pos
                self._camera = camera
                self._selected_device_idx, self._selected_output_idx, self._selected_backend = spec
                self._last_error = None
                return True
            except Exception as exc:
                self._candidate_errors[self._format_spec(spec)] = f"{type(exc).__name__}: {exc}"[:256]
        self._selected_device_idx = None
        self._selected_output_idx = None
        self._selected_backend = None
        errors = list(self._candidate_errors.values())
        self._last_error = "; ".join(errors[-3:])[:500] or "DXcam could not initialize any candidate"
        return False

    @staticmethod
    def _format_spec(spec: tuple[int, int | None, str]) -> str:
        device, output, backend = spec
        output_label = "primary" if output is None else str(output)
        return f"device={device},output={output_label},backend={backend}"

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "available": self._camera is not None and not self._closed and self._last_error is None,
                "name": self.name,
                "region": self._region,
                "device_idx": self._selected_device_idx,
                "output_idx": self._selected_output_idx,
                "backend": self._selected_backend,
                "requested_device_idx": self._requested_device_idx,
                "requested_output_idx": self._requested_output_idx,
                "requested_backend": self._requested_backend,
                "winrt_available": self._winrt_available,
                "candidate_count": len(self._candidate_specs),
                "candidate_errors": dict(self._candidate_errors),
                "frames": self._frames,
                "last_error": self._last_error,
            }

    def read(self) -> Any:
        with self._lock:
            camera, region, closed = self._camera, self._region, self._closed
        if camera is None or closed:
            return None
        try:
            frame = camera.grab(region=region) if region is not None else camera.grab()
            with self._lock:
                self._frames += 1
                self._last_error = None
            return frame
        except Exception as exc:
            with self._lock:
                message = f"{type(exc).__name__}: {exc}"[:256]
                if self._candidate_specs:
                    spec = self._candidate_specs[self._candidate_pos]
                    self._candidate_errors[self._format_spec(spec)] = message
                    self._activate_candidate_locked(self._candidate_pos + 1)
                    if self._camera is None:
                        self._last_error = message
                else:
                    self._last_error = message
            return None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            camera = self._camera
            self._camera = None
        self._release_camera(camera)


class DesktopMirrorFrameSource:
    """Select DXcam first and fall back to MSS without changing the worker API.

    Both backends stay alive during probing.  This matters on systems where
    DXGI is denied for one GPU while GDI/MSS can still capture another output.
    """

    name = "desktop_mirror"

    def __init__(
        self,
        *,
        region: Mapping[str, Any] | None = None,
        monitor_index: int = -1,
        dxcam_device_idx: int = -1,
        dxcam_output_idx: int = -1,
        dxcam_backend: str = "auto",
    ) -> None:
        self._lock = threading.Lock()
        self._sources: list[FrameSource] = [
            DxcamFrameSource(
                region=region,
                device_idx=dxcam_device_idx,
                output_idx=dxcam_output_idx,
                backend=dxcam_backend,
            ),
            MssFrameSource(monitor_index=monitor_index, region=region),
        ]
        self._active_index = 0

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            active_index = self._active_index
            sources = list(self._sources)
        statuses = [dict(source.status()) for source in sources]
        active = statuses[active_index] if statuses else {"available": False}
        result = dict(active)
        result["name"] = self.name
        result["backend"] = getattr(sources[active_index], "name", "unknown") if sources else "unknown"
        result["active_backend_index"] = active_index
        result["backends"] = statuses
        result["available"] = any(bool(item.get("available")) for item in statuses)
        if not result.get("available"):
            errors = [str(item.get("last_error")) for item in statuses if item.get("last_error")]
            if errors:
                result["last_error"] = " | ".join(errors)[:500]
        return result

    def read(self) -> Any:
        with self._lock:
            sources = list(self._sources)
            active_index = self._active_index
        if not sources:
            return None
        for offset in range(len(sources)):
            index = (active_index + offset) % len(sources)
            frame = sources[index].read()
            if frame is not None:
                with self._lock:
                    self._active_index = index
                return frame
        return None

    def close(self) -> None:
        with self._lock:
            sources = list(self._sources)
        for source in sources:
            source.close()


class OpenVinoLocalDetector:
    """Safe OpenVINO adapter seam for YOLOX/depth/OCR model bundles.

    Model-specific preprocessing is deliberately injected through ``infer``;
    an absent model or runtime never fabricates detections and reports an
    explicit unavailable status instead. This keeps inference off the 120 Hz
    control thread while allowing a deployed bundle to provide real results.
    """

    name = "openvino"

    def __init__(self, *, model_path: str | None = None, infer: Callable[[Any], Mapping[str, Any]] | None = None) -> None:
        self._model_path = model_path
        self._infer = infer
        self._compiled = False
        self._last_error: str | None = None
        self._frames = 0
        if infer is not None:
            self._compiled = True
        elif importlib.util.find_spec("openvino") is None:
            self._last_error = "openvino is not installed"
        elif not model_path or not os.path.exists(model_path):
            self._last_error = "YOLOX-Tiny model_path is not configured"
        else:
            # The concrete YOLOX/depth/OCR graph is deployment-specific. Keep
            # the adapter unavailable until a validated infer callable is
            # supplied instead of loading an untrusted arbitrary graph.
            self._last_error = "model bundle requires a validated infer adapter"

    def status(self) -> Mapping[str, Any]:
        return {
            "available": self._compiled,
            "name": self.name,
            "models": ["yolox_tiny", "depth", "ocr"],
            "frames": self._frames,
            "last_error": self._last_error,
        }

    def observe(self, frame: Any, *, now: float) -> VisionObservation:
        if not self._compiled or self._infer is None:
            raise RuntimeError(self._last_error or "OpenVINO detector is unavailable")
        self._frames += 1
        payload = self._infer(frame)
        if not isinstance(payload, Mapping):
            raise ValueError("OpenVINO infer adapter must return an object")
        return VisionObservation(
            entities=tuple(payload.get("entities") or ()),
            events=tuple(payload.get("events") or ()),
            source=str(payload.get("source") or "openvino"),
            observed_at=now,
            frame_id=str(payload.get("frame_id")) if payload.get("frame_id") is not None else None,
            remove_entity_ids=tuple(payload.get("remove_entity_ids") or ()),
            remove_source=payload.get("remove_source"),
            uncertainties=tuple(str(item)[:160] for item in (payload.get("uncertainties") or ())[:16]),
        )


class OpenAICompatibleSemanticBackend:
    """Change-triggered structured VLM adapter with a 30/minute hard cap."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        max_per_minute: int = 30,
        timeout_s: float = 8.0,
        request_fn: Callable[[Request, float], Any] | None = None,
    ) -> None:
        self.endpoint = str(endpoint).strip()
        self.model = str(model).strip()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.max_per_minute = min(30, max(1, int(max_per_minute)))
        self.timeout_s = min(30.0, max(1.0, float(timeout_s)))
        self._request_fn = request_fn
        self._calls: deque[float] = deque(maxlen=30)
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._last_call_at: float | None = None

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "available": bool(self.endpoint and self.model),
                "name": self.name,
                "model": self.model,
                "max_per_minute": self.max_per_minute,
                "calls_last_minute": len(self._calls),
                "last_call_age_ms": None if self._last_call_at is None else round(max(0.0, (time.monotonic() - self._last_call_at) * 1000.0), 1),
                "last_error": self._last_error,
            }

    @staticmethod
    def _frame_data(frame: Any) -> str:
        if isinstance(frame, bytes):
            return base64.b64encode(frame).decode("ascii")
        save = getattr(frame, "save", None)
        if callable(save):
            from io import BytesIO
            buf = BytesIO()
            save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        # numpy-like arrays are encoded only when Pillow is available.
        try:
            from PIL import Image  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            image = Image.fromarray(np.asarray(frame).astype(np.uint8))
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            raise ValueError(f"frame cannot be encoded: {exc}") from exc

    def _call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        response = self._request_fn(request, self.timeout_s) if self._request_fn else urlopen(request, timeout=self.timeout_s)
        raw = response.read() if hasattr(response, "read") else response
        result = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        if not isinstance(result, Mapping):
            raise ValueError("semantic backend response must be an object")
        return result

    def observe(self, frame: Any, *, world: Mapping[str, Any], now: float) -> VisionObservation:
        current = time.monotonic()
        with self._lock:
            while self._calls and current - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) >= self.max_per_minute:
                self._last_error = "semantic rate limit reached"
                raise RuntimeError(self._last_error)
            self._calls.append(current)
            self._last_call_at = current
        schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array"},
                "events": {"type": "array"},
                "uncertainties": {"type": "array"},
            },
            "required": ["entities", "events", "uncertainties"],
            "additionalProperties": False,
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "vrc_world", "strict": True, "schema": schema}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe only observable VRChat world entities, interactions, spatial relations and events. Do not identify players or reproduce chat."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + self._frame_data(frame)}},
            ]}],
        }
        try:
            result = self._call(payload)
            choices = result.get("choices") if isinstance(result, Mapping) else None
            content = ((choices or [{}])[0] or {}).get("message", {}).get("content", "{}") if isinstance(choices, list) else "{}"
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) for item in content
                    if isinstance(item, Mapping)
                ) or "{}"
            structured = json.loads(content) if isinstance(content, str) else content
            if not isinstance(structured, Mapping):
                raise ValueError("semantic response content must be a JSON object")
            self._last_error = None
            return VisionObservation(
                entities=tuple(structured.get("entities") or ()),
                events=tuple(structured.get("events") or ()),
                source="openai_vlm",
                observed_at=now,
                frame_id=f"vlm-{int(current * 1000)}",
                uncertainties=tuple(str(item)[:160] for item in (structured.get("uncertainties") or ())[:16]),
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:256]
            raise


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
        capture_only: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.source = source
        self.interval_s = min(2.0, max(0.01, float(interval_s)))
        self.queue: Queue[CapturedFrame] = Queue(maxsize=max(1, min(4, int(queue_size))))
        self.capture_only = bool(capture_only)
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
            candidates = [item for item in (self.runtime.detector, self.runtime.semantic) if item is not None]
            if not candidates and not self.capture_only:
                self._last_error = "no detector or semantic backend is configured"
                return False
            available = False
            errors: list[str] = []
            for candidate in candidates:
                try:
                    status = candidate.status()
                    if not isinstance(status, Mapping) or status.get("available") is not False:
                        available = True
                        break
                    errors.append(str(status.get("last_error") or status.get("reason") or "unavailable"))
                except Exception:
                    available = True
                    break
            if candidates and not available:
                self._last_error = "; ".join(errors)[:256] or "configured vision backends are unavailable"
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
                "capture_only": self.capture_only,
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
        def array_field(mapping: Mapping[str, Any], name: str) -> tuple[Any, ...]:
            raw = mapping.get(name)
            if raw is None:
                return ()
            if not isinstance(raw, (list, tuple, set, frozenset)):
                raise ValueError(f"{name} must be an array")
            return tuple(raw)

        if isinstance(observation, VisionObservation):
            raw_remove_ids = observation.remove_entity_ids
            if raw_remove_ids is None:
                normalized_remove_ids: tuple[Any, ...] = ()
            elif isinstance(raw_remove_ids, (tuple, list, set, frozenset)):
                normalized_remove_ids = tuple(raw_remove_ids)
            else:
                raise ValueError("remove_entity_ids must be an array")
            raw_remove_source = observation.remove_source
            if raw_remove_source is not None:
                if not isinstance(raw_remove_source, str) or not raw_remove_source.strip():
                    raise ValueError("remove_source must be a non-empty string")
                normalized_remove_source: str | None = raw_remove_source.strip()
            else:
                normalized_remove_source = None
            value = replace(
                observation,
                remove_entity_ids=normalized_remove_ids,
                remove_source=normalized_remove_source,
                uncertainties=tuple(str(item)[:160] for item in (observation.uncertainties or ())[:16]),
            )
        else:
            if not isinstance(observation, Mapping):
                raise ValueError("observation must be an object")
            normalized_remove_ids = array_field(observation, "remove_entity_ids")
            raw_remove_source = observation.get("remove_source")
            if raw_remove_source is not None:
                if not isinstance(raw_remove_source, str) or not raw_remove_source.strip():
                    raise ValueError("remove_source must be a non-empty string")
                normalized_remove_source: str | None = raw_remove_source.strip()
            else:
                normalized_remove_source = None
            value = VisionObservation(
                entities=array_field(observation, "entities"),
                events=array_field(observation, "events"),
                remove_entity_ids=normalized_remove_ids,
                remove_source=normalized_remove_source,
                source=str(observation.get("source") or "vision"),
                observed_at=observation.get("observed_at"),
                frame_id=str(observation.get("frame_id")) if observation.get("frame_id") is not None else None,
                uncertainties=tuple(str(item)[:160] for item in (observation.get("uncertainties") or ())[:16]),
            )
        ingest_kwargs: dict[str, Any] = {
            "source": value.source,
            "observed_at": value.observed_at,
        }
        if value.uncertainties:
            ingest_kwargs["uncertainties"] = value.uncertainties
        if value.remove_entity_ids or value.remove_source is not None:
            ingest_kwargs["remove_entity_ids"] = value.remove_entity_ids
            ingest_kwargs["remove_source"] = value.remove_source
        result = self.store.ingest(value.entities, value.events, **ingest_kwargs)
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
            detector_available = detector is not None
            if detector is not None:
                try:
                    detector_status = detector.status()
                    detector_available = not (
                        isinstance(detector_status, Mapping)
                        and detector_status.get("available") is False
                    )
                except Exception:
                    detector_available = True
            if detector_available and detector is not None:
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
    "DxcamFrameSource",
    "DesktopMirrorFrameSource",
    "MssFrameSource",
    "OpenVinoLocalDetector",
    "OpenAICompatibleSemanticBackend",
    "SemanticBackend",
    "VisionObservation",
    "VisionRuntime",
    "VisionWorker",
    "optional_dependency_status",
]
