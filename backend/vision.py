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
    # 将生命周期字段放在末尾，保持与原观测协议的已有位置参数兼容。
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


_OPTIONAL_STATUS_CACHE: dict[str, bool] | None = None
_OPTIONAL_STATUS_AT = 0.0
_OPTIONAL_STATUS_LOCK = threading.Lock()
_OPTIONAL_STATUS_TTL_S = 5.0


def optional_dependency_status(*, refresh: bool = False) -> dict[str, bool]:
    """不导入重量级模型包，只报告可选能力是否存在。"""
    global _OPTIONAL_STATUS_CACHE, _OPTIONAL_STATUS_AT
    now = time.monotonic()
    with _OPTIONAL_STATUS_LOCK:
        if (
            not refresh
            and _OPTIONAL_STATUS_CACHE is not None
            and now - _OPTIONAL_STATUS_AT < _OPTIONAL_STATUS_TTL_S
        ):
            return dict(_OPTIONAL_STATUS_CACHE)
    # 父包不存在时（例如非 Windows 主机），``find_spec`` 对嵌套模块会抛出
    # ``ModuleNotFoundError``。这里采用尽力探测，不能让缺少 WinRT wheel
    # 阻止后端启动；细分状态键用于解释 DXcam 的 WinRT 候选为何被加入或
    # 未被加入采集探测列表。
    dxcam_available = _module_available("dxcam")
    winrt_available = _module_available("winrt")
    winrt_capture_available = _module_available("winrt.windows.graphics.capture")
    result = {
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
    with _OPTIONAL_STATUS_LOCK:
        _OPTIONAL_STATUS_CACHE = dict(result)
        _OPTIONAL_STATUS_AT = now
    return result


def _module_available(module_name: str) -> bool:
    """返回可选模块是否可以在不导入的情况下被解析。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def encode_frame_jpeg(
    frame: Any,
    *,
    max_width: int = 0,
    quality: int = 70,
) -> tuple[bytes, int, int]:
    """把一帧编码成 JPEG 字节，并返回 ``(数据, 宽, 高)``。

    ``max_width`` 大于 0 时按比例降采样到该宽度以内。给 LLM 的图不需要原始
    分辨率——降采样同时压低了 token 成本和「盯着模糊像素脑补」的风险。

    已经是 ``bytes`` 的帧按原样返回（假定调用方已编码），此时尺寸未知，返回
    ``(data, 0, 0)``：宁可说不知道，也不猜一个尺寸。
    """
    if isinstance(frame, bytes):
        return frame, 0, 0
    from io import BytesIO

    quality = min(95, max(30, int(quality)))
    image = None
    save = getattr(frame, "save", None)
    if callable(save) and hasattr(frame, "size"):
        image = frame
    else:
        try:
            from PIL import Image  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            image = Image.fromarray(np.asarray(frame).astype(np.uint8))
        except Exception as exc:
            raise ValueError(f"frame cannot be encoded: {exc}") from exc
    try:
        width, height = int(image.size[0]), int(image.size[1])
        if max_width > 0 and width > max_width and width > 0:
            scale = float(max_width) / float(width)
            target = (max_width, max(1, int(round(height * scale))))
            # 到这里 image 可能是任何实现了 save/size 的对象，不一定来自 Pillow，
            # 所以采样常量要能在 Pillow 缺席时退化——2 就是 PIL 的 BILINEAR。
            resample: Any = 2
            try:
                from PIL import Image as _Image  # type: ignore[import-not-found]
                resample = _Image.BILINEAR
            except Exception:
                pass
            image = image.resize(target, resample)
            width, height = int(image.size[0]), int(image.size[1])
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), width, height
    except Exception as exc:
        raise ValueError(f"frame cannot be encoded: {exc}") from exc


def find_window_region(title: str) -> dict[str, int] | None:
    """通过标题查找 Windows 顶层窗口并返回其屏幕坐标。

    返回值是 ``{"left": x, "top": y, "right": x2, "bottom": y2}``，可直接
    传给 ``DxcamFrameSource`` 或 ``MssFrameSource`` 的 ``region`` 参数。
    仅在 Windows 上可用；其他平台返回 ``None``。非 Windows 或窗口未找到时
    也返回 ``None``，不抛出异常。

    矩形会被夹到虚拟桌面范围内。窗口被拖出屏幕边缘时 ``GetWindowRect`` 会
    返回越界坐标，而 DXcam 直接拒绝整块区域（``ValueError: Invalid Region``），
    采集因此归零。夹取按**虚拟桌面**而不是主显示器：多屏时副屏坐标本就超出
    主屏范围，按主屏夹会把副屏上的窗口整个裁掉。

    被夹掉时额外返回 ``clamped_px``（四边各自被夹掉的像素数）与
    ``clamped``。采集区域一变，FOV→``bearing_deg`` 的映射基准就跟着变，
    这件事必须能被看见，不能默默发生。
    """
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return None
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return None
        # SM_XVIRTUALSCREEN=76, SM_YVIRTUALSCREEN=77,
        # SM_CXVIRTUALSCREEN=78, SM_CYVIRTUALSCREEN=79
        virtual_left = user32.GetSystemMetrics(76)
        virtual_top = user32.GetSystemMetrics(77)
        virtual_width = user32.GetSystemMetrics(78)
        virtual_height = user32.GetSystemMetrics(79)
        if virtual_width <= 0 or virtual_height <= 0:
            # 取不到虚拟桌面尺寸时不猜：原样返回，让采集器自己报错，
            # 好过按一个编造的边界把区域裁错。
            return {
                "left": rect.left,
                "top": rect.top,
                "right": rect.right,
                "bottom": rect.bottom,
            }
        virtual_right = virtual_left + virtual_width
        virtual_bottom = virtual_top + virtual_height
        left = max(virtual_left, min(int(rect.left), virtual_right - 1))
        top = max(virtual_top, min(int(rect.top), virtual_bottom - 1))
        right = max(left + 1, min(int(rect.right), virtual_right))
        bottom = max(top + 1, min(int(rect.bottom), virtual_bottom))
        region: dict[str, int] = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }
        clipped = {
            "left": left - int(rect.left),
            "top": top - int(rect.top),
            "right": int(rect.right) - right,
            "bottom": int(rect.bottom) - bottom,
        }
        if any(value for value in clipped.values()):
            region["clamped"] = True
            region["clamped_px"] = clipped
        return region
    except Exception:
        return None


def _normalize_region(region: Mapping[str, Any] | None) -> dict[str, int] | None:
    """把采集区域补全成同时含 ``left/top/right/bottom/width/height`` 的形式。

    ``find_window_region`` 返回 ``left/top/right/bottom``，而 MSS 的监视器字典用
    ``left/top/width/height``。两边各认一半，结果是窗口裁剪在 MSS 路径上只改了
    原点、尺寸仍是整块显示器。这里统一补齐，两种写法都能被两个采集器正确接受。
    """
    if region is None:
        return None
    try:
        left = int(region["left"])
        top = int(region["top"])
        if "right" in region and "bottom" in region:
            right = int(region["right"])
            bottom = int(region["bottom"])
        elif "width" in region and "height" in region:
            right = left + int(region["width"])
            bottom = top + int(region["height"])
        else:
            raise KeyError("right/bottom or width/height")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"capture region is missing usable bounds: {exc}") from exc
    if right <= left or bottom <= top:
        raise ValueError("capture region must have positive width and height")
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


class MssFrameSource:
    """可选的纯 mss 桌面采集器，不依赖插件 SDK 或模型包。

    ``mss`` 将监视器 ``0`` 暴露为虚拟桌面，将 ``1..N`` 暴露为物理输出。
    BitBlt 失败可能只影响某个输出，因此采集器会探测物理输出，并在抓取失败
    后切换到下一个输出，不会永远卡在第一个监视器上。
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

            region = _normalize_region(region)
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
                # 优先选择实际输出，而不是 MSS 的虚拟桌面条目。
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
    """使用 DXcam 的可选 Windows 桌面镜像采集器。

    这里刻意延迟导入 DXcam，使未安装可选采集包的机器仍可使用插件。
    ``grab`` 只读取最新帧，本类不保留帧队列。
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
        # 分开计「尝试」与「真的拿到帧」。DXcam 的 ``new_frame_only=True`` 在没有
        # 新帧时合法返回 ``None``，把两者合成一个计数器会让「相机在线但一帧不产」
        # 看起来和正常采集完全一样。
        self._frames = 0
        self._grabs_attempted = 0
        self._empty_grabs = 0
        self._last_error: str | None = None
        self._exhausted_error: str | None = None
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
        # 将能力状态与当前后端分开：``auto`` 可能先使用 DXGI，抓取失败后
        # 才切换到 WinRT。这样 status 可以在首帧到达前说明回退原因。
        self._winrt_available = _module_available("winrt.windows.graphics.capture")
        try:
            import dxcam  # type: ignore[import-not-found]

            self._dxcam = dxcam
            normalized = _normalize_region(region)
            if normalized is not None:
                self._region = tuple(normalized[key] for key in ("left", "top", "right", "bottom"))
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
                # ``None`` 和输出 0 通常都表示主显示器，避免重复探测同一输出。
                device: [None, *[
                    index for index in sorted(set(output_map.get(device, [0])))
                    if index != 0
                ]]
                for device in devices
            }
            # output_info 不可用时，保留一个小型有界探测列表，不要默默只信任
            # 主输出。
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
        base = {"device_idx": device, "output_idx": output, "output_color": "RGB"}
        # DXcam 的后处理默认走 cv2，而冻结宿主没有 cv2：默认路径下 grab() 会
        # 永远返回 None，而 status 仍然报告 available=True。numpy 后端是
        # Cython 加速路径且自带 cv2 兜底，因此无条件优先——开发机和宿主由此
        # 走同一条路，不会再出现只在部署时才暴露的采集失败。
        first_exc: TypeError | None = None
        try:
            return self._dxcam.create(**base, backend=backend, processor_backend="numpy")
        except TypeError as exc:
            first_exc = exc
        # 较旧的 DXcam 版本不暴露 backend 参数。仅对历史默认的 DXGI 路径重试是
        # 安全的；显式请求 WinRT 时不能偷偷创建 DXGI 摄像头，却在 status 中
        # 报告 ``backend=winrt``。
        if backend != "dxgi":
            raise first_exc  # type: ignore[misc]
        # dxgi 的向后兼容回退：去掉 backend 参数，兼容不支持该参数的旧版 DXcam。
        return self._dxcam.create(**base, processor_backend="numpy")

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
                # 构造成功不等于能采集：越界区域下每个 candidate 都能建出相机，
                # 却在 grab 时抛同一个 ValueError。轮换一圈后如果每个 candidate
                # 都留下过错误，就不能再把 _last_error 清空——否则 status() 会在
                # 采集已经彻底不可用时报告 available=True、last_error=None，
                # agent 看到的是「还没开始」而不是「已经坏了」。
                self._last_error = self._exhausted_error_locked()
                return True
            except Exception as exc:
                self._candidate_errors[self._format_spec(spec)] = f"{type(exc).__name__}: {exc}"[:256]
        self._selected_device_idx = None
        self._selected_output_idx = None
        self._selected_backend = None
        errors = list(self._candidate_errors.values())
        self._last_error = "; ".join(errors[-3:])[:500] or "DXcam could not initialize any candidate"
        return False

    def _exhausted_error_locked(self) -> str | None:
        """所有 candidate 都失败过时返回汇总错误，否则返回 ``None``。

        判据是「每个 spec 都在 candidate_errors 里留过记录」，而不是「刚才这次失败了」：
        单个 candidate 偶发失败后切到另一个能用的输出属于正常回退，不该报错。
        """
        if not self._candidate_specs:
            return None
        if any(
            self._format_spec(spec) not in self._candidate_errors
            for spec in self._candidate_specs
        ):
            return None
        errors = list(self._candidate_errors.values())
        return (
            "all DXcam candidates failed: " + "; ".join(errors[-3:])
        )[:500]

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
                # ``frames`` 保持原语义（成功产出的帧数）以兼容既有读取方；
                # ``grabs_attempted``/``empty_grabs`` 用来区分「没在采」和
                # 「在采但一帧都没出来」。
                "frames": self._frames,
                "grabs_attempted": self._grabs_attempted,
                "empty_grabs": self._empty_grabs,
                "last_error": self._last_error,
            }

    # 连续拿到 ``None`` 多少次算「相机在线但不产帧」。默认采集间隔是 100 ms
    # （``VisionConfig.interval_ms``），所以 30 次约等于 3 秒——足够跨过切场景、
    # Alt-Tab 这类正常的短暂无新帧，又不至于让真正的故障沉默太久。
    _EMPTY_GRAB_LIMIT = 30

    def read(self) -> Any:
        with self._lock:
            camera, region, closed = self._camera, self._region, self._closed
        if camera is None or closed:
            return None
        try:
            frame = camera.grab(region=region) if region is not None else camera.grab()
            with self._lock:
                self._grabs_attempted += 1
                if frame is None:
                    # ``new_frame_only=True`` 时没有新帧就返回 None，这本身合法，
                    # 不能当异常。但连续不产帧和采集坏掉在外部看来无法区分，
                    # 所以计数并在越过阈值后明确报错——沉默才是这里真正的 bug。
                    self._empty_grabs += 1
                    if self._empty_grabs >= self._EMPTY_GRAB_LIMIT:
                        self._last_error = (
                            f"DXcam camera is live but produced no frame in "
                            f"{self._empty_grabs} consecutive grabs"
                        )
                else:
                    self._frames += 1
                    self._empty_grabs = 0
                    self._last_error = self._exhausted_error_locked()
            return frame
        except Exception as exc:
            with self._lock:
                self._grabs_attempted += 1
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
    """优先选择 DXcam，失败时回退到 MSS，且不改变 worker 接口。

    探测期间两个后端都保持可用。这对某些系统很重要：一个 GPU 可能拒绝
    DXGI，而 GDI/MSS 仍能采集另一个输出。
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


class WindowTrackedFrameSource:
    """按 TTL 重新解析目标窗口矩形，窗口移动或改分辨率后重建内部采集源。

    DXcam 与 MSS 都在构造时就把区域固定下来，没有改区域的接口，所以这里只能
    重建。重建 DXGI 复制会话是重操作，因此只在矩形**真的**变了之后才做，TTL
    到点只是解析一次窗口坐标。

    窗口暂时找不到时（最小化、切到别的桌面、VRChat 还没起来）保留上一次的矩形，
    而不是立刻退回全屏：把整个桌面喂给检测器比暂时抓一块过期区域更糟。
    """

    name = "window_tracked"

    def __init__(
        self,
        *,
        title: str,
        factory: Callable[[Mapping[str, int] | None], FrameSource],
        interval_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        resolver: Callable[[str], dict[str, int] | None] = find_window_region,
    ) -> None:
        self._lock = threading.Lock()
        self._title = str(title)[:256]
        self._factory = factory
        self._interval_s = max(0.0, float(interval_s))
        self._clock = clock
        self._resolver = resolver
        self._closed = False
        self._rebuilds = 0
        self._last_error: str | None = None
        region = self._resolve()
        self._region: dict[str, int] | None = region
        self._window_found = region is not None
        self._checked_at = clock()
        self._source: FrameSource | None = factory(region)

    def _resolve(self) -> dict[str, int] | None:
        try:
            return self._resolver(self._title)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:256]
            return None

    def _refresh(self) -> None:
        if self._interval_s <= 0.0:
            return
        with self._lock:
            now = self._clock()
            if self._closed or now - self._checked_at < self._interval_s:
                return
            self._checked_at = now
        region = self._resolve()
        with self._lock:
            self._window_found = region is not None
            if region is None or region == self._region or self._closed:
                return
            old = self._source
            self._region = region
            # 先把旧采集源摘掉：同一输出上并存两个 DXGI 复制会话会直接失败，
            # 因此必须先关旧的再建新的。这中间的 read() 返回 None，只掉一帧。
            self._source = None
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        try:
            replacement = self._factory(region)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:256]
            return
        with self._lock:
            if self._closed:
                stale = replacement
            else:
                self._source = replacement
                self._rebuilds += 1
                self._last_error = None
                stale = None
        if stale is not None:
            try:
                stale.close()
            except Exception:
                pass

    def read(self) -> Any:
        self._refresh()
        with self._lock:
            source = None if self._closed else self._source
        if source is None:
            return None
        return source.read()

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            source = self._source
            region = dict(self._region) if self._region else None
            found = self._window_found
            rebuilds = self._rebuilds
            last_error = self._last_error
        if source is None:
            result: dict[str, Any] = {"available": False}
        else:
            try:
                result = dict(source.status())
            except Exception as exc:
                result = {"available": False, "last_error": f"{type(exc).__name__}: {exc}"[:256]}
        result["name"] = self.name
        result["backend"] = getattr(source, "name", "unknown")
        result["window_title"] = self._title
        result["window_found"] = found
        result["window_region"] = region
        # 夹取量单独抬到顶层。它藏在 window_region 里等于没报告——采集区域被裁掉
        # 一块之后，FOV→bearing_deg 的映射基准就变了，读 status 的人必须一眼看到。
        result["window_clamped"] = bool(region.get("clamped")) if region else False
        result["window_clamped_px"] = dict(region.get("clamped_px") or {}) if region else {}
        result["window_rebuilds"] = rebuilds
        result["window_track_interval_ms"] = round(self._interval_s * 1000.0, 1)
        if last_error and not result.get("last_error"):
            result["last_error"] = last_error
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            source = self._source
            self._source = None
        if source is not None:
            source.close()


class OpenVinoLocalDetector:
    """面向 YOLOX/深度/OCR 模型包的安全 OpenVINO 适配接口。

    特定模型的预处理通过 ``infer`` 显式注入；模型或运行时缺失时不会伪造
    检测结果，而是报告明确的不可用状态。这样可以让推理脱离 120 Hz 控制
    线程，同时允许已部署的模型包提供真实结果。
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
            self._last_error = "model_path is not configured"
        else:
            # 具体的 YOLOX/深度/OCR 图取决于部署环境。应等待经过验证的 infer
            # 可调用对象，而不是加载不可信的任意图；在此之前保持适配器不可用。
            self._last_error = "model bundle requires a validated infer adapter"

    def status(self) -> Mapping[str, Any]:
        # 这里刻意不声明模型列表。本类只是注入式 ``infer`` 的适配壳，自己不加载
        # 任何图；之前硬编码的 ``["yolox_tiny", "depth", "ocr"]`` 在没有 infer 时
        # 是纯谎报，有 infer 时也无从得知对方真正跑的是什么。真实的本地检测器是
        # ``local_perception.OpenVinoLocalDetector``，其 status 才带能力声明。
        return {
            "available": self._compiled,
            "name": self.name,
            "adapter": "injected" if self._infer is not None else "none",
            "model_path_configured": bool(self._model_path),
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
    """按变化触发的结构化 VLM 适配器，每分钟最多调用 30 次。"""

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
        # 只有 Pillow 可用时，才编码类 numpy 的数组。
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
            if candidates and not available and not self.capture_only:
                self._last_error = "; ".join(errors)[:256] or "configured vision backends are unavailable"
                return False
            if candidates and not available:
                # 显式请求的仅采集 worker 在部署模型期间仍可持续传递最新帧。
                # ``VisionRuntime.process_frame`` 会跳过不可用后端，因此这条
                # 路径不会伪造实体。
                self._last_error = "; ".join(errors)[:256] or "configured vision backends are unavailable"
            self._stop.clear()
            self._running = True
            # 新建采集源等待首帧期间，不要暴露旧的世界快照。采集循环只有在
            # 收到真实帧后才打开这个门控。
            self.runtime.set_capture_state(False, "awaiting_first_frame")
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
        # 关闭采集源前先标记为过期。即使生产者/消费者线程需要一点时间退出，
        # 或旧观测仍在 TTL 内，消费者也会立即看到未知世界。
        self.runtime.set_capture_state(False, "stopped")
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
        activated = False
        while not self._stop.is_set():
            captured_at = self._clock()
            try:
                frame = self.source.read()
                if frame is not None:
                    # read() 可能在 close() 后才返回；停止后的帧必须丢弃，
                    # 否则会把生命周期门重新打开并把旧画面送入推理队列。
                    with self._lock:
                        stopped_after_read = self._stop.is_set()
                        if not stopped_after_read and not activated:
                            self.runtime.set_capture_state(True, "running")
                            activated = True
                    if stopped_after_read:
                        _release_frame(frame)
                        break
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
        frame_cache_interval_s: float = 1.0,
        frame_cache_max_width: int = 960,
        frame_cache_quality: int = 70,
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
        # 单槽最新帧缓存。给 agent 看的图不走世界状态，也不进 120 Hz 调度线程：
        # 它只在采集 worker 自己的消费线程里按间隔编码一次，之后所有拉取都命中
        # 缓存。存最新一帧而不是队列，是因为过期的画面比没有画面更危险。
        self._frame_cache_interval_s = max(0.0, float(frame_cache_interval_s))
        self._frame_cache_max_width = max(0, int(frame_cache_max_width))
        self._frame_cache_quality = min(95, max(30, int(frame_cache_quality)))
        self._frame_cache: dict[str, Any] | None = None
        self._frame_cache_at: float | None = None
        self._frame_cache_error: str | None = None
        # 采集生命周期与检测器可用性分离。将门控放在运行时中，可让所有消费者
        # （HTTP、世界桥、自主控制和导航器）统一认定已停止的来源为未知，即使
        # 有界世界存储中仍保留最近一帧。
        self._capture_active = True
        self._capture_reason = "active"
        self.store.set_backend_status("vision_runtime", self.status())

    def set_capture_state(self, active: bool, reason: str | None = None) -> None:
        """发布世界消费者是否可以使用新鲜帧。

        本方法不会删除持久化世界记忆，也不会伪造删除事件；它只会在新帧到达前
        屏蔽瞬时观测。关闭或重建采集句柄时，这一点尤其重要。
        """
        with self._lock:
            self._capture_active = bool(active)
            self._capture_reason = (
                "active" if active else (str(reason or "capture_stopped").strip()[:160] or "capture_stopped")
            )
            if not active:
                # 停止采集时丢掉缓存帧。留着它会让停止后的拉取返回停止前的画面，
                # 而 agent 无法区分「刚才」和「现在」。
                self._frame_cache = None
                self._frame_cache_at = None
        self.store.set_backend_status("vision_runtime", self.status())

    def capture_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._capture_active,
                "reason": self._capture_reason,
            }

    def _cache_frame(self, frame: Any, captured_at: float) -> None:
        """按间隔把最新帧编码进单槽缓存。

        必须在帧被 ``_release_frame`` 释放之前调用。编码失败只记录原因，绝不
        让采集或检测因此中断——看不到图是降级，掉帧是故障。
        """
        if self._frame_cache_interval_s < 0:
            return
        with self._lock:
            last = self._frame_cache_at
            interval = self._frame_cache_interval_s
        if last is not None and (captured_at - last) < interval:
            return
        try:
            data, width, height = encode_frame_jpeg(
                frame,
                max_width=self._frame_cache_max_width,
                quality=self._frame_cache_quality,
            )
        except Exception as exc:
            with self._lock:
                self._frame_cache_error = f"{type(exc).__name__}: {exc}"[:256]
            return
        with self._lock:
            self._frame_cache = {
                "data": data,
                "mime": "image/jpeg",
                "width": width,
                "height": height,
                "bytes": len(data),
            }
            self._frame_cache_at = captured_at
            self._frame_cache_error = None

    def latest_frame(self, *, max_age_ms: int = 3000) -> dict[str, Any]:
        """返回最近一次缓存的画面，超龄或采集停止时明确拒绝。

        这条路径刻意与 ``world_state`` 完全分离：帧只喂给 agent 理解，不产生
        实体、不产生事件，也不能拿去满足 ``body_reach_and_grab`` 的
        ``preconditions``。从画面得出的任何结论都是低置信视觉猜测。

        ``max_age_ms <= 0`` 表示不限龄，这是留给运行时内部调用方的逃生口。
        LLM 那一侧不允许走到这里——工具与 ``BackendService`` 都把下限抬到了
        250 ms，否则「要最新的画面」写成 0 反而会拿到最旧的一张。
        """
        now = self._clock()
        capture = self.capture_state()
        with self._lock:
            cached = self._frame_cache
            cached_at = self._frame_cache_at
            error = self._frame_cache_error
        if not capture["active"]:
            return {
                "available": False,
                "reason": capture["reason"] or "capture_stopped",
                "capture_active": False,
            }
        if cached is None or cached_at is None:
            return {
                "available": False,
                "reason": error or "no_frame_cached",
                "capture_active": True,
            }
        age_ms = max(0.0, (now - cached_at) * 1000.0)
        try:
            limit_ms = max(0.0, float(max_age_ms))
        except (TypeError, ValueError, OverflowError):
            limit_ms = 3000.0
        if limit_ms and age_ms > limit_ms:
            # 过期的画面比没有画面更危险：agent 会拿它当现在。
            return {
                "available": False,
                "reason": "frame_stale",
                "age_ms": round(age_ms, 1),
                "capture_active": True,
            }
        return {
            "available": True,
            "capture_active": True,
            "age_ms": round(age_ms, 1),
            "mime": cached["mime"],
            "width": cached["width"],
            "height": cached["height"],
            "bytes": cached["bytes"],
            "data": cached["data"],
        }

    def _mask_stopped_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """返回未知视图，同时不修改持久化世界存储。"""
        with self._lock:
            active = self._capture_active
            reason = self._capture_reason
        result = dict(snapshot)
        result["capture_active"] = active
        result["capture_reason"] = reason
        if active:
            return result
        result["available"] = False
        result["entities"] = []
        result["events"] = []
        uncertainties = [str(item)[:160] for item in (result.get("uncertainties") or ())]
        if "visual_capture_stopped" not in uncertainties:
            uncertainties.append("visual_capture_stopped")
        result["uncertainties"] = uncertainties[:16]
        status = dict(result.get("status") or {})
        status["entity_count"] = 0
        status["event_count"] = 0
        result["status"] = status
        return result

    def _mask_stopped_delta(self, delta: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            active = self._capture_active
            reason = self._capture_reason
        result = dict(delta)
        result["capture_active"] = active
        result["capture_reason"] = reason
        if active:
            return result
        world = result.get("world")
        if isinstance(world, Mapping):
            result["world"] = self._mask_stopped_snapshot(world)
        else:
            result["world"] = self._mask_stopped_snapshot({})
        result["navigation"] = {"status": "unknown", "safe_navigation": False}
        result["social"] = {
            "status": "unknown",
            "players_persisted": False,
            "chat_persisted": False,
        }
        result["uncertainty"] = list(result["world"].get("uncertainties") or ())
        result["changes"] = {
            "entities": [],
            "events": [],
            "removed_entity_ids": [],
            "removed_entity_count": 0,
        }
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            detector = self.detector
            semantic = self.semantic
            last_error = self._last_error
            capture_active = self._capture_active
            capture_reason = self._capture_reason
            frame_cached = self._frame_cache is not None
            frame_cache_at = self._frame_cache_at
            frame_cache_error = self._frame_cache_error
            frame_cache_interval_s = self._frame_cache_interval_s
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
            "capture_active": capture_active,
            "capture_reason": capture_reason,
            "detector": backend_status(detector, label="detector"),
            "semantic": backend_status(semantic, label="semantic"),
            "optional_dependencies": optional_dependency_status(),
            # 画面缓存独立于世界状态；这里只报告它是否可用，不把画面内容
            # 混进感知结论。
            "frame_cache": {
                "cached": frame_cached,
                "age_ms": (
                    None if frame_cache_at is None
                    else round(max(0.0, (self._clock() - frame_cache_at) * 1000.0), 1)
                ),
                "interval_s": frame_cache_interval_s,
                "last_error": frame_cache_error,
            },
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

    def ingest(
        self,
        observation: VisionObservation | Mapping[str, Any],
        *,
        _reactivate: bool = True,
    ) -> dict[str, Any]:
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
        capture_active = self.capture_state()["active"]
        # 明确的外部观测可以重新打开世界视图；采集 worker 的过期帧绝不
        # 能在 stop 之后写入存储，即使它已经越过了 process_frame 的首个检查。
        if not capture_active:
            if not _reactivate:
                result = self._mask_stopped_snapshot(self.store.snapshot())
                result["vision"] = self.status()
                return result
            self.set_capture_state(True, "external_observation")
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
        if not self.capture_state()["active"]:
            result = self._mask_stopped_snapshot(self.store.snapshot(now=processing_now))
            result["vision"] = self.status()
            return result
        try:
            observation_now = min(processing_now, float(observed_at)) if observed_at is not None else processing_now
        except (TypeError, ValueError, OverflowError):
            observation_now = processing_now
        # 在检测之前缓存：帧在 worker 的 finally 里就会被释放，之后再想编码
        # 只能拿到已关闭的句柄。缓存自身按间隔限流，不是每帧都编。
        self._cache_frame(frame, processing_now)
        # attach_vision/set_backends 可能与采集线程并行；在锁内取得稳定引用，
        # 后续推理不持有运行时锁，避免阻塞生命周期控制。
        with self._lock:
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
            if detector_available and detector is not None and self.capture_state()["active"]:
                self.ingest(detector.observe(frame, now=observation_now), _reactivate=False)
            if semantic is not None and self.capture_state()["active"]:
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
                    ), _reactivate=False)
            with self._lock:
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
        self.store.set_backend_status("vision_runtime", self.status())
        result = self._mask_stopped_snapshot(self.store.snapshot(now=processing_now))
        result["vision"] = self.status()
        return result

    def snapshot(self) -> dict[str, Any]:
        result = self._mask_stopped_snapshot(self.store.snapshot())
        result["vision"] = self.status()
        return result

    def delta(
        self,
        after_revision: int = 0,
        *,
        wait_ms: int = 250,
        limit: int = 16,
    ) -> dict[str, Any]:
        """返回受采集生命周期门控的世界增量。"""
        return self._mask_stopped_delta(
            self.store.delta(after_revision, wait_ms=wait_ms, limit=limit)
        )


# 原有适配器保留在上方，以兼容源码级引用；实际部署实现位于面向模型的模块中。
# 在此导入可以避免循环依赖：``local_perception`` 使用的所有协议/数据类定义
# 已经完成初始化。
from .local_perception import OpenVinoLocalDetector as OpenVinoLocalDetector  # noqa: E402,F401


__all__ = [
    "CapturedFrame",
    "encode_frame_jpeg",
    "find_window_region",
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
    "WindowTrackedFrameSource",
    "optional_dependency_status",
]
