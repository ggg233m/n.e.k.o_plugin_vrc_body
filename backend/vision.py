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
import math
import os
import re
import secrets
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


def overlay_boxes_geometry(
    boxes: Any,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """把归一化 bbox 换算成像素矩形，并丢掉画不出来的框。

    单独成函数是为了能在没有 Pillow 的机器上测：归一化→像素的换算是这里唯一
    可能出 off-by-one 的地方，而绘制那层必须要 PIL。``tests/test_vision_frame.py``
    明确不依赖可选视觉依赖，所以算术必须能被单独 import。

    ``bbox`` 是采集区域的 0..1 归一化坐标（``local_perception`` 按轴各自归一化，
    无 letterbox），缓存 JPEG 是同一区域的等比降采样，所以直接乘宽高即可，不需要
    letterbox 校正。退化框（宽或高不足 1 px）与非有限值被跳过而不是钳成一条线：
    画一条线出来会让人以为检测到了什么。
    """
    try:
        canvas_w = int(width)
        canvas_h = int(height)
    except (TypeError, ValueError, OverflowError):
        return []
    if canvas_w <= 0 or canvas_h <= 0:
        return []
    result: list[dict[str, Any]] = []
    for item in boxes or ():
        if not isinstance(item, Mapping):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            values = [float(bbox[index]) for index in range(4)]
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        left = int(round(min(max(values[0], 0.0), 1.0) * canvas_w))
        top = int(round(min(max(values[1], 0.0), 1.0) * canvas_h))
        right = int(round(min(max(values[2], 0.0), 1.0) * canvas_w))
        bottom = int(round(min(max(values[3], 0.0), 1.0) * canvas_h))
        # 先钳进画布再判退化，顺序不能反：整个落在画布外的框钳完会塌到角上，
        # 若此时把 left/top 抬到 canvas-1，它就复活成一个 1×1 的角点——而一个
        # 孤立像素会被读成「这里检测到了东西」。要求宽高各至少 1 px，也就顺带
        # 保证了留下来的框 left <= canvas_w-1、top <= canvas_h-1，可以直接画。
        left = min(max(left, 0), canvas_w)
        top = min(max(top, 0), canvas_h)
        right = min(max(right, 0), canvas_w)
        bottom = min(max(bottom, 0), canvas_h)
        if right - left < 1 or bottom - top < 1:
            continue
        attributes = item.get("attributes")
        attributes = attributes if isinstance(attributes, Mapping) else {}
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        target_id = str(item.get("id") or "").strip()
        # 没有实体 ID 的框无法被 LLM 安全地交给导航器。宁可不画，也不能生成一个
        # 看得见却无法解析回 target_id 的 T 编号。
        if not target_id:
            continue
        result.append({
            # T 编号只对本次叠框结果有效；真正提交导航时仍必须使用 target_id。
            "ref": f"T{len(result) + 1}",
            "id": target_id,
            "label": str(item.get("label") or "?"),
            "confidence": confidence,
            "rect": (left, top, right, bottom),
            "clipped": bool(attributes.get("apparent_height_clipped")),
            "bearing_deg": attributes.get("bearing_deg"),
        })
    return result


def draw_detection_overlay(
    data: bytes,
    boxes: Any,
    *,
    quality: int = 70,
    warning: str | None = None,
    geometry_out: list[dict[str, Any]] | None = None,
) -> tuple[bytes, int]:
    """在 JPEG 副本上画检测框，返回 ``(数据, 画出的框数)``。

    ``geometry_out`` 用于把同一次绘制生成的临时 T 编号交给调用方，进而随工具
    结果返回完整 target_id；它不改变原有返回值，外部诊断调用仍可只取数量。

    调用方传入的是已编码的 JPEG 字节；这里解码、绘制、重新编码，**绝不写回
    帧缓存**——缓存里必须保留原始像素，否则唤醒推送会拿到烧了框的图，而那张图
    再也无法还原。

    需要 Pillow。缺席时抛 ``ValueError``，由调用方降级成"给原图 + 说明原因"：
    看不到框是降级，掉帧才是故障。
    """
    from io import BytesIO

    try:
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]
    except Exception as exc:
        raise ValueError(f"overlay requires Pillow: {exc}") from exc
    try:
        image = Image.open(BytesIO(data))
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"frame cannot be decoded: {exc}") from exc
    geometry = overlay_boxes_geometry(boxes, width=image.size[0], height=image.size[1])
    if geometry_out is not None:
        # 调用方传入新列表收集与本次画面完全相同的 T 编号映射。复制字典，避免
        # 后续绘制代码意外把内部对象暴露给 JSON 返回值。
        geometry_out.extend(dict(box) for box in geometry)
    draw = ImageDraw.Draw(image)
    for box in geometry:
        left, top, right, bottom = box["rect"]
        # 贴边的框换个颜色并标 CLIPPED：表观高度已饱和，不能再当距离的单调
        # 函数用，看图的人必须能一眼分辨。
        color = (255, 96, 0) if box["clipped"] else (0, 224, 96)
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        # T1/T2 比完整 session ID 更容易被多模态模型从 960px 图里读出；完整 ID
        # 通过 overlay.candidates 同步返回，不能从图片文字里猜。
        caption = f"{box['ref']} {box['label']} {box['confidence']:.2f}"
        if box["clipped"]:
            caption += " CLIPPED"
        bearing = box.get("bearing_deg")
        if isinstance(bearing, (int, float)) and math.isfinite(float(bearing)):
            caption += f" {float(bearing):+.0f}°"
        # 标签画在框内侧顶部：画在外面时贴着画面上边的框会把文字挤出画布。
        text_y = top + 2 if top + 16 < image.size[1] else max(0, top - 14)
        draw.rectangle((left, text_y, left + 8 * len(caption) + 4, text_y + 12), fill=(0, 0, 0))
        draw.text((left + 2, text_y), caption, fill=color)
    if warning:
        # 警告烧在图上而不是只放进 JSON：JSON 字段容易被忽略，画面上的红条不会。
        draw.rectangle((0, 0, image.size[0], 16), fill=(160, 0, 0))
        draw.text((4, 3), warning[:120], fill=(255, 255, 255))
    try:
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=min(95, max(30, int(quality))))
        return buf.getvalue(), len(geometry)
    except Exception as exc:
        raise ValueError(f"overlay cannot be encoded: {exc}") from exc


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


#: 可见比例低于此值就算「画面已经不是目标窗口了」。留一点余量：任务栏、输入法
#: 候选框、Steam 通知都会盖掉窗口边角，那不该报警。
_WINDOW_OBSCURED_BELOW = 0.6


def window_visibility(title: str, *, grid: int = 12) -> dict[str, Any]:
    """报告目标窗口此刻是否真的露在外面。

    DXGI 桌面复制抓的是**合成后的桌面**、按输出而不是按窗口。VRChat 一被最小化
    或被别的窗口盖住，采集依旧成功，只是像素换成了压在上面那个窗口的内容——
    ``find_window_region`` 照样返回矩形、``window_found`` 照样是 true，检测器却在
    别人的画面里找人。这个失败必须能被看见，否则 world_state 会凭空长出玩家。

    可见度用**采样**而不是矩形求差：多个遮挡窗口互相重叠时面积会重复累加，两个
    各盖 60% 的窗口能算出 120% 被盖，实际还露着 40%。这里在窗口矩形上取
    ``grid × grid`` 个点，逐点问 ``WindowFromPoint`` 最上层是谁，重叠因此天然只
    算一次。``WindowFromPoint`` 本身会跳过隐藏、禁用和 ``WS_EX_TRANSPARENT``
    窗口，正好是我们想要的语义：点不到的覆盖物不算覆盖。

    非 Windows、窗口不存在或 Win32 调用失败时返回 ``{"found": False}``，不抛异常。
    可见度只是诊断信息，不该让采集本身挂掉。
    """
    try:
        import ctypes
        from ctypes import wintypes

        # 用独立的 WinDLL 实例而不是共享的 ``ctypes.windll``：下面要设 argtypes/
        # restype，而 windll 是进程级缓存，改它会波及别处的调用。
        user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]
        user32.FindWindowW.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]

        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return {"found": False}
        if user32.IsIconic(hwnd):
            # 最小化后 GetWindowRect 仍会返回一个屏外的旧矩形，在那上面采样毫无
            # 意义，直接判 0。
            return {"found": True, "minimized": True, "visible_ratio": 0.0}
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return {"found": True, "minimized": False}
        width = int(rect.right) - int(rect.left)
        height = int(rect.bottom) - int(rect.top)
        if width <= 0 or height <= 0:
            return {"found": True, "minimized": False, "visible_ratio": 0.0}

        root = user32.GetAncestor(hwnd, 2) or hwnd  # GA_ROOT=2
        steps = max(1, int(grid))
        point = wintypes.POINT()
        hits = 0
        total = 0
        blockers: dict[int, int] = {}
        for row in range(steps):
            point.y = int(rect.top) + int((row + 0.5) * height / steps)
            for col in range(steps):
                point.x = int(rect.left) + int((col + 0.5) * width / steps)
                total += 1
                topmost = user32.WindowFromPoint(point)
                if not topmost:
                    continue
                owner = user32.GetAncestor(topmost, 2) or topmost
                if owner == root:
                    hits += 1
                else:
                    blockers[owner] = blockers.get(owner, 0) + 1

        result: dict[str, Any] = {
            "found": True,
            "minimized": False,
            "visible_ratio": round(hits / total, 3),
        }
        if blockers:
            worst, _count = max(blockers.items(), key=lambda item: item[1])
            # 只解析盖得最多的那一个的标题：「被谁盖住了」是运维唯一想知道的，
            # 逐个取标题只是白花 Win32 调用。
            name = _hwnd_title(user32, worst)
            if name:
                result["occluded_by"] = name
        return result
    except Exception:
        return {"found": False}


def _hwnd_title(user32: Any, hwnd: Any) -> str:
    try:
        import ctypes

        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value[:128]
    except Exception:
        return ""


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

    找到矩形不等于抓到了游戏画面：DXGI 抓的是合成后的桌面，窗口被盖住时采集照样
    成功、内容却是压在上面那个窗口的。因此每个 TTL 还会顺手问一次
    ``window_visibility``，把结果原样抬进 ``status()``。这一层仍然只报告、照常出帧；
    要不要因此跳过推理由 ``VisionRuntime._observe_obscured`` 决定——采集源不该替
    消费者判断画面有没有用。
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
        visibility: Callable[[str], Mapping[str, Any]] | None = window_visibility,
    ) -> None:
        self._lock = threading.Lock()
        self._title = str(title)[:256]
        self._factory = factory
        self._interval_s = max(0.0, float(interval_s))
        self._clock = clock
        self._resolver = resolver
        self._visibility = visibility
        self._closed = False
        self._rebuilds = 0
        self._last_error: str | None = None
        region = self._resolve()
        self._region: dict[str, int] | None = region
        self._window_found = region is not None
        self._visible: dict[str, Any] = self._probe_visibility()
        self._checked_at = clock()
        self._source: FrameSource | None = factory(region)

    def _resolve(self) -> dict[str, int] | None:
        try:
            return self._resolver(self._title)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:256]
            return None

    def _probe_visibility(self) -> dict[str, Any]:
        """探测失败一律当「不知道」，不写进 ``_last_error``。

        可见度是诊断信息，把它的异常混进采集错误里，会让一个纯诊断故障看起来
        像采集挂了。
        """
        if self._visibility is None:
            return {}
        try:
            return dict(self._visibility(self._title))
        except Exception:
            return {}

    def _refresh(self) -> None:
        if self._interval_s <= 0.0:
            return
        with self._lock:
            now = self._clock()
            if self._closed or now - self._checked_at < self._interval_s:
                return
            self._checked_at = now
        region = self._resolve()
        # 可见度必须留在区域比较之外：它每次 Alt-Tab 都会变，塞进 region 会让
        # 相等判断失手，于是每切一次窗口就重建一次 DXGI 会话。
        visible = self._probe_visibility()
        with self._lock:
            self._window_found = region is not None
            self._visible = visible
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
            visible = dict(self._visible)
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
        # 可见度探测失败时 ratio 留 None（「不知道」），而不是 0。分不清这两者的话，
        # 一个坏掉的探测会被读成「窗口被完全盖住」，那是喊狼来了。
        minimized = bool(visible.get("minimized"))
        ratio = visible.get("visible_ratio")
        result["window_minimized"] = minimized
        result["window_visible_ratio"] = ratio
        result["window_obscured"] = bool(
            minimized or (ratio is not None and float(ratio) < _WINDOW_OBSCURED_BELOW)
        )
        occluder = visible.get("occluded_by")
        if occluder:
            result["window_occluded_by"] = occluder
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
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "semantic_type": {
                                "type": "string",
                                "enum": ["npc", "player", "avatar", "person", "humanoid", "object", "unknown"],
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "bbox": {
                                "anyOf": [
                                    {
                                        "type": "array",
                                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                    {"type": "null"},
                                ]
                            },
                            "state": {"type": "string"},
                        },
                        "required": ["id", "label", "semantic_type", "confidence", "bbox", "state"],
                        "additionalProperties": False,
                    },
                },
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "target_id": {"type": ["string", "null"]},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["type", "target_id", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "uncertainties": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["entities", "events", "uncertainties"],
            "additionalProperties": False,
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "vrc_world", "strict": True, "schema": schema}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": (
                    "Find and classify observable VRChat entities, including small, seated, crouched, "
                    "white or low-contrast NPCs that a standing-person detector may miss. Return a "
                    "normalized [left, top, right, bottom] bbox for every localized entity and classify "
                    "it as npc, player, avatar, person, humanoid, object or unknown. Do not identify "
                    "players, infer hidden entities, or reproduce chat."
                )},
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


def _semantic_entity_mapping(value: WorldEntity | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, WorldEntity):
        return {
            "id": value.id,
            "label": value.label,
            "confidence": value.confidence,
            "bbox": value.bbox,
            "state": value.state,
            "attributes": dict(value.attributes or {}),
            "relations": tuple(value.relations),
            "source": tuple(value.source),
            "observed_at": value.observed_at,
            "ttl_s": value.ttl_s,
        }
    return dict(value) if isinstance(value, Mapping) else {}


def _semantic_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    left, top, right, bottom = (
        min(1.0, max(0.0, left)),
        min(1.0, max(0.0, top)),
        min(1.0, max(0.0, right)),
        min(1.0, max(0.0, bottom)),
    )
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _semantic_iou(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _semantic_type(value: Mapping[str, Any]) -> str:
    attributes = value.get("attributes") if isinstance(value.get("attributes"), Mapping) else {}
    raw = value.get("semantic_type") or attributes.get("semantic_type") or value.get("label") or "unknown"
    normalized = str(raw).replace("\x00", "").strip().lower()[:32]
    aliases = {
        "non-player character": "npc",
        "character": "humanoid",
        "anime character": "humanoid",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {
        "npc", "player", "avatar", "person", "humanoid", "object",
        # 这些负类别不能作为自主 selector，但主 LLM 必须能明确标出干扰物，
        # 否则海报/镜子只能被压成 unknown，下一轮还会反复消耗图片 token。
        "poster", "screen", "mirror", "unknown",
    } else "unknown"


def _semantic_crop(data: bytes, bbox: tuple[float, float, float, float] | None) -> bytes | None:
    if bbox is None:
        return None
    try:
        from io import BytesIO
        from PIL import Image  # type: ignore[import-not-found]

        image = Image.open(BytesIO(data)).convert("RGB")
        width, height = image.size
        box = (
            max(0, min(width - 1, int(round(bbox[0] * width)))),
            max(0, min(height - 1, int(round(bbox[1] * height)))),
            max(1, min(width, int(round(bbox[2] * width)))),
            max(1, min(height, int(round(bbox[3] * height)))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        crop = image.crop(box)
        output = BytesIO()
        crop.save(output, format="JPEG", quality=65)
        return output.getvalue()
    except Exception:
        return None


def _semantic_descriptor(data: bytes | None) -> tuple[float, ...] | None:
    if not data:
        return None
    try:
        from io import BytesIO
        from PIL import Image  # type: ignore[import-not-found]

        image = Image.open(BytesIO(data)).convert("RGB").resize((8, 8))
        return tuple(float(item) / 255.0 for pixel in image.getdata() for item in pixel)
    except Exception:
        return None


def _semantic_similarity(
    first: tuple[float, ...] | None,
    second: tuple[float, ...] | None,
) -> float:
    if first is None or second is None or len(first) != len(second) or not first:
        return 0.0
    difference = sum(abs(a - b) for a, b in zip(first, second)) / len(first)
    return max(0.0, 1.0 - difference)


@dataclass(frozen=True)
class SemanticJob:
    data: bytes
    captured_at: float
    frame_id: str
    revision: int
    world: Mapping[str, Any]
    source: str = "openai_vlm"


class SemanticCandidateCache:
    """只驻留内存的语义候选与外观原型。"""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_candidates: int = 32,
        ttl_s: float = 30.0,
        session_token: str | None = None,
    ) -> None:
        self._clock = clock
        self.max_candidates = max(4, min(128, int(max_candidates)))
        self.ttl_s = max(5.0, min(300.0, float(ttl_s)))
        self.session_token = str(session_token or secrets.token_hex(4))[:24]
        self._lock = threading.RLock()
        self._items: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    def _prune_locked(self, now: float) -> None:
        expired = [
            candidate_id
            for candidate_id, item in self._items.items()
            if now - float(item.get("received_at", now)) > self.ttl_s
        ]
        for candidate_id in expired:
            self._items.pop(candidate_id, None)
        if len(self._items) <= self.max_candidates:
            return
        ranked = sorted(
            self._items.items(),
            key=lambda pair: float(pair[1].get("received_at", 0.0)),
            reverse=True,
        )[: self.max_candidates]
        self._items = dict(ranked)

    def _new_id_locked(self) -> str:
        value = f"semantic:session:{self.session_token}:{self._next_id}"
        self._next_id += 1
        return value

    def bind(
        self,
        raw: Mapping[str, Any],
        *,
        job: SemanticJob,
    ) -> tuple[dict[str, Any], str | None]:
        now = self._clock()
        bbox = _semantic_bbox(raw.get("bbox"))
        semantic_type = _semantic_type(raw)
        crop = _semantic_crop(job.data, bbox)
        descriptor = _semantic_descriptor(crop)
        matched_id: str | None = None
        matched_iou = 0.0

        # 先复用同一帧 YOLO 已经建立的稳定 ID；语义层只补分类，不拆出第二个实体。
        for item in job.world.get("entities") or ():
            if not isinstance(item, Mapping):
                continue
            overlap = _semantic_iou(bbox, _semantic_bbox(item.get("bbox")))
            if overlap >= 0.30 and overlap > matched_iou:
                candidate_id = str(item.get("id") or "").strip()
                if candidate_id:
                    matched_id = candidate_id
                    matched_iou = overlap

        with self._lock:
            self._prune_locked(now)
            if matched_id is None and bbox is not None:
                best_score = 0.0
                for candidate_id, item in self._items.items():
                    if str(item.get("semantic_type") or "unknown") != semantic_type:
                        continue
                    overlap = _semantic_iou(bbox, item.get("bbox"))
                    appearance = _semantic_similarity(descriptor, item.get("descriptor"))
                    # 轻微擦边不构成同一实体；否则两个相邻角色只要框有一像素
                    # 重叠就会永久共用 ID。外观可独立匹配，但也要过高阈值。
                    score = max(
                        overlap if overlap >= 0.30 else 0.0,
                        appearance if appearance >= 0.88 else 0.0,
                    )
                    if score > best_score:
                        best_score = score
                        matched_id = candidate_id
            explicit_id = str(raw.get("id") or "").replace("\x00", "").strip()[:96]
            if matched_id is None:
                # 没有框的语义事实可以保留显式世界 ID；可定位候选必须使用会话 ID，
                # 防止模型每次随意生成的新名字破坏稳定绑定。
                matched_id = explicit_id if bbox is None and explicit_id else self._new_id_locked()

            previous = self._items.get(matched_id, {})
            candidate = {
                "id": matched_id,
                "semantic_type": semantic_type,
                "label": str(raw.get("label") or semantic_type or "unknown")[:64],
                "confidence": min(1.0, max(0.0, float(raw.get("confidence") or 0.0))),
                "bbox": bbox,
                "first_observed_at": previous.get("first_observed_at", job.captured_at),
                "last_observed_at": job.captured_at,
                "received_at": now,
                "frame_id": job.frame_id,
                "revision": job.revision,
                "state": str(raw.get("state") or "visible")[:64],
                "attributes": (
                    dict(raw.get("attributes") or {})
                    if isinstance(raw.get("attributes"), Mapping) else {}
                ),
                "source": job.source,
                "crop": crop if crop is not None else previous.get("crop"),
                "descriptor": descriptor if descriptor is not None else previous.get("descriptor"),
            }
            self._items[matched_id] = candidate
            self._prune_locked(now)

        attributes = dict(raw.get("attributes") or {}) if isinstance(raw.get("attributes"), Mapping) else {}
        if bbox is not None:
            center_x = (bbox[0] + bbox[2]) * 0.5
            center_y = (bbox[1] + bbox[3]) * 0.5
            attributes.setdefault("screen_center", [round(center_x, 5), round(center_y, 5)])
            attributes.setdefault("screen_size", [round(bbox[2] - bbox[0], 5), round(bbox[3] - bbox[1], 5)])
            attributes.setdefault("bearing_deg", round((center_x - 0.5) * 90.0, 3))
            attributes.setdefault("apparent_height", round(bbox[3] - bbox[1], 5))
        attributes.update({
            "semantic_type": semantic_type,
            "semantic_verified": semantic_type != "unknown",
            "semantic_source_frame_id": job.frame_id,
            "semantic_source_revision": job.revision,
            "identity_scope": "session",
            "identity_method": "semantic_bbox_appearance",
            "memory_scope": "session",
        })
        normalized = dict(raw)
        normalized.update({
            "id": matched_id,
            "label": str(raw.get("label") or semantic_type or "unknown")[:64],
            "bbox": bbox,
            "state": str(raw.get("state") or "visible")[:64],
            "attributes": attributes,
            "source": [job.source],
            "observed_at": job.captured_at,
            "ttl_s": min(5.0, max(0.5, float(raw.get("ttl_s") or 2.0))),
        })
        return normalized, explicit_id or None

    def enrich_observation(
        self,
        observation: VisionObservation,
        *,
        frame: Any,
        now: float,
    ) -> VisionObservation:
        """把已完成的慢语义分类附着到当前检测帧。

        VLM 返回时它分析的原帧通常已经落后数秒，不能把旧时间戳伪装成当前
        事实。这里等待下一次本地检测重新看到同一稳定 ID，再把分类作为缓存属性
        附着；实体的位置和 observed_at 始终来自当前本地帧。
        """
        with self._lock:
            self._prune_locked(now)
            cached = {candidate_id: dict(item) for candidate_id, item in self._items.items()}
        if not cached or not observation.entities:
            return observation

        encoded_frame: bytes | None = frame if isinstance(frame, bytes) else None
        enriched: list[WorldEntity | Mapping[str, Any]] = []
        for value in observation.entities:
            raw = _semantic_entity_mapping(value)
            entity_id = str(raw.get("id") or "")[:96]
            candidate = cached.get(entity_id)
            candidate_id = entity_id if candidate is not None else ""
            bbox = _semantic_bbox(raw.get("bbox"))

            if candidate is None and bbox is not None:
                # ID 注册表短暂重建时先尝试严格同屏框重叠；仍不匹配才计算 8x8
                # 外观描述子。阈值刻意偏高，宁可等下一轮 VLM 也不把相似角色串号。
                best_score = 0.0
                current_descriptor: tuple[float, ...] | None = None
                for cached_id, item in cached.items():
                    overlap = _semantic_iou(bbox, item.get("bbox"))
                    score = overlap if overlap >= 0.55 else 0.0
                    if score <= best_score and item.get("descriptor") is not None:
                        if encoded_frame is None:
                            try:
                                encoded_frame, _, _ = encode_frame_jpeg(frame, max_width=960, quality=65)
                            except Exception:
                                encoded_frame = b""
                        if current_descriptor is None and encoded_frame:
                            current_descriptor = _semantic_descriptor(
                                _semantic_crop(encoded_frame, bbox)
                            )
                        appearance = _semantic_similarity(current_descriptor, item.get("descriptor"))
                        if appearance >= 0.94:
                            score = max(score, appearance)
                    if score > best_score:
                        best_score = score
                        candidate = item
                        candidate_id = cached_id

            semantic_type = str((candidate or {}).get("semantic_type") or "unknown")
            if candidate is None or semantic_type == "unknown":
                enriched.append(value)
                continue

            attributes = dict(raw.get("attributes") or {}) if isinstance(raw.get("attributes"), Mapping) else {}
            semantic_attributes = candidate.get("attributes")
            if isinstance(semantic_attributes, Mapping):
                for key, item in semantic_attributes.items():
                    attributes.setdefault(str(key)[:64], item)
            attributes.update({
                "semantic_type": semantic_type,
                "semantic_verified": True,
                "semantic_candidate_id": candidate_id,
                "semantic_source_frame_id": candidate.get("frame_id"),
                "semantic_source_revision": candidate.get("revision"),
                "semantic_result_age_ms": round(
                    max(0.0, now - float(candidate.get("received_at", now))) * 1000.0,
                    1,
                ),
                "semantic_memory_scope": "session",
            })
            sources = raw.get("source")
            if isinstance(sources, str):
                source_values = [sources]
            elif isinstance(sources, (list, tuple, set)):
                source_values = [str(item) for item in sources]
            else:
                source_values = []
            source_values.append(str(candidate.get("source") or "openai_vlm"))
            local_confidence = min(1.0, max(0.0, float(raw.get("confidence") or 0.0)))
            semantic_confidence = min(1.0, max(0.0, float(candidate.get("confidence") or 0.0)))
            raw.update({
                "label": str(candidate.get("label") or raw.get("label") or semantic_type)[:64],
                "confidence": min(local_confidence, semantic_confidence),
                "attributes": attributes,
                "source": list(dict.fromkeys(source_values))[:8],
                # observed_at/框/距离故意不改，必须继续代表当前本地检测帧。
            })
            enriched.append(raw)
        return replace(observation, entities=tuple(enriched))

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            candidates = []
            total_bytes = 0
            for item in self._items.values():
                crop = item.get("crop")
                crop_bytes = len(crop) if isinstance(crop, bytes) else 0
                total_bytes += crop_bytes
                candidates.append({
                    "id": item.get("id"),
                    "semantic_type": item.get("semantic_type"),
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "bbox": list(item["bbox"]) if item.get("bbox") is not None else None,
                    "frame_id": item.get("frame_id"),
                    "revision": item.get("revision"),
                    "observation_age_ms": round(max(0.0, now - float(item.get("last_observed_at", now))) * 1000.0, 1),
                    "result_age_ms": round(max(0.0, now - float(item.get("received_at", now))) * 1000.0, 1),
                    "crop_bytes": crop_bytes,
                })
            return {
                "storage": "memory_bounded",
                "persistent": False,
                "ttl_s": self.ttl_s,
                "max_candidates": self.max_candidates,
                "candidate_count": len(candidates),
                "bytes": total_bytes,
                "candidates": candidates,
            }


class SemanticWorker:
    """VLM 单槽异步 worker；慢推理不能阻塞检测与导航。"""

    def __init__(self, runtime: "VisionRuntime", *, clock: Callable[[], float] = time.monotonic) -> None:
        self.runtime = runtime
        self._clock = clock
        self.queue: Queue[SemanticJob] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._last_submit_at: float | None = None
        self._last_result_at: float | None = None
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._loop,
                name="neko-semantic-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(self, job: SemanticJob) -> bool:
        self._ensure_started()
        try:
            self.queue.put_nowait(job)
        except Full:
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            with self._lock:
                self._dropped += 1
            try:
                self.queue.put_nowait(job)
            except Full:
                return False
        with self._lock:
            self._submitted += 1
            self._last_submit_at = self._clock()
        return True

    def clear(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
            with self._lock:
                self._dropped += 1

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, timeout_s))
        with self._lock:
            alive = thread is not None and thread.is_alive()
            self._running = bool(alive)
            if not alive:
                self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.queue.get(timeout=0.2)
            except Empty:
                continue
            started = self._clock()
            try:
                self.runtime._process_semantic_job(job)
                with self._lock:
                    self._processed += 1
                    self._last_result_at = self._clock()
                    self._last_latency_ms = round(max(0.0, self._last_result_at - started) * 1000.0, 3)
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_result_at = self._clock()
                    self._last_latency_ms = round(max(0.0, self._last_result_at - started) * 1000.0, 3)
                    self._last_error = f"{type(exc).__name__}: {exc}"[:256]
        with self._lock:
            self._running = False

    def status(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            return {
                "running": self._running,
                "queue_size": 1,
                "queue_depth": self.queue.qsize(),
                "submitted": self._submitted,
                "processed": self._processed,
                "dropped": self._dropped,
                "last_submit_age_ms": None if self._last_submit_at is None else round(max(0.0, now - self._last_submit_at) * 1000.0, 1),
                "last_result_age_ms": None if self._last_result_at is None else round(max(0.0, now - self._last_result_at) * 1000.0, 1),
                "last_latency_ms": self._last_latency_ms,
                "last_error": self._last_error,
            }


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

    def _obscured(self) -> bool:
        """采集源是否报告目标窗口被盖住。探测不出来就当没被盖住。

        这里刻意宁放不误封：把探测故障读成"被遮挡"会让检测在窗口其实可见时
        整段停掉，那是用一个诊断故障换掉全部感知。
        """
        try:
            return bool(dict(self.source.status()).get("window_obscured", False))
        except Exception:
            return False

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self.queue.get(timeout=min(0.2, self.interval_s))
            except Empty:
                continue
            try:
                self.runtime.process_frame(
                    packet.frame,
                    observed_at=packet.captured_at,
                    source_obscured=self._obscured(),
                )
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
        main_llm_semantic: bool = False,
        main_llm_min_interval_s: float = 12.0,
        detect_interval_s: float = 0.0,
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
        self._semantic_candidates = SemanticCandidateCache(clock=clock)
        self._semantic_worker = SemanticWorker(self, clock=clock)
        # 主 LLM 语义桥只有一个待处理槽位。它保存与 revision 原子配对的一张
        # JPEG，插件取走后作为被动上下文并入已有对话；不会在后端另起模型调用。
        self._main_llm_semantic_enabled = bool(main_llm_semantic)
        self._main_llm_min_interval_s = max(5.0, float(main_llm_min_interval_s))
        self._main_llm_request_ttl_s = max(30.0, self._main_llm_min_interval_s * 2.0)
        self._main_llm_request: dict[str, Any] | None = None
        self._main_llm_request_counter = 0
        self._main_llm_last_requested_at: float | None = None
        self._main_llm_last_committed_at: float | None = None
        self._main_llm_requests_created = 0
        self._main_llm_requests_committed = 0
        self._main_llm_requests_cancelled = 0
        self._main_llm_results_rejected = 0
        self._main_llm_last_cancelled_request_id: str | None = None
        self._main_llm_last_cancel_reason: str | None = None
        # 检测最小间隔。线程上限管的是「一次推理占几个核」，这个管的是「一秒里
        # 推几次」——两者相乘才是实际 CPU 占用，少任何一个都收不住。0 表示每帧
        # 都检测（保持既有行为，也是所有现存测试的假设）。
        self._detect_interval_s = max(0.0, float(detect_interval_s))
        self._last_detect_at: float | None = None
        self._detect_skipped = 0
        self._obscured_frames = 0
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

    def _cancel_main_llm_request_locked(self, reason: str) -> str | None:
        """在持锁状态下取消单槽，并留下供宿主覆盖旧回调的轻量墓碑。"""
        request = self._main_llm_request
        request_id = (
            str(request.get("request_id") or "")[:128]
            if isinstance(request, Mapping) else ""
        )
        self._main_llm_request = None
        if not request_id:
            return None
        self._main_llm_last_cancelled_request_id = request_id
        self._main_llm_last_cancel_reason = (
            str(reason or "request_cancelled").replace("\x00", "").strip()[:96]
            or "request_cancelled"
        )
        self._main_llm_requests_cancelled += 1
        return request_id

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
                self._cancel_main_llm_request_locked(self._capture_reason)
        if not active:
            # 尚未开始的旧语义任务必须丢掉；已经在推理的任务返回后也会经过
            # captured_at/采集门控，最多进入候选记忆，不能伪装成当前可见实体。
            self._semantic_worker.clear()
        self.store.set_backend_status("vision_runtime", self.status())

    def capture_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._capture_active,
                "reason": self._capture_reason,
            }

    def close(self) -> None:
        """停止仅属于运行时的异步语义线程。"""
        self._semantic_worker.stop()
        with self._lock:
            self._cancel_main_llm_request_locked("vision_runtime_closed")

    def _build_semantic_job(self, frame: Any, captured_at: float) -> SemanticJob:
        data: bytes | None = None
        frame_id: str | None = None
        with self._lock:
            cached = self._frame_cache
            cached_at = self._frame_cache_at
            if (
                isinstance(cached, Mapping)
                and cached_at is not None
                and abs(float(cached_at) - float(captured_at)) <= 1e-6
                and isinstance(cached.get("data"), bytes)
            ):
                data = cached["data"]
                pair = cached.get("detection_pair")
                if isinstance(pair, Mapping):
                    frame_id = str(pair.get("frame_id") or "") or None
        if data is None:
            data, _, _ = encode_frame_jpeg(
                frame,
                max_width=self._frame_cache_max_width,
                quality=self._frame_cache_quality,
            )
        world = self.store.snapshot(now=self._clock())
        status = world.get("status") if isinstance(world.get("status"), Mapping) else {}
        try:
            revision = int(status.get("revision", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            revision = 0
        return SemanticJob(
            data=bytes(data),
            captured_at=float(captured_at),
            frame_id=frame_id or f"semantic-source-{revision}-{int(captured_at * 1000)}",
            revision=revision,
            world=world,
        )

    @staticmethod
    def _normalize_main_llm_selector(value: Any) -> dict[str, Any]:
        """收紧发给主 LLM 的 selector，避免把任意目标文本重复塞进上下文。"""
        if not isinstance(value, Mapping):
            return {}
        result: dict[str, Any] = {}
        semantic_type = str(value.get("semantic_type") or "").replace("\x00", "").strip().lower()[:32]
        if semantic_type:
            result["semantic_type"] = semantic_type
        label = str(value.get("label") or "").replace("\x00", "").strip()[:64]
        if label:
            result["label"] = label
        try:
            confidence = float(value.get("min_confidence"))
        except (TypeError, ValueError, OverflowError):
            confidence = None
        if confidence is not None and math.isfinite(confidence):
            result["min_confidence"] = min(1.0, max(0.0, confidence))
        return result

    def request_main_llm_semantics(
        self,
        selector: Mapping[str, Any] | None,
        *,
        reason: str = "semantic_target_unresolved",
        force: bool = False,
    ) -> dict[str, Any]:
        """为宿主主 LLM 建立一个最新帧语义任务，但不触发任何模型推理。"""
        now = self._clock()
        normalized_selector = self._normalize_main_llm_selector(selector)
        with self._lock:
            if not self._main_llm_semantic_enabled:
                return {"accepted": False, "reason": "main_llm_semantic_not_configured"}
            if not self._capture_active:
                return {"accepted": False, "reason": self._capture_reason or "capture_stopped"}
            current = self._main_llm_request
            if isinstance(current, Mapping):
                age_s = max(0.0, now - float(current.get("created_at", now)))
                if current.get("state") in {"pending", "processing"} and age_s <= self._main_llm_request_ttl_s:
                    return {
                        "accepted": True,
                        "created": False,
                        "reason": "request_already_pending",
                        "request_id": current.get("request_id"),
                        "revision": current.get("revision"),
                    }
            if (
                not force
                and self._main_llm_last_requested_at is not None
                and now - self._main_llm_last_requested_at < self._main_llm_min_interval_s
            ):
                retry_s = self._main_llm_min_interval_s - (now - self._main_llm_last_requested_at)
                return {
                    "accepted": False,
                    "reason": "main_llm_semantic_rate_limited",
                    "retry_after_ms": round(max(0.0, retry_s) * 1000.0),
                }
            cached = self._frame_cache
            cached_at = self._frame_cache_at
            if not isinstance(cached, Mapping) or cached_at is None or not isinstance(cached.get("data"), bytes):
                return {"accepted": False, "reason": "no_frame_cached"}
            frame_age_s = max(0.0, now - float(cached_at))
            if frame_age_s > 3.0:
                return {
                    "accepted": False,
                    "reason": "frame_stale",
                    "age_ms": round(frame_age_s * 1000.0, 1),
                }
            pair = cached.get("detection_pair")
            if not isinstance(pair, Mapping):
                # 没有本地 detector 时仍允许主 LLM 做第一阶段开放类别检测；它
                # 必须返回 bbox，不能伪造 target_id。合成空配对只锚定像素时间，
                # 不会把当前 world 的旧实体画到这张图上。
                world = self.store.snapshot(now=now)
                world_status = world.get("status") if isinstance(world.get("status"), Mapping) else {}
                pair = {
                    "frame_id": f"main-llm-source-{int(cached_at * 1000)}",
                    "revision": int(world_status.get("revision", 0) or 0),
                    "captured_at": float(cached_at),
                    "observed_at": float(cached_at),
                    "entities": [],
                }
            try:
                revision = int(pair.get("revision", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                revision = 0
            captured_at = float(pair.get("captured_at", cached_at) or cached_at)
            self._main_llm_request_counter += 1
            request_id = (
                f"semantic-request:{self._semantic_candidates.session_token}:"
                f"{self._main_llm_request_counter}"
            )
            # bytes 是不可变对象，这里只持有同一引用；没有第二份 JPEG，也没有队列。
            self._main_llm_request = {
                "request_id": request_id,
                "state": "pending",
                "created_at": now,
                "captured_at": captured_at,
                "reason": str(reason or "semantic_target_unresolved").replace("\x00", "").strip()[:96],
                "selector": normalized_selector,
                "frame_id": str(pair.get("frame_id") or "")[:128] or None,
                "revision": revision,
                "data": cached["data"],
                "mime": str(cached.get("mime") or "image/jpeg"),
                "width": int(cached.get("width", 0) or 0),
                "height": int(cached.get("height", 0) or 0),
                "detection_pair": dict(pair),
            }
            self._main_llm_last_requested_at = now
            self._main_llm_requests_created += 1
            return {
                "accepted": True,
                "created": True,
                "reason": None,
                "request_id": request_id,
                "revision": revision,
            }

    def main_llm_semantic_request(
        self,
        *,
        after_request_id: str | None = None,
    ) -> dict[str, Any]:
        """读取精确配对的待处理任务；同 ID 只返回轻量 no_new_request。"""
        now = self._clock()
        with self._lock:
            if not self._main_llm_semantic_enabled:
                return {"available": False, "reason": "main_llm_semantic_not_configured"}
            request = self._main_llm_request
            if not isinstance(request, Mapping):
                cancelled_id = self._main_llm_last_cancelled_request_id
                if after_request_id and str(after_request_id) == cancelled_id:
                    return {
                        "available": False,
                        "reason": "request_cancelled",
                        "request_id": cancelled_id,
                        "cancellation_reason": self._main_llm_last_cancel_reason,
                    }
                return {"available": False, "reason": "no_pending_request"}
            request_id = str(request.get("request_id") or "")
            age_s = max(0.0, now - float(request.get("created_at", now)))
            if age_s > self._main_llm_request_ttl_s:
                cancelled_id = self._cancel_main_llm_request_locked("request_expired")
                return {
                    "available": False,
                    "reason": "request_cancelled",
                    "request_id": cancelled_id,
                    "cancellation_reason": "request_expired",
                }
            if request.get("state") != "pending":
                return {"available": False, "reason": f"request_{request.get('state', 'unavailable')}"}
            if after_request_id and str(after_request_id) == request_id:
                return {
                    "available": False,
                    "reason": "no_new_request",
                    "request_id": request_id,
                }
            data = request.get("data")
            pair = request.get("detection_pair")
            frame = {
                "available": True,
                "capture_active": True,
                "age_ms": round(max(0.0, now - float(request.get("captured_at", now))) * 1000.0, 1),
                "mime": str(request.get("mime") or "image/jpeg"),
                "width": int(request.get("width", 0) or 0),
                "height": int(request.get("height", 0) or 0),
                "bytes": len(data) if isinstance(data, bytes) else 0,
                "data": data,
                "frame_id": request.get("frame_id"),
                "revision": int(request.get("revision", 0) or 0),
            }
            metadata = {
                "request_id": request_id,
                "request_state": "pending",
                "request_age_ms": round(age_s * 1000.0, 1),
                "request_ttl_ms": round(self._main_llm_request_ttl_s * 1000.0),
                "reason": request.get("reason"),
                "selector": dict(request.get("selector") or {}),
            }
        if not isinstance(frame.get("data"), bytes):
            return {"available": False, "reason": "request_frame_unavailable"}
        result = self._apply_overlay(
            frame,
            frame_age_ms=float(frame["age_ms"]),
            detection_pair=pair,
        )
        result.update(metadata)
        return result

    def commit_main_llm_semantics(
        self,
        request_id: Any,
        frame_revision: Any,
        entities: Any,
    ) -> dict[str, Any]:
        """校验并缓存主 LLM 分类；旧帧位置绝不直接回放到当前世界。"""
        normalized_request_id = str(request_id or "").replace("\x00", "").strip()[:128]
        try:
            normalized_revision = int(frame_revision)
        except (TypeError, ValueError, OverflowError):
            return {"accepted": False, "reason": "frame_revision must be an integer"}
        if not isinstance(entities, (list, tuple)) or len(entities) > 32:
            return {"accepted": False, "reason": "entities must be an array with at most 32 items"}
        now = self._clock()
        with self._lock:
            request = self._main_llm_request
            if not self._main_llm_semantic_enabled or not isinstance(request, Mapping):
                self._main_llm_results_rejected += 1
                reason = (
                    "request_cancelled"
                    if normalized_request_id
                    and normalized_request_id == self._main_llm_last_cancelled_request_id
                    else "no_pending_request"
                )
                return {"accepted": False, "reason": reason}
            if request.get("state") != "pending":
                self._main_llm_results_rejected += 1
                return {"accepted": False, "reason": "request_not_pending"}
            if normalized_request_id != str(request.get("request_id") or ""):
                self._main_llm_results_rejected += 1
                return {"accepted": False, "reason": "request_id_mismatch"}
            if normalized_revision != int(request.get("revision", 0) or 0):
                self._main_llm_results_rejected += 1
                return {"accepted": False, "reason": "frame_revision_mismatch"}
            if now - float(request.get("created_at", now)) > self._main_llm_request_ttl_s:
                self._cancel_main_llm_request_locked("request_expired")
                self._main_llm_results_rejected += 1
                return {"accepted": False, "reason": "request_expired"}
            self._main_llm_request["state"] = "processing"
            frozen = dict(request)

        pair = frozen.get("detection_pair") if isinstance(frozen.get("detection_pair"), Mapping) else {}
        paired_entities = {
            str(item.get("id") or "")[:96]: dict(item)
            for item in (pair.get("entities") or ())
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        }
        raw_entities: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for index, value in enumerate(entities):
            if not isinstance(value, Mapping):
                rejected.append({"index": index, "reason": "entity_must_be_object"})
                continue
            target_id = str(value.get("target_id") or "").replace("\x00", "").strip()[:96]
            paired = paired_entities.get(target_id) if target_id else None
            bbox = _semantic_bbox((paired or {}).get("bbox")) if paired is not None else _semantic_bbox(value.get("bbox"))
            if target_id and paired is None:
                rejected.append({"index": index, "reason": "target_id_not_in_request"})
                continue
            if bbox is None:
                rejected.append({"index": index, "reason": "bbox_required_for_unpaired_entity"})
                continue
            try:
                confidence = float(value.get("confidence"))
            except (TypeError, ValueError, OverflowError):
                confidence = math.nan
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                rejected.append({"index": index, "reason": "confidence_out_of_range"})
                continue
            semantic_type = _semantic_type(value)
            label = str(value.get("label") or (paired or {}).get("label") or semantic_type)
            label = label.replace("\x00", "").strip()[:64] or semantic_type
            raw_entities.append({
                "id": target_id or None,
                "label": label,
                "semantic_type": semantic_type,
                "confidence": confidence,
                "bbox": bbox,
                "state": "visible",
                "attributes": {"classified_by_current_main_llm": True},
                "ttl_s": 2.0,
            })

        job = SemanticJob(
            data=bytes(frozen.get("data") or b""),
            captured_at=float(frozen.get("captured_at", now) or now),
            frame_id=str(frozen.get("frame_id") or f"main-llm-{normalized_revision}"),
            revision=normalized_revision,
            world={"entities": list(paired_entities.values())},
            source="main_llm_vlm",
        )
        try:
            normalized = self._normalize_semantic_observation(
                VisionObservation(
                    entities=tuple(raw_entities),
                    source="main_llm_vlm",
                    observed_at=job.captured_at,
                    frame_id=job.frame_id,
                ),
                job,
            )
        except Exception as exc:
            with self._lock:
                if self._main_llm_request is not None:
                    self._main_llm_request["state"] = "pending"
                self._main_llm_results_rejected += 1
            return {"accepted": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}

        bindings = [
            {
                "target_id": str(item.get("id") or ""),
                "semantic_type": _semantic_type(item),
                "label": str(item.get("label") or "")[:64],
                "confidence": round(float(item.get("confidence", 0.0) or 0.0), 4),
            }
            for item in normalized.entities
            if isinstance(item, Mapping)
        ]
        with self._lock:
            self._main_llm_request = None
            self._main_llm_last_committed_at = now
            self._main_llm_requests_committed += 1
        # 不 ingest 旧帧：下一次本地检测会用稳定 ID/外观把这些属性附到当前位置。
        self.store.set_backend_status("vision_runtime", self.status())
        return {
            "accepted": True,
            "request_id": normalized_request_id,
            "frame_revision": normalized_revision,
            "bindings": bindings,
            "classified": len(bindings),
            "rejected": rejected,
            "position_update": "deferred_to_next_local_detection",
        }

    def clear_main_llm_semantic_request(self, reason: str = "goal_cleared") -> None:
        """目标结束时释放单槽图片，避免下一次对话消费已经无关的任务。"""
        with self._lock:
            self._cancel_main_llm_request_locked(reason)

    def _normalize_semantic_observation(
        self,
        observation: VisionObservation,
        job: SemanticJob,
    ) -> VisionObservation:
        entities: list[Mapping[str, Any]] = []
        id_map: dict[str, str] = {}
        for item in observation.entities:
            raw = _semantic_entity_mapping(item)
            if not raw:
                continue
            try:
                normalized, original_id = self._semantic_candidates.bind(raw, job=job)
            except (TypeError, ValueError, OverflowError):
                continue
            entities.append(normalized)
            if original_id:
                id_map[original_id] = str(normalized.get("id") or "")

        events: list[Mapping[str, Any]] = []
        for item in observation.events:
            raw_event: dict[str, Any]
            if isinstance(item, WorldEvent):
                raw_event = {
                    "type": item.kind,
                    "target_id": item.target_id,
                    "confidence": item.confidence,
                    "data": dict(item.data or {}),
                    "source": tuple(item.source),
                }
            elif isinstance(item, Mapping):
                raw_event = dict(item)
            else:
                continue
            old_target = str(raw_event.get("target_id") or "")
            if old_target in id_map:
                raw_event["target_id"] = id_map[old_target]
            raw_event["source"] = [job.source]
            raw_event["observed_at"] = job.captured_at
            events.append(raw_event)

        return VisionObservation(
            entities=tuple(entities),
            events=tuple(events),
            source=job.source,
            observed_at=job.captured_at,
            frame_id=job.frame_id,
            uncertainties=tuple(observation.uncertainties),
        )

    def _process_semantic_job(self, job: SemanticJob) -> None:
        with self._lock:
            semantic = self.semantic
        if semantic is None:
            return
        observation = semantic.observe(
            job.data,
            world=job.world,
            now=job.captured_at,
        )
        if not isinstance(observation, VisionObservation):
            raise ValueError("semantic backend must return VisionObservation")
        normalized = self._normalize_semantic_observation(observation, job)
        # 停止采集后仍允许候选缓存保留刚完成的分类，但绝不把旧结果发布成当前世界。
        if not self.capture_state()["active"]:
            return
        self.ingest(normalized, _reactivate=False)

    def _cache_frame(
        self,
        frame: Any,
        captured_at: float,
        *,
        observation: VisionObservation | None = None,
        world: Mapping[str, Any] | None = None,
    ) -> bool:
        """按间隔把最新帧编码进单槽内存缓存。

        必须在帧被 ``_release_frame`` 释放之前调用。编码失败只记录原因，绝不
        让采集或检测因此中断——看不到图是降级，掉帧是故障。

        ``observation`` 与 ``world`` 同时提供时，把该检测批次中实际出现的实体
        和 JPEG 放进同一个内存对象。它不创建文件、不保留历史，也不额外调用
        ``store.ingest``；下一次成功缓存会直接覆盖整组数据。
        """
        if self._frame_cache_interval_s < 0:
            return False
        with self._lock:
            last = self._frame_cache_at
            interval = self._frame_cache_interval_s
        if last is not None and (captured_at - last) < interval:
            return False
        try:
            data, width, height = encode_frame_jpeg(
                frame,
                max_width=self._frame_cache_max_width,
                quality=self._frame_cache_quality,
            )
        except Exception as exc:
            with self._lock:
                self._frame_cache_error = f"{type(exc).__name__}: {exc}"[:256]
            return False
        detection_pair: dict[str, Any] | None = None
        if observation is not None and isinstance(world, Mapping):
            exact_ids: set[str] = set()
            for item in observation.entities:
                try:
                    raw: Mapping[str, Any] = (
                        {
                            "id": item.id,
                            "label": item.label,
                            "confidence": item.confidence,
                            "bbox": item.bbox,
                            "state": item.state,
                            "attributes": item.attributes,
                            "relations": item.relations,
                            "source": item.source,
                            "observed_at": item.observed_at,
                            "ttl_s": item.ttl_s,
                        }
                        if isinstance(item, WorldEntity)
                        else item
                    )
                    normalized = WorldEntity.from_mapping(
                        raw,
                        now=captured_at,
                        default_source=observation.source,
                        default_ttl_s=float(getattr(self.store, "default_ttl_s", 2.0)),
                    )
                except (TypeError, ValueError):
                    continue
                exact_ids.add(normalized.id)
            paired_entities = tuple(
                dict(item)
                for item in (world.get("entities") or ())
                if isinstance(item, Mapping) and str(item.get("id") or "") in exact_ids
            )
            status = world.get("status") if isinstance(world.get("status"), Mapping) else {}
            try:
                revision = int(status.get("revision", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                revision = 0
            try:
                paired_observed_at = (
                    min(captured_at, float(observation.observed_at))
                    if observation.observed_at is not None
                    else captured_at
                )
            except (TypeError, ValueError, OverflowError):
                paired_observed_at = captured_at
            detection_pair = {
                "frame_id": observation.frame_id or f"{observation.source}-{revision}",
                "revision": revision,
                "captured_at": captured_at,
                "observed_at": paired_observed_at,
                "entities": paired_entities,
            }
        with self._lock:
            self._frame_cache = {
                "data": data,
                "mime": "image/jpeg",
                "width": width,
                "height": height,
                "bytes": len(data),
                "detection_pair": detection_pair,
            }
            self._frame_cache_at = captured_at
            self._frame_cache_error = None
        return True

    def latest_frame(self, *, max_age_ms: int = 3000, overlay: bool = False) -> dict[str, Any]:
        """返回最近一次缓存的画面，超龄或采集停止时明确拒绝。

        这条路径刻意与 ``world_state`` 完全分离：帧只喂给 agent 理解，不产生
        实体、不产生事件，也不能拿去满足 ``body_reach_and_grab`` 的
        ``preconditions``。从画面得出的任何结论都是低置信视觉猜测。

        ``max_age_ms <= 0`` 表示不限龄，这是留给运行时内部调用方的逃生口。
        LLM 那一侧不允许走到这里——工具与 ``BackendService`` 都把下限抬到了
        250 ms，否则「要最新的画面」写成 0 反而会拿到最旧的一张。

        ``overlay=True`` 时叠加检测框，用于对照「检测器看到的」与「画面里实际
        有的」。叠框只画在副本上，缓存中的原始像素不受影响。
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
        result = {
            "available": True,
            "capture_active": True,
            "age_ms": round(age_ms, 1),
            "mime": cached["mime"],
            "width": cached["width"],
            "height": cached["height"],
            "bytes": cached["bytes"],
            "data": cached["data"],
        }
        detection_pair = cached.get("detection_pair")
        if isinstance(detection_pair, Mapping):
            result["frame_id"] = str(detection_pair.get("frame_id") or "") or None
            result["revision"] = int(detection_pair.get("revision", 0) or 0)
        if overlay:
            result = self._apply_overlay(
                result,
                frame_age_ms=age_ms,
                detection_pair=detection_pair,
            )
        return result

    def _apply_overlay(
        self,
        frame: dict[str, Any],
        *,
        frame_age_ms: float,
        detection_pair: Any,
    ) -> dict[str, Any]:
        """只把与缓存像素同批次的检测实体画到副本上。

        这里绝不读取最新 ``world.snapshot``。像素和框在检测完成后已经原子地放进
        同一个单槽内存对象；没有配对数据就明确降级，不能拿旧图叠新框。
        """
        if not isinstance(detection_pair, Mapping):
            frame["overlay"] = {
                "requested": True,
                "paired": False,
                "revision": None,
                "frame_id": None,
                "frame_age_ms": round(frame_age_ms, 1),
                "world_age_ms": None,
                "skew_ms": None,
                "entities_available": 0,
                "candidates": [],
                "drawn": False,
                "reason": "frame_detection_pair_unavailable",
            }
            return frame
        entities = list(detection_pair.get("entities") or ())
        try:
            captured_at = float(detection_pair.get("captured_at"))
            observed_at = float(detection_pair.get("observed_at"))
            skew_ms = abs(captured_at - observed_at) * 1000.0
            world_age_ms = frame_age_ms + (captured_at - observed_at) * 1000.0
        except (TypeError, ValueError, OverflowError):
            skew_ms = math.inf
            world_age_ms = None
        try:
            revision = int(detection_pair.get("revision", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            revision = 0
        overlay_status: dict[str, Any] = {
            "requested": True,
            "paired": True,
            "revision": revision,
            "frame_id": str(detection_pair.get("frame_id") or "") or None,
            "frame_age_ms": round(frame_age_ms, 1),
            "world_age_ms": round(world_age_ms, 1) if world_age_ms is not None else None,
            "skew_ms": round(skew_ms, 1) if math.isfinite(skew_ms) else None,
            "entities_available": len(entities),
            # 只有真正烧进图里的框才会出现在这里。绘制失败时保持空列表，避免
            # LLM 在没看见 T 编号的情况下仅凭 JSON 猜目标。
            "candidates": [],
        }
        warning = None
        if not math.isfinite(skew_ms) or skew_ms > 250.0:
            warning = "PAIR INVALID" if not math.isfinite(skew_ms) else f"PAIR SKEW {skew_ms:.0f}ms"
            overlay_status["skew_warning"] = True
        try:
            drawn_geometry: list[dict[str, Any]] = []
            data, drawn = draw_detection_overlay(
                frame["data"],
                entities,
                quality=self._frame_cache_quality,
                warning=warning,
                geometry_out=drawn_geometry,
            )
        except Exception as exc:
            # 画不出框就给原图。看不到框是降级，掉帧才是故障。
            overlay_status["drawn"] = False
            overlay_status["reason"] = f"{type(exc).__name__}: {exc}"[:200]
            frame["overlay"] = overlay_status
            return frame
        overlay_status["drawn"] = True
        overlay_status["boxes_drawn"] = drawn
        overlay_status["boxes_skipped"] = max(0, len(entities) - drawn)
        overlay_status["candidates"] = [
            {
                "ref": str(box["ref"]),
                "target_id": str(box["id"]),
                "label": str(box["label"]),
                "confidence": round(float(box["confidence"]), 4),
                "bearing_deg": box.get("bearing_deg"),
                "clipped": bool(box.get("clipped")),
            }
            for box in drawn_geometry
        ]
        frame["data"] = data
        frame["bytes"] = len(data)
        frame["overlay"] = overlay_status
        return frame

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
        # 停止采集后旧账本也属于过期视觉信息。推进 through_revision 让消费者
        # 丢弃游标以前的历史，但绝不把这些实体/事件重新注入主 LLM。
        journal = dict(result.get("journal") or {})
        journal["entries"] = []
        journal["has_more"] = False
        journal["truncated"] = False
        journal["through_revision"] = int(result.get("revision", 0) or 0)
        result["journal"] = journal
        return result

    def status(self) -> dict[str, Any]:
        status_now = self._clock()
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
            frame_pair = (
                self._frame_cache.get("detection_pair")
                if isinstance(self._frame_cache, Mapping)
                else None
            )
            detect_interval_s = self._detect_interval_s
            detect_skipped = self._detect_skipped
            obscured_frames = self._obscured_frames
            last_detect_at = self._last_detect_at
            main_llm_enabled = self._main_llm_semantic_enabled
            main_llm_request = self._main_llm_request
            main_llm_status = {
                "enabled": main_llm_enabled,
                "mode": "merged_conversation",
                "storage": "memory_single_slot",
                "persistent": False,
                "min_interval_s": self._main_llm_min_interval_s,
                "request_id": (
                    str(main_llm_request.get("request_id") or "") or None
                    if isinstance(main_llm_request, Mapping) else None
                ),
                "request_state": (
                    str(main_llm_request.get("state") or "none")
                    if isinstance(main_llm_request, Mapping) else "none"
                ),
                "request_revision": (
                    int(main_llm_request.get("revision", 0) or 0)
                    if isinstance(main_llm_request, Mapping) else None
                ),
                "request_age_ms": (
                    round(max(0.0, status_now - float(main_llm_request.get("created_at", status_now))) * 1000.0, 1)
                    if isinstance(main_llm_request, Mapping) else None
                ),
                "requests_created": self._main_llm_requests_created,
                "requests_committed": self._main_llm_requests_committed,
                "requests_cancelled": self._main_llm_requests_cancelled,
                "last_cancelled_request_id": self._main_llm_last_cancelled_request_id,
                "last_cancel_reason": self._main_llm_last_cancel_reason,
                "results_rejected": self._main_llm_results_rejected,
                "last_commit_age_ms": (
                    None if self._main_llm_last_committed_at is None
                    else round(max(0.0, status_now - self._main_llm_last_committed_at) * 1000.0, 1)
                ),
            }
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
        semantic_status = backend_status(semantic, label="semantic")
        if main_llm_enabled:
            semantic_status = {
                "available": True,
                "backend": "main_llm",
                "model": "current_host_multimodal_llm",
                **main_llm_status,
            }
        return {
            "enabled": detector is not None or semantic is not None or main_llm_enabled,
            "capture_active": capture_active,
            "capture_reason": capture_reason,
            "detector": backend_status(detector, label="detector"),
            "semantic": semantic_status,
            "main_llm_semantic": main_llm_status,
            "semantic_worker": self._semantic_worker.status(),
            "semantic_candidates": self._semantic_candidates.snapshot(),
            "optional_dependencies": optional_dependency_status(),
            # 画面缓存独立于世界状态；这里只报告它是否可用，不把画面内容
            # 混进感知结论。
            "frame_cache": {
                "cached": frame_cached,
                "storage": "memory_single_slot",
                "persistent": False,
                "paired": isinstance(frame_pair, Mapping),
                "frame_id": (
                    str(frame_pair.get("frame_id") or "") or None
                    if isinstance(frame_pair, Mapping)
                    else None
                ),
                "revision": (
                    int(frame_pair.get("revision", 0) or 0)
                    if isinstance(frame_pair, Mapping)
                    else None
                ),
                "age_ms": (
                    None if frame_cache_at is None
                    else round(max(0.0, (self._clock() - frame_cache_at) * 1000.0), 1)
                ),
                "interval_s": frame_cache_interval_s,
                "last_error": frame_cache_error,
            },
            # 节流是可观测的：跳过多少帧必须能读到，否则「限流生效」与「采集挂了」
            # 在外面看起来一样。interval_s 为 0 表示每帧都检测。
            "detect_throttle": {
                "interval_s": detect_interval_s,
                "skipped_frames": detect_skipped,
                "age_ms": (
                    None if last_detect_at is None
                    else round(max(0.0, (self._clock() - last_detect_at) * 1000.0), 1)
                ),
            },
            # 被遮挡而整帧丢弃的次数。它跟 skipped_frames 是两回事：限流是「这一帧
            # 不看」，遮挡是「这一帧看不见」。混在一起读，就分不清检测器在省 CPU
            # 还是窗口一直被别的应用压着。
            "obscured_frames": obscured_frames,
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
            self._last_semantic_at = None
        self._semantic_worker.clear()
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

    def _observe_obscured(self, processing_now: float, observation_now: float) -> dict[str, Any]:
        """窗口被盖住时只声明看不见，不对这一帧做任何推理。

        DXGI 抓的是合成后的桌面，VRChat 被别的窗口压住时采集依旧"成功"，拿到的
        却是上层窗口的像素。对它推理会把浏览器或聊天窗里的人形当成世界里的玩家
        写进世界状态——那比没有观测危险得多，因为下游分不出真假。

        因此这里既不写实体也不删实体：已有实体按各自 TTL 自然老化，
        ``visual_capture_occluded`` 则明确告诉消费者「观测缺失是有原因的」。它不在
        ``INFORMATIONAL_UNCERTAINTIES`` 白名单里，所以会照常阻断移动——看不见就
        不该走。写入沿用检测节流的节拍，避免每帧都推高世界修订号。
        """
        with self._lock:
            self._obscured_frames += 1
            due = (
                self._detect_interval_s <= 0.0
                or self._last_detect_at is None
                or processing_now - self._last_detect_at >= self._detect_interval_s
            )
            if due:
                self._last_detect_at = processing_now
            else:
                self._detect_skipped += 1
        if due:
            self.ingest(
                VisionObservation(
                    entities=(),
                    events=(),
                    source="vision",
                    observed_at=observation_now,
                    uncertainties=("visual_capture_occluded",),
                ),
                _reactivate=False,
            )
        self.store.set_backend_status("vision_runtime", self.status())
        result = self._mask_stopped_snapshot(self.store.snapshot(now=processing_now))
        result["vision"] = self.status()
        return result

    def process_frame(
        self,
        frame: Any,
        *,
        force_semantic: bool = False,
        observed_at: float | None = None,
        source_obscured: bool = False,
    ) -> dict[str, Any]:
        """处理一帧画面而不阻塞身体调度器。

        采集循环由调用方负责。语义推理受到频率限制，只在明确请求或冷却时间
        到期时运行。

        ``source_obscured`` 为真表示采集拿到的不是目标窗口的画面（被别的窗口
        压住、或已最小化）。这种帧一律不推理：见 ``_observe_obscured``。
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
        if source_obscured:
            return self._observe_obscured(processing_now, observation_now)
        # attach_vision/set_backends 可能与采集线程并行；在锁内取得稳定引用，
        # 后续推理不持有运行时锁，避免阻塞生命周期控制。
        with self._lock:
            detector = self.detector
            semantic = self.semantic
        cache_written = False
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
                with self._lock:
                    detect_due = (
                        self._detect_interval_s <= 0.0
                        or self._last_detect_at is None
                        or processing_now - self._last_detect_at >= self._detect_interval_s
                    )
                    if detect_due:
                        self._last_detect_at = processing_now
                    else:
                        self._detect_skipped += 1
                # 跳过时**不**写世界状态。补一个空观测会把「这一帧没看」伪造成
                # 「这一帧什么都没有」，实体全部消失；不写则由 store 自己让
                # age_ms 长上去，「多久之前看到的」仍然是真话。
                if detect_due:
                    detector_observation = detector.observe(frame, now=observation_now)
                    detector_observation = self._semantic_candidates.enrich_observation(
                        detector_observation,
                        frame=frame,
                        now=processing_now,
                    )
                    detector_world = self.ingest(detector_observation, _reactivate=False)
                    # worker 只会在 process_frame 返回后释放 frame，所以检测完成后
                    # 仍可安全编码。此时像素、实体、revision 一次写入同一个单槽
                    # 内存对象，不需要磁盘临时文件。
                    cache_written = self._cache_frame(
                        frame,
                        observation_now,
                        observation=detector_observation,
                        world=detector_world,
                    )
            else:
                # 没有可用检测器时仍允许主 LLM 看原图，但 overlay 会明确报告未配对。
                cache_written = self._cache_frame(frame, observation_now)
            if semantic is not None and self.capture_state()["active"]:
                with self._lock:
                    semantic_due = (
                        force_semantic
                        or self._last_semantic_at is None
                        or processing_now - self._last_semantic_at >= self.semantic_cooldown_s
                    )
                if semantic_due:
                    job = self._build_semantic_job(frame, observation_now)
                    if self._semantic_worker.submit(job):
                        with self._lock:
                            self._last_semantic_at = processing_now
            with self._lock:
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
            if not cache_written:
                # 推理失败时保留看原图的能力；没有配对元数据，因此绝不会误画框。
                self._cache_frame(frame, observation_now)
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
from .local_perception import cap_openmp_threads as cap_openmp_threads  # noqa: E402,F401


__all__ = [
    "cap_openmp_threads",
    "CapturedFrame",
    "draw_detection_overlay",
    "encode_frame_jpeg",
    "find_window_region",
    "FrameDetector",
    "FrameSource",
    "DxcamFrameSource",
    "DesktopMirrorFrameSource",
    "MssFrameSource",
    "OpenVinoLocalDetector",
    "OpenAICompatibleSemanticBackend",
    "overlay_boxes_geometry",
    "SemanticBackend",
    "VisionObservation",
    "VisionRuntime",
    "VisionWorker",
    "WindowTrackedFrameSource",
    "window_visibility",
    "optional_dependency_status",
]
