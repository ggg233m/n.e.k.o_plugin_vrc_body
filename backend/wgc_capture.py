"""Windows.Graphics.Capture 的 D3D11 胶水层。

这一层刻意从 ``vision.py`` 里分出来：它全是 ctypes/COM 细节，与视觉编排无关，
混在一起会让本就很长的 vision.py 更难读。对外只暴露 ``WgcSession``——打开、
读一帧 RGB ndarray、关闭。

**为什么需要它。** DXGI 桌面复制抓的是合成后的桌面，VRChat 被别的窗口盖住时
采集照样成功，拿到的却是上层窗口的像素。Windows.Graphics.Capture 按**窗口**
而不是按输出捕获，DWM 直接交出该窗口自己的合成内容，因此遮挡与采集无关。

实测（1922x1041，RTX 环境）：
  - ``try_get_next_frame`` 本身 ~0 ms（帧已在 GPU 上）
  - 全尺寸回读（CopyResource + Map + BGRA→RGB）中位数 22 ms
  - 同条件下 PrintWindow 约 70 ms，DXGI 桌面复制则直接抓到错误内容

仍然抓不到的唯一情形是**窗口最小化**：DWM 不再为它合成，没有内容可交。
调用方必须自己保留最小化判断。
"""

from __future__ import annotations

import ctypes
import threading
import uuid
from ctypes import wintypes
from typing import Any

#: ID3D11Texture2D
_IID_ID3D11TEXTURE2D = "6f15aaf2-d208-4e89-9ab4-489535d34f9c"
#: IDXGIDevice
_IID_IDXGIDEVICE = "54ec77fa-1377-44e6-8c32-88fd5f44c84c"

_D3D_DRIVER_TYPE_HARDWARE = 1
_D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
_D3D11_SDK_VERSION = 7
_D3D11_USAGE_STAGING = 3
_D3D11_CPU_ACCESS_READ = 0x20000
_D3D11_MAP_READ = 1
_DXGI_FORMAT_B8G8R8A8_UNORM = 87

# ID3D11Device / ID3D11DeviceContext 的 vtable 槽位。这些序号由接口定义固定，
# 不随驱动或 Windows 版本变化。
_VTBL_QUERY_INTERFACE = 0
_VTBL_RELEASE = 2
_VTBL_CREATE_TEXTURE2D = 5
_VTBL_CONTEXT_MAP = 14
_VTBL_CONTEXT_UNMAP = 15
_VTBL_CONTEXT_COPY_RESOURCE = 47


class _D3D11_TEXTURE2D_DESC(ctypes.Structure):
    _fields_ = [
        ("Width", ctypes.c_uint),
        ("Height", ctypes.c_uint),
        ("MipLevels", ctypes.c_uint),
        ("ArraySize", ctypes.c_uint),
        ("Format", ctypes.c_uint),
        ("SampleDescCount", ctypes.c_uint),
        ("SampleDescQuality", ctypes.c_uint),
        ("Usage", ctypes.c_uint),
        ("BindFlags", ctypes.c_uint),
        ("CPUAccessFlags", ctypes.c_uint),
        ("MiscFlags", ctypes.c_uint),
    ]


class _D3D11_MAPPED_SUBRESOURCE(ctypes.Structure):
    _fields_ = [
        ("pData", ctypes.c_void_p),
        ("RowPitch", ctypes.c_uint),
        ("DepthPitch", ctypes.c_uint),
    ]


def _vtbl_call(ptr: Any, index: int, restype: Any, *argtypes: Any) -> Any:
    """取 COM 对象 vtable 第 ``index`` 项并包成可调用对象。"""
    vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


def _query_interface(ptr: Any, iid_str: str) -> Any:
    """COM QueryInterface。失败时返回 ``None`` 而不是抛异常。"""
    iid = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID(iid_str).bytes_le)
    out = ctypes.c_void_p()
    call = _vtbl_call(
        ptr,
        _VTBL_QUERY_INTERFACE,
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    if call(ptr, ctypes.byref(iid), ctypes.byref(out)) != 0:
        return None
    return out


def _release(ptr: Any) -> None:
    if not ptr:
        return
    try:
        _vtbl_call(ptr, _VTBL_RELEASE, ctypes.c_ulong)(ptr)
    except Exception:
        pass


def wgc_supported() -> bool:
    """WGC 按窗口捕获是否可用。只探测、不构造会话。

    需要 Win10 1903+ 的 ``GraphicsCaptureItem`` interop，以及 winrt wheel。
    任一缺失都返回 ``False``，让调用方回退到桌面复制。
    """
    try:
        import winrt.windows.graphics.capture.interop as interop  # noqa: F401
        import winrt.windows.graphics.capture as capture

        return hasattr(interop, "create_for_window") and hasattr(
            capture.Direct3D11CaptureFramePool, "create_free_threaded"
        )
    except Exception:
        return False


class WgcSession:
    """针对单个 HWND 的 Windows.Graphics.Capture 会话。

    构造成功即代表捕获已启动；构造失败会抛异常，由调用方决定回退。
    ``read()`` 返回 ``(height, width, 3)`` 的 RGB ``ndarray``，无新帧时返回
    ``None``——那不是错误，只是这一拍游戏还没出新画面。

    线程安全：``read()`` 与 ``close()`` 由内部锁串行化。帧池用
    ``create_free_threaded`` 建立，因此不需要调用方提供 DispatcherQueue。
    """

    def __init__(self, hwnd: int, *, capture_cursor: bool = False) -> None:
        import numpy as np
        import winrt.windows.graphics.capture as wgc
        import winrt.windows.graphics.capture.interop as cap_interop
        import winrt.windows.graphics.directx as directx
        import winrt.windows.graphics.directx.direct3d11.interop as d3d_interop

        self._np = np
        self._lock = threading.Lock()
        self._closed = False
        self._device: Any = None
        self._context: Any = None
        self._dxgi_device: Any = None
        self._staging: Any = None
        self._pool: Any = None
        self._session: Any = None
        self._frames = 0

        try:
            self._item = cap_interop.create_for_window(int(hwnd))
            size = self._item.size
            self._width = int(size.width)
            self._height = int(size.height)
            if self._width <= 0 or self._height <= 0:
                raise RuntimeError(f"capture item has empty size {self._width}x{self._height}")

            self._create_device()
            winrt_device = d3d_interop.create_direct3d11_device_from_dxgi_device(
                self._dxgi_device.value
            )
            self._pool = wgc.Direct3D11CaptureFramePool.create_free_threaded(
                winrt_device,
                directx.DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED,
                # 2 个缓冲：够吸收一次抖动，又不会让我们读到太旧的帧。
                2,
                size,
            )
            self._session = self._pool.create_capture_session(self._item)
            # 黄色捕获边框和鼠标指针都会污染检测输入。这两个属性在较早的
            # Windows 版本上不存在，设不上就算了，不该让会话建不起来。
            for attr, value in (
                ("is_border_required", False),
                ("is_cursor_capture_enabled", bool(capture_cursor)),
            ):
                try:
                    setattr(self._session, attr, value)
                except Exception:
                    pass
            self._create_staging()
            self._session.start_capture()
        except Exception:
            self._teardown()
            raise

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    def _create_device(self) -> None:
        d3d11 = ctypes.WinDLL("d3d11")
        d3d11.D3D11CreateDevice.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        device = ctypes.c_void_p()
        context = ctypes.c_void_p()
        level = ctypes.c_uint()
        hr = d3d11.D3D11CreateDevice(
            None,
            _D3D_DRIVER_TYPE_HARDWARE,
            None,
            # BGRA_SUPPORT 是 WGC 帧池的硬要求，格式不匹配会在建池时失败。
            _D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            0,
            _D3D11_SDK_VERSION,
            ctypes.byref(device),
            ctypes.byref(level),
            ctypes.byref(context),
        )
        if hr != 0 or not device.value:
            raise RuntimeError(f"D3D11CreateDevice failed: hr=0x{hr & 0xFFFFFFFF:08x}")
        self._device = device
        self._context = context
        dxgi_device = _query_interface(device, _IID_IDXGIDEVICE)
        if dxgi_device is None:
            raise RuntimeError("QueryInterface(IDXGIDevice) failed")
        self._dxgi_device = dxgi_device

    def _create_staging(self) -> None:
        """GPU 纹理 CPU 读不了，必须先拷到 STAGING 纹理再 Map。

        staging 纹理只建一次并复用。每帧新建会在驱动里反复分配显存，
        那是这条路径上最容易被写错的性能坑。
        """
        desc = _D3D11_TEXTURE2D_DESC(
            Width=self._width,
            Height=self._height,
            MipLevels=1,
            ArraySize=1,
            Format=_DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDescCount=1,
            SampleDescQuality=0,
            Usage=_D3D11_USAGE_STAGING,
            BindFlags=0,
            CPUAccessFlags=_D3D11_CPU_ACCESS_READ,
            MiscFlags=0,
        )
        staging = ctypes.c_void_p()
        call = _vtbl_call(
            self._device,
            _VTBL_CREATE_TEXTURE2D,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = call(self._device, ctypes.byref(desc), None, ctypes.byref(staging))
        if hr != 0 or not staging.value:
            raise RuntimeError(f"CreateTexture2D(staging) failed: hr=0x{hr & 0xFFFFFFFF:08x}")
        self._staging = staging

    def read(self) -> Any:
        """取最新一帧为 RGB ndarray；无新帧返回 ``None``。"""
        with self._lock:
            if self._closed or self._pool is None:
                return None
            pool = self._pool

        frame = pool.try_get_next_frame()
        if frame is None:
            return None
        try:
            return self._frame_to_array(frame)
        finally:
            try:
                frame.close()
            except Exception:
                pass

    def _frame_to_array(self, frame: Any) -> Any:
        import winrt.windows.graphics.directx.direct3d11.interop as d3d_interop

        content = frame.content_size
        # 窗口被拉伸后帧尺寸会变，而 staging 纹理还是旧的。尺寸不符时直接丢帧，
        # 由上层的窗口跟踪去重建会话——在这里悄悄拉伸会让 bearing_deg 的基准漂掉。
        if int(content.width) != self._width or int(content.height) != self._height:
            return None

        surface = d3d_interop.get_dxgi_surface_from_object(frame.surface)
        surface_ptr = getattr(surface, "value", surface)
        texture = _query_interface(surface_ptr, _IID_ID3D11TEXTURE2D)
        if texture is None:
            return None
        try:
            copy = _vtbl_call(
                self._context,
                _VTBL_CONTEXT_COPY_RESOURCE,
                None,
                ctypes.c_void_p,
                ctypes.c_void_p,
            )
            copy(self._context, self._staging, texture)

            mapped = _D3D11_MAPPED_SUBRESOURCE()
            map_call = _vtbl_call(
                self._context,
                _VTBL_CONTEXT_MAP,
                ctypes.c_long,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.POINTER(_D3D11_MAPPED_SUBRESOURCE),
            )
            hr = map_call(self._context, self._staging, 0, _D3D11_MAP_READ, 0, ctypes.byref(mapped))
            if hr != 0 or not mapped.pData:
                return None
            try:
                # RowPitch 通常大于 width*4（驱动按对齐补齐），必须按它算步长，
                # 否则整张图会斜着错位。
                stride = int(mapped.RowPitch)
                buffer = (ctypes.c_ubyte * (stride * self._height)).from_address(mapped.pData)
                raw = self._np.frombuffer(buffer, dtype=self._np.uint8).reshape(
                    self._height, stride // 4, 4
                )
                # BGRA -> RGB；copy() 必须保留：Unmap 之后那块内存就不属于我们了。
                out = raw[:, : self._width, :3][:, :, ::-1].copy()
            finally:
                unmap = _vtbl_call(
                    self._context, _VTBL_CONTEXT_UNMAP, None, ctypes.c_void_p, ctypes.c_uint
                )
                unmap(self._context, self._staging, 0)
            with self._lock:
                self._frames += 1
            return out
        finally:
            _release(texture)

    @property
    def frames(self) -> int:
        with self._lock:
            return self._frames

    def _teardown(self) -> None:
        for attr in ("_session", "_pool"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        for attr in ("_staging", "_dxgi_device", "_context", "_device"):
            ptr = getattr(self, attr, None)
            if ptr is not None:
                _release(ptr)
                setattr(self, attr, None)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._teardown()


__all__ = ["WgcSession", "wgc_supported"]
