# -*- coding: utf-8 -*-
"""SteamVR 合成器镜像纹理双目取帧（由 ``research/tools/openvr_mirror_probe.py`` 移入 backend）。

在线建图要用它，而 backend 不能依赖 ``research/tools/``；探针本身的说明（实测记录）仍在那个
文件的 docstring，这里只保留取帧所需的部分。``openvr`` 由调用方惰性导入。

pyopenvr 1.26 的 ``getMirrorTextureD3D11`` 封装把 ``byref(device)``（设备指针的地址）
传给了 C API，所以这里绕过封装直接调函数表。
"""
from __future__ import annotations

import ctypes
from typing import Any

import cv2
import numpy as np

from .wgc_capture import (
    _D3D11_MAPPED_SUBRESOURCE,
    _D3D11_TEXTURE2D_DESC,
    _D3D_DRIVER_TYPE_HARDWARE,
    _D3D11_CPU_ACCESS_READ,
    _D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    _D3D11_MAP_READ,
    _D3D11_SDK_VERSION,
    _D3D11_USAGE_STAGING,
    _IID_ID3D11TEXTURE2D,
    _VTBL_CONTEXT_COPY_RESOURCE,
    _VTBL_CONTEXT_MAP,
    _VTBL_CONTEXT_UNMAP,
    _VTBL_CREATE_TEXTURE2D,
    _query_interface,
    _release,
    _vtbl_call,
)

# ID3D11View::GetResource：IUnknown(0-2) + ID3D11DeviceChild(3-6) 之后第一个槽。
_VTBL_VIEW_GET_RESOURCE = 7
# ID3D11Texture2D::GetDesc：IUnknown(0-2) + DeviceChild(3-6) + Resource(7-9) 之后。
_VTBL_TEXTURE2D_GET_DESC = 10

# 只接受 8 位四通道；其余格式（HDR 浮点、10bit）如实报错，不猜转换。
_RGBA_FORMATS = {27, 28, 29}   # R8G8B8A8 TYPELESS / UNORM / UNORM_SRGB
_BGRA_FORMATS = {87, 90, 91}   # B8G8R8A8 UNORM / TYPELESS / UNORM_SRGB

# GPU 端缩小用。ID3D11DeviceContext 的槽位顺序已由 wgc_capture 里的
# Map=14 / Unmap=15 / CopyResource=47 对齐验证过，所以同一张表里
# CopySubresourceRegion=46、GenerateMips=54 可以直接用。
_VTBL_CREATE_SRV = 7                       # ID3D11Device::CreateShaderResourceView
_VTBL_CONTEXT_COPY_SUBRESOURCE_REGION = 46
_VTBL_CONTEXT_GENERATE_MIPS = 54

_D3D11_USAGE_DEFAULT = 0
_D3D11_BIND_SHADER_RESOURCE = 0x8
_D3D11_BIND_RENDER_TARGET = 0x20
_D3D11_RESOURCE_MISC_GENERATE_MIPS = 0x1

# GenerateMips 要求**有类型**的格式；TYPELESS 不能建 RTV。同族转换是位兼容的，
# CopySubresourceRegion 允许跨同族格式拷贝。
_TYPED_FORMAT = {27: 28, 90: 87}


def _mip_for(width: int, height: int, target: tuple[int, int]) -> int:
    """返回仍**不小于** target 的最深 mip。0 表示没有可用的缩小级别。

    mip 链是逐级折半，2880×1620 只能给 1440×810 / 720×405 / 360×202。
    640×360 不在链上，所以取 720×405（mip 2）再由 CPU 补最后一步——
    回读量已经降到 1/16，剩下那次缩放在 CPU 上是微秒级。
    """
    level = 0
    while width // 2 >= target[0] and height // 2 >= target[1]:
        width, height, level = width // 2, height // 2, level + 1
    return level


class MirrorEye:
    """一只眼的镜像纹理：SRV → 纹理 → staging，staging 只建一次。"""

    def __init__(self, compositor: Any, device: Any, context: Any, eye: int,
                 gpu_downscale: tuple[int, int] | None = None) -> None:
        self._compositor = compositor
        self._context = context
        self.eye = eye
        self._device = device
        self._scratch = ctypes.c_void_p()
        self._scratch_srv = ctypes.c_void_p()
        self._mip = 0
        self.gpu_size: tuple[int, int] | None = None
        self._srv = ctypes.c_void_p()
        fn = compositor.function_table.getMirrorTextureD3D11
        error = fn(eye, device, ctypes.byref(self._srv))
        if error != 0 or not self._srv.value:
            raise RuntimeError(f"getMirrorTextureD3D11(eye={eye}) 失败：CompositorError={error}")

        resource = ctypes.c_void_p()
        _vtbl_call(self._srv, _VTBL_VIEW_GET_RESOURCE, None,
                   ctypes.POINTER(ctypes.c_void_p))(self._srv, ctypes.byref(resource))
        if not resource.value:
            raise RuntimeError(f"eye={eye}: SRV.GetResource 返回空")
        try:
            texture = _query_interface(resource, _IID_ID3D11TEXTURE2D)
        finally:
            _release(resource)
        if texture is None:
            raise RuntimeError(f"eye={eye}: 镜像资源不是 Texture2D")
        self._texture = texture

        desc = _D3D11_TEXTURE2D_DESC()
        _vtbl_call(texture, _VTBL_TEXTURE2D_GET_DESC, None,
                   ctypes.POINTER(_D3D11_TEXTURE2D_DESC))(texture, ctypes.byref(desc))
        self.desc = {name: int(getattr(desc, name)) for name, _ in desc._fields_}
        self.texture_ptr = int(texture.value)
        if desc.Format not in _RGBA_FORMATS | _BGRA_FORMATS:
            raise RuntimeError(f"eye={eye}: 不支持的纹理格式 DXGI_FORMAT={desc.Format}")
        if desc.SampleDescCount != 1:
            # 多重采样纹理不能直接 CopyResource 到 staging，需要先 resolve。
            raise RuntimeError(f"eye={eye}: 多重采样纹理（count={desc.SampleDescCount}）未处理")
        self._bgra = desc.Format in _BGRA_FORMATS

        if gpu_downscale is not None:
            self._mip = _mip_for(desc.Width, desc.Height, gpu_downscale)
            if self._mip:
                self._setup_scratch(device, desc, eye)

        # 不缩小时 CopyResource 要求规格完全一致，所以照抄源规格；缩小时
        # staging 只需装得下那一级 mip。
        if self._mip:
            stage_w, stage_h = desc.Width >> self._mip, desc.Height >> self._mip
            stage_fmt, stage_mips = self._scratch_format, 1
            self.gpu_size = (stage_w, stage_h)
        else:
            stage_w, stage_h = desc.Width, desc.Height
            stage_fmt, stage_mips = desc.Format, desc.MipLevels
        staging_desc = _D3D11_TEXTURE2D_DESC(
            Width=stage_w, Height=stage_h, MipLevels=stage_mips,
            ArraySize=desc.ArraySize, Format=stage_fmt,
            SampleDescCount=1, SampleDescQuality=0,
            Usage=_D3D11_USAGE_STAGING, BindFlags=0,
            CPUAccessFlags=_D3D11_CPU_ACCESS_READ, MiscFlags=0,
        )
        staging = ctypes.c_void_p()
        hr = _vtbl_call(device, _VTBL_CREATE_TEXTURE2D, ctypes.c_long, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
            device, ctypes.byref(staging_desc), None, ctypes.byref(staging))
        if hr != 0 or not staging.value:
            raise RuntimeError(f"eye={eye}: CreateTexture2D(staging) hr=0x{hr & 0xFFFFFFFF:08x}")
        self._staging = staging
        self._stage_size = (stage_w, stage_h)

    def _setup_scratch(self, device: Any, desc: Any, eye: int) -> None:
        """建一张带完整 mip 链的中间纹理 + 它的 SRV（GenerateMips 要 SRV）。

        失败时清掉 ``self._mip`` 退回全分辨率回读，而不是抛异常——GPU 缩放是
        纯优化，不该让取帧整体不可用。
        """
        fmt = _TYPED_FORMAT.get(int(desc.Format), int(desc.Format))
        scratch_desc = _D3D11_TEXTURE2D_DESC(
            Width=desc.Width, Height=desc.Height, MipLevels=self._mip + 1,
            ArraySize=1, Format=fmt, SampleDescCount=1, SampleDescQuality=0,
            Usage=_D3D11_USAGE_DEFAULT,
            BindFlags=_D3D11_BIND_SHADER_RESOURCE | _D3D11_BIND_RENDER_TARGET,
            CPUAccessFlags=0, MiscFlags=_D3D11_RESOURCE_MISC_GENERATE_MIPS,
        )
        hr = _vtbl_call(device, _VTBL_CREATE_TEXTURE2D, ctypes.c_long, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
            device, ctypes.byref(scratch_desc), None, ctypes.byref(self._scratch))
        if hr != 0 or not self._scratch.value:
            print(f"eye={eye}: mip 纹理创建失败 hr=0x{hr & 0xFFFFFFFF:08x}，退回全分辨率回读")
            self._mip = 0
            return
        # SRV 传 NULL desc 即覆盖全部 mip，正是 GenerateMips 需要的。
        hr = _vtbl_call(device, _VTBL_CREATE_SRV, ctypes.c_long, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
            device, self._scratch, None, ctypes.byref(self._scratch_srv))
        if hr != 0 or not self._scratch_srv.value:
            print(f"eye={eye}: mip SRV 创建失败 hr=0x{hr & 0xFFFFFFFF:08x}，退回全分辨率回读")
            _release(self._scratch)
            self._scratch = ctypes.c_void_p()
            self._mip = 0
            return
        self._scratch_format = fmt

    def copy(self) -> None:
        """只提交 GPU 拷贝，不等待。配合 :meth:`map_rgb` 实现两眼同帧取样。

        开了 GPU 缩放时是三步：镜像 → mip0（同分辨率，走显存内拷贝）→
        GenerateMips → 取第 ``_mip`` 级到 staging。三步都在 GPU 上排队，
        回读量降到 1/4**mip，这才是省掉那 37 MB PCIe 传输的地方。
        """
        if not self._mip:
            _vtbl_call(self._context, _VTBL_CONTEXT_COPY_RESOURCE, None, ctypes.c_void_p,
                       ctypes.c_void_p)(self._context, self._staging, self._texture)
            return
        copy_region = _vtbl_call(
            self._context, _VTBL_CONTEXT_COPY_SUBRESOURCE_REGION, None,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p)
        # 源 box 传 NULL = 整个 subresource。TYPELESS→UNORM 是同族，允许。
        copy_region(self._context, self._scratch, 0, 0, 0, 0, self._texture, 0, None)
        _vtbl_call(self._context, _VTBL_CONTEXT_GENERATE_MIPS, None, ctypes.c_void_p)(
            self._context, self._scratch_srv)
        copy_region(self._context, self._staging, 0, 0, 0, 0, self._scratch, self._mip, None)

    def map_rgb(self, size: tuple[int, int] | None = None) -> Any:
        """把 staging 读成 RGB ndarray。Map 失败返回 ``None``，不返回半张图。

        ``size=(w, h)`` 时先在映射内存上直接 INTER_AREA 缩小再转通道。实测
        （2880×1620，2026-09-26）：原来的 ``[:, :, :3][:, :, ::-1].copy()``
        是负步长三通道收集，单眼 33 ms；四通道直接缩到 640×360 只要 8 ms，
        双目抓帧因此从 ~93 ms 降到 ~30 ms。

        构造时给了 ``gpu_downscale`` 的话，staging 里已经是缩过的图，这里
        只补 mip 链凑不出的最后一步（720×405 → 640×360）。
        """
        mapped = _D3D11_MAPPED_SUBRESOURCE()
        hr = _vtbl_call(self._context, _VTBL_CONTEXT_MAP, ctypes.c_long, ctypes.c_void_p,
                        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                        ctypes.POINTER(_D3D11_MAPPED_SUBRESOURCE))(
            self._context, self._staging, 0, _D3D11_MAP_READ, 0, ctypes.byref(mapped))
        if hr != 0 or not mapped.pData:
            return None
        # staging 可能已经是 GPU 缩过的尺寸，不能用源纹理的 desc。
        width, height = self._stage_size
        if size is not None and (width, height) == tuple(size):
            size = None            # GPU 已经到位，省掉 CPU 这一次 resize
        try:
            stride = int(mapped.RowPitch)
            buffer = (ctypes.c_ubyte * (stride * height)).from_address(mapped.pData)
            raw = np.frombuffer(buffer, dtype=np.uint8).reshape(height, stride // 4, 4)[:, :width]
            # 结果必须是新分配的数组：Unmap 之后那块内存就不属于我们了。
            # cv2.resize / cvtColor 都写入新缓冲区，满足这一点。
            if size is not None:
                raw = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB if self._bgra else cv2.COLOR_RGBA2RGB)
        finally:
            _vtbl_call(self._context, _VTBL_CONTEXT_UNMAP, None, ctypes.c_void_p,
                       ctypes.c_uint)(self._context, self._staging, 0)

    def read(self) -> Any:
        """单眼取一帧。**双目不要连续调两次它**，用 :func:`read_stereo`。"""
        self.copy()
        return self.map_rgb()

    def close(self) -> None:
        _release(self._scratch_srv)
        _release(self._scratch)
        _release(self._staging)
        _release(self._texture)
        if self._srv.value:
            # 按文档用 ReleaseMirrorTextureD3D11，而不是对 SRV 直接 Release。
            self._compositor.function_table.releaseMirrorTextureD3D11(self._srv)
            self._srv = ctypes.c_void_p()


def read_stereo(left: MirrorEye, right: MirrorEye,
                size: tuple[int, int] | None = None) -> tuple[Any, Any]:
    """两眼**同一合成帧**的画面。

    逐眼 ``read()`` 时，左眼的 Map 会阻塞到 GPU 完成（整张 2880×1620 回读约
    37 ms），右眼的拷贝要等它之后才提交，两眼因此相隔约 3 个 90 Hz 合成帧。
    静止时看不出来；行走时实测每帧立体点从 1118 掉到 155（run2），剩下的点
    水平视差里还混着运动光流，导致尺度随深度系统性漂移。

    这里两次 CopyResource 连着入队，GPU 按序执行、间隔远小于一帧，再统一 Map。
    """
    left.copy()
    right.copy()
    return left.map_rgb(size), right.map_rgb(size)


def create_device() -> tuple[Any, Any]:
    d3d11 = ctypes.WinDLL("d3d11")
    d3d11.D3D11CreateDevice.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    device, context, level = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_uint()
    hr = d3d11.D3D11CreateDevice(None, _D3D_DRIVER_TYPE_HARDWARE, None,
                                 _D3D11_CREATE_DEVICE_BGRA_SUPPORT, None, 0,
                                 _D3D11_SDK_VERSION, ctypes.byref(device),
                                 ctypes.byref(level), ctypes.byref(context))
    if hr != 0 or not device.value:
        raise RuntimeError(f"D3D11CreateDevice failed: hr=0x{hr & 0xFFFFFFFF:08x}")
    return device, context


def projection_intrinsics(system: Any, eye: int, out_size: tuple[int, int],
                          src_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """``getProjectionRaw`` 的半角正切 → 源分辨率内参，再按输出尺寸缩放。返回 fx, fy, cx, cy。"""
    left_b, right_b, top_b, bottom_b = system.getProjectionRaw(eye)
    src_w, src_h = src_size
    sx, sy = out_size[0] / src_w, out_size[1] / src_h
    fx = src_w / (right_b - left_b) * sx
    fy = src_h / (bottom_b - top_b) * sy
    cx = src_w * (-left_b) / (right_b - left_b) * sx
    cy = src_h * (-top_b) / (bottom_b - top_b) * sy
    return fx, fy, cx, cy


__all__ = ["MirrorEye", "create_device", "projection_intrinsics", "read_stereo"]
