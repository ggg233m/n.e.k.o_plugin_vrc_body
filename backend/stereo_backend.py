# -*- coding: utf-8 -*-
"""可插拔的双目视差后端：SGBM（默认，无外部依赖）与 LAS2-M（可选，需 torch-xpu + timm）。

两种后端同一接口：``matcher.compute(left_gray, right_gray) -> float32 视差（px，<=0 无效）``，
``matcher.describe() -> dict``（进 ``status.stereo_backend``）。

LAS2 只在 `.venv_las`（Py3.14.9 + torch 2.14.1+xpu + timm 1.0.30）里可用；发行包（Py3.11）
没有 torch，``make_stereo_backend("las2")`` 会立即抛错，由调用者决定退回 SGBM。

⚠️ LAS2 编译的进程环境（2026-10-25 实测，缺一条就编不过，自动退回 eager）：
  * PATH 上要有 MSVC ``cl``（先跑 vcvars64.bat），否则 ``Failed to find C compiler``；
  * ``PYTHONUTF8=1``，否则 inductor 用 gbk 解 cl 输出报 ``UnicodeDecodeError``；
  * 仓库根 ``_tc_run314.bat`` 把两条连同 spawn / 独立缓存目录一起设好。
首次编译 70–330 s（LAS2 doc 订正 B.4），发生在第一次 ``.compute``，所以只能在双目解耦线程里用。

License：网络结构取自 LiteAnyStereo（MIT），见 THIRD_PARTY_NOTICES.md 与 ``las2_model``。
"""
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from .nav_mapping import disparity_range_px, make_sgbm, stereo_disparity

__all__ = ("STEREO_BACKENDS", "make_stereo_backend")

STEREO_BACKENDS = ("sgbm", "las2")
LAS2_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "las_ckpt", "LAS2_M.pth")

# LAS2 每进程只建一份：权重占显存，编译 70–330 s。会话重开（缩放变化）时复用同一个已编译的
# 适配器；旧会话的线程若还卡在编译里，新会话的首帧在适配器锁上等它编完，不会并发编两次。
_las2_lock = threading.Lock()
_las2: "_LAS2Adapter | None" = None


def make_stereo_backend(kind: str = "sgbm", *, fx: float = 202.5) -> Any:
    """工厂：``kind = "sgbm"`` CPU SGBM；``kind = "las2"`` LAS2-M（XPU，half + 两眼合批 + torch.compile）。

    ``fx`` 只给 SGBM 用：视差范围随 fx 缩放（720 / 1080 / 1440 档 nd = 64 / 96 / 128，
    ``disparity_range_px``），不缩放会把近场裁掉。LAS2 的视差上限是网络固定的 192 px
    （48 层代价体 × 4），1080 档 fx 303.75 下对应 Z ≈ 0.2 追踪米，用不到。

    为什么只做 M：S 快但精度输给 SGBM（EPE 1.975 vs 1.783）；L 的发布权重 27.3 M ≠ 论文
    11.4 M、精度反比 M 差 43%（LAS2 doc §8.1）。
    """
    if kind == "sgbm":
        return _SGBMAdapter(make_sgbm(disparity_range_px(fx)))
    if kind == "las2":
        global _las2
        with _las2_lock:
            if _las2 is None:
                _las2 = _las2_adapter()
            return _las2
    raise ValueError(f"unknown stereo backend {kind!r} (expected one of {STEREO_BACKENDS})")


class _SGBMAdapter:
    __slots__ = ("_m",)

    def __init__(self, sgbm: Any) -> None:
        self._m = sgbm

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        return stereo_disparity(left_gray, right_gray, self._m)

    def describe(self) -> dict[str, Any]:
        return {"kind": "sgbm", "num_disparities": int(self._m.getNumDisparities())}


def _las2_adapter() -> Any:
    """载权重、搬上 XPU、包编译。缺 torch / timm / 权重 / XPU 都在这里抛错。"""
    import torch
    try:
        import timm  # noqa: F401 - FasterNet-T0 主干
    except ImportError as exc:
        raise RuntimeError("LAS2 backend needs timm; the release package (Py3.11) has neither torch "
                           "nor timm, run from .venv_las") from exc
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("LAS2 backend needs an XPU device (torch.xpu.is_available() is False)")
    if not os.path.isfile(LAS2_CKPT):
        raise FileNotFoundError(f"LAS2 checkpoint missing: {LAS2_CKPT}")

    from .las2_model import LiteAnyStereoV2M

    sd = torch.load(LAS2_CKPT, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    m = LiteAnyStereoV2M()
    m.load_state_dict(sd, strict=True)
    dev = torch.device("xpu")
    model = m.to(dev).eval().half()
    return _LAS2Adapter(model, dev)


class _LAS2Adapter:
    """灰度 → 复制成 3 通道 → replicate pad 到 32 的倍数 → 网络（half）→ 去 pad → float32。

    灰度复制而不是真彩：Middlebury 实测学习型立体在 gray3 上逐场胜 6–9/10（FACTS「真彩色」行），
    镜像采集端本来就给灰度。

    ``dynamic=False``：每个新尺寸编一次（采集尺寸会话内不变）。某尺寸的首次编译失败（没有 cl、
    没设 PYTHONUTF8 …）就**永久**退回 eager 并把原因留在 ``describe()``——eager 720 档约 105 ms，
    慢一倍但结果相同，比整条双目断掉好。``compute`` 持锁：编译期与跨会话复用都只有一个线程在跑。
    """

    __slots__ = ("_model", "_dev", "_compiled", "_ok_shapes", "compile_error", "_shape", "_pad", "_lock")

    def __init__(self, model: Any, device: Any) -> None:
        import torch
        self._model = model
        self._dev = device
        self._compiled = torch.compile(lambda a, b: model(a.half(), b.half()), dynamic=False)
        self._ok_shapes: set[tuple[int, int]] = set()
        self.compile_error: str | None = None
        self._shape: tuple[int, int] | None = None
        self._pad: list[int] = [0, 0, 0, 0]
        self._lock = threading.Lock()

    def _eager(self, a: Any, b: Any) -> Any:
        return self._model(a.half(), b.half())

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        with self._lock:
            return self._compute(left_gray, right_gray)

    def _compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        H, W = left_gray.shape[:2]
        if self._shape != (H, W):
            ph, pw = (-H) % 32, (-W) % 32
            self._pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
            self._shape = (H, W)
        pad = self._pad

        def to_dev(g: np.ndarray) -> Any:
            t = torch.from_numpy(np.ascontiguousarray(g)).to(self._dev)[None, None].float()
            return F.pad(t.expand(1, 3, -1, -1), pad, mode="replicate")

        with torch.no_grad():
            a, b = to_dev(left_gray), to_dev(right_gray)
            if self.compile_error is not None:
                d = self._eager(a, b)
            elif (H, W) in self._ok_shapes:
                d = self._compiled(a, b)          # 已编译过的尺寸：运行期错误照常抛
            else:
                try:
                    d = self._compiled(a, b)
                    self._ok_shapes.add((H, W))
                except Exception as exc:  # noqa: BLE001 - 只兜编译期
                    self.compile_error = f"{type(exc).__name__}: {exc}"[:300]
                    d = self._eager(a, b)
        d = d[0, 0, pad[2]:pad[2] + H, pad[0]:pad[0] + W]
        return d.float().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        """``compiled``：None = 还没跑过首帧（正在编译或没帧），True = 编译版，False = 退回 eager。"""
        compiled = False if self.compile_error else (True if self._ok_shapes else None)
        return {"kind": "las2", "compiled": compiled, "compile_error": self.compile_error}
