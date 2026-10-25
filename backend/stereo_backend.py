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
import time
from typing import Any

import numpy as np

from .nav_mapping import disparity_range_px, make_sgbm, stereo_disparity

__all__ = ("STEREO_BACKENDS", "las2_prewarm", "las2_prewarm_status", "make_stereo_backend")

STEREO_BACKENDS = ("sgbm", "las2")
LAS2_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "las_ckpt", "LAS2_M.pth")

# LAS2 每进程只建一份：权重占显存，编译 70–330 s。会话重开（缩放变化）时复用同一个已编译的
# 适配器；旧会话的线程若还卡在编译里，新会话的首帧在适配器锁上等它编完，不会并发编两次。
_las2_lock = threading.Lock()
_las2: "_LAS2Adapter | None" = None

# 启动预热（``las2_prewarm``）：热缓存下首帧仍要 ~80 s（AOT 反序列化 + 载 Triton 内核），新尺寸
# 冷编译 ~5 min。挪到后端启动时的后台线程里做，会话首帧就不用等。每进程只成功一次。
# state：idle 没起过 / loading 载权重 / compiling 跑首帧 / ready 编译版 / eager 编译失败退回 eager / failed 载不起来。
_warm_lock = threading.Lock()
_warm: dict[str, Any] = {"state": "idle"}


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



def las2_prewarm(shape: tuple[int, int]) -> dict[str, Any]:
    """后台线程载 LAS2 并按 ``shape = (H, W)`` 跑一帧空图，把 torch.compile 的开销挪到启动时。

    编译缓存按**补到 32 倍数后的**尺寸命中（606 → 608），所以 shape 只要落在同一档就有效。
    预热和会话首帧共用同一个适配器（``make_stereo_backend`` 单例 + 适配器锁）：会话在预热中途开起来，
    首帧就在锁上等预热编完，不会编两次。重复调用直接返回当前状态；只有 failed 会重试。
    """
    h, w = int(shape[0]), int(shape[1])
    with _warm_lock:
        if _warm["state"] not in ("idle", "failed"):
            return _warm_view()
        _warm.clear()
        _warm.update(state="loading", shape=[h, w], error=None, started=time.perf_counter())
    threading.Thread(target=_prewarm_run, args=(h, w), name="las2-prewarm", daemon=True).start()
    return las2_prewarm_status()


def _prewarm_set(**kv: Any) -> None:
    with _warm_lock:
        _warm.update(kv)


def _prewarm_run(h: int, w: int) -> None:
    try:
        m = make_stereo_backend("las2")
        _prewarm_set(state="compiling")
        z = np.zeros((h, w, 3), np.uint8)
        m.compute(z, z)
        d = m.describe()
        _prewarm_set(state="ready" if d.get("compiled") else "eager", error=d.get("compile_error"))
    except Exception as exc:  # noqa: BLE001 - 缺 torch / XPU / 权重：如实报，会话照常退回 SGBM
        _prewarm_set(state="failed", error=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        _prewarm_set(done=time.perf_counter())


def _warm_view() -> dict[str, Any]:
    out = dict(_warm)
    t0, t1 = out.pop("started", None), out.pop("done", None)
    out["elapsed_s"] = None if t0 is None else round((t1 if t1 is not None else time.perf_counter()) - t0, 1)
    return out


def las2_prewarm_status() -> dict[str, Any]:
    """``{"state", "shape", "error", "elapsed_s"}``；elapsed_s 在跑时是已用时，跑完是总耗时。"""
    with _warm_lock:
        return _warm_view()

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
    """RGB → replicate pad 到 32 的倍数 → 网络（half）→ 去 pad → float32。

    ⚠️ **LAS2 喂 RGB 比 gray3 更准**（2026-10-25，Middlebury 10 场 eager half，
    `.tmp/las2_color_probe.py`）：RGB 比 gray3 EPE 720 档 −5.5%（10/10 场胜）、1080 档 −9.5%
    （9/10），耗时相同；BGR 介于两者之间 ⇒ 通道序要对。FACTS「真彩色」行的「学习型 gray3 胜
    6–9/10」是 AANet/StereoNet 的结论，**对 LAS2 不成立**。上游 loader 只对单通道图 tile 成
    三通道，彩色图保留 RGB（`core/stereo_datasets.py:71`）。``nav_online`` 选 las2 时递 RGB
    （镜像读回已按 DXGI 格式转成 RGB，`openvr_mirror.map_rgb`）；拿不到 RGB 时收灰度，照旧复制。
    VRChat 镜像上的收益还没测。

    ``dynamic=False``：每个新尺寸编一次（采集尺寸会话内不变）。某尺寸的首次编译失败（没有 cl、
    没设 PYTHONUTF8 …）就**永久**退回 eager 并把原因留在 ``describe()``——eager 720 档约 105 ms，
    慢一倍但结果相同，比整条双目断掉好。``compute`` 持锁：编译期与跨会话复用都只有一个线程在跑。
    """

    __slots__ = ("_model", "_dev", "_compiled", "_ok_shapes", "compile_error", "_shape", "_pad", "_lock",
                 "last_timing")

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
        # 上一帧分段墙钟（ms）：upload = numpy→XPU+pad；launch = 前向调用返回（XPU 异步，
        # 这段是 Python 派发，GIL 等待会落在这里）；sync = .cpu() 等 GPU 跑完 + 回传。
        # 只用 perf_counter，不加任何 XPU 同步/算子。
        self.last_timing: dict[str, float] | None = None

    def _eager(self, a: Any, b: Any) -> Any:
        return self._model(a.half(), b.half())

    def compute(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        with self._lock:
            return self._compute(left_rgb, right_rgb)

    def _compute(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        H, W = left_rgb.shape[:2]
        if self._shape != (H, W):
            ph, pw = (-H) % 32, (-W) % 32
            self._pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
            self._shape = (H, W)
        pad = self._pad

        def to_dev(img: np.ndarray) -> Any:
            # (H, W, 3) RGB → (1, 3, H, W)；(H, W) 灰度 → 复制成 3 通道（没有 RGB 时的退路）。
            t = torch.from_numpy(np.ascontiguousarray(img)).to(self._dev)
            t = t.permute(2, 0, 1)[None].float() if t.ndim == 3 else t[None, None].float().expand(1, 3, -1, -1)
            return F.pad(t, pad, mode="replicate")

        t0 = time.perf_counter()
        with torch.no_grad():
            a, b = to_dev(left_rgb), to_dev(right_rgb)
            t1 = time.perf_counter()
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
            t2 = time.perf_counter()
        d = d[0, 0, pad[2]:pad[2] + H, pad[0]:pad[0] + W]
        out = d.float().cpu().numpy()
        t3 = time.perf_counter()
        self.last_timing = {"upload": (t1 - t0) * 1e3, "launch": (t2 - t1) * 1e3, "sync": (t3 - t2) * 1e3}
        return out

    def describe(self) -> dict[str, Any]:
        """``compiled``：None = 还没跑过首帧（正在编译或没帧），True = 编译版，False = 退回 eager。"""
        compiled = False if self.compile_error else (True if self._ok_shapes else None)
        return {"kind": "las2", "compiled": compiled, "compile_error": self.compile_error}
