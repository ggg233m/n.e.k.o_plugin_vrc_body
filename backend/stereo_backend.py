# -*- coding: utf-8 -*-
"""可插拔的双目视差后端：SGBM（默认，无外部依赖）与 LAS2-M（可选，需 torch-xpu + timm）。

两种后端同一接口：`matcher.compute(left_gray, right_gray) -> float32 视差（px，<=0 无效）`，
`matcher.describe() -> dict`（进 `status.stereo_backend`）。

LAS2 有两种跑法：

* **子进程（默认，`subprocess=True`）**：主后端全程不 import torch，由
  `backend/stereo_worker.py` 在独立解释器里载模型、编译、推理。共享内存传图，命名信号量
  同步。好处有三条：①主后端可以留在 `.venv`（Py3.12）而 LAS2 用 `.venv_las`（Py3.14）；
  ②独立 GIL，torch.compile 每帧几百次 GIL 交接不再被建图线程拖长（订正 B.2 实测最坏
  两个数量级）；③编译/崩溃都不带走主后端，只丢一帧。
* **进程内（`subprocess=False`，旧路径）**：只有从 `.venv_las` 启动时才可用。

⚠️ LAS2 编译的进程环境（2026-10-25 实测，缺一条就编不过，自动退回 eager）：
  * PATH 上要有 MSVC `cl`（`_msvc_env()` 会把 vcvars64 的环境并进子进程）；
  * `PYTHONUTF8=1`，否则 inductor 用 gbk 解 cl 输出报 `UnicodeDecodeError`；
  * 仓库根 `_tc_run314.bat` 把两条连同 spawn / 独立缓存目录一起设好 —— 子进程沿用同一份
    缓存目录（`_tc_inductor_314` / `_tc_triton_314`），热缓存下首帧 ~80 s 而不是 5 min。
首次编译 70–330 s（LAS2 doc 订正 B.4），发生在子进程的预热里，主进程只轮询 PHASE。

License：网络结构取自 LiteAnyStereo（MIT），见 THIRD_PARTY_NOTICES.md 与 `las2_model`。
"""
from __future__ import annotations

import atexit
import collections
import contextlib
import json
import os
import subprocess as _subprocess
import sys
import threading
import time
import uuid
from typing import Any

import numpy as np

from .nav_mapping import disparity_range_px, make_sgbm, stereo_disparity
from .stereo_worker import (
    CTRL, CTRL_BYTES, H_IX, LAUNCH_MS, LAST_MS, PHASE, PHASE_COMPILING, PHASE_EAGER, PHASE_EXITED,
    PHASE_FAILED, PHASE_LOADING, PHASE_READY, PROTOCOL_VERSION, SEQ_DONE, SEQ_REQ, SYNC_MS, FRAMES,
    UPLOAD_MS, VERSION_IX, W_IX, WORKER_PID, control, disparity, read_error,
)

__all__ = ("STEREO_BACKENDS", "NamedSemaphore", "StereoBusy", "StereoWorkerError", "close_las2",
           "las2_prewarm", "las2_prewarm_status", "las2_worker_status", "make_stereo_backend")

STEREO_BACKENDS = ("sgbm", "las2")
LAS2_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "las_ckpt", "LAS2_M.pth")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SHAPE = (606, 1080)       # 现行驱动 1440×810 + capture_width=1080 ⇒ mip1
DEFAULT_TIMEOUT_S = 0.5           # 单帧等子进程的上限；10 Hz 的预算是 100 ms，留足余量
SPAWN_TIMEOUT_S = 60.0            # 子进程启动 + 载入引擎的上限（torch 冷 import 约 10 s，权重几秒）


class StereoBusy(TimeoutError):
    """子进程还在载入/编译，或上一帧没回来。**丢这一帧**，不算错误。

    单独一个类型是因为它跟真正的故障处理完全不同：编译期每 ~100 ms 就有一帧到这里，
    记进 `status.errors` 会把真正的故障挤掉（只留 5 条），而它其实是预期状态。
    """


class StereoWorkerError(RuntimeError):
    """子进程起不来 / 已崩溃 / 预热失败。本会话退回 SGBM。"""


# ---- 命名信号量 ----------------------------------------------------------------
class NamedSemaphore:
    """跨进程命名信号量。Windows 用 Win32 `CreateSemaphoreW/OpenSemaphoreW`。

    为什么不用标准库：
    * `multiprocessing.Semaphore(name=...)` 在 Windows 上**不支持**取名（只认 POSIX）；
    * 内部 `_multiprocessing.SemLock(..., name, unlink)` 虽然收得下名字，实测（2026-10-26，
      Py3.12 父 + Py3.14 子）子进程 release 之后**父进程永远读到 0** —— 每次取值都像新开的
      一个，坏的。所以直接用 Win32 API。

    句柄由创建它的进程持有；`close()` 关本进程的句柄，对象本身在最后一个句柄关闭时消失。
    """

    __slots__ = ("_handle", "_closed", "_name")

    SEMAPHORE_MODIFY_STATE = 0x0002
    SYNCHRONIZE = 0x00100000
    _ACCESS = SEMAPHORE_MODIFY_STATE | SYNCHRONIZE

    def __init__(self, handle: Any, name: str | None = None) -> None:
        self._handle = handle
        self._closed = False
        self._name = name

    # -- 构造 --
    @classmethod
    def create(cls, name: str, initial: int = 0, maximum: int = 1) -> "NamedSemaphore":
        if os.name != "nt":
            return cls._posix(name, initial, maximum)
        import ctypes
        k32 = _kernel32(ctypes)
        handle = k32.CreateSemaphoreW(None, int(initial), int(maximum), name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return cls(handle, name)

    @classmethod
    def open(cls, name: str, timeout_s: float = SPAWN_TIMEOUT_S) -> "NamedSemaphore":
        if os.name != "nt":
            return cls._posix(name, 0, 1, create=False)
        import ctypes
        k32 = _kernel32(ctypes)
        deadline = time.monotonic() + max(0.1, timeout_s)
        last = 0
        while True:
            handle = k32.OpenSemaphoreW(cls._ACCESS, False, name)
            if handle:
                return cls(handle, name)
            last = ctypes.get_last_error()
            if time.monotonic() >= deadline:
                raise StereoWorkerError(f"打不开命名信号量 {name}（CreateSemaphoreW 失败，err={last}）")
            time.sleep(0.02)

    @classmethod
    def _posix(cls, name: str, initial: int, maximum: int, create: bool = True) -> "NamedSemaphore":
        import _multiprocessing
        import multiprocessing.synchronize as sync
        sem = _multiprocessing.SemLock(sync.SEMAPHORE, int(initial), int(maximum),
                                       name if name.startswith("/") else "/" + name, create)
        return cls(sem, name)

    # -- 用法 --
    def release(self, count: int = 1) -> bool:
        if self._closed:
            return False
        if os.name != "nt":
            for _ in range(int(count)):
                self._handle.release()
            return True
        import ctypes
        return bool(_kernel32(ctypes).ReleaseSemaphore(self._handle, int(count), None))

    def acquire(self, timeout_s: float | None = None) -> bool:
        """等信号量。`timeout_s=None` = 无限等；超时返回 False（不抛）。"""
        if self._closed:
            return False
        if os.name != "nt":
            return bool(self._handle.acquire(timeout=timeout_s))
        import ctypes
        k32 = _kernel32(ctypes)
        ms = 0xFFFFFFFF if timeout_s is None else max(0, int(timeout_s * 1000))
        rc = k32.WaitForSingleObject(self._handle, ms)
        return rc in (0x0, 0x80)          # WAIT_OBJECT_0 / WAIT_ABANDONED

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if os.name == "nt":
            try:
                import ctypes
                _kernel32(ctypes).CloseHandle(self._handle)
            except Exception:  # noqa: BLE001 - 关闭失败不能挡住退出
                pass
        self._handle = None


_K32: Any = None


def _kernel32(ctypes: Any) -> Any:
    """按需绑一次 kernel32（函数签名固定，避免每次调用都重新推断）。"""
    global _K32
    if _K32 is not None:
        return _K32
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateSemaphoreW.restype = ctypes.c_void_p
    k32.CreateSemaphoreW.argtypes = (ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_wchar_p)
    k32.OpenSemaphoreW.restype = ctypes.c_void_p
    k32.OpenSemaphoreW.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)
    k32.ReleaseSemaphore.restype = ctypes.c_int
    k32.ReleaseSemaphore.argtypes = (ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p)
    k32.WaitForSingleObject.restype = ctypes.c_uint32
    k32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    k32.CloseHandle.argtypes = (ctypes.c_void_p,)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    _K32 = k32
    return k32


# LAS2 每进程只建一份：权重占显存，编译 70–330 s。会话重开（缩放变化）时复用同一个已编译的
# 适配器；旧会话的线程若还卡在编译里，新会话的首帧在适配器锁上等它编完，不会并发编两次。
_las2_lock = threading.Lock()
_las2: "_LAS2Adapter | None" = None

# 子进程模式单例（形状变了才重建；见 _las2_subprocess）。锁比 _las2_lock 单独一把，
# 因为建子进程要几秒，不能把进程内路径一起堵住。
_sub_lock = threading.Lock()
_sub: "_LAS2SubprocessAdapter | None" = None
# 关闭闸门：close_las2() 之后禁止再起新子进程。为什么需要 —— prewarm 跑在后台线程里，
# 后端 stop() 时它可能正卡在 _spawn_las2 中间（建共享内存 / 等握手）。那时 close_las2()
# 看到 _sub 还是 None 就直接返回了，紧接着 prewarm 线程把 adapter 建完、塞进 _sub，
# 于是这个刚起来的子进程再也没人关（实测后端退出后残留 2 个 worker 进程）。
_closing = threading.Event()

# 启动预热（`las2_prewarm`）：热缓存下首帧仍要 ~80 s（AOT 反序列化 + 载 Triton 内核），新尺寸
# 冷编译 ~5 min。挪到后端启动时的后台线程里做，会话首帧就不用等。每进程只成功一次。
# state：idle 没起过 / loading 载权重 / compiling 跑首帧 / ready 编译版 / eager 编译失败退回 eager / failed 载不起来。
_warm_lock = threading.Lock()
_warm: dict[str, Any] = {"state": "idle"}

# 总闸（只给测试/CI 用）：设 NEKO_LAS_DISABLE=1 ⇒ 一切 LAS2 请求立刻失败，调用方退回 SGBM。
# 为什么需要：service.start() 会顺手 prewarm，任何「真起服务」的用例都会拉起一个 3.14 子进程
# 并在里面开始 5 min 的编译 —— 测试套件不该有这种副作用。生产上没人设这个变量。
_DISABLED = os.environ.get("NEKO_LAS_DISABLE") == "1"


def make_stereo_backend(kind: str = "sgbm", *, fx: float = 202.5, subprocess: bool = False,
                        shape: tuple[int, int] | None = None, engine: str | None = None,
                        timeout_s: float = DEFAULT_TIMEOUT_S,
                        spec_extra: dict[str, Any] | None = None) -> Any:
    """工厂：`kind = "sgbm"` CPU SGBM；`kind = "las2"` LAS2-M（XPU，half + 两眼合批 + torch.compile）。

    `fx` 只给 SGBM 用：视差范围随 fx 缩放（720 / 1080 / 1440 档 nd = 64 / 96 / 128，
    `disparity_range_px`），不缩放会把近场裁掉。LAS2 的视差上限是网络固定的 192 px
    （48 层代价体 × 4），1080 档 fx 303.75 下对应 Z ≈ 0.2 追踪米，用不到。

    `subprocess=True` 时 las2 起独立进程（需同时给 `shape`，缺省按 `DEFAULT_SHAPE`）；
    `engine="echo"` 是测试用假引擎，不起 torch。

    为什么只做 M：S 快但精度输给 SGBM（EPE 1.975 vs 1.783）；L 的发布权重 27.3 M ≠ 论文
    11.4 M、精度反比 M 差 43%（LAS2 doc §8.1）。
    """
    if kind == "sgbm":
        return _SGBMAdapter(make_sgbm(disparity_range_px(fx)))
    if kind == "las2":
        if _DISABLED and (engine or os.environ.get("NEKO_LAS_ENGINE") or "las2").lower() != "echo":
            # 只挡真 LAS2：echo 是测试引擎，不载 torch、无副作用，IPC 用例必须能跑。
            raise StereoWorkerError("NEKO_LAS_DISABLE=1：本进程禁用 LAS2（echo 引擎不受影响）")
        if subprocess:
            return _las2_subprocess(shape or DEFAULT_SHAPE, fx=fx, engine=engine,
                                    timeout_s=timeout_s, spec_extra=spec_extra)
        global _las2
        with _las2_lock:
            if _las2 is None:
                _las2 = _las2_adapter()
            return _las2
    raise ValueError(f"unknown stereo backend {kind!r} (expected one of {STEREO_BACKENDS})")


def _las2_python() -> str:
    """LAS2 子进程用哪个解释器。

    优先 `NEKO_LAS_PYTHON`（测试/换机时钉死），其次仓库里的 `.venv_las`（Py3.14 + torch-xpu），
    最后回落到当前解释器 —— 那时子进程 import torch 会失败，父进程收到 StereoWorkerError
    并退回 SGBM，行为与「没有 LAS2」一致。
    """
    explicit = os.environ.get("NEKO_LAS_PYTHON")
    if explicit:
        return explicit
    for rel in (os.path.join(".venv_las", "Scripts", "python.exe"),
                os.path.join(".venv_las", "bin", "python")):
        candidate = os.path.join(PROJECT_DIR, rel)
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def _las2_env() -> dict[str, str]:
    """子进程环境：当前环境 + inductor/triton 那套（与 `_tc_run314.bat` 同一份缓存目录）。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"                      # inductor 用 gbk 解 cl 输出会 UnicodeDecodeError
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "16")
    # 🔴 Windows 上只有 spawn 走得通（2026-10-26 实测两条死路）：
    #   * "subprocess"：torch 的 SubprocPool 用 subprocess.Popen(pass_fds=...) 传结果管道，
    #     而 Windows **不支持 pass_fds** ⇒ 编译直接 BackendCompilerFailed 退回 eager
    #     （实测报 "AssertionError: pass_fds not supported on Windows"，见
    #     torch/_inductor/compile_worker/subproc_pool.py:221）；
    #   * "fork"：Windows 没有。
    # spawn 用 `-c "from multiprocessing.spawn import spawn_main; ..."` 起进程，不重放我们的
    # argv（Python 3.14 的 multiprocessing/spawn.py:83-97），所以不会凭空多出 stereo_worker。
    env["TORCHINDUCTOR_WORKER_START"] = "spawn"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(PROJECT_DIR, "_tc_inductor_314"))
    env.setdefault("TRITON_CACHE_DIR", os.path.join(PROJECT_DIR, "_tc_triton_314"))
    # 编译中间产物落到仓库根（C 盘只剩 1 GB，inductor 的临时目录很容易把它写满）。
    env.setdefault("TMP", PROJECT_DIR)
    env.setdefault("TEMP", PROJECT_DIR)
    # 子进程要从仓库根 import `backend.stereo_worker`，cwd 已经是仓库根；再加一份保险。
    parent = os.path.dirname(PROJECT_DIR)
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = parent if not old else parent + os.pathsep + old
    # MSVC cl：inductor 找不到编译器就退回 eager（慢一倍，但结果相同）。找到就并进 PATH。
    msvc = _msvc_env()
    if msvc:
        env.update(msvc)
    return env


_MSVC_CACHE: dict[str, str] | None = None


def _msvc_env() -> dict[str, str]:
    """从 vcvars64.bat 抓一份环境（PATH/INCLUDE/LIB）。找不到就返回空 dict。

    `cl` 不在 PATH 上时 torch.compile 会直接报 `Failed to find C compiler` 并退回 eager。
    抓一次缓存住：vcvars 要跑 ~2 s，而这只在起 LAS2 子进程时发生一次。
    """
    global _MSVC_CACHE
    if _MSVC_CACHE is not None:
        return _MSVC_CACHE
    _MSVC_CACHE = {}
    if os.name != "nt" or os.environ.get("NEKO_LAS_SKIP_VCVARS"):
        return _MSVC_CACHE
    shutil_which = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "where.exe")
    try:
        import subprocess
        found = subprocess.run([shutil_which, "cl"], capture_output=True, text=True, timeout=10)
        if found.returncode == 0:
            return _MSVC_CACHE                 # 已经在 PATH 上，不用抓
    except Exception:  # noqa: BLE001 - 环境探测失败不该挡住起进程
        return _MSVC_CACHE
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    bat = next((p for p in candidates if os.path.isfile(p)), None)
    if bat is None:
        return _MSVC_CACHE
    try:
        # `call bat && set` 是把 vcvars 的环境导出的标准做法；输出是 UTF-8（PYTHONUTF8 在 cmd 里不管用，
        # 但 set 只打 ASCII 变量名，值里的中文路径用 utf-8 解一般也对；解不了就退回 replace）。
        proc = _subprocess.run(["cmd", "/c", f'call "{bat}" >nul && set'], capture_output=True,
                               timeout=90, env=dict(os.environ, VSLANG="1033"))
        text = proc.stdout.decode("utf-8", "replace")
        out: dict[str, str] = {}
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            if sep and key and key.upper() not in ("PATH", "INCLUDE", "LIB", "LIBPATH"):
                continue
            if sep and key:
                out[key] = value
        _MSVC_CACHE = out
    except Exception:  # noqa: BLE001 - 抓不到就让 inductor 自己去试
        _MSVC_CACHE = {}
    return _MSVC_CACHE


def las2_prewarm(shape: tuple[int, int], *, subprocess: bool = True, fx: float = 202.5,
                 engine: str | None = None, spec_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """后台线程载 LAS2 并按 `shape = (H, W)` 跑一帧空图，把 torch.compile 的开销挪到启动时。

    编译缓存按**补到 32 倍数后的**尺寸命中（606 → 608），所以 shape 只要落在同一档就有效。
    预热和会话首帧共用同一个适配器（工厂单例 + 适配器锁）：会话在预热中途开起来，
    首帧就在锁上等预热编完，不会编两次。重复调用直接返回当前状态；只有 failed 会重试。

    子进程模式下预热跑在**子进程里**（PHASE 从 loading 走到 compiling/ready），主进程只轮询。
    """
    h, w = int(shape[0]), int(shape[1])
    with _warm_lock:
        if _warm["state"] not in ("idle", "failed"):
            return _warm_view()
        _warm.clear()
        _warm.update(state="loading", shape=[h, w], error=None, started=time.perf_counter(),
                     subprocess=bool(subprocess))
        first = _warm_view()
    threading.Thread(target=_prewarm_run, name="las2-prewarm", daemon=True,
                     args=(h, w, bool(subprocess), float(fx), engine, spec_extra)).start()
    return first


def _prewarm_set(**kv: Any) -> None:
    with _warm_lock:
        _warm.update(kv)


def _prewarm_run(h: int, w: int, subprocess: bool, fx: float, engine: str | None,
                 spec_extra: dict[str, Any] | None) -> None:
    try:
        m = make_stereo_backend("las2", fx=fx, subprocess=subprocess, shape=(h, w),
                                engine=engine, spec_extra=spec_extra)
        _prewarm_set(state="compiling")
        z = np.zeros((h, w, 3), np.uint8)
        try:
            m.compute(z, z)
        except StereoBusy:
            # 子进程还在自己预热（PHASE=loading/compiling）：轮询它，别当失败。
            _poll_prewarm(m, h, w)
        d = m.describe()
        _prewarm_set(state="ready" if d.get("compiled") else "eager", error=d.get("compile_error"),
                     pid=d.get("pid"))
    except Exception as exc:  # noqa: BLE001 - 缺 torch / XPU / 权重：如实报，会话照常退回 SGBM
        _prewarm_set(state="failed", error=f"{type(exc).__name__}: {exc}"[:300])
    finally:
        _prewarm_set(done=time.perf_counter())


def _poll_prewarm(m: Any, h: int, w: int) -> None:
    """等子进程把 PHASE 走到终态；期间把 loading/compiling 同步进 `_warm` 给 status 看。"""
    while True:
        st = m.prewarm_state()
        state = st.get("state")
        if state in ("ready", "eager"):
            return
        if state in ("failed", "exited"):
            raise StereoWorkerError(st.get("error") or f"worker {state}")
        if state in ("loading", "compiling") and state != _warm.get("state"):
            _prewarm_set(state=state)
        time.sleep(0.05)


def wait_prewarm_thread(timeout_s: float = 10.0) -> bool:
    """等后台预热线程收尾。退出路径上先等它，免得它在解释器关闭时还在起子进程。"""
    for th in threading.enumerate():
        if th.name == "las2-prewarm" and th.is_alive():
            th.join(timeout=timeout_s)
            return not th.is_alive()
    return True


def _warm_view() -> dict[str, Any]:
    out = dict(_warm)
    t0, t1 = out.pop("started", None), out.pop("done", None)
    out["elapsed_s"] = None if t0 is None else round((t1 if t1 is not None else time.perf_counter()) - t0, 1)
    return out


def las2_prewarm_status() -> dict[str, Any]:
    """`{"state", "shape", "error", "elapsed_s", "subprocess", "pid"}`；elapsed_s 跑时是已用时，跑完是总耗时。"""
    with _warm_lock:
        out = _warm_view()
    sub = _sub
    if sub is not None and out.get("pid") is None:
        out["pid"] = sub.pid
    return out


def las2_worker_status() -> dict[str, Any] | None:
    """子进程的现状；从没起过返回 None。status 里靠它看「活没活 / 编没编成 / 累计帧数」。"""
    sub = _sub
    if sub is None:
        return None
    if sub._closed:
        # 已经关掉的子进程不算"现状"，否则 status 会一直挂着一条 exited 记录。
        return None
    return sub.worker_status()


def close_las2() -> None:
    """关 LAS2 子进程并释放共享内存/信号量。幂等；也注册在 atexit 上。

    先竖闸门再抢 _sub_lock：抢锁意味着「如果有 spawn 正在跑，等它跑完」，
    拿到锁时它建出来的 adapter 一定已经在 _sub 里，能一起关掉。
    """
    global _sub
    _closing.set()
    try:
        with _sub_lock:
            sub, _sub = _sub, None
        if sub is not None:
            try:
                sub.close()
            except Exception:  # noqa: BLE001 - 退出路径上不能抛
                pass
    finally:
        # 闸门只挡「close 正在进行」这段时间，不永久关死：关闭完成后允许再起（测试用例之间、
        # 以及运维手动重启都会用到）。真正禁止再起的是解释器退出，那时也没人再调了。
        _closing.clear()


atexit.register(close_las2)
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

class _LAS2SubprocessAdapter:
    """LAS2 独立进程适配器：主进程只负责搬图和等信号，torch 全在子进程里。

    `compute` 的三条出口严格区分（调用方按类型分流，见 nav_online._stereo_worker_loop）：
    * 正常 → float32 (H, W)；
    * `StereoBusy` → 子进程还在载入/编译，或这一帧没在 `timeout_s` 内回来。**丢帧**；
    * `StereoWorkerError` → 进程起不来/崩了。本会话退回 SGBM。

    ⚠️ **迟到帧不能冒充下一帧**：子进程算完一帧就 release 一次 done，父进程超时后那个令牌
    还留在信号量上。所以每次 compute 都给 `SEQ_REQ` 发一个新序号，拿到 done 后要核对
    `SEQ_DONE == seq`，不等就继续等剩下的时间。少了这一步，超时那帧的视差会被当成下一帧
    的结果返回 —— 深度错一整帧，比丢帧危险得多。
    """

    __slots__ = ("shape_hw", "fx", "proc", "shm_left", "shm_right", "shm_disp", "sem_ready",
                 "sem_done", "timeout_s", "engine", "_ctrl", "_disp", "_left", "_right",
                 "_err_thread", "_log", "_closed", "_lock", "timeouts", "frames", "last_timing",
                 "job")

    def __init__(self, shape_hw: tuple[int, int], fx: float, proc: Any, shm_left: Any,
                 shm_right: Any, shm_disp: Any, sem_ready: NamedSemaphore, sem_done: NamedSemaphore,
                 timeout_s: float, engine: str, job: Any = None) -> None:
        h, w = int(shape_hw[0]), int(shape_hw[1])
        self.shape_hw, self.fx = (h, w), float(fx)
        self.proc, self.shm_left, self.shm_right, self.shm_disp = proc, shm_left, shm_right, shm_disp
        self.sem_ready, self.sem_done = sem_ready, sem_done
        self.timeout_s, self.engine = float(timeout_s), engine
        # Job Object 句柄：job 内所有进程在句柄关闭时被内核收掉（含 inductor 编译池）。
        # 本进程被强杀时句柄随之关闭 —— 正是我们要的效果。
        self.job = job
        self._ctrl = control(shm_disp.buf)
        self._disp = disparity(shm_disp.buf, h, w)
        self._left = np.ndarray((h, w, 3), np.uint8, buffer=shm_left.buf)
        self._right = np.ndarray((h, w, 3), np.uint8, buffer=shm_right.buf)
        self._log: collections.deque[str] = collections.deque(maxlen=40)
        self._closed = False
        self._lock = threading.Lock()
        self.timeouts = 0
        self.frames = 0
        self.last_timing: dict[str, float] | None = None
        self._err_thread = threading.Thread(target=self._drain_stderr, name="las2-worker-stderr",
                                            daemon=True)
        self._err_thread.start()

    # -- 子进程 stderr：调试时最有用的一路，转进主进程日志/status --
    def _drain_stderr(self) -> None:
        stream = getattr(self.proc, "stderr", None)
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self._log.append(line)
        except (OSError, ValueError):
            return

    # -- 状态 --
    @property
    def pid(self) -> int:
        """父进程 Popen 的 pid（= venv 启动器包装层）。真正在算的进程见 real_pid。"""
        return int(self.proc.pid)

    @property
    def real_pid(self) -> int:
        """真正跑推理那个解释器的 pid（包装层会再起一个）。还没握手时回落 Popen 的 pid。"""
        try:
            got = int(self._ctrl[WORKER_PID])
        except (TypeError, ValueError, IndexError):
            got = 0
        return got if got > 0 else self.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _phase(self) -> float:
        try:
            return float(self._ctrl[PHASE])
        except Exception:  # noqa: BLE001 - 共享内存已关
            return PHASE_FAILED

    def prewarm_state(self) -> dict[str, Any]:
        """`{"state", "error"}`，state 与 `las2_prewarm` 的状态机同名。"""
        if self._closed:
            return {"state": "exited", "error": "adapter closed"}
        phase = self._phase()
        if phase == PHASE_READY:
            return {"state": "ready", "error": None}
        if phase == PHASE_EAGER:
            try:
                err = read_error(self.shm_disp.buf)
            except (TypeError, ValueError):
                err = ""
            return {"state": "eager", "error": err or None}
        if phase == PHASE_FAILED:
            try:
                err = read_error(self.shm_disp.buf)
            except (TypeError, ValueError):
                err = ""
            return {"state": "failed", "error": err or "worker failed"}
        if phase == PHASE_EXITED:
            return {"state": "exited", "error": f"worker exited (rc={self.proc.poll()})"}
        if phase == PHASE_COMPILING:
            return {"state": "compiling", "error": None}
        # loading（或子进程还没来得及写 PHASE）
        if not self.alive():
            code = self.proc.poll()
            return {"state": "failed", "error": f"worker exited during load (rc={code})"}
        return {"state": "loading", "error": None}

    def worker_status(self) -> dict[str, Any]:
        if self._closed:
            return {"alive": False, "pid": self.pid, "frames": self.frames, "timeouts": self.timeouts,
                    "last_ms": None, "shape": list(self.shape_hw), "state": "exited",
                    "error": "adapter closed", "engine": self.engine, "log": list(self._log)[-8:]}
        try:
            frames = int(self._ctrl[FRAMES])
        except Exception:  # noqa: BLE001 - 共享内存已关
            frames = self.frames
        st = self.prewarm_state()
        try:
            last_ms = float(self._ctrl[LAST_MS]) or None
        except Exception:  # noqa: BLE001
            last_ms = None
        return {"alive": self.alive(), "pid": self.pid, "frames": max(frames, self.frames),
                "timeouts": self.timeouts, "last_ms": last_ms, "shape": list(self.shape_hw),
                "state": st["state"], "error": st["error"], "engine": self.engine,
                "log": list(self._log)[-8:]}

    def describe(self) -> dict[str, Any]:
        st = self.prewarm_state()
        compiled = True if st["state"] == "ready" else (False if st["state"] == "eager" else None)
        return {"kind": "las2", "subprocess": True, "alive": self.alive(), "compiled": compiled,
                "compile_error": st["error"], "pid": self.pid, "frames": self.frames,
                "timeouts": self.timeouts, "last_ms": self.last_timing and self.last_timing.get("launch"),
                "engine": self.engine}

    # -- 一帧 --
    def compute(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        with self._lock:
            return self._compute(left_rgb, right_rgb)

    def _compute(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        if self._closed:
            raise StereoWorkerError("LAS2 worker 已关闭")
        if not self.alive():
            raise StereoWorkerError(
                f"LAS2 subprocess exited with code {self.proc.poll()}（{self._log[-1] if self._log else '无输出'}）")
        st = self.prewarm_state()
        if st["state"] in ("failed", "exited"):
            raise StereoWorkerError(f"LAS2 worker {st['state']}：{st['error']}")
        if st["state"] in ("loading", "compiling"):
            raise StereoBusy(f"LAS2 worker {st['state']}")

        h, w = self.shape_hw
        if left_rgb.shape[:2] != (h, w):
            raise StereoWorkerError(f"LAS2 worker 尺寸错：共享内存 {h}x{w}，收到 {left_rgb.shape[:2]}"
                                    "（shape 变了要 close 重建，见 make_stereo_backend）")
        self._left[:] = left_rgb
        self._right[:] = right_rgb
        seq = int(self._ctrl[SEQ_REQ]) + 1
        self._ctrl[SEQ_REQ] = float(seq)          # 先写序号再 release：子进程读到的一定是本帧
        self.sem_ready.release()

        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.sem_done.acquire(remaining):
                self.timeouts += 1
                # 进程在等的时候死了：报「死了」，不是「慢」。
                if not self.alive():
                    raise StereoWorkerError(
                        f"LAS2 subprocess exited with code {self.proc.poll()}"
                        f"（{self._log[-1] if self._log else '无输出'}）")
                raise StereoBusy(f"LAS2 subprocess timeout（{self.timeout_s * 1e3:.0f} ms）")
            # 迟到的 done（上一帧超时后才算完）：序号对不上，继续等本帧。
            if int(self._ctrl[SEQ_DONE]) >= seq:
                break
        self.frames += 1
        self.last_timing = {"upload": float(self._ctrl[UPLOAD_MS]), "launch": float(self._ctrl[LAUNCH_MS]),
                            "sync": float(self._ctrl[SYNC_MS])}
        return self._disp.copy()

    # -- 收尾 --
    def close(self) -> None:
        if self._closed:
            return
        # ⚠️ 必须先取状态再置 _closed：prewarm_state() 见到 _closed 会立刻报 "exited"，
        # 于是「编译中」的分支永远进不去，然后死等 5 s（2026-10-26 实测踩过）。
        compiling = self.prewarm_state()["state"] in ("loading", "compiling")
        self._closed = True
        global _sub
        if _sub is self:
            _sub = None
        proc = self.proc
        try:
            if proc.poll() is None:
                # 先礼后兵：CTRL=1 + release ready，子进程在 ready 上醒来就自己退（能写 PHASE_EXITED）。
                # ⚠️ 编译中不走这一套：那时子进程的主循环还没开始，CTRL 没人看。terminate() 会打断
                # inductor 的编译，但半成品缓存是安全的（torch 按 key 校验），下次启动本来也要重预热。
                if not compiling:
                    with contextlib.suppress(Exception):
                        self._ctrl[CTRL] = 1.0
                        self.sem_ready.release()
                try:
                    proc.wait(timeout=1.0 if compiling else 5.0)
                except _subprocess.TimeoutExpired:
                    pass
            # ⚠️ 不管上面等没等到，都要按**真实** worker 的 pid 收整棵树：
            #   ① .venv_las/Scripts/python.exe 是 venv 启动器包装层，terminate() 只杀它，
            #      真正在算的解释器（WORKER_PID）会留下来继续占 XPU 并抢信号量；
            #   ② 编译中时子进程不看 CTRL，上面那 1 s 必然等不到。
            # taskkill /T 连包装层带真实解释器一起收，覆盖这两种情况。
            _kill_tree(self.real_pid, wait_s=5.0)
            with contextlib.suppress(Exception):
                proc.wait(timeout=3.0)
        finally:
            if proc.stderr is not None:
                with contextlib.suppress(Exception):
                    proc.stderr.close()
            if self.job is not None:
                # 关 job 句柄 = 内核杀光 job 内全部进程（KILL_ON_JOB_CLOSE）。
                # 比 taskkill /T 更彻底：连 multiprocessing 起的孙进程一起收。
                with contextlib.suppress(Exception):
                    import ctypes
                    ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self.job)
                self.job = None
            for sem in (self.sem_ready, self.sem_done):
                with contextlib.suppress(Exception):
                    sem.close()
            # ⚠️ 顺序有讲究：self._ctrl/_disp/_left/_right 是 numpy 视图，它们持有底层 mmap 的
            # memoryview ⇒ 只要这些数组还活着，SharedMemory.close() 就只是减少了引用计数，
            # Windows 上的命名对象**不会消失**（测试里表现为 close() 之后还能按名字打开）。
            # 必须先丢引用，再 close，最后 unlink（POSIX 上 unlink 才真正生效）。
            self._ctrl = None
            self._disp = None
            self._left = None
            self._right = None
            for shm in (self.shm_disp, self.shm_left, self.shm_right):
                with contextlib.suppress(Exception):
                    shm.close()
            for shm in (self.shm_disp, self.shm_left, self.shm_right):
                with contextlib.suppress(Exception):
                    shm.unlink()


def _kill_tree(pid: int, *, wait_s: float = 5.0) -> None:
    """杀掉 pid 及其所有子孙进程。Windows 上没有进程组，靠 taskkill /T /F。

    为什么需要：``.venv_las/Scripts/python.exe`` 是 venv 启动器包装层，它再起真实解释器。
    ``Popen.terminate()`` 只杀包装层，真正在算的那个（= WORKER_PID）会变孤儿并继续占着 XPU。
    """
    if pid <= 0:
        return
    if os.name != "nt":
        import signal as _signal
        try:
            os.kill(pid, _signal.SIGTERM)
        except OSError:
            pass
        return
    try:
        _subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                        capture_output=True, timeout=max(1.0, wait_s))
    except Exception:  # noqa: BLE001 - 收尸失败不能挡住退出
        pass


def _job_for_pid(pid: int) -> Any:
    """把 pid 放进一个「句柄关闭就杀光」的 Job Object，返回句柄（失败返回 None）。

    为什么需要：父进程被 /F 强杀时，worker 和它的 inductor 编译池收不到任何信号。
    worker 自己会靠探测父 pid 退出，但**编译池进程不会** —— 它们是 multiprocessing 起的
    孙进程，父被强杀后只会一直挂着（2026-10-26 实测：一次强杀留下 16 个池进程过夜）。

    内核级兜底：KILL_ON_JOB_CLOSE ⇒ 父进程一死，它的 job 句柄表条目随之消失，最后一个句柄
    关闭时内核把 job 里所有进程（含孙进程）一起收掉。不依赖任何用户态代码。
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC), ("IoInfo", _IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _EXT()
    info.BasicLimitInformation.LimitFlags = 0x00002000     # KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(job)
        return None
    ph = k32.OpenProcess(0x1F0FFF, False, int(pid))
    if not ph:
        k32.CloseHandle(job)
        return None
    try:
        if not k32.AssignProcessToJobObject(job, ph):
            k32.CloseHandle(job)
            return None
    finally:
        k32.CloseHandle(ph)
    return job

def _las2_subprocess(shape_hw: tuple[int, int], *, fx: float, engine: str | None,
                     timeout_s: float, spec_extra: dict[str, Any] | None) -> _LAS2SubprocessAdapter:
    """起（或复用）LAS2 子进程。形状/引擎变了就重建 —— 共享内存尺寸绑死在形状上。"""
    global _sub
    h, w = int(shape_hw[0]), int(shape_hw[1])
    engine = (engine or os.environ.get("NEKO_LAS_ENGINE") or "las2").strip().lower()
    if _closing.is_set():
        # 后端正在退出：别再造新进程（造出来没人关，见 _closing 的注释）。
        raise StereoWorkerError("LAS2 已关闭（后端正在退出）")
    with _sub_lock:
        cur = _sub
        if cur is not None and cur.shape_hw == (h, w) and cur.engine == engine and cur.alive():
            cur.timeout_s = float(timeout_s)
            return cur
        if cur is not None:
            with contextlib.suppress(Exception):
                cur.close()
            _sub = None
        adapter = _spawn_las2(shape_hw=(h, w), fx=fx, engine=engine, timeout_s=timeout_s,
                              spec_extra=spec_extra)
        _sub = adapter
        return adapter


def _create_shm(name: str, size: int) -> Any:
    """建一块命名共享内存；同名对象还在（上一次硬崩留下）就先打开扔掉它再重试一次。

    Windows 上命名对象在最后一个句柄关闭时消失，所以「还在」意味着确实有活着的持有者 ——
    那多半是上一次后端没走 close()。宁可清掉重来，也不要让整个 LAS2 因为名字冲突起不来。
    """
    from multiprocessing import shared_memory
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        pass
    try:
        stale = shared_memory.SharedMemory(name=name)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass
    return shared_memory.SharedMemory(name=name, create=True, size=size)


def _spawn_las2(*, shape_hw: tuple[int, int], fx: float, engine: str, timeout_s: float,
                spec_extra: dict[str, Any] | None) -> _LAS2SubprocessAdapter:
    """建共享内存/信号量 → 起子进程 → 等它把 PHASE 写成 loading（= 协议通了）才返回。"""
    h, w = shape_hw
    base = "neko_las2_" + uuid.uuid4().hex[:12]
    names = {"left": f"{base}_left", "right": f"{base}_right", "disp": f"{base}_disp",
             "sem_ready": f"{base}_ready", "sem_done": f"{base}_done"}
    shm_left = shm_right = shm_disp = None
    sem_ready = sem_done = None
    proc = None
    try:
        from multiprocessing import shared_memory
        shm_left = _create_shm(names["left"], h * w * 3)
        shm_right = _create_shm(names["right"], h * w * 3)
        shm_disp = _create_shm(names["disp"], CTRL_BYTES + h * w * 4)
        # 控制块先清干净：PHASE 落在 0（loading）以外的话父进程会读到上一个进程的残值。
        np.frombuffer(shm_disp.buf, np.float32, 32, 0)[:] = 0.0
        sem_ready = NamedSemaphore.create(names["sem_ready"])
        sem_done = NamedSemaphore.create(names["sem_done"])

        spec = {
            "left": names["left"], "right": names["right"], "disp": names["disp"],
            "sem_ready": names["sem_ready"], "sem_done": names["sem_done"],
            "h": h, "w": w, "parent_pid": os.getpid(), "fx": float(fx), "engine": engine,
            "delay_s": float(os.environ.get("NEKO_LAS_DELAY_S") or 0.0),
            "exit_after": int(os.environ.get("NEKO_LAS_EXIT_AFTER") or 0),
        }
        # spec_extra 优先级最高（测试用：直通 delay_s / exit_after）。
        if spec_extra:
            spec.update({k: v for k, v in spec_extra.items() if k in spec or k != "shape_bytes"})

        python = _las2_python()
        creationflags = getattr(_subprocess, "CREATE_NO_WINDOW", 0)
        proc = _subprocess.Popen(
            [python, "-m", "backend.stereo_worker", json.dumps(spec, ensure_ascii=False)],
            cwd=PROJECT_DIR, env=_las2_env(), stdin=_subprocess.DEVNULL,
            stdout=_subprocess.DEVNULL, stderr=_subprocess.PIPE, creationflags=creationflags,
        )
        adapter = _LAS2SubprocessAdapter(shape_hw=(h, w), fx=fx, proc=proc, shm_left=shm_left,
                                         shm_right=shm_right, shm_disp=shm_disp, sem_ready=sem_ready,
                                         sem_done=sem_done, timeout_s=timeout_s, engine=engine,
                                         job=_job_for_pid(int(proc.pid)))
        # 握手分两段，缺一不可：
        #   ① 等它把 H/协议版本写进控制块（= 共享内存连上了）；② 等它把引擎**载起来**
        #   （PHASE 到 COMPILING/EAGER/READY，或直接 FAILED/退出）。
        # 为什么②不能省：缺 torch / 解释器不对时子进程也是先连上共享内存、再在 import 里死。
        # 只做①的话 make_stereo_backend 会「成功」返回一个注定失败的适配器，调用方要等到
        # 第一帧才知道该退回 SGBM —— 而退回这件事必须在建 matcher 时就决定（见 nav_online）。
        # ②等到的是「模型载入完成」而不是「编译完成」：编译要 70–330 s，那是预热的事，
        # 主进程绝不能在 spawn 里等它（否则后端启动就卡死）。
        deadline = time.monotonic() + SPAWN_TIMEOUT_S
        while time.monotonic() < deadline:
            if not adapter.alive():
                raise StereoWorkerError(
                    f"LAS2 worker 起不来（{python}，rc={proc.returncode}）："
                    f"{' | '.join(list(adapter._log)[-3:]) or '无输出'}")
            phase = adapter._phase()
            if phase in (PHASE_FAILED, PHASE_COMPILING, PHASE_READY, PHASE_EAGER) and \
                    float(adapter._ctrl[H_IX]) == float(h):
                break
            time.sleep(0.02)
        else:
            raise StereoWorkerError(f"LAS2 worker {SPAWN_TIMEOUT_S}s 内没完成握手（{adapter.prewarm_state()}）")
        # 校验握手：子进程必须回报同一份尺寸与协议版本。
        if float(adapter._ctrl[H_IX]) == float(h) and int(adapter._ctrl[VERSION_IX]) != int(PROTOCOL_VERSION):
            raise StereoWorkerError(f"LAS2 worker 协议版本不符：{adapter._ctrl[VERSION_IX]}"
                                    f" != {PROTOCOL_VERSION}")
        st = adapter.prewarm_state()
        if st["state"] in ("failed", "exited"):
            raise StereoWorkerError(f"LAS2 worker 载入失败：{st['error']}")
        return adapter
    except BaseException:
        # 起不来就把已经建的对象全部收掉，别留孤儿进程/命名对象。
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=2.0)
        for sem in (sem_done, sem_ready):
            with contextlib.suppress(Exception):
                sem.close()
        for shm in (shm_disp, shm_left, shm_right):
            if shm is not None:
                with contextlib.suppress(Exception):
                    shm.close()
                with contextlib.suppress(Exception):
                    shm.unlink()
        raise