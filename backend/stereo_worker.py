# -*- coding: utf-8 -*-
"""LAS2 独立推理进程：从共享内存读左右图，算视差写回同一块共享内存。

为什么要独立进程
----------------
LAS2 的 torch.compile 每次进 Python 派发几百次 GIL 释放/重取。2026-10-24 实测（Docs/
双目解耦线程与学习型立体取舍实测（2026-10-24）.md 订正 B.2）：只要别的线程**连续**跑纯
Python，每次重取都要等满 Windows 定时器量化后的超时（15.625 ms），一帧约 730 次交接 ⇒
最坏被拖长两个数量级。主后端里建图/回环线程就有这种长段纯 Python。独立进程 = 独立
解释器 = 独立 GIL，这条链彻底断开。

进程边界 = 协议边界
-------------------
三块共享内存 + 两个命名信号量（**父进程创建、子进程 open**，见 `stereo_backend.NamedSemaphore` + `_LAS2SubprocessAdapter`）：

* `left` / `right`：`H*W*3` uint8，RGB 原图；
* `disp`：`384 + H*W*4` 字节。前 384 = `float32[32]` 控制块 + 256 B UTF-8 错误文本，
  之后是 `H*W` float32 视差。控制块布局见下；
* `ready`：父进程写完图、`SEQ_REQ += 1` 后 release；
* `done`：子进程写完视差、`SEQ_DONE += 1` 后 release。

控制块（`float32[32]`，两个进程都按同一张表读写）：

===== ============ =========================================================
下标   名字         含义
===== ============ =========================================================
0     PHASE        -2 已退出 / -1 失败 / 0 载入中 / 1 编译中 / 2 ready / 3 eager / 4 正在退出
1     SEQ_REQ      父进程写图次数（父写）
2     SEQ_DONE     子进程完成次数（子写）。**配对判据**：`SEQ_DONE < SEQ_REQ` 就是还没好
3     FRAMES       子进程累计算完的帧数
4     LAST_MS      上一帧推理墙钟 ms
5     HEARTBEAT    子进程每 100 ms 自增（父进程看它判断「卡住」而不是「忙」）
6/7   H / W        尺寸回读（父进程拿错尺寸就是协议错，立刻报）
8     VERSION      协议版本，必须等于 `PROTOCOL_VERSION`
10–12 UPLOAD/     numpy→XPU+pad / 前向返回 / .cpu() 同步 三段 ms（诊断用）
      LAUNCH/
      SYNC
16    CTRL         父进程写 1.0 = 请求退出
===== ============ =========================================================

退出
----
两条路，都不依赖 `terminate()` 打断：
1. `CTRL = 1.0` + release ready：子进程在 ready 上醒来后看到 CTRL 就置 PHASE=-2 退出；
2. 父进程 `terminate()`（`close()` 里先走 1，超时才用它）。

`terminate()` 会打断 `torch.compile` 的编译，留下的半成品缓存是安全的（torch 自己按 key
校验），但进程直接死掉不会写 PHASE。

⚠️ 这个文件**必须能在没有 torch 的解释器里 import**：主后端（`.venv`，无 torch）也会
import 本模块拿协议常量，所以 torch 只在 `_load_las2` 里惰性导入。
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from typing import Any

import numpy as np

# ---- 协议常量（父进程 stereo_backend 也 import 这张表，改这里就是改协议）----------
PROTOCOL_VERSION = 1.0
CTRL_FLOATS = 32                  # float32[32]
ERR_BYTES = 256                   # 错误文本
CTRL_BYTES = CTRL_FLOATS * 4 + ERR_BYTES      # 384：控制块总长（视差数据的起点）

PHASE, SEQ_REQ, SEQ_DONE, FRAMES, LAST_MS, HEARTBEAT = 0, 1, 2, 3, 4, 5
H_IX, W_IX, VERSION_IX = 6, 7, 8
# 真实解释器的 pid。⚠️ 不等于父进程 Popen 的 pid：`.venv_las/Scripts/python.exe` 是 venv 的
# **启动器包装层**（255 KB 的 exe），它会再起一个真实解释器（106 KB 的 F:/py314/python.exe），
# 两个进程是父子关系、argv 完全相同。terminate() 只会杀掉包装层，真正在算的那个变孤儿。
# 所以 worker 一进来就把自己的 pid 写进控制块，父进程按这个 pid 收尸。
WORKER_PID = 9
UPLOAD_MS, LAUNCH_MS, SYNC_MS, CTRL = 10, 11, 12, 16
ERR_OFFSET = CTRL_FLOATS * 4      # 128

PHASE_EXITED, PHASE_FAILED = -2.0, -1.0
PHASE_LOADING, PHASE_COMPILING, PHASE_READY, PHASE_EAGER, PHASE_STOPPING = 0.0, 1.0, 2.0, 3.0, 4.0

# 子进程心跳间隔。父进程用它区分「在编译」和「死了」。
HEARTBEAT_S = 0.1


def control(view: memoryview) -> np.ndarray:
    """`disp` 共享内存开头 `CTRL_BYTES` 字节的 float32 视图（32 个）。"""
    return np.frombuffer(view, np.float32, CTRL_FLOATS, 0)


def disparity(view: memoryview, h: int, w: int) -> np.ndarray:
    """`disp` 共享内存里视差部分的 (h, w) float32 视图。"""
    return np.frombuffer(view, np.float32, h * w, CTRL_BYTES).reshape(h, w)


def _set_error(view: memoryview, text: str) -> None:
    """把失败原因写进共享内存（父进程读得到，不用去管道里捞）。"""
    raw = text.encode("utf-8", "replace")[:ERR_BYTES - 1]
    view[ERR_OFFSET:ERR_OFFSET + len(raw)] = raw
    view[ERR_OFFSET + len(raw):CTRL_BYTES] = b"\x00" * (ERR_BYTES - len(raw))


def read_error(view: memoryview) -> str:
    raw = bytes(view[ERR_OFFSET:CTRL_BYTES])
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


# ---- 引擎 ------------------------------------------------------------------------
class EchoEngine:
    """测试用假引擎：`disp = left[..., 0]`。

    为什么要有它：进程边界的每条断言（超时隔离、崩溃、清理）都不该依赖 4 GB 的 torch。
    echo 的输出逐像素可验，且与 right 无关 ⇒ 测试能用「结果是不是这一帧的 left」直接
    判出「迟到帧有没有冒充下一帧」。
    """

    compiled = True
    compile_error: str | None = None

    def __init__(self, delay_s: float = 0.0) -> None:
        # 慢引擎：warmup 与每帧都先 sleep。用来把 worker 停在不同相位上做测试
        # （尤其是 COMPILING —— 那个窗口里 worker 主循环还没开始，CTRL 没人看）。
        self.delay_s = max(0.0, float(delay_s))

    def compute(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if self.delay_s:
            time.sleep(self.delay_s)
        t0 = time.perf_counter()
        out = np.ascontiguousarray(left[..., 0]).astype(np.float32)
        ms = (time.perf_counter() - t0) * 1e3
        return out, {"upload": 0.0, "launch": ms, "sync": 0.0}


class _LAS2Engine:
    """真 LAS2：包一层 `compute`，顺手把 `last_timing` 带出来。"""

    def __init__(self, matcher: Any) -> None:
        self._m = matcher

    @property
    def compiled(self) -> bool | None:
        return self._m.describe().get("compiled")

    @property
    def compile_error(self) -> str | None:
        return self._m.describe().get("compile_error")

    def compute(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        out = self._m.compute(left, right)
        return out, dict(self._m.last_timing or {})


def _load_las2(fx: float) -> Any:
    """进程内 LAS2 适配器。torch 相关的一切都在这条路径上（惰性导入）。"""
    from .stereo_backend import make_stereo_backend

    return _LAS2Engine(make_stereo_backend("las2", fx=fx, subprocess=False))


# ---- 主循环 ----------------------------------------------------------------------
def _open_bus(spec: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    """连上父进程建好的对象。

    共享内存用 `multiprocessing.shared_memory`（跨 3.12/3.14 兼容）；命名信号量不能用
    `multiprocessing.Semaphore`（Windows 上不支持取名），也不能用内部
    `_multiprocessing.SemLock`（实测跨进程时子进程永远读到 0）。用 Win32 API，见
    `stereo_backend.NamedSemaphore`。
    """
    from multiprocessing import shared_memory

    from .stereo_backend import NamedSemaphore

    shm_left = shared_memory.SharedMemory(name=spec["left"])
    shm_right = shared_memory.SharedMemory(name=spec["right"])
    shm_disp = shared_memory.SharedMemory(name=spec["disp"])
    sem_ready = NamedSemaphore.open(spec["sem_ready"])
    sem_done = NamedSemaphore.open(spec["sem_done"])
    return shm_left, shm_right, shm_disp, sem_ready, sem_done


class _ParentGone(RuntimeError):
    """父进程没了：别再算，也别留着占 XPU。"""


def run(spec: dict[str, Any]) -> int:
    """解析 spec → 建引擎 → 报 PHASE → 循环「收帧 / 推理 / 回写」。"""
    h, w = int(spec["h"]), int(spec["w"])
    fx = float(spec.get("fx", 202.5))
    engine_name = str(spec.get("engine") or "las2")
    delay_s = float(spec.get("delay_s") or 0.0)
    exit_after = int(spec.get("exit_after") or 0)
    parent_pid = int(spec.get("parent_pid") or 0)

    try:
        shm_left, shm_right, shm_disp, sem_ready, sem_done = _open_bus(spec)
    except Exception:
        print(f"stereo_worker: 连不上父进程的共享对象：{traceback.format_exc()}", file=sys.stderr, flush=True)
        return 2

    ctrl = control(shm_disp.buf)
    disp = disparity(shm_disp.buf, h, w)
    left = np.ndarray((h, w, 3), np.uint8, buffer=shm_left.buf)
    right = np.ndarray((h, w, 3), np.uint8, buffer=shm_right.buf)

    # 尺寸/版本先自检：父进程按错尺寸开共享内存是**协议错**，宁可立刻报死也不要算错。
    ctrl[H_IX], ctrl[W_IX], ctrl[VERSION_IX] = h, w, PROTOCOL_VERSION
    ctrl[WORKER_PID] = float(os.getpid())     # 父进程按这个 pid 杀，见 WORKER_PID 的注释
    ctrl[HEARTBEAT] = 1.0        # 进程已经起来了：从 1 开始，之后按 HEARTBEAT_S 递增
    ctrl[PHASE] = PHASE_LOADING

    def alive() -> bool:
        return parent_pid <= 0 or _parent_alive(parent_pid)

    try:
        if engine_name == "echo":
            engine: Any = EchoEngine(delay_s)
        else:
            engine = _load_las2(fx)

        # 预热：跑一帧全零图。torch.compile 的 70–330 s 冷编译就发生在这里，父进程
        # 靠轮询 PHASE 看进度（而不是等 compute 超时）。
        ctrl[PHASE] = PHASE_COMPILING
        _warmup(engine, h, w, alive, ctrl)
        if engine.compiled is False:
            # 编译失败退回 eager（慢一倍但结果相同）。原因要留给父进程：否则 status 里只有
            # 一个 "eager"，没人知道是缺 cl、缺 PYTHONUTF8 还是 inductor 炸了。
            _set_error(shm_disp.buf, f"compile fallback: {engine.compile_error or 'unknown'}"[:ERR_BYTES - 1])
            ctrl[PHASE] = PHASE_EAGER
        else:
            ctrl[PHASE] = PHASE_READY
        print(f"stereo_worker: ready (engine={engine_name}, {h}x{w}, pid={os.getpid()})",
              file=sys.stderr, flush=True)
    except _ParentGone:
        return 0
    except BaseException as exc:  # noqa: BLE001 - 载不起来要如实报给父进程
        ctrl[PHASE] = PHASE_FAILED
        _set_error(shm_disp.buf, f"{type(exc).__name__}: {exc}"[:ERR_BYTES - 1])
        print(f"stereo_worker: 预热失败：{traceback.format_exc()}", file=sys.stderr, flush=True)
        return 1

    frames = 0
    last_beat = time.perf_counter()
    # terminate() 兜底：被 SIGTERM 打断时先把 PHASE 标成「已退出」再走，父进程看得到。
    with_signal = hasattr(signal, "SIGTERM")
    if with_signal:
        signal.signal(signal.SIGTERM, lambda *_: (_set_phase(ctrl, PHASE_EXITED), os._exit(0)))

    while True:
        # 用带超时的 acquire 而不是无限等：空闲期（父进程还没送帧）也要维持心跳，否则父
        # 进程没法把「在编译」和「已经卡死」分开 —— 这正是 HEARTBEAT 存在的理由。
        if not sem_ready.acquire(HEARTBEAT_S):
            last_beat = _beat(ctrl, last_beat)
            if not alive():
                break
            continue
        # 父进程要退出：CTRL 置 1 并 release ready（见模块文档「退出」）。
        if float(ctrl[CTRL]) != 0.0:
            break
        seq_req = int(ctrl[SEQ_REQ])
        if delay_s > 0:
            time.sleep(delay_s)
        try:
            out, timing = engine.compute(left, right)
        except _ParentGone:
            break
        except BaseException as exc:  # noqa: BLE001 - 单帧失败只记，不带走进程
            _set_error(shm_disp.buf, f"{type(exc).__name__}: {exc}"[:ERR_BYTES - 1])
            print(f"stereo_worker: 推理失败：{traceback.format_exc()}", file=sys.stderr, flush=True)
            # 仍要把 done 交回去，否则父进程会白等满 timeout。
            ctrl[SEQ_DONE] = float(seq_req)
            sem_done.release()
            continue
        # 每帧都查一次父进程：父进程被强杀（任务管理器 / job_kill / 崩溃）时，
        # 子进程收不到任何信号，只能靠主动探测。
        if not alive():
            break
        disp[:] = out
        frames += 1
        # 先写结果与序号、再 release：父进程看到 done 时数据已经就绪。
        ctrl[SEQ_DONE] = float(seq_req)
        ctrl[FRAMES] = float(frames)
        ctrl[LAST_MS] = float(timing.get("launch", 0.0) + timing.get("sync", 0.0))
        ctrl[UPLOAD_MS] = float(timing.get("upload", 0.0))
        ctrl[LAUNCH_MS] = float(timing.get("launch", 0.0))
        ctrl[SYNC_MS] = float(timing.get("sync", 0.0))
        sem_done.release()
        if exit_after and frames >= exit_after:
            # 测试用：模拟原生崩溃/被杀（不走清理路径，就是要看父进程能不能认出来）。
            os._exit(9)

    _set_phase(ctrl, PHASE_EXITED)
    # numpy 视图持有 mmap 的 memoryview ⇒ 不先丢掉引用，SharedMemory.close() 会抛
    # BufferError（Windows 上还表现为命名对象不消失，父进程 close 后再也清不掉）。
    del ctrl, disp, left, right
    for shm in (shm_left, shm_right, shm_disp):
        try:
            shm.close()
        except BufferError:
            pass
    sem_ready.close()
    sem_done.close()
    return 0


def _beat(ctrl: np.ndarray, last_beat: float) -> float:
    """空闲心跳：到点就 +1，返回新的上次时刻。父进程看它区分「在忙」和「死了」。"""
    now = time.perf_counter()
    if now - last_beat >= HEARTBEAT_S:
        ctrl[HEARTBEAT] = float(ctrl[HEARTBEAT]) + 1.0
        return now
    return last_beat


def _set_phase(ctrl: np.ndarray, phase: float) -> None:
    try:
        ctrl[PHASE] = phase
    except Exception:  # noqa: BLE001 - 退出路径上共享内存可能已经没了
        pass


def _parent_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = wintypes.DWORD(0)
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259          # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def _warmup(engine: Any, h: int, w: int, alive: Any, ctrl: np.ndarray) -> None:
    """跑一帧全零图触发编译。编译期长，每 100 ms 摸一下父进程还在不在。"""
    import threading

    done = threading.Event()
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            z = np.zeros((h, w, 3), np.uint8)
            box["out"] = engine.compute(z, z)
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, name="las2-warmup", daemon=True).start()
    last_beat = time.perf_counter()
    while not done.wait(HEARTBEAT_S):
        # 编译期最长 5 min，父进程要靠心跳把「还在编」和「已经死了」分开。
        last_beat = _beat(ctrl, last_beat)
        if not alive():
            raise _ParentGone()
    if "err" in box:
        raise box["err"]


def main(argv: list[str] | None = None) -> int:
    # 防线（排除法，不是白名单）：只有当本模块**被 multiprocessing 当成 spawn 载体**时才退出。
    # 判据用 multiprocessing 自己的语义函数 `is_forking()`（= argv 里有 --multiprocessing-fork），
    # 不用字符串匹配 `-c "from multiprocessing..."`（那条在版本间更脆）。
    #
    # 为什么需要：TORCHINDUCTOR_WORKER_START=spawn 时 inductor 起编译 worker 走的是
    # `python -c "from multiprocessing.spawn import spawn_main; spawn_main(...)" --multiprocessing-fork`，
    # **不重放我们的 argv**（Python 3.14 multiprocessing/spawn.py:83-97）。所以正常路径根本不会命中。
    # 留着这道是为了挡住"有人把本模块当 spawn 载体"这种误用：那种进程会 open 同一份共享内存和
    # ready/done 信号量、跟真 worker 抢帧，把协议劈成两半 —— 安静退出比抢帧安全。
    # ⚠️ 不用 parent_process()：Windows 上所有进程都由 spawn 起来，parent_process() 恒非 None，
    #    那会把正常的 -m 启动也误杀。
    from multiprocessing.spawn import is_forking
    if is_forking(sys.argv):
        return 0
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m backend.stereo_worker '<spec-json>'", file=sys.stderr)
        return 2
    try:
        spec = json.loads(args[0])
    except json.JSONDecodeError as exc:
        print(f"stereo_worker: spec 不是合法 JSON：{exc}", file=sys.stderr)
        return 2
    return run(spec)


if __name__ == "__main__":
    sys.exit(main())