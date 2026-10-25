# -*- coding: utf-8 -*-
"""LAS2 独立推理进程（backend/stereo_worker.py）的进程边界单元测试。

为什么这一层必须真起进程
------------------------
LAS2 的 torch.compile 首编 70-330 s，只能放在独立进程里跑，否则位姿线程被卡住。
于是进程边界就成了**协议**边界：共享内存布局、命名信号量的 ready/done 配对、SEQ 序号、
崩溃与超时的区分。这些恰恰是 mock 掉的那一段 —— mock 了自己等于什么都没测。

本文件全程 engine="echo"：disp = left[..., 0].astype(float32)，逐像素可验，
**不需要 torch / XPU / 权重**，所以在 .venv（Py3.12，无 torch）里就能跑。

三条铁律，每条都有独立用例
--------------------------
1. 一次 compute 只配一次 ready/done；**上一帧迟到的 done 必须按 SEQ_DONE 丢弃**，
   否则超时那帧的视差会被当成下一帧的结果（TimeoutIsolationTest）。
2. 子进程 os._exit(9) => 下一帧 StereoWorkerError，不能退化成超时（CrashTest）。
3. close() 之后命名共享内存全部消失、子进程已退出，且可重复调用（CleanupTest）。
外加「空闲 3 s 不误杀」（IdleTest）—— watchdog 写错方向就会把健康进程杀掉。

⚠️ 命名对象（shm / 信号量）由**父进程**创建、子进程 open（Lead 定稿：Low 完整性级别是
.venv_las 的环境配置错误，正在修，不是设计约束）。因此所有清理断言都在父进程侧做：
close() 后再 open 同名对象必须失败。

⚠️ 每个用例拉起的子进程都在 addCleanup 里 close/kill，不留孤儿。

Windows 上的两个坑（都是实测踩出来的，改这块前先读）
----------------------------------------------------
1. **Popen.poll() 比内核退出晚 5-50 ms**：TerminateProcess 之后进程其实已经死了，但
   Popen.poll() 可能还是 None，而适配器的 alive() 读的就是 poll()。所以「杀进程之后立刻断言
   抛 StereoWorkerError」会偶发拿到竞态窗口里的假结果。RestartTest 先等 alive() 收敛再断言。
2. **子进程适配器是进程级单例**（stereo_backend._sub）：同形状同引擎会直接复用。所以
   test_stereo_backend.SubprocessBackendTest 的 setUp 必须先 close_las2()，否则上一个用例
   留下的活 worker 会让「起不来」用例根本走不到 spawn，断言恒假绿。

⚠️ delay_s / exit_after 直通 spec（spec_extra）。带这两个参数的 worker 会被**故意弄坏**
   （变慢 / 自杀），所以这类用例一律**不走 las2_prewarm**（预热自己会跑一帧），
   而是直接 make_stereo_backend 后轮询 adapter.prewarm_state() 等 ready。
"""
from __future__ import annotations

import contextlib
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from unittest import mock

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import stereo_backend

REPO = Path(__file__).resolve().parents[1]
WORKER_PY = REPO / "backend" / "stereo_worker.py"
_IS_WIN = sys.platform == "win32"

# 进程还没落地时整组跳过，且跳过原因可见（不静默变绿）。
_HAS_WORKER = WORKER_PY.is_file()
_MISSING_API = [n for n in ("StereoBusy", "StereoWorkerError", "las2_worker_status", "close_las2")
                if getattr(stereo_backend, n, None) is None]
_HAS_BACKEND = _HAS_WORKER and not _MISSING_API
_skip_no_worker = unittest.skipUnless(
    _HAS_BACKEND,
    f"LAS2 子进程后端尚未落地：worker={'有' if _HAS_WORKER else '缺'}，"
    f"stereo_backend 缺符号={_MISSING_API or '无'}")

# ---- 协议常量（照 Lead 定稿抄）-------------------------------------------------
CTRL_BYTES = 384                  # float32[32] 控制块 + 256 B UTF-8 错误文本
ERR_OFFSET = 128                  # 32 * 4
PHASE, SEQ_REQ, SEQ_DONE, FRAMES, LAST_MS, HEARTBEAT = 0, 1, 2, 3, 4, 5
H_IX, W_IX, VERSION_IX = 6, 7, 8
UPLOAD_MS, LAUNCH_MS, SYNC_MS, CTRL = 10, 11, 12, 16
PHASE_LOADING, PHASE_COMPILING, PHASE_READY, PHASE_EAGER, PHASE_STOPPING = 0, 1, 2, 3, 4
PHASE_FAILED, PHASE_EXITED = -1, -2
VERSION = 1.0


def _timing_ms(d: dict) -> float:
    """adapter.describe()/worker_status 的「上一帧毫秒」，两种命名都认。"""
    for key in ("last_ms", "last_ms_p50", "frame_ms"):
        if key in d and d[key] is not None:
            return float(d[key])
    raise AssertionError(f"describe() 里没有上一帧耗时字段：{sorted(d)}")


def _state_of(obj) -> str | None:
    """prewarm_state() 可能是 {"state": ...} 也可能直接是字符串。"""
    if isinstance(obj, dict):
        return obj.get("state")
    return obj if isinstance(obj, str) else None


def _require(name: str):
    """取 stereo_backend 里的符号；缺了就是 task-1 没落地，直接红而不是 skip 成假绿。"""
    obj = getattr(stereo_backend, name, None)
    if obj is None:
        raise AssertionError(f"stereo_backend.{name} 尚未落地（task-1 未完成）")
    return obj


def _pair(h: int = 24, w: int = 40, *, seed: int = 0, shift: int = 6):
    """返回 (left, right)，都是 (H, W, 3) uint8；left 由 right 循环右移 shift 得到。

    echo 引擎取 left 的第一通道，所以不同 seed 的第一通道必然不同 —— 这正是
    「迟到帧冒充下一帧」用例能分辨真假结果的前提。
    """
    rng = np.random.default_rng(seed)
    right = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    left = np.ascontiguousarray(np.roll(right, shift, axis=1))
    return left, right


def _echo(left: np.ndarray) -> np.ndarray:
    """echo 引擎的结果，逐像素。"""
    return np.ascontiguousarray(left[..., 0]).astype(np.float32)


# ---- Windows 内核对象小工具 ----------------------------------------------------
_SEM_ACCESS = 0x0002 | 0x00100000        # SEMAPHORE_MODIFY_STATE | SYNCHRONIZE
_WAIT_OBJECT_0, _WAIT_ABANDONED = 0x0, 0x80
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259

if _IS_WIN:  # pragma: no cover - 仅在 Windows 上定义
    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _K32.CreateSemaphoreW.restype = ctypes.c_void_p
    _K32.CreateSemaphoreW.argtypes = (ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_wchar_p)
    _K32.OpenSemaphoreW.restype = ctypes.c_void_p
    _K32.OpenSemaphoreW.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)
    _K32.ReleaseSemaphore.restype = ctypes.c_int
    _K32.ReleaseSemaphore.argtypes = (ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p)
    _K32.WaitForSingleObject.restype = ctypes.c_uint32
    _K32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    _K32.CloseHandle.argtypes = (ctypes.c_void_p,)
    _K32.OpenProcess.restype = ctypes.c_void_p
    _K32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    _K32.GetExitCodeProcess.restype = ctypes.c_int
    _K32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    _K32.TerminateProcess.restype = ctypes.c_int
    _K32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)


def _sem_create(name: str):
    handle = _K32.CreateSemaphoreW(None, 0, 1, name)      # 初值 0，max 1
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _sem_open(name: str):
    return _K32.OpenSemaphoreW(_SEM_ACCESS, 0, name)


def _sem_release(handle) -> bool:
    return bool(_K32.ReleaseSemaphore(handle, 1, None))


def _sem_take(handle, timeout_s: float) -> bool:
    rc = _K32.WaitForSingleObject(handle, max(0, int(timeout_s * 1000)))
    return rc in (_WAIT_OBJECT_0, _WAIT_ABANDONED)


def _sem_close(handle) -> None:
    if handle:
        _K32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    if not _IS_WIN:  # pragma: no cover
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    handle = _K32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong(0)
        if not _K32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        _K32.CloseHandle(handle)


def _kill(pid: int) -> None:
    """硬杀一个进程（模拟运行期崩溃/被任务管理器结束）。"""
    if _IS_WIN:
        handle = _K32.OpenProcess(0x0001, 0, int(pid))       # PROCESS_TERMINATE
        if handle:
            try:
                _K32.TerminateProcess(handle, 9)
            finally:
                _K32.CloseHandle(handle)
        return
    with contextlib.suppress(OSError):  # pragma: no cover
        os.kill(int(pid), 9)


def _wait_dead(pid: int, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.02)
    return False


# ---- 手工扮演父进程（直接打 worker 的协议） ------------------------------------
class RawWorker:
    """不经 stereo_backend，自己建 shm + 命名信号量并起 backend.stereo_worker。

    用途：验证 worker 本体的行为（echo 公式、CTRL 停机），把适配器的 bug 与
    worker 的 bug 分开。协议照 Lead 定稿：父建子开，ready 父 release、done 子 release。
    """

    def __init__(self, h: int = 16, w: int = 32, *, engine: str = "echo",
                 delay_s: float = 0.0, exit_after: int = 0, fx: float = 202.5) -> None:
        base = "neko_las2_" + uuid.uuid4().hex[:12]
        self.h, self.w = h, w
        self.names = {
            "left": f"{base}_left", "right": f"{base}_right", "disp": f"{base}_disp",
            "sem_ready": f"{base}_ready", "sem_done": f"{base}_done",
        }
        self.spec = {
            "left": self.names["left"], "right": self.names["right"], "disp": self.names["disp"],
            "sem_ready": self.names["sem_ready"], "sem_done": self.names["sem_done"],
            "h": h, "w": w, "parent_pid": os.getpid(), "fx": fx,
            "engine": engine, "delay_s": delay_s, "exit_after": exit_after,
        }
        self.shm_left = shared_memory.SharedMemory(name=self.names["left"], create=True, size=h * w * 3)
        self.shm_right = shared_memory.SharedMemory(name=self.names["right"], create=True, size=h * w * 3)
        self.shm_disp = shared_memory.SharedMemory(name=self.names["disp"], create=True,
                                                   size=CTRL_BYTES + h * w * 4)
        self.sem_ready = _sem_create(self.names["sem_ready"])
        self.sem_done = _sem_create(self.names["sem_done"])
        python = os.environ.get("NEKO_LAS_PYTHON") or sys.executable
        # stderr 落文件而不是 PIPE：管道写满会阻塞子进程（长用例里就等于假死）。
        self._err = tempfile.NamedTemporaryFile(prefix="las2_worker_", suffix=".err",
                                                delete=False, mode="w+b")
        self.proc = subprocess.Popen(
            [python, "-m", "backend.stereo_worker", json.dumps(self.spec)],
            cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=self._err,
        )
        self._closed = False

    # -- 控制块与画面 --
    @property
    def ctrl(self) -> np.ndarray:
        return np.frombuffer(self.shm_disp.buf, np.float32, 32, 0)

    def disp(self) -> np.ndarray:
        return np.frombuffer(self.shm_disp.buf, np.float32, self.h * self.w,
                             CTRL_BYTES).reshape(self.h, self.w)

    def error_text(self) -> str:
        raw = bytes(self.shm_disp.buf[ERR_OFFSET:CTRL_BYTES])
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    def stderr_text(self) -> str:
        with contextlib.suppress(Exception):
            self._err.flush()
            with open(self._err.name, "rb") as fh:
                return fh.read().decode("utf-8", "replace")[-2000:]
        return ""

    def wait_phase(self, phase: int, timeout: float = 60.0) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            got = float(self.ctrl[PHASE])
            if got == float(phase):
                return got
            if got == float(PHASE_FAILED):
                raise AssertionError(f"worker 报 PHASE=-1 失败：{self.error_text()!r}")
            if self.proc.poll() is not None:
                raise AssertionError(f"worker 已退出 rc={self.proc.returncode}：{self.stderr_text()!r}")
            time.sleep(0.01)
        raise AssertionError(f"等 PHASE={phase} 超时，当前 {float(self.ctrl[PHASE])}")

    # -- 一帧：写图 -> SEQ_REQ++ -> release ready -> 等 done --
    def submit(self, left: np.ndarray, right: np.ndarray) -> int:
        np.frombuffer(self.shm_left.buf, np.uint8, self.h * self.w * 3).reshape(self.h, self.w, 3)[:] = left
        np.frombuffer(self.shm_right.buf, np.uint8, self.h * self.w * 3).reshape(self.h, self.w, 3)[:] = right
        self.ctrl[SEQ_REQ] += 1.0
        if not _sem_release(self.sem_ready):
            raise AssertionError("release ready 失败")
        return int(self.ctrl[SEQ_REQ])

    def await_done(self, timeout: float = 10.0) -> bool:
        return _sem_take(self.sem_done, timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc.poll() is None and _IS_WIN:
            with contextlib.suppress(Exception):
                self.ctrl[CTRL] = 1.0                       # 父写 1.0 = 请求退出
                _sem_release(self.sem_ready)                # 唤醒阻塞在 ready 上的子进程
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=5)
        for handle in (self.sem_ready, self.sem_done):
            with contextlib.suppress(Exception):
                _sem_close(handle)
        for shm in (self.shm_left, self.shm_right, self.shm_disp):
            with contextlib.suppress(Exception):
                shm.close()
        for shm in (self.shm_left, self.shm_right, self.shm_disp):
            with contextlib.suppress(Exception):
                shm.unlink()
        with contextlib.suppress(Exception):
            os.unlink(self._err.name)


# ---- 监视父进程建了哪些命名对象/怎么起子进程 --------------------------------
_FIXED_UUID = uuid.UUID("0123456789abcdef0123456789abcdef")
_PREDICTED_BASE = "neko_las2_" + _FIXED_UUID.hex[:12]


@contextlib.contextmanager
def _capture_spawn():
    """拦住子进程拉起，返回 {"argv": [...]}；同时把 uuid4 钉死，便于预测名字。

    Popen 可能被 from subprocess import Popen 提前绑定，所以两处都打。
    """
    cap: dict = {}
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        if isinstance(argv, (list, tuple)):
            parts = [str(x) for x in argv]
            # 只认真正的 worker 启动：实现可能先用 where.exe / 其他探测命令找解释器。
            if "argv" not in cap and any("backend.stereo_worker" in x for x in parts):
                cap["argv"] = parts
        return real_popen(*args, **kwargs)

    patches = [mock.patch.object(subprocess, "Popen", new=spy)]
    if getattr(stereo_backend, "Popen", None) is not None:
        patches.append(mock.patch.object(stereo_backend, "Popen", new=spy))
    patches.append(mock.patch.object(uuid, "uuid4", return_value=_FIXED_UUID))
    if getattr(stereo_backend, "uuid4", None) is not None:
        patches.append(mock.patch.object(stereo_backend, "uuid4", return_value=_FIXED_UUID))
    for p in patches:
        p.start()
    try:
        yield cap
    finally:
        for p in reversed(patches):
            p.stop()


@contextlib.contextmanager
def _record_shm():
    """记录父进程以 create=True 建的每个共享内存 (name -> size)。

    比从 spec 里猜名字稳：不管实现从哪拿名字、用不用 uuid，建出来的对象都被记下。
    """
    rec: dict[str, int] = {}
    real = shared_memory.SharedMemory

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, name=None, create=False, size=0):
            super().__init__(name=name, create=create, size=size)
            if create:
                rec[str(self.name)] = int(self.size)

    patches = [mock.patch.object(shared_memory, "SharedMemory", Recording)]
    if getattr(stereo_backend, "SharedMemory", None) is not None:
        patches.append(mock.patch.object(stereo_backend, "SharedMemory", Recording))
    for p in patches:
        p.start()
    try:
        yield rec
    finally:
        for p in reversed(patches):
            p.stop()


def _spawn_spec(cap: dict) -> dict | None:
    """从拦截到的 argv 里取 spec；取不到返回 None（不猜）。"""
    for part in (cap.get("argv") or []):
        if isinstance(part, str) and part.lstrip().startswith("{"):
            with contextlib.suppress(json.JSONDecodeError):
                spec = json.loads(part)
                if isinstance(spec, dict) and "disp" in spec:
                    return spec
    return None


class _WorkerCase(unittest.TestCase):
    """公共夹具：把子进程解释器钉到当前 sys.executable（.venv，无 torch 也无妨）。"""

    H, W = 24, 40

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {"NEKO_LAS_PYTHON": sys.executable})
        env.start()
        self.addCleanup(env.stop)
        close = getattr(stereo_backend, "close_las2", None)
        if close is not None:
            close()                                    # 清掉上一用例可能的残留
            self.addCleanup(close)                     # 兜底：绝不留孤儿进程
        warm = getattr(stereo_backend, "_warm", None)
        if isinstance(warm, dict):
            warm.clear()
            warm["state"] = "idle"
            self.addCleanup(warm.clear)
            self.addCleanup(warm.update, state="idle")

    def _wait_ready(self, m, timeout: float = 90.0) -> dict:
        """等 worker 编译到 ready（走 adapter.prewarm_state()，不依赖 las2_prewarm）。"""
        deadline = time.monotonic() + timeout
        st: dict = {}
        while time.monotonic() < deadline:
            st = m.prewarm_state()
            state = _state_of(st)
            if state in ("ready", "eager"):
                return st
            if state == "failed":
                self.fail(f"worker 预热失败：{st}")
            time.sleep(0.01)
        self.fail(f"worker 在 {timeout}s 内没到 ready：prewarm_state={st} "
                  f"worker_status={_safe_worker_status()} describe={_safe_describe(m)}")

    def _wait_prewarm(self, want=("ready",), timeout: float = 90.0) -> dict:
        deadline = time.monotonic() + timeout
        st: dict = {}
        while time.monotonic() < deadline:
            st = stereo_backend.las2_prewarm_status()
            if st.get("state") in want:
                return st
            if st.get("state") == "failed":
                self.fail(f"prewarm 失败：{st}")
            time.sleep(0.01)
        self.fail(f"prewarm 在 {timeout}s 内没到 {want}，停在 {st}")

    def spawn_adapter(self, timeout_s: float = 30.0, **kw):
        """直接起 worker（不预热）并等 ready；给「故意弄坏 worker」的用例用。"""
        m = stereo_backend.make_stereo_backend(
            "las2", subprocess=True, shape=(self.H, self.W), engine="echo", timeout_s=timeout_s, **kw)
        self.addCleanup(m.close)
        self._wait_ready(m)
        return m

    def warm_adapter(self, timeout_s: float = 30.0, **kw):
        """走 las2_prewarm -> ready -> make；正常路径的用例用。"""
        stereo_backend.las2_prewarm((self.H, self.W), subprocess=True, engine="echo", **kw)
        self._wait_prewarm()
        m = stereo_backend.make_stereo_backend(
            "las2", subprocess=True, shape=(self.H, self.W), engine="echo", timeout_s=timeout_s)
        self.addCleanup(m.close)
        return m


def _safe_worker_status():
    with contextlib.suppress(Exception):
        return _require("las2_worker_status")()
    return "<error>"


def _safe_describe(m):
    with contextlib.suppress(Exception):
        return m.describe()
    return "<error>"


# =============================================================================
# a. 全链路：起子进程 -> prewarm ready -> compute -> 与 echo 公式逐像素一致
# =============================================================================
@_skip_no_worker
class EchoEndToEndTest(_WorkerCase):
    def test_prewarm_ready_then_echo_pixelwise(self) -> None:
        first = stereo_backend.las2_prewarm((self.H, self.W), subprocess=True, engine="echo")
        self.assertIn(first["state"], ("loading", "compiling", "ready"), first)   # 立即返回，不等
        st = self._wait_prewarm()
        self.assertEqual(st["state"], "ready", st)
        self.assertIsNone(st["error"], st)
        self.assertEqual(tuple(st["shape"]), (self.H, self.W), st)
        self.assertGreaterEqual(st["elapsed_s"], 0.0, st)

        m = stereo_backend.make_stereo_backend(
            "las2", subprocess=True, shape=(self.H, self.W), engine="echo", timeout_s=30.0)
        self.addCleanup(m.close)

        # 两幅「已知图」：不同 seed => 第一通道不同，逐像素对上才算通。
        for seed in (0, 1):
            left, right = _pair(self.H, self.W, seed=seed)
            out = m.compute(left, right)
            self.assertEqual(out.shape, (self.H, self.W), f"seed={seed}")
            self.assertEqual(out.dtype, np.float32, f"seed={seed}")
            np.testing.assert_array_equal(out, _echo(left), f"seed={seed} 的 disp 与 echo 公式不符")

        d = m.describe()
        self.assertEqual(d["kind"], "las2")
        self.assertIs(d["subprocess"], True, d)
        self.assertIs(d["alive"], True, d)
        self.assertIsInstance(d["pid"], int, d)
        self.assertGreaterEqual(d["frames"], 2, d)
        self.assertIsInstance(d["timeouts"], int, d)
        for key in ("compiled", "compile_error"):
            self.assertIn(key, d)
        _timing_ms(d)                                  # 有上一帧耗时才谈得上分段诊断

        # 父进程侧能读到同一个 worker 状态
        ws = _require("las2_worker_status")()
        self.assertIsNotNone(ws, "worker 已起来，las2_worker_status() 不该是 None")
        self.assertTrue(ws["alive"], ws)
        self.assertEqual(int(ws["pid"]), int(d["pid"]), ws)
        self.assertGreaterEqual(int(ws["frames"]), 2, ws)
        for key in ("timeouts", "shape", "state", "error"):
            self.assertIn(key, ws)
        _timing_ms(ws)


# =============================================================================
# b. 超时：delay_s > timeout_s => StereoBusy；迟到的 done 不能冒充下一帧
# =============================================================================
@_skip_no_worker
class TimeoutIsolationTest(_WorkerCase):
    def test_delay_over_timeout_raises_stereo_busy(self) -> None:
        m = self.spawn_adapter(timeout_s=0.2, spec_extra={"delay_s": 0.5})
        left, right = _pair(self.H, self.W)
        t0 = time.monotonic()
        with self.assertRaises(TimeoutError) as ctx:
            m.compute(left, right)
        spent = time.monotonic() - t0
        self.assertIsInstance(ctx.exception, _require("StereoBusy"))
        self.assertLess(spent, 0.5, f"超时应该 ~0.2 s 返回，实际 {spent:.2f}s（等满了 delay？）")
        self.assertGreaterEqual(m.describe()["timeouts"], 1)
        # 只是慢，不是死：子进程还活着
        self.assertTrue(m.describe()["alive"])

    def test_env_delay_fallback(self) -> None:
        """环境变量回落路径：NEKO_LAS_DELAY_S 在 spec_extra 缺席时生效。"""
        with mock.patch.dict(os.environ, {"NEKO_LAS_DELAY_S": "0.5"}):
            m = self.spawn_adapter(timeout_s=0.2)
            with self.assertRaises(TimeoutError):
                m.compute(*_pair(self.H, self.W))

    def test_late_frame_is_not_returned_for_the_next_compute(self) -> None:
        """核心回归：第 1 帧超时后，它的 done 会迟到落到信号量上。

        实现若「拿到任意 done 就当本帧结果」，第二次 compute 会立刻把 disp 缓冲里
        **第 1 帧**的视差当成第 2 帧返回。正确实现必须按 SEQ_DONE == SEQ_REQ 判别并
        丢弃迟到帧；delay 恒定时第二次 compute 必然也超时（StereoBusy）。两条路都可以，
        唯独不能返回 A 帧的数据。
        """
        m = self.spawn_adapter(timeout_s=0.35, spec_extra={"delay_s": 0.6})
        a_l, a_r = _pair(self.H, self.W, seed=0, shift=6)
        b_l, b_r = _pair(self.H, self.W, seed=7, shift=6)
        self.assertFalse(np.array_equal(_echo(a_l), _echo(b_l)), "两幅待测图第一通道必须不同")

        with self.assertRaises(TimeoutError):
            m.compute(a_l, a_r)

        time.sleep(0.75)          # 让 A 帧在这段时间里算完，done 上留一个陈旧令牌

        try:
            out = m.compute(b_l, b_r)
        except _require("StereoBusy"):
            return                # 正确：陈旧 done 被丢弃，本帧还没好
        self.assertFalse(np.array_equal(out, _echo(a_l)), "第二次 compute 返回了上一帧的视差")
        np.testing.assert_array_equal(out, _echo(b_l), "第二次 compute 的结果与 echo 公式不符")


# =============================================================================
# c. 崩溃：exit_after=1 => 下一帧 StereoWorkerError，不是超时
# =============================================================================
@_skip_no_worker
class CrashTest(_WorkerCase):
    def _spawn_crashy(self, timeout_s: float, **spec_extra):
        """起一个注定会崩的 worker。

        若实现会在 spawn 阶段预跑一帧（那帧就把进程弄死了），make_stereo_backend 本身
        就抛 StereoWorkerError —— 返回 None，用例改判「崩溃面已覆盖」，不硬凑第二帧。
        """
        try:
            return self.spawn_adapter(timeout_s=timeout_s, spec_extra=spec_extra)
        except _require("StereoWorkerError"):
            return None

    def _assert_crash_surface(self, m) -> None:
        left, right = _pair(self.H, self.W, seed=3)
        with self.assertRaises(RuntimeError) as ctx:
            m.compute(left, right)          # 子进程已经 os._exit(9)，这里必须报「死了」而不是「慢」
        self.assertIsInstance(ctx.exception, _require("StereoWorkerError"))
        self.assertFalse(m.describe()["alive"], "进程死了，describe().alive 必须转 False")

    def test_exit_after_first_frame(self) -> None:
        m = self._spawn_crashy(20.0, exit_after=1)
        if m is None:
            return                      # spawn 阶段就崩：StereoWorkerError 已抛出，崩溃面覆盖
        left, right = _pair(self.H, self.W, seed=2)
        out = m.compute(left, right)        # 第 1 帧正常返回，随后子进程立刻 _exit(9)
        np.testing.assert_array_equal(out, _echo(left))
        deadline = time.monotonic() + 10.0
        while m.describe()["alive"] and time.monotonic() < deadline:
            time.sleep(0.01)
        self._assert_crash_surface(m)

    def test_env_exit_after_fallback(self) -> None:
        with mock.patch.dict(os.environ, {"NEKO_LAS_EXIT_AFTER": "1"}):
            m = self._spawn_crashy(20.0)
            if m is None:
                return
            left, right = _pair(self.H, self.W, seed=4)
            with contextlib.suppress(_require("StereoWorkerError")):
                m.compute(left, right)
            deadline = time.monotonic() + 10.0
            while m.describe()["alive"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self._assert_crash_surface(m)


# =============================================================================
# d. 清理：close() 后命名对象消失、子进程退出、可重复调用
# =============================================================================
@_skip_no_worker
class CleanupTest(_WorkerCase):
    def test_close_releases_named_objects_and_child(self) -> None:
        with _record_shm() as created, _capture_spawn() as cap:
            stereo_backend.las2_prewarm((self.H, self.W), subprocess=True, engine="echo")
            self._wait_prewarm()
            m = stereo_backend.make_stereo_backend(
                "las2", subprocess=True, shape=(self.H, self.W), engine="echo", timeout_s=30.0)
            self.addCleanup(m.close)
            out = m.compute(*_pair(self.H, self.W))
        self.assertEqual(out.shape, (self.H, self.W))

        # -- 布局契约：左/右 H*W*3，disp 384 + H*W*4 --
        sizes = sorted(created.values())
        self.assertEqual(sizes, sorted([self.H * self.W * 3, self.H * self.W * 3,
                                        CTRL_BYTES + self.H * self.W * 4]),
                         f"共享内存尺寸不对：{created}")
        names = list(created)

        # -- 命名契约：neko_las2_<12 hex>_{left,right,disp}（三个一组）--
        for nm in names:
            self.assertTrue(nm.startswith("neko_las2_"), f"共享内存名没按前缀规则：{nm}")

        # -- spec 契约（拦截到就看，拦不到不猜）--
        spec = _spawn_spec(cap)
        if spec is not None:
            for key in ("left", "right", "disp", "sem_ready", "sem_done",
                        "h", "w", "parent_pid", "fx", "engine", "delay_s", "exit_after"):
                self.assertIn(key, spec, f"spec 缺字段 {key}")
            self.assertNotIn("shape_bytes", spec)          # Lead 已明确删掉
            self.assertEqual((spec["h"], spec["w"]), (self.H, self.W))
            self.assertEqual(spec["engine"], "echo")
            self.assertEqual(int(spec["parent_pid"]), os.getpid())
            self.assertEqual(sorted([spec["left"], spec["right"], spec["disp"]]), sorted(names))
        argv = cap.get("argv") or []
        if argv:
            # 启动命令契约：[python, "-m", "backend.stereo_worker", spec_json]
            self.assertEqual(argv[0], sys.executable, f"NEKO_LAS_PYTHON 没生效：{argv[:1]}")
            self.assertEqual(argv[1:3], ["-m", "backend.stereo_worker"], argv[:3])

        pid = int(m.describe()["pid"])
        self.assertTrue(_pid_alive(pid), "compute 之后子进程应该在跑")

        # -- close：进程退出 + 命名对象全消失 --
        m.close()
        self.assertTrue(_wait_dead(pid, 8.0), f"close() 之后子进程 {pid} 还在")
        for nm in names:
            with self.assertRaises(FileNotFoundError, msg=f"close 后 {nm} 还能打开"):
                shared_memory.SharedMemory(name=nm)
        st = _require("las2_worker_status")()
        self.assertTrue(st is None or st.get("alive") is False, f"close 后仍报 alive：{st}")

        # -- 幂等：close() 可重复调用，close_las2() 也一样 --
        m.close()
        m.close()
        stereo_backend.close_las2()
        stereo_backend.close_las2()

    def test_close_before_ready_is_safe(self) -> None:
        """还没 ready 就 close：不能挂着等编译，也不能抛。"""
        m = stereo_backend.make_stereo_backend(
            "las2", subprocess=True, shape=(self.H, self.W), engine="echo", timeout_s=30.0,
            spec_extra={"delay_s": 0.5})           # 首帧要 0.5 s，close 不等它
        self.addCleanup(m.close)
        pid = int(m.describe()["pid"])
        t0 = time.monotonic()
        m.close()
        self.assertLess(time.monotonic() - t0, 8.0, "close() 卡住了")
        self.assertTrue(_wait_dead(pid, 8.0))
        stereo_backend.close_las2()


# =============================================================================
# e. 空闲期不误杀：无帧 3 s 后子进程仍 alive
# =============================================================================
@_skip_no_worker
class IdleTest(_WorkerCase):
    def test_worker_survives_three_seconds_idle(self) -> None:
        m = self.warm_adapter(timeout_s=30.0)
        left, right = _pair(self.H, self.W, seed=5)
        np.testing.assert_array_equal(m.compute(left, right), _echo(left))
        pid = int(m.describe()["pid"])

        hb0 = float(m._ctrl[HEARTBEAT]) if hasattr(m, "_ctrl") and m._ctrl is not None else None
        time.sleep(3.0)                       # 故意不喂帧

        self.assertTrue(_pid_alive(pid), "空闲 3 s 后子进程被误杀")
        # 心跳与帧解耦（Lead 定稿）：空闲 3 s 里也应每 100 ms 涨一次。
        if hb0 is not None:
            hb1 = float(m._ctrl[HEARTBEAT])
            self.assertGreater(hb1, hb0, f"空闲 3 s 心跳没涨：{hb0} -> {hb1}")
            self.assertGreaterEqual(hb1 - hb0, 10.0, f"3 s 内至少该涨 ~30 次：{hb0} -> {hb1}")
        self.assertTrue(m.describe()["alive"])
        ws = _require("las2_worker_status")()
        self.assertTrue(ws["alive"], ws)
        self.assertEqual(int(ws["pid"]), pid)

        # 而且还能继续算（不是僵尸）
        left2, right2 = _pair(self.H, self.W, seed=6)
        np.testing.assert_array_equal(m.compute(left2, right2), _echo(left2))
        self.assertGreaterEqual(m.describe()["frames"], 2)


# =============================================================================
# worker 本体（不经适配器）：手工握手 / echo 公式 / CTRL 停机
# =============================================================================
@unittest.skipUnless(_IS_WIN, "命名信号量协议只在 Windows 上有意义")
@_skip_no_worker
class RawWorkerProtocolTest(unittest.TestCase):
    H, W = 16, 32

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {"NEKO_LAS_PYTHON": sys.executable})
        env.start()
        self.addCleanup(env.stop)

    def test_echo_engine_and_control_block(self) -> None:
        w = RawWorker(self.H, self.W, engine="echo")
        self.addCleanup(w.close)
        w.wait_phase(PHASE_READY, timeout=60.0)
        # 心跳契约（Lead 定稿）：每 100 ms 自增一次，**与有没有帧无关**；初值 1.0（0 = 还没起来）。
        # 所以准备好 300 ms 之后至少应该是 2，且空闲不喂帧也继续涨。
        time.sleep(0.35)
        hb0 = float(w.ctrl[HEARTBEAT])
        self.assertGreaterEqual(hb0, 2.0, f"ready 300 ms 后 HEARTBEAT 应 ≥2，实际 {hb0}")
        time.sleep(0.35)
        self.assertGreater(float(w.ctrl[HEARTBEAT]), hb0, "空闲期心跳没有推进（每 100 ms 应自增）")

        frames0 = int(w.ctrl[FRAMES])
        left, right = _pair(self.H, self.W, seed=11)
        seq = w.submit(left, right)
        self.assertTrue(w.await_done(10.0), "等 done 超时（ready/done 配对反了？）")
        self.assertEqual(int(w.ctrl[SEQ_DONE]), seq, "SEQ_DONE 没跟上 SEQ_REQ")
        self.assertEqual(int(w.ctrl[FRAMES]), frames0 + 1, "FRAMES 没按帧递增")
        self.assertEqual(int(w.ctrl[H_IX]), self.H)
        self.assertEqual(int(w.ctrl[W_IX]), self.W)
        self.assertEqual(float(w.ctrl[VERSION_IX]), VERSION)
        np.testing.assert_array_equal(w.disp(), _echo(left))

        # 第二帧：确认序号递增、结果不串味
        left2, right2 = _pair(self.H, self.W, seed=12)
        seq2 = w.submit(left2, right2)
        self.assertTrue(w.await_done(10.0))
        self.assertEqual(int(w.ctrl[SEQ_DONE]), seq2)
        self.assertEqual(int(w.ctrl[FRAMES]), frames0 + 2)
        np.testing.assert_array_equal(w.disp(), _echo(left2))

    def test_ctrl_requests_clean_exit(self) -> None:
        w = RawWorker(self.H, self.W)
        self.addCleanup(w.close)
        w.wait_phase(PHASE_READY, timeout=60.0)
        w.ctrl[CTRL] = 1.0                  # 父写 1.0 = 要求退出
        _sem_release(w.sem_ready)           # 唤醒阻塞在 ready 上的子进程
        self.assertIsNotNone(w.proc.wait(timeout=10), "CTRL 之后子进程没退出")
        self.assertEqual(w.proc.returncode, 0, w.stderr_text())
        self.assertEqual(float(w.ctrl[PHASE]), float(PHASE_EXITED), "退出前应把 PHASE 置 -2")


# =============================================================================
# 运行期崩溃后重建：kill 掉子进程 -> close_las2 -> 再起得来
# =============================================================================
@_skip_no_worker
class RestartTest(_WorkerCase):
    def test_rebuild_after_worker_killed(self) -> None:
        """模拟运行期被外力 kill（不是 exit_after 的自杀路径）：
        父进程要能认出「死了」，close_las2() 收干净，之后还能重新建起来出帧。
        """
        m = self.warm_adapter(timeout_s=30.0)
        left, right = _pair(self.H, self.W, seed=8)
        np.testing.assert_array_equal(m.compute(left, right), _echo(left))
        pid = int(m.describe()["pid"])
        self.assertTrue(_pid_alive(pid))

        # 外力 kill（等同任务管理器结束进程 / 驱动崩溃）：不走 CTRL，也不给清理机会。
        _kill(pid)
        self.assertTrue(_wait_dead(pid, 10.0), f"kill 之后 {pid} 还在")

        # ⚠️ Windows 上 Popen.poll() 比内核退出晚 5-50 ms（进程已死但 poll() 仍是 None），
        # 适配器的 alive() 读的就是 poll()。等它收敛，否则会拿到竞态窗口里的假结果。
        deadline = time.monotonic() + 10.0
        while m.alive() and time.monotonic() < deadline:
            time.sleep(0.01)

        # 死了就要报「死了」，不是超时
        with self.assertRaises(RuntimeError) as ctx:
            m.compute(*_pair(self.H, self.W, seed=9))
        self.assertIsInstance(ctx.exception, _require("StereoWorkerError"))
        self.assertFalse(m.describe()["alive"])

        # 收干净再重建
        stereo_backend.close_las2()
        m2 = stereo_backend.make_stereo_backend(
            "las2", subprocess=True, shape=(self.H, self.W), engine="echo", timeout_s=30.0)
        self.addCleanup(m2.close)
        self._wait_ready(m2)
        self.assertIs(m2.describe()["alive"], True, m2.describe())
        self.assertNotEqual(int(m2.describe()["pid"]), pid, "重建应该起一个新进程")
        left2, right2 = _pair(self.H, self.W, seed=10)
        np.testing.assert_array_equal(m2.compute(left2, right2), _echo(left2), "重建后算错了")


# =============================================================================
# 契约面：异常类型
# =============================================================================
@_skip_no_worker
class ContractTest(unittest.TestCase):
    def test_exception_hierarchy(self) -> None:
        self.assertTrue(issubclass(_require("StereoBusy"), TimeoutError))
        self.assertTrue(issubclass(_require("StereoWorkerError"), RuntimeError))
        # 超时不能被崩溃侧的 except 顺手吞掉：两者必须是不同分支
        self.assertFalse(issubclass(_require("StereoBusy"), RuntimeError))
        self.assertFalse(issubclass(_require("StereoWorkerError"), TimeoutError))


if __name__ == "__main__":
    unittest.main()
