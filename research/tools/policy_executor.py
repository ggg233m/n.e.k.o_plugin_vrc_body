# -*- coding: utf-8 -*-
"""动作块执行器 + 安全门：10 Hz 取步，限速/限加速度/超时停车。

输入是 PolicyRunner 产出的 12 步 ``(vx, vz)`` m/s 块。执行器只负责把当前步转成
``/input`` 轴并交给 ``send_axes(x, y, duration_ms)``；发不发由调用方实现。

安全门（闭环首跑的最小集）：
  * ``|v| <= v_max``（默认 1.2 m/s）
  * 逐步加速度 ``|Δv| <= dv_max``（默认 0.5 m/s per 0.1 s）
  * ``|wp| < wp_stop`` 或 stale 的块直接全零
  * ``now - obs_t > stale_s``（1 Hz 默认 1.3 s）立即停车
  * 停止文件（默认 ``runs/STOP``）存在即停车（人工急停，不依赖 UI）
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

K_STEPS = 12
STEP_S = 0.1
V_AXIS_DEAD = 0.1


def mps_to_axes(vx: float, vz: float) -> tuple[float, float]:
    """§9.1：v (m/s) → /input Horizontal(x) / Vertical(y)，各轴独立钳位。"""
    def one(v, vmax):
        if abs(v) <= V_AXIS_DEAD:
            return 0.0
        return math.copysign(min(1.0, V_AXIS_DEAD + 0.9 * abs(v) / vmax), v)
    return one(vx, 3.0), one(vz, 4.0)


class SafetyGate:
    def __init__(self, v_max: float = 1.2, dv_max: float = 0.5, floor_frac: float = 0.6):
        self.v_max = float(v_max)
        self.dv_max = float(dv_max)
        # 「不许比跟随器差」的下限系数（2026-11-02）。
        #
        # 为什么要有它：VLA 接管后 A* 的摇杆被静音（nav_online.py 的 drive_enabled 分支），
        # 但跟随器的 forward **仍在同一拍算出来**、只是没发。于是「策略比几何跟随器更差」
        # 这件事在结构上没有兜底 —— 实测策略稳态只有 0.25-0.5 m/s，而跟随器是同一条
        # 路径的 4.0 m/s（轴 1.0）。
        #
        # 这条下限把「替代」改成「调制」：策略仍然决定方向与快慢，但它**不允许**把速度
        # 压到跟随器本可以给出的 60% 以下。跟随器 forward<=0 时（到点/未定位/原地转）
        # 下限为 0，即完全放行策略的停车决定 —— 该停的地方必须能停。
        #
        # 0 关闭（逐位回到旧行为）。
        self.floor_frac = max(0.0, float(floor_frac))

    def apply(self, block, gated: bool, stale: bool, wp_stop: float,
              floor_mps: float = 0.0):
        """限速/限加速；``floor_mps`` 是「不许比跟随器差」的速度下限（米/秒）。

        ``floor_mps`` 只抬**前进分量**（vz，机体系前方），不碰侧向：
        侧向是 VLA 的修正自由度，把它一起抬会变成「强行横着走」。
        ``floor_mps <= 0`` 时逐位等于旧行为。
        """
        b = np.asarray(block, dtype=np.float64).reshape(K_STEPS, 2).copy()
        if gated or stale:
            return np.zeros((K_STEPS, 2))
        if floor_mps > 0.0:
            # 只在策略自己的前进速度低于下限时抬；策略已经更快就原样保留。
            b[:, 1] = np.maximum(b[:, 1], float(floor_mps))
        speeds = np.linalg.norm(b, axis=1)
        for i in range(K_STEPS):
            if speeds[i] > self.v_max:
                b[i] *= self.v_max / speeds[i]
            if i > 0:
                d = b[i] - b[i - 1]
                n = float(np.linalg.norm(d))
                if n > self.dv_max:
                    b[i] = b[i - 1] + d * (self.dv_max / n)
        return b


class PolicyExecutor:
    def __init__(self, send_axes, stale_s: float = 1.3, stop_file: str | None = None,
                 dry_run: bool = True, gate: SafetyGate | None = None,
                 hold_provider: Callable[[], bool] | None = None,
                 floor_provider: Callable[[], float] | None = None):
        self.send_axes = send_axes
        self.stale_s = float(stale_s)
        self.stop_file = Path(stop_file) if stop_file else None
        self.dry_run = bool(dry_run)
        self.gate = gate or SafetyGate()
        # 外部转向代管的按住信号（方案 §9.1 原地转门）：True 时按住当前块不发轴。
        # 读不到（provider 抛错/为 None）按 False 处理——hold 缺失不能变成「策略不动」，
        # 那是 stale 阈值该管的事。与 near_obstacle veto 的容错方向一致。
        self._hold_provider = hold_provider
        # 跟随器本拍**本可以**给出的前进速度（m/s）。策略低于它时被抬到下限。
        # 读不到返回 0 = 不设下限（跟随器自己也不走，没有「更好」可言）。
        self._floor_provider = floor_provider
        self.n_holds = 0
        self.n_floored = 0
        self._block = np.zeros((K_STEPS, 2))
        self._obs_t = -1e9
        self._gated = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="policy-exec", daemon=True)
        # 诊断计数：submit 次数 / 实际发轴次数 / halt 次数（2026-11-01 在线排障欠的债）
        self.n_submits = 0
        self.n_sends = 0
        self.n_halts = 0
        self.last_send = None

    def submit(self, block, obs_t: float, meta: dict | None = None) -> None:
        meta = meta or {}
        gated = bool(meta.get("gated", False)) or bool(meta.get("stale", False))
        floor = 0.0
        if self._floor_provider is not None and not gated:
            try:
                floor = float(self._floor_provider() or 0.0)
            except Exception:  # noqa: BLE001 - 读不到就不设下限，绝不因此停车
                floor = 0.0
            floor = max(0.0, floor) * self.gate.floor_frac
        if floor > 0.0:
            self.n_floored += 1
        self.n_submits += 1
        with self._lock:
            self._block = self.gate.apply(block, gated, bool(meta.get("stale", False)), 0.0,
                                          floor_mps=floor)
            self._obs_t = float(obs_t)
            self._gated = gated

    def start(self): self._th.start()
    def stop(self): self._stop.set()

    def _halt(self):
        if not self.dry_run:
            self.send_axes(0.0, 0.0, 400)

    def _loop(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            if self.stop_file is not None and self.stop_file.exists():
                self._halt(); time.sleep(STEP_S); continue
            with self._lock:
                age = now - self._obs_t
                block, gated = self._block, self._gated
            if age > self.stale_s or gated:
                self.n_halts += 1
                self._halt(); time.sleep(STEP_S); continue
            if self._hold_provider is not None:
                try:
                    hold = bool(self._hold_provider())
                except Exception:  # noqa: BLE001 - 读不到按没有处理
                    hold = False
                if hold:
                    # 转向代管原地转中：块留着（不失效），只按住轴——转到位后无缝续走。
                    self.n_holds += 1
                    time.sleep(STEP_S); continue
            k = min(K_STEPS - 1, max(0, int(age / STEP_S)))
            vx, vz = block[k]
            x, y = mps_to_axes(vx, vz)
            self.n_sends += 1
            self.last_send = (round(x, 3), round(y, 3))
            if not self.dry_run:
                self.send_axes(x, y, 400)
            time.sleep(STEP_S)
