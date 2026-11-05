# -*- coding: utf-8 -*-
"""动作块执行器 + 安全门：10 Hz 取步，限速/限加速度/超时停车。

输入是 PolicyRunner 产出的 12 步 ``(vx, vz)`` m/s 块。执行器只负责把当前步转成
``/input`` 轴并交给 ``send_axes(x, y, duration_ms)``；发不发由调用方实现。

安全门（闭环首跑的最小集）：
  * ``|v| <= v_max``（默认 1.2 m/s）
  * 逐步加速度 ``|Δv| <= dv_max``（默认 0.5 m/s per 0.1 s）
  * 停止只来自五种来源（2026-11-06 收敛，见 policy_modes）：显式导航状态 /
    停止头 / 安全门 / 过期保护 / ESTOP。``|wp|≈0`` **不是**停止条件，
    它只表示「没有路径先验」。
  * ``now - obs_t > stale_s``（2 Hz 默认 1.3 s）立即停车（过期保护）
  * 停止文件（默认 ``runs/STOP``）存在即停车（人工急停，不依赖 UI）
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from policy_modes import STOP_SOURCES, decide_stop

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


def omega_to_turn(w_rad_s: float, w_max: float = 1.57) -> float:
    """ω（**弧度/秒，右转为正**）→ /input 的转向轴（-1..1，右正）。

    `w_max` 同单位（**弧度/秒**）的满舵角速度，属于世界/角色标定，不是模型参数。

    ⚠️ 2026-11-05 修：此前函数体用了未定义变量 `w_deg_s`，一调用就 NameError。
    同时把单位契约写死为 rad/s —— 数据标签（build_policy_dataset.omega_rad_s_at）
    产出的就是 rad/s，执行器早期注释按 deg/s 写的，两处不一致。
    """
    if w_max <= 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, float(w_rad_s) / float(w_max))))


def omega_to_delta_deg(w_rad_s: float, w_max: float = 1.57, step_ms: int = 100) -> float:
    """把右正的 ω 转成 scheduler 使用的左正单步转角（度）。"""
    turn = omega_to_turn(w_rad_s, w_max)
    return -math.degrees(turn * float(w_max)) * (int(step_ms) / 1000.0)


class SafetyGate:
    def __init__(self, v_max: float = 1.2, dv_max: float = 0.5, floor_frac: float = 0.6,
                 w_max: float = 1.57, dw_max: float = 3.14):
        self.v_max = float(v_max)
        self.dv_max = float(dv_max)
        # ω 是**弧度/秒**、右转为正（与 vx 右正同向）。
        # 默认 1.57 rad/s = 90 deg/s 满舵；3.14 rad/s^2 约等于「0.5 s 打满」。
        self.w_max = float(w_max)
        self.dw_max = float(dw_max)
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

    def apply(self, block, stop: bool, stale: bool, wp_stop: float,
              floor_mps: float = 0.0):
        """限速/限加速；``floor_mps`` 是「不许比跟随器差」的速度下限（米/秒）。

        ``stop``：上游（runtime / executor）已经判定的停止，``stale``：过期保护。
        两者任一为真即整块归零 —— 这里**不再**看 wp，``wp_stop`` 参数只为保持
        旧调用点的位置参数兼容，值被忽略（wp≈0 只表示没有路径先验）。

        ``floor_mps`` 只抬**前进分量**（vz，机体系前方），不碰侧向：
        侧向是 VLA 的修正自由度，把它一起抬会变成「强行横着走」。
        ``floor_mps <= 0`` 时逐位等于旧行为。
        """
        # reshape(-1) 跟随动作维度：2 通道（vx, vz）与 3 通道（vx, vz, ω）共用
        b = np.asarray(block, dtype=np.float64).reshape(K_STEPS, -1).copy()
        if b.shape[1] < 2:
            raise ValueError("动作块至少要 2 维（vx, vz），收到 %r" % (b.shape,))
        if stop or stale:
            return np.zeros_like(b)
        if floor_mps > 0.0:
            # 只在策略自己的前进速度低于下限时抬；策略已经更快就原样保留。
            b[:, 1] = np.maximum(b[:, 1], float(floor_mps))
        speeds = np.linalg.norm(b[:, :2], axis=1)
        for i in range(K_STEPS):
            if speeds[i] > self.v_max:
                b[i, :2] *= self.v_max / speeds[i]
            if i > 0:
                d = b[i, :2] - b[i - 1, :2]
                n = float(np.linalg.norm(d))
                if n > self.dv_max:
                    b[i, :2] = b[i - 1, :2] + d * (self.dv_max / n)
            if b.shape[1] > 2:
                # ω 单独限幅：与线速度解耦，否则「转得快」会被当成交速度超限压掉
                w = float(b[i, 2])
                w = max(-self.w_max, min(self.w_max, w))
                if i > 0:
                    dw = w - float(b[i - 1, 2])
                    if abs(dw) > self.dw_max:
                        w = float(b[i - 1, 2]) + math.copysign(self.dw_max, dw)
                b[i, 2] = w
        return b


class PolicyExecutor:
    def __init__(self, send_axes, stale_s: float = 1.3, stop_file: str | None = None,
                 dry_run: bool = True, gate: SafetyGate | None = None,
                 hold_provider: Callable[[], bool] | None = None,
                 floor_provider: Callable[[], float] | None = None,
                 act_dim: int = 3, w_max: float = 1.57,
                 send_turn: Callable[[float], bool] | None = None,
                 step_ms: int = 100):
        self.send_axes = send_axes
        # 转向走独立通道（度/次）。None = 没有接线，此时统计丢弃次数而不是静默吞掉。
        self.send_turn = send_turn
        self.step_ms = int(step_ms)
        self.n_turn_dropped = 0
        self.n_turn_sent = 0
        self.act_dim = int(act_dim)
        # 满舵对应的角速度（**弧度/秒**），用于把模型输出的 ω 归一化到 -1..1。
        # 数据标签是 rad/s（build_policy_dataset.omega_rad_s_at），不是 deg/s。
        self.w_max = float(w_max)
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
        self._block = np.zeros((K_STEPS, int(act_dim)))
        self._obs_t = -1e9
        # `_gated` 保留为旧名（service.py 的 /policy/state 读它），语义已改为「停止」。
        self._gated = True
        self._stop_flag = True
        self._stop_source = None
        self._stop_reason = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="policy-exec", daemon=True)
        # 诊断计数：submit 次数 / 实际发轴次数 / halt 次数（2026-11-01 在线排障欠的债）
        self.n_submits = 0
        self.n_sends = 0
        self.n_halts = 0
        # 按来源归因的 halt 计数（2026-11-06）：`stop_recall` 必须说清「在哪种
        # 停止源下测的」，所以每一种来源都单独记一份，禁止把门控数字混在一起。
        self.n_halts_by_source = {k: 0 for k in STOP_SOURCES}
        self.last_stop_source = None
        self.last_stop_reason = None
        self.last_send = None

    def _estop_requested(self) -> bool:
        return bool(self.stop_file is not None and self.stop_file.exists())

    def submit(self, block, obs_t: float, meta: dict | None = None) -> None:
        meta = meta or {}
        # 2026-11-06：唯一语义是 `stop`；`gated` 只作旧调用方（policy_runner 早期
        # 版本、注入式测试替身）的兼容别名。`safety_veto` 由感知线程给出。
        upstream_stop = bool(meta.get("stop", meta.get("gated", False)))
        source = meta.get("stop_source")
        if upstream_stop and not source:
            source = "safety_gate" if meta.get("safety_veto") else "nav_state"
        stale = bool(meta.get("stale", False))
        safety = bool(meta.get("safety_veto", False))
        # 定位安全门：与 wp / nav mode 完全独立。None = 没有定位信息，不适用。
        loc_ok = meta.get("localization_ok")
        decision = decide_stop(stale=stale, safety_veto=safety,
                               localization_ok=loc_ok,
                               localization_reason=meta.get("localization_reason"),
                               prior_stop=upstream_stop, prior_source=source,
                               prior_reason=meta.get("stop_reason"))
        stop = bool(decision["stop"])
        floor = 0.0
        if self._floor_provider is not None and not stop:
            try:
                floor = float(self._floor_provider() or 0.0)
            except Exception:  # noqa: BLE001 - 读不到就不设下限，绝不因此停车
                floor = 0.0
            floor = max(0.0, floor) * self.gate.floor_frac
        if floor > 0.0:
            self.n_floored += 1
        self.n_submits += 1
        with self._lock:
            self._block = self.gate.apply(block, stop, stale, 0.0,
                                          floor_mps=floor)
            self._obs_t = float(obs_t)
            self._gated = stop
            self._stop_flag = stop
            self._stop_source = decision["source"]
            self._stop_reason = decision["reason"]

    def start(self): self._th.start()
    def stop(self): self._stop.set()

    def _send(self, x: float, y: float, turn: float, turn_deg: float = 0.0) -> None:
        """下发：摇杆两轴走 send_axes，转向走**独立通道**。

        ⚠️ 2026-11-05 修：此前 turn 用关键字塞进 send_axes，而 VRChat 的摇杆
        只有 x(横移)/y(前进) 两轴，`policy_head._send_axes(x, y, d)` 签名接不住，
        `except TypeError` 一兜底就把 turn 丢掉 —— **模型输出的 ω 从未到达执行闭环**。
        真实转向通道是 service 的 scheduler.submit("turn", {"delta_deg": ...})，
        单位是**度**。所以这里把 turn（归一化 -1..1）换算成 delta_deg 再单独发。
        """
        sent = False
        try:
            sent = bool(self.send_axes(x, y, self.step_ms))
        except TypeError:
            self.send_axes(x, y, self.step_ms)
            sent = True
        if abs(turn_deg) > 1e-6:
            if self.send_turn is not None:
                self.send_turn(float(turn_deg))
            else:
                self.n_turn_dropped += 1
        return sent

    def _halt(self):
        """停车：摇杆归零 + 转向归零。急停必须同时清两条通道。"""
        if not self.dry_run:
            self._send(0.0, 0.0, 0.0, 0.0)

    def _loop(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            with self._lock:
                age = now - self._obs_t
                block, stop, source, reason = (self._block, self._stop_flag,
                                               self._stop_source, self._stop_reason)
            # 执行器视野内的两种来源在这里合并：ESTOP（STOP 文件）与过期保护。
            # 优先级（estop > stale > 上游）由 policy_modes.decide_stop 唯一决定。
            decision = decide_stop(estop=self._estop_requested(),
                                   stale=age > self.stale_s,
                                   prior_stop=stop, prior_source=source,
                                   prior_reason=reason)
            if decision["stop"]:
                self.n_halts += 1
                self.last_stop_source = decision["source"]
                self.last_stop_reason = decision["reason"]
                if decision["source"] in self.n_halts_by_source:
                    self.n_halts_by_source[decision["source"]] += 1
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
            row = block[k]
            vx, vz = float(row[0]), float(row[1])
            # ⚠️ 模型输出的是 **rad/s**（见 build_policy_dataset.omega_rad_s_at）。
            # 早期这里叫 w_deg，容易误以为要换单位；实际直接透传，单位由 w_max 决定。
            w_rad = float(row[2]) if len(row) > 2 else 0.0
            x, y = mps_to_axes(vx, vz)
            turn = omega_to_turn(w_rad, self.w_max)
            # 一个块步长内的转角：归一化轴 × 满舵角速度 × 步长。
            # 数据标签约定 ω>0 = 右转；service 的 correction_deg 约定正值 = 左转，
            # 因此这里必须反号。漏掉这个符号会让模型每次转向都朝反方向执行。
            turn_deg = omega_to_delta_deg(w_rad, self.w_max, self.step_ms)
            self.n_sends += 1
            self.last_send = (round(x, 3), round(y, 3), round(turn, 3))
            if abs(turn_deg) > 1e-6:
                self.n_turn_sent += 1
            if not self.dry_run:
                self._send(x, y, turn, turn_deg)
            time.sleep(STEP_S)
