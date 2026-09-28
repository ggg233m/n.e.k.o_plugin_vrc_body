# -*- coding: utf-8 -*-
"""run_motion.py —— 从 run 目录读「运动量」：OSC 路程 + HMD 航向。

给拓扑建图补两类**非视觉**证据：
  1. **走了多远**（OSC 速度按真实报文间隔积分）——用来判定「两个簇之间是否真的
     离开过又回来」，把 c4↔c6 这种「同一次访问内的相似视角」剔除。
  2. **转了多少度**（HMD 四元数 yaw 解卷绕）——用来把「站着不动只转头」和
     「走到新地方」区分开，给过长的簇（c22 吞了 26 s）提供切分依据。

时间基准：OSC / HMD 的 `t` 都是 `time.monotonic()`，与 `run.json` 里
`events[].obs_start_monotonic` 同一把钟。所以
    telemetry_t = t - anchor
    telemetry_t = video_file_t + video_timebase.offset_s
两边必须对得上，本模块只管 telemetry 秒。

## 已知坑（别踩）
- OSC 只有**回传遥测**，给不了世界坐标；世界位置必须用「本地速度按 HMD yaw
  旋进世界」积分。`VelocityX/Z` 是 avatar **本地系**（README 已实机验证）：
  +Z 前进、+X 右侧横移。
  ⚠️ 早期这里写过「`VelocityX` 恒为 0（一路向前＝本地 +Z）」，**那是错的** ——
  本 run 实测横移 `|dvx|` 累计 1.2205 m，出现在 596/895 步上。正因为横移不是
  恒 0，航位推算必须用 `disp()` 的**矢量**位移，不能用 `dist()` 的标量路程
  （旧实现混用造成终点差 1.5496 m）。
- `hypot(Vx,Vz)` 是**路程速度**（与朝向无关），语义是"走了多远"，
  只用于回环判定与导航边长度，**不能当位移用**。
- 首个 Velocity 报文之前是 **NaN**（实测晚 0.906 s），必须**掩码**，
  否则积分起点会凭空多出一段。
- `AngularY` **不参与**朝向积分（是指令回声，已证伪）——转向一律用 HMD yaw。
- 旋转公式**不在本文件**：唯一实现是 `backend/pose_math.py`。本模块只负责读遥测与积分，不做旋转。
"""
from __future__ import annotations

import json
import os

import numpy as np


def load_run_json(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        return json.load(f)


def load_anchor(run_dir: str):
    """obs_start_monotonic（在 events[] 里，不是顶层键）。缺失返回 None。"""
    rj = load_run_json(run_dir)
    for ev in rj.get("events", []) or []:
        if isinstance(ev, dict) and ev.get("obs_start_monotonic") is not None:
            return float(ev["obs_start_monotonic"])
    return rj.get("obs_start_monotonic")


def load_offset(run_dir: str):
    """video_timebase.offset_s（遥测 = 视频文件时间 + offset）。缺失返回 None。"""
    return (load_run_json(run_dir).get("video_timebase") or {}).get("offset_s")


def _series(run_dir: str, fname: str, pick):
    """通用：读 jsonl → (t_rel, value) 已按时间排序。pick(row) 返回 float 或 None。"""
    anchor = load_anchor(run_dir)
    if anchor is None:
        raise ValueError("run.json 缺少 obs_start_monotonic，无法对齐时间轴")
    ts, vs = [], []
    with open(os.path.join(run_dir, fname), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            v = pick(r)
            if v is None:
                continue
            ts.append(float(r["t"]) - anchor)
            vs.append(float(v))
    order = np.argsort(ts) if ts else np.array([], dtype=int)
    return np.array(ts)[order], np.array(vs)[order]


class OscPath:
    """OSC 速度 → 累计路程 D(telemetry_t)，并暴露**报文新鲜度**。

    ## 为什么必须记新鲜度（勿简化）
    VRChat 的速度回传是**变化驱动**的：**静止不发包，匀速也不发包**。所以
    「一段时间内没有报文」既可能是站住不动，也可能是持续匀速——**不能一律判成静止**。

    ⚠️ **不要**再写「零阶保持会把长空档外推成一大段位移」——**那是反的**，已实测证伪
    （2026-09-25，`Docs/SLAM米制复测报告（2026-09-24）.md` §3.1）：
    本 run 31 个空档里，「前速度≠0 且后速度=0」的**一个都没有**；而**超过 1 秒的空档
    全部是"已经停住"的**（前速度全是 0），零阶保持在这些段上外推 `0 × Δt = 0`，**恰好正确**。
    需要外推非零速度的空档全部 ≤ 0.875 s。

    所以：**空档 ⟺ 速度不变（含不变为 0）**，`zoh` 是协议语义、不是"乐观口径"；
    `stop` 是**已知错误的下界**（少算 24.03 m / 37%），只作为保守对照保留。

    因此本类同时给出：
      * ``dist(t0, t1, policy="zoh")``  —— 保持最后速度跨空档（**协议语义，默认**）
      * ``dist(t0, t1, policy="stop")`` —— 空档（>GAP_S）按 0 增量（**已知错误的下界**）
      * ``max_gap_in(t0, t1)``          —— 区间内最大报文间隔（秒）
      * ``coverage_frac(t0, t1)``       —— 被"有报文覆盖"的时长占比
    调用方必须**显式选口径**并如实上报新鲜度，不得静默取一种。

    ⚠️ **已知风险（未修）**：通道没有心跳，所以「合法匀速静默」与「丢了一个 0 包」
    在信息上完全等价，**不可判定**。**不要用超时阈值去猜** —— 那会把匀速长走误判成停住。
    详见同文档 §3.2。
    """

    #: 报文间隔超过该值(秒)即视为空档（无观测）。
    GAP_S = 0.25

    def __init__(self, run_dir: str):
        vx_t, vx = _series(run_dir, "osc.jsonl",
                           lambda r: r["args"][0]
                           if r.get("addr", "").endswith("/VelocityX") else None)
        vz_t, vz = _series(run_dir, "osc.jsonl",
                           lambda r: r["args"][0]
                           if r.get("addr", "").endswith("/VelocityZ") else None)
        if len(vz_t) == 0:
            raise ValueError("osc.jsonl 里没有 VelocityZ，算不出路程")
        # ZOH：在每条报文处用「另一通道最近一次已知值」补齐；首个报文前留 NaN → 掩码
        t = vz_t
        vx_at = np.interp(t, vx_t, vx, left=np.nan, right=vx[-1]) if len(vx_t) else np.zeros_like(vz)
        vx_use = np.nan_to_num(vx_at, nan=0.0)
        speed = np.hypot(vx_use, vz)
        dt = np.diff(t)
        good = dt > 0
        # 口径 A：ZOH —— 保持最后速度跨空档（旧行为，保持向后兼容）
        step = np.concatenate([[0.0], np.where(good, speed[:-1] * np.where(good, dt, 0.0), 0.0)])
        # 口径 B：stop —— 空档（>GAP_S）按 0 增量，只累加"有报文覆盖"的位移
        covered = good & (dt <= self.GAP_S)
        step_stop = np.concatenate([[0.0], np.where(covered, speed[:-1] * np.where(covered, dt, 0.0), 0.0)])

        def _cum(comp, mask):
            """矢量的某一分量按 mask 累加。步长用**步首**速度，与 step 完全同构 ⇒
            |disp(单步)| == dist(单步)，两条口径的模长严格一致。"""
            return np.cumsum(np.concatenate(
                [[0.0], np.where(mask, comp[:-1] * np.where(mask, dt, 0.0), 0.0)]))

        self.t = t
        self.dt = dt
        self.cum = np.cumsum(step)
        self.cum_stop = np.cumsum(step_stop)
        # 矢量位移的累积分量（供 disp() 用）
        self.cum_vx = _cum(vx_use, good)
        self.cum_vz = _cum(vz, good)
        self.cum_vx_stop = _cum(vx_use, covered)
        self.cum_vz_stop = _cum(vz, covered)
        self.total = float(self.cum[-1]) if len(self.cum) else 0.0
        self.total_stop = float(self.cum_stop[-1]) if len(self.cum_stop) else 0.0
        self.max_gap = float(np.max(dt)) if len(dt) else 0.0
        self.n_holes = int(np.sum(dt > self.GAP_S)) if len(dt) else 0

    # ---- 距离（必须显式选口径） -------------------------------------------
    def dist(self, t0: float, t1: float, policy: str = "zoh") -> float:
        """[t0,t1] 区间走过的路程（米）。

        policy="zoh"：跨空档保持最后速度（乐观，旧口径，默认以兼容既有调用）。
        policy="stop"：空档按 0 增量（保守）。
        """
        if t1 <= t0:
            return 0.0
        cum = self.cum_stop if policy == "stop" else self.cum
        return float(np.interp(t1, self.t, cum) - np.interp(t0, self.t, cum))

    # ---- 位移矢量（航位推算必须用这个） -----------------------------------
    def disp(self, t0: float, t1: float, policy: str = "zoh") -> tuple[float, float]:
        """[t0,t1] 区间内的 **avatar 本地位移矢量** (Δx_local, Δz_local)，米。

        与 ``dist()`` 的区别（别混用）：
          * ``dist`` = ∫|v|dt —— **路程**，标量。语义是"走了多远"，
            用于回环判定（是否真的离开过又回来）、导航图边长度等。
          * ``disp`` = ∫v dt —— **位移矢量**，保留方向。航位推算必须用它：
            avatar 本地速度 `(vx, vz)` 只有先保留方向、再按朝向旋进世界，
            才等价于在线航位推算（`backend/nav_online.py`）的口径。

        旧实现用 `dist` 沿 yaw 走，等于把**横移当成了前进** —— 只在 vx≈0 时正确。
        本 run 实测该偏差造成终点差 1.5496 m（横移 `|dvx|` 累计 1.2205 m，
        出现在 596/895 步上 —— 不是"横移很少见"，而是每一步都有一点点）。

        旋转本身不在这里：唯一实现是 `backend/pose_math.py`。
        """
        if t1 <= t0:
            return (0.0, 0.0)
        cx = self.cum_vx_stop if policy == "stop" else self.cum_vx
        cz = self.cum_vz_stop if policy == "stop" else self.cum_vz
        return (float(np.interp(t1, self.t, cx) - np.interp(t0, self.t, cx)),
                float(np.interp(t1, self.t, cz) - np.interp(t0, self.t, cz)))

    # ---- 新鲜度 -----------------------------------------------------------
    def _breakpoints(self, t0: float, t1: float):
        """[t0,t1] 内外的断点序列：端点 + 落在区间内的报文时刻。"""
        inner = self.t[(self.t > t0) & (self.t < t1)]
        return np.concatenate([[t0], inner, [t1]]) if t1 > t0 else np.array([t0])

    def max_gap_in(self, t0: float, t1: float) -> float:
        """区间内最大报文间隔（秒）。端点也计入 → 完全没有报文时就是区间长度。"""
        if t1 <= t0:
            return 0.0
        bp = self._breakpoints(t0, t1)
        return float(np.max(np.diff(bp))) if len(bp) > 1 else 0.0

    def coverage_frac(self, t0: float, t1: float) -> float:
        """区间内"有报文覆盖"（相邻间隔 <= GAP_S）的时长占比，∈[0,1]。"""
        if t1 <= t0:
            return 1.0
        bp = self._breakpoints(t0, t1)
        if len(bp) < 2:
            return 0.0
        d = np.diff(bp)
        return float(np.sum(d[d <= self.GAP_S]) / (t1 - t0))

    def freshness(self, t0: float, t1: float) -> dict:
        """一次取全：口径落差 + 新鲜度，供上游如实上报（不做静默选择）。"""
        z = self.dist(t0, t1, "zoh")
        s = self.dist(t0, t1, "stop")
        return {
            "duration_s": round(float(t1 - t0), 4),
            "dist_zoh_m": round(z, 4),
            "dist_stop_m": round(s, 4),
            "dist_spread_m": round(z - s, 4),
            "max_gap_s": round(self.max_gap_in(t0, t1), 4),
            "coverage_frac": round(self.coverage_frac(t0, t1), 4),
        }


class HmdYaw:
    """HMD 四元数 → 解卷绕累计 yaw（度）。"""

    def __init__(self, run_dir: str):
        t, q = [], []
        with open(os.path.join(run_dir, "hmd_frames.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rot = (r.get("hmd") or {}).get("rotation_xyzw")
                if not rot:
                    continue
                t.append(float(r["t"]))
                q.append(rot)
        anchor = load_anchor(run_dir)
        if anchor is None:
            raise ValueError("run.json 缺少 obs_start_monotonic，无法对齐时间轴")
        t = np.array(t) - anchor
        q = np.array(q, dtype=float)
        if len(t) == 0:
            raise ValueError("hmd_frames.jsonl 为空")
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = np.degrees(np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z)))
        u = np.zeros_like(yaw)
        u[0] = yaw[0]
        for i in range(1, len(yaw)):
            u[i] = u[i - 1] + ((yaw[i] - yaw[i - 1] + 180.0) % 360.0 - 180.0)
        self.t = t
        self.yaw = u

    def turn(self, t0: float, t1: float) -> float:
        """[t0,t1] 之间的**净**转角（度，带符号）。"""
        return float(np.interp(t1, self.t, self.yaw) - np.interp(t0, self.t, self.yaw))

    def abs_turn(self, t0: float, t1: float) -> float:
        """[t0,t1] 之间累计**绝对**转角（度）——来回摇头也会计入。"""
        m = (self.t >= t0) & (self.t <= t1)
        if m.sum() < 2:
            return 0.0
        return float(np.abs(np.diff(self.yaw[m])).sum())


def main():
    import argparse
    ap = argparse.ArgumentParser(description="查看某个 run 的运动量")
    ap.add_argument("--run", required=True)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    a = ap.parse_args()
    p = OscPath(a.run)
    y = HmdYaw(a.run)
    print("run          : %s" % a.run)
    print("遥测时长     : %.1f s" % (p.t[-1] - p.t[0]))
    print("OSC 总路程   : %.2f m" % p.total)
    print("HMD 累计转角 : %.1f°" % y.abs_turn(p.t[0], p.t[-1]))
    if a.t0 is not None and a.t1 is not None:
        print("[%.2f, %.2f] 路程 %.2f m，净转角 %+.1f°，绝对转角 %.1f°"
              % (a.t0, a.t1, p.dist(a.t0, a.t1), y.turn(a.t0, a.t1),
                 y.abs_turn(a.t0, a.t1)))


if __name__ == "__main__":
    main()
