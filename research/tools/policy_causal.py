# -*- coding: utf-8 -*-
"""因果版 [wp]/[topo]：只用 t_obs **之前**的轨迹，不碰未来。

为什么必须另写一套（2026-11-05）：
  旧的 waypoint_body() 在 t_obs 处沿**整条** DR 轨迹往前取点，
  near 模式取 min(0.8m, 1.2s行程)，far 取 4m —— 两者都在用未来。
  topo 的 lateral 用的是「当前点到该边弦」的横向距离，那条弦的终点也在未来。

本模块提供的替代量，全部满足 causal(t) <= data(<=t)：
    [path]  past_curve   最近 CURVE_M 米内的平均转角（弧度，右正）
            past_lateral 最近 CURVE_M 米内，当前点到**已走过**那段的弦的横向偏差
            past_prog    最近 CURVE_M 米内已走过的弧长（饱和）
            past_speed   最近 1 s 的平均速度
这些量在部署时都能算（只需要历史位姿/速度），所以可以当真机输入。
"""
from __future__ import annotations

import numpy as np

CURVE_M = 6.0        # 回看窗口（米），与旧 DEFAULT_EDGE_M 同量级
SPEED_WIN_S = 1.0    # 速度回看窗口（秒）


def _at_s(path: dict, sq: float):
    """沿弧长取点（只在 [0, s_now] 内取，调用方保证 sq <= s_now）。"""
    s = path["s"]
    p = path["p"]
    if len(s) < 2:
        return p[0]
    sq = float(max(0.0, min(sq, s[-1])))
    m = int(np.searchsorted(s, sq, side="right") - 1)
    m = max(0, min(m, len(s) - 2))
    ds = s[m + 1] - s[m]
    f = 0.0 if ds <= 1e-9 else (sq - s[m]) / ds
    return p[m] + f * (p[m + 1] - p[m])


def causal_topo(t_obs: float, path: dict, fwd: np.ndarray,
                curve_m: float = CURVE_M) -> tuple[float, float, float]:
    """返回 (prog, lateral, remain)，**三者都只依赖 t_obs 之前的轨迹**。

    prog    = 回看窗口内已走过的弧长（0..curve_m）
    lateral = 当前点到「窗口起点→当前点」弦的有符号横向距离（右正）
    remain  = 0.0（因果口径下没有「离边终点还有多远」这种东西）
    """
    t = path["t"]
    s = path["s"]
    if len(t) < 2 or t_obs < t[0] or t_obs > t[-1]:
        return (0.0, 0.0, 0.0)
    j = int(np.searchsorted(t, t_obs, side="right") - 1)
    j = max(0, min(j, len(t) - 1))
    s_now = float(s[j])
    s_from = max(0.0, s_now - float(curve_m))
    p_now = _at_s(path, s_now)
    p_from = _at_s(path, s_from)
    d = p_now - p_from
    fx, fz = float(fwd[0]), float(fwd[1])
    # 右向 (fz, -fx)：与 waypoint_body 的约定一致
    lateral = float(d[0] * fz + d[1] * -fx)
    prog = float(s_now - s_from)
    return (round(prog, 4), round(lateral, 4), 0.0)


def past_speed(t_obs: float, streams: dict, win_s: float = SPEED_WIN_S) -> float:
    """最近 win_s 秒的平均速度（只用 <= t_obs 的样本）。"""
    t = streams["t_m"]
    vx = streams["vx_m"]
    vz = streams["vz_m"]
    if len(t) == 0:
        return 0.0
    hi = int(np.searchsorted(t, t_obs, side="right"))
    lo = int(np.searchsorted(t, t_obs - win_s, side="left"))
    if hi <= lo:
        return 0.0
    sp = np.hypot(vx[lo:hi], vz[lo:hi])
    return float(sp.mean())
