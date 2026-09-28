# -*- coding: utf-8 -*-
"""定位融合（OSC 航位推算 + 视觉重定位）与路径跟随（pure pursuit）。

依据 run7 对 run6 地图的离线重定位（2026-09-27，单次运行）：
* 单条重定位给出的对齐，水平偏离中位 0.45 m、p90 1.7 m，偏航 p90 13°，离群约 37%；
* 最长 24 m 没有任何重定位；OSC 航位推算漂移约 2.8% 路程。
所以重定位**不能直接当位置用**：这里把每条重定位变成一个"odom→map 对齐样本"，
在最近一段路程的窗口里取稳健中位、按门限剔除离群，没有新样本时沿用上次对齐，
不确定度按路程线性增长。跟随器按不确定度决定走、减速还是停下等定位。

坐标：odom 系与导航系都是右手、朝向从 +x 逆时针（弧度）。调用方负责把 OSC/HMD
的 yaw 约定换成这个（在线 yaw 符号见 ``pose_math`` 的未决项）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _compose(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """SE(2)：a ∘ b，均为 (x, y, theta)。"""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def _inverse(a: Sequence[float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap(-a[2]))


@dataclass
class FuserConfig:
    window_m: float = 10.0          # 只用最近这么多路程内的重定位样本
    max_samples: int = 15
    gate_m: float = 1.0             # 离群门限（与 3σ 取大）
    gate_yaw_rad: float = math.radians(15.0)
    init_agree_m: float = 0.8       # 初始化：至少 2 个样本互相落在这个范围内
    init_agree_yaw_rad: float = math.radians(10.0)
    sigma_floor_m: float = 0.25
    drift_per_m: float = 0.028


class RelocFuser:
    def __init__(self, cfg: FuserConfig | None = None) -> None:
        self.cfg = cfg or FuserConfig()
        self._samples: list[tuple[float, tuple[float, float, float]]] = []   # (odom 路程, 对齐)
        self._align: tuple[float, float, float] | None = None
        self._align_sigma = 0.0
        self._align_dist = 0.0      # 对齐最近一次被有效样本支撑时的路程
        self._odom: tuple[float, float, float] | None = None
        self._dist = 0.0
        self.rejected = 0
        self.accepted = 0

    def predict(self, odom_pose: Sequence[float], dist_m: float) -> None:
        """odom 位姿（绝对）与累计路程（OSC，世界米，单调）。"""
        self._odom = (float(odom_pose[0]), float(odom_pose[1]), float(odom_pose[2]))
        self._dist = max(self._dist, float(dist_m))

    def add_reloc(self, map_pose: Sequence[float], odom_pose_at: Sequence[float],
                  dist_at_m: float) -> bool:
        """一条重定位：该帧在地图里的位姿，以及**同一时刻**的 odom 位姿与路程。

        返回是否被采纳（初始化前的样本先缓存，返回 False）。
        """
        align = _compose(map_pose, _inverse(odom_pose_at))
        cfg = self.cfg
        if self._align is not None and self._odom is not None:
            pred = _compose(self._align, self._odom)
            obs = _compose(align, self._odom)
            gate = max(cfg.gate_m, 3.0 * self.sigma_m())
            if (math.hypot(obs[0] - pred[0], obs[1] - pred[1]) > gate
                    or abs(wrap(obs[2] - pred[2])) > max(cfg.gate_yaw_rad, 3.0 * self._yaw_sigma())):
                self.rejected += 1
                return False
        self._samples.append((float(dist_at_m), align))
        self._samples = self._samples[-cfg.max_samples:]
        return self._refit()

    def _window(self) -> list[tuple[float, float, float]]:
        lo = self._dist - self.cfg.window_m
        return [a for d, a in self._samples if d >= lo]

    def _refit(self) -> bool:
        if self._odom is None:
            return False
        cfg = self.cfg
        win = self._window()
        if not win:
            return False
        # 在"当前 odom 位姿映射到地图"这一层比较，和 odom 原点无关（杠杆臂自动计入）。
        pts = np.array([_compose(a, self._odom) for a in win])
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        cth = math.atan2(np.median(np.sin(pts[:, 2])), np.median(np.cos(pts[:, 2])))
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        dy = np.abs([wrap(t - cth) for t in pts[:, 2]])
        if self._align is None:
            inl = (d <= cfg.init_agree_m / 2) & (dy <= cfg.init_agree_yaw_rad)
            if inl.sum() < 2:
                return False
        else:
            inl = (d <= cfg.gate_m) & (dy <= cfg.gate_yaw_rad)
            if not inl.any():
                return False
        p = pts[inl]
        cur = (float(np.median(p[:, 0])), float(np.median(p[:, 1])),
               math.atan2(float(np.mean(np.sin(p[:, 2]))), float(np.mean(np.cos(p[:, 2])))))
        spread = float(np.median(np.hypot(p[:, 0] - cur[0], p[:, 1] - cur[1])))
        self._align = _compose(cur, _inverse(self._odom))
        self._align_sigma = max(cfg.sigma_floor_m, 1.4826 * spread / math.sqrt(len(p)) + cfg.sigma_floor_m)
        self._align_dist = self._dist
        self.accepted += 1
        return True

    def _yaw_sigma(self) -> float:
        return math.radians(3.0) + self.cfg.drift_per_m * max(0.0, self._dist - self._align_dist) / 5.0

    def sigma_m(self) -> float:
        return self._align_sigma + self.cfg.drift_per_m * max(0.0, self._dist - self._align_dist)

    def estimate(self) -> dict[str, Any]:
        if self._align is None or self._odom is None:
            return {"state": "unknown", "reason": "not_relocalized"}
        x, y, th = _compose(self._align, self._odom)
        return {"state": "localized", "xy": (x, y), "theta": th, "sigma_m": self.sigma_m(),
                "since_reloc_m": self._dist - self._align_dist}

@dataclass
class FollowConfig:
    lookahead_m: float = 0.8
    max_speed: float = 1.0          # OSC 前进量上限（0..1）
    slow_speed: float = 0.4
    max_turn_rate: float = math.radians(90.0)   # rad/s
    turn_in_place_rad: float = math.radians(50.0)
    goal_tol_m: float = 0.3
    off_path_m: float = 0.6         # 估计位置离路径超过这个就要求重规划
    slow_sigma_m: float = 0.5
    stop_sigma_m: float = 0.9
    scan_turn_rate: float = math.radians(30.0)  # 等定位时原地慢转，换视角找回环


class PathFollower:
    """输出 {state, forward, turn_rate, reason}；``forward`` 是 OSC 前进轴（0..1），
    ``turn_rate`` 是期望偏航角速度（rad/s，逆时针为正）。不发命令、不读栅格，
    只吃 ``NavGrid.plan`` 的路径合约——执行层自己决定怎么变成 OSC/HMD。"""

    def __init__(self, waypoints: Sequence[Sequence[float]], cfg: FollowConfig | None = None) -> None:
        self.cfg = cfg or FollowConfig()
        self.path = np.asarray(waypoints, float).reshape(-1, 2)
        if len(self.path) < 1:
            raise ValueError("空路径")
        seg = np.diff(self.path, axis=0)
        self._seg_len = np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.zeros(0)
        self._cum = np.concatenate([[0.0], np.cumsum(self._seg_len)])
        self._progress = 0.0        # 沿路径弧长，只增不减，防止回头

    @property
    def length_m(self) -> float:
        return float(self._cum[-1])

    def _project(self, p: np.ndarray) -> tuple[float, float]:
        """返回 (弧长, 横向距离)，只在当前进度前后一段里找，避免折返路段串线。"""
        best = (self._progress, math.inf)
        for i, L in enumerate(self._seg_len):
            if self._cum[i + 1] < self._progress - 1.0 or self._cum[i] > self._progress + 3.0:
                continue
            a = self.path[i]
            t = 0.0 if L <= 0 else float(np.clip(np.dot(p - a, self.path[i + 1] - a) / (L * L), 0, 1))
            d = float(np.hypot(*(a + t * (self.path[i + 1] - a) - p)))
            if d < best[1]:
                best = (float(self._cum[i] + t * L), d)
        if len(self._seg_len) == 0:
            best = (0.0, float(np.hypot(*(self.path[0] - p))))
        return best

    def _point_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length_m)
        i = int(np.searchsorted(self._cum, s, side="right") - 1)
        i = min(i, len(self._seg_len) - 1)
        if i < 0:
            return self.path[0].copy()
        L = self._seg_len[i]
        t = 0.0 if L <= 0 else (s - self._cum[i]) / L
        return self.path[i] + t * (self.path[i + 1] - self.path[i])

    def step(self, est: dict[str, Any], *, local_stop: bool = False) -> dict[str, Any]:
        cfg = self.cfg
        if local_stop:
            return {"state": "stopped", "forward": 0.0, "turn_rate": 0.0, "reason": "local_obstacle"}
        if est.get("state") != "localized":
            return {"state": "wait_localization", "forward": 0.0, "turn_rate": cfg.scan_turn_rate,
                    "reason": est.get("reason", "localization_unknown")}
        p = np.asarray(est["xy"], float)
        sigma = float(est.get("sigma_m", 0.0))
        if float(np.hypot(*(self.path[-1] - p))) <= cfg.goal_tol_m:
            return {"state": "arrived", "forward": 0.0, "turn_rate": 0.0, "reason": "goal_reached"}
        if sigma > cfg.stop_sigma_m:
            return {"state": "wait_localization", "forward": 0.0, "turn_rate": cfg.scan_turn_rate,
                    "reason": "localization_uncertain", "sigma_m": sigma}
        s, lateral = self._project(p)
        self._progress = max(self._progress, s)
        if lateral > cfg.off_path_m:
            return {"state": "replan", "forward": 0.0, "turn_rate": 0.0, "reason": "off_path",
                    "lateral_m": lateral}
        target = self._point_at(self._progress + cfg.lookahead_m)
        err = wrap(math.atan2(target[1] - p[1], target[0] - p[0]) - float(est["theta"]))
        # 角速度按 pure pursuit 曲率 2·sin(α)/L 换算，再限幅。
        turn = float(np.clip(2.0 * math.sin(err) / cfg.lookahead_m * cfg.max_speed,
                             -cfg.max_turn_rate, cfg.max_turn_rate))
        if abs(err) > cfg.turn_in_place_rad:
            return {"state": "turning", "forward": 0.0, "turn_rate": math.copysign(cfg.max_turn_rate, err),
                    "reason": "heading_error", "sigma_m": sigma}
        speed = cfg.slow_speed if sigma > cfg.slow_sigma_m else cfg.max_speed
        speed *= max(0.3, math.cos(err))
        remain = self.length_m - self._progress
        speed = min(speed, max(0.25, remain / 1.0))
        return {"state": "following", "forward": float(speed), "turn_rate": turn,
                "reason": "ok", "sigma_m": sigma, "lateral_m": lateral,
                "progress_m": self._progress}
