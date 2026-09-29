# -*- coding: utf-8 -*-
"""关键帧上的 4 自由度（x y z yaw）位姿图优化，roll/pitch 取 odom。

**来历**：本来是 `.tmp/nav_incremental_replay.py` 里的一个内部类。`Docs/回环优化耗时剖析
（2026-09-29）.md` 量出它单步要 6.9–11.3 s，而随插件运行的 `backend/nav_loop.py`
`LoopCloser` 只要 3.3–12.5 ms——慢的是这份 4 自由度原型，不是线上代码。
为了让「修好它」这件事能被回归测试钉住（`.tmp/` 是 gitignore 的，测试引用不到），
把它搬进仓库，并把残差函数向量化。

**为什么残差要向量化**：`least_squares` 只给了 `jac_sparsity` 没给解析雅可比，
所以 scipy 用有限差分，一次优化要调用残差约 722 次。原实现是纯 Python 逐边循环、
且每次调用都重算常量（`np.linalg.inv(Z)`、`_tilt`），单次 9.5 ms ⇒ 722 × 9.5 ≈ 6.9 s。
向量化后单次 0.157 ms，整体 7126 → 469 ms（÷15），残差与原实现最大绝对差 1.6e-13。

**用法**：``PoseGraph4(odom)`` → 按时间顺序 ``add_node`` → ``add_loop`` → ``optimize()``。
``residual_reference()`` 是原来的朴素实现，**只为对拍测试保留**，生产路径不要用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def yaw_of(T: np.ndarray) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def rz(a: float | np.ndarray) -> np.ndarray:
    """绕 z 轴旋转。标量给 3×3；数组给 (N,3,3)。"""
    if np.ndim(a) == 0:
        c, s = math.cos(float(a)), math.sin(float(a))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    out = np.zeros((len(a), 3, 3))
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    out[:, 2, 2] = 1.0
    return out


@dataclass
class _EdgeConst:
    """一次 optimize 内不变的边常量。节点/回环只增不减，所以每次 optimize 重算一遍即可
    （实测 1.3 ms），不必做增量缓存——缓存失效才是 bug 温床。"""

    edges: list[tuple[int, int, np.ndarray, float, float]]  # (a, b, Z, sigma_t, sigma_r)
    ea: np.ndarray            # (E,) 边起点在 ids 中的下标
    eb: np.ndarray            # (E,) 边终点下标
    Zi_R: np.ndarray          # (E,3,3) inv(Z) 的旋转部分
    Zi_t: np.ndarray          # (E,3)   inv(Z) 的平移部分
    st: np.ndarray            # (E,) 平移残差尺度
    sr: np.ndarray            # (E,) 旋转残差尺度
    tilt: np.ndarray          # (N,3,3) 每节点去掉 yaw 后的 roll/pitch（常量）
    fixed: np.ndarray         # (4,) 首节点固定，不参与优化


class PoseGraph4:
    """关键帧上的 4DoF 图优化。odom 边：相邻关键帧的 odom 相对位姿；回环边：Link.transform。

    节点必须按时间顺序、id 递增加入。``pose(k)`` 给下游；``optimize()`` 后 ``x`` 被就地更新。
    """

    def __init__(self, odom: dict[int, np.ndarray]) -> None:
        self.odom = odom
        self.ids: list[int] = []
        self.loops: list[tuple[int, int, np.ndarray]] = []
        self.x: dict[int, np.ndarray] = {}      # id → (x, y, z, yaw)
        self._tilts: dict[int, np.ndarray] = {}

    # ---- 常量 ----
    def _tilt(self, k: int) -> np.ndarray:
        """去掉 yaw 后剩下的 roll/pitch（固定用 odom 的）。odom 不变，所以算一次就缓存。"""
        t = self._tilts.get(k)
        if t is None:
            T = self.odom[k]
            t = rz(-yaw_of(T)) @ T[:3, :3]
            self._tilts[k] = t
        return t

    # ---- 查询 ----
    def pose(self, k: int, v: np.ndarray | None = None) -> np.ndarray:
        v = self.x[k] if v is None else v
        T = np.eye(4)
        T[:3, :3] = rz(v[3]) @ self._tilt(k)
        T[:3, 3] = v[:3]
        return T

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, k: object) -> bool:
        return k in self.x

    # ---- 构建 ----
    def add_node(self, k: int) -> None:
        if not self.ids:
            T = self.odom[k]
        else:
            p = self.ids[-1]
            T = self.pose(p) @ np.linalg.inv(self.odom[p]) @ self.odom[k]
        self.ids.append(k)
        self.x[k] = np.array([*T[:3, 3], yaw_of(T)])

    def add_loop(self, a: int, b: int, Z: np.ndarray) -> None:
        self.loops.append((a, b, Z))

    # ---- 残差 ----
    def prepare(self) -> _EdgeConst:
        edges: list[tuple[int, int, np.ndarray, float, float]] = []
        for p, q in zip(self.ids, self.ids[1:]):
            Z = np.linalg.inv(self.odom[p]) @ self.odom[q]
            d = float(np.linalg.norm(Z[:3, 3]))
            edges.append((p, q, Z, 0.02 + 0.03 * d, 0.01 + 0.01 * d))
        for a, b, Z in self.loops:
            edges.append((a, b, Z, 0.10, math.radians(3.0)))
        idx = {k: i for i, k in enumerate(self.ids)}
        Zi = np.linalg.inv(np.array([Z for _a, _b, Z, _st, _sr in edges]))
        return _EdgeConst(
            edges=edges,
            ea=np.array([idx[a] for a, _b, _Z, _st, _sr in edges], int),
            eb=np.array([idx[b] for _a, b, _Z, _st, _sr in edges], int),
            Zi_R=np.ascontiguousarray(Zi[:, :3, :3]),
            Zi_t=np.ascontiguousarray(Zi[:, :3, 3]),
            st=np.array([st for _a, _b, _Z, st, _sr in edges], float),
            sr=np.array([sr for _a, _b, _Z, _st, sr in edges], float),
            tilt=np.array([self._tilt(k) for k in self.ids]),
            fixed=np.asarray(self.x[self.ids[0]], float).copy(),
        )

    def _unpack(self, x: np.ndarray, fixed: np.ndarray) -> dict[int, np.ndarray]:
        v = {self.ids[0]: fixed}
        for i, k in enumerate(self.ids[1:]):
            v[k] = np.asarray(x, float)[4 * i:4 * i + 4]
        return v

    def residual(self, x: np.ndarray, const: _EdgeConst | None = None) -> np.ndarray:
        """向量化残差。每条边 4 项：平移 3 项 / st，yaw 误差 / sr。"""
        c = const if const is not None else self.prepare()
        V = np.empty((len(self.ids), 4))
        V[0] = c.fixed
        V[1:] = np.asarray(x, float).reshape(-1, 4)
        # 每个节点的完整旋转 R = rz(yaw) @ tilt
        Rfull = rz(V[:, 3]) @ c.tilt                    # (N,3,3)
        Ra, Rb = Rfull[c.ea], Rfull[c.eb]                # (E,3,3)
        # M = inv(Ta) @ Tb：R = RaᵀRb，t = Raᵀ(t_b − t_a)
        R = np.transpose(Ra, (0, 2, 1)) @ Rb
        t = np.einsum('eji,ej->ei', Ra, V[c.eb, :3] - V[c.ea, :3])
        # E = inv(Z) @ M
        ER = c.Zi_R @ R
        Et = np.einsum('eij,ej->ei', c.Zi_R, t) + c.Zi_t
        out = np.empty((len(c.st), 4))
        out[:, :3] = Et / c.st[:, None]
        out[:, 3] = np.arctan2(ER[:, 1, 0], ER[:, 0, 0]) / c.sr
        return out.ravel()

    def residual_reference(self, x: np.ndarray, const: _EdgeConst | None = None) -> np.ndarray:
        """朴素逐边实现，**只用于对拍测试**（单次约 9.5 ms @55 节点）。勿用于生产路径。"""
        c = const if const is not None else self.prepare()
        v = self._unpack(x, c.fixed)
        r: list[float] = []
        for a, b, Z, st, sr in c.edges:
            Ta, Tb = self.pose(a, v[a]), self.pose(b, v[b])
            E = np.linalg.inv(Z) @ (np.linalg.inv(Ta) @ Tb)
            r.extend(E[:3, 3] / st)
            r.append(math.atan2(E[1, 0], E[0, 0]) / sr)
        return np.asarray(r, float)

    # ---- 优化 ----
    def optimize(self, max_nfev: int = 50, reference_residual: bool = False) -> float:
        """4DoF 图优化，就地更新 ``x``，返回末值 cost。

        ``reference_residual=True`` 走朴素残差，只为测试对拍；生产路径不要开。
        ⚠️ ``max_nfev`` 别往下调：实测 50→20 省 63% 时间，但 cost 从 302 劣化到 551。
        """
        if not self.loops:
            return 0.0
        c = self.prepare()
        if reference_residual:
            res = lambda z: self.residual_reference(z, c)  # noqa: E731
        else:
            res = lambda z: self.residual(z, c)  # noqa: E731
        x0 = np.concatenate([self.x[k] for k in self.ids[1:]])
        idx = {k: i for i, k in enumerate(self.ids)}

        # 稀疏雅可比：每条边只牵涉两端节点的 4 个量，否则数值差分随节点数平方变慢。
        sp = lil_matrix((4 * len(c.edges), len(x0)), dtype=int)
        for e, (a, b, *_r) in enumerate(c.edges):
            for n_ in (a, b):
                if idx[n_] > 0:
                    sp[4 * e:4 * e + 4, 4 * (idx[n_] - 1):4 * idx[n_]] = 1

        sol = least_squares(res, x0, loss="huber", f_scale=3.0,
                            max_nfev=max_nfev, jac_sparsity=sp)
        for i, k in enumerate(self.ids[1:]):
            self.x[k] = np.array(sol.x[4 * i:4 * i + 4], float)
        return float(sol.cost)

    def status(self) -> dict[str, Any]:
        return {"nodes": len(self.ids), "loops": len(self.loops)}
