# -*- coding: utf-8 -*-
"""三态栅格 → 可行走中心区 → A* + 拉直 → 路径合约。

输入是 ``.tmp/cloud_to_grid.py`` 输出的 PGM（178 free / 0 obstacle / 89 unknown）
和同名 ``*_map.json``（原点、分辨率、行序）。原型 ``.tmp/navmesh_proto.py`` 靠轨迹
对齐搜索找原点，这里直接读 json，不再猜。

坐标：**导航系** = RTAB-Map 地图系的 xy 平面（z 朝上），单位世界米
（追踪米 × world_scale）。朝向 = 从 +x 逆时针的弧度。

红线（与原型一致）：
* unknown 永远不当 free；可走 = 观测 free ∪ 走过的走廊，且不压观测障碍；
* 中心点离"非可走"（障碍 + 未知 + 图外）至少 avatar 半径；
* 起点/终点不在中心区、跨区域都拒绝，并给出原因，执行器只吃合约不读栅格。
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

FREE, OCC, UNK = 178, 0, 89


@dataclass
class GridMeta:
    resolution_m: float          # 追踪米 / 格
    origin_xy_m: tuple[float, float]
    world_scale: float = 0.755

    @property
    def cell_world_m(self) -> float:
        return self.resolution_m * self.world_scale


@dataclass
class PathContract:
    accepted: bool
    reason: str
    waypoints_xy_m: list[list[float]] = field(default_factory=list)
    length_m: float = 0.0
    corridor_half_width_m: float = 0.0
    start_snapped_m: float = 0.0
    frame: str = "nav_map_xy_world_m"

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason, "frame": self.frame,
                "waypoints_xy_m": self.waypoints_xy_m, "length_m": round(self.length_m, 3),
                "corridor_half_width_m": round(self.corridor_half_width_m, 3),
                "start_snapped_m": round(self.start_snapped_m, 3)}


class NavGrid:
    def __init__(self, grid: np.ndarray, meta: GridMeta) -> None:
        g = np.asarray(grid, np.uint8)
        bad = ~np.isin(g, (FREE, OCC, UNK))
        if bad.any():
            # 非三态值（插值/压缩伪影）一律按 unknown，不猜成 free。
            g = g.copy()
            g[bad] = UNK
        self.grid = g
        self.meta = meta
        self.center = np.zeros_like(g, bool)
        self.labels = np.zeros(g.shape, np.int32)
        self.clearance = np.zeros(g.shape, np.float32)   # 世界米，到非可走
        self.radius_m = 0.0

    @classmethod
    def load(cls, pgm: str | Path) -> "NavGrid":
        pgm = Path(pgm)
        side = pgm.with_suffix(".json")
        if not side.exists():
            raise FileNotFoundError(f"{side} 不存在：栅格必须带原点，不接受对齐搜索")
        info = json.loads(side.read_text(encoding="utf-8"))
        g = cv2.imread(str(pgm), cv2.IMREAD_UNCHANGED)
        if g is None:
            raise FileNotFoundError(str(pgm))
        if g.shape != (int(info["rows"]), int(info["cols"])) or info.get("row0") != "max_y":
            raise ValueError(f"{pgm} 与 {side.name} 的尺寸或行序不一致")
        # 不回落默认值：world_scale 随 avatar 变，猜一个会把整张图按错的比例换成世界米。
        if "world_scale" not in info:
            raise ValueError(f"{side.name} 缺 world_scale")
        return cls(g, GridMeta(float(info["resolution_m"]),
                               (float(info["origin_xy_m"][0]), float(info["origin_xy_m"][1])),
                               float(info["world_scale"])))

    # ---- 坐标 ----
    def to_cell(self, xy_world: Sequence[float]) -> tuple[int, int]:
        m, s = self.meta, self.meta.world_scale
        c = int(math.floor((xy_world[0] / s - m.origin_xy_m[0]) / m.resolution_m))
        row_from_bottom = int(math.floor((xy_world[1] / s - m.origin_xy_m[1]) / m.resolution_m))
        return self.grid.shape[0] - 1 - row_from_bottom, c

    def to_world(self, rc: Sequence[int]) -> tuple[float, float]:
        m, s = self.meta, self.meta.world_scale
        x = m.origin_xy_m[0] + (rc[1] + 0.5) * m.resolution_m
        y = m.origin_xy_m[1] + (self.grid.shape[0] - 1 - rc[0] + 0.5) * m.resolution_m
        return x * s, y * s

    def inside(self, rc: Sequence[int]) -> bool:
        return 0 <= rc[0] < self.grid.shape[0] and 0 <= rc[1] < self.grid.shape[1]

    # ---- 建可走区 ----
    def build(self, radius_m: float = 0.25, walked: Iterable[np.ndarray] = (),
              walked_half_m: float = 0.20) -> dict[str, Any]:
        """``walked``：导航系世界米的折线列表，**调用方已用 OSC 门控切掉位姿跳变**。"""
        g, cw = self.grid, self.meta.cell_world_m
        occ, free_obs = g == OCC, g == FREE
        walked_mask = np.zeros(g.shape, np.uint8)
        line = np.zeros(g.shape, np.uint8)
        thick = max(1, int(round(2 * walked_half_m / cw)))
        for poly in walked:
            pts = np.array([self.to_cell(p)[::-1] for p in np.asarray(poly, float)], np.int32)
            if len(pts) >= 2:
                cv2.polylines(walked_mask, [pts.reshape(-1, 1, 2)], False, 1, thick)
                cv2.polylines(line, [pts.reshape(-1, 1, 2)], False, 1, 1)
        passable = (free_obs | (walked_mask > 0)) & ~occ
        # pad 一圈 0：图外与未知、障碍一样算"非可走"。
        dt = cv2.distanceTransform(np.pad(passable, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1] * cw
        d_occ = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * cw
        center = dt >= radius_m
        # 身体实际走过的中心线就是"装得下"的证据，但离观测障碍仍需 ≥ 半径。
        center |= (line > 0) & passable & (d_occ >= radius_m)
        n, labels = cv2.connectedComponents(center.astype(np.uint8), connectivity=8)
        self.center, self.labels, self.clearance, self.radius_m = center, labels, dt, float(radius_m)
        areas = np.bincount(labels.ravel(), minlength=n)[1:] * cw * cw
        total = float(center.sum()) * cw * cw
        return {"cell_world_m": round(cw, 4), "avatar_radius_m": radius_m,
                "observed_free_m2": round(float(free_obs.sum()) * cw * cw, 1),
                "obstacle_m2": round(float(occ.sum()) * cw * cw, 1),
                "walkable_center_m2": round(total, 1), "regions": int(n - 1),
                "largest_region_share": round(float(areas.max()) / total, 3) if total else 0.0}

    # ---- 查询 ----
    def classify(self, xy_world: Sequence[float]) -> str:
        """free_center / too_narrow / obstacle / unknown / outside。"""
        rc = self.to_cell(xy_world)
        if not self.inside(rc):
            return "outside"
        if self.center[rc]:
            return "free_center"
        v = self.grid[rc]
        return "obstacle" if v == OCC else ("unknown" if v == UNK else "too_narrow")

    def snap_to_center(self, xy_world: Sequence[float], max_m: float) -> tuple[tuple[int, int], float] | None:
        rc = self.to_cell(xy_world)
        if self.inside(rc) and self.center[rc]:
            return rc, 0.0
        k = int(math.ceil(max_m / self.meta.cell_world_m))
        r0, c0 = max(0, rc[0] - k), max(0, rc[1] - k)
        win = self.center[r0:rc[0] + k + 1, c0:rc[1] + k + 1]
        cand = np.argwhere(win)
        if not len(cand):
            return None
        d = np.hypot(cand[:, 0] + r0 - rc[0], cand[:, 1] + c0 - rc[1]) * self.meta.cell_world_m
        i = int(np.argmin(d))
        if d[i] > max_m:
            return None
        return (int(cand[i, 0] + r0), int(cand[i, 1] + c0)), float(d[i])

    def _unknown_near(self, xy_world: Sequence[float], r_world: float) -> bool:
        """目标周围一个身体半径内有 unknown：多半是边缘外还没看清，不能吸附进来。"""
        r0, c0 = self.to_cell(xy_world)
        k = int(math.ceil(r_world / self.meta.cell_world_m))
        h, w = self.grid.shape
        win = self.grid[max(0, r0 - k):min(h, r0 + k + 1), max(0, c0 - k):min(w, c0 + k + 1)]
        return win.size == 0 or bool((win == UNK).any())

    def plan(self, start_xy: Sequence[float], goal_xy: Sequence[float],
             start_snap_m: float = 0.5, goal_snap_m: float = 0.5) -> PathContract:
        """起点允许吸附到最近中心格（定位噪声）。终点只在"已观测但太窄/障碍"时吸附 ≤ goal_snap_m
        （点在墙边、窄处很常见）；终点在 unknown / 图外必须拒绝，不猜。"""
        if self.radius_m <= 0:
            raise RuntimeError("先调用 build()")
        goal_kind = self.classify(goal_xy)
        if goal_kind in ("too_narrow", "obstacle") and not self._unknown_near(goal_xy, self.radius_m):
            gs = self.snap_to_center(goal_xy, goal_snap_m)
            if gs is None:
                return PathContract(False, f"goal_{goal_kind}")
            goal_xy = self.to_world(gs[0])
        elif goal_kind != "free_center":
            return PathContract(False, f"goal_{goal_kind}")
        snapped = self.snap_to_center(start_xy, start_snap_m)
        if snapped is None:
            return PathContract(False, f"start_{self.classify(start_xy)}")
        s, snap_d = snapped
        t = self.to_cell(goal_xy)
        if self.labels[s] != self.labels[t]:
            return PathContract(False, "goal_disconnected", start_snapped_m=snap_d)
        path = self._astar(s, t)
        if path is None:
            return PathContract(False, "no_path", start_snapped_m=snap_d)
        pulled = self._string_pull(path)
        pts = [list(self.to_world(p)) for p in pulled]
        pts[-1] = [float(goal_xy[0]), float(goal_xy[1])]
        length = float(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        return PathContract(True, "ok", [[round(v, 3) for v in p] for p in pts], length,
                            float(min(self.clearance[p] for p in path)), snap_d)

    def segment_clear(self, a_xy: Sequence[float], b_xy: Sequence[float]) -> bool:
        return self._line_ok(self.to_cell(a_xy), self.to_cell(b_xy))

    # ---- 内部 ----
    def _cost(self) -> np.ndarray:
        # 贴墙加价（0.6 m 内线性到 ×3），让路径尽量走中间。
        return np.where(self.center, 1.0 + 2.0 * np.clip(0.6 - self.clearance, 0, None) / 0.6, np.inf)

    def _astar(self, s: tuple[int, int], t: tuple[int, int]) -> list[tuple[int, int]] | None:
        h, w = self.grid.shape
        cost = self._cost().ravel()
        g = np.full(h * w, np.inf)
        prev = np.full(h * w, -1, np.int64)
        si, ti = s[0] * w + s[1], t[0] * w + t[1]
        g[si] = 0.0
        pq = [(0.0, si)]
        nb = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
        closed = np.zeros(h * w, bool)
        while pq:
            _, cur = heapq.heappop(pq)
            if closed[cur]:
                continue
            closed[cur] = True
            if cur == ti:
                out = [cur]
                while prev[out[-1]] >= 0:
                    out.append(int(prev[out[-1]]))
                return [(i // w, i % w) for i in reversed(out)]
            r, c = divmod(cur, w)
            for dr, dc, d in nb:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                ni = nr * w + nc
                if closed[ni] or not math.isfinite(cost[ni]):
                    continue
                ng = g[cur] + d * cost[ni]
                if ng < g[ni]:
                    g[ni], prev[ni] = ng, cur
                    heapq.heappush(pq, (ng + math.hypot(nr - t[0], nc - t[1]), ni))
        return None

    def _line_ok(self, a: Sequence[int], b: Sequence[int]) -> bool:
        if not (self.inside(a) and self.inside(b)):
            return False
        # 两处都不能省：
        # 1) 步数要用 L1（|dr|+|dc|）。切比雪夫步数下每步位移可达 √2 格。
        # 2) 取整必须用 floor 而不是 round：round 取的是**最近**格，不是线段
        #    **所在**格；to_cell() 本身就是 floor 口径，只有 floor 才与它自洽。
        #    L1 步数保证每步 |Δr|+|Δc| == 1，floor 后逐样本必 8-连通，不会整格
        #    跳过（随机 4000 组 dr/dc∈[0,60] 全覆盖；只加 L1 而保留 round 仍漏 ~37%，
        #    典型反例 (5,5)→(9,6) 会漏掉真正压过的 (8,5)）。
        #    +1e-9 让正好落在整数坐标上的端点不被浮点误差掉到上一格。
        n = int(abs(b[0] - a[0]) + abs(b[1] - a[1])) + 1
        r = np.floor(np.linspace(a[0], b[0], n) + 1e-9).astype(int)
        c = np.floor(np.linspace(a[1], b[1], n) + 1e-9).astype(int)
        return bool(self.center[r, c].all())

    def _string_pull(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_ok(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out
