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
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

FREE, OCC, UNK = 178, 0, 89


class TerrainClass(IntEnum):
    """地图上单格的地面类别。

    为什么是**代价**而不是**类别**：机器人与地形的交互是连续的，"能不能过"
    是个连续量。Shaban et al.（CoRL 2022）与 RELLIS-OCC（2024）都落在四档代价上，
    而不是二值 free/not-free。二值化在这里的具体代价是：**正确地图与错误地图
    会给出同一个答案**（都"过不去"），于是下游任何以"能否通行"为判据的信号
    都没有梯度。⇒ 必须保留中间档。
    """

    UNKNOWN = 0     # 没有地面证据。**不是 free**（见 nav_grid 红线）
    FLAT = 1        # 地板
    STAIRS = 2      # 楼梯：可上可下，代价略高于平地
    WATER = 3       # 水：**可涉但昂贵**，不是障碍
    PIT = 4         # 坑 / 断崖：不可通行
    OBSTACLE = 5    # 实心障碍：不可通行


# 每类的**基准**代价（无量纲，乘算，不是米）。取值理由：
#   FLAT     = 1.0   基准
#   STAIRS   = 1.25  上下一级台阶有实打实的时间与打滑风险，但远没到"危险"
#   WATER    = 4.0   **关键**：刻意给成有限值而不是 inf。见下方 TERRAIN_HARD_BLOCK 注释。
#   PIT/OBC  = inf   真的过不去
# 上下都可以调，但**改这里等于改规划行为**，必须跑 tools/mapping_gate 看漂移。
TERRAIN_COST: dict[int, float] = {
    TerrainClass.FLAT: 1.0,
    TerrainClass.STAIRS: 1.25,
    TerrainClass.WATER: 4.0,
    TerrainClass.PIT: math.inf,
    TerrainClass.OBSTACLE: math.inf,
}

#: 置信度低时，代价往"保守"方向插值的目标值。
#: 为什么要这一项：网络/规则给出的类别必然有错，而"自信地判错"比"不确定"贵得多。
#: 让 cost' = lerp(cost, 保守值, 1-conf)，等价于给判错自动加价。
TERRAIN_CONSERVATIVE_COST = 8.0

#: 代价上限。inf 在 Dijkstra 里能收敛，但序列化/哈希/比较都不友好，
#: 而且"地图炸了"会变成 inf-inf=nan。封顶让异常可见又可诊断。
TERRAIN_COST_MAX = 1.0e3

#: 地形通道缺失时的策略。
#:   "lethal"  = 无地形证据一律按不可通行（本项目默认，保守）
#:   "inherit" = 忽略地形层，退回纯几何代价（**只用于 A/B**，见 Docs 诚实性约束）
TERRAIN_MISSING = ("lethal", "inherit")


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
        # ---- 地形层（S0）----
        # 三个通道默认**全为 None**。None 的含义是"这张图没有地形信息"，
        # 与"有地形信息但判成 UNKNOWN"是**两件不同的事**，必须能区分：
        # 前者让 _cost() 逐位退回改动前的行为（可用 tools/mapping_gate 证明），
        # 后者按 TERRAIN_MISSING 策略处理，默认禁行。
        self.terrain_class: np.ndarray | None = None   # uint8，TerrainClass 取值
        self.terrain_conf: np.ndarray | None = None     # float32 0..1，证据强度
        self.terrain_missing: str = "lethal"

    @property
    def has_terrain(self) -> bool:
        return self.terrain_class is not None

    def attach_terrain(self, klass: np.ndarray, conf: np.ndarray | None = None, *,
                       missing: str = "lethal") -> None:
        """挂上地面类别图。形状必须与三态图逐位相同，否则**抛错不做广播**——
        广播会静默把一张错位/错分辨率的类别图当成合法证据用，那种错查不出来。

        ``conf`` 缺省视为全 1（完全信任）。给低置信度是让代价自动向保守方向插值，
        这样"自信地判错"比"不确定"贵——见 TERRAIN_CONSERVATIVE_COST。
        """
        if missing not in TERRAIN_MISSING:
            raise ValueError(f"missing 只能是 {TERRAIN_MISSING}，收到 {missing!r}")
        k = np.asarray(klass)
        if k.shape != self.grid.shape:
            raise ValueError(f"terrain_class 形状 {k.shape} != 栅格 {self.grid.shape}")
        valid = {int(c) for c in TerrainClass}
        bad = sorted({int(v) for v in np.unique(k)} - valid)
        if bad:
            # 越界类别一律当 UNKNOWN，不 clamp 成最近的合法值：clamp 会把
            # "上游算错了"伪装成"上游判成某个合法类别"，那正是约束 1 禁止的。
            raise ValueError(f"terrain_class 含非法取值 {bad}（合法 {sorted(valid)}）")
        # ⚠️ 全部校验做完再赋值。曾经把 conf 的校验放在赋值之后，结果
        # attach(conf=NaN) 抛了错却把 terrain_class 留在了图上——调用方以为
        # "什么都没发生"，实际拿到一张半成品地形图，且 has_terrain 已经是 True。
        c: np.ndarray | None = None
        if conf is not None:
            c = np.asarray(conf, np.float32)
            if c.shape != self.grid.shape:
                raise ValueError(f"terrain_conf 形状 {c.shape} != 栅格 {self.grid.shape}")
            if not np.isfinite(c).all():
                raise ValueError("terrain_conf 含 NaN/Inf")
            c = np.clip(c, 0.0, 1.0)
        self.terrain_class = k.astype(np.uint8, copy=True)
        self.terrain_conf = None if c is None else c.astype(np.float32, copy=True)
        self.terrain_missing = missing

    def terrain_cost_map(self) -> np.ndarray | None:
        """把类别图 + 置信度烘成**纯地形代价**（不含几何项），供离线诊断与门使用。

        返回 None 表示这张图没有地形层。代价语义见 TERRAIN_COST。
        """
        if self.terrain_class is None:
            return None
        k = self.terrain_class
        out = np.full(k.shape, TERRAIN_COST_MAX, np.float32)
        for cls, base in TERRAIN_COST.items():
            if math.isfinite(base):
                out[k == int(cls)] = float(base)
        conf = (np.ones(k.shape, np.float32) if self.terrain_conf is None else self.terrain_conf)
        # 置信越低越往保守值插值。用 lerp 而不是乘法：inf 不能参与乘法
        # （0*inf=nan），而 PIT/OBSTACLE 恰好是 inf。
        out = out + (TERRAIN_CONSERVATIVE_COST - out) * (1.0 - conf)
        return np.minimum(out, TERRAIN_COST_MAX).astype(np.float32)

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
    def _passable(self) -> np.ndarray:
        """可通行掩码 = 中心区 **且** 代价没被顶到 C_max。

        为什么必须和 ``_cost()`` 一起看，而不是只看 ``center``：
        ``_line_ok`` 原来只查 ``center``，于是 ``_string_pull`` 会把 A* 好不容易
        绕开坑的 105 格路径，拉成一条**横穿坑**的 2 点直线——地形代价对搜索有效、
        对拉直无效，分数一样但结果是错的。实测：加了地形层后 A* 路径穿过坑 0 格，
        ``plan()`` 出来的折点却正好横跨坑。

        无地形层时 ``_cost()`` 与改动前逐位相同（``C_max`` 判据不会命中），
        所以这条掩码对既有行为是惰性的，可用 tools/mapping_gate 证明。
        """
        if self.terrain_class is None:
            return self.center
        return self.center & (self._cost() < TERRAIN_COST_MAX)
    def _cost(self) -> np.ndarray:
        # 贴墙加价（0.6 m 内线性到 ×3），让路径尽量走中间。
        base = np.where(self.center, 1.0 + 2.0 * np.clip(0.6 - self.clearance, 0, None) / 0.6, np.inf)
        if self.terrain_class is None:
            # 没有地形层 ⇒ **逐位退回**加地形之前的行为。这条分支是
            # tools/mapping_gate 证明"S0 没有动到既有规划"的依据，别在这里"顺手优化"。
            return base
        t = self.terrain_cost_map()
        if t is None:                       # 不可达，但显式写出来比隐含假设好
            return base
        if self.terrain_missing == "inherit":
            # 只在有类别证据的格上用地形代价；UNKNOWN 格退回几何代价。
            known = self.terrain_class != int(TerrainClass.UNKNOWN)
            return np.where(self.center & known, base * np.minimum(t, TERRAIN_COST_MAX), base)
        # 默认 "lethal"：UNKNOWN 与 PIT/OBSTACLE 一样，按"实际不可通行"处理。
        # ⚠️ 注意这里给的是**有限的 C_max，不是 inf**。这是刻意的：
        #   ① inf 参与任何算术（序列化、哈希、inf-inf）都会变成 nan，难诊断；
        #   ② 硬阻断会让"正确地图"和"错误地图"给出同一个结果（都过不去），
        #      下游以能否通行为判据的信号就没有梯度了——见 TerrainClass 的 docstring。
        #   代价：**地形层本身不硬阻断路径**，只把那条路变贵约 1000 倍。
        #   真正"不走"的保证要由规划侧的**路径代价预算**给出，那是 S1 的事，
        #   属于导航行为，本次不动。
        # 这里**不**用 base * t ——base 在非中心格是 inf，inf * 0 会出 nan。
        out = np.where(self.center, np.minimum(base * np.minimum(t, TERRAIN_COST_MAX),
                                               TERRAIN_COST_MAX), np.inf)
        return out

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
        return bool(self._passable()[r, c].all())

    def _string_pull(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_ok(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out
