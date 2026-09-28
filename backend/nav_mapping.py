# -*- coding: utf-8 -*-
"""关键帧 → 三态栅格的**增量**建图（首访世界边走边建，navmesh 随之长大）。

与离线 ``.tmp/cloud_to_grid.py`` 是同一套判定，只是按关键帧累积：
* 每个关键帧存自己 base 系（x 前 y 左 z 上，追踪米）下的局部点，不存地图系坐标；
* 栅格化时用**当前**位姿把点投进地图系——回环后位姿变了，下一次栅格化自然跟着变，
  不会把旧位姿下的格子残留在图里；
* 高度相对**观测它的那个关键帧**：h = z_map − z_node + cam_h，头部俯仰、轨迹 z
  漂移都不进入判定（RTAB-Map 自带栅格在 base 系判高，run6 上 3 m 处障碍 39%）；
* 三态：free = 有地面点且非障碍；obstacle = 障碍点 ≥ min_pts 且 3×3 邻居支撑；
  其余 unknown。不做射线追踪，free 只来自真正看见的地面，unknown 永不当 free。

栅格原点对齐到分辨率整数倍，所以地图变大时已有格子的编号含义不变。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .nav_grid import FREE, OCC, UNK, GridMeta, NavGrid


@dataclass
class MapperConfig:
    res_m: float = 0.10             # 追踪米；0.05 在 run6 上碎成很多小岛
    range_m: float = 3.0            # 离观测关键帧的水平距离上限（追踪米）
    ground_tol_m: float = 0.30
    obst_top_m: float = 2.0
    min_pts: int = 3
    pad_m: float = 1.0
    cam_h_band: tuple[float, float] = (1.2, 2.6)   # 地面候选：相机下方这个范围
    cam_h_default: float = 1.73     # 候选太少时的回退值（run6 实测 1.728）
    world_scale: float = 0.755
    # 入图时按关键帧局部系体素化（带计数），栅格化只处理体素中心：run6 每帧约 1 万点 → 2.5 千体素。
    # xy 取栅格的一半，量化误差 ≤ 2.5 cm；z 2 cm，远小于 ground_tol。
    vox_xy_m: float = 0.05
    vox_z_m: float = 0.02
    # 增量栅格化：每个关键帧的格计数算一次，回环只按整格平移挪下标（见 _shift）。
    # cam_h 按 1 cm 直方图取中位，变化超过 cam_h_hyst_m 才换（换了要全部重分类）。
    cam_h_hyst_m: float = 0.02
    # 单个关键帧的地面可能整体偏离全局 cam_h（低头/蹲下/HMD 高度抖、视差系统误差）：run6 有 5 帧偏 0.3–0.4 m，
    # 地面被整片判成障碍，正是"走过的地方变障碍、点了不动"的来源。按该帧 1.2 m 外地面的高度众数修正，
    # 限幅 ±ground_offset_max_m；众数不够突出（< 25% 点）就不修。
    ground_offset_max_m: float = 0.5
    ground_offset_min_range_m: float = 1.2
    # 障碍点数还须 ≥ 地面点数 × 这个比例：单帧视差噪声打出的几个高点压不过几十帧看到的地面。
    occ_ground_ratio: float = 0.3   # run6：路径上的障碍格 25→12


def _voxelize(p: np.ndarray, xy_m: float, z_m: float) -> tuple[np.ndarray, np.ndarray]:
    """N×3 → (M×3 体素中心, M 点数)。键打包成 int64 再一维 unique，比 unique(axis=0) 快一个量级。"""
    if not len(p):
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    q = np.floor(p / np.array([xy_m, xy_m, z_m], np.float32)).astype(np.int64)
    q = np.clip(q, -(1 << 20), (1 << 20) - 1) + (1 << 20)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    u, cnt = np.unique(key, return_counts=True)
    m = (1 << 21) - 1
    idx = np.column_stack([(u >> 42) & m, (u >> 21) & m, u & m]) - (1 << 20)
    centers = (idx.astype(np.float32) + 0.5) * np.array([xy_m, xy_m, z_m], np.float32)
    return centers, cnt.astype(np.int32)


def _ground_offset(h: np.ndarray, cnt: np.ndarray, rxy: np.ndarray, lim: float, min_r: float) -> float:
    """关键帧地面相对全局 cam_h 的偏移（追踪米）：±lim 内 2 cm 直方图、10 cm 平滑后的众数。"""
    sel = (np.abs(h) <= lim + 0.1) & (np.hypot(rxy[:, 0], rxy[:, 1]) >= min_r)
    tot = float(cnt[sel].sum())
    if tot < 300:
        return 0.0
    n = int((2 * lim + 0.2) / 0.02) + 1
    b = np.clip(((h[sel] + lim + 0.1) / 0.02).astype(np.int64), 0, n - 1)
    sm = np.convolve(np.bincount(b, cnt[sel], minlength=n), np.ones(5), "same")
    i = int(np.argmax(sm))
    if sm[i] < 0.25 * tot:
        return 0.0
    return float(np.clip((i + 0.5) * 0.02 - lim - 0.1, -lim, lim))


def make_sgbm() -> Any:
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5,
                                 P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def stereo_disparity(left_gray: np.ndarray, right_gray: np.ndarray, matcher: Any = None) -> np.ndarray:
    """左目视差（像素，float32）；无效处 ≤ 0。"""
    m = matcher if matcher is not None else make_sgbm()
    return m.compute(left_gray, right_gray).astype(np.float32) / 16.0


def stereo_points(left_gray: np.ndarray, right_gray: np.ndarray, *, fx: float, cx: float,
                  cy: float, baseline_m: float, max_range_m: float = 3.5,
                  step: int = 2, matcher: Any = None, disp: np.ndarray | None = None) -> np.ndarray:
    """校正好的双目 → base 系点（N×3，追踪米）。光学系 z 前 x 右 y 下 → base x 前 y 左 z 上。
    ``disp`` 给了就不再算一遍视差（回环检测要同一张视差图）。"""
    if disp is None:
        disp = stereo_disparity(left_gray, right_gray, matcher)
    d = disp[::step, ::step]
    v, u = np.mgrid[0:disp.shape[0]:step, 0:disp.shape[1]:step]
    ok = d > 1.0
    z = fx * baseline_m / d[ok]
    keep = z <= max_range_m
    z = z[keep]
    x = (u[ok][keep] - cx) * z / fx
    y = (v[ok][keep] - cy) * z / fx
    return np.column_stack([z, -x, -y]).astype(np.float32)


class KeyframeGridMapper:
    def __init__(self, cfg: MapperConfig | None = None) -> None:
        self.cfg = cfg or MapperConfig()
        self._pts: dict[int, np.ndarray] = {}       # 体素中心（关键帧 base 系，追踪米）
        self._cnt: dict[int, np.ndarray] = {}       # 每个体素里的原始点数
        self._pose: dict[int, np.ndarray] = {}
        self._osc: dict[int, float] = {}
        self._trail: dict[int, list[tuple[np.ndarray, np.ndarray, float | None]]] = {}
        self._trail_arr: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cam_h = self.cfg.cam_h_default
        # 缓存：_rot[k] = (R·p 的 xy, R·p 的 z = 相对关键帧的高度, 点数, 算它用的 R)；
        # 只改平移时高度不变，分类也不变。_base[k] = (ix, iy, n_ground, n_obst, 算它用的 t_xy, cam_h)。
        self._rot: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._base: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self._hist: dict[int, np.ndarray] = {}
        lo, hi = self.cfg.cam_h_band
        self._hist_sum = np.zeros(int(round((hi - lo) / 0.01)), np.int64)
        # 全局格计数累加器（[ix − lo_x, iy − lo_y]）。每个关键帧记下它当前加进去的 (base, 整格平移)，
        # 变了就先减旧的再加新的：新关键帧只算自己，回环只挪整格，不重投点。
        self._acc_g: np.ndarray | None = None
        self._acc_o: np.ndarray | None = None
        self._acc_lo = (0, 0)
        self._acc_cam_h: float | None = None
        self._applied: dict[int, tuple[tuple, int, int]] = {}
        self._dirty: set[int] = set()

    def __len__(self) -> int:
        return len(self._pts)

    def add_keyframe(self, node_id: int, points_base: np.ndarray, pose_map_base: np.ndarray,
                     osc_dist_m: float | None = None) -> None:
        """``osc_dist_m``：该关键帧时刻的 OSC 累计路程（世界米）；给了才能门控走过的走廊。"""
        k = int(node_id)
        p = np.asarray(points_base, np.float32).reshape(-1, 3)
        p = p[np.hypot(p[:, 0], p[:, 1]) <= self.cfg.range_m]
        self._pts[k], self._cnt[k] = _voxelize(p, self.cfg.vox_xy_m, self.cfg.vox_z_m)
        self._pose[k] = np.asarray(pose_map_base, np.float64).reshape(4, 4)
        self._drop_cache(k)
        if osc_dist_m is not None:
            self._osc[k] = float(osc_dist_m)

    def _drop_cache(self, k: int) -> None:
        self._rot.pop(k, None)
        self._base.pop(k, None)
        h = self._hist.pop(k, None)
        if h is not None:
            self._hist_sum -= h

    def _rotated(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = self._pose[k][:3, :3]
        c = self._rot.get(k)
        if c is not None and (c[3] is R or np.array_equal(c[3], R)):
            return c
        self._drop_cache(k)
        q = self._pts[k] @ R.T.astype(np.float32)
        c = (q[:, :2].copy(), q[:, 2].copy(), self._cnt[k], R.copy())
        self._rot[k] = c
        lo, hi = self.cfg.cam_h_band
        band = (c[1] < -lo) & (c[1] > -hi)
        b = np.clip(((-c[1][band] - lo) / 0.01).astype(np.int64), 0, len(self._hist_sum) - 1)
        h = np.bincount(b, c[2][band], minlength=len(self._hist_sum)).astype(np.int64)
        self._hist[k] = h
        self._hist_sum += h
        return c

    def _update_cam_h(self) -> None:
        tot = int(self._hist_sum.sum())
        if tot < 200:
            return
        cw = np.cumsum(self._hist_sum)
        est = self.cfg.cam_h_band[0] + (int(np.searchsorted(cw, tot / 2.0)) + 0.5) * 0.01
        if abs(est - self.cam_h) > self.cfg.cam_h_hyst_m:
            self.cam_h = est

    def _kf_base(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """关键帧 k 在"算它那一刻的平移 t0"下的格计数 (ix, iy, n_ground, n_obst, t0, cam_h)。
        朝向或 cam_h 变了才重算；只有平移变了就按整格平移复用（见 ``_shift``）。"""
        c = self.cfg
        cached = self._base.get(k)
        if cached is not None and cached[5] == self.cam_h:
            return cached
        t = self._pose[k][:2, 3].copy()
        rxy, rel, cnt, _R = self._rot[k]
        h = rel + np.float32(self.cam_h)
        h = h - np.float32(_ground_offset(h, cnt, rxy, c.ground_offset_max_m, c.ground_offset_min_range_m))
        g = np.abs(h) <= c.ground_tol_m
        o = (h > c.ground_tol_m) & (h < c.obst_top_m)
        sel = g | o
        if not sel.any():
            z = np.zeros(0, np.int64)
            out = (z, z, z, z, t, self.cam_h)
            self._base[k] = out
            return out
        xy = rxy[sel].astype(np.float64) + t
        ix = np.floor(xy[:, 0] / c.res_m).astype(np.int64)
        iy = np.floor(xy[:, 1] / c.res_m).astype(np.int64)
        # 关键帧只覆盖 range_m 半径的小方块：在局部稠密方块里 bincount 去重，比 np.unique（排序）快一个量级。
        x0, y0 = int(ix.min()), int(iy.min())
        bw = int(iy.max()) - y0 + 1
        flat = (ix - x0) * bw + (iy - y0)
        size = (int(ix.max()) - x0 + 1) * bw
        w = cnt[sel]
        ng = np.bincount(flat, w * g[sel], minlength=size)
        no = np.bincount(flat, w * o[sel], minlength=size)
        nz = np.flatnonzero((ng > 0) | (no > 0))
        out = (nz // bw + x0, nz % bw + y0, ng[nz], no[nz], t, self.cam_h)
        self._base[k] = out
        return out

    def _shift(self, k: int, base: tuple) -> tuple[int, int]:
        """当前平移相对 base 的 t0 挪了多少整格。整格平移与逐点重投最多差一格，而逐点重投本身
        也要量化到格；回环（853 帧全体平移）从逐帧重投 ~1 s 变成只挪下标。"""
        d = (self._pose[k][:2, 3] - base[4]) / self.cfg.res_m
        return int(round(float(d[0]))), int(round(float(d[1])))

    def _sync_acc(self) -> None:
        """全局累加器只处理 (base, 整格平移) 变了的关键帧：旧贡献负权、新贡献正权，一次 bincount。"""
        if self._acc_cam_h != self.cam_h:
            self._acc_g = self._acc_o = None
            self._applied.clear()
            self._acc_cam_h = self.cam_h
        work: list[tuple[tuple, int, int, float]] = []
        for k in self._pts:
            base = self._kf_base(k)
            sx, sy = self._shift(k, base)
            prev = self._applied.get(k)
            if prev is not None and prev[0] is base and prev[1] == sx and prev[2] == sy:
                continue
            if prev is not None and len(prev[0][0]):
                work.append((*prev, -1.0))
            if len(base[0]):
                work.append((base, sx, sy, 1.0))
            self._applied[k] = (base, sx, sy)
        for k in [k for k in self._applied if k not in self._pts]:
            prev = self._applied.pop(k)
            if len(prev[0][0]):
                work.append((*prev, -1.0))
        if not work:
            return
        ix = np.concatenate([b[0] + sx for b, sx, _sy, _g in work])
        iy = np.concatenate([b[1] + sy for b, _sx, sy, _g in work])
        wg = np.concatenate([b[2] * sg for b, _sx, _sy, sg in work])
        wo = np.concatenate([b[3] * sg for b, _sx, _sy, sg in work])
        self._grow_acc(int(ix.min()), int(iy.min()), int(ix.max()), int(iy.max()))
        (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
        flat = (iy - y0) * W + (ix - x0)
        self._acc_g += np.bincount(flat, wg, minlength=H * W).reshape(H, W)
        self._acc_o += np.bincount(flat, wo, minlength=H * W).reshape(H, W)

    def _grow_acc(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        m = 64                                   # 每次多留 6.4 m，边走边长不必每帧重分配
        if self._acc_g is None:
            x0, y0 = xmin - m, ymin - m
            shape = (ymax + 1 + m - y0, xmax + 1 + m - x0)
            self._acc_g, self._acc_o, self._acc_lo = np.zeros(shape), np.zeros(shape), (x0, y0)
            return
        (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
        if xmin >= x0 and ymin >= y0 and xmax < x0 + W and ymax < y0 + H:
            return
        nx0 = xmin - m if xmin < x0 else x0
        ny0 = ymin - m if ymin < y0 else y0
        nx1 = xmax + 1 + m if xmax >= x0 + W else x0 + W
        ny1 = ymax + 1 + m if ymax >= y0 + H else y0 + H
        for name in ("_acc_g", "_acc_o"):
            new = np.zeros((ny1 - ny0, nx1 - nx0))
            new[y0 - ny0:y0 - ny0 + H, x0 - nx0:x0 - nx0 + W] = getattr(self, name)
            setattr(self, name, new)
        self._acc_lo = (nx0, ny0)

    def add_trail(self, node_id: int, pose_map_frame: np.ndarray, osc_dist_m: float | None = None) -> None:
        """关键帧之间的逐帧位姿（地图系，追踪米）挂到最近的前一个关键帧上，存成相对位姿，
        回环后跟着关键帧走。关键帧 1 Hz 时一跳可达 4 m，只靠关键帧连线门控会被尺度误差误断。"""
        k = int(node_id)
        if k not in self._pose:
            return
        rel = np.linalg.inv(self._pose[k]) @ np.asarray(pose_map_frame, np.float64).reshape(4, 4)
        self._trail.setdefault(k, []).append((rel[:2, 3].copy(), rel[:2, :2].copy(),
                                              None if osc_dist_m is None else float(osc_dist_m)))
        self._trail_arr.pop(k, None)

    def _trail_xy_d(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        c = self._trail_arr.get(k)
        if c is None:
            tr = self._trail.get(k, ())
            c = (np.array([t for t, _r, _d in tr], float).reshape(-1, 2),
                 np.array([np.nan if d is None else d for _t, _r, d in tr], float))
            self._trail_arr[k] = c
        return c

    def walked(self, *, gate_m: float = 0.15, gate_frac: float = 0.05,
               max_hop_m: float = 1.5) -> list[np.ndarray]:
        """关键帧 + 逐帧轨迹（按 id 顺序，当前位姿）→ 走过的折线，导航系世界米。

        双目看不到脚下 ~1.6 m 以内的地面，身体实际走过的地方只能靠这条链当通行证据。
        相邻两点的 SLAM 位移超过 OSC 位移 ×(1+gate_frac) + gate_m 就断开（位姿跳变不能画成走廊）；
        没有 OSC 路程时退回 max_hop_m（追踪米）硬门限。"""
        s = self.cfg.world_scale
        xs: list[np.ndarray] = []
        ds: list[np.ndarray] = []
        for k in sorted(self._pose):
            T = self._pose[k]
            txy, td = self._trail_xy_d(k)
            xs.append(T[:2, 3][None])
            ds.append(np.array([self._osc.get(k, np.nan)], float))
            if len(txy):
                xs.append(T[:2, 3] + txy @ T[:2, :2].T)
                ds.append(td)
        if not xs:
            return []
        p = np.concatenate(xs) * s
        d = np.concatenate(ds)
        step = np.hypot(*(p[1:] - p[:-1]).T)
        dd = d[1:] - d[:-1]                        # 任一端没有 OSC 路程 → NaN → 退回硬门限
        limit = np.where(np.isnan(dd), max_hop_m * s, np.nan_to_num(dd) * (1 + gate_frac) + gate_m)
        cuts = np.flatnonzero(~(step <= limit)) + 1
        return [q for q in np.split(p, cuts) if len(q) >= 2]

    def update_poses(self, poses: dict[int, np.ndarray]) -> float:
        """回环优化后整体换位姿；返回关键帧的最大平移变化（追踪米），便于记录。"""
        moved = 0.0
        for k, T in poses.items():
            if k in self._pose:
                T = np.asarray(T, np.float64).reshape(4, 4)
                moved = max(moved, float(np.linalg.norm(T[:3, 3] - self._pose[k][:3, 3])))
                self._pose[k] = T
        return moved

    def anchor(self, xy_track: tuple[float, float]) -> tuple[int, tuple[float, float]] | None:
        """把地图系一点挂到最近的关键帧上（该关键帧水平系下的偏移）。回环后地图系会动，
        目标用 ``resolve`` 跟着关键帧走，而不是钉死在旧坐标上。"""
        if not self._pose:
            return None
        p = np.asarray(xy_track, float)
        k = min(self._pose, key=lambda i: float(np.hypot(*(self._pose[i][:2, 3] - p))))
        T = self._pose[k]
        yaw = math.atan2(T[1, 0], T[0, 0])
        d = p - T[:2, 3]
        c, s = math.cos(yaw), math.sin(yaw)
        return k, (c * d[0] + s * d[1], -s * d[0] + c * d[1])

    def resolve(self, anc: tuple[int, tuple[float, float]]) -> tuple[float, float] | None:
        k, (lx, ly) = anc
        T = self._pose.get(k)
        if T is None:
            return None
        yaw = math.atan2(T[1, 0], T[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        return float(T[0, 3] + c * lx - s * ly), float(T[1, 3] + s * lx + c * ly)

    def rasterize(self, extra_xy: np.ndarray | None = None) -> NavGrid:
        """当前全部关键帧 → NavGrid（未 build）。``extra_xy``：还要包进图里的点（追踪米，如当前位姿）。

        增量：全局累加器只更新变了的关键帧——新关键帧投一次；回环只改平移，按整格挪下标；
        朝向或 cam_h 变了才重投。输出只裁剪累加器，不再每次拼接全部关键帧的格。"""
        c = self.cfg
        for k in self._pts:
            self._rotated(k)
        self._update_cam_h()
        self._sync_acc()
        pts = []
        if self._acc_g is not None:
            seen = (self._acc_g > 0.5) | (self._acc_o > 0.5)
            rows, cols = np.flatnonzero(seen.any(axis=1)), np.flatnonzero(seen.any(axis=0))
            if len(rows):
                x0, y0 = self._acc_lo
                pts.append(np.array([[x0 + cols[0], y0 + rows[0]], [x0 + cols[-1], y0 + rows[-1]]], float)
                           * c.res_m + 0.5 * c.res_m)
        anchors = [T[:2, 3] for T in self._pose.values()]
        if extra_xy is not None:
            anchors += list(np.asarray(extra_xy, float).reshape(-1, 2))
        if anchors:
            pts.append(np.array(anchors, float))
        allxy = np.vstack(pts) if pts else np.zeros((1, 2))
        # 原点对齐到 res 整数倍：图长大时旧格子编号含义不变，调试时两帧可以直接逐格比。
        lo_i = np.floor((allxy.min(axis=0) - c.pad_m) / c.res_m).astype(np.int64)
        lo = lo_i * c.res_m
        hi = allxy.max(axis=0) + c.pad_m
        w, h = (np.ceil((hi - lo) / c.res_m).astype(int) + 1)
        n_g = np.zeros((h, w))
        n_o = np.zeros((h, w))
        if self._acc_g is not None:
            (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
            lx, ly = int(lo_i[0]), int(lo_i[1])
            ox0, oy0 = max(lx, x0), max(ly, y0)
            ox1, oy1 = min(lx + w, x0 + W), min(ly + h, y0 + H)
            if ox1 > ox0 and oy1 > oy0:
                n_g[oy0 - ly:oy1 - ly, ox0 - lx:ox1 - lx] = self._acc_g[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
                n_o[oy0 - ly:oy1 - ly, ox0 - lx:ox1 - lx] = self._acc_o[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
        # 权重 = 原始点数，min_pts 语义不变；累加器是整数加减，用 ±0.5 比较免得 1e-12 级残差翻转。
        occ = ((n_o > c.min_pts - 0.5) & (n_o >= c.occ_ground_ratio * n_g)).astype(np.uint8)
        occ &= (cv2.filter2D(occ, -1, np.ones((3, 3), np.float32)) >= 3).astype(np.uint8)
        g = np.full((h, w), UNK, np.uint8)
        g[(n_g > 0.5) & (occ == 0)] = FREE
        g[occ == 1] = OCC
        return NavGrid(g[::-1].copy(), GridMeta(c.res_m, (float(lo[0]), float(lo[1])), c.world_scale))


@dataclass
class Frontier:
    goal_xy_m: tuple[float, float]      # 导航系世界米，保证是可走中心格、与起点同区
    path_dist_m: float                  # 起点沿中心区的测地距离
    size_m: float                       # 边界长度（格数 × 格宽）

    def to_dict(self) -> dict[str, Any]:
        return {"goal_xy_m": [round(self.goal_xy_m[0], 3), round(self.goal_xy_m[1], 3)],
                "path_dist_m": round(self.path_dist_m, 2), "size_m": round(self.size_m, 2)}


def _geodesic(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """中心区上的 8 邻域 Dijkstra（格为单位）；不可达为 inf。"""
    import heapq

    dist = np.full(mask.shape, np.inf, np.float32)
    dist[start] = 0.0
    pq = [(0.0, start)]
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    H, W = mask.shape
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc, w in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and d + w < dist[rr, cc]:
                dist[rr, cc] = d + w
                heapq.heappush(pq, (d + w, (rr, cc)))
    return dist


def frontiers(ng: NavGrid, start_xy: tuple[float, float], *, min_size_m: float = 0.5,
              chunk_m: float = 1.5, reach_m: float = 0.6, start_snap_m: float = 0.5,
              limit: int = 8) -> list[Frontier]:
    """探索目标：观测 free 里紧挨 unknown 的边界，目标格取**可走中心区**里离边界 ≤ reach_m、
    且与起点连通的格。目标本身永远不在 unknown 里——走过去以后前面的 unknown 才会被看清。

    视野扇形的边界通常整圈连成一个分量；按 chunk_m 的方块切段，每段各出一个目标，
    否则只剩离起点最近的那一段（往往是身后），前方的边界永远不会被选中。"""
    snapped = ng.snap_to_center(start_xy, start_snap_m)
    if snapped is None:
        return []
    s, _ = snapped
    cw = ng.meta.cell_world_m
    g = ng.grid
    unk = (g == UNK).astype(np.uint8)
    touch = cv2.dilate(unk, np.ones((3, 3), np.uint8)) > 0
    edge = (g == FREE) & touch
    n, lab = cv2.connectedComponents(edge.astype(np.uint8), connectivity=8)
    reach = ng.center & (ng.labels == ng.labels[s])
    geo = _geodesic(reach, s)
    k = max(1, int(math.ceil(reach_m / cw)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    sizes = np.bincount(lab.ravel(), minlength=n)
    # 段 = (分量, 方块)；太碎的段（方块切角剩下的几格）并不单独出目标。
    blk = max(1, int(round(chunk_m / cw)))
    rr_e, cc_e = np.nonzero(lab > 0)
    keys = lab[rr_e, cc_e].astype(np.int64) * 1_000_000 + (rr_e // blk) * 1000 + (cc_e // blk)
    best: dict[tuple[int, int], Frontier] = {}
    for key in np.unique(keys):
        comp = int(key // 1_000_000)
        sel = keys == key
        if sizes[comp] * cw < min_size_m or sel.sum() * cw < min(min_size_m, chunk_m / 3):
            continue
        m = np.zeros(g.shape, np.uint8)
        m[rr_e[sel], cc_e[sel]] = 1
        cand = (cv2.dilate(m, ker) > 0) & np.isfinite(geo)
        if not cand.any():
            continue
        rr, cc = np.nonzero(cand)
        j = int(np.argmin(geo[rr, cc]))
        rc = (int(rr[j]), int(cc[j]))
        f = Frontier(ng.to_world(rc), float(geo[rc]) * cw, float(sel.sum()) * cw)
        if rc not in best or f.size_m > best[rc].size_m:     # 两段落到同一个目标格只留一个
            best[rc] = f
    out = sorted(best.values(), key=lambda f: f.path_dist_m)
    return out[:limit]


class NavSession:
    """增量地图上的"去某处 / 探索"会话：每次地图更新（新关键帧或回环）都重新栅格化、
    重新规划、换一个新的跟随器。

    * 目标挂在最近关键帧上（``KeyframeGridMapper.anchor``），回环挪动地图系后目标跟着走；
    * 探索模式下目标是 frontier，每次更新重选，但 ``keep_m`` 内还有 frontier 就不换，免得来回摆；
    * 估计位姿 ``est`` 与 ``PathFollower.step`` 同格式（导航系世界米），由调用方给——在线建图时
      就是 OSC 门控过的当前 SLAM 位姿，回环后它和地图一起跳。
    """

    def __init__(self, mapper: KeyframeGridMapper, *, radius_m: float = 0.25,
                 keep_m: float = 1.0, follow_cfg: Any = None,
                 request_update: Callable[[], Any] | None = None) -> None:
        self.mapper = mapper
        self.radius_m = radius_m
        self.keep_m = keep_m
        self.follow_cfg = follow_cfg
        # 给了就异步：跟随器要重规划 / 到达 frontier 时只发请求，由建图线程去算；
        # 不给（离线回放、测试）就当场 on_map_update。
        self.request_update = request_update
        self.ng: NavGrid | None = None
        self.mode = "idle"          # idle / goto / explore / done
        self._anchor: tuple[int, tuple[float, float]] | None = None
        self._goal_req: tuple[float, float] | None = None   # goto 的原始目标，建图线程下次更新时挂锚
        self._gen = 0               # 意图（goto/explore/cancel/到达）每变一次 +1；过期的规划结果丢弃
        self._awaiting = False      # 已请求异步重规划、结果还没回来
        self.follower: Any = None
        self.contract: Any = None
        self.last: dict[str, Any] = {}

    @property
    def _s(self) -> float:
        return self.mapper.cfg.world_scale

    def _resolve(self, anchor: tuple[int, tuple[float, float]] | None) -> tuple[float, float] | None:
        if anchor is None:
            return None
        p = self.mapper.resolve(anchor)
        return None if p is None else (p[0] * self._s, p[1] * self._s)

    def goal_xy(self) -> tuple[float, float] | None:
        """当前目标（导航系世界米），已按最新关键帧位姿解析；goto 还没挂锚时返回原始目标。"""
        if self._anchor is None:
            return self._goal_req
        return self._resolve(self._anchor)

    def _anchor_of(self, xy_world: tuple[float, float]) -> tuple[int, tuple[float, float]] | None:
        return self.mapper.anchor((xy_world[0] / self._s, xy_world[1] / self._s))

    def _intent(self, mode: str, anchor: Any = None, goal_req: tuple[float, float] | None = None) -> None:
        self.mode, self._anchor, self._goal_req, self.follower = mode, anchor, goal_req, None
        self._gen += 1
        self._awaiting = mode in ("goto", "explore")

    def goto(self, xy_world: tuple[float, float]) -> None:
        self._intent("goto", goal_req=(float(xy_world[0]), float(xy_world[1])))

    def explore(self) -> None:
        self._intent("explore")

    def cancel(self) -> None:
        self._intent("idle")

    def snapshot(self) -> dict[str, Any]:
        """``compute`` 需要的会话意图。调用方在锁里取，``compute`` 在锁外跑。"""
        return {"gen": self._gen, "mode": self.mode, "anchor": self._anchor, "goal_req": self._goal_req}

    def compute(self, est: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
        """重的那半：栅格化 + build + frontier/规划。不改会话字段（mapper 缓存除外），
        返回交给 ``apply``。mapper 同一时刻只能有一个线程在写/算。"""
        from .nav_follow import PathFollower

        mode, anchor = snap["mode"], snap["anchor"]
        here = est.get("xy") if est.get("state") == "localized" else None
        extra = None if here is None else np.array([[here[0] / self._s, here[1] / self._s]])
        ng = self.mapper.rasterize(extra)
        # 脚下 ~1.6 m 双目看不到，身体走过的走廊（OSC 门控）是当前位置可走的唯一证据。
        stats = ng.build(radius_m=self.radius_m, walked=self.mapper.walked())
        res: dict[str, Any] = {"gen": snap["gen"], "ng": ng, "mode": mode, "anchor": anchor,
                               "contract": None, "follower": None}
        info: dict[str, Any] = {"mode": mode, "grid": stats}
        res["info"] = info
        if mode in ("idle", "done") or here is None:
            info["reason"] = "not_localized" if here is None else mode
            return res
        if mode == "goto" and anchor is None and snap["goal_req"] is not None:
            anchor = res["anchor"] = self._anchor_of(snap["goal_req"])
        if mode == "explore":
            fs = frontiers(ng, here)
            info["frontiers"] = len(fs)
            if not fs:
                res["mode"], res["anchor"] = "done", None
                info["reason"] = "no_frontier"
                return res
            cur = self._resolve(anchor)
            pick = fs[0]
            if cur is not None:
                near = [f for f in fs if math.hypot(f.goal_xy_m[0] - cur[0], f.goal_xy_m[1] - cur[1]) <= self.keep_m]
                if near:
                    pick = near[0]
            anchor = res["anchor"] = self._anchor_of(pick.goal_xy_m)
        goal = self._resolve(anchor)
        if goal is None:
            res["mode"] = "idle"
            info["reason"] = "goal_anchor_lost"
            return res
        c = ng.plan(here, goal)
        res["contract"] = c
        info.update(c.to_dict())
        if c.accepted:
            res["follower"] = PathFollower(c.waypoints_xy_m, self.follow_cfg)
        return res

    def apply(self, res: dict[str, Any]) -> dict[str, Any]:
        """轻的那半：换上新栅格；意图没被改过才换规划（算的时候用户又 goto/cancel 了就只换栅格）。"""
        self.ng = res["ng"]
        info = res["info"]
        self._awaiting = res["gen"] != self._gen
        if res["gen"] != self._gen:
            info = {**info, "reason": "superseded"}
            self.last = info
            return info
        self.mode, self._anchor, self.follower = res["mode"], res["anchor"], res["follower"]
        if res["anchor"] is not None:
            self._goal_req = None
        if res["contract"] is not None:
            self.contract = res["contract"]
        self.last = info
        return info

    def on_map_update(self, est: dict[str, Any]) -> dict[str, Any]:
        """新关键帧 / 回环之后调用（同步版）。返回本次规划结果（给日志和 LLM 看）。"""
        return self.apply(self.compute(est, self.snapshot()))

    def _need_update(self, est: dict[str, Any]) -> None:
        if self.request_update is None:
            self.on_map_update(est)
        else:
            self.follower, self._awaiting = None, True
            self.request_update()

    def step(self, est: dict[str, Any], *, local_stop: bool = False) -> dict[str, Any]:
        idle = {"forward": 0.0, "turn_rate": 0.0}
        if self.mode == "idle":
            return {"state": "idle", "reason": "no_goal", **idle}
        if self.mode == "done":
            return {"state": "explore_done", "reason": "no_frontier", **idle}
        if self.follower is None:
            if self._awaiting and self.ng is not None:
                return {"state": "wait_map", "reason": "replanning", **idle}
            if self.ng is None or est.get("state") != "localized":
                return {"state": "wait_map", "reason": self.last.get("reason", "no_map"), **idle}
            # 规划被拒（目标在 unknown / 不连通……）：等下一次地图更新，不猜。
            return {"state": "blocked", "reason": self.last.get("reason", "no_plan"), **idle}
        out = self.follower.step(est, local_stop=local_stop)
        if out["state"] == "replan":
            self._need_update(est)
            return {**out, "forward": 0.0, "turn_rate": 0.0}
        if out["state"] == "arrived":
            if self.mode == "explore":
                self._intent("explore")     # 到了就当看过，下一次更新选新的 frontier
                self._need_update(est)
                return {**out, "state": "frontier_reached"}
            self._intent("idle")
        return out
