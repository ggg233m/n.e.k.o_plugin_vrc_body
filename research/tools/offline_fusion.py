# -*- coding: utf-8 -*-
r"""离线多视角融合（固化档第一步）—— 用**已采纳（merged）位姿**重建三态地图，清远场假障碍；
多场录制可融合成同一张**世界先验**（``--write-prior``，在线由 ``backend/nav_prior.py`` 消费）。

    python research/tools/offline_fusion.py 20261001_044153                  # auto：会话有 merged 就用它
    python research/tools/offline_fusion.py 20261001_044153 --pose final     # A/B 对照：录制自带 final_poses
    python research/tools/offline_fusion.py 20261005_235232 --session 20261005_235237 --pose merged
    # 世界先验（P0.3b 下一步②后半）：多场录制融合，落 <world>/prior/（原子写，rigid 口径）
    python research/tools/offline_fusion.py 20261001_044153 20261005_235232 20261006_001523 --write-prior

**它是什么**：把 2026-10-02 实验 `Docs/archive/离线多视角融合v1-假障碍清除（2026-10-02）.md`
（原脚本 `.tmp/_fusion_v1.py`）收编成正式工具。方法 = 重放录制关键帧，与在线 mapper 相同
的逐帧管线（体素化 5cm/2cm → R → cam_h → `_ground_correction` → 高/地分类），但每格计数按
**观测距离分层**（近 <1.5 m / 中 1.5–3 m / 远 >3 m）：

  * baseline：复刻在线计数规则（`O≥min_pts & O≥0.3G` + 3×3 多数），同一条数据管线；
  * fused：近距确认(`o_near≥min_pts`) | 中距多帧确认(`o_mid≥min_pts` 且 ≥2 关键帧 且 `O≥0.3G`)；
    baseline 判障但不满足前两条的格：有近/中距地面证据 → FREE，否则 → UNK（诚实降级）。

评估：run5-7 同口径「已知内障碍%」= 障碍格/(障碍格+free 格)，按轨迹距离环 R∈{1.5,2.5,3.0}。

**本工具新增（2026-10-06，P0.3b 下一步②"merged 表喂下游"）**：位姿源三选一 —
`final`（录制自带 `final_poses.npz`，缺则 kf `T_map`）、`merged`（世界目录 `xsession/` 下
`session_pose_table`：**已采纳表优先**，ids 对不上自动回退原表，绝不混 gauge）、
`auto`（默认：会话在世界里就走 merged，否则 final）。`--world` 缺省时自动在
`navmesh_memory/*/sessions/<sid>` 里找。⚠️ 录制名与会话 sid 可能不同（如录制
`20261005_235232` ↔ 会话 `20261005_235237`），对不上时用 `--session` 指名。

实验遗留（同文档 §发现 3）：**位姿质量是融合能力上限**——0929 用未修正位姿清障能力减半。
本工具的 A/B 就是拿这条断言在"并树后的 merged 位姿"上复检。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_grid import FREE, OCC                                # noqa: E402
from backend.nav_mapping import (MapperConfig, _ground_correction, _ground_model,  # noqa: E402
                                 _ray_voxels, assemble_labels, label_cells, majority3)
from backend.nav_prior import write_prior                          # noqa: E402
from backend.nav_xsession import session_pose_table                # noqa: E402

C = MapperConfig()
RES = C.res_m
NEAR_M, MID_M = 1.5, 3.0
MIN_PTS = C.min_pts
RINGS = (1.5, 2.5, 3.0)
KOFF = 1 << 20
# 射线体素在打包键里占的位数：iz ∈ [0, _ray_z())。见 ``ray_z()``。
RAY_ZBITS = 6


def load_events(rec: Path) -> list[dict]:
    return [json.loads(l) for l in
            (rec / "events.jsonl").read_text(encoding="utf-8").strip().splitlines() if l.strip()]


def find_world(sid: str) -> Path | None:
    """在 ``navmesh_memory/*/sessions/<sid>`` 里找这个会话属于哪个世界。"""
    for w in sorted((ROOT / "navmesh_memory").glob("*/sessions/" + sid)):
        if w.is_dir():
            return w.parents[1]
    return None


def table_poses(world_dir: Path, sid: str) -> tuple[dict[int, np.ndarray], list[int], str, str | None] | None:
    """会话位姿表 → ({kf_id: T}, 排序后的 id 列表, 来源描述, 坐标系根 base_sid)。表读不到返回 None。

    base_sid 是**全名**（来源描述里只截后 6 位给人看）：写先验时要拿它跟其它会话比根，
    截断的名字比不出"同根"。
    """
    tab = session_pose_table(world_dir, sid)
    if tab is None:
        return None
    ids = [int(k) for k in tab["ids"]]
    pose_of = {int(k): np.asarray(T, np.float64) for k, T in zip(ids, tab["T_map"])}
    base = str(tab["base_sid"]) if tab.get("base_sid") else None
    return pose_of, sorted(pose_of), f"{tab['source']}(base={(base or '-')[-6:]})", base


def load_poses(rec: Path, events: list[dict], mode: str, world_dir: Path | None, sid: str):
    """返回 (kf_npz, pose, src, fallback_count, base_sid)。

    merged 模式绝不混 gauge：表里没有的 k（如 refresh 撤下的帧）用**最近的表内 id** 兜底，
    不从 kf ``T_map``（原表系）取；final 模式保持原脚本行为（final_poses → kf T_map）。
    ``base_sid`` 只在从会话表取位姿时给出（merged/rigid），final 模式为 None。
    """
    cache: dict[int, np.lib.npyio.NpzFile] = {}

    def kf_npz(k: int):
        if k not in cache:
            cache[k] = np.load(rec / "kf" / f"{k:06d}.npz")
        return cache[k]

    n_fallback = 0
    if mode == "rigid" and world_dir is not None:
        got = table_poses(world_dir, sid)
        raw_path = Path(world_dir) / "sessions" / sid / "poses.npz"
        try:
            with np.load(raw_path) as z:
                raw_of = {int(k): np.asarray(T, np.float64) for k, T in zip(z["ids"], z["T_map"])}
        except (OSError, KeyError, ValueError):
            raise SystemExit(f"[fail] 读不到会话原表 {raw_path}")
        if got is None:
            raise SystemExit(f"[fail] 世界目录里没有会话 {sid} 的位姿表")
        merged_of, _ids, src, base = got
        common = sorted(set(raw_of) & set(merged_of))
        P = np.array([raw_of[k][:2, 3] for k in common])
        Q = np.array([merged_of[k][:2, 3] for k in common])
        sol, *_ = np.linalg.lstsq(np.column_stack([P, np.ones(len(P))]), Q, rcond=None)
        M, t2 = sol[:2], sol[2]
        U, _s, Vt = np.linalg.svd(M)
        R2 = U @ Vt
        if np.linalg.det(R2) < 0:
            R2 = U @ np.diag([1.0, -1.0]) @ Vt
        t2 = Q.mean(0) - P.mean(0) @ R2                      # 旋转定死后再取平移（最小二乘闭式）
        Rz = np.eye(3)
        Rz[:2, :2] = R2.T                                    # 列向量约定：R_new = Rz @ R_raw
        resid = float(np.median(np.linalg.norm(P @ R2 + t2 - Q, axis=1)))
        print(f"[rigid] 只取并树的刚体分量：n={len(common)} · 非刚体残余（中位）{resid:.3f} m", flush=True)

        def pose(k: int) -> np.ndarray:
            nonlocal n_fallback
            T = raw_of.get(int(k))
            if T is None:                                    # 原表缺该帧（理论不发生）：保位不动
                n_fallback += 1
                return np.eye(4)
            out = T.copy()
            out[:2, 3] = T[:2, 3] @ R2 + t2
            out[:3, :3] = Rz @ T[:3, :3]
            return out

        return kf_npz, pose, f"rigid({src})", n_fallback, base

    if mode in ("auto", "merged") and world_dir is not None:
        got = table_poses(world_dir, sid)
        if got is not None:
            pose_of, ids_sorted, src, base = got
            arr_ids = np.asarray(ids_sorted)

            def pose(k: int) -> np.ndarray:
                nonlocal n_fallback
                hit = pose_of.get(int(k))
                if hit is not None:
                    return hit
                n_fallback += 1
                j = int(np.clip(np.searchsorted(arr_ids, int(k)), 0, len(arr_ids) - 1))
                return pose_of[int(arr_ids[j])]

            return kf_npz, pose, src, n_fallback, base
        if mode == "merged":
            raise SystemExit(f"[fail] 世界目录里找不到会话 {sid}（--world {world_dir}）；"
                             f"或改用 --pose final")

    fp = rec / "final_poses.npz"
    if fp.exists():
        z = np.load(fp)
        pose_of = {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
        src = "final_poses"
    else:
        pose_of, src = {}, "kf_T_map"

    def pose(k: int) -> np.ndarray:
        hit = pose_of.get(int(k))
        if hit is not None:
            return hit
        return np.asarray(kf_npz(int(k))["T_map"], np.float64)

    return kf_npz, pose, src, n_fallback, None


def estimate_cam_h(kf_npz, events, alive_check) -> float:
    """全场 cam_h：逐帧取相机下方 [1.2,2.6] m 带内 rz 的 1 cm 直方图众数，跨帧加权中位。"""
    cands, wts = [], []
    for e in events:
        if e.get("kind") != "kf":
            continue
        k = int(e["k"])
        if not alive_check(k):
            continue
        p = kf_npz(k)["pts"]
        rz = p[:, 2]
        m = (rz >= -2.6) & (rz <= -1.2)
        if m.sum() < 100:
            continue
        bins = np.floor(rz[m] / 0.01).astype(np.int64)
        u, cnt = np.unique(bins, return_counts=True)
        mode = u[np.argmax(cnt)]
        band = np.abs(bins - mode) <= 6  # ±6 cm
        cands.append(-float(np.median(rz[m][band])))
        wts.append(int(m.sum()))
    if not cands:
        return C.cam_h_default
    cands = np.asarray(cands)
    wts = np.asarray(wts, float)
    order = np.argsort(cands)
    cw = np.cumsum(wts[order])
    return float(cands[order][np.searchsorted(cw, cw[-1] / 2)])


def ray_z() -> int:
    """障碍高度带 (ground_tol, obst_top) 的层数。与 ``KeyframeGridMapper._ray_z`` 同一算式，
    但**不复制那个数**：直接问 cfg，改 ``ray_z_m`` 时两边一起动。"""
    return int(math.ceil((C.obst_top_m - C.ground_tol_m) / C.ray_z_m))


def kf_rows(k, pts, T, cam_h, *, ray_clear=False):
    """单关键帧 → ``(cu, rows[6])``，``ray_clear=True`` 时 → ``(cu, rows, (hit, mis))``。

    ``rows`` 六列 = ``g0,g1,g2,o0,o1,o2``（地面/障碍 × 近/中/远带）。
    ``hit``/``mis`` 是**该关键帧的射线体素**（``K×3`` 的 ``(ix, iy, iz)``，地图系格下标），
    语义与 ``KeyframeGridMapper._ray_hit/_ray_mis`` 逐位一致——同一个 ``_ray_voxels`` 算出来的。
    """
    R, t = T[:3, :3], T[:2, 3]
    # 与在线 mapper **同一道距离门**（``nav_mapping.py:472``，在 ``add_keyframe`` 里）。
    # 在线其实过了两层：采集时 ``stereo_points(max_range_m=range_m)`` 卡的是**光学深度 z**，
    # 入图时这里卡的是**水平半径**。npz 里存的是过了第一层的点（044153 实测 r 到 10.11 m），
    # 所以这一层在离线是**真缺**：044153 有 20.8% 的点 r > 5.0，会被这道门丢掉。
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= C.range_m]
    # 局部系体素化（与 mapper 同参：xy 5cm / z 2cm），计数聚合
    vkey = (np.floor(pts[:, 0] / C.vox_xy_m).astype(np.int64) * (1 << 42)
            + np.floor(pts[:, 1] / C.vox_xy_m).astype(np.int64) * (1 << 21)
            + np.floor(pts[:, 2] / C.vox_z_m).astype(np.int64))
    u, inv, cnt = np.unique(vkey, return_inverse=True, return_counts=True)
    # 体素中心 = 该体素内点均值（mapper 用体素中心；均值等价且更稳）
    n_v = len(u)
    centers = np.zeros((n_v, 3), np.float64)
    np.add.at(centers, inv, pts.astype(np.float64))
    centers /= cnt[:, None]
    w = cnt.astype(np.float64)

    pw = centers @ R.T
    rxy = pw[:, :2]
    h_raw = pw[:, 2] + cam_h
    gc = _ground_correction(h_raw, w, rxy, C.ground_offset_max_m, C.ground_offset_min_range_m,
                            C.ground_plane_max_deg, C.ground_plane_band_m)
    h = h_raw - gc
    g = np.abs(h) <= C.ground_tol_m
    o = (h > C.ground_tol_m) & (h < C.obst_top_m)
    sel = g | o
    if sel.sum() < 20:
        return None
    # 射线清除要用**全量**点（含高带）的高度，所以在 ``sel`` 过滤之前算。
    # 与在线 ``_sync_ray`` 走的是同一个 ``_ray_voxels``：那是唯一实现，绝不在这里重写一份
    # （在线那份的稠密去重、10 cm 端点去重都是量过代价调出来的，抄一遍必然漂移）。
    # ``under`` 与在线同义：关键帧正下方地面修正量，用来定"相机离地多高"。
    ray = None
    if ray_clear:
        corr, under = _ground_model(h_raw, w, rxy, C.ground_offset_max_m,
                                    C.ground_offset_min_range_m, C.ground_plane_max_deg,
                                    C.ground_plane_band_m)
        hit, mis = _ray_voxels(rxy, h_raw - corr, t, cam_h - under, C)
        Z = ray_z()
        hit = hit[(hit[:, 2] >= 0) & (hit[:, 2] < Z)]
        mis = mis[(mis[:, 2] >= 0) & (mis[:, 2] < Z)]
        ray = (hit, mis)
    rxy, w, g, o = rxy[sel], w[sel], g[sel], o[sel]
    d = np.hypot(rxy[:, 0], rxy[:, 1])
    band = np.where(d < NEAR_M, 0, np.where(d < MID_M, 1, 2)).astype(np.int64)
    xy = rxy + t
    ix = np.floor(xy[:, 0] / RES).astype(np.int64)
    iy = np.floor(xy[:, 1] / RES).astype(np.int64)
    ckey = (ix + KOFF) * (4 * KOFF) + (iy + KOFF)  # 偏移打包：坐标可负（±100 km 内安全）
    cu, cinv = np.unique(ckey, return_inverse=True)
    rows = np.zeros((len(cu), 6), np.float64)
    for cls_mask, base in ((g, 0), (o, 3)):
        for b in range(3):
            m = cls_mask & (band == b)
            if m.any():
                rows[:, base + b] = np.bincount(cinv[m], weights=w[m], minlength=len(cu))
    if ray_clear:
        return cu, rows, ray
    return cu, rows


def aggregate(rec: Path, mode: str, world_dir: Path | None, sid: str, *,
              ray_clear: bool | None = None) -> dict:
    """``ray_clear``：None = 用 ``MapperConfig`` 的当前值（与在线同一开关）；可显式覆盖做 A/B。"""
    events = load_events(rec)
    kf_npz, pose, src, n_fb, base = load_poses(rec, events, mode, world_dir, sid)
    rc = C.ray_clear if ray_clear is None else bool(ray_clear)
    # refresh 语义（replay 同款）：kf k 事件带 refresh ⇒ 上一帧 k-1 的点让出（不计入终态）。
    alive = []
    prev = None
    n_refresh = 0
    for e in events:
        if e.get("kind") != "kf":
            continue
        if e.get("refresh") and prev is not None:
            alive.pop()  # 丢掉 k-1
            n_refresh += 1
        alive.append(int(e["k"]))
        prev = int(e["k"])

    t0 = time.perf_counter()
    cam_h = estimate_cam_h(kf_npz, events, lambda k: k in set(alive))
    n_est = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows_all = []
    rays: list[tuple[np.ndarray, np.ndarray]] = []
    traj = []
    for k in alive:
        z = kf_npz(k)
        T = pose(k)
        r = kf_rows(k, z["pts"], T, cam_h, ray_clear=rc)
        if r is not None:
            rows_all.append(r[:2])
            if rc:
                rays.append(r[2])
        traj.append(T[:2, 3])
    for e in events:
        if e.get("kind") == "trail":
            k = int(e["k"])
            try:
                T = pose(k) @ np.asarray(e["T_dr"], np.float64)
            except (FileNotFoundError, KeyError):
                continue
            traj.append(T[:2, 3])
    dt = time.perf_counter() - t0

    keys = np.concatenate([r[0] for r in rows_all])
    vals = np.concatenate([r[1] for r in rows_all])
    uk, inv = np.unique(keys, return_inverse=True)
    G = np.zeros((len(uk), 6), np.float64)
    S = np.zeros((len(uk), 6), np.float64)  # support：该格该带非零的关键帧数
    for j in range(6):
        G[:, j] = np.bincount(inv, weights=vals[:, j], minlength=len(uk))
        S[:, j] = np.bincount(inv, weights=(vals[:, j] > 0).astype(float), minlength=len(uk))
    ix = (uk >> 22) - KOFF
    iy = (uk & (4 * KOFF - 1)) - KOFF
    hit = mis = None
    if rc:
        hk, hn, mk, mn = ray_keys(rays)
    else:
        hk = hn = mk = mn = None
    return dict(cam_h=cam_h, cam_h_est_s=round(n_est, 1), fusion_s=round(dt, 1),
                n_kf=len(alive), n_refresh=n_refresh, pose_src=src, pose_fallback=n_fb,
                base=base, rec=rec.name, sid=sid, ray_clear=rc,
                ix=ix, iy=iy, G=G, S=S, traj=np.asarray(traj),
                ray_hit_key=hk, ray_hit_n=hn, ray_mis_key=mk, ray_mis_n=mn)


def ray_zbits() -> int:
    """射线层号在打包键里占的位数。按 ``ray_z()`` 现算，不写死。"""
    return max(1, (ray_z() - 1).bit_length())


def ray_keys(rays: list[tuple[np.ndarray, np.ndarray]]):
    """各帧射线体素 → 打包键 ``(格键 << zbits) | iz`` 的 (打中/看穿) 去重计数。

    格键与 ``G`` 的格键**同一坐标系**（都是 ``(ix+KOFF)*4KOFF + (iy+KOFF)``），所以跨场合并
    只是「拼起来再 unique」，与 ``merge_aggs`` 处理 G/S 完全同一个套路。
    层号进键 ⇒ ``ray_veto`` 才能按**层**做 ``any``（在线正是逐层判的，见那里的说明）。
    """
    if not rays:
        z = np.zeros(0, np.int64)
        return z, z, z, z

    def pack(arrs):
        if not arrs:
            return np.zeros(0, np.int64), np.zeros(0, np.float64)
        a = np.concatenate(arrs)
        u, c = np.unique(a, return_counts=True)
        return u, c.astype(np.float64)

    hk, hn = pack([((h[:, 0] + KOFF) * (4 * KOFF) + (h[:, 1] + KOFF)) << ray_zbits() | h[:, 2]
                   for h, _m in rays if len(h)])
    mk, mn = pack([((m[:, 0] + KOFF) * (4 * KOFF) + (m[:, 1] + KOFF)) << ray_zbits() | m[:, 2]
                   for _h, m in rays if len(m)])
    return hk, hn, mk, mn


def ray_veto(agg: dict, *, beta: float | None = None,
             walk_w: float | None = None) -> np.ndarray:
    """本聚合的射线清除判据 → 长度 = ``G`` 的布尔数组，``True`` = 该格被看穿清掉。

    判据与在线 ``KeyframeGridMapper._ray_veto`` **逐字同源**：
    ``veto = ~((hit > 0) & (mis < beta · hit)).any(层)``
    —— 即"没有任何一层满足 看穿 < beta×打中"。

    ⚠️ **必须逐层 ``any``，不能先把层加起来**。反例（每层各 1 打中、看穿 5/0）：
    逐层看第 2 层满足条件 ⇒ 不清除；先求和得 hit=2/mis=5 ⇒ 5 ≥ 2 ⇒ 误判成清除。
    这是"看起来只是省一次循环"的静默错误，所以层号进了打包键。

    另加身体走过的中心线每格 ``ray_walk_w`` 次看穿（在线 ``ray_walk_w``，默认 3.0，**所有层**都加，
    与在线把一维线广播进 ``(Z, n)`` 的写法一致）：走过就是活的通行证据。
    """
    c = MapperConfig()
    beta = c.ray_beta if beta is None else float(beta)
    walk_w = c.ray_walk_w if walk_w is None else float(walk_w)
    n = len(agg["ix"])
    if not n:
        return np.zeros(0, bool)
    zb = ray_zbits()
    zmask = (1 << zb) - 1
    cell = (agg["ix"] + KOFF) * (4 * KOFF) + (agg["iy"] + KOFF)
    order = np.argsort(cell, kind="stable")
    cs = cell[order]

    def spread(keys, counts, z: int) -> np.ndarray:
        out = np.zeros(n, np.float64)
        if keys is None or not len(keys):
            return out
        sel = (keys & zmask) == z
        if not sel.any():
            return out
        k = keys[sel] >> zb
        pos = np.searchsorted(cs, k)
        ok = (pos < n) & (cs[np.minimum(pos, n - 1)] == k)
        np.add.at(out, order[pos[ok]], counts[sel][ok])
        return out

    # 走过的格（每层都 +walk_w 次看穿）。
    wl = np.zeros(n, np.float64)
    if walk_w:
        tr = np.asarray(agg["traj"], np.float64)
        if len(tr):
            tcell = ((np.floor(tr[:, 0] / RES).astype(np.int64) + KOFF) * (4 * KOFF)
                     + (np.floor(tr[:, 1] / RES).astype(np.int64) + KOFF))
            pos = np.searchsorted(cs, tcell)
            ok = (pos < n) & (cs[np.minimum(pos, n - 1)] == tcell)
            wl[order[pos[ok]]] = walk_w

    cond_any = np.zeros(n, bool)
    for z in range(ray_z()):
        hz = spread(agg.get("ray_hit_key"), agg.get("ray_hit_n"), z)
        mz = spread(agg.get("ray_mis_key"), agg.get("ray_mis_n"), z) + wl
        cond_any |= (hz > 0) & (mz < beta * hz)
    return ~cond_any


def merge_aggs(aggs: list[dict]) -> dict:
    """多场录制 → 一份聚合（**同一世界系**：各自 rigid 位姿已落并树根坐标系）。

    先验的"世界级"就在这里：跨会话的票合并后，`classify` 的"中距多帧确认"与射线式
    降级都多看几场会话的眼睛——单场没看全的角落，另一场看见了。
    ⚠️ 混不同世界系/不同 world_scale 的会话是**错的**，调用方负责先验（--write-prior 有闸）。
    """
    keys = np.concatenate([(a["ix"] + KOFF) * (4 * KOFF) + (a["iy"] + KOFF) for a in aggs])
    vals = np.concatenate([a["G"] for a in aggs])
    sup = np.concatenate([a["S"] for a in aggs])
    uk, inv = np.unique(keys, return_inverse=True)
    G = np.zeros((len(uk), 6), np.float64)
    S = np.zeros((len(uk), 6), np.float64)
    for j in range(6):
        G[:, j] = np.bincount(inv, weights=vals[:, j], minlength=len(uk))
        S[:, j] = np.bincount(inv, weights=sup[:, j], minlength=len(uk))
    ix = (uk >> 22) - KOFF
    iy = (uk & (4 * KOFF - 1)) - KOFF
    # 射线票与 G/S 同一套路合并（键已含层号，直接拼起来 unique）。
    hk, hn, mk, mn = ray_keys([])
    rhk, rhn, rmk, rmn = [], [], [], []
    for a in aggs:
        if a.get("ray_hit_key") is not None and len(a["ray_hit_key"]):
            rhk.append(a["ray_hit_key"]); rhn.append(a["ray_hit_n"])
        if a.get("ray_mis_key") is not None and len(a["ray_mis_key"]):
            rmk.append(a["ray_mis_key"]); rmn.append(a["ray_mis_n"])
    if rhk:
        allk = np.concatenate(rhk)
        allv = np.concatenate(rhn)
        u, inv = np.unique(allk, return_inverse=True)
        hk = u; hn = np.bincount(inv, allv, minlength=len(u))
    if rmk:
        allk = np.concatenate(rmk)
        allv = np.concatenate(rmn)
        u, inv = np.unique(allk, return_inverse=True)
        mk = u; mn = np.bincount(inv, allv, minlength=len(u))
    return dict(ix=ix, iy=iy, G=G, S=S,
                traj=np.concatenate([a["traj"] for a in aggs]),
                n_kf=sum(a["n_kf"] for a in aggs),
                n_refresh=sum(a["n_refresh"] for a in aggs),
                pose_fallback=sum(a["pose_fallback"] for a in aggs),
                cam_h=[a["cam_h"] for a in aggs],
                fusion_s=round(sum(a["fusion_s"] for a in aggs), 1),
                ray_clear=all(a.get("ray_clear", False) for a in aggs),
                ray_hit_key=hk, ray_hit_n=hn, ray_mis_key=mk, ray_mis_n=mn,
                per=aggs)


def dense(agg):
    x0, y0 = agg["ix"].min(), agg["iy"].min()
    H = int(agg["ix"].max() - x0 + 1)
    W = int(agg["iy"].max() - y0 + 1)
    flat = ((agg["ix"] - x0) * W + (agg["iy"] - y0))

    def put(cols):
        out = np.zeros((H, W, len(cols)), np.float64)
        for j, c in enumerate(cols):
            out[:, :, j].flat[flat] = agg["G"][:, c]
        return out

    gband = put([0, 1, 2])
    oband = put([3, 4, 5])

    def sup(cols):
        out = np.zeros((H, W, len(cols)), np.float64)
        for j, c in enumerate(cols):
            out[:, :, j].flat[flat] = agg["S"][:, c]
        return out

    sband = sup([0, 1, 2, 3, 4, 5])
    return x0, y0, gband, oband, sband


def majority(occ):
    """**已废弃**：改调 ``backend.nav_mapping.majority3``（唯一实现，带 BORDER_CONSTANT）。
    保留这个名字只为兼容可能的外部引用；它的旧实现少了 ``BORDER_CONSTANT``，
    图外沿会镜像回填、白送 3×3 支撑。"""
    return majority3(occ)


def classify(gband, oband, sband, veto=None):
    """三态判定 → ``(occ_baseline, labels_fused, occ_b3, stats)``。

    **规则本体走 ``backend.nav_mapping.label_cells``**（唯一实现），本函数只负责：
    把分带计数铺成 ``label_cells`` 要的入参、按**在线同款**算出 ``base_occ``、再算 stats。

    ⚠️ 2026-10-08 前的版本是自己重写的一份规则，**漏了两处门**：
      * ``base_occ`` 前置门（``(n_o > min_pts) & (n_o ≥ ratio·n_g)`` + 3×3 多数）——
        离线直接 ``majority(near_ok | mid_ok)``，等于近带票单条即定罪，绕过三道密度门；
      * ``solo`` 单帧例外（票 ≥q_solo_pts 且远带占比 ≤q_far_tol）——离线没有，
        中距真障碍被 q_mid_kf≥2 卡掉。
    044153 实测：只在线有 580 格、只离线有 1229 格、IoU 0.662。现在两边同一份。

    ``veto``：射线清除掩码（``(H,W)`` bool，True = 该格被看穿）。接法见下面注释。
    """
    # 射线清除：在线 ``rasterize`` 是先把 ``n_o`` 清零、**再**算 ``base_occ`` 与走 ``_by_quality``
    # （``nav_mapping.py:1087-1091``），所以被看穿的格连候选都不是。离线要等价，
    # 就必须在**算 base_occ 之前**把障碍票清零——只清 ``oband`` 的分带票不够，
    # 因为 ``base_occ`` 用的是 ``O = oband.sum()``。
    if veto is not None:
        oband = np.where(veto[:, :, None], 0.0, oband)
    G = gband.sum(axis=2)
    O = oband.sum(axis=2)
    o_near, o_mid, o_far = oband[:, :, 0], oband[:, :, 1], oband[:, :, 2]
    g_near, g_mid = gband[:, :, 0], gband[:, :, 1]
    sup_o_mid = sband[:, :, 4]

    # 与在线 ``rasterize`` 逐字同源的两步（在线用的是累加器整块，这里用的是分带求和，等价）。
    occ_b = majority3((O >= MIN_PTS) & (O >= C.occ_ground_ratio * G))
    # 规则本体：与在线同一份（含 base_occ 前置门与 solo 单帧例外）。
    occ_f, restored, rejected = label_cells(C, occ_b, G, O, o_near, o_mid, g_near, sup_o_mid)

    # RangeMax 复刻（run5-7 口径）：只用近+中距观测建图，远场票全部丢弃。**离线专有**，
    # 不接进在线（它是历史对照口径，不是生产规则）。
    G_nm = gband[:, :, :2].sum(axis=2)
    O_nm = oband[:, :, :2].sum(axis=2)
    occ_b3 = majority3((O_nm >= MIN_PTS) & (O_nm >= C.occ_ground_ratio * G_nm))

    g_evid = (g_near + g_mid) >= MIN_PTS
    # 三态装配走 ``backend.nav_mapping.assemble_labels``（唯一实现）——**含在线的第 ⑦ 步**
    # （``rejected & ~restored ⇒ UNK``）。改动前离线只有第 ④ 步，FREE 是在线的超集：
    # 044153 上多出 7,106 格（占其 FREE 的 10.3%，其中 4,409 格**只有远带**地面票）。
    # 这一步不是"收紧口径"，是补回在线本来就有的诚实降级。
    #
    # ⚠️ **编码转换**：``assemble_labels`` 返回 ``nav_grid`` 编码（FREE 178 / OCC 0 / UNK 89），
    # 而本工具与 ``write_prior`` 用的是 **先验编码**（0 UNK / 1 FREE / 2 OCC）。
    # 两套编码里 UNK 与 OCC 的数值正好交叉，**直接混用不会报错**，只会把障碍写成可走。
    # 转换只在这一处做。
    _g = assemble_labels(C, occ_f, restored, rejected, G)
    labels_f = np.full(_g.shape, 0, np.uint8)          # 0 = UNK
    labels_f[_g == FREE] = 1                            # 1 = FREE
    labels_f[_g == OCC] = 2                             # 2 = OCC

    # ⚠️ 与在线**仍有一处故意不同**（保留，不修）：在线第 ④ 步的门是 ``n_g > 0.5``，
    # 即**任何**地面观测（含远带）都算；这与"unknown 永不当 free"的红线精神有张力——
    # 远带 δz 一个视差像素就值米级（C31）。实测（.tmp/free_gate.py，三场融合先验）：
    # 只有远带地面票的 FREE 占 37.6%，收紧到"近+中距 ≥min_pts"会掉 38.3%。
    # **但那次测量不支持"收紧才对"**：那些格离轨迹中位仅 0.89 m、票中位 11 张，
    # 很可能是走过的地面本身。⇒ 保持原行为，判据留给有真值的那天。
    free_f = labels_f == 1

    base_occ_mask = occ_b
    cleared_free = base_occ_mask & (labels_f == 1)
    cleared_unk = base_occ_mask & (labels_f == 0)
    kept = base_occ_mask & occ_f
    added = occ_f & ~base_occ_mask
    stats = dict(
        occ_baseline=int(occ_b.sum()), occ_fused=int(occ_f.sum()),
        kept=int(kept.sum()), cleared_free=int(cleared_free.sum()),
        cleared_unk=int(cleared_unk.sum()), added=int(added.sum()),
        cleared_free_with_gnd_evid=int((cleared_free & g_evid).sum()),
        cleared_far_only_share=round(float((o_far[cleared_free | cleared_unk].sum() /
                                            max(o_near[cleared_free | cleared_unk].sum()
                                                + o_mid[cleared_free | cleared_unk].sum()
                                                + o_far[cleared_free | cleared_unk].sum(), 1.0))), 3),
    )
    return occ_b, labels_f, occ_b3, stats


def rings_metrics(traj, x0, y0, occ, free):
    H, W = occ.shape
    out = {}
    for R in RINGS:
        m = np.zeros((H, W), np.uint8)
        rpx = int(R / RES)
        for p in traj:
            cx, cy = int((p[0] / RES) - x0), int((p[1] / RES) - y0)
            if 0 <= cx < H and 0 <= cy < W:
                cv2.circle(m, (cy, cx), rpx, 1, -1)
        ring = m > 0
        n_occ = int((occ & ring).sum())
        n_free = int((free & ring).sum())
        n_all = int(ring.sum())
        out[f"R{R}"] = dict(
            known_obstacle_pct=round(100.0 * n_occ / max(n_occ + n_free, 1), 2),
            free_pct=round(100.0 * n_free / max(n_all, 1), 2),
            unknown_pct=round(100.0 * (n_all - n_occ - n_free) / max(n_all, 1), 2),
            occ=n_occ, free=n_free)
    return out


def panel(labels, title, traj, x0, y0):
    H, W = labels.shape
    img = np.full((H, W, 3), 255, np.uint8)
    img[labels == 0] = (217, 217, 217)
    img[labels == 1] = (102, 187, 106)
    img[labels == 2] = (229, 57, 53)
    for p in traj:
        cx, cy = int(p[0] / RES) - x0, int(p[1] / RES) - y0
        if 0 <= cx < H and 0 <= cy < W:
            cv2.circle(img, (cy, cx), 1, (222, 120, 21), -1)  # 轨迹
    img = cv2.resize(img, None, fx=min(1.0, 900 / max(H, W)), fy=min(1.0, 900 / max(H, W)),
                     interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(img, 28, 8, 8, 8, cv2.BORDER_CONSTANT, value=(60, 60, 60))


def session_world_scale(world_dir: Path, sid: str) -> float | None:
    """会话 session.json 里的 world_scale（换过 avatar 就变）。缺字段/读不到返回 None。"""
    try:
        meta = json.loads((Path(world_dir) / "sessions" / sid / "session.json").read_text(encoding="utf-8"))
        ws = float(meta.get("world_scale"))
    except (OSError, ValueError, TypeError):
        return None
    return ws if ws > 0.0 and np.isfinite(ws) else None


def veto_grid(agg: dict, x0: int, y0: int, shape: tuple[int, int],
              **kw) -> np.ndarray | None:
    """射线清除 → ``(H, W)`` 稠密掩码（``True`` = 该格被看穿）。没开射线清除返回 ``None``。

    抽成独立函数有两个理由，都不是洁癖：
      * ``main`` 里这段原来内联在 ``dense()`` 之后、``H, W`` 赋值**之前**，
        踩过一次 `UnboundLocalError`（而当时的 A/B 脚本直接调 ``aggregate``/``classify``，
        绕过了 `main` ⇒ **测试全绿但命令行根本跑不起来**）。取成显式入参就没有这个顺序依赖。
      * 它有一处容易写错的对齐（稀疏键 → 稠密 ``flat``），值得单独测。
    """
    if not agg.get("ray_clear"):
        return None
    sparse = ray_veto(agg, **kw)
    H, W = shape
    out = np.zeros((H, W), bool)
    flat = (agg["ix"] - x0) * W + (agg["iy"] - y0)
    if len(flat):
        if flat.min() < 0 or flat.max() >= H * W:
            raise ValueError(f"射线票的格越出稠密图幅（flat ∈ [{flat.min()}, {flat.max()}]，"
                             f"容量 {H * W}）：x0/y0/shape 与 agg 不是同一套坐标系")
        out.flat[flat] = sparse
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec", nargs="+", help="录制目录名（navmesh_recordings/ 下）；给多个 = 多场融合成同一张图")
    ap.add_argument("--session", default=None,
                    help="会话 sid，逗号分隔、与 rec 一一对应（默认与录制同名；录制名与会话 sid 不同时指名，"
                         "如 20261005_235232 ↔ 20261005_235237）")
    ap.add_argument("--world", default=None, help="世界目录（默认自动在 navmesh_memory/*/sessions/<sid> 找）")
    ap.add_argument("--pose", choices=("auto", "final", "merged", "rigid"), default="auto",
                    help="auto=会话在世界上就用已采纳(merged)表；final=录制自带 final_poses；"
                         "merged=强制用会话表；rigid=原表位姿只加并树的刚体分量（保会话内部形状）")
    ap.add_argument("--write-prior", action="store_true",
                    help="把融合结果固化为**世界先验** <world>/prior/prior.npz+json（原子写）。"
                         "强制 rigid 口径（弹性形变会拉坏多视一致性，见 ROADMAP 2026-10-06 A/B）")
    ap.add_argument("--out", default=None, help="输出目录（默认 .tmp/offline_fusion）")
    ap.add_argument("--ray-clear", dest="ray_clear", action="store_true", default=None,
                    help="开射线清除（默认跟随 MapperConfig.ray_clear=True，与在线同一开关）")
    ap.add_argument("--no-ray-clear", dest="ray_clear", action="store_false",
                    help="关射线清除（A/B 对照用：量它到底改了多少格）")
    args = ap.parse_args()

    recs = [ROOT / "navmesh_recordings" / r for r in args.rec]
    for r in recs:
        if not (r / "events.jsonl").is_file():
            raise SystemExit(f"[fail] 没有这个录制：{r}")
    sids = [s.strip() for s in (args.session or "").split(",") if s.strip()]
    if sids and len(sids) != len(recs):
        raise SystemExit(f"[fail] --session 给了 {len(sids)} 个 sid，录制有 {len(recs)} 个（一一对应）")
    if not sids:
        sids = [r.name for r in recs]
    mode = args.pose
    if args.write_prior:
        if args.pose not in ("auto", "rigid"):
            raise SystemExit("[fail] --write-prior 只接受 --pose rigid（默认 auto 即 rigid）："
                             "merged/final 的位姿会混 gauge 或带弹性形变")
        mode = "rigid"

    world_dir = Path(args.world) if args.world else None
    if world_dir is None and mode in ("auto", "merged", "rigid"):
        found = {w for w in (find_world(s) for s in sids) if w is not None}
        if len(found) > 1:
            raise SystemExit(f"[fail] 多场录制落在不同世界目录：{sorted(str(w) for w in found)}")
        if found:
            world_dir = found.pop()
    if args.write_prior and world_dir is None:
        raise SystemExit("[fail] --write-prior 需要世界目录（会话没在世界里找到 ⇒ 无坐标系可固化）")
    out_dir = Path(args.out) if args.out else (ROOT / ".tmp" / "offline_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    aggs = [aggregate(rec, mode, world_dir, sid, ray_clear=args.ray_clear)
            for rec, sid in zip(recs, sids)]
    agg = aggs[0] if len(aggs) == 1 else merge_aggs(aggs)
    x0, y0, gband, oband, sband = dense(agg)
    H, W = gband.shape[:2]
    veto = veto_grid(agg, x0, y0, (H, W))
    occ_b, labels_f, occ_b3, st = classify(gband, oband, sband, veto)
    st["ray_clear"] = bool(agg.get("ray_clear"))
    if veto is not None:
        st["ray_cleared_cells"] = int((veto & (oband.sum(axis=2) > 0)).sum())

    G = gband.sum(axis=2)
    free_b = (G > 0.5) & ~occ_b
    free_f = labels_f == 1
    free_3 = (gband[:, :, :2].sum(axis=2) > 0.5) & ~occ_b3

    m_b = rings_metrics(agg["traj"], x0, y0, occ_b, free_b)
    m_f = rings_metrics(agg["traj"], x0, y0, labels_f == 2, free_f)
    m_3 = rings_metrics(agg["traj"], x0, y0, occ_b3, free_3)

    lab_b = np.zeros((H, W), np.uint8)
    lab_b[free_b] = 1
    lab_b[occ_b] = 2
    pb = panel(lab_b, "baseline", agg["traj"], x0, y0)
    pf = panel(labels_f, "fused", agg["traj"], x0, y0)
    for img, title, ox in ((pb, "baseline (count rule)", 10),
                           (pf, "fused (quality layered)", pb.shape[1] + 20)):
        cv2.putText(img, title, (ox, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.full((max(pb.shape[0], pf.shape[0]), pb.shape[1] + pf.shape[1] + 20, 3), 40, np.uint8)
    canvas[:pb.shape[0], :pb.shape[1]] = pb
    canvas[:pf.shape[0], pb.shape[1] + 20:] = pf
    tag = f"{args.rec[0]}_pose{mode}" if len(recs) == 1 else f"{args.rec[0]}+{len(recs) - 1}_pose{mode}"
    cv2.imwrite(str(out_dir / f"compare_{tag}.png"), canvas)

    result = dict(recs=args.rec, sessions=sids, world=str(world_dir) if world_dir else None,
                  pose_mode=mode, n_kf=agg["n_kf"], n_refresh=agg["n_refresh"],
                  pose_fallback=agg["pose_fallback"],
                  cam_h=(agg["cam_h"] if len(aggs) == 1 else
                         [round(c, 3) for c in agg["cam_h"]]),
                  fusion_s=agg["fusion_s"],
                  per_session=[{"rec": a["rec"], "sid": a["sid"], "pose_src": a["pose_src"],
                                "base": a["base"], "n_kf": a["n_kf"], "fallback": a["pose_fallback"]}
                               for a in aggs],
                  transitions=st, baseline_all_rings=m_b, rmax3_rings=m_3, fused_rings=m_f)

    if args.write_prior:
        bases = {a["base"] for a in aggs}
        if None in bases or len(bases) != 1:
            raise SystemExit(f"[fail] 会话位姿表坐标系根不一致（{sorted(str(b) for b in bases)}）："
                             "先验必须落在**同一棵树**里；先跑 tools/xsession_align.py --world-tree")
        base_sid = str(bases.pop())
        scales = {sid: session_world_scale(world_dir, sid) for sid in sids}
        if any(v is None for v in scales.values()):
            raise SystemExit(f"[fail] 会话 session.json 缺 world_scale（{scales}）：不能固化先验（尺度不明）")
        if max(scales.values()) - min(scales.values()) > 1e-6:
            raise SystemExit(f"[fail] 会话 world_scale 不一致（{scales}）：换过 avatar，先验不能混尺度")
        ws = float(min(scales.values()))
        # 存储布局：行 = y、列 = x（labels_f 是行 = x、列 = y）⇒ 转置；origin = 最小格下标 × 分辨率。
        npz_path = write_prior(
            world_dir, labels_f.T, (x0 * RES, y0 * RES), RES, world_scale=ws, base_sid=base_sid,
            meta={"tool": "research/tools/offline_fusion.py", "pose_mode": mode,
                  "sessions": [{"sid": a["sid"], "rec": a["rec"], "n_kf": a["n_kf"],
                                "pose_src": a["pose_src"], "pose_fallback": a["pose_fallback"]}
                               for a in aggs],
                  "transitions": st, "fused_rings": m_f, "cam_h": [round(a["cam_h"], 3) for a in aggs],
                  "extent_m": {"x": [x0 * RES, (x0 + H) * RES], "y": [y0 * RES, (y0 + W) * RES]}})
        result["prior"] = str(npz_path)

    (out_dir / f"metrics_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[out] {out_dir / f'compare_{tag}.png'}")


if __name__ == "__main__":
    main()