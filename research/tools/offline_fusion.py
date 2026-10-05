# -*- coding: utf-8 -*-
r"""离线多视角融合（固化档第一步）—— 用**已采纳（merged）位姿**重建三态地图，清远场假障碍。

    python research/tools/offline_fusion.py 20261001_044153                  # auto：会话有 merged 就用它
    python research/tools/offline_fusion.py 20261001_044153 --pose final     # A/B 对照：录制自带 final_poses
    python research/tools/offline_fusion.py 20261005_235232 --session 20261005_235237 --pose merged

**它是什么**：把 2026-10-02 实验 `Docs/离线多视角融合v1-假障碍清除（2026-10-02）.md`
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
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_mapping import MapperConfig, _ground_correction   # noqa: E402
from backend.nav_xsession import session_pose_table                # noqa: E402

C = MapperConfig()
RES = C.res_m
NEAR_M, MID_M = 1.5, 3.0
MIN_PTS = C.min_pts
RINGS = (1.5, 2.5, 3.0)
KOFF = 1 << 20


def load_events(rec: Path) -> list[dict]:
    return [json.loads(l) for l in
            (rec / "events.jsonl").read_text(encoding="utf-8").strip().splitlines() if l.strip()]


def find_world(sid: str) -> Path | None:
    """在 ``navmesh_memory/*/sessions/<sid>`` 里找这个会话属于哪个世界。"""
    for w in sorted((ROOT / "navmesh_memory").glob("*/sessions/" + sid)):
        if w.is_dir():
            return w.parents[1]
    return None


def table_poses(world_dir: Path, sid: str) -> tuple[dict[int, np.ndarray], list[int], str] | None:
    """会话位姿表 → ({kf_id: T}, 排序后的 id 列表, 来源描述)。表读不到返回 None。"""
    tab = session_pose_table(world_dir, sid)
    if tab is None:
        return None
    ids = [int(k) for k in tab["ids"]]
    pose_of = {int(k): np.asarray(T, np.float64) for k, T in zip(ids, tab["T_map"])}
    base = tab.get("base_sid") or "-"
    return pose_of, sorted(pose_of), f"{tab['source']}(base={str(base)[-6:]})"


def load_poses(rec: Path, events: list[dict], mode: str, world_dir: Path | None, sid: str):
    """返回 (kf_npz, pose, src, fallback_count)。

    merged 模式绝不混 gauge：表里没有的 k（如 refresh 撤下的帧）用**最近的表内 id** 兜底，
    不从 kf ``T_map``（原表系）取；final 模式保持原脚本行为（final_poses → kf T_map）。
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
        merged_of, _ids, src = got
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

        return kf_npz, pose, f"rigid({src})", n_fallback

    if mode in ("auto", "merged") and world_dir is not None:
        got = table_poses(world_dir, sid)
        if got is not None:
            pose_of, ids_sorted, src = got
            arr_ids = np.asarray(ids_sorted)

            def pose(k: int) -> np.ndarray:
                nonlocal n_fallback
                hit = pose_of.get(int(k))
                if hit is not None:
                    return hit
                n_fallback += 1
                j = int(np.clip(np.searchsorted(arr_ids, int(k)), 0, len(arr_ids) - 1))
                return pose_of[int(arr_ids[j])]

            return kf_npz, pose, src, n_fallback
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

    return kf_npz, pose, src, n_fallback


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


def kf_rows(k, pts, T, cam_h):
    """单关键帧 → 每格每带计数行：(cu, rows[6] = g0,g1,g2, o0,o1,o2)。"""
    R, t = T[:3, :3], T[:2, 3]
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
    return cu, rows


def aggregate(rec: Path, mode: str, world_dir: Path | None, sid: str) -> dict:
    events = load_events(rec)
    kf_npz, pose, src, n_fb = load_poses(rec, events, mode, world_dir, sid)
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
    traj = []
    for k in alive:
        z = kf_npz(k)
        T = pose(k)
        r = kf_rows(k, z["pts"], T, cam_h)
        if r is not None:
            rows_all.append(r)
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
    return dict(cam_h=cam_h, cam_h_est_s=round(n_est, 1), fusion_s=round(dt, 1),
                n_kf=len(alive), n_refresh=n_refresh, pose_src=src, pose_fallback=n_fb,
                ix=ix, iy=iy, G=G, S=S, traj=np.asarray(traj))


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
    return (cv2.filter2D(occ.astype(np.uint8), -1, np.ones((3, 3), np.float32)) >= 3)


def classify(gband, oband, sband):
    G = gband.sum(axis=2)
    O = oband.sum(axis=2)
    o_near, o_mid, o_far = oband[:, :, 0], oband[:, :, 1], oband[:, :, 2]
    g_near, g_mid = gband[:, :, 0], gband[:, :, 1]
    sup_o_mid = sband[:, :, 4]

    occ_b = majority((O >= MIN_PTS) & (O >= C.occ_ground_ratio * G))

    # RangeMax 复刻（run5-7 口径）：只用近+中距观测建图，远场票全部丢弃
    G_nm = gband[:, :, :2].sum(axis=2)
    O_nm = oband[:, :, :2].sum(axis=2)
    occ_b3 = majority((O_nm >= MIN_PTS) & (O_nm >= C.occ_ground_ratio * G_nm))

    near_ok = o_near >= MIN_PTS  # 近距确认：1.5 m 内假障碍 0.06%（run5-7），近看票本身即决定性
    # 中距半可靠：除多帧确认外，仍须过地面压制（障碍票 ≥ 30% 地面票），防止纯地面格被倾斜点翻成障碍
    mid_ok = (o_mid >= MIN_PTS) & (sup_o_mid >= 2) & (O >= C.occ_ground_ratio * G)
    occ_f = majority(near_ok | mid_ok)

    g_evid = (g_near + g_mid) >= MIN_PTS
    free_f = (G > 0.5) & ~occ_f
    labels_f = np.zeros(occ_f.shape, np.uint8)  # 0 unk 1 free 2 occ
    labels_f[free_f] = 1
    labels_f[occ_f] = 2

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec", help="录制目录名（navmesh_recordings/ 下）")
    ap.add_argument("--session", default=None,
                    help="会话 sid（默认与录制同名；录制名与会话 sid 不同时指名，如 20261005_235232 ↔ 20261005_235237）")
    ap.add_argument("--world", default=None, help="世界目录（默认自动在 navmesh_memory/*/sessions/<sid> 找）")
    ap.add_argument("--pose", choices=("auto", "final", "merged", "rigid"), default="auto",
                    help="auto=会话在世界上就用已采纳(merged)表；final=录制自带 final_poses；"
                         "merged=强制用会话表；rigid=原表位姿只加并树的刚体分量（保会话内部形状）")
    ap.add_argument("--out", default=None, help="输出目录（默认 .tmp/offline_fusion）")
    args = ap.parse_args()

    rec = ROOT / "navmesh_recordings" / args.rec
    sid = args.session or args.rec
    world_dir = Path(args.world) if args.world else None
    if world_dir is None and args.pose in ("auto", "merged", "rigid"):
        world_dir = find_world(sid)
    out_dir = Path(args.out) if args.out else (ROOT / ".tmp" / "offline_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate(rec, args.pose, world_dir, sid)
    x0, y0, gband, oband, sband = dense(agg)
    occ_b, labels_f, occ_b3, st = classify(gband, oband, sband)
    H, W = occ_b.shape

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
    tag = f"{args.rec}_pose{args.pose}"
    cv2.imwrite(str(out_dir / f"compare_{tag}.png"), canvas)

    result = dict(rec=args.rec, session=sid,
                  world=str(world_dir) if world_dir else None, pose_mode=args.pose,
                  cam_h=round(agg["cam_h"], 3), cam_h_est_s=agg["cam_h_est_s"],
                  fusion_s=agg["fusion_s"], n_kf=agg["n_kf"], n_refresh=agg["n_refresh"],
                  pose_src=agg["pose_src"], pose_fallback=agg["pose_fallback"],
                  transitions=st, baseline_all_rings=m_b, rmax3_rings=m_3, fused_rings=m_f)
    (out_dir / f"metrics_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[out] {out_dir / f'compare_{tag}.png'}")


if __name__ == "__main__":
    main()