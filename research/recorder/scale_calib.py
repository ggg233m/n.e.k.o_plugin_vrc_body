# -*- coding: utf-8 -*-
"""scale_calib.py -- offline auto-calibration of camera height (cam_h).

North Star context
------------------
DA-V2 is a *relative* (inverse) depth model; the only metric anchor in this
pipeline is ``cam_h`` (the floor prior). The whole map scale is strictly linear
in ``cam_h`` (stage3_horizon.py:55  k = a*cam_h*fy ; pointcloud_build.py:135/139).
Therefore "auto-adapt scale" == "auto-estimate cam_h". The floor prior alone only
fixes the ratio k/cam_h, never cam_h itself, so an external metric reference is
required. The free one here is OSC path length (metres) vs the depth-reconstructed
displacement.

Method
------
For a frame pair (i, j) that has *real translation* and *fresh OSC coverage*:
  * relative rotation R_ij is known from HMD yaw (do NOT solve it);
  * build the cached-scale camera-local point cloud of frame i:
        Z = k_cached / d ,  Xc=(u-cx)*Z/fx ,  Yc=(v-vh)*Z/fy
    (k_cached, vh come from the frozen depth cache; they were fit with cam_h=1.5,
     so this cloud is in "1.5 m" units -- everything scales by s = cam_h_true/1.5);
  * the true world point is  P = C_i + R_i (s * P_i)  and also  P = C_j + R_j (s * P_j);
  * reproject a near-field seed pixel of frame i into frame j (parametrised by s,
    fixed-point iterate), read frame j depth there -> P_j (cached scale);
  * least-squares solve the single unknown s from
        s * (P_j - R_ij P_i) = R_j^T (C_i - C_j)        (true-scale translation)
  * cam_h = 1.5 * s.

Robust statistics (median + IQR + outlier trim) give the point estimate and its
spread. Failure channels (OSC holes / pure rotation / far field) are counted and
reported. Everything is parameterised; nothing in the main pipeline is touched.

Pure depth reprojection: no RGB needed (the cache has only depth). OSC supplies the
absolute scale that depth alone cannot give.

Run:
    PYTHONPATH="<repo>/.slam_probe/ov_env" \
      "C:/Users/35269/scoop/apps/python/current/python.exe" -u scale_calib.py \
      --run <RD> --cache <repo>/.tmp/_depth_cache20.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# BLAS single-thread BEFORE importing numpy (see MEMORY / pose_graph.py).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_motion import OscPath, HmdYaw  # noqa: E402
from pose_graph import dead_reckon       # noqa: E402


# 写进产物 JSON，避免下一个人再从 osc_policy 的值去推断尺度口径（曾误判过一次）。
OSC_POLICY_NOTE = (
    "osc_policy 只作用于 min_trans_m 门槛与 trans 稳定性分箱；驱动几何的航位位置 "
    "C_i/C_j 由 dead_reckon() 建立，其内部固定用 zoh 积分。所以本参数不改变 cam_h "
    "解算结果（参考 run 实测 stop vs zoh 差 0.05%：1.3056 vs 1.3049）。"
    "真正影响 cam_h 的是 max_resid_per_trans（深度链 vs OSC 链一致性门）："
    "0.50 -> 1.3049 / 0.15 -> 1.4245 / 0.10 -> 1.4436，单调上升且 IQR 收窄 3.3 倍。"
    "见 Docs/尺度标定门依赖-9%分歧（2026-09-24）.md 与 .tmp/scale_calib_gate_sweep.json。"
)

# 单一真值源：map_from_capture.py 会 import 这两个常量，用来判断它复用的历史标定
# 报告是不是在同一档门下求出来的（不同档在参考 run 上差约 9% 的 cam_h）。
DEFAULT_OSC_POLICY = "zoh"
DEFAULT_MAX_RESID_PER_TRANS = 0.15


# ----------------------------------------------------------------------------
# rotation about the vertical (camera y) axis, matching pointcloud_build.py world
# placement: world_x =  cos*y*Xc + sin*y*Z ,  world_z = -sin*y*Xc + cos*y*Z
# ----------------------------------------------------------------------------
def _R_wc(yaw):
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def _build_dr(run_dir, yaw_sign, yaw_lag_s=0.0):
    """Dead-reckoned (x, z, yaw_rad) on a 0.05 s grid; interpolate to any time.

    yaw_lag_s shifts the yaw sample (HMD yaw at time t maps to OSC time t-yaw_lag_s),
    used to probe the known OSC<->HMD ~0.5 s lag.
    """
    dr = dead_reckon(run_dir, dt=0.05, yaw_sign=yaw_sign)
    g = dr["grid"]

    def interp(t):
        t = np.asarray(t, dtype=np.float64)
        return (np.interp(t, g, dr["x"]),
                np.interp(t, g, dr["z"]),
                np.interp(t + yaw_lag_s, g, dr["yaw_rad"]))
    return interp


def estimate_cam_h(run_dir, cache_path, **params):
    """Estimate cam_h from a frozen depth cache + OSC path.

    Returns a dict with the point estimate, IQR spread, diff vs 1.5, failure
    channel counts, and sub-set stability (per time window / per translation bin).
    """
    # ---- parameters (all override-able) ------------------------------------
    p = dict(
        near_min=0.3, near_max=6.0,           # Z range kept metric (reliable band)
        gap_min_frames=3, gap_max_frames=16,  # frame-index gap window (0.15-0.80 s)
        min_trans_m=0.4,                      # real translation required between frames
        min_cov=0.5, max_gap_s=0.30,          # OSC freshness inside the pair window
        max_dyaw_deg=50.0,                    # relative yaw cap (keep reproj in-frame)
        seed_step_u=20, seed_step_v=20,       # seed-pixel subsample in frame i
        n_fp_iter=3,
        # 深度链 vs OSC 链的一致性门：median(|resid|)/|t| 必须低于此值。
        #
        # 旧默认 0.5 允许两条米制链的位移相差 50%，放进来足够多的不一致帧对，
        # 把 cam_h 系统性地**拉低**。在参考 run 上扫描该门
        # （见 .tmp/scale_calib_gate_sweep.json）：
        #   gate 0.50 -> cam_h 1.3049, IQR 0.2715, n=1218
        #   gate 0.30 -> cam_h 1.3710, IQR 0.1747, n= 805
        #   gate 0.20 -> cam_h 1.4036, IQR 0.1318, n= 545
        #   gate 0.15 -> cam_h 1.4245, IQR 0.1014, n= 376
        #   gate 0.10 -> cam_h 1.4436, IQR 0.0820, n= 179
        # 估计单调上升 10.6%，同时 IQR 收窄 3.3 倍 ⇒ 紧门是**条件数更好**，
        # 不是过拟合。0.15 是 n（376）仍远高于 min_pairs=30 的最紧工作点。
        #
        # ⚠️ 这会改变参考 run 的 cam_h 约 9%。它选的是条件数更好的工作点，
        # **并不**由此确立哪条链在米制上是对的 —— 那需要一个独立几何锚点。
        # 详见 Docs/尺度标定门依赖-9%分歧（2026-09-24）.md。
        max_resid_per_trans=DEFAULT_MAX_RESID_PER_TRANS,
        min_valid_pix=12,                     # pixels required to trust a pair
        trim_iqr_k=3.0,                       # global outlier trim (x IQR)
        min_pairs=30,                         # below this -> "not observable"
        yaw_sign=-1,                          # image-calibrated sign (yaw_sign_check.py)
        yaw_lag_s=0.0,                         # probe OSC<->HMD lag
        assumed_cam_h=1.5,
        max_anchor_frames=10 ** 9,            # smoke-test limiter (default: all)
        # NOTE: this parameter does NOT enter the scale solve. The dead-reckoned
        # positions C_i/C_j that drive the geometry come from dead_reckon(), which
        # always integrates with policy="zoh" internally (recorder/pose_graph.py).
        # osc_policy only feeds (a) the min_trans_m gate and (b) the trans
        # stability binning. Measured on the reference run, stop vs zoh changes
        # cam_h by 0.05% (1.3056 vs 1.3049) -- the old default "stop" was a
        # mislabel, not a numeric error. Default is now "zoh" so the recorded
        # label matches what the geometry actually uses.
        osc_policy=DEFAULT_OSC_POLICY,
    )
    p.update(params)

    # ---- load cache ---------------------------------------------------------
    f = np.load(cache_path, mmap_mode=None)
    D16 = f["d16"].astype(np.float32)        # (N, H, W) inverse depth (was float16)
    K = f["k"].astype(np.float64)            # cached per-frame k (cam_h assumed=1.5)
    VH = f["vh"].astype(np.float64)          # cached per-frame horizon row
    t_tele = f["t_tele"].astype(np.float64)
    fx = float(f["fx"]); fy = float(f["fy"]); cy = float(f["cy"])
    W = int(f["W"]); H = int(f["H"])
    cx = (W - 1) / 2.0
    N = D16.shape[0]
    f.close()

    osc = OscPath(run_dir)
    dr_interp = _build_dr(run_dir, p["yaw_sign"], p["yaw_lag_s"])
    osc_end = float(osc.t[-1])               # OSC coverage end (telemetry s)

    # ---- precompute per-frame pose + freshness mask -------------------------
    # valid frame = usable cache index with finite k, telemetry inside OSC span
    fr_x = np.empty(N); fr_z = np.empty(N); fr_yaw = np.empty(N)
    for c in range(N):
        xx, zz, yy = dr_interp(t_tele[c])
        fr_x[c], fr_z[c], fr_yaw[c] = xx, zz, yy
    k_ok = np.isfinite(K) & np.isfinite(VH)
    t_ok = t_tele <= (osc_end - 0.05)
    fresh = k_ok & t_ok

    # seed pixel grid (u, v) in frame i
    us = np.arange(0, W, p["seed_step_u"], dtype=np.float64)
    vs = np.arange(0, H, p["seed_step_v"], dtype=np.float64)
    UU, VV = np.meshgrid(us, vs)
    Uf = UU.ravel(); Vf = VV.ravel()

    # ---- failure counters --------------------------------------------------
    cnt = dict(cand=0, osc_hole=0, pure_rot=0, far=0, lowtrans=0,
               notfresh=0, badfit=0, weak=0, valid=0)

    s_all = []          # per-pair scale estimates (1.5-reference)
    s_t = []            # telemetry time of each pair (for time-window stability)
    s_trans = []        # OSC displacement of each pair (for trans-bin stability)

    # anchor frames = subset that can be the "i" of a pair
    anchors = [c for c in range(N) if fresh[c]]
    if len(anchors) > p["max_anchor_frames"]:
        anchors = anchors[::len(anchors) // p["max_anchor_frames"]]

    t0 = time.time()
    n_done = 0
    for ci in anchors:
        # target frames j within the gap window, after ci
        lo = ci + p["gap_min_frames"]
        hi = min(N, ci + p["gap_max_frames"] + 1)
        if hi <= lo:
            continue
        d_arr = D16[ci]
        for cj in range(lo, hi):
            if not fresh[cj]:
                continue
            cnt["cand"] += 1
            ti, tj = t_tele[ci], t_tele[cj]
            fr = osc.freshness(ti, tj)
            # freshness gate
            if fr["coverage_frac"] < p["min_cov"] or fr["max_gap_s"] > p["max_gap_s"]:
                cnt["osc_hole"] += 1
                continue
            trans = fr["dist_zoh_m"] if p["osc_policy"] == "zoh" else fr["dist_stop_m"]
            if trans < p["min_trans_m"]:
                cnt["lowtrans"] += 1
                continue
            dyaw = abs(fr_yaw[cj] - fr_yaw[ci])
            if np.degrees(dyaw) > p["max_dyaw_deg"]:
                cnt["pure_rot"] += 1
                continue

            # ---- solve s for this pair ---------------------------------------
            Ki, vhi = K[ci], VH[ci]
            Kj, vhj = K[cj], VH[cj]
            ui0 = Uf.astype(int); vi0 = Vf.astype(int)
            Zi = Ki / np.maximum(d_arr[vi0, ui0], 1e-6)  # cached depth at seed grid
            good = np.isfinite(Zi) & (Zi >= p["near_min"]) & (Zi <= p["near_max"])
            if good.sum() < p["min_valid_pix"]:
                cnt["far"] += 1
                continue
            U = Uf[good]; V = Vf[good]; Zi = Zi[good]
            Pi = np.stack([(U - cx) * Zi / fx,
                           (V - vhi) * Zi / fy,
                           Zi], axis=1)                  # (M,3) cached cam-i cloud

            # relative pose
            Rij = _R_wc(fr_yaw[ci] - fr_yaw[cj])        # cam-i -> cam-j frame
            Ci = np.array([fr_x[ci], p["assumed_cam_h"], fr_z[ci]])
            Cj = np.array([fr_x[cj], p["assumed_cam_h"], fr_z[cj]])
            t_ij_cam = _R_wc(-fr_yaw[cj]) @ (Ci - Cj)   # true-scale translation (m)
            base = (Rij @ Pi.T)                          # 3 x M, cached (s=1)

            s = 1.0
            d_j = D16[cj]
            for _ in range(p["n_fp_iter"]):
                Pj_star = (s * base).T + t_ij_cam        # cam-j frame, true scale
                Zj = Pj_star[:, 2]
                okf = Zj > 0.1
                uj = fx * Pj_star[:, 0] / np.where(okf, Zj, 1.0) + cx
                vj = fy * Pj_star[:, 1] / np.where(okf, Zj, 1.0) + vhj
                ui = np.clip(np.round(uj).astype(int), 0, W - 1)
                vi = np.clip(np.round(vj).astype(int), 0, H - 1)
                Dj = d_j[vi, ui]
                Zc_j = Kj / np.maximum(Dj, 1e-6)
                valid = okf & np.isfinite(Dj) & (Zc_j >= p["near_min"]) & (Zc_j <= p["near_max"]) \
                    & (uj >= 1.0) & (uj <= W - 2.0) & (vj >= 1.0) & (vj <= H - 2.0)
                if valid.sum() < p["min_valid_pix"]:
                    break
                Pj = np.stack([(uj - cx) * Zc_j / fx,
                               (vj - vhj) * Zc_j / fy,
                               Zc_j], axis=1)[valid]
                ray = Pj - (s * base).T[valid]          # Pj - Rij(s Pi)
                num = float(np.sum(ray @ t_ij_cam))
                den = float(np.sum(ray * ray))
                if den <= 1e-12:
                    break
                s = num / den

            if valid.sum() < p["min_valid_pix"]:
                cnt["weak"] += 1
                continue
            # consistency: residual / translation magnitude
            resid = np.linalg.norm(ray - t_ij_cam[None, :], axis=1) / \
                (np.linalg.norm(t_ij_cam) + 1e-9)
            if np.median(resid) > p["max_resid_per_trans"]:
                cnt["weak"] += 1
                continue
            cnt["valid"] += 1
            s_all.append(s)
            s_t.append(0.5 * (ti + tj))
            s_trans.append(trans)
            n_done += 1

        if (ci - anchors[0]) % 200 == 0 and ci != anchors[0]:
            el = time.time() - t0
            print("[scale_calib] frame %d/%d  valid_pairs=%d  elapsed=%.0fs"
                  % (ci, anchors[-1], n_done, el), flush=True)

    if len(s_all) < p["min_pairs"]:
        return {
            "success": False,
            "reason": "not_observable",
            "n_valid_pairs": len(s_all),
            "failure_counts": cnt,
            "params": p,
            "osc_policy_note": OSC_POLICY_NOTE,
        }

    s_all = np.array(s_all)
    med = float(np.median(s_all))
    q1 = float(np.percentile(s_all, 25))
    q3 = float(np.percentile(s_all, 75))
    iqr = q3 - q1
    # outlier trim (3 x IQR) then final stats
    lo_t, hi_t = med - p["trim_iqr_k"] * iqr, med + p["trim_iqr_k"] * iqr
    st = s_all[(s_all >= lo_t) & (s_all <= hi_t)]
    med_t = float(np.median(st))
    q1t = float(np.percentile(st, 25))
    q3t = float(np.percentile(st, 75))
    cam_h = p["assumed_cam_h"] * med_t
    cam_h_q1 = p["assumed_cam_h"] * q1t
    cam_h_q3 = p["assumed_cam_h"] * q3t

    # ---- stability: by time window ----------------------------------------
    s_t = np.array(s_t); s_trans = np.array(s_trans)
    def grp_stats(mask):
        sub = st[(s_t >= 0) & mask] if False else s_all[mask]
        if sub.size < 5:
            return None
        return {"n": int(sub.size),
                "cam_h_med": round(p["assumed_cam_h"] * float(np.median(sub)), 4),
                "cam_h_iqr": round(p["assumed_cam_h"] * float(np.percentile(sub, 75) - np.percentile(sub, 25)), 4)}
    tmin, tmax = float(s_t.min()), float(s_t.max())
    twin = []
    for q in range(4):
        a = tmin + (tmax - tmin) * q / 4.0
        b = tmin + (tmax - tmin) * (q + 1) / 4.0
        m = (s_t >= a) & (s_t < b) if q < 3 else (s_t >= a) & (s_t <= b)
        g = grp_stats(m)
        if g:
            twin.append({"window_s": [round(a, 2), round(b, 2)], **g})
    # by translation bin
    tbin = []
    edges = [0.4, 0.8, 1.2, 2.0, 1e9]
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (s_trans >= e0) & (s_trans < e1)
        g = grp_stats(m)
        if g:
            tbin.append({"trans_m": [round(e0, 2), round(e1, 2)], **g})

    return {
        "success": True,
        "cam_h_point_estimate": cam_h,
        "cam_h_iqr_lo": cam_h_q1,
        "cam_h_iqr_hi": cam_h_q3,
        "cam_h_spread_iqr": (cam_h_q3 - cam_h_q1),
        "s_point_estimate": med_t,
        "s_iqr": iqr,
        "diff_vs_assumed_1p5": cam_h - p["assumed_cam_h"],
        "rel_diff_vs_1p5": (cam_h / p["assumed_cam_h"] - 1.0),
        "n_valid_pairs": int(len(s_all)),
        "n_trimmed_pairs": int(st.size),
        "failure_counts": cnt,
        "stability_by_time_window": twin,
        "stability_by_trans_bin": tbin,
        "params": p,
        "osc_policy_note": OSC_POLICY_NOTE,
    }


def main():
    ap = argparse.ArgumentParser(description="auto-calibrate cam_h from depth+OSC")
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None, help="write report json here")
    ap.add_argument("--max-anchor-frames", type=int, default=10 ** 9)
    ap.add_argument("--gap-max-frames", type=int, default=16)
    ap.add_argument("--min-trans-m", type=float, default=0.4)
    ap.add_argument("--osc-policy", default=DEFAULT_OSC_POLICY,
                    help="只作用于 min_trans_m 门槛与 trans 分箱；驱动几何的航位"
                         "固定用 zoh，故本参数不改变 cam_h（实测差 0.05%%）")
    ap.add_argument("--max-resid-per-trans", type=float,
                    default=DEFAULT_MAX_RESID_PER_TRANS,
                    # 注意：本 help 不能出现裸 %（argparse 还会再格式化一次，
                    # 先经 .format 再被 argparse 的 % 吃会抛 unsupported format character）。
                    help="深度链 vs OSC 链一致性门（默认 {}；旧默认 0.5 会把 "
                         "cam_h 拉低约 9 个百分点）".format(DEFAULT_MAX_RESID_PER_TRANS))
    ap.add_argument("--yaw-lag-s", type=float, default=0.0)
    ap.add_argument("--yaw-sign", type=int, default=-1)
    a = ap.parse_args()

    res = estimate_cam_h(a.run, a.cache, max_anchor_frames=a.max_anchor_frames,
                         gap_max_frames=a.gap_max_frames, min_trans_m=a.min_trans_m,
                         max_resid_per_trans=a.max_resid_per_trans,
                         osc_policy=a.osc_policy, yaw_lag_s=a.yaw_lag_s,
                         yaw_sign=a.yaw_sign)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
