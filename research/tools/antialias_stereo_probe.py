# -*- coding: utf-8 -*-
"""游戏内抗锯齿选项会不会影响我们的点云（只读离线重算）。

背景：抗锯齿（MSAA / FXAA / TAA / 关闭）**只在游戏渲染时作用**，我们读的是
SteamVR 合成器的镜像纹理 —— 拿到的已经是渲染并后处理完的左右两张图。所以我们
既不读取、也不控制这个选项。它进入我们链路的唯一途径是：**改变 SGBM 看到的那对图**。

两条可能的路：
  路 1  对称：两眼被同样地处理（画面变软、边缘变糊），但左右一致。
        预期损失是"匹配证据变少" ⇒ 视差空洞变多 ⇒ 点变少。
  路 2  不对称：两眼被处理得不一样（同一道屏幕空间滤波在两眼的亚像素相位不同）
        ⇒ 右图相对左图多一个虚假水平位移 ⇒ 视差被整体带偏。

本探针用 20261010_033237 里真实录到的 720x405 左右原图分别量这两条路：

  M0  真实这对图的两眼一致性（**直接测量，不是模拟**）—— 路 2 的现状。
  M1  对称高斯模糊的剂量-响应 —— 路 1。这是抗锯齿**最忠实**的代理：
      模糊正是 AA 对像素做的事，且高斯是全图模糊，比 MSAA（只动边缘像素）更狠，
      所以给的是上界。
  M2  只模糊一只眼 —— 路 2 的最坏情况（现实中不会这么严重，作对照）。
  M3  右眼亚像素整体平移 —— 路 2 的机制本身。

判据是**点云**里的东西：障碍带格子的数量与位置，以及地面点的密度。

用法：
  .venv/Scripts/python.exe -W ignore research/tools/antialias_stereo_probe.py --n 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from backend.nav_mapping import disparity_range_px, make_sgbm, stereo_disparity

FX = 202.5
CX, CY = 360.0, 202.5
BASELINE = 0.126
RES_M = 0.10
GROUND_TOL = 0.30
OBST_TOP = 2.0
CAM_H = 1.73
MIN_PTS = 3
RANGE_M = 5.0
STEP = 2


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def warp_x(img: np.ndarray, dx: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def to_points(disp: np.ndarray, step: int = STEP) -> np.ndarray:
    d = disp[::step, ::step]
    v, u = np.mgrid[0:disp.shape[0]:step, 0:disp.shape[1]:step]
    ok = (d > 1.0)
    z = FX * BASELINE / np.where(ok, d, 1.0)
    ok = ok & (z <= RANGE_M)
    z = z[ok]
    x = (u[ok] - CX) * z / FX
    y = (v[ok] - CY) * z / FX
    return np.column_stack([z, -x, -y]).astype(np.float32)


def occ_keys(pts: np.ndarray) -> np.ndarray:
    h = CAM_H + pts[:, 2]
    p = pts[(h > GROUND_TOL) & (h < OBST_TOP)]
    if len(p) == 0:
        return np.zeros(0, np.int64)
    k = (np.floor(p[:, 0] / RES_M).astype(np.int64) * 100000
         + np.floor(p[:, 1] / RES_M).astype(np.int64))
    uniq, cnt = np.unique(k, return_counts=True)
    return uniq[cnt >= MIN_PTS]


def fold(keys: np.ndarray, box: int = 5):
    """把格子键折叠到 box*0.1 m 的网格，用于容忍整体位移。

    为什么不比 0.1 m 格：4 m 处 0.1 m 只相当于 0.06 px 视差，比 SGBM 的
    量化步长（1/16 px）还小 ⇒ 逐格 IoU 会把这个场景里无关紧要的抖动
    放大成"40% 的格子变了"。box=5（0.5 m）才是能分辨"多了一片障碍"与
    "整片挪了半格"的尺度。"""
    if len(keys) == 0:
        return set()
    cx = keys // 100000
    cy = keys % 100000
    return set(zip((cx // box).tolist(), (cy // box).tolist()))


def frame_stats(disp: np.ndarray) -> dict:
    ok = disp > 1.0
    v, u = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]
    z = FX * BASELINE / np.where(ok, disp, 1.0)
    h = CAM_H - (v - CY) * z / FX
    keep = ok & (z <= RANGE_M)
    return {
        "valid_frac": float(ok.mean()),
        "inrange": int(keep.sum()),
        "ground": int((keep & (h < GROUND_TOL)).sum()),
        "band": int((keep & (h > GROUND_TOL) & (h < OBST_TOP)).sum()),
        "occ": occ_keys(to_points(disp)),
    }


def lat_ramp_median(gray: np.ndarray) -> float:
    """中位横向边缘宽度：|dI/dx| 在强边缘处的倒数。AA 会把它变宽。"""
    gx = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    vals = gx[gx > 12.0]
    if vals.size < 200:
        return float("nan")
    return float(np.median(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--dir", default="navmesh_recordings/20261010_033237/kf_img")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--stride", type=int, default=34)
    ap.add_argument("--out", default=".tmp/aa_probe")
    args = ap.parse_args()

    d = ROOT / args.dir
    names = sorted(p.name for p in d.glob("*_L.png"))
    picks = names[:: max(1, args.stride)][: args.n]
    if not picks:
        print("no *_L.png")
        return 1
    nd = disparity_range_px(FX)
    matcher = make_sgbm(nd)
    print("frames dir : " + str(d))
    print("pairs total: %d   sampled: %d" % (len(names), len(picks)))
    print("SGBM numDisparities=%d fx=%s baseline=%s step=%d res=%s band=(%s,%s) cam_h=%s"
          % (nd, FX, BASELINE, STEP, RES_M, GROUND_TOL, OBST_TOP, CAM_H))
    print("ground_tol_abs = %.2f m (cam_h - ground_tol)" % (CAM_H - GROUND_TOL))

    pairs = []
    for nm in picks:
        L = cv2.imread(str(d / nm), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(d / nm.replace("_L.png", "_R.png")), cv2.IMREAD_GRAYSCALE)
        if L is not None and R is not None:
            pairs.append((nm, L, R))
    print("loaded %d pairs" % len(pairs))

    # ---------------- M0：真实两眼的一致性 ----------------
    lap_r, ramp_r, swap_med = [], [], []
    b_valid, b_ground, b_band, b_occ = [], [], [], []
    for nm, L, R in pairs:
        ll = float(np.mean(np.abs(cv2.Laplacian(L.astype(np.float32), cv2.CV_32F, ksize=3))))
        rr = float(np.mean(np.abs(cv2.Laplacian(R.astype(np.float32), cv2.CV_32F, ksize=3))))
        if rr > 0:
            lap_r.append(ll / rr)
        ramp_l, ramp_rr = lat_ramp_median(L), lat_ramp_median(R)
        if ramp_rr == ramp_rr and ramp_l == ramp_l:
            ramp_r.append(ramp_l / ramp_rr)
        d_lr = stereo_disparity(L, R, matcher)
        st = frame_stats(d_lr)
        b_valid.append(st["valid_frac"])
        b_ground.append(st["ground"])
        b_band.append(st["band"])
        b_occ.append(len(st["occ"]))
        # 右视差用「左右翻转后互换」求，才符合 OpenCV 的正视差约定
        d_rl = stereo_disparity(cv2.flip(R, 1), cv2.flip(L, 1), matcher)[:, ::-1]
        H, W = d_lr.shape
        vv, uu = np.mgrid[0:H, 0:W]
        ui = np.rint(uu - d_lr).astype(np.int64)
        ok = (d_lr > 1.0) & (d_rl > 1.0) & (ui >= 0) & (ui < W)
        if ok.sum() > 500:
            err = np.abs(d_lr[ok] - d_rl[vv[ok], ui[ok]])
            swap_med.append((float(np.median(err)), float(np.percentile(err, 90)),
                             float((err <= 1.0).mean())))

    print("")
    print("== M0  real pair, this recording (AA setting as configured) ==")
    print("  Laplacian energy L/R     : median %.4f   (p10 %.4f  p90 %.4f)   1.0 = eyes alike"
          % (float(np.median(lap_r)), float(np.percentile(lap_r, 10)),
             float(np.percentile(lap_r, 90))))
    print("  |dI/dx| strength L/R     : median %.4f   (1.0 = same edge softness)"
          % float(np.median(ramp_r)))
    print("  LR disparity residual    : median %.3f px  p90 %.3f px  pass(<=1px) %.1f%%"
          % (float(np.median([a for a, _, _ in swap_med])),
             float(np.median([b for _, b, _ in swap_med])),
             100.0 * float(np.median([c for _, _, c in swap_med]))))
    print("  valid disparity pixels   : %.1f%% of frame" % (100 * float(np.median(b_valid))))
    print("  points in range          : ground %.0f  band %.0f (step=%d sampling)"
          % (float(np.median(b_ground)), float(np.median(b_band)), STEP))
    print("  OCC cells per keyframe   : %.0f" % float(np.median(b_occ)))

    # ---------------- M1/M2/M3 ----------------
    cases = [("M1 sym blur sigma=0.4", 0.4, 0.4, 0.0),
             ("M1 sym blur sigma=0.6", 0.6, 0.6, 0.0),
             ("M1 sym blur sigma=0.8", 0.8, 0.8, 0.0),
             ("M1 sym blur sigma=1.2", 1.2, 1.2, 0.0),
             ("M2 one eye sigma=0.6", 0.0, 0.6, 0.0),
             ("M2 one eye sigma=1.2", 0.0, 1.2, 0.0),
             ("M3 shift right 0.10 px", 0.0, 0.0, 0.10),
             ("M3 shift right 0.25 px", 0.0, 0.0, 0.25)]

    rows = []
    for label, sL, sR, dx in cases:
        g_r, b_r, v_r, iou, iou1, dd, o_r, new_b, lost_b = [], [], [], [], [], [], [], [], []
        for nm, L, R in pairs:
            base = stereo_disparity(L, R, matcher)
            Lt, Rt = blur(L, sL), blur(R, sR)
            if dx:
                Rt = warp_x(Rt, dx)
            test = stereo_disparity(Lt, Rt, matcher)
            A, B = frame_stats(base), frame_stats(test)
            if A["ground"] < 100:
                continue
            v = base > 1.0
            dd.append(float(np.median(np.abs((test - base)[v]))))
            g_r.append(B["ground"] / A["ground"])
            b_r.append(B["band"] / max(1, A["band"]))
            v_r.append(B["valid_frac"] / max(1e-6, A["valid_frac"]))
            sa, sb = set(A["occ"].tolist()), set(B["occ"].tolist())
            iou.append(len(sa & sb) / max(1, len(sa | sb)))
            fa, fb = fold(A["occ"]), fold(B["occ"])
            iou1.append(len(fa & fb) / max(1, len(fa | fb)))
            o_r.append(len(B["occ"]) / max(1, len(A["occ"])))
            new_b.append(len(fb - fa) / max(1, len(fa | fb)))
            lost_b.append(len(fa - fb) / max(1, len(fa | fb)))
        rows.append({
            "case": label, "d_track": float(np.median(dd)), "valid_r": float(np.median(v_r)),
            "ground_r": float(np.median(g_r)), "band_r": float(np.median(b_r)),
            "occ_iou": 100 * float(np.median(iou)), "occ_iou_1cell": 100 * float(np.median(iou1)),
            "occ_ratio": float(np.median(o_r)),
            "new_box_pct": 100 * float(np.median(new_b)),
            "lost_box_pct": 100 * float(np.median(lost_b)),
        })

    print("")
    print("== M1/M2/M3  injected degradation ==")
    print("%-23s%8s%8s%9s%8s%8s%10s%10s%10s"
          % ("case", "|dd|med", "valid x", "ground x", "band x", "OCC x",
             "OCC IoU", "new box%", "lost box%"))
    print("-" * 103)
    print("%-23s%8s%8s%9s%8s%8s%9.1f%%%9s%9s"
          % ("(baseline)", "-", "1.000", "1.000", "1.000", "1.000", 100.0, "-", "-"))
    for r in rows:
        print("%-23s%8.3f%8.3f%9.3f%8.3f%8.3f%9.1f%%%9.1f%%%9.1f%%"
              % (r["case"], r["d_track"], r["valid_r"], r["ground_r"], r["band_r"],
                 r["occ_ratio"], r["occ_iou_1cell"], r["new_box_pct"], r["lost_box_pct"]))

    # ---------------- M4：对回环检测（ORB）的副作用 ----------------
    # 点云不是唯一的下游：回环用 ORB 特征。模糊会吃掉角点 ⇒ 约束变弱 ⇒ 位姿漂移
    # ⇒ 关键帧摆错位置 ⇒ 点云**叠错地方**。这是与「深度误差」正交的另一条路。
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m4 = []
    for label, sL, sR in (("no blur", 0.0, 0.0), ("sym sigma=0.6", 0.6, 0.6),
                          ("sym sigma=1.2", 1.2, 1.2), ("one eye sigma=1.2", 0.0, 1.2)):
        kp_l, kp_r, good = [], [], []
        for nm, L, R in pairs:
            L2, R2 = blur(L, sL), blur(R, sR)
            a, da = orb.detectAndCompute(L2, None)
            b, db = orb.detectAndCompute(R2, None)
            if da is None or db is None:
                continue
            kp_l.append(len(a))
            kp_r.append(len(b))
            good.append(len(bf.match(da, db)))
        m4.append((label, float(np.median(kp_l)), float(np.median(kp_r)), float(np.median(good))))

    print("")
    print("== M4  loop-closure features (ORB nfeatures=1000, crossCheck Hamming) ==")
    print("%-22s%10s%10s%12s%10s" % ("case", "kp left", "kp right", "matches", "vs base"))
    print("-" * 66)
    base_m = m4[0][3] if m4 else 1.0
    for label, kl, kr, g in m4:
        print("%-22s%10.0f%10.0f%12.0f%9.3fx" % (label, kl, kr, g, g / max(1.0, base_m)))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "aa_probe.json").write_text(json.dumps({
        "frames": [n for n, _, _ in pairs], "fx": FX, "baseline": BASELINE, "step": STEP,
        "res_m": RES_M, "ground_tol": GROUND_TOL, "obst_top": OBST_TOP, "cam_h": CAM_H,
        "min_pts": MIN_PTS,
        "m0": {"lap_ratio_med": float(np.median(lap_r)),
               "lap_ratio_p10": float(np.percentile(lap_r, 10)),
               "lap_ratio_p90": float(np.percentile(lap_r, 90)),
               "grad_ratio_med": float(np.median(ramp_r)),
               "lr_residual_med_px": float(np.median(swap_med)),
               "valid_frac": float(np.median(b_valid)),
               "ground_pts": float(np.median(b_ground)),
               "band_pts": float(np.median(b_band)),
               "occ_per_frame": float(np.median(b_occ))},
        "cases": rows,
        "m4_orb": [{"case": a, "kp_left": b, "kp_right": c, "matches": d}
                   for a, b, c, d in m4]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("")
    print("wrote " + str(out / "aa_probe.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
