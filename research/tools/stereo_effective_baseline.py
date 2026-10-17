# -*- coding: utf-8 -*-
r"""生效基线探针：从地板平面反推相机高与「实际生效的」双目基线。

回答一个问题：**渲染实际用的基线 B 是多少？**

为什么不能用 sensors.baseline_m
--------------------------------
那个数是 getEyeToHeadTransform 读回来的 —— **接口声明**，不是渲染证据。
实测 15/15 场逐位相同（0.12600000202655792 = 2 x float32(0.063)），
**连 avatar 缩放那一场也是同一个值**。要裁「渲染实际用了什么」，只能用图像。

判据
----
地板是水平面。相机离地 h，基线 B，图像行 v：

    d(v) = (B*cos(theta)/h) * (v - cy)  +  (fx*B*sin(theta)/h)
    斜率 s = B*cos(theta)/h      截距 b = fx*B*sin(theta)/h
    =>  tan(theta) = b / (fx*s)
    =>  rho = B/h = s * sqrt(1 + tan^2(theta)) = s / cos(theta)

rho 只由「同一坐标系里两个长度之比」决定 => 对任何整体缩放不变。
这正是它能跨头像尺度直接比较的原因，而且不需要先知道 B 或 h。
配合声明的基线：h_track = b_declared / rho。

绝对刻度：若相机在头像眼位（VRChat 语义），则 B_world = rho * eye_height。
这条依赖外部前提，见「边界」。

用法::

    python research/tools/stereo_effective_baseline.py --rec 20261013_223033
    python research/tools/stereo_effective_baseline.py --rec A --rec B --rec C
    python research/tools/stereo_effective_baseline.py --rec A --frames 60 --compare-camh

输出：
  0. 用法自检   图像重算 vs 存盘 pts（逐位一致才说明两者同源；不过就没资格下结论）
  1. 逐帧拟合   rho / 俯仰 / 内点率；异常帧会列出来
  2. 结果       rho 中位 + 自举 95% CI + 俯仰无关的 h_track
  3. 对比       多场时给两两比值，并对两个假设各报 sigma
  4. 绝对刻度   有眼高记录时给 B_world，并把 world_scale 的张力一起报出来

边界（不要读成「已经清楚」）
--------------------------
* rho 是比值。它单独**分不开**「B 与 h 都缩放」与「两个都不动」。
  把后者排除掉要靠**另一条事实**：相机在头像眼位 => 眼高变则 h 变。
  输出里的 sigma 只在**这个前提成立时**才是指向 B 的证据。
* 绝对刻度依赖「相机在眼位」，本工具没有独立锚验证它。
  2026-10-18 实测：默认头像 B_world = 9.00 cm，而账面 0.126 x 0.755 = 9.51 cm
  —— 差 5.4%，反推 world_scale = 0.714 而实物尺子外锚是 0.762。**未定案**。
* 需要 kf_img/ 存图（record_images 是选项，默认关）。全仓 15 场里只有 4 场有。
* 地板必须占满所取行带；家具/地毯覆盖严重的帧会被内点率筛掉。
* 只验过 720 采集宽度。mip1（1440）下 rho 是否仍不变未验。

相关的既有坑（都踩过）
----------------------
* 不做俯仰修正会得出相反结论：不修正的「两行读数」版本给出比值 1.134，
  而正确的比值是 1.001。修正项就是上面那条 cos(theta)。
* 含中文的路径不能用 cv2.imread：它在 Windows 上返回 None 且不抛异常，静默丢一场。
* estimate_cam_h **不是**俯仰无关量：它取 pts[:,2] 的众数，而地板点的 pts[:,2] 随
  俯仰在 1.248-1.868 之间移动。本工具的 h_track 才是俯仰无关的。
  （--compare-camh 会把两者并排打出来。）

依据：Docs/图像视差实测-生效基线随头像缩放（2026-10-18）.md
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

from backend.nav_mapping import (  # noqa: E402
    disparity_range_px, make_sgbm, stereo_disparity, stereo_points)

# 采集端内参：镜像路径恒为 720x405 / fx 202.5（见 FACTS 一）。cy 不写死，
# 一律按画幅中位取 —— 改分辨率时自动跟着走。
FX = 202.5
# 驱动声明的基线（追踪米）。它 15/15 场不变，正是「声明 != 生效」的那个数。
B_DECLARED = 0.126


def imread_unicode(path: Path) -> np.ndarray | None:
    """读图，支持非 ASCII 路径。

    不能用 cv2.imread(str(path))：Windows 上 OpenCV 走窄字符 API，含中文的路径
    **返回 None 且不抛异常** —— 本工具第一版因此静默漏掉一场录制。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def fit_floor_rho(disp: np.ndarray, cy: float, fx: float = FX, *,
                  vmin: float = 0.60, tol: float = 0.35, seed: int = 0,
                  ransac: int = 400, sample_cap: int = 30000) -> dict | None:
    """单帧：从视差图拟合地板平面 -> rho = B/h（俯仰已修正）。

    只取 v >= vmin*h 的行带（地板保证在下半幅），RANSAC 拟合
    d = s*(v-cy) + b，再解俯仰 theta = atan(b/(fx*s)) 并去倾斜：
    rho = s/cos(theta)。**这一步不能省**（省掉会偏 13%，见模块 docstring）。

    RANSAC 在抽样子集上跑（提速约 4x），最后一次在**全量**上精修。
    返回 None = 这一帧不可用（有效点太少 / 拟合退化）。
    """
    h, _w = disp.shape
    v0 = int(h * vmin)
    band = disp[v0:]
    ok = band > 1.0
    if int(ok.sum()) < 8000:
        return None
    vs, _us = np.nonzero(ok)
    v = (vs + v0).astype(np.float64)
    d = band[ok].astype(np.float64)
    x = v - cy
    n = len(x)

    rng = np.random.default_rng(seed)
    if n > sample_cap:
        pick = rng.choice(n, sample_cap, replace=False)
        xs_, ds_ = x[pick], d[pick]
    else:
        xs_, ds_ = x, d
    m_ = len(xs_)
    best = None
    for _ in range(ransac):
        i, j = rng.integers(0, m_, 2)
        if abs(xs_[i] - xs_[j]) < 25.0:      # 两点太近 -> 斜率被噪声主导
            continue
        s = (ds_[i] - ds_[j]) / (xs_[i] - xs_[j])
        b = ds_[i] - s * xs_[i]
        inl = int((np.abs(ds_ - (s * xs_ + b)) < tol).sum())
        if best is None or inl > best[0]:
            best = (inl, s, b)
    if best is None:
        return None
    _, s, b = best
    m = np.abs(d - (s * x + b)) < tol
    for _ in range(4):                       # 全量迭代重加权
        if int(m.sum()) < 500:
            return None
        A_ = np.column_stack([x[m], np.ones(int(m.sum()))])
        sol, *_ = np.linalg.lstsq(A_, d[m], rcond=None)
        s, b = float(sol[0]), float(sol[1])
        m = np.abs(d - (s * x + b)) < tol
    if abs(s) < 1e-4:
        return None
    tan_t = b / (fx * s)
    resid = d[m] - (s * x[m] + b)
    return {"rho": float(s * np.sqrt(1.0 + tan_t * tan_t)),
            "slope": float(s), "intercept": float(b),
            "pitch_deg": float(np.degrees(np.arctan(tan_t))),
            "frac": float(m.sum()) / n, "n": int(n),
            "rmse_px": float(np.sqrt(np.mean(resid ** 2)))}


def selfcheck_image_vs_pts(rec: Path, k: int, left: np.ndarray, right: np.ndarray) -> str:
    """自检：用存盘原图重算一遍点云，与 kf/<k>.npz 里存的比。

    为什么要它：本工具的前提是「kf_img 与 pts 是同一对图算出来的」。
    不是的话，从图反推的东西跟管线用的东西不是一个东西，后面全白算。
    """
    f = rec / "kf" / ("%06d.npz" % k)
    if not f.exists():
        return "跳过：没有 kf/%06d.npz" % k
    with np.load(f) as z:
        if "pts" not in z.files:
            return "跳过：npz 里没有 pts"
        pts = np.asarray(z["pts"], np.float64)
    h, w = left.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    disp = stereo_disparity(left, right, make_sgbm(disparity_range_px(FX)))
    got = stereo_points(left, right, fx=FX, cx=cx, cy=cy, baseline_m=B_DECLARED,
                        max_range_m=5.0, step=2, disp=disp).astype(np.float64)
    if len(pts) != len(got):
        return ("**不一致**：存 %d 点 / 重算 %d 点 —— 原图与点云不同源，结论无效"
                % (len(pts), len(got)))
    key = lambda a: a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]   # noqa: E731
    diff = float(np.abs(key(pts) - key(got)).max())
    tag = "逐位一致" if diff < 1e-5 else "**不一致** —— 结论无效"
    return "存 %d 点 / 重算 %d 点，max|diff| = %.2e => %s" % (len(pts), len(got), diff, tag)


def read_meta(rec: Path) -> dict:
    """从 meta.json 取头像眼高 / ScaleFactor / world_scale（缺就返回 None）。"""
    f = rec / "meta.json"
    out: dict = {"eye_height": None, "scale_factor": None, "world_scale": None,
                 "compensated": None}
    if not f.exists():
        return out
    try:
        m = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    av = m.get("avatar") or {}
    out["eye_height"] = av.get("EyeHeightAsMeters") or av.get("/avatar/eyeheight")
    out["scale_factor"] = av.get("ScaleFactor")
    out["world_scale"] = (m.get("config") or {}).get("world_scale")
    # 补偿是否生效：有 ref 字段才说明那版代码上线了（见 FACTS 二「事故档」）。
    out["compensated"] = ("avatar_ref_eye_height_m" in m)
    return out


def probe(rec: Path, args) -> dict | None:
    """单场：返回 rho 分布 + 元数据。None = 本场不可用。"""
    kfs = sorted((rec / "kf").glob("*.npz"), key=lambda p: int(p.stem))
    if not kfs:
        print("    [skip] 没有 kf/")
        return None
    img_dir = rec / "kf_img"
    if not img_dir.is_dir():
        print("    [skip] 没有 kf_img/（录制时没开 record_images）")
        return None
    nd = disparity_range_px(FX)
    matcher = make_sgbm(nd)
    meta = read_meta(rec)
    if not meta["eye_height"] and args.default_eye_height:
        # 老录制的 meta 没有 avatar 块（15 场里 12 场如此）。要跨场比就得显式假定，
        # 并让输出把它标出来 —— 不能让读者以为那是记录在案的眼高。
        meta["eye_height"] = float(args.default_eye_height)
        meta["eye_height_assumed"] = True

    first = kfs[0].stem
    L0 = imread_unicode(img_dir / ("%s_L.png" % first))
    R0 = imread_unicode(img_dir / ("%s_R.png" % first))
    if L0 is None or R0 is None:
        print("    [fail] 读不到首帧原图：%s" % (img_dir / ("%s_L.png" % first)))
        return None
    h_img, w_img = L0.shape[:2]
    cy = h_img / 2.0
    print("    [cfg] 图 %dx%d  fx=%.1f  cy=%.1f  ND=%d  声明基线 %.3f m"
          % (w_img, h_img, FX, cy, nd, B_DECLARED))
    print("    [chk] %s" % selfcheck_image_vs_pts(rec, int(kfs[0].stem), L0, R0))

    rows = []
    for kf in kfs[::max(1, args.step)]:
        k = int(kf.stem)
        L = L0 if k == int(kfs[0].stem) else imread_unicode(img_dir / ("%06d_L.png" % k))
        R = R0 if k == int(kfs[0].stem) else imread_unicode(img_dir / ("%06d_R.png" % k))
        if L is None or R is None or L.shape != R.shape:
            continue
        r = fit_floor_rho(stereo_disparity(L, R, matcher), cy, FX,
                          vmin=args.vmin, tol=args.tol, seed=args.seed)
        if r is None or r["frac"] < args.min_frac or r["rmse_px"] > args.max_rmse:
            continue
        r["k"] = k
        rows.append(r)
        if len(rows) >= args.frames:
            break
    if len(rows) < args.min_frames:
        print("    [fail] 可用帧只有 %d < %d" % (len(rows), args.min_frames))
        return None

    rho = np.array([r["rho"] for r in rows])
    rng = np.random.default_rng(args.seed + 7)
    boot = np.array([np.median(rng.choice(rho, len(rho), replace=True))
                     for _ in range(args.bootstrap)])
    med = float(np.median(rho))
    return {"rec": rec.name, "n_frames": len(rows), "meta": meta,
            "rho": med, "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "pitch": float(np.median([r["pitch_deg"] for r in rows])),
            "frac": float(np.median([r["frac"] for r in rows])),
            "rmse": float(np.median([r["rmse_px"] for r in rows])),
            "h_track": B_DECLARED / med, "rho_all": rho.tolist(), "rows": rows}


def report_compare(base: dict, other: dict, n_boot: int = 4000, seed: int = 0) -> None:
    """两条记录的两两比值 + 两个假设的 sigma。

    只对**头像眼高不同**的两场才有意义（同眼高的两场只是复现性检查）。
    H1 都缩放   => B 与 h 同乘 k => rho 不变      预期 1.000
    H2 只缩高度 => B 不动、h 乘 k => rho 乘 1/k    预期 1/k
    """
    eh_a, eh_b = base["meta"]["eye_height"], other["meta"]["eye_height"]
    if not eh_a or not eh_b:
        return
    k = float(eh_b) / float(eh_a)
    ratio = other["rho"] / base["rho"]
    # 不确定度**直接自举比值**：不要把两场的半宽拼起来。本仓的自举 CI 是不对称的
    # （下侧比上侧远），只取上侧半宽会把不确定度低估约 2x —— 2026-10-18 就是这么
    # 把 6 sigma 写成 12-17 sigma 的。
    a_rho = np.asarray(base.get("rho_all", [base["rho"]]), float)
    b_rho = np.asarray(other.get("rho_all", [other["rho"]]), float)
    rng = np.random.default_rng(seed + 13)
    boot = np.array([np.median(rng.choice(b_rho, len(b_rho), replace=True))
                     / np.median(rng.choice(a_rho, len(a_rho), replace=True))
                     for _ in range(n_boot)])
    u = float((np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2.0)
    pred_h2 = 1.0 / k
    print("    %s -> %s : 眼高 %.4f -> %.4f m (k=%.4f)" % (base["rec"], other["rec"], eh_a, eh_b, k))
    print("      比值 rho2/rho1 = %.4f +- %.4f  （比值自举 95%% CI）" % (ratio, u))
    print("      H1 都缩放   预期 %.4f   %s"
          % (1.0, "一致" if abs(ratio - 1.0) < 2 * u else "偏离 %.1f sigma" % (abs(ratio - 1.0) / u)))
    print("      H2 只缩高度 预期 %.4f   %s"
          % (pred_h2, "一致" if abs(ratio - pred_h2) < 2 * u
             else "**排除** %.1f sigma" % (abs(ratio - pred_h2) / u)))


def report_absolute(r: dict, declared_world_scale: float = 0.755) -> None:
    """有眼高记录时给 B_world，并把与账面的张力一起报出来。"""
    eh = r["meta"]["eye_height"]
    if not eh:
        print("    本场 meta 没记眼高 => 只报 rho 与 h_track，不给 B_world")
        print("    （要用假定值给 --default-eye-height）")
        return
    b_world = r["rho"] * float(eh)
    ledger = B_DECLARED * declared_world_scale
    src = "假定" if r["meta"].get("eye_height_assumed") else "meta 记录"
    print("    头像眼高（%s）= %.4f m" % (src, eh))
    print("    B_world = rho x 眼高 = %.5f x %.4f = **%.2f cm**"
          % (r["rho"], float(eh), b_world * 100))
    print("    账面   = %.3f x %.3f      = %.2f cm" % (B_DECLARED, declared_world_scale, ledger * 100))
    gap = b_world / ledger - 1.0
    print("    差 %+.1f%%  => 反推 world_scale = %.3f（账面 %.3f）"
          % (gap * 100, b_world / B_DECLARED, declared_world_scale))
    if abs(gap) > 0.02:
        print("    ⚠️ 差 >2%：**未定案**。三种可能分不开：①尺子锚（0.762）一起偏；")
        print("       ②相机不在头像眼位；③平面拟合有系统偏。见模块 docstring「边界」。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rec", action="append", required=True,
                    help="录制目录名（navmesh_recordings/ 下）；可给多个做对比")
    ap.add_argument("--frames", type=int, default=40, help="每场取多少帧")
    ap.add_argument("--min-frames", type=int, default=8, help="少于这么多帧就算本场不可用")
    ap.add_argument("--step", type=int, default=8, help="关键帧抽样步长")
    ap.add_argument("--vmin", type=float, default=0.60, help="只用 v >= vmin*H 的下半幅")
    ap.add_argument("--tol", type=float, default=0.35, help="RANSAC 内点容差（像素）")
    ap.add_argument("--min-frac", type=float, default=0.55, help="内点率下限")
    ap.add_argument("--max-rmse", type=float, default=0.60, help="拟合 RMSE 上限（像素）")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--default-eye-height", type=float, default=None,
                    help="meta 没记眼高的场次按这个值假定（输出会标「假定」）")
    ap.add_argument("--compare-camh", action="store_true",
                    help="并排打印 estimate_cam_h（俯仰相关，只为对照）")
    ap.add_argument("--json", default="", help="把结果写到该路径")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = []
    for name in args.rec:
        rec = ROOT / "navmesh_recordings" / name
        print("=== %s ===" % name)
        if not rec.is_dir():
            print("    [fail] 没有这个录制")
            print()
            continue
        r = probe(rec, args)
        print()
        if r is None:
            continue
        ms = r["meta"]
        print("  拟合 %d 帧  俯仰 p50 %+.1f deg  内点率 p50 %.1f%%  RMSE p50 %.3f px"
              % (r["n_frames"], r["pitch"], 100 * r["frac"], r["rmse"]))
        print("  **rho = B/h = %.5f**   自举 95%% CI [%.5f, %.5f]"
              % (r["rho"], r["ci_lo"], r["ci_hi"]))
        print("  h_track = %.3f / rho = **%.4f m**（俯仰无关）" % (B_DECLARED, r["h_track"]))
        if args.compare_camh:
            try:
                sys.path.insert(0, str(ROOT / "research" / "tools"))
                import pointcloud_view as V  # noqa: PLC0415
                old = V.estimate_cam_h(rec)
                print("  estimate_cam_h = %.4f m（俯仰相关，仅对照；差 %+.3f m）"
                      % (old, old - r["h_track"]))
            except Exception as exc:  # noqa: BLE001
                print("  estimate_cam_h 读不到：%s" % type(exc).__name__)
        if ms["scale_factor"] is not None and abs(float(ms["scale_factor"]) - 1.0) > 1e-3:
            print("  ⚠️ 本场 ScaleFactor = %.4f" % ms["scale_factor"])
            if not ms["compensated"]:
                print("     且 meta 无 avatar_ref_eye_height_m => **补偿未生效**")
                print("     （录于修复提交 64e609b 之前）。rho 是尺度无关量、不受影响，")
                print("     但**不要拿本场点云做绝对长度对比**。见 FACTS 二「事故档」。")
        print()
        print("  --- 绝对刻度 ---")
        report_absolute(r, float(ms["world_scale"] or 0.755))
        print()
        out.append({k: v for k, v in r.items() if k != "rows"})

    if len(out) >= 2:
        print("=== 对比（只有眼高不同的场才是指向 B 的证据）===")
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                report_compare(out[i], out[j], args.bootstrap, args.seed)
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("[json] %s" % args.json)
    print("[done] %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()