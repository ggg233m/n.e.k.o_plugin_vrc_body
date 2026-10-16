# -*- coding: utf-8 -*-
r"""立体亚像素审计：SGBM 到底有多准？双眼有没有被不对称处理？

回答两个问题，**都必须先过用法自检**：

  Q1  SGBM 的**绝对**亚像素偏差是多少？
      做法：取左图 L，按**已知**视差 d_true 造一张合成 R_syn，跑**完全相同**的
      SGBM + 亚像素细化流水线，与 d_true 比。合成对里两眼来自同一张图
      ⇒ 不存在任何不对称处理 ⇒ 测到的就是 SGBM 的偏差。

  Q2  真实左右眼有没有被**不对称**处理（逐眼渲染 / TAA）？
      做法：同一流水线跑真实左右对，比较 resid = d* − d_sgbm 随 frac(d) 的
      **基频幅值**。分箱宽度会改变量到的摆幅，所以用与分箱无关的量：

          resid(frac) ≈ C + A·sin(2π·frac) + B·cos(2π·frac)
          amp = sqrt(A^2 + B^2)

      真实 amp ≈ 合成 amp  ⇒ 摆动全是估计器的，图像无不对称
      真实 amp 远大于合成 amp ⇒ 图像确实被不对称处理过

为什么这题值得固化
------------------
2026-10-17 用它结掉了「飞点是不是 SGBM 太差」：结论是**都不是**，
而是视差预算（f·b = 25.5，5 m 处 1 px = 0.99 m）。
见 `Docs/飞点归因终局-视差预算（2026-10-17）.md`。
以后任何「换个匹配器会不会更准」的问题都该先跑这个。

⚠️ **本工具不产生真值**
    「真值」一律是**合成对照**（同一张图造的两眼），不是物理真值。
    它同时含 SGBM 偏差与合成插值误差（三次插值会软化图像）
    ⇒ Q1 的读数是**上界**，不是精确值。

⚠️ **本工具测的是「两眼之间」的不对称**
    若采集端对**两只眼施加了同样的**处理（例如 TAA 对称作用于两眼），
    本工具**看不出来**。要验那一条必须改采集端重录一场（录像里没有
    「未处理」的对照图）。这是明确留空的边界，不要把它读成「图像没问题」。

用法::

    python research/tools/stereo_subpixel_audit.py --rec 20261013_223033
    python research/tools/stereo_subpixel_audit.py --rec 20261013_223033 --syn-frames 5
    python research/tools/stereo_subpixel_audit.py --rec 20261013_223033 --no-synth

输出：
  0. 用法自检        极线对齐（最佳 dy 必须 = 0）+ warp 自检（d±1 必须更差）
  1. Q1 绝对偏差     amp(d_sgbm − d_true)，逐 frac(d) 的偏差曲线
  2. Q2 不对称判据   amp 真实 / amp 合成，及其裁定
  3. 深度换算        把 px 偏差按 z^2/(f·b) 换成米

已知边界（见文档 §八）
----------------------
* 真值是合成对照，不是物理真值；amp(d_sgbm − d_true) 是**上界**。
* 只在一场数据上验证过（20261013_223033）。跨数据集未复现。
* 合成对照的「真值」依赖三次插值，插值本身会软化右图。
* 改抽样步长会改变样本量；换值时请一起看输出里的 n。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_mapping import (disparity_range_px, make_sgbm,      # noqa: E402
                                 stereo_disparity)

# 亚像素细化的候选偏移与搜索窗 —— 与 Q2 的判据绑定，不要随手改
CAND = np.arange(-2.0, 2.0001, 0.25)
WIN = np.arange(-2, 3)


def sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, order: int = 3) -> np.ndarray:
    """按浮点像素坐标双/三次插值采样。"""
    return map_coordinates(img, [ys, xs], order=order, mode="nearest")


def refine(Lf, Rf, d, ys, xs, dv):
    """5x5 SSD + 抛物线亚像素细化 → (d*, 有效下标)。

    ⚠️ 与 SGBM 的整数量化无关：这是**独立**求出来的亚像素值，
    所以 d* − d_sgbm 才能拿来当「相位是否被扰动」的判据。
    """
    WX, WY = np.meshgrid(WIN, WIN)
    ssd = np.zeros((len(CAND), len(xs)))
    for ci, off in enumerate(CAND):
        acc = np.zeros(len(xs))
        for kx in range(WX.size):
            gx_ = xs + WX.ravel()[kx]
            gy_ = ys + WY.ravel()[kx]
            lv = sample(Lf, gx_, gy_, order=1)
            rv = sample(Rf, gx_ - dv + off, gy_, order=1)
            acc += (lv - rv) ** 2
        ssd[ci] = acc
    jm = ssd.argmin(0)
    vj = (jm > 0) & (jm < len(CAND) - 1)
    ii = np.flatnonzero(vj)
    j = jm[vj]
    s0, s1, s2 = ssd[j - 1, ii], ssd[j, ii], ssd[j + 1, ii]
    den = s0 - 2 * s1 + s2
    delta = np.zeros(len(j))
    good = np.abs(den) > 1e-12
    delta[good] = 0.5 * (s0[good] - s2[good]) / den[good]
    return (dv[ii] - CAND[j] - delta * 0.25), ii


def fundamental(frac: np.ndarray, resid: np.ndarray):
    """拟合基频 → (amp, 常数项)。**与分箱无关**，这是 Q2 判据的关键。

    为什么不用分箱：箱宽会改变量到的摆幅（宽箱把摆动平均掉），
    真实对与合成对照若用了不同箱宽，两者**不可比**。
    """
    w = 2 * np.pi * frac
    X = np.column_stack([np.ones(len(frac)), np.sin(w), np.cos(w)])
    coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return float(np.hypot(coef[1], coef[2])), float(coef[0])


def tex_mask(Lu, xv, yv, pct=40.0):
    """纹理掩膜：只保留梯度能量高于 pct 分位的像素（无纹理处视差本就不定）。"""
    gs = cv2.Sobel(Lu, cv2.CV_32F, 1, 0, ksize=3)      # ⚠️ 不接受 float64
    e = uniform_filter(gs.astype(np.float64) ** 2, 7)
    ev = sample(e, xv, yv, order=1)
    return ev > np.percentile(ev, pct), ev


def load_gray(path: Path):
    """读灰度图 → (uint8 原图, float64 副本)。

    ⚠️ SGBM 要求 CV_8U；插值采样要 float。**两者不能是同一个数组**
    （踩过：直接 .astype(float64) 喂 SGBM ⇒ Assertion failed left.depth() == CV_8U）。
    """
    u = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if u is None:
        return None, None
    return u, u.astype(np.float64)


def self_checks(Lu, Ru, matcher, nd, stride=37):
    """用法自检。任一不过就没有资格出结论。"""
    H, W = Lu.shape
    Lf, Rf = Lu.astype(np.float64), Ru.astype(np.float64)
    d = stereo_disparity(Lu, Ru, matcher)
    ys, xs = np.mgrid[0:H, 0:W]
    m = ((d > 2) & (d < nd - 2) & (ys >= 4) & (ys < H - 4)
         & (xs >= 8) & (xs < W - 8))
    sub = np.flatnonzero(m.ravel())[::stride]
    yy = (sub // W).astype(np.float64)
    xx = (sub % W).astype(np.float64)
    dd = d[yy.astype(int), xx.astype(int)]

    print()
    print("=== 0. 用法自检（不过就没有资格出结论）===")
    print("  (1) 极线对齐：最佳匹配必须落在 dy=0")
    prof = []
    for dy in range(-3, 4):
        a = sample(Lf, xx, yy)
        b = sample(Rf, xx - dd, yy + dy)
        prof.append((dy, float(np.mean(np.abs(a - b)))))
    best = min(prof, key=lambda t: t[1])[0]
    for dy, v in prof:
        print("      dy=%+d  |L-R| 均值 %7.3f%s" % (dy, v, "   <- 最佳" if dy == best else ""))
    ok_epi = (best == 0)

    print("  (2) warp 自检：用该视差 warp 的残差必须**小于** d±1")
    resid = {}
    for off in (-2.0, -1.0, 0.0, 1.0, 2.0):
        a = sample(Lf, xx, yy)
        b = sample(Rf, xx - dd + off, yy)
        resid[off] = float(np.mean(np.abs(a - b)))
    b0 = min(resid, key=lambda k: resid[k])
    for off in (-2.0, -1.0, 0.0, 1.0, 2.0):
        print("      d%+0.0f  |L-warp(R)| 均值 %7.3f%s"
              % (off, resid[off], "   <- 最小" if off == b0 else ""))
    ok_warp = (b0 == 0.0)

    print("  => 极线 %s / warp %s"
          % ("对齐 OK" if ok_epi else "**未对齐**",
             "符号正确 OK" if ok_warp else "**符号或坐标系错**"))
    if not (ok_epi and ok_warp):
        print("  !! 自检未过：后面的数字**不可信**。先查内参/校正/坐标约定。")
    return ok_epi and ok_warp


def run_synth(kfs, kdir_img, matcher, nd, fb, fx, cx, cy, n_frames, ds,
              stride, seed, pct):
    """Q1：合成对照。返回 (样本数组, 逐 frac 统计行)。"""
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(kfs), min(n_frames, len(kfs)), replace=False).tolist())
    S = []
    for i in pick:
        k = int(kfs[i].stem)
        Lu, Lf = load_gray(kdir_img / ("%06d_L.png" % k))
        if Lu is None:
            continue
        H, W = Lf.shape
        ys, xs = np.mgrid[0:H, 0:W]
        reg = (ys > 60) & (ys < H - 60) & (xs > 120) & (xs < W - 400)
        idx = np.flatnonzero(reg.ravel())[::stride]
        yv = (idx // W).astype(np.float64)
        xv = (idx % W).astype(np.float64)
        tex, _ = tex_mask(Lu, xv, yv, pct)
        for dvt in ds:
            Rs = sample(Lf, xs + dvt, ys, order=3)
            Ru = np.clip(np.round(Rs), 0, 255).astype(np.uint8)
            d = stereo_disparity(Lu, Ru, matcher)
            dvv = d[yv.astype(int), xv.astype(int)]
            ok = (dvv > 3) & (dvv < nd - 3) & tex
            if ok.sum() < 150:
                continue
            dstar, ii = refine(Lf, Rs, d, yv[ok], xv[ok], dvv[ok])
            S.append(np.column_stack([
                np.full(len(dstar), dvt),
                np.full(len(dstar), np.mod(dvt, 1.0)),
                dvv[ok][ii], dstar]))
        print("      合成 kf%-6d 累计样本 %d" % (k, sum(len(s) for s in S)), flush=True)
    return (np.vstack(S) if S else np.zeros((0, 4))), pick


def run_real(kfs, kdir_img, matcher, nd, n_frames, stride, seed, pct):
    """Q2 的真实对：同一流水线，但两眼是真实录到的。"""
    rng = np.random.default_rng(seed + 1)
    pick = sorted(rng.choice(len(kfs), min(n_frames, len(kfs)), replace=False).tolist())
    R = []
    for i in pick:
        k = int(kfs[i].stem)
        Lu, Lf = load_gray(kdir_img / ("%06d_L.png" % k))
        Ru, Rf = load_gray(kdir_img / ("%06d_R.png" % k))
        if Lu is None or Ru is None:
            continue
        H, W = Lf.shape
        ys, xs = np.mgrid[0:H, 0:W]
        d = stereo_disparity(Lu, Ru, matcher)
        ok = ((d > 3) & (d < nd - 3) & (ys >= 5) & (ys < H - 5)
              & (xs >= 10) & (xs < W - 10))
        idx = np.flatnonzero(ok.ravel())[::stride]
        yv = (idx // W).astype(np.float64)
        xv = (idx % W).astype(np.float64)
        dvv = d[yv.astype(int), xv.astype(int)]
        tex, _ = tex_mask(Lu, xv, yv, pct)
        dstar, ii = refine(Lf, Rf, d, yv[tex], xv[tex], dvv[tex])
        R.append(np.column_stack([
            dvv[tex][ii], np.mod(dvv[tex][ii], 1.0), dvv[tex][ii], dstar]))
        print("      真实 kf%-6d 累计样本 %d" % (k, sum(len(r) for r in R)), flush=True)
    return (np.vstack(R) if R else np.zeros((0, 4))), pick


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rec", required=True, help="录制目录名（navmesh_recordings/ 下）")
    ap.add_argument("--fx", type=float, default=202.5)
    ap.add_argument("--baseline", type=float, default=0.126)
    ap.add_argument("--cx", type=float, default=360.0)
    ap.add_argument("--cy", type=float, default=202.5)
    ap.add_argument("--syn-frames", type=int, default=3)
    ap.add_argument("--real-frames", type=int, default=12)
    ap.add_argument("--stride-syn", type=int, default=23)
    ap.add_argument("--stride-real", type=int, default=29)
    ap.add_argument("--d-min", type=float, default=5.0)
    ap.add_argument("--d-max", type=float, default=12.0)
    ap.add_argument("--d-step", type=float, default=0.1)
    ap.add_argument("--tex-pct", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-synth", action="store_true", help="只跑 Q2（用缓存的合成样本）")
    ap.add_argument("--syn-cache", default=".tmp/_stereo_syn.npy")
    a = ap.parse_args()

    rec = ROOT / "navmesh_recordings" / a.rec
    kdir, kdir_img = rec / "kf", rec / "kf_img"
    if not kdir.is_dir():
        raise SystemExit("[fail] 没有这个录制：%s" % rec)
    kfs = sorted(kdir.glob("*.npz"), key=lambda p: int(p.stem))
    fb = a.fx * a.baseline
    nd = disparity_range_px(a.fx)
    matcher = make_sgbm(nd)
    print("[cfg] rec=%s  帧 %d  ND=%d  f·b=%.3f  （1 px ⇒ z^2/%.1f）"
          % (a.rec, len(kfs), nd, fb, fb))

    f0 = kdir_img / ("%06d_L.png" % int(kfs[0].stem))
    r0 = kdir_img / ("%06d_R.png" % int(kfs[0].stem))
    Lu0, _ = load_gray(f0)
    Ru0, _ = load_gray(r0)
    if Lu0 is None or Ru0 is None:
        raise SystemExit("[fail] 缺 kf_img（本工具需要原始左右图）：%s" % f0)
    self_checks(Lu0, Ru0, matcher, nd)

    DS = np.round(np.arange(a.d_min, a.d_max + 1e-9, a.d_step), 2)
    t0 = time.perf_counter()
    cache = ROOT / a.syn_cache
    if a.no_synth and cache.exists():
        SYN = np.load(cache)
        print()
        print("[synth] 复用缓存 %s（%d 样本）" % (cache.name, len(SYN)))
    else:
        print()
        print("=== 1. 合成对照（Q1）===")
        print("    扫 d_true ∈ [%.1f, %.1f] 步长 %.2f，%d 帧"
              % (a.d_min, a.d_max, a.d_step, a.syn_frames))
        SYN, _ = run_synth(kfs, kdir_img, matcher, nd, fb, a.fx, a.cx, a.cy,
                           a.syn_frames, DS, a.stride_syn, a.seed, a.tex_pct)
        if len(SYN) < 1000:
            raise SystemExit("[fail] 合成样本太少（%d），检查图像与掩膜" % len(SYN))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, SYN)
    print("    n=%d  (%.0fs)" % (len(SYN), time.perf_counter() - t0))

    # ⚠️ 两个**不同**的量，别混（踩过：合并时把前者当成后者的合成列）
    #   dm : d_sgbm − d_true   —— Q1 的绝对偏差
    #   de : d* − d_sgbm       —— Q2 的估计器摆动，真实对要与**合成对的同一量**比
    aM, cM = fundamental(SYN[:, 1], SYN[:, 2] - SYN[:, 0])
    aS, cS = fundamental(SYN[:, 1], SYN[:, 3] - SYN[:, 2])
    print()
    print("    Q1  SGBM 绝对亚像素偏差 amp(d_sgbm − d_true) = %.4f px   常数项 %+.4f px"
          % (aM, cM))
    print("    Q2  估计器固有摆动   amp(d* − d_sgbm)       = %.4f px   常数项 %+.4f px"
          % (aS, cS))
    print("    %-12s %7s %16s %16s" % ("frac(d)", "n", "d_sgbm−d_true", "d*−d_true"))
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (SYN[:, 1] >= lo) & (SYN[:, 1] < lo + 0.1)
        if m.sum() < 3:
            continue
        print("    %.1f-%.1f      %7d %+15.4f %+15.4f"
              % (lo, lo + 0.1, int(m.sum()),
                 np.median(SYN[m, 2] - SYN[m, 0]),
                 np.median(SYN[m, 3] - SYN[m, 0])))

    print()
    print("=== 2. 双眼不对称检验（Q2）===")
    REAL, _ = run_real(kfs, kdir_img, matcher, nd, a.real_frames,
                       a.stride_real, a.seed, a.tex_pct)
    if len(REAL) < 1000:
        raise SystemExit("[fail] 真实对样本太少（%d）" % len(REAL))
    aR, cR = fundamental(REAL[:, 1], REAL[:, 3] - REAL[:, 2])
    print()
    print("    %-34s %9s %14s %12s" % ("", "n", "amp(d*−d_sgbm)", "常数项"))
    print("    %-34s %9d %14.4f %12.4f"
          % ("合成对照（同一张图造的两眼）", len(SYN), aS, cS))   # ⚠️ 必须是 aS 不是 aM
    print("    %-34s %9d %14.4f %12.4f" % ("真实左右对", len(REAL), aR, cR))
    ratio = aR / max(aS, 1e-9)
    print()
    print("    比值 真实/合成 = %.3f" % ratio)
    if ratio < 1.5:
        print("    => 摆动 <= 估计器固有水平 ⇒ **图像没有可测的不对称**（H 证伪）")
    else:
        print("    => 真实摆动明显更大 ⇒ 图像**确有**不对称处理")
    print("    ⚠️ 这只测「两眼之间」。对称地糊两眼看不出来，见模块 docstring。")

    print()
    print("=== 3. 深度换算（把 px 换成米）===")
    print("    %-10s %10s %14s %14s" % ("z", "视差", "1 px ⇒", "%.4f px ⇒" % aM))
    for z in (1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        d = fb / z
        print("    %-10.1f %10.2f %13.3f m %13.3f m"
              % (z, d, z * z / fb, aM * z * z / fb))   # 深度换算按 Q1 的绝对偏差 aM
    print()
    print("[done] %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
