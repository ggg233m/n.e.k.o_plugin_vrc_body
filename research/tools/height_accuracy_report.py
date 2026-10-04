# -*- coding: utf-8 -*-
r"""高度图精度：**按观测距离分档**量，并出图。

    python research/tools/height_accuracy_report.py 20261001_044153
    python research/tools/height_accuracy_report.py            # 默认 044153 + 045615

为什么必须分档
--------------
``terrain_hist_probe`` 量到的 3x3 平面残差 ~1.5cm 是**全图混合**的，被近场格主导，
不能直接回答"S1 能在多远的地方判台阶"。而文档明确记了一条随距离退化的机制：
平移残差（回环后 0.183 m）在 5 m 处折合约 18 cm 横向位移。**这个必须实测。**

量法（不需要真值）
------------------
地面高度按构造为 0（``h`` 是离地高），所以绝对精度无法直接验。但**平整地面上，
高度图在空间上应当是平的 —— 它的空间噪声就是高度误差**。于是对每一格用 3x3 邻域
做平面拟合，残差即该处的高度不确定度；再按该格的**最近观测距离**分档。

自变量用**最近**观测距离而不是平均：远处的一次误观测不该把一个近处被反复确认的
格拖进远档。

主图
----
1. 残差中位 / p90 / p99 **随距离**的折线（+ 5cm、15cm 参考线）
2. 俯视图：观测距离热力
3. 俯视图：平面残差热力
4. 每个距离档的残差直方图叠加

图里**全用英文/数字标签**：本机控制台是 GBK，中文标签在这条链路上反复出问题。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "research" / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from terrain_hist_probe import BIN, H_LO, NBIN, cell_hists      # noqa: E402
from mapping_gate import recorded_config                           # noqa: E402

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 1e9])
MIN_PTS = 20
GROUND_PEAK = 0.12          # 只在峰高 ±12cm 的"地面附近"格上量
GRAY = (38, 38, 38)


def fields(hmap: np.ndarray, rng: np.ndarray):
    """返回 (众数峰高场, 均值场, 有效掩码)。"""
    gh, gw = hmap.shape[:2]
    w = hmap.sum(axis=2)
    sm = cv2.GaussianBlur(hmap.reshape(-1, 1, NBIN), (5, 1), 0).reshape(gh, gw, NBIN)
    big = (w >= MIN_PTS) & (sm.max(axis=2) > 0)
    peak = np.full((gh, gw), np.nan, np.float32)
    peak[big] = (sm.argmax(axis=2)[big].astype(np.float32) + 0.5) * BIN + H_LO
    centers = np.arange(NBIN) * BIN + H_LO + BIN / 2
    mean_h = np.where(w > 0, (hmap * centers).sum(axis=2) / np.maximum(w, 1e-6), np.nan)
    return peak, mean_h, big


def plane_rms(fld: np.ndarray, msk: np.ndarray) -> np.ndarray:
    """3x3 最小二乘平面的 RMS 残差；邻域不全是有效格时留 NaN（不做外推）。"""
    v = np.where(msk, np.nan_to_num(fld), 0.0).astype(np.float32)
    m = msk.astype(np.float32)
    ker = np.ones((3, 3), np.float32)
    n = cv2.filter2D(m, -1, ker, borderType=cv2.BORDER_CONSTANT)
    s1 = cv2.filter2D(v, -1, ker, borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.filter2D(v * v, -1, ker, borderType=cv2.BORDER_CONSTANT)
    out = np.full(fld.shape, np.nan, np.float32)
    ok = n == 9
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(s2[ok] / n[ok] - (s1[ok] / n[ok]) ** 2, 0.0)
    out[ok] = np.sqrt(6.0 * var / 9.0)
    return out


def colorize(a: np.ndarray, vmin: float, vmax: float, msk: np.ndarray) -> np.ndarray:
    """把标量场画成伪彩（jet），无效格留深灰。"""
    t = np.clip((a - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~msk] = GRAY
    return img


def fit(img: np.ndarray, width: int) -> np.ndarray:
    sc = width / img.shape[1]
    return cv2.resize(img, (width, max(1, int(img.shape[0] * sc))),
                      interpolation=cv2.INTER_AREA)


def panel_stats(med, p90, p99, counts):
    H, W = 420, 760
    img = np.full((H, W, 3), 30, np.uint8)
    ymax = 0.12
    x0, y0, x1, y1 = 70, 30, W - 30, H - 70
    cv2.rectangle(img, (x0, y0), (x1, y1), (110, 110, 110), 1)
    for gv in (0.02, 0.05, 0.10):
        gy = int(y1 - (gv / ymax) * (y1 - y0))
        col = (70, 70, 200) if abs(gv - 0.05) < 1e-6 else (60, 60, 60)
        cv2.line(img, (x0, gy), (x1, gy), col, 1)
        cv2.putText(img, f"{gv*100:.0f}cm", (x0 - 55, gy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (180, 180, 180), 1, cv2.LINE_AA)
    xs = np.linspace(x0, x1, len(med))
    ok_pts = [i for i in range(len(med))
              if counts[i] >= 20 and np.isfinite(med[i])]
    for series, col, lab in ((med, (90, 220, 90), "median"), (p90, (60, 180, 250), "p90"),
                             (p99, (60, 60, 240), "p99")):
        for i, j in zip(ok_pts[:-1], ok_pts[1:]):
            ys_i = y1 - (np.clip(series[i], 0, ymax) / ymax) * (y1 - y0)
            ys_j = y1 - (np.clip(series[j], 0, ymax) / ymax) * (y1 - y0)
            cv2.line(img, (int(xs[i]), int(ys_i)), (int(xs[j]), int(ys_j)), col, 2)
        if ok_pts:
            k = ok_pts[len(ok_pts) // 2]
            yk = y1 - (np.clip(series[k], 0, ymax) / ymax) * (y1 - y0)
            cv2.putText(img, lab, (x1 - 96, int(yk) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv2.LINE_AA)
    cv2.putText(img, "height residual (3x3 plane RMS) vs observation range",
                (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    for i in range(len(med)):
        if counts[i] < 20 or not np.isfinite(med[i]):
            continue
        lab = f"{EDGES[i]:.0f}-{EDGES[i+1]:.0f}m" if EDGES[i + 1] < 1e8 else f">{EDGES[i]:.0f}m"
        cv2.putText(img, lab, (int(xs[i]) - 22, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"n={counts[i]}", (int(xs[i]) - 24, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (120, 120, 120), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="*", default=["20261001_044153", "20260929_045615"])
    ap.add_argument("--out", type=Path, default=ROOT / "tmp")
    args = ap.parse_args()

    for name in args.recs:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.exists():
            print(f"[skip] {name}", file=sys.stderr)
            continue
        cfg, cinfo = recorded_config(rec)
        hmap, _lo, _m, rng = cell_hists(rec, cfg)
        peak, mean_h, big = fields(hmap, rng)
        ground = big & (np.abs(peak) < GROUND_PEAK) & np.isfinite(rng)
        rms_mode = plane_rms(peak, ground)
        rms_mean = plane_rms(mean_h, ground)

        med, p90, p99, cnt, hists = [], [], [], [], []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = ground & (rng >= lo) & (rng < hi)
            r = rms_mode[m]
            r = r[np.isfinite(r)]
            cnt.append(int(r.size))
            if r.size < 20:
                med.append(np.nan); p90.append(np.nan); p99.append(np.nan); hists.append(None)
                continue
            med.append(float(np.median(r))); p90.append(float(np.percentile(r, 90)))
            p99.append(float(np.percentile(r, 99)))
            hists.append(r)

        print(f"\n=== {name}  range_m={cinfo['range_m']}  地面附近格 {int(ground.sum())} ===")
        print(f"{'观测距离':>12} {'格数':>7} {'中位':>9} {'p90':>9} {'p99':>9}   众数高度均值偏差")
        for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            if cnt[i] < 20:
                print(f"{lo:5.0f}-{min(hi,999):4.0f} m {cnt[i]:7d}  (样本不足)")
                continue
            m = ground & (rng >= lo) & (rng < hi)
            bias = float(np.median((mean_h - peak)[m]))
            tag = f"{lo:.0f}-{hi:.0f}m" if hi < 1e8 else f">{lo:.0f}m"
            print(f"{tag:>12} {cnt[i]:7d} {med[i]*100:8.2f}cm {p90[i]*100:8.2f}cm "
                  f"{p99[i]*100:8.2f}cm   {bias*100:+.2f}cm")

        # ---- 出图 ----
        args.out.mkdir(parents=True, exist_ok=True)
        top = panel_stats(np.array(med), np.array(p90), np.array(p99), np.array(cnt))
        m_r = np.isfinite(rms_mode) & ground
        c1 = colorize(rng, 0, 5, ground)
        c2 = colorize(np.nan_to_num(rms_mode, nan=0.0), 0, 0.05, m_r)
        for c, title in ((c1, "observation range (m)"), (c2, "3x3 plane RMS (m)")):
            cv2.putText(c, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        big_map = np.hstack([fit(c1, 560), fit(c2, 560)])

        hbins = np.linspace(0, 0.10, 61)
        hp = np.zeros((300, 760, 3), np.uint8)
        hp[:] = 30
        cols = [(90, 220, 90), (60, 180, 250), (60, 60, 240), (200, 120, 255),
                (255, 200, 60), (255, 120, 60), (255, 255, 255)]
        for i, h in enumerate(hists):
            if h is None:
                continue
            cnt_h, _ = np.histogram(h, hbins)
            if cnt_h.sum() == 0:
                continue
            cnt_h = cnt_h / cnt_h.sum() * 100
            col = cols[i % len(cols)]
            prev = None
            for j, v in enumerate(cnt_h):
                x = 60 + int(j / len(cnt_h) * 660)
                y = 250 - int(min(v, 12) / 12 * 200)
                if prev is not None:
                    cv2.line(hp, prev, (x, y), col, 1)
                prev = (x, y)
        cv2.putText(hp, "residual distribution per range band", (60, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        for i in range(len(hists)):
            if hists[i] is None:
                continue
            tag = f"{EDGES[i]:.0f}-{EDGES[i+1]:.0f}m" if EDGES[i + 1] < 1e8 else f">{EDGES[i]:.0f}m"
            cv2.putText(hp, tag, (60 + i * 100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        cols[i % len(cols)], 1, cv2.LINE_AA)

        W = max(top.shape[1], big_map.shape[1], hp.shape[1])
        canvas = np.full((top.shape[0] + big_map.shape[0] + hp.shape[0] + 20, W, 3), 30, np.uint8)
        y = 0
        for im in (top, big_map, hp):
            canvas[y:y + im.shape[0], :im.shape[1]] = im
            y += im.shape[0] + 10
        p = args.out / f"height_accuracy_{name}.png"
        cv2.imwrite(str(p), canvas)
        print(f"图已写出：{p}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
