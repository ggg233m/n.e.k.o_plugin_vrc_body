# -*- coding: utf-8 -*-
"""ω 标签来源对比：HMD 四元数 vs AngularY，谁更该当标签？

问题（2026-11-02）：用户问「ω 靠 HMD 四元数换算是不是更准确」。

关键区分：**精度 ≠ 构念效度**。
  * HMD 四元数 175-195 Hz、连续无量化 —— 精度高，但它测的是**头的朝向**，
    六点追踪下人可以转头而不转身体；
  * AngularY 是 VRChat 自己报的**角色角速度参数**，量化步长 8、±1024 削顶 1.055% ——
    精度差，但测的是**身体/胶囊体**，与 VelocityX/Z 同一族信号。

本脚本用两个可判定的量来分辨：

1. **拟合关系**：HMD 偏航角速度 vs AngularY，求 corr 与斜率 K（复核 1.243-1.697 与 2.44-3.93 的分歧）。
2. **高频能量比**：转头是高频抖动、身体转向是低频。分别算「0.1 s 窗口速率」与
   「2 s 窗口速率」的方差比 var(短)/var(长)。**若 HMD 的比值显著高于 AngularY，
   说明它多出来的那部分是头动噪声，而不是身体转向。**

另外做两件保护：
  * HMD 恒等四元数（丢追踪）段标记并跳过；单帧跳变 > 45 度视为跳变，剔除；
  * 只在 HMD 与 AngularY 都有数据且时间对齐的区间上比。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_omega_source.py \
        --root G:/vrc_recording --out .venv_xpu/runs/omega_source
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_jsonl(p: Path):
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def quat_yaw(q):
    """(N,4) xyzw -> 偏航角（弧度）。用 Z-up 约定：atan2(2(wz+xy), 1-2(y^2+z^2))。

    ⚠️ 早先绕错轴是 |r|<=0.07 那个伪结论的来源（Y-up vs Z-up）。
    这里同时算两种，把「哪种才对」交给数据说话。
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Z-up（绕世界 Z 轴旋转）
    yawz = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # Y-up（绕世界 Y 轴旋转）
    yawy = np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return yawz, yawy


def unwrap_safe(yaw, cos_ok):
    """展开相位；把丢追踪 / 跳变处置为 NaN，后续用有效段算速率。"""
    y = np.unwrap(yaw)
    d = np.diff(y, prepend=y[0])
    bad = (~cos_ok) | (np.abs(d) > math.radians(45))
    y = y.copy()
    y[bad] = np.nan
    return y


def rate_at(t, yaw, tq, win):
    """在 tq 处用 [tq-win, tq] 的端点差分求速率（rad/s），跳过 NaN。"""
    out = np.full(len(tq), np.nan)
    for i, q in enumerate(tq):
        m = (t >= q - win) & (t <= q) & ~np.isnan(yaw)
        if m.sum() < 5:
            continue
        tt, yy = t[m], yaw[m]
        if tt[-1] - tt[0] < win * 0.5:
            continue
        out[i] = (yy[-1] - yy[0]) / max(tt[-1] - tt[0], 1e-6)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="G:/vrc_recording")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sessions", type=int, default=10)
    a = ap.parse_args(argv)

    root = Path(a.root)
    sess = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] == "20")
    rows = []
    for s in sess[: a.max_sessions]:
        fh, fa = s / "hmd_frames.jsonl", s / "avatar_params.jsonl"
        if not (fh.exists() and fa.exists()):
            continue
        try:
            hmd = read_jsonl(fh)
            par = read_jsonl(fa)
        except Exception:
            continue
        th = np.array([r["t"] for r in hmd if "hmd" in r], float)
        q = np.array([r["hmd"]["rotation_xyzw"] for r in hmd if "hmd" in r], float)
        if len(th) < 1000:
            continue
        n = np.linalg.norm(q, axis=1)
        cos_ok = np.abs(q[:, 3] / np.maximum(n, 1e-9)) < 0.99995   # 非恒等（有姿态）
        yz, yy = quat_yaw(q / np.maximum(n, 1e-9)[:, None])

        ta = np.array([r["t"] for r in par if r.get("name") == "AngularY"], float)
        va = np.array([r["value"] for r in par if r.get("name") == "AngularY"], float)
        if len(ta) < 500:
            continue

        # 0.4 s 网格（与动作块 1.2 s 同量级，且够密）
        tq = np.arange(max(th[0], ta[0]) + 1.0, min(th[-1], ta[-1]) - 1.0, 0.4)
        if len(tq) < 200:
            continue

        res = {"sid": s.name, "n_q": int(len(tq))}
        for tag, yaw in (("zup", yz), ("yup", yy)):
            yu = unwrap_safe(yaw, cos_ok)
            r_short = rate_at(th, yu, tq, 0.2)
            r_long = rate_at(th, yu, tq, 2.0)
            good = ~np.isnan(r_short) & ~np.isnan(r_long)
            if good.sum() < 100:
                continue
            v_a = np.interp(tq[good], ta, va)
            c = float(np.corrcoef(r_short[good], v_a)[0, 1]) if v_a.std() > 0 else 0.0
            # 拟合 K: va = -K * rate
            X = np.stack([r_short[good]], 1)
            K = float(-(X[:, 0] @ v_a) / max(X[:, 0] @ X[:, 0], 1e-12))
            ratio = float(np.var(r_short[good]) / max(np.var(r_long[good]), 1e-12))
            res[tag] = dict(corr=c, K=K, hi_ratio=ratio)

        # AngularY 自己的短/长窗能量比（在它自己的采样率上近似：用事件间隔重采样）
        taq = tq[~np.isnan(tq)]
        va_s = np.interp(tq, ta, va)
        res["angularY_hint"] = dict(
            n=int(len(va_s)), std=float(np.std(va_s)),
            frac_zero=float((va_s == 0).mean()))
        rows.append(res)
        print("  %s  n=%d  zup(corr=%.3f K=%.2f hi=%.2f)  yup(corr=%.3f K=%.2f hi=%.2f)"
              % (s.name, res["n_q"],
                 res.get("zup", {}).get("corr", float("nan")),
                 res.get("zup", {}).get("K", float("nan")),
                 res.get("zup", {}).get("hi_ratio", float("nan")),
                 res.get("yup", {}).get("corr", float("nan")),
                 res.get("yup", {}).get("K", float("nan")),
                 res.get("yup", {}).get("hi_ratio", float("nan"))), flush=True)

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "omega_source.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    for tag in ("zup", "yup"):
        cs = [r[tag]["corr"] for r in rows if tag in r]
        ks = [r[tag]["K"] for r in rows if tag in r]
        hs = [r[tag]["hi_ratio"] for r in rows if tag in r]
        if cs:
            print()
            print("  %s: corr %.3f..%.3f (mean %.3f)  K %.2f..%.2f (mean %.2f)  高频能量比 %.2f..%.2f"
                  % (tag, min(cs), max(cs), float(np.mean(cs)),
                     min(ks), max(ks), float(np.mean(ks)), min(hs), max(hs)))
    print("[done] %s" % (o / "omega_source.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
