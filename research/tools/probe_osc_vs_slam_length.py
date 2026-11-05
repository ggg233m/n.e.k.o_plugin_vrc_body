# -*- coding: utf-8 -*-
"""量 OSC 速度积分出来的路程 与 SLAM 位姿路程 的比值 —— 判定 OSC 速度是什么单位。

为什么关键：训练集 wp 由 build_dr_path 积分 OSC 速度得到，在线 wp 由 PathFollower
的网格弧长给出（追踪米）。若两者单位不同，wp 就有系统性 1/0.755 或 1.32 倍的错配。

做法：用 navmesh_recordings 里既有 osc.jsonl 又有 final_poses.npz 的场次，
  * OSC 路程 = ∫|v|dt
  * SLAM 路程 = Σ|Δpose|
比值 OSC/SLAM 若 ≈1.32 则 OSC 是世界米；若 ≈1.0 则 OSC 已经是追踪米。
"""
from __future__ import annotations

import argparse
import json
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="navmesh_recordings")
    a = ap.parse_args(argv)

    root = Path(a.root)
    for sess in sorted(p for p in root.iterdir() if p.is_dir()):
        osc_p, pose_p = sess / "osc.jsonl", sess / "final_poses.npz"
        if not (osc_p.exists() and pose_p.exists()):
            continue
        try:
            rows = read_jsonl(osc_p)
        except Exception:
            continue
        # osc.jsonl 里速度的字段名各场可能不同，两种都试
        tv, vv = [], []
        for r in rows:
            v = None
            for k in ("vx", "VelocityX", "x"):
                if k in r:
                    v = r
                    break
            if v is None:
                continue
            vx = float(v.get("vx", v.get("VelocityX", 0.0)) or 0.0)
            vz = float(v.get("vz", v.get("VelocityZ", 0.0)) or 0.0)
            tv.append(float(r.get("t", 0.0)))
            vv.append(math_hypot(vx, vz))
        if len(tv) < 10:
            continue
        tv = np.asarray(tv)
        vv = np.asarray(vv)
        dt = np.diff(tv)
        good = (dt > 0) & (dt < 2.5)
        osc_len = float((vv[:-1][good] * dt[good]).sum())

        z = np.load(pose_p)
        key = "T" if "T" in z else list(z.keys())[0]
        T = np.asarray(z[key], dtype=np.float64)
        if T.ndim == 3:
            xy = T[:, :2, 3]
        else:
            xy = T[:, :2]
        slam_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        if slam_len < 1.0:
            continue
        print("  %-22s osc=%.1f m  slam=%.1f m  ratio=%.3f"
              % (sess.name, osc_len, slam_len, osc_len / slam_len), flush=True)
    return 0


def math_hypot(x, y):
    return float((x * x + y * y) ** 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
