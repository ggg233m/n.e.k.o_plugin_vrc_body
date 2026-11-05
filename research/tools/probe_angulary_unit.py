# -*- coding: utf-8 -*-
"""AngularY 的单位到底是 deg/s 还是别的 —— 用「积分角度对齐 HMD 转角」来判定。

VRChat 官方文档只写 `AngularY — Angular velocity on the Y axis`，**没给单位**。
之前假设 deg/s 并据此把大值当异常裁掉，8.8% 的正常数据被误伤。

判别方法：把 AngularY 积分成角度，与同期 HMD 偏航角的变化量比。
  若 int(w*K) dt ≈ Δyaw，则 K 就是正确换算系数；K 若≈1 说明单位就是 deg/s。
同时对每个候选 K 求残差，选使 |int - Δyaw| 最小的那个。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_policy_dataset as B


def main() -> int:
    root = Path("G:/vrc_recording")
    rows = []
    for s in sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] == "20")[:6]:
        if not (s / "avatar_params.jsonl").exists() or not (s / "hmd_frames.jsonl").exists():
            continue
        st = B.load_streams(s)
        st["fwd"] = B.hmd_forward(st["quat"])
        if len(st["ay_t"]) < 500:
            continue
        fx, fz = st["fwd"][:, 0], st["fwd"][:, 1]
        yaw = np.unwrap(np.arctan2(fx, fz))          # 弧度，+x 方向（右）为正
        th = st["h_t"]

        # 在整段录制上按 1 s 窗口做积分对比（避开单点量化噪声）
        T0, T1 = max(th[0], st["ay_t"][0]) + 2.0, min(th[-1], st["ay_t"][-1]) - 2.0
        seg = []
        t = T0
        while t + 1.0 < T1:
            m = (th >= t) & (th <= t + 1.0)
            ma = (st["ay_t"] >= t) & (st["ay_t"] <= t + 1.0)
            if m.sum() >= 10 and ma.sum() >= 2:
                dyaw = float(yaw[m][-1] - yaw[m][0])
                # AngularY 矩形积分（按事件间隔，末值保持）
                ta = st["ay_t"][ma]; va = st["ay_v"][ma]
                if len(ta) >= 2:
                    dt = np.diff(ta)
                    integ = float(np.sum(va[:-1] * dt))
                    seg.append((dyaw, integ))
            t += 1.0
        if len(seg) < 20:
            continue
        a = np.array([x[0] for x in seg])
        b = np.array([x[1] for x in seg])
        # dyaw(rad) = K * integ(-AngularY * dt) / 1000 ; 求 K
        mask = np.abs(b) > 1e-6
        if mask.sum() < 10:
            continue
        K = float((a[mask] @ b[mask]) / (b[mask] @ b[mask]))
        pred = K * b
        r2 = 1.0 - float(((a - pred) ** 2).sum()) / max(float(((a - a.mean()) ** 2).sum()), 1e-12)
        rows.append((s.name, len(seg), K, r2, float(np.std(a)), float(np.std(b))))
        print("  %s  n=%3d  K=%+.6f rad/unit  R2=%+.3f" % (s.name, len(seg), K, r2))

    if rows:
        Ks = [r[2] for r in rows]
        print()
        print("  平均 K = %.6f rad/unit  (1/K = %.2f unit/rad = %.4f unit/deg)"
              % (np.mean(Ks), 1 / np.mean(Ks), (1 / np.mean(Ks)) * math.pi / 180))
        print("  若单位就是 deg/s，则 K 应等于 %.6f (pi/180)" % (math.pi / 180))
        print("  ⇒ 单位更可能是：%s"
              % ("deg/s（K≈pi/180）" if abs(np.mean(Ks) - math.pi / 180) < math.pi / 180 * 0.5
                 else "不是 deg/s，K=%.6f rad/unit" % np.mean(Ks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
