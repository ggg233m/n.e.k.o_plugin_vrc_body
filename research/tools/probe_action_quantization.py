# -*- coding: utf-8 -*-
"""VRChat 的实际控制量是 WASD 按键 + 鼠标位移，不是 (vx,vz,omega)。

官方 GameScaling 的动作格式就是结构化的：
    X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10
    = 鼠标位移(per-mille) + 6 组按键(每组 33ms)

VRChat 对应关系：
    平移 <- W/A/S/D 按键组合（速度由按键和 Walk/Run 决定，是**离散档位**）
    转向 <- 鼠标 X

本探针检查：机体系速度是否真的量化在少数几个档位上。
如果速度是离散的，那「结构化动作标签」就能直接从现有数据反推，
不需要重新录制。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

s = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
tr = [x for x in s if x["split"] == "train"]
A = np.stack([np.asarray(x["act"]) for x in tr]).astype(np.float64)  # (N,12,3)
M = np.stack([np.asarray(x["mask"]) for x in tr])
m = M > 0
v = A[:, :, :2][m]
w = A[:, :, 2][m]
spd = np.linalg.norm(v, axis=1)

print("[速度模长分布] n=%d" % spd.size)
for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
    print("   p%-3d = %.4f" % (q, np.percentile(spd, q)))

# 非零速度的取值聚集程度：直方图峰值
nz = spd[spd > 0.05]
print()
print("[非零速度] n=%d" % nz.size)
h, edges = np.histogram(nz, bins=60, range=(0, 6))
top = np.argsort(h)[::-1][:8]
print("  最高频的 8 个区间:")
for i in sorted(top):
    if h[i] == 0: continue
    print("    [%.2f, %.2f)  n=%6d  %5.1f%%" % (edges[i], edges[i+1], h[i], 100*h[i]/nz.size))

# 速度向量的分量是否落在少数几个值上（说明是按键组合）
print()
print("[分量离散度] 用 0.01 m/s 量化后，不同取值的个数")
for k, nm in ((0, "vx"), (1, "vz")):
    q = np.round(A[:, :, k][m], 2)
    u, c = np.unique(q, return_counts=True)
    print("  %s: 唯一值 %d 个，前 8 个: %s" % (
        nm, len(u), " ".join("%.2f(%d)" % (u[i], c[i]) for i in np.argsort(c)[::-1][:8])))

# omega 的离散度
qw = np.round(w, 2)
uw, cw = np.unique(qw, return_counts=True)
print("  omega: 唯一值 %d 个，前 8 个: %s" % (
    len(uw), " ".join("%.2f(%d)" % (uw[i], cw[i]) for i in np.argsort(cw)[::-1][:8])))

# 关键：把速度转到机体系后，看是否接近「前进/后退/左右」四个轴
print()
print("[机体系] act 已经是机体系(右,前)。看前进分量/侧向分量的比值");
fwd = A[:, :, 1][m]; lat = A[:, :, 0][m]
mv = spd > 0.2
ratio = np.abs(lat[mv]) / np.maximum(np.abs(fwd[mv]), 1e-6)
print("  |lat|/|fwd| p10=%.3f p50=%.3f p90=%.3f" % tuple(np.percentile(ratio, [10,50,90])))
print("  |lat|/|fwd| < 0.1 (纯前后):  %.1f%%" % (100*np.mean(ratio < 0.1)))
print("  |lat|/|fwd| > 10  (纯横移):  %.1f%%" % (100*np.mean(ratio > 10)))
