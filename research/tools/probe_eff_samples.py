# -*- coding: utf-8 -*-
"""量化两条评审意见：
  §4 有效数据量 —— 动作块 1.2 s、网格 0.2 s，相邻样本重叠 83%；
  §5 动作分布 —— 横移/后退/慢速是否真的稀缺。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

ds = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
S = 8
s = T.build_index(ds, None)
tr = [x for x in s if x["split"] == "train"]
print("[split] train=%d val=%d total=%d" % (len(tr), len(s)-len(tr), len(s)))

# ---------- §4 有效样本量 ----------
GRID_S, CHUNK_S = 0.2, 1.2
overlap = CHUNK_S / GRID_S                      # 每个独立窗口被 6 个样本覆盖
print()
print("[§4] 动作块 %.1f s / 网格 %.1f s = 每个独立窗口 %.0f 个样本" % (CHUNK_S, GRID_S, overlap))
print("     名义 train=%d  →  每轮独立窗口 ≈ %d" % (len(tr), int(len(tr)/overlap)))

# 标签自相关：同一 session 内 lag-k 样本的 act[0] 相关
by_sid = {}
for x in tr:
    by_sid.setdefault(x["sid"], []).append(x)
lags, cors = [], []
for sid, xs in by_sid.items():
    xs = sorted(xs, key=lambda z: z["t"])
    if len(xs) < 50: continue
    v = np.array([np.asarray(z["act"])[0][:2] for z in xs])
    v = v - v.mean(0, keepdims=True)
    base = (v*v).sum()
    for k in (1, 2, 5, 10, 25):
        c = (v[:-k]*v[k:]).sum()/base
        cors.append((k, c))
for k in (1, 2, 5, 10, 25):
    vals = [c for kk, c in cors if kk == k]
    print("     act[0] 自相关 lag=%2d (=%.1f s): %.4f" % (k, k*GRID_S, np.mean(vals)))

# ---------- §5 动作分布 ----------
A = np.stack([np.asarray(x["act"])[:, :2] for x in tr]).astype(np.float64)   # (N,12,2)
a0 = A[:, 0, :]
spd = np.linalg.norm(a0, axis=1)
lat, fwd = a0[:, 0], a0[:, 1]   # (右, 前)
print()
print("[§5] 首步速度 |v|: p10=%.2f p50=%.2f p90=%.2f m/s" % tuple(np.percentile(spd,[10,50,90])))
print("     静止(<0.1)=%.1f%%  慢速(0.1~0.5)=%.1f%%  中(0.5~1.5)=%.1f%%  快(>1.5)=%.1f%%"
      % (100*np.mean(spd<0.1), 100*np.mean((spd>=0.1)&(spd<0.5)),
         100*np.mean((spd>=0.5)&(spd<1.5)), 100*np.mean(spd>=1.5)))
mv = spd > 0.2
print("     运动中样本的 右向/前向:  |lat|>|fwd| 占比 %.1f%%" % (100*np.mean(np.abs(lat[mv]) > np.abs(fwd[mv]))))
print("     前进(fwd>0.2)=%.1f%%  后退(fwd<-0.2)=%.1f%%  横移为主=%.1f%%"
      % (100*np.mean(fwd>0.2), 100*np.mean(fwd<-0.2), 100*np.mean(np.abs(lat)>np.abs(fwd))))
print("     转向 |w|: p50=%.3f p90=%.3f rad/s"
      % tuple(np.percentile(np.abs(np.stack([np.asarray(x["act"])[:,2] for x in tr])), [50,90])))
