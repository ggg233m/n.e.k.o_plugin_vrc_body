# -*- coding: utf-8 -*-
"""现有数据能否支撑「视觉避障」任务？逐项体检。"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

s = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
tr = [x for x in s if x["split"] == "train"]
A = np.stack([np.asarray(x["act"]) for x in tr])
M = np.stack([np.asarray(x["mask"]) for x in tr])
m = M > 0

print("=" * 72)
print("  视觉避障所需行为在现有 10 场录制里的覆盖")
print("=" * 72)

v = A[:, :, :2][m]
lat = np.abs(v[:, 0])
fwd = np.abs(v[:, 1])
mv = np.linalg.norm(v, axis=1) > 0.2
fwdc = A[:, :, 1][m]

lateral_ratio = float(np.mean(mv & (lat > fwd)))
backward = float(np.mean(fwdc < -0.2))
turning = float(np.mean(np.abs(A[:, :, 2][m]) > 0.3))

checks = [
    ("左右绕行(横移为主)", lateral_ratio, 0.30, "运动样本里应 >=30% 横移主导"),
    ("后退", backward, 0.15, "绕障后重定位"),
    ("原地转身", turning, 0.10, "死路掉头"),
]
for name, got, want, why in checks:
    mark = "OK" if got >= want else "缺"
    print("  [%s] %-20s 实测 %5.1f%%  要求 >=%4.1f%%   %s"
          % (mark, name, 100 * got, 100 * want, why))

by_sid = {}
for i, x in enumerate(tr):
    by_sid.setdefault(x["sid"], []).append(i)
turns = 0;
tot = 0
for sid, idx in by_sid.items():
    idx = sorted(idx, key=lambda i: tr[i]["t"])
    w = np.array([A[i, 0, 2] for i in idx])
    if w.size < 10:
        continue
    w = w[np.abs(w) > 0.0]
    if w.size < 10:
        continue
    sign = np.sign(w)
    turns += int((np.abs(np.diff(sign)) > 0).sum())
    tot += w.size
print()
print("  omega 符号翻转: %d 次 / %d 步 = 每 %.1f 步一次"
      % (turns, tot, tot / max(turns, 1)))

meta = Path("G:/vrc_recording")
have = [p for p in meta.iterdir() if p.is_dir() and (p / "meta.json").exists()]
keys = set()
if have:
    keys = set(json.loads((have[0] / "meta.json").read_text(encoding="utf-8")).keys())
print()
print("  录制 meta 里有用的几何/碰撞字段:")
for k in ("collision", "obstacle", "depth", "raycast", "navmesh", "world_key"):
    print("    %-12s %s" % (k, "有" if k in keys else "无"))
print()
print("  全部字段: %s" % ", ".join(sorted(keys)))
