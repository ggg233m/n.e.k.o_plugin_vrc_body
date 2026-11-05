# -*- coding: utf-8 -*-
"""VRChat 自然运动方式核查：低横移是不是「转身后前进」的必然结果？

假设（用户提出）：正常玩家在 VRChat 里不用 A/D 横移，而是先转头/转身，
再按 W 前进。如果成立，那么「横移 0.5%」不代表「没有横向机动能力」，
而是横向机动**由转向+前进实现**。

检验方法：
  H1  若横向位移来自转向，则轨迹曲率与 ω 应高度相关
  H2  若人靠转身走，则「转向段」与「前进段」应交替（分段性）
  H3  机体系横移小，但**世界系**轨迹应有明显方向变化
"""
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
vx = A[:, :, 0][m]
vz = A[:, :, 1][m]
w = A[:, :, 2][m]

print("=" * 74)
print("  H1: 横向位移是否由转向实现")
print("=" * 74)
sp = np.hypot(vx, vz)
mv = sp > 0.2

# 前进为主时，ω 与「前进中带横移」的关系
# 若人用转身走，则运动时 |vx|/|vz| 小，但 |ω| 可以很大
print("  运动中样本 (|v|>0.2, n=%d):" % int(mv.sum()))
print("    |横移|/|前进| 中位 %.4f" % np.median(np.abs(vx[mv]) / np.maximum(np.abs(vz[mv]), 1e-6)))
print("    |ω| 中位 %.3f  p90 %.3f rad/s" % (np.median(np.abs(w[mv])), np.percentile(np.abs(w[mv]), 90)))
print("    |ω|>0.3 占比 %.1f%%" % (100 * np.mean(np.abs(w[mv]) > 0.3)))

# 关键：横向「位移」= 前进 × sin(航向偏角)。若人靠转向实现横向，
# 则把 ω 积分出来的等效横向位移，应远大于实测的机身横移
dt = 0.1
equiv_lat = np.abs(vz[mv] * np.tan(np.clip(w[mv] * dt, -1.2, 1.2)))
actual_lat = np.abs(vx[mv])
print("  运动中：等效横向位移(由ω推) 中位 %.3f m/s" % np.median(equiv_lat))
print("          实测横移分量      中位 %.3f m/s" % np.median(actual_lat))
print("          -> 比值 %.2f 倍（>1 说明该横移靠转向实现）"
      % (np.median(equiv_lat) / max(np.median(actual_lat), 1e-6)))

print()
print("=" * 74)
print("  H2: 转向与前进是否交替（分段性）")
print("=" * 74)
by = {}
for i, x in enumerate(tr):
    by.setdefault(x["sid"], []).append(i)
seg_turn_only = 0
seg_move_only = 0
seg_both = 0
seg_none = 0
for sid, idx in by.items():
    idx = sorted(idx, key=lambda i: tr[i]["t"])
    for i in idx:
        a = A[i]; mm = M[i] > 0
        if not mm.any():
            continue
        sp_i = np.hypot(a[mm, 0], a[mm, 1]).max()
        w_i = np.abs(a[mm, 2]).max()
        moving = sp_i > 0.2
        turning = w_i > 0.3
        if moving and turning:
            seg_both += 1
        elif moving:
            seg_move_only += 1
        elif turning:
            seg_turn_only += 1
        else:
            seg_none += 1
tot = seg_both + seg_move_only + seg_turn_only + seg_none
print("  块级分类 (n=%d):" % tot)
print("    只走不转 %5.1f%%" % (100*seg_move_only/tot))
print("    只转不走 %5.1f%%   <- 「转身后前进」的转身段" % (100*seg_turn_only/tot))
print("    边走边转 %5.1f%%" % (100*seg_both/tot))
print("    停       %5.1f%%" % (100*seg_none/tot))

print()
print("=" * 74)
print("  H3: 世界系轨迹的方向变化（机体系横移小 ≠ 轨迹直）")
print("=" * 74)
import json
REC = Path("G:/vrc_recording")
for sess in sorted(p for p in REC.iterdir() if p.is_dir() and (p / "osc.jsonl").exists())[:5]:
    rows = [json.loads(l) for l in (sess / "osc.jsonl").open("r", encoding="utf-8")]
    ang = [r.get("AngularY") for r in rows if r.get("AngularY") is not None]
    if not ang:
        continue
    a = np.array(ang, dtype=np.float64)
    print("  %s  AngularY 非零占比 %4.1f%%  中位|值| %.3f"
          % (sess.name, 100 * np.mean(np.abs(a) > 1e-9), np.median(np.abs(a))))
