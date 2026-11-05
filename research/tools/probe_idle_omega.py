# -*- coding: utf-8 -*-
"""idle 类的 ω 分布：idle 是「没走动」，那「没走动但在转头」占多少？

如果 idle 里大量样本 ω 明显非 0，那么：
  * 「黑图 0.061 vs 真图 0.792」就不能归因为「模型看到东西想动」，
    而是「模型该输出转速时却输出了 0」——方向正好相反。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

s = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
tr = [x for x in s if x["split"] == "train"]

for cl in ("idle", "move", "trans"):
    xs = [x for x in tr if x["cls"] == cl]
    if not xs: continue
    A = np.stack([np.asarray(x["act"]) for x in xs])     # (N,12,3)
    M = np.stack([np.asarray(x["mask"]) for x in xs])
    spd = np.linalg.norm(A[:, :, :2], axis=-1)           # (N,12)
    omg = np.abs(A[:, :, 2])
    m = M > 0
    turn_rate = (omg[m] > 0.3).mean()                    # ω>0.3 rad/s ≈ 17 deg/s
    still_turn = ((spd[m] <= 0.1) & (omg[m] > 0.3)).mean()
    print("[%s] n=%d 平移|v| p50=%.3f p90=%.3f | |ω| p50=%.3f p90=%.3f"
          % (cl, len(xs), np.median(spd[m]), np.percentile(spd[m],90),
             np.median(omg[m]), np.percentile(omg[m],90)))
    print("       ω>0.3 占比 %.1f%%   平移静止但 ω>0.3 占比 %.1f%%"
          % (100*turn_rate, 100*still_turn))
    # 每块的「最大 ω」——判断这一块里有没有转头
    per_block_max = omg.max(1)
    print("       块内 max|ω|>0.3 的样本占比 %.1f%%"
          % (100*(per_block_max > 0.3).mean()))

# 专门看 idle 类：copy 基线为何是 0
print()
idle = [x for x in tr if x["cls"] == "idle"]
A = np.stack([np.asarray(x["act"]) for x in idle])
print("[idle] 首步速度 max = %.4f （copy 把它抄 8 步，所以 copy≈0）"
      % np.linalg.norm(A[:,0,:2],axis=-1).max())
print("[idle] 首步 |ω| p50=%.3f p90=%.3f"
      % (np.median(np.abs(A[:,0,2])), np.percentile(np.abs(A[:,0,2]),90)))
print("[idle] 全块 |ω| 均值 %.3f，其中 >0.3 的步占 %.1f%%"
      % (np.abs(A[:,:,2]).mean(), 100*(np.abs(A[:,:,2])>0.3).mean()))
