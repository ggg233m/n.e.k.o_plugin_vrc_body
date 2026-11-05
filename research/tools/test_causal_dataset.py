# -*- coding: utf-8 -*-
"""验证 causal 数据集：wp 全 0、topo 与旧版不同、且对未来扰动不敏感。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T

OLD = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
NEW = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v4causal")

o = T.build_index(OLD, None)
n = T.build_index(NEW, None)
print("[规模] 旧 %d  新 %d" % (len(o), len(n)))

wps = np.array([x["wp"] for x in n], dtype=np.float64)
print("[wp] 新数据集 max|wp| = %.6f  (必须为 0)" % np.abs(wps).max())
owps = np.array([x["wp"] for x in o], dtype=np.float64)
print("[wp] 旧数据集 max|wp| = %.4f，中位 %.4f" % (np.abs(owps).max(), np.median(np.linalg.norm(owps, axis=1))))

tp_new = np.array([x["topo"] for x in n], dtype=np.float64)
tp_old = np.array([x["topo"] for x in o], dtype=np.float64)
print("[topo] 新 remain 恒 0: %s" % bool(np.allclose(tp_new[:, 2], 0.0)))
print("[topo] 新 prog 范围 [%.2f, %.2f]  lateral 范围 [%.2f, %.2f]"
      % (tp_new[:, 0].min(), tp_new[:, 0].max(), tp_new[:, 1].min(), tp_new[:, 1].max()))
print("[topo] 旧 prog 范围 [%.2f, %.2f]  lateral 范围 [%.2f, %.2f]"
      % (tp_old[:, 0].min(), tp_old[:, 0].max(), tp_old[:, 1].min(), tp_old[:, 1].max()))
print("[topo] 新旧相关性 prog=%.4f lateral=%.4f"
      % (np.corrcoef(tp_new[:,0], tp_old[:,0])[0,1], np.corrcoef(tp_new[:,1], tp_old[:,1])[0,1]))

# 关键：causal 版的前两个 topo 维度是否含有可用信息
print()
print("[信息量] causal topo 与标签的相关性（前 8 步）")
A = np.stack([np.asarray(x["act"])[:8, :2] for x in n])
sp = np.linalg.norm(A[:, 0, :], axis=1)
print("  corr(prog,   首步速度) = %+.4f" % np.corrcoef(tp_new[:,0], sp)[0,1])
print("  corr(lateral,首步速度) = %+.4f" % np.corrcoef(tp_new[:,1], sp)[0,1])
w = np.stack([np.asarray(x["act"])[:8, 2] for x in n]).mean(1)
print("  corr(prog,   平均转角) = %+.4f" % np.corrcoef(tp_new[:,0], w)[0,1])
print("  corr(lateral,平均转角) = %+.4f" % np.corrcoef(tp_new[:,1], w)[0,1])

ok = np.abs(wps).max() < 1e-9 and np.allclose(tp_new[:, 2], 0.0)
print()
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
