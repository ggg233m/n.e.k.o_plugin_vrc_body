# -*- coding: utf-8 -*-
r"""量化「wp 等于标签积分」这一假设。

near 模式: target_s = min(s0+0.8, s_end)         -> 被 0.8 m 截断
far  模式: target_s = min(s0+4.0, s_end)         -> s_end 恒较小(除速度>3.33m/s)
          s_end = at(t_obs + K_STEPS*STEP_S)     -> 正好是「动作块 1.2 s 之后的位置」

若 far 的 wp ≈ sum_k(steps) dt * act_k，则 wp **就是标签本身**（仅差逐步机体系
旋转），泄漏不是「部分」而是「完全」。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_policy_lora as T

def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"
    S = 8
    s = T.build_index(Path(ds), None)
    s = [x for x in s if x["split"] == "train"]
    wp = np.array([x["wp"] for x in s], dtype=np.float64)            # (N,2) right,fwd
    A  = np.stack([np.asarray(x["act"])[:S, :2] for x in s]).astype(np.float64)  # (N,S,2)
    M  = np.stack([np.asarray(x["mask"])[:S] for x in s]).astype(np.float64)
    ok = M.sum(-1) == S
    wp, A = wp[ok], A[ok]
    print("[data] %s  可用样本 %d" % (Path(ds).name, len(wp)))

    for dt in (0.1,):
        integ = A.sum(1) * dt
        for k in range(2):
            a, b = wp[:, k], integ[:, k]
            c = np.corrcoef(a, b)[0, 1]
            rel = np.abs(a - b) / (np.abs(b) + 1e-6)
            print("  分量%d corr(wp, sum(act)*dt)=%.4f  |wp-integ| p50=%.4f m  rel p50=%.1f%%"
                  % (k, c, np.median(np.abs(a - b)), 100*np.median(rel)))
        d_wp = np.linalg.norm(wp, axis=1); d_in = np.linalg.norm(integ, axis=1)
        err = np.linalg.norm(wp - integ, axis=1)
        print("  |wp| p50=%.3f  |integ| p50=%.3f  |wp-integ| p50=%.3f  p90=%.3f"
              % (np.median(d_wp), np.median(d_in), np.median(err), np.percentile(err, 90)))
        # 方向一致性
        cosang = (wp * integ).sum(1) / (d_wp * d_in + 1e-9)
        print("  cos(wp, integ) p50=%.4f  p10=%.4f  | 占比 cos>0.9: %.1f%%"
              % (np.median(cosang), np.percentile(cosang, 10), 100*np.mean(cosang > 0.9)))
    # 线性可解性：用 wp 单变量最小二乘还原 integ（仅作下界参考）
    X = np.concatenate([wp, np.ones((len(wp),1))], 1)
    W, *_ = np.linalg.lstsq(X, integ, rcond=None)
    pred = X @ W
    print("  仅用 wp 线性回归预测 sum(act): 残差 p50=%.4f m (占 |integ| p50 的 %.1f%%)"
          % (np.median(np.linalg.norm(pred-integ,axis=1)),
             100*np.median(np.linalg.norm(pred-integ,axis=1))/np.median(d_in)))
    return 0

raise SystemExit(main())
