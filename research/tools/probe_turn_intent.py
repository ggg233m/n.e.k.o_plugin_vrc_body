# -*- coding: utf-8 -*-
"""转向意图 vs omega 序列：哪种标签更可学？

回应评审（2026-11-05）：omega 作为「未来 1.2 s 人类角速度」的单值回归目标不合适，
不代表 VLA 不该控制转向。更合适的标签是「未来 0.3~0.6 s 的转向意图」。

本探针把同一批数据派生成两种标签，在**完全相同的 LWO 10 折**下对比：
  (R) 回归：未来 8 步的 omega 单值序列        -> 报 MSE / 可解释比例
  (C) 分类：未来 0.3~0.6 s 的转向意图 3 类    -> 报 accuracy / macro-F1

意图定义：窗口内平均角速度 (rad/s)
    >  +0.3  左转?（omega 右正，所以正=右）
    <  -0.3  左转
    其余     直行/停
具体符号按 ANGULARY 约定（右转为正）已经反映在 act[:,2] 里。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import train_policy_lora as T
import policy_classify as C

S = 8
ROOT = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
WIN = (3, 6)          # 0.3 s ~ 0.6 s（步 3..5）
THR = 0.3             # rad/s ~ 17 deg/s


def quad(X):
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def feats(v, drop_wp=True):
    o = np.zeros((len(v), 8))
    for i, x in enumerate(v):
        if not drop_wp:
            o[i, :2] = x["wp"]
        o[i, 2:5] = x.get("topo") or (0, 0, 0)
        o[i, 5:7] = x.get("body") or (0, 0)
        o[i, 7] = float(x.get("mode", 0))
    return o


def intent_of(act_block, mask_block):
    """返回 0/1/2 = 左/直/右，以及窗口平均 omega。"""
    a = np.asarray(act_block)
    m = np.asarray(mask_block)
    w = a[WIN[0]:WIN[1], 2]
    ok = m[WIN[0]:WIN[1]] > 0
    if not ok.any():
        return None, 0.0
    mean_w = float(w[ok].mean())
    if mean_w > THR:
        return 2, mean_w      # 右
    if mean_w < -THR:
        return 0, mean_w      # 左
    return 1, mean_w          # 直行


alls = T.build_index(ROOT, None)
C.relabel(alls)
sids = sorted({x["sid"] for x in alls})

lab, vals = [], []
for x in alls:
    k, v = intent_of(x["act"], x["mask"])
    lab.append(-1 if k is None else k)
    vals.append(v)
lab = np.array(lab); vals = np.array(vals)
keep = lab >= 0
print("[data] %d 样本，有效意图 %d" % (len(alls), keep.sum()))
for i, nm in enumerate(("左转", "直行/停", "右转")):
    print("   %-6s n=%6d  %5.1f%%" % (nm, int((lab == i).sum()), 100*np.mean(lab == i)))

# 标签噪声：窗口内 omega 符号是否稳定
stable = 0; tot = 0
for x in alls:
    a = np.asarray(x["act"]); m = np.asarray(x["mask"])
    w = a[WIN[0]:WIN[1], 2][m[WIN[0]:WIN[1]] > 0]
    if w.size < 2:
        continue
    tot += 1
    if (w > 0).all() or (w <= 0).all():
        stable += 1
print()
print("[标签稳定性] 窗口内 omega 符号一致: %d/%d = %.1f%%" % (stable, tot, 100*stable/max(tot,1)))
print("   不一致说明：0.3~0.6 s 内正在反向，单值回归会把两侧抵消成 0")


def lwo_eval(mode):
    """mode: reg / cls。返回 pooled 指标 + 逐折。"""
    fold_mse, fold_acc, fold_f1, base_acc = [], [], [], []
    allpred, alltrue = [], []
    for hold in sids:
        tri = [i for i, x in enumerate(alls) if x["sid"] != hold]
        vai = [i for i, x in enumerate(alls) if x["sid"] == hold]
        tri = [i for i in tri if keep[i]]
        vai = [i for i in vai if keep[i]]
        if len(tri) < 100 or len(vai) < 20:
            continue
        Xtr = feats([alls[i] for i in tri]); Xva = feats([alls[i] for i in vai])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr, Fva = quad((Xtr-mu)/sd), quad((Xva-mu)/sd)
        if mode == "reg":
            Y = np.stack([np.asarray(alls[i]["act"])[:S, 2:3] for i in tri])
            A = np.stack([np.asarray(alls[i]["act"])[:S, 2:3] for i in vai])
            M = np.stack([np.asarray(alls[i]["mask"])[:S] for i in vai])
            Ym = Y.reshape(len(tri), -1)
            best = (1e18, None)
            for lam in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0):
                W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Ym)
                p = (Fva@W).reshape(len(vai), S, 1)
                e = (((p-A)**2).sum(-1)*M).sum()/max(M.sum(), 1.0)
                if e < best[0]: best = (e, p)
            fold_mse.append(best[0])
        else:
            ytr = lab[tri]; yva = lab[vai]
            # one-hot 最小二乘（等价 ridge 分类头）
            Yh = np.eye(3)[ytr]
            best = (-1, None)
            for lam in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
                W = np.linalg.solve(Ftr.T@Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T@Yh)
                s = Fva@W
                pr = s.argmax(1)
                a = float((pr == yva).mean())
                if a > best[0]: best = (a, pr)
            pr = best[1]
            fold_acc.append(best[0])
            # macro F1
            f1s = []
            for c in range(3):
                tp = ((pr == c) & (yva == c)).sum()
                fp = ((pr == c) & (yva != c)).sum()
                fn = ((pr != c) & (yva == c)).sum()
                if tp == 0:
                    continue
                prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
                f1s.append(2*prec*rec/max(prec+rec, 1e-9))
            fold_f1.append(float(np.mean(f1s)) if f1s else 0.0)
            base_acc.append(float(np.bincount(ytr, minlength=3).max()/len(ytr)))
            allpred.append(pr); alltrue.append(yva)
    if mode == "reg":
        return np.array(fold_mse)
    return np.array(fold_acc), np.array(fold_f1), np.array(base_acc), 


print()
print("=" * 78)
print("  (R) 回归目标：未来 8 步 omega 序列（当前做法）")
print("=" * 78)
mse = lwo_eval("reg")
print("  逐折 MSE: " + " ".join("%.3f" % v for v in np.sort(mse)))
print("  macro=%.4f  median=%.4f  std=%.4f" % (mse.mean(), np.median(mse), mse.std()))

print()
print("=" * 78)
print("  (C) 分类目标：未来 0.3~0.6 s 转向意图（左/直/右）")
print("=" * 78)
acc, f1, base = lwo_eval("cls")
print("  逐折 accuracy: " + " ".join("%.3f" % v for v in np.sort(acc)))
print("  macro acc=%.4f  median=%.4f  std=%.4f" % (acc.mean(), np.median(acc), acc.std()))
print("  macro F1 =%.4f  median=%.4f" % (f1.mean(), np.median(f1)))
print("  多数类基线 acc=%.4f  ->  提升 %+.1f%%" % (base.mean(), 100*(acc.mean()/base.mean()-1)))
