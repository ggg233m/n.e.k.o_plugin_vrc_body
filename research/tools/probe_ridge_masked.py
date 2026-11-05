# -*- coding: utf-8 -*-
r"""用训练脚本的 masked_mse 口径复现队友的 45 参数二次 ridge（对表 1.2392）。

上一版复现失败的原因已定位：口径不一致。
  * 训练侧 masked_mse：((pred-act)^2).sum(-1) * mask，再除以 mask.sum()
    → 同一批 val 上 copy 基线 ≈ 1.80
  * 我上一版按整块展平求平方和 → copy 基线 30.26，差 17 倍

所以这次严格照搬 masked_mse，并同时给出：
  A. 原有 split（val 是每 session 的后 20%——注意这**不是**留出 session）
  B. 按 session 留出（真正的泛化口径）
  C. 与前 400 条对齐（队友可能只用 val 前 400）

每个口径都报 copy / zero / ridge 三个数，方便逐位对表。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def quad(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    cr = [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate([X, X ** 2] + cr, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lam", type=float, nargs="+",
                    default=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    ap.add_argument("--n", type=int, default=400, help="与 eval 共用的子集条数")
    ap.add_argument("--moments", action="store_true",
                    help="同时试「只给一阶矩(8维)/二阶矩(16维/含平方)」等更小模型")
    ap.add_argument("--topo-mode", default="full", choices=["full", "no-lateral", "drop"],
                    help="topo 三项的来源是 synthesize_topo()：边 = 沿 DR 轨迹每 5 m 切段，"
                         "lateral = 当前点到该段弦的横向距离——**该弦用到当前点之后最多 5 m 的轨迹**。"
                         "no-lateral 只留 prog/remain，drop 全去。用来测 topo 是否也是泄漏")
    ap.add_argument("--drop-wp", action="store_true",
                    help="把 wp 两维置 0（= 训练侧 --drop-wp）。部署时没有未来轨迹，"
                         "这是唯一诚实的输入口径")
    a = ap.parse_args()

    import train_policy_lora as T
    s = T.build_index(Path(a.dataset), None)
    S = a.steps

    def feats(v):
        o = np.zeros((len(v), 8))
        for i, x in enumerate(v):
            wp = x["wp"]; tp = x.get("topo") or (0, 0, 0); bd = x.get("body") or (0, 0)
            if not a.drop_wp:
                o[i, :2] = wp
            if a.topo_mode == "full":
                o[i, 2:5] = tp
            elif a.topo_mode == "no-lateral":
                o[i, 2] = tp[0]; o[i, 4] = tp[2]      # 只留 prog / remain
            o[i, 5:7] = bd; o[i, 7] = float(x.get("mode", 0))
        return o

    def pack(v):
        """返回 (特征, 标签, mask)，标签形状 (N,S,2)。"""
        X = feats(v)
        A = np.stack([np.asarray(x["act"])[:, :2] for x in v])
        M = np.stack([np.asarray(x["mask"]) for x in v])
        return X, A, M

    def masked_mse(pred, act, mask):
        """**逐位照搬 train_policy_lora.masked_mse**：((p-a)^2).sum(-1) 按 mask 归一。"""
        sq = ((pred - act) ** 2).sum(-1)
        return float((sq * mask).sum() / max(mask.sum(), 1.0))

    def run(trl, val, tag):
        Xtr, Atr, Mtr = pack(trl)
        Xva, Ava, Mva = pack(val)
        # 训练只用「前 S 步都有效」的样本（否则标签是 0 填充，会污染拟合）
        okt = Mtr[:, :S].sum(-1) == S
        Xtr, Atr = Xtr[okt], Atr[okt][:, :S]
        # ⚠️ **评测不做这个过滤**：模型侧（eval_omega_wp）是在全部样本上按 mask 加权，
        # 这里若也过滤掉不满 8 步的样本，两边评的就不是同一批，copy 基线会差 54%
        #（1.80 vs 2.78，2026-11-03 实测）。所以 A/M 保持原样，只截前 S 步。
        Ava, Mv = Ava[:, :S], Mva[:, :S]
        Mt = Mtr[okt][:, :S]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ftr = quad((Xtr - mu) / sd)
        Fva = quad((Xva - mu) / sd)
        Ytr = Atr.reshape(len(Atr), -1)
        best = (1e9, None, None)
        for lam in a.lam:
            W = np.linalg.solve(Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1]), Ftr.T @ Ytr)
            pred = (Fva @ W).reshape(len(Fva), S, 2)
            m = masked_mse(pred, Ava, Mv)
            if m < best[0]:
                best = (m, lam, pred)
        copy = masked_mse(np.repeat(Ava[:, :1], S, axis=1), Ava, Mv)
        zero = masked_mse(np.zeros_like(Ava), Ava, Mv)
        print("  %-24s n_tr=%5d n_va=%5d dim=%2d lam=%-8g "
              "ridge=%.4f  copy=%.4f  zero=%.4f"
              % (tag, len(Ftr), len(Fva), Ftr.shape[1], best[1], best[0], copy, zero))
        return best[0]

    tr = [x for x in s if x["split"] == "train"]
    # 与 eval_omega_wp 共用选样口径：两者必须评**同一批 400 条**
    va_all = [x for x in s if x["split"] == "val"]
    print("[data] train=%d val(全部)=%d" % (len(tr), len(va_all)), flush=True)
    print()
    run(tr, va_all, "原有 split (全部 val)")
    va400 = T.select_val_subset(Path(a.dataset), a.n)
    run(tr, va400, "与 eval 同批 %d 条" % len(va400))

    # ---- D. 按 session/world 整组留出（真正的泛化口径）----
    # 旧代码这里引用了未定义的 va，实际一次都没跑过（NameError）。修掉。
    # 10 场录制 = 10 个不同世界，正好可以做 leave-one-world-out。
    sids = sorted({x["sid"] for x in s})
    if len(sids) >= 2:
        print()
        print("按 session/world 留出（每折：训练=其余 9 场，验证=留出那场全部帧）")
        scores = []
        for hold in sids:
            trh = [x for x in tr if x["sid"] != hold]
            vah = [x for x in s if x["sid"] == hold]
            if not trh or not vah:
                continue
            scores.append((hold, run(trh, vah, "  LWO %s" % hold)))
        if scores:
            vals = np.array([v for _, v in scores])
            print()
            print("  留出汇总  mean=%.4f  median=%.4f  min=%.4f  max=%.4f"
                  % (vals.mean(), np.median(vals), vals.min(), vals.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
