# -*- coding: utf-8 -*-
r"""实验：**DR→地图 的非刚体差，能不能被"按里程分段线性形变"消掉？**

背景（`Docs/漂移形态诊断（2026-10-05）.md` §9.2）
------------------------------------------------
回环**彼此**很自洽（045615 in-sample 中位 6.4 cm），但**回环整体与 DR 之间**差一个
非刚体变换（尺度 −9.9% / 旋转 −6.6° / 剪切 19.7%）。`pose_graph4` 是 4-DOF，
结构上修不了尺度与剪切 ⇒ 有人提议用"分段线性形变"补。

本工具要判的是：**这个形变场是光滑可外推的结构，还是纯过拟合？**

模型
----
``p_map ≈ A(s) @ p_dr + b(s)``，s = 里程，2D 平面（**yaw 已实测恒等**，见探查输出）。

| 变体 | 说明 |
|---|---|
| ``identity`` | 直接用 DR（下界基线） |
| ``global_affine`` | 常数 A、b —— 文档已测的那条 |
| ``seg_const`` | 按里程分 K 段，每段一个仿射（**段界跳变**） |
| ``seg_linear`` | 节点仿射 + 段内**线性插值**（连续，A/b 逐元素插值） |

验收（关键：交叉留出，否则分段模型必然"赢"）
--------------------------------------------
1. **split-half**：奇数 kf 拟合 / 偶数 kf 评估（测插值是否过拟合）
2. **extrapolate**：前 70% 里程拟合 / 后 30% 评估（测能否**外推**到没数据的段）

⚠️ 外推才是真问题：文档 §9.3 指出末段 ~1/4 会话没被任何回环碰过、保留冻结常量偏移。
分段模型若只在插值上赢、外推上输 ⇒ 那只是记忆，救不了 3D。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT))


# ---------- 数据 ----------
def load(sid: str):
    z = np.load(ROOT / "navmesh_memory" / "wrld_home-7cf435ea" / "sessions" / sid / "poses.npz")
    d = np.asarray(z["dist_m"], float)
    o = np.argsort(d)
    return (d[o], np.asarray(z["T_map"], float)[o][:, :2, 3],
            np.asarray(z["T_dr"], float)[o][:, :2, 3])


# ---------- 模型 ----------
def fit_affine(P: np.ndarray, Q: np.ndarray):
    """最小二乘：Q ≈ A @ P + b。返回 (A(2,2), b(2,))。样本 < 3 时退化为单位+零。"""
    if len(P) < 3:
        return np.eye(2), np.zeros(2)
    X = np.column_stack([P, np.ones(len(P))])
    sol, *_ = np.linalg.lstsq(X, Q, rcond=None)
    return sol[:2, :2].T.copy(), sol[2].copy()


class SegDeform:
    """按里程分 K 段的仿射形变场；mode='const' 段内常数，'linear' 段内线性插值（连续）。"""

    def __init__(self, edges: np.ndarray, mode: str):
        self.edges, self.mode = edges, mode
        self.nodes: list[tuple[np.ndarray, np.ndarray]] = []

    def fit(self, d: np.ndarray, P: np.ndarray, Q: np.ndarray) -> "SegDeform":
        K = len(self.edges) - 1
        if self.mode == "const":
            for i in range(K):
                m = (d >= self.edges[i]) & (d < self.edges[i + 1]) if i < K - 1 \
                    else (d >= self.edges[i]) & (d <= self.edges[i + 1])
                self.nodes.append(fit_affine(P[m], Q[m]))
        else:
            # 节点取在段中心，用**重叠窗口**（半径为半段长）估 ⇒ 相邻节点共享数据 ⇒ 连续
            cen = 0.5 * (self.edges[:-1] + self.edges[1:])
            half = 0.5 * np.diff(self.edges)
            for c, h in zip(cen, half):
                m = np.abs(d - c) <= max(h, 1e-6)
                self.nodes.append(fit_affine(P[m], Q[m]))
            self.cen = cen
        return self

    def __call__(self, d: np.ndarray, P: np.ndarray) -> np.ndarray:
        K = len(self.edges) - 1
        out = np.empty_like(P)
        if self.mode == "const":
            for i in range(K):
                m = (d >= self.edges[i]) & (d < self.edges[i + 1]) if i < K - 1 \
                    else (d >= self.edges[i]) & (d <= self.edges[i + 1])
                if m.any():
                    out[m] = P[m] @ self.nodes[i][0].T + self.nodes[i][1]
            # 外推：超出范围的用最近一段（分段常数模型唯一合理的外推）
            lo, hi = d < self.edges[0], d > self.edges[-1]
            if lo.any():
                out[lo] = P[lo] @ self.nodes[0][0].T + self.nodes[0][1]
            if hi.any():
                out[hi] = P[hi] @ self.nodes[-1][0].T + self.nodes[-1][1]
        else:
            # 段内线性插值（A/b 逐元素），两端 clamp（不外推，避免爆掉）
            t = np.clip((d - self.cen[0]) / max(self.cen[-1] - self.cen[0], 1e-9), 0.0, 1.0)
            u = np.clip(t * (K - 1), 0, K - 1)
            i0 = np.clip(np.floor(u).astype(int), 0, K - 1)
            i1 = np.clip(i0 + 1, 0, K - 1)
            w = u - i0
            for k in range(len(d)):
                a0, b0 = self.nodes[i0[k]]
                a1, b1 = self.nodes[i1[k]]
                A = a0 + w[k] * (a1 - a0)
                bb = b0 + w[k] * (b1 - b0)
                out[k] = A @ P[k] + bb
        return out


def edges_by_quantile(d: np.ndarray, K: int) -> np.ndarray:
    e = np.quantile(d, np.linspace(0, 1, K + 1))
    e[0], e[-1] = d.min(), d.max()
    return np.unique(e)


def edges_by_meter(d: np.ndarray, step: float) -> np.ndarray:
    K = max(1, int(np.ceil((d.max() - d.min()) / step)))
    return np.linspace(d.min(), d.max(), K + 1)


# ---------- 评估 ----------
def metrics(err: np.ndarray) -> str:
    return f"中位 {np.median(err):6.3f} · p90 {np.percentile(err,90):6.3f} · rms {np.sqrt((err**2).mean()):6.3f}"


def run(sid: str) -> None:
    d, P_map, P_dr = load(sid)
    n = len(d)
    print(f"\n{'='*78}\n会话 {sid} · {n} 关键帧 · 里程 {d.min():.1f}→{d.max():.1f} m"
          f" · DR→地图修正量中位 {np.median(np.linalg.norm(P_map-P_dr,axis=1)):.3f} m")
    print(f"{'='*78}")

    splits = {
        "split-half（奇拟合/偶评估）": (np.arange(n) % 2 == 1, np.arange(n) % 2 == 0),
        "extrapolate（前70%拟合/后30%评估）": (d <= np.quantile(d, 0.70), d > np.quantile(d, 0.70)),
    }
    variants = [("identity", None, None), ("global_affine", None, None)]
    for K in (4, 8, 16, 32):
        variants.append((f"seg_const K={K}", K, "const"))
        variants.append((f"seg_linear K={K}", K, "linear"))

    for sname, (tr, te) in splits.items():
        print(f"\n── {sname} ──  (train {tr.sum()} / test {te.sum()})")
        base = None
        for vname, K, mode in variants:
            if vname == "identity":
                pred = P_dr[te]
            elif vname == "global_affine":
                A, b = fit_affine(P_dr[tr], P_map[tr])
                pred = P_dr[te] @ A.T + b
            else:
                e = edges_by_quantile(d[tr], K)
                m = SegDeform(e, mode).fit(d[tr], P_dr[tr], P_map[tr])
                pred = m(d[te], P_dr[te])
            err = np.linalg.norm(pred - P_map[te], axis=1)
            if base is None:
                base = np.median(err)
            print(f"  {vname:<20} {metrics(err)}   改善 {100*(1-np.median(err)/base):+6.1f}%")


if __name__ == "__main__":
    for sid in (sys.argv[1:] or ["20261001_044153", "20261005_235237"]):
        run(sid)
