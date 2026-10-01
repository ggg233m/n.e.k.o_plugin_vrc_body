# -*- coding: utf-8 -*-
"""ORB 词袋（BoW）索引 —— 让回环**候选发现不再依赖当前位置**。

背景（`Docs/停顿后地图错位-根因诊断（2026-10-01）.md` §9.9）：
`LoopCloser._candidates` 按 **当前估计位置的欧氏距离** 取最近 N 个当候选。
而位置本身就是漂移量 ⇒ 漂移一大，轨迹自我折叠，59 个不相干的帧比真匹配"几何上更近"，
真配对排到第 60 名，永远进不了前 N —— 这是"用错误量去找错误量的修正"。
修复：再加一路 **外观** 候选（本模块），与位置候选取并集。

为什么不用现成 DBoW2 —— ⚠️ **两个"现成跑不通"的事实（已实测，2026-10-02）**：

1. **cv2 的 Python 绑定里根本没有 DBoW2 的检索部分。**
   `cv2` 4.14.0 只有 3 个 BOW 符号：`BOWTrainer` / `BOWKMeansTrainer` / `BOWImgDescriptorExtractor`。
   `cv2.ORBVocabulary` / `cv2.DBoW2` / `cv2.Database` / `cv2.InvertedFile` **一律不存在**
   ——`ORBVocabulary`（带倒排索引 + 打分）是 ORB-SLAM 自己写在 C++ 里的，OpenCV 从未发布它。
   ⇒ cv2.BOW 只能**训词表**和**算每帧的 BoW 直方图**，**没有"检索最像的历史帧"这一步**，
   而我们要的恰恰就是那一步。
2. **cv2 的 k-means 只吃 float32，喂不了 ORB 的 uint8 二进制描述子。**
   `BOWKMeansTrainer(16,(3,10,0.001),1,KMEANS_PP_CENTERS).cluster(D_uint8)`
   → `error: (-215:Assertion failed) data0.dims <= 2 && type == CV_32F`；
   `cluster(D.astype(np.float32))` 能跑，但那是**欧氏 k-means**——对汉明空间是错的度量。
   ⇒ 想用它就得先把汉明距离硬塞进欧氏 k-means，**先引入一个错误的度量，再谈可审计**。

至于仓库里那份 `.slam_probe/ORB_SLAM3/Thirdparty/DBoW2`：**许可没问题**（BSD，见 `README.txt`），
但它是 **C++ / 要链接**；ORB-SLAM3 已因 GPLv3 出局，把它拖进在线主链路不合算。
**再强调一次：真正的障碍是"cv2 没有检索 API"，不是"许可"**（我原先写的"依赖扩展/要管内存"
是错的说法，已删）。

本模块只用 numpy，**汉明 k-means（质心按位取多数值）+ 稠密直方图 + idf 打分**，自写、可审计。
词汇树需**离线训练**后随包发布（见 `train()` / `save()` / `load()`），在线只做检索。

实测口径（录制 `20261001_044153`，见 `.tmp/_bow_check.py`）：
* 朴素"对全部历史帧做描述子互匹配"要 ~5 ms/历史帧 ⇒ 第 385 帧要 **1835 ms**，在线不可用。
  本模块把单帧查询压到毫秒级。
* ⚠️ 本模块只解决"找到候选"，**不解决**验证门。配合 `search_max_m` 抬到 10.0 才完整
  （外观候选到位后，漂移半径封顶才成为可见约束）。
"""
from __future__ import annotations

import numpy as np

__all__ = ["BowVocabulary", "BowIndex", "bits"]


def bits(des: np.ndarray) -> np.ndarray:
    """(N,32) uint8 ORB 描述子 → (N,256) uint8 位平面。"""
    return np.unpackbits(np.ascontiguousarray(des, dtype=np.uint8), axis=1)


_POPCOUNT = np.array([bin(i).count("1") for i in range(65536)], np.uint8)


class BowVocabulary:
    """汉明词汇树。质心 = 按位多数投票（不是均值），分裂用汉明 k-means。

    结构：`levels[l]` 形状 (n_nodes_l, branching, 256) uint8，n_nodes_l = branching ** l。
    叶子数 = branching ** depth。`transform()` 返回的 word id 就是叶子在树里的序号。
    """

    def __init__(self, branching: int = 16, depth: int = 3, seed: int = 0):
        self.branching = int(branching)
        self.depth = int(depth)
        self.seed = int(seed)
        self.levels: list[np.ndarray] = []
        self.n_words = self.branching ** self.depth

    # ---------- 训练 ----------
    def train(self, des: np.ndarray, iters: int = 10, max_rows: int = 120_000) -> "BowVocabulary":
        """在给定描述子上训练词汇树。描述子应来自**与在线场景同类**的世界。"""
        des = np.asarray(des, dtype=np.uint8)
        if len(des) > max_rows:
            rng = np.random.default_rng(self.seed)
            des = des[rng.choice(len(des), max_rows, replace=False)]
        A = bits(des).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        cur = A
        groups = np.zeros(len(A), np.int64)     # 当前每个描述子所属节点
        for _ in range(self.depth):
            cents = np.empty((len(cur) and (groups.max() + 1) or 1, self.branching, 256), np.uint8)
            n_nodes = int(groups.max()) + 1
            cents = np.empty((n_nodes, self.branching, 256), np.uint8)
            new_groups = np.empty(len(A), np.int64)
            for node in range(n_nodes):
                m = groups == node
                B = cur[m]
                if len(B) < self.branching:
                    cents[node] = 0
                    new_groups[m] = 0
                    continue
                c = self._kmeans(B, self.branching, iters, rng)
                cents[node] = (c > 0.5).astype(np.uint8) * 255
                d = self._dist(B, c)
                new_groups[m] = d.argmin(1)
            self.levels.append(cents)
            groups = groups * self.branching + new_groups
        return self

    @staticmethod
    def _dist(A: np.ndarray, C: np.ndarray) -> np.ndarray:
        """汉明距离 (N,K)：‖a−b‖² = |a| + |b| − 2a·b，位平面上等价于汉明距离。"""
        return A.sum(1)[:, None] + C.sum(1)[None, :] - 2.0 * (A @ C.T)

    def _kmeans(self, A: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(A), k, replace=False)
        c = A[idx].copy()
        for _ in range(iters):
            lab = self._dist(A, c).argmin(1)
            for j in range(k):
                m = lab == j
                if m.any():
                    c[j] = A[m].mean(0)
        return (c > 0.5).astype(np.float32)

    # ---------- 变换 ----------
    def _packed(self) -> list[np.ndarray]:
        """质心打包成 uint64（每层 (nodes, branching, 4)），供快速汉明距离用。只算一次。"""
        if getattr(self, "_lv64", None) is None:
            out = []
            for cents in self.levels:
                p = np.ascontiguousarray(np.packbits(cents.reshape(-1, 256), axis=1))
                out.append(p.view(np.uint64).reshape(cents.shape[0], cents.shape[1], 4))
            self._lv64 = out
        return self._lv64

    def transform(self, des: np.ndarray) -> np.ndarray:
        """(N,32) 描述子 → (N,) 叶子 word id。

        走**打包 popcount** 而不是"位平面 float32 + einsum"：后者每层要物化 (N,k,256) float32
        （N=1000、k=16 时 16 MB/层）且 einsum 走不到 BLAS，实测 57.7 ms/帧；
        打包版每层只要 (N,k,4) uint64（512 KB），**实测 6.5 ms/帧，快 8.9×，结果逐字一致**
        （`.tmp/_speed.py`，前 60 帧逐一比对）。
        """
        if not self.levels:
            raise RuntimeError("词汇树未训练：先 train() 或 load()")
        A = np.ascontiguousarray(des, dtype=np.uint8).view(np.uint64)      # (N,4)
        node = np.zeros(len(A), np.int64)
        for C in self._packed():
            x = np.ascontiguousarray(A[:, None, :] ^ C[node])              # (N,k,4)
            d = _POPCOUNT[x.view(np.uint16)].sum(-1, dtype=np.uint16)      # (N,k,8) → (N,k)
            node = node * self.branching + d.argmin(1)
        return node

    # ---------- 存取 ----------
    def save(self, path: str) -> None:
        np.savez_compressed(path,
                            levels=np.array(self.levels, dtype=object),
                            branching=np.array(self.branching),
                            depth=np.array(self.depth),
                            seed=np.array(self.seed))

    @classmethod
    def load(cls, path: str) -> "BowVocabulary":
        z = np.load(path, allow_pickle=True)
        v = cls(int(z["branching"]), int(z["depth"]), int(z["seed"]))
        v.levels = [np.asarray(x, np.uint8) for x in z["levels"]]
        return v


class BowIndex:
    """按关键帧累积的词袋直方图 + idf 加权检索。在线只做 add / query。

    打分 = **直方图交集 / min(双方 L1 范数)**，对"两帧描述子总数不同"鲁棒。
    ⚠️ 两侧的范数都必须按**各自全部词**算，不能只算查询词那一撮 —— 早期版本只在查询词
    支撑集上算文档范数，把分数虚抬到 0.86（真值 0.47）且排名完全失真（`.tmp/_bow_dbg.py`）。
    """

    def __init__(self, vocab: BowVocabulary, cap_docs: int = 8192):
        self.v = vocab
        self.V = vocab.n_words
        self._ids: list[int] = []
        self._df = np.zeros(self.V, np.int32)
        self._cap = int(cap_docs)
        self._C = np.zeros((256, self.V), np.float32)     # 词频，按块增长
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(self, kf_id: int, des: np.ndarray) -> None:
        w = self.v.transform(des)
        cnt = np.bincount(w, minlength=self.V).astype(np.float32)
        if self._n == self._C.shape[0]:
            if self._n >= self._cap:
                return                                    # 超上限则不再收新帧（见 cap_docs）
            self._C = np.resize(self._C, (min(self._cap, self._C.shape[0] * 2), self.V))
        self._C[self._n] = cnt
        self._df[cnt > 0] += 1
        self._ids.append(int(kf_id))
        self._n += 1

    def query(self, des: np.ndarray, top_n: int = 4, min_score: float = 0.0
              ) -> list[tuple[int, float]]:
        """返回 [(kf_id, score)]，score ∈ [0,1]，越大越像同一处。"""
        if self._n == 0:
            return []
        n = self._n
        q = np.bincount(self.v.transform(des), minlength=self.V).astype(np.float32)
        nz = np.nonzero(q)[0]
        if not len(nz):
            return []
        idf_all = np.log((n + 1.0) / (self._df.astype(np.float32) + 1.0)) + 1.0
        C = self._C[:n]
        D = C * idf_all                                   # (n, V)
        qv = (q * idf_all)[nz]
        qs = float(qv.sum())
        if qs <= 0:
            return []
        dn = D.sum(1)
        num = np.minimum(D[:, nz], qv[None, :]).sum(1)
        s = num / np.minimum(dn, qs)
        order = np.argsort(-s)
        out = [(self._ids[int(j)], float(s[j])) for j in order[:max(top_n, 0)] if s[j] >= min_score]
        return out[:top_n]
