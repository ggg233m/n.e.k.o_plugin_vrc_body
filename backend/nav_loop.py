# -*- coding: utf-8 -*-
"""回环检测 + 平移位姿图：给在线航位推算加上"回到见过的地方就对齐"。

为什么只优化平移
----------------
朝向来自 HMD（SteamVR 追踪，不随路程漂移），漂的只有 OSC 速度积分出来的平移（实测约 2.8% 路程）。
所以位姿图的变量只有每个关键帧的 (x, y)，边是线性的：
* 里程计边：相邻关键帧的航位推算位移，σ = σ0 + frac·路程；
* 回环边：PnP 解出的相对位姿（旋到地图系）。
整个问题是一个加权图拉普拉斯线性方程，x、y 共用一个矩阵，不需要 SE(3) 求解器。

回环怎么验
----------
候选：比当前帧早、OSC 路程至少隔 ``min_path_m``（同一次经过不算回环）、当前估计下在漂移半径
（base + frac·路程）+ 双目可视距离之内、朝向差 ≤ ``max_view_deg`` 的关键帧，按距离取前几个。
验证：ORB 互为最近邻 → 候选关键帧的双目 3D 点 + 当前帧 2D → PnP RANSAC + LM；
门槛 = 内点数、覆盖、重投影误差，再加两道几何门：
* PnP 的相对 yaw 必须和 HMD 的相对 yaw 一致（HMD 不漂，这是最强的外点过滤）；
* 回环给的相对位置与当前图的预测差不超过漂移半径（远处的相似纹理不能把人拽走）。
收进图之后做 Cauchy IRLS：彼此矛盾的回环里残差大的被降权。

坐标同 nav_mapping：地图系 x 前 y 左 z 上，追踪米；路程、σ、半径用世界米。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 列 = 光学系（x 右 y 下 z 前）的轴在 base（x 前 y 左 z 上）中的坐标。
M_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass
class LoopConfig:
    orb_features: int = 1000
    max_depth_m: float = 8.0          # 追踪米；再远视差 < 3 px，深度噪声太大
    min_path_m: float = 3.0           # 世界米：两帧之间 OSC 路程至少这么多才算"重访"
    search_base_m: float = 1.0        # 世界米
    search_frac: float = 0.06         # 漂移半径随路程增长（实测 2.8%，留两倍余量）
    search_max_m: float = 6.0
    # 追踪米：两帧相机最多隔这么远还能看到同一片东西。
    # 3.0（= 2.27 世界米）对 102.45° FOV 偏保守：见 Docs/archive/停顿后地图错位-根因诊断（2026-10-01）.md §九，
    # 录制 20261001_044153 上尾部 0 回环的唯一病灶就是它 —— 唯一能启动尾部链条的那条回环
    # (a=181,b=385) 真实相隔 5.45 追踪米，被 3.0 判 too_far；放到 6.0 后尾部 16 条回环全部出现，
    # 且经交叉留出验证为**真实漂移的修正**（留出组残差 4.91→0.27 m、深度点云倒角 3.86→0.08 m）。
    # ⚠️ 5.45 vs 6.0 余量只有 10%，且漂移半径封顶 search_max_m=6.0 会在更长的录制上复发（§九.6）。
    max_offset_m: float = 6.0
    max_view_deg: float = 60.0
    max_candidates: int = 4
    # ---- 外观候选（词袋）：让候选发现**不再只依赖当前估计位置** ----
    # 病灶（`Docs/archive/停顿后地图错位-根因诊断（2026-10-01）.md` §10）：位置候选的排序键就是被漂移
    # 污染的位置本身 ⇒ 漂移让轨迹自我折叠，真配对 (181,385) 排到第 60 名，59 个不相干帧在
    # "几何上更近"。外观候选与位置无关，是唯一能跳出这个自指的入口。
    # 0 = 关闭（**默认关**）。> 0 时每帧额外试这么多外观候选，与位置候选取并集。
    bow_candidates: int = 0
    # 先取这么多外观候选，**过完 min_path_m 再截断**到 bow_candidates（顺序不能反，见
    # `_bow_candidates` 的注释）。
    bow_shortlist: int = 32
    # 离线训好的词汇树（tools/train_bow_vocab.py）。空字符串 = 不启用。
    # ⚠️ 词汇树是世界相关的：跨路线迁移实测只掉 0.5~4.6 pt，但**跨世界未测**，换世界要重训。
    bow_vocab: str = "models/bow_vocab.npz"
    ratio: float = 0.8
    min_inliers: int = 50              # run6：30→50 回环 26→19 条，对参照中位 0.51→0.45 m
    reproj_px: float = 3.0
    min_coverage: float = 0.12
    yaw_tol_deg: float = 6.0
    odo_sigma0_m: float = 0.03        # 世界米
    odo_sigma_frac: float = 0.03
    loop_sigma_m: float = 0.10
    loop_sigma_frac: float = 0.03     # 回环 σ 随两帧距离增长（远点深度更不准）
    robust_k: float = 3.0
    world_scale: float = 0.755        # 由 OnlineNavConfig.world_scale 覆盖，别在这里改


@dataclass
class KeyframeFeatures:
    uv: np.ndarray        # N×2 左目像素（全部关键点，作查询）
    des: np.ndarray       # N×32
    xyz: np.ndarray       # M×3 头部 base 系（追踪米），只含有深度的点（作地图）
    des3d: np.ndarray     # M×32
    K: np.ndarray         # 3×3
    size: tuple[int, int]
    eye_y: float          # 左眼在头部 base 系的 y（= 半基线，左为正）


def extract_features(left_gray: np.ndarray, disp: np.ndarray, *, fx: float, cx: float, cy: float,
                     baseline_m: float, orb: Any, max_depth_m: float = 8.0) -> KeyframeFeatures:
    """左目 ORB + 同一张视差图给深度。点换到**头部** base 系：左眼在头中心左侧半个基线。"""
    h, w = left_gray.shape[:2]
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    kps, des = orb.detectAndCompute(left_gray, None)
    if des is None or not kps:
        z2, z3 = np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32)
        zd = np.zeros((0, 32), np.uint8)
        return KeyframeFeatures(z2, zd, z3, zd, K, (w, h), baseline_m / 2.0)
    uv = np.array([k.pt for k in kps], np.float32)
    ui = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    d = disp[vi, ui]
    ok = d > 1.0
    z = np.where(ok, fx * baseline_m / np.maximum(d, 1e-6), np.inf)
    ok &= z <= max_depth_m
    X = np.column_stack([(uv[ok, 0] - cx) * z[ok] / fx, (uv[ok, 1] - cy) * z[ok] / fx, z[ok]])
    P = X @ M_OPT.T + np.array([0.0, baseline_m / 2.0, 0.0])
    return KeyframeFeatures(uv, des, P.astype(np.float32), des[ok], K, (w, h), baseline_m / 2.0)


def _match_mutual(query: np.ndarray, train: np.ndarray, ratio: float) -> np.ndarray:
    if len(query) < 2 or len(train) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    fwd = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(query, train, k=2)
           if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    rev = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(train, query, k=2)
           if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    pairs = [(q, t) for q, t in fwd.items() if rev.get(t) == q]
    return np.array(pairs, int).reshape(-1, 2)


def relative_pose(map_kf: KeyframeFeatures, query: KeyframeFeatures, cfg: LoopConfig
                  ) -> tuple[dict[str, Any] | None, str]:
    """``query`` 头部在 ``map_kf`` 头部 base 系下的位姿 (R_ab, t_ab)，或 (None, 拒绝原因)。"""
    pairs = _match_mutual(query.des, map_kf.des3d, cfg.ratio)
    if len(pairs) < cfg.min_inliers:
        return None, "few_matches"
    obj = map_kf.xyz[pairs[:, 1]].astype(np.float64)
    img = query.uv[pairs[:, 0]].astype(np.float64)
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), query.K, None,
                                                 flags=cv2.SOLVEPNP_EPNP, reprojectionError=cfg.reproj_px,
                                                 iterationsCount=200, confidence=0.999)
    except cv2.error:
        return None, "pnp_error"
    if not ok or inl is None or len(inl) < cfg.min_inliers:
        return None, "few_inliers"
    idx = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[idx].reshape(-1, 1, 3), img[idx].reshape(-1, 1, 2),
                                      query.K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    cam = obj[idx] @ R.T + t
    if (cam[:, 2] <= 0.05).mean() > 0.1:
        return None, "behind_camera"
    uvw = cam @ query.K.T
    err = float(np.median(np.linalg.norm(uvw[:, :2] / uvw[:, 2:3] - img[idx], axis=1)))
    w, h = query.size
    cells = {(int(u / w * 6), int(v / h * 4)) for u, v in img[idx]}
    coverage = len(cells) / 24.0
    if coverage < cfg.min_coverage:
        return None, "low_coverage"
    # 地图点 P 在 a 的头部 base 系；PnP：X = R·P + t 是 b 的左眼光学系；b 头部 = M·X + e。
    e = np.array([0.0, query.eye_y, 0.0])
    R_ba = M_OPT @ R
    t_ba = M_OPT @ t + e
    R_ab = R_ba.T
    t_ab = -R_ab @ t_ba
    return {"R_ab": R_ab, "t_ab": t_ab, "inliers": int(len(idx)), "coverage": round(coverage, 3),
            "reproj_px": round(err, 2)}, "ok"
def _solve_laplacian(n: int, I: np.ndarray, J: np.ndarray, Z: np.ndarray, W: np.ndarray,
                     x0: np.ndarray) -> np.ndarray:
    """min Σ w‖x_j − x_i − z‖²，x_0 = x0 固定 → n×2。边 (I, J, Z, W) 已向量化。"""
    g = np.zeros((n, 2))
    for d in range(2):
        g[:, d] = np.bincount(J, W * Z[:, d], minlength=n) - np.bincount(I, W * Z[:, d], minlength=n)
    rows = np.concatenate([I, J, I, J])
    cols = np.concatenate([I, J, J, I])
    vals = np.concatenate([W, W, -W, -W])
    keep = rows != 0                      # 首节点那一行换成 x_0 = x0
    rows = np.append(rows[keep], 0)
    cols = np.append(cols[keep], 0)
    vals = np.append(vals[keep], 1.0)
    g[0] = x0
    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import splu
    except ImportError:                   # 没装 scipy：退回稠密（几百帧内仍可用）
        H = np.zeros((n, n))
        np.add.at(H, (rows, cols), vals)
        return np.linalg.solve(H, g)
    return splu(csc_matrix((vals, (rows, cols)), shape=(n, n))).solve(g)


def rotation_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


@dataclass
class _Node:
    R: np.ndarray            # 3×3 地图系朝向（来自 HMD，视为真值）
    dr_xy: np.ndarray        # 航位推算平移（追踪米）
    dist_m: float            # OSC 累计路程（世界米）
    feat: KeyframeFeatures | None


@dataclass
class _Loop:
    a: int
    b: int
    z: np.ndarray            # x_b − x_a，地图系，追踪米
    sigma: float             # 追踪米
    info: dict[str, Any]
    weight: float = 1.0


class LoopCloser:
    """关键帧（HMD 朝向 + 航位推算平移 + 左目特征）→ 回环 → 平移位姿图 → 修正后的位姿。

    节点必须按时间顺序、id 递增加入（与 KeyframeGridMapper 的 node_id 相同）。
    ``poses()`` 给 ``KeyframeGridMapper.update_poses``；``correct(T_dr)`` 把最新航位推算位姿
    接到最后一个关键帧的修正量上，给控制用。"""

    def __init__(self, cfg: LoopConfig | None = None) -> None:
        self.cfg = cfg or LoopConfig()
        self._nodes: dict[int, _Node] = {}
        self._order: list[int] = []
        self._x: dict[int, np.ndarray] = {}
        self.loops: list[_Loop] = []
        self.rejects: dict[str, int] = {}
        # 回环健康度：距上次**接受**回环过去了多少关键帧 / 多少 OSC 路程。
        # 「尾部长期 0 回环」此前没有任何外显，只能事后翻录制才发现（见
        # Docs/archive/停顿后地图错位-根因诊断（2026-10-01）.md §七.3），这里让它自己报警。
        self._last_loop_kf: int | None = None
        self._last_loop_idx: int | None = None
        self._last_loop_dist_m: float | None = None
        # PnP 过了、再比 HMD 朝向的每个候选：(b, a, 带符号 yaw 差°, 是否接受)。VRChat 自己转/传送了人
        # （重生点、传送门）而 HMD 不知道时，这里会出现一串同号的大偏差。
        self.yaw_checks: deque[tuple[int, int, float, bool]] = deque(maxlen=64)
        self.orb = cv2.ORB_create(nfeatures=self.cfg.orb_features)
        # 外观候选（词袋）。默认关；开了但词汇树不存在也只是没有外观候选，回环照常跑。
        self._bow: Any = None
        self._bow_ready = False
        self._bow_reason = "disabled" if self.cfg.bow_candidates <= 0 else "not_loaded"
        self._bow_docs = 0
        if self.cfg.bow_candidates > 0:
            self._init_bow()

    def _init_bow(self) -> None:
        """载入词汇树、建空索引。**任何失败都不抛** —— 外观只是候选来源之一。"""
        raw = str(self.cfg.bow_vocab or "").strip()
        if not raw:
            self._bow_reason = "no_vocab_path"
            return
        p = Path(raw)
        if not p.is_absolute():
            root = Path(__file__).resolve().parent.parent
            cand = root / raw
            p = cand if cand.exists() else Path(raw)
        if not p.exists():
            self._bow_reason = "vocab_missing"
            return
        try:
            from .nav_bow import BowIndex, BowVocabulary
        except ImportError:
            # 离线回放常把 nav_loop 当单文件加载（没有父包），退回绝对导入。
            try:
                from nav_bow import BowIndex, BowVocabulary  # type: ignore[no-redef]
            except ImportError:                             # pragma: no cover
                self._bow_reason = "import_error"
                return
        try:
            self._bow = BowIndex(BowVocabulary.load(str(p)))
        except Exception:                                   # pragma: no cover
            self._bow = None
            self._bow_reason = "load_error"
            return
        self._bow_ready = True
        self._bow_reason = "ok"

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, k: object) -> bool:
        return k in self._nodes

    # ---- 查询 ----
    def pose(self, k: int) -> np.ndarray:
        n = self._nodes[k]
        T = np.eye(4)
        T[:3, :3] = n.R
        T[:2, 3] = self._x[k]
        return T

    def poses(self) -> dict[int, np.ndarray]:
        return {k: self.pose(k) for k in self._order}

    def offset(self) -> np.ndarray:
        """最后一个关键帧的修正量（修正后 − 航位推算），追踪米。"""
        if not self._order:
            return np.zeros(2)
        k = self._order[-1]
        return self._x[k] - self._nodes[k].dr_xy

    def correct_at(self, k: int, T_dr: np.ndarray) -> np.ndarray:
        """按关键帧 k 的修正量平移一个航位推算位姿（k 之后、下一个关键帧之前的逐帧位姿用）。"""
        T = np.array(T_dr, float, copy=True)
        T[:2, 3] = T[:2, 3] + self._x[k] - self._nodes[k].dr_xy
        return T

    def correct(self, T_dr: np.ndarray) -> np.ndarray:
        T = np.array(T_dr, float, copy=True)
        T[:2, 3] = T[:2, 3] + self.offset()
        return T

    # ---- 建图 ----
    def _drift_radius(self, path_m: float) -> float:
        c = self.cfg
        return min(c.search_max_m, c.search_base_m + c.search_frac * path_m)

    def add_keyframe(self, k: int, T_dr: np.ndarray, dist_m: float,
                     feat: KeyframeFeatures | None) -> list[dict[str, Any]]:
        """加入关键帧并尝试回环；返回本次接受的回环（空列表 = 没有，位姿图没变）。"""
        k = int(k)
        if self._order and k <= self._order[-1]:
            raise ValueError("关键帧 id 必须递增")
        T_dr = np.asarray(T_dr, float)
        node = _Node(T_dr[:3, :3].copy(), T_dr[:2, 3].copy(), float(dist_m), feat)
        if self._order:
            p = self._order[-1]
            self._x[k] = self._x[p] + (node.dr_xy - self._nodes[p].dr_xy)
        else:
            self._x[k] = node.dr_xy.copy()
        self._nodes[k] = node
        self._order.append(k)
        accepted = []
        for a in self._candidates(k):
            loop, why = self._verify(a, k)
            if loop is None:
                self.rejects[why] = self.rejects.get(why, 0) + 1
                continue
            self.loops.append(loop)
            accepted.append({"a": a, "b": k, **loop.info})
        # 索引**必须在 query 之后**才收当前帧：先 add 再 query 会检索到自己，
        # 产生 a==b 的自回环（z≈0、内点≈1000，所有门都过），实测回环数 623→949、覆盖帧 19→36 全是假的。
        if self._bow_ready and node.feat is not None and len(node.feat.des):
            try:
                self._bow.add(k, node.feat.des)
                self._bow_docs = len(self._bow)
            except Exception:                               # pragma: no cover
                self._bow_ready = False
                self._bow_reason = "add_error"
        if accepted:
            self._last_loop_kf = k
            self._last_loop_idx = len(self._order) - 1     # 本帧已在上面 append，索引即末位
            self._last_loop_dist_m = node.dist_m
            self.optimize()
        return accepted

    def _candidates(self, b: int) -> list[int]:
        c, s = self.cfg, self.cfg.world_scale
        nb = self._nodes[b]
        if nb.feat is None or not len(nb.feat.des):
            return []
        fb = nb.R[:2, 0] / max(1e-9, float(np.hypot(*nb.R[:2, 0])))
        out = []
        for a in self._order[:-1]:
            na = self._nodes[a]
            path = nb.dist_m - na.dist_m
            if path < c.min_path_m or na.feat is None or len(na.feat.des3d) < c.min_inliers:
                continue
            gap = float(np.hypot(*(self._x[b] - self._x[a])))
            if gap * s > self._drift_radius(path) + c.max_offset_m * s:
                continue
            fa = na.R[:2, 0] / max(1e-9, float(np.hypot(*na.R[:2, 0])))
            if math.degrees(math.acos(float(np.clip(fa @ fb, -1.0, 1.0)))) > c.max_view_deg:
                continue
            out.append((gap, a))
        res = [a for _g, a in sorted(out)[:c.max_candidates]]
        if self._bow_ready:
            res += self._bow_candidates(b, nb)
        seen: set[int] = set()
        uniq = []
        for a in res:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        return uniq

    def _bow_candidates(self, b: int, nb: "_Node") -> list[int]:
        """外观候选。**先过门、再截断** —— 顺序反了会全军覆没。

        ⚠️ 顺序为什么不能反：长时间停顿时 `dist_m` 不动，一堆积压的重复帧在外观上都"最像"，
        会把 top-N 名额占满；若先截断再过 `min_path_m`，这些名额全被废掉 —— 9-29 录制
        （41% 是停顿）实测存活 **0.0~0.2 / 12**，97~100% 的帧一个真候选都不剩。
        所以先取 `bow_shortlist` 个、用 `min_path_m` 筛掉"同一次经过"，再截断到 `bow_candidates`。

        ⚠️ `min_path_m` 必须在这里**自己过**：`_verify` 不查这个门（它只写在位置分支里），
        绕过它就会混进大量相邻帧平凡对（实测 63%，路程中位 1.4 m vs 正常 15.9 m）。
        """
        c = self.cfg
        if nb.feat is None or not len(nb.feat.des):
            return []
        try:
            ranked = self._bow.query(nb.feat.des, top_n=int(max(1, c.bow_shortlist)))
        except Exception:                                   # pragma: no cover
            self._bow_ready = False
            self._bow_reason = "query_error"
            return []
        out: list[int] = []
        for a, _score in ranked:
            if a == b:
                continue
            na = self._nodes.get(a)
            if na is None or na.feat is None or len(na.feat.des3d) < c.min_inliers:
                continue
            if nb.dist_m - na.dist_m < c.min_path_m:        # 同一次经过不算重访
                continue
            out.append(a)
            if len(out) >= int(c.bow_candidates):
                break
        return out

    def _verify(self, a: int, b: int) -> tuple[_Loop | None, str]:
        c, s = self.cfg, self.cfg.world_scale
        na, nb = self._nodes[a], self._nodes[b]
        rel, why = relative_pose(na.feat, nb.feat, c)
        if rel is None:
            return None, why
        # HMD 朝向不漂：PnP 的相对转角必须与它一致，这是最强的外点过滤。
        R_err = (na.R @ rel["R_ab"]).T @ nb.R
        yaw_err = rotation_angle_deg(R_err)
        signed = round(math.degrees(math.atan2(R_err[1, 0], R_err[0, 0])), 1)
        if yaw_err > c.yaw_tol_deg:
            self.yaw_checks.append((b, a, signed, False))
            return None, "rotation_mismatch"
        t_ab = rel["t_ab"]
        if float(np.hypot(*t_ab[:2])) > c.max_offset_m:
            return None, "too_far"
        z = (na.R @ t_ab)[:2]
        resid = float(np.hypot(*(self._x[b] - self._x[a] - z))) * s
        if resid > self._drift_radius(nb.dist_m - na.dist_m):
            return None, "outside_drift_radius"
        sigma = (c.loop_sigma_m + c.loop_sigma_frac * float(np.linalg.norm(t_ab)) * s) / s
        info = {**{k: rel[k] for k in ("inliers", "coverage", "reproj_px")},
                "rot_err_deg": round(yaw_err, 2), "correction_m": round(resid, 3),
                "offset_m": round(float(np.hypot(*t_ab[:2])) * s, 3)}
        self.yaw_checks.append((b, a, signed, True))
        return _Loop(a, b, z, sigma, info), "ok"
    def optimize(self, iterations: int = 8) -> float:
        """加权线性最小二乘（首节点固定）+ Cauchy IRLS；返回节点最大平移变化（追踪米）。

        法方程是加权图拉普拉斯，x、y 共用。用稀疏 LU：稠密 solve 在 853 帧、OpenBLAS 20 线程与
        VRChat 抢核时实测一次 6 s，IRLS 最多 8 次，建图线程握着导航锁，控制线程跟着卡死。"""
        c, s = self.cfg, self.cfg.world_scale
        n = len(self._order)
        if n < 2:
            return 0.0
        idx = {k: i for i, k in enumerate(self._order)}
        edges: list[tuple[int, int, np.ndarray, float]] = []
        for p, k in zip(self._order, self._order[1:]):
            path = max(0.0, self._nodes[k].dist_m - self._nodes[p].dist_m)
            sig = (c.odo_sigma0_m + c.odo_sigma_frac * path) / s
            edges.append((idx[p], idx[k], self._nodes[k].dr_xy - self._nodes[p].dr_xy, sig))
        old = {k: self._x[k].copy() for k in self._order}
        x0 = self._nodes[self._order[0]].dr_xy
        x = np.array([self._x[k] for k in self._order])
        oi = np.array([e[0] for e in edges], int)
        oj = np.array([e[1] for e in edges], int)
        oz = np.array([e[2] for e in edges], float).reshape(-1, 2)
        ow = np.array([1.0 / (e[3] * e[3]) for e in edges], float)
        li = np.array([idx[L.a] for L in self.loops], int)
        lj = np.array([idx[L.b] for L in self.loops], int)
        lz = np.array([L.z for L in self.loops], float).reshape(-1, 2)
        lsig = np.array([L.sigma for L in self.loops], float)
        lw = np.array([L.weight for L in self.loops], float)
        for _ in range(max(1, iterations)):
            I = np.concatenate([oi, li])
            J = np.concatenate([oj, lj])
            Z = np.concatenate([oz, lz])
            W = np.concatenate([ow, np.maximum(lw, 1e-6) / (lsig * lsig)])
            x = _solve_laplacian(n, I, J, Z, W, x0)
            if not self.loops:
                break
            r = np.hypot(*(x[lj] - x[li] - lz).T)
            w_new = 1.0 / (1.0 + (r / (c.robust_k * lsig)) ** 2)
            changed = bool(np.any(np.abs(w_new - lw) > 1e-3))
            lw = w_new
            if not changed:
                break
        for L, w in zip(self.loops, lw):
            L.weight = float(w)
        for k, i in idx.items():
            self._x[k] = x[i]
        return max(float(np.hypot(*(self._x[k] - old[k]))) for k in self._order)

    def status(self) -> dict[str, Any]:
        s = self.cfg.world_scale
        # 健康度：还没出过回环时，"距上次回环"= 从起点到现在（一样是单调增长、能报警）。
        if not self._order:
            kf_since, path_since = 0, 0.0
        elif self._last_loop_idx is None:
            kf_since = len(self._order)
            path_since = self._nodes[self._order[-1]].dist_m - self._nodes[self._order[0]].dist_m
        else:
            kf_since = len(self._order) - 1 - self._last_loop_idx
            path_since = self._nodes[self._order[-1]].dist_m - (self._last_loop_dist_m or 0.0)
        return {"keyframes": len(self._order), "loops": len(self.loops),
                "last_loop_keyframe": self._last_loop_kf,
                "kf_since_last_loop": kf_since,
                "path_since_last_loop_m": round(path_since, 1),
                "loops_downweighted": sum(1 for L in self.loops if L.weight < 0.5),
                # 外观候选的状态必须外显：默认关，但一旦开了却没词汇树，最容易变成
                # "以为在跑其实没跑"。bow_reason 直接说原因（ok / vocab_missing / ...）。
                "bow_ready": bool(self._bow_ready),
                "bow_reason": self._bow_reason,
                "bow_docs": int(self._bow_docs),
                "bow_candidates": int(self.cfg.bow_candidates),
                "correction_m": round(float(np.hypot(*self.offset())) * s, 3),
                "rejects": dict(self.rejects),
                "yaw_checks": [{"b": b, "a": a, "yaw_err_deg": e, "accepted": ok}
                               for b, a, e, ok in list(self.yaw_checks)[-16:]]}
