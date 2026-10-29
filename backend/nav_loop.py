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
    # ⚠️ 斜率项的**前提已被本仓库实测否定**：回环残差与路程**无关** —— 拟合 c = a + b·L 得
    # 地板 a=+25.6 cm、b=0.009 %/m、**R²=0.000**；6 组 / 5 场会话 / 两条独立管线 R² 全 ≤0.091
    # ⇒「漂移是路程的 2.3–4.5%、线性增长」不成立（Docs/漂移形态诊断（2026-10-05）.md、Docs/FACTS.md:134）。
    # 旧注释引的「实测 2.8%」出自 2026-09-27 的 run6 测量，已被上面这份取代。
    # 实测该闸门平时几乎不 binding（8 场录制 outside_drift_radius 只拒 0–2 条/场）；且它**越远越松**，
    # 与"最需要防长间隔假回环"相反。2026-10-14 复核：尾段 346↔401 正是被它卡掉（残差 2.967 vs 半径 2.965，差 2 mm），
    # 而该残差 ≈ 典型地板 22–26 cm 的 13 倍，属异常值，拦截未必是误伤。
    # 暂不改值：放宽零收益、且会引入方向不可验证（30° 残差）的尾段回环；要收紧先扫 8 场确认不丢真回环。
    search_frac: float = 0.06         # 线性前提已撤回，当前仅作宽松安全网；见上
    # 🔴 2026-10-29 6.0 → 20.0：**上面那条预警复发了**（「封顶会在更长的录制上复发」）。
    #
    # 本场 20261029_060258（889 m / 1338 帧 / 736 s）实测：
    #   * 后半段(b>801) 有 343 个「物理回访」对（T_dr 距离 <2 m），却只产生 2 条回环；
    #   * 用图像内容（不依赖位姿）全库搜索证实：kf=1300 与 kf=1 **是同一地点**
    #     （good matches 341，与成功回环的 307 中位同量级），但 T_dr 认为相距 13.6 m；
    #   * 本场 min(6.0, 1.0+0.06*769) = 6.0 ⇒ 需要 13.6 m 的真对**永远进不来**。
    #
    # 实测漂移率只有 **0.9–3.3%**（4 对图像证实的同地点对），模型假设的 6% 是偏保守的；
    # 坏的不是斜率，是 search_max_m 这个**硬上限把长路程的漂移封顶**。
    # 上限 6→20 后：回环 49→85 条，后半段回环 2→38 条，
    # 且 38 条质量全部过硬（inliers 中位 200 / min 59，reproj 0.68–1.31 px，coverage 0.125–0.417）；
    # 位姿图对 4 个「图像证实同地点」对的平均距离 10.07 → **7.62 m**（kf1300↔kf1：13.78 → 2.35 m）。
    #
    # ⚠️ 20.0 是**按本场 889 m 定**的：对应 2.3% 漂移率（在实测 0.9–3.3% 区间内）。
    # 更长的录制（>1 km）应继续上抬，或改成按路程比例且不封顶。
    # ⚠️ 已知代价：correction_m 最大值 2.65 → 7.91 m —— 放宽后能接上更远的真回环，
    # 修正量自然更大；这些回环的 inlier/coverage 都是优质的，不是假匹配。
    search_max_m: float = 20.0
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
    # 0 = 关闭。> 0 时每帧额外试这么多外观候选，与位置候选取并集。
    # 2026-10-10 默认改 4：20261010_033237（287 s/476 m）留出 30% 重访对，残差中位 3.52/3.02→0.64/0.47 m；
    # 4 场老录制留出无一变坏（001019 2.19/3.54→0.70/0.75）。代价：每关键帧 +27 ms（822 kf）/ +51 ms（3014 kf，
    # 主要是 BowIndex.query 随帧数线性涨）。8 个候选、放大搜索半径都没有额外收益。
    bow_candidates: int = 4
    # 先取这么多外观候选，**过完 min_path_m 再截断**到 bow_candidates（顺序不能反，见
    # `_bow_candidates` 的注释）。
    bow_shortlist: int = 32
    # 离线训好的词汇树（research/tools/train_bow_vocab.py）。空字符串 = 不启用。
    # ⚠️ 词汇树是世界相关的：跨路线迁移实测只掉 0.5~4.6 pt，但**跨世界未测**，换世界要重训。
    bow_vocab: str = "models/bow_vocab.npz"
    ratio: float = 0.8
    min_inliers: int = 50              # run6：30→50 回环 26→19 条，对参照中位 0.51→0.45 m
    reproj_px: float = 3.0
    min_coverage: float = 0.12
    yaw_tol_deg: float = 6.0
    # 2026-10-14 默认 ×2（0.03/0.03 → 0.06/0.06）：实测里程增量误差约路程 17%，0.03 把位姿压得太死。
    # 证据（.tmp/metric_arbiter.py + odo_online.py，三场带原图录制）σ×2 的回访错位 |e| 中位：
    # 033237 28.4→26.2、223033 34.6→26.6、003723 26.1→19.1 cm，NCC 同向变好；σ×4/×8 无额外收益。
    # 当初判否依据的点云倒角已被证为**钝尺子**（注入 20 cm 只变 ~1 cm、极小点≈随机，见记忆
    # metric-arbiter-chamfer-blind），故该否决作废。位姿代价：非尾段最大挪 5.6 cm，未引入假回环。
    odo_sigma0_m: float = 0.06        # 世界米
    odo_sigma_frac: float = 0.06
    loop_sigma_m: float = 0.10
    loop_sigma_frac: float = 0.03     # 回环 σ 随两帧距离增长（远点深度更不准）
    robust_k: float = 3.0
    # 是否把 **z** 也纳入位姿图（2026-10-28）。
    #
    # 默认 **False**（保持既有行为，T[2,3] 恒 0）—— 因为 DR 没有 z 里程：
    # `T_dr[2,3]` 从来是 0，odometry 边在 z 维上全为「不变」，图在 z 方向只剩回环边，
    # 除回环节点外全欠定。贸然打开会让 z 只在少数节点上有定义。
    #
    # 打开后的正确用法（见 `Docs/`）：先用 `z_edges_only` 探针把回环 z 边**算出来存下来**，
    # 量它们的自洽性（三元闭合），确认 z 真的可观测，再谈把它当优化变量。
    # 这条路线来自 VINS-Mono 的先例：「roll/pitch 可观测 ⇒ 只砍 roll/pitch，保留 x,y,z,yaw」；
    # 我们朝向全由 HMD 给定，却连 z 一起砍了。
    enable_z: bool = False
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
    """min Σ w‖x_j − x_i − z‖²，x_0 = x0 固定 → n×D。边 (I, J, Z, W) 已向量化。

    ``D`` 由 ``Z`` 的列数与 ``x0`` 的长度决定（2=只解 xy，3=解 xyz）。
    法方程是**加权图拉普拉斯**，与维数无关 —— 所以从 2 改到 3 只改这一处循环，
    LU 分解本身天然支持多个右端项（``splu(...).solve(g)`` 的 g 是 n×D）。

    为什么要支持 3 维（2026-10-28）：原先写死 2 维，把 z 一起砍掉了。
    但 VINS-Mono 的先例是「roll/pitch 可观测 ⇒ 只砍 roll/pitch，保留 x,y,z,yaw」；
    我们朝向来自 HMD（连 yaw 都不用估），却连 z 也砍了 ⇒ 位姿表 T[2,3] 恒为 0，
    任何竖向偏移都被 cam_h/gc 吸收，**地面高度无法成为优化变量**。
    实测该录制 T[2,3] 的 std = 0.000000（759 帧），见 `Docs/`。
    """
    D = int(np.asarray(Z).shape[1])
    g = np.zeros((n, D))
    for d in range(D):
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
    dr_xy: np.ndarray        # 航位推算平移（水平，追踪米）
    dist_m: float            # OSC 累计路程（世界米）
    feat: KeyframeFeatures | None
    # 航位推算的竖直分量（追踪米）。目前 DR 恒给 0（`T_dr[2,3]` 从来是 0），
    # 所以默认 0 不改变任何既有行为；留这个字段是为了让 z 进位姿图之后
    # 「odometry 边」能在三维里成立 —— 否则 z 维只有回环边、没有里程边，
    # 图会退化（除回环节点外全欠定）。
    dr_z: float = 0.0


@dataclass
class _Loop:
    a: int
    b: int
    # x_b − x_a，地图系，追踪米。**2 或 3 列**：由 `LoopConfig.enable_z` 决定。
    # PnP 给的相对平移本来就完整三维（`rel["t_ab"]`），原先 `[:2]` 把它丢了。
    z: np.ndarray
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
        # 回环覆盖（2026-10-29）：**哪些帧段落没有回环锚定**，不再只能事后翻 events 才知道。
        # 为什么需要它：20261029_060258（890 m）末段 **423 m（48%）无任何回环**，
        # 而 20261029_024332（405 m）只有 1 m —— 差异巨大却没有任何在线信号。
        # 下游（点云导出、导航）必须能知道「这段轨迹是纯航位推算，漂移可能 5–19 m」。
        # 口径：一个关键帧算「被回环锚定」当且仅当它出现在某条接受回环的 a 或 b 上。
        # 不用「距上次回环多少米」是因为那只看得见尾部，看不见中间的空洞。
        self._loop_anchored: set[int] = set()
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
        x = np.asarray(self._x[k], float).ravel()
        T[:2, 3] = x[:2]
        if self.cfg.enable_z and x.size >= 3:
            # z 只在打开 enable_z 时才非 0。默认路径下 `T[2,3]` 仍然**逐位恒 0**
            # （既有行为不变：所有消费方都假定它是 0）。
            T[2, 3] = x[2]
        return T

    def poses(self) -> dict[int, np.ndarray]:
        return {k: self.pose(k) for k in self._order}

    def offset(self) -> np.ndarray:
        """最后一个关键帧的修正量（修正后 − 航位推算），追踪米。2 或 3 维。"""
        if not self._order:
            return np.zeros(3 if self.cfg.enable_z else 2)
        k = self._order[-1]
        x = np.asarray(self._x[k], float).ravel()
        d = np.asarray(self._nodes[k].dr_xy, float)
        if not self.cfg.enable_z:
            return x[:2] - d
        # ⚠️ z 的「航位推算」是 `dr_z`（默认恒 0），不是 `dr_xy[2]`（不存在）。
        # 写错会让 offset 的 z 分量变成 −x[2]，把整片点云按两步翻转。
        return np.array([x[0] - d[0], x[1] - d[1], x[2] - float(self._nodes[k].dr_z)])

    def correct_at(self, k: int, T_dr: np.ndarray) -> np.ndarray:
        """按关键帧 k 的修正量平移一个航位推算位姿（k 之后、下一个关键帧之前的逐帧位姿用）。"""
        T = np.array(T_dr, float, copy=True)
        x = np.asarray(self._x[k], float).ravel()
        d = np.asarray(self._nodes[k].dr_xy, float)
        T[:2, 3] = T[:2, 3] + x[:2] - d[:2]
        if self.cfg.enable_z and x.size >= 3:
            T[2, 3] = T[2, 3] + x[2] - float(self._nodes[k].dr_z)
        return T

    def correct(self, T_dr: np.ndarray) -> np.ndarray:
        T = np.array(T_dr, float, copy=True)
        off = self.offset()
        T[:2, 3] = T[:2, 3] + off[:2]
        if self.cfg.enable_z and off.size >= 3:
            T[2, 3] = T[2, 3] + off[2]
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
        node = _Node(T_dr[:3, :3].copy(), T_dr[:2, 3].copy(), float(dist_m), feat,
                     dr_z=float(T_dr[2, 3]) if self.cfg.enable_z else 0.0)
        if self._order:
            p = self._order[-1]
            self._x[k] = self._extend(self._x[p] + self._delta(node, self._nodes[p]))
        else:
            self._x[k] = self._extend(node.dr_xy.copy())
        self._nodes[k] = node
        self._order.append(k)
        accepted = []
        for a in self._candidates(k):
            loop, why = self._verify(a, k)
            if loop is None:
                self.rejects[why] = self.rejects.get(why, 0) + 1
                continue
            self.loops.append(loop)
            # 覆盖统计：a、b 两个端点都算「被回环锚定」（回环边约束的是这两个位姿）。
            self._loop_anchored.add(int(a))
            self._loop_anchored.add(int(k))
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

    def _extend(self, v: np.ndarray) -> np.ndarray:
        """把 2 列向量补成 3 列（`enable_z=False` 的帧 z 恒 0）。"""
        v = np.asarray(v, float).ravel()
        if not self.cfg.enable_z:
            return v[:2].copy() if v.size >= 2 else np.zeros(2)
        out = np.zeros(3)
        out[:min(3, v.size)] = v[:3]
        return out
    
    def _delta(self, a: "_Node", b: "_Node") -> np.ndarray:
        """b − a 的航位推算增量（2 或 3 维）。"""
        d = np.asarray(a.dr_xy, float) - np.asarray(b.dr_xy, float)
        if not self.cfg.enable_z:
            return d[:2]
        return np.array([d[0], d[1], float(a.dr_z) - float(b.dr_z)])

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
            # ⚠️ 不能用 `np.hypot(*v)`：三维时它只接受 2 个参数会 TypeError。
            # 这里只关心**水平**候选半径（DR 漂移是水平量），所以显式取前两维。
            gap = float(np.linalg.norm((self._x[b] - self._x[a])[:2]))
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
        # ⚠️ 原先这里是 `(na.R @ t_ab)[:2]` —— PnP 的相对平移本来就有**完整三维**，
        # `[:2]` 把竖直分量丢掉了。这就是「z 没有观测」的真正原因：不是测不到，是被切片扔了。
        # 2026-10-28 起改由 `enable_z` 决定：默认仍只取 xy（保持既有行为与既有回环数），
        # 打开后 z 维进入位姿图。
        z3 = na.R @ t_ab
        z = z3 if self.cfg.enable_z else z3[:2]
        resid = float(np.linalg.norm((self._x[b] - self._x[a]) - z)) * s
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
        # D = 2（只解 xy，默认）或 3（z 也进图，`enable_z=True`）。
        # ⚠️ 这里不要写死 2：`.reshape(-1, 2)` 会在 z 维存在时**静默截断**，
        # 而 `_solve_laplacian` 是按 Z 的列数决定维数的 ⇒ 两者必须一致。
        D = 3 if c.enable_z else 2
        idx = {k: i for i, k in enumerate(self._order)}
        edges: list[tuple[int, int, np.ndarray, float]] = []
        for p, k in zip(self._order, self._order[1:]):
            path = max(0.0, self._nodes[k].dist_m - self._nodes[p].dist_m)
            sig = (c.odo_sigma0_m + c.odo_sigma_frac * path) / s
            edges.append((idx[p], idx[k], self._delta(self._nodes[k], self._nodes[p]), sig))
        old = {k: self._x[k].copy() for k in self._order}
        x0 = self._extend(self._nodes[self._order[0]].dr_xy)
        x = np.array([self._x[k] for k in self._order])
        oi = np.array([e[0] for e in edges], int)
        oj = np.array([e[1] for e in edges], int)
        oz = np.array([e[2] for e in edges], float).reshape(-1, D)
        ow = np.array([1.0 / (e[3] * e[3]) for e in edges], float)
        li = np.array([idx[L.a] for L in self.loops], int)
        lj = np.array([idx[L.b] for L in self.loops], int)
        lz = np.array([np.asarray(L.z, float).ravel()[:D].tolist() for L in self.loops],
                      float).reshape(-1, D)
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
            r = np.linalg.norm(x[lj] - x[li] - lz, axis=1)
            w_new = 1.0 / (1.0 + (r / (c.robust_k * lsig)) ** 2)
            changed = bool(np.any(np.abs(w_new - lw) > 1e-3))
            lw = w_new
            if not changed:
                break
        for L, w in zip(self.loops, lw):
            L.weight = float(w)
        for k, i in idx.items():
            self._x[k] = x[i]
        # ⚠️ 不能用 `np.hypot(*v)`：它只接受 2 个参数，三维时会 TypeError。
        # 用 `np.linalg.norm` 对任意维都对（D=2 时与 hypot 逐位同值）。
        return max(float(np.linalg.norm(self._x[k] - old[k])) for k in self._order)

    def coverage(self) -> dict[str, Any]:
        """回环锚定情况：哪些帧没被任何回环约束过（> 直接决定该段位姿可信度）。

        背景（2026-10-29）：`20261029_060258`（890 m / 1496 帧）末段 **423 m（48%）无回环**，
        而 `20261029_024332`（405 m）只有 **1 m**。差异巨大，却没有任何在线信号 ——
        下游（点云导出、导航）拿到一张图时无从判断哪段是纯航位推算。

        口径：关键帧「被锚定」⟺ 它出现在某条**接受**回环的 a 或 b 上（回环边约束两端位姿）。

        ⚠️ **主指标是 uncovered_m（末段未锚定路程），不是 anchored_frac。**
        一条回环只约束两个端点，所以实测 anchored_frac 只有 **4–5%**（060258 60/1496、
        024332 39/772）—— 这是正常的，不是缺陷。anchored_frac 只作「到底有没有回环」的
        粗略参考，别拿它当覆盖率。
        """
        c = self.cfg
        order = self._order
        if not order:
            return {"anchored_kf": 0, "anchored_frac": None,
                    "uncovered_m": None, "uncovered_frac": None,
                    "uncovered_since_kf": None}
        anchored = self._loop_anchored & set(order)
        total_m = max(0.0, self._nodes[order[-1]].dist_m - self._nodes[order[0]].dist_m)
        # 尾部无回环：从最后一个被锚定的关键帧量到末尾。一个都没锚定时 = 全片。
        if anchored:
            last = max(order.index(k) for k in anchored)   # 取**最后出现**的锚定帧
            tail_k = order[last]
            tail_m = max(0.0, self._nodes[order[-1]].dist_m - self._nodes[tail_k].dist_m)
        else:
            tail_k = order[0]
            tail_m = total_m
        return {
            "anchored_kf": int(len(anchored)),
            "anchored_frac": round(len(anchored) / len(order), 4),
            "uncovered_m": round(tail_m * c.world_scale, 1),
            "uncovered_frac": round(tail_m / total_m, 4) if total_m > 0 else None,
            "uncovered_since_kf": tail_k,
        }

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
                "correction_m": round(float(np.linalg.norm(self.offset()[:2])) * s, 3),
                "rejects": dict(self.rejects),
                # 回环覆盖（2026-10-29）：**末段/空洞**的量化。
                # uncovered_m / uncovered_frac：从「最后一个被锚定的关键帧」到当前末尾的路程。
                # 20261029_060258 实测 423 m / 48% —— 那段只有纯航位推算，漂移 5–19 m。
                # ⚠️ 它只量**尾部**。中段空洞要等轨迹走回来才看得见，所以另给
                # anchored_frac（全片被锚定帧占比）作为补充：两者一起看才完整。
                **self.coverage(),
                "yaw_checks": [{"b": b, "a": a, "yaw_err_deg": e, "accepted": ok}
                               for b, a, e, ok in list(self.yaw_checks)[-16:]]}
