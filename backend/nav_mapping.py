# -*- coding: utf-8 -*-
"""关键帧 → 三态栅格的**增量**建图（首访世界边走边建，navmesh 随之长大）。

与离线 ``.tmp/cloud_to_grid.py`` 是同一套判定，只是按关键帧累积：
* 每个关键帧存自己 base 系（x 前 y 左 z 上，追踪米）下的局部点，不存地图系坐标；
* 栅格化时用**当前**位姿把点投进地图系——回环后位姿变了，下一次栅格化自然跟着变，
  不会把旧位姿下的格子残留在图里；
* 高度相对**观测它的那个关键帧**：h = z_map − z_node + cam_h，头部俯仰、轨迹 z
  漂移都不进入判定（RTAB-Map 自带栅格在 base 系判高，run6 上 3 m 处障碍 39%）；
* 三态：free = 有地面点且非障碍；obstacle = 障碍点 ≥ min_pts 且 3×3 邻居支撑；
  其余 unknown。不做射线追踪，free 只来自真正看见的地面，unknown 永不当 free。
* **障碍判定按观测距离分层**（``q_tiers``，2026-10-05）：视差深度误差 δz = z²/(fx·b)·δd
  随距离平方增长，fx 202.5 / 基线 0.126 下 3 m 是 0.35 m、5 m 是 0.98 m，而障碍高度带
  (0.3, 2.0) 只有 1.7 m 宽——1.5 m 以外"真地面被抖进障碍带"是必然的，min_pts / 3×3 多数 /
  occ_ground_ratio 只压密度、补不回那一层不存在的信息（"空地中间冒孤岛"的物理来源）。
  所以近带单帧定案、中带要 ≥q_mid_kf 个关键帧一致（或"票够密且不含远带成分"的单帧例外）、
  远带没有定案权，只用来**否证**。实测走廊障碍 2168→502（044153），
  见 Docs/虚假障碍-根因与分层修复（2026-10-05）.md 与 tools/q_tier_ab.py。

栅格原点对齐到分辨率整数倍，所以地图变大时已有格子的编号含义不变。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .nav_grid import FREE, OCC, UNK, GridMeta, NavGrid


@dataclass
class MapperConfig:
    res_m: float = 0.10             # 追踪米；0.05 在 run6 上碎成很多小岛
    # 离观测关键帧的水平距离上限（追踪米）。相机平视、垂直视场 90°，地面要到 cam_h≈1.73 m 外才进画面，
    # 3 m 时每帧只有 1.3 m 深的一条带，要走遍才建得全。5 m（2026-09-30，run5/6/7 全因果回放）：
    # 已观测面积 ×1.5–1.8、可走面积 ×1.5–2.0，路径上障碍仍为 0，空地里的假障碍孤岛反被远处视线看穿清掉；
    # 基线 0.126 下 5 m 处视差约 5 px，地面高度误差约 0.1 m/像素，仍在 ground_tol 内。
    range_m: float = 5.0
    ground_tol_m: float = 0.30
    obst_top_m: float = 2.0
    min_pts: int = 3
    pad_m: float = 1.0
    cam_h_band: tuple[float, float] = (1.2, 2.6)   # 地面候选：相机下方这个范围
    cam_h_default: float = 1.73     # 候选太少时的回退值（run6 实测 1.728）
    world_scale: float = 0.755      # 由 OnlineNavConfig.world_scale 覆盖，别在这里改
    # 入图时按关键帧局部系体素化（带计数），栅格化只处理体素中心：run6 每帧约 1 万点 → 2.5 千体素。
    # xy 取栅格的一半，量化误差 ≤ 2.5 cm；z 2 cm，远小于 ground_tol。
    vox_xy_m: float = 0.05
    vox_z_m: float = 0.02
    # 增量栅格化：每个关键帧的格计数算一次，回环只按整格平移挪下标（见 _shift）。
    # cam_h 按 1 cm 直方图取中位，变化超过 cam_h_hyst_m 才换（换了要全部重分类）。
    #
    # ⚠️ 2026-10-05 调参：这两个值原来取错了量级，直接造成建图卡顿。cam_h 唯一的消费者是
    # 地面判定，而 ground_tol 就有 0.30 m —— **1~2 cm 的精度纯属浪费**，而 2 cm 的滞回带
    # 恰好卡死在直方图相邻两桶的抖动幅度上（0.02 的比较在浮点上还不严格）。实测 044153
    # 回放里 cam_h 在 1.745 / 1.765 之间来回翻了 9 次，每翻一次就是一次**全体关键帧重算**：
    # 末段单次 1.6 s，9 次合计占全程栅格化时间 31%（见 Docs/虚假障碍-根因与分层修复 §九）。
    # 滞回放宽到远大于桶噪声（6 cm），再按当前规模限流一次，末段重算从 1.6 s 降到不再发生。
    cam_h_hyst_m: float = 0.06
    # 单个关键帧的地面可能整体偏离全局 cam_h（低头/蹲下/HMD 高度抖、视差系统误差）：run6 有 5 帧偏 0.3–0.4 m，
    # 地面被整片判成障碍，正是"走过的地方变障碍、点了不动"的来源。按该帧 1.2 m 外地面的高度众数修正，
    # 限幅 ±ground_offset_max_m；众数不够突出（< 25% 点）就不修。
    ground_offset_max_m: float = 0.5
    ground_offset_min_range_m: float = 1.2
    # 常数偏移之外再拟合一个斜面：run6 另有 5 帧地面斜 12–20°（远处比近处高 0.3–0.4 m），常数修不掉，
    # 走过的路上 12 格假障碍里 8 格来自它们。坡度限幅 max_deg：拟合只取偏移众数 ±band 内的点，
    # 墙和箱子进不来；限幅防止少量点把平面掰到把墙根也吞进地面。
    ground_plane_max_deg: float = 15.0
    ground_plane_band_m: float = 0.12
    # 障碍点数还须 ≥ 地面点数 × 这个比例：单帧视差噪声打出的几个高点压不过几十帧看到的地面。
    occ_ground_ratio: float = 0.3   # run6：路径上的障碍格 25→12
    # ---- 观测质量分层（近/中/远）----
    # 视差深度误差 δz = z²/(fx·b)·δd。面板上的 fx 202.5、基线 0.126 ⇒ 1 像素视差 ≈ z²/25.5 m：
    # 1.5 m 处 9 cm、3 m 处 35 cm、5 m 处 98 cm。而障碍高度带 (ground_tol 0.3, obst_top 2.0)
    # 宽 1.7 m —— 1.5 m 以外"真地面"散进障碍带是必然的，不是阈值没调好，是信息压根不在那一层。
    # 所以扁平计数（min_pts / 3×3 多数 / occ_ground_ratio）只能压密度，补不回丢失的信息，
    # 这就是"空地中间冒孤岛"的来源。改成按观测距离分带定案：
    #   近带 <q_near_m：误差还在带边沿内，单帧即可定案，**故意不带**地面压制条件（否则 ratio 误杀真细障碍）；
    #   中带 q_near~q_mid：须 ≥q_mid_kf 个关键帧 + 地面压制，多帧一致性压住量化倾斜；
    #   远带 >q_mid：没有定案权，只有否证权——纯远场票的格降级成 FREE（有近/中距地面）或 UNK。
    # 口径与 Docs/离线多视角融合v1-假障碍清除（2026-10-02）.md 一致：该录制上障碍占比
    # 31.4%→13.3%（R1.5）/ 11.1%（R3.0），机理与 ray_clear 正交，两者叠加。
    q_tiers: bool = True
    q_near_m: float = 1.5
    q_mid_m: float = 3.0
    q_mid_kf: int = 2
    # 近带的独立票数门槛，**0 = 不启用**（默认，保持历史行为逐格不变）。调到 1~min_pts
    # 才会让"近距票够、但被地面压制挡掉"的格独立定案。
    #
    # ⚠️ 实测（tools/q_near_pts_ab.py，044153，两把尺子同时看）：**开着是净亏，别开。**
    #   q_near_pts  OCC    尺子1 走过∩OCC   尺子2 漏放
    #        0     4057        1.35%          202      ← 基线
    #        2     4320        1.51%          201      交换率 −6.1×
    #        1     4414        1.84%          201      交换率 −2.0×
    # 202 个漏放格只回来 1 个。原因是 **ray_clear 的看穿清零在 ``_by_quality`` 上游**：
    # tools/low_ceiling_nav 量到 1368/1405 的 n_o 在进质量分层之前就已经被置 0，
    # 票压根没走到这一层，降门槛降的是空气。
    # 保留这个开关是因为它让"近带不带地面压制"这句文档从死代码变成真代码（单测覆盖），
    # 而且**一旦上游 veto 修好，它就是现成的近距通道**。在那之前保持 0。
    # ─────────────────────────────────────────────────────────────────────────
    # ⚠️ 想改"真结构被误清掉"？**不要在这里调票数**。往这边看：
    #
    #   1. 症状一般在**上游**，不在这一层。``ray_clear`` 的看穿清零在 ``_by_quality``
    #      之前就把 n_o 削了（tools/low_ceiling_nav.py 在 044153 上量到 1368/1405 个
    #      有票的格 n_o 被置 0）。票没走到这一层，这里怎么调都是调空气。
    #   2. 先量，再改。跑 ``python -m tools.nav_audit <录制>``——它给两把尺子：
    #        尺子 1  走过∩OCC 率        抓**路径上**的虚假障碍
    #        尺子 2  近距平面证据的漏放  抓**路径外**的真结构被清掉
    #      **只报尺子 1 会把人往反方向推**：放宽阈值总能降低尺子 1，而尺子 1 对
    #      "真结构被清掉"权重为零。044153 上单看尺子 1 会得出"别动 ray_clear"的结论，
    #      实际情况是它一次清掉 14210 格、其中有 200 来格是坐在沙发后面的真矮墙。
    #   3. 🎯 **票数门和几何门都试过，都净亏**（见 ``ray_near_exempt`` 处的实测表）。
    #      真要继续，判据不是"回收几格 / 封死几格"的比值，而是
    #      ``tools/ray_exempt_conn.py`` 的连通性——比值数的是格数，代价取决于位置。
    #
    # 本开关保留只为让"近带不带地面压制"这句文档从死代码变成真代码（单测覆盖），
    # 实测净亏，**保持 0**：
    #        q_near_pts  OCC    尺子1        尺子2 漏放
    #             0     4057    1.35%           202      ← 基线
    #             2     4320    1.51%           201      回收 1 格
    #             1     4414    1.84%           201      回收 1 格
    # ─────────────────────────────────────────────────────────────────────────
    q_near_pts: int = 0
    # 单帧例外：一堵墙如果只被一个关键帧看到（刚走近就看见了），单帧多帧确认会把它误杀成 unknown。
    # 放宽的条件是"这张票干净"：格内障碍点够多（≥q_solo_pts）**且**远带票占比 ≤q_far_tol。
    # 实测（tools/q_tier_why.py，20261001_044153）：被清掉的格远带票占比中位数是 1.00、障碍点中位数 14；
    # 留住的格是 0.15 / 149。远带票占比是这个场景里最能分开"真结构"与"量化倾斜"的一维。
    q_solo_pts: int = 15
    q_far_tol: float = 0.15
    # 降级判回**空地**要多大范围内的地面证据。默认 = q_near_m，只认 1.5 m 内看清的地面；
    # 调到 q_mid_m（3.0）就是 Docs/离线多视角融合v1 那一版的近+中距口径：判回空地的格多得多，
    # 图外可探索面积涨得快，但中距地面本身就是被抖出来的，等于拿噪声治噪声。实测两者对
    # "走廊上还有几个障碍"几乎没差别（见 tools/q_tier_ab.json），差别只在图外那圈——所以
    # 默认取保守的那个，不伪造 free。
    q_free_m: float = 1.5
    # 射线清除：相机到每个观测点的视线穿过的障碍高度体素记一次"看穿"。一个体素被看穿的关键帧数
    # ≥ ray_beta × 被打中的关键帧数，就不再算障碍。没有它，障碍只增不减：定位误差、回环挪位每次
    # 都在旁边再画一份墙，旧的那份没有任何机制清掉（21 min 实测 150 块障碍里 94 块是空地中间的孤岛）。
    # 身体走过的中心线每格额外算 ray_walk_w 次看穿。run6：路径上假障碍 8→0，多帧障碍一格不丢。
    # β 2→1（2026-09-30）：21 min 录制（去掉静止重复帧后）路径上障碍格 195→23、可走 223→261 m²；
    # 代价是 run5 障碍格少 13%（665→581，路径上本来就是 0），run6 可走 +5 m²。
    ray_clear: bool = True
    ray_beta: float = 1.0
    ray_z_m: float = 0.10           # 障碍高度带 (ground_tol, obst_top) 的分层
    ray_step_m: float = 0.10        # 视线水平采样步长 = 一格；0.05 时 5 m 视距每帧多花 ~25 ms，结果几乎不变
    ray_stop_m: float = 0.20        # 离端点这么近就不算看穿（端点本身的深度噪声）
    ray_walk_w: float = 3.0
    # 近距票豁免看穿清零，**默认关，实测净亏**（几何门版，2026-10-04，044153 + 045615）。
    #
    # 🎯 **判据是 tools/ray_exempt_conn.py 的连通性，不是 tools/ray_exempt_ab.py 的交换比。**
    #   ab 报的是"回收的真结构格 / 新封死的路径格"，两段录制都 >1，看着像划算：
    #         044153   135/29 = 4.7:1        045615   47/33 = 1.4:1
    #   但连通性量下来两段都是亏的：
    #         044153   可达域 −1104 格、隔离出 1 块区域、4 对走过的路变得走不通、绕行 +13%
    #         045615   可达域 −954 格、隔离出 2 块区域、0 对断裂、绕行 +3%
    #   交换比与真实代价**反向**：赔率最好看的那段（4.7:1）损伤反而最重。因为代价不取决于
    #   格数，取决于那几格**卡在什么位置**——窄处一格就能掐断一条道，29 格可以是 3 条通路。
    #   ⚠️ 别拿 ab 的比值当定案依据，它只是"这一刀动了哪些格"的规模感。
    #
    # 门是"这里有没有一张连贯的近距水平面"（``_near_plane``），**不是**票数：按票数试过一次，
    # 045615 上回收 123 个真结构却封死 1155 个路径格，赔率 9:1。
    ray_near_exempt: bool = False
    # _near_plane 的门。三个变体在 044153 + 045615 上量过（tools/ray_exempt_ab.py，
    # 括号里是**票面**比值"回收的真结构格 / 新封死的路径格"——它不是判据，见上一条注释）：
    #   每格平均高度 + 邻居≥6   135/29 (4.7:1)   47/33  (1.4:1)   ← 三个里最不差，代码里留它
    #   众数层     + 邻居≥6      59/10 (5.9:1)   30/127 (0.2:1)   亏
    #   众数层     + 边缘感知      70/16 (4.4:1)   44/145 (0.3:1)   亏
    # ⚠️ "留它"≠"该打开"：三个变体**没有一个**在 tools/ray_exempt_conn.py 的连通性上翻正，
    #    开关保持 False。留着是因为它是几何门这个方向的现成实现（单测覆盖）。
    # 后两个都是"看见错杀就去修"，结果更差——flat 与 inside 是一对互相支撑的误杀集合。
    # **别再盲调这两个门**，细节见 ``_near_plane_parts`` 的注释。要继续先得第三段录制。
    ray_plane_pts: int = 20        # 原始点票数（噪声地板：1.5 m 外真地面散进障碍带是必然的）
    ray_plane_near: float = 0.5     # 近距票占障碍票的比例（1.5 m 处 δz 9 cm，地面抖不出 1 m）
    ray_plane_tol: float = 0.15     # 3×3 邻域每格平均高度极差（随机置换对照 16×/30× 信噪比）
    # 覆盖伴生网格（纯显示，不进三态判定）：观测点离该关键帧相机水平距离 ≤ cov_near_m 算"近看"。
    # 近看计数=0 的已观测格 = 只远看过的信息洞（走过去补扫）；近看点数再分稀疏/高质量（显示层阈值）。
    # 语义与上面 q_near_m/q_mid_m 无关（这里问"信息洞"，那里问"证据够不够定案"），改一个不会串到另一个。
    cov_near_m: float = 3.0

    # ---- 高度分带占用（多层 / 飞行；**旁路产物，不进地面层判定**）----
    # 现状：`obst_top_m = 2.0` 以上的点被**整段丢弃**。实测 044153 有 28.7% 的双目点落在头顶
    # 2 m 以上（2~3 m 占 7.0%、3~5 m 占 20.9%）——在 VRChat 里那正是二楼、阳台、天桥和
    # 下层的天花板。系统对多层建筑是**结构性失明**，不是精度不够。
    # 这里只做一件事：把被丢掉的高度**记下来**，按带存点计数。地面层的 `_acc_g`/`_acc_o`、
    # 质量分层规则、射线清除**一律不动**（导航行为零变化），带数据另开一条路给上层用。
    #
    # 带边界（h 的上界，追踪米）：(-inf, -ground_tol] / (obst_top_m, hi_band_m] /
    # (hi_band_m, hi_band_top_m] / (hi_band_top_m, +inf)。
    # h = rel_z + cam_h 是**离地高**而不是"离相机高"，所以跨关键帧可比，回环挪动时跟着关键帧走。
    #
    # 精度够用：墙面平面误差实测 0.06 m，avatar 半径 0.25 m，余量 4 倍。
    # 这次**不做**楼板/天花板识别（那要带内高度聚类，量过数据再定），只回答"这个高度有没有东西"。
    hi_bands: bool = True
    # 边界按 `tools/band_height_hist.py` 在 044153 上量到的自然峰设，不再是随手取的整数：
    # 头顶之上的点合并起来是 3.0~4.0 m 占 60.8%（壳厚约 1 m，见下），原来 3.5/6.0 的边界
    # **正好劈在那个面上**（lo 堆在自己上沿、mid 堆在自己下沿），两层都拿不到完整的一张面。
    hi_band_m: float = 3.0
    hi_band_top_m: float = 4.5


def _voxelize(p: np.ndarray, xy_m: float, z_m: float) -> tuple[np.ndarray, np.ndarray]:
    """N×3 → (M×3 体素中心, M 点数)。键打包成 int64 再一维 unique，比 unique(axis=0) 快一个量级。"""
    if not len(p):
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
    q = np.floor(p / np.array([xy_m, xy_m, z_m], np.float32)).astype(np.int64)
    q = np.clip(q, -(1 << 20), (1 << 20) - 1) + (1 << 20)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    u, cnt = np.unique(key, return_counts=True)
    m = (1 << 21) - 1
    idx = np.column_stack([(u >> 42) & m, (u >> 21) & m, u & m]) - (1 << 20)
    centers = (idx.astype(np.float32) + 0.5) * np.array([xy_m, xy_m, z_m], np.float32)
    return centers, cnt.astype(np.int32)


def _ground_offset(h: np.ndarray, cnt: np.ndarray, rxy: np.ndarray, lim: float, min_r: float) -> float:
    """关键帧地面相对全局 cam_h 的偏移（追踪米）：±lim 内 2 cm 直方图、10 cm 平滑后的众数。"""
    sel = (np.abs(h) <= lim + 0.1) & (np.hypot(rxy[:, 0], rxy[:, 1]) >= min_r)
    tot = float(cnt[sel].sum())
    if tot < 300:
        return 0.0
    n = int((2 * lim + 0.2) / 0.02) + 1
    b = np.clip(((h[sel] + lim + 0.1) / 0.02).astype(np.int64), 0, n - 1)
    sm = np.convolve(np.bincount(b, cnt[sel], minlength=n), np.ones(5), "same")
    i = int(np.argmax(sm))
    if sm[i] < 0.25 * tot:
        return 0.0
    return float(np.clip((i + 0.5) * 0.02 - lim - 0.1, -lim, lim))


def _ground_correction(h: np.ndarray, cnt: np.ndarray, rxy: np.ndarray, lim: float, min_r: float,
                       max_deg: float, band: float) -> np.ndarray | float:
    """关键帧地面的逐点高度修正（追踪米）：先取常数偏移众数，再在它 ±band 内加权拟合 h = a + b·x + c·y，
    两轮剔除残差 > 6 cm 的点。点不够就只用常数偏移；坡度超过 max_deg 按比例压回；总修正限幅 ±lim。"""
    return _ground_model(h, cnt, rxy, lim, min_r, max_deg, band)[0]


def _ground_model(h: np.ndarray, cnt: np.ndarray, rxy: np.ndarray, lim: float, min_r: float,
                  max_deg: float, band: float) -> tuple[np.ndarray | float, float]:
    """(逐点修正, 关键帧正下方的修正)。后者给射线清除定相机离地高度。"""
    off = _ground_offset(h, cnt, rxy, lim, min_r)
    if max_deg <= 0.0:
        return off, off
    sel = np.flatnonzero((np.abs(h - off) <= band) & (np.hypot(rxy[:, 0], rxy[:, 1]) >= min_r))
    if float(cnt[sel].sum()) < 300:
        return off, off
    sol = None
    for _ in range(3):
        if len(sel) < 50:
            return off, off
        A = np.column_stack([np.ones(len(sel)), rxy[sel].astype(np.float64)])
        w = np.sqrt(cnt[sel].astype(np.float64))
        sol = np.linalg.lstsq(A * w[:, None], (h[sel] - off) * w, rcond=None)[0]
        sel = sel[np.abs(h[sel] - off - A @ sol) <= 0.06]
    grad = math.hypot(sol[1], sol[2])
    cap = math.tan(math.radians(max_deg))
    if grad > cap:
        sol[1:] *= cap / grad
    corr = off + sol[0] + sol[1] * rxy[:, 0] + sol[2] * rxy[:, 1]
    return np.clip(corr, -lim, lim).astype(np.float32), float(np.clip(off + sol[0], -lim, lim))


def _ray_voxels(rxy: np.ndarray, h: np.ndarray, t: np.ndarray, cam_h: float, cfg: "MapperConfig"
                ) -> tuple[np.ndarray, np.ndarray]:
    """一个关键帧的 (打中, 看穿) 障碍高度体素，各为 K×3 (ix, iy, iz)，地图系按平移 t 定格，iz 从 ground_tol 起。

    看穿 = 相机（离地 cam_h）到每个端点的视线在障碍高度带内经过、且离端点 > ray_stop_m 的体素，
    扣掉本帧自己打中的。端点先按 10 cm 去重：同一方向的一簇点只追一条线。"""
    res, tol, top, zb = cfg.res_m, cfg.ground_tol_m, cfg.obst_top_m, cfg.ray_z_m
    empty = np.zeros((0, 3), np.int64)

    def cells(xy: np.ndarray, hh: np.ndarray) -> np.ndarray:
        # 一次分配写三列。column_stack + astype(int64) 会多出两次整块拷贝，
        # 这里一行就要处理 6 万多个采样点（见下面 miss 那行的耗时对比）。
        out = np.empty((len(xy), 3), np.int64)
        out[:, 0] = np.floor((xy[:, 0] + t[0]) / res)
        out[:, 1] = np.floor((xy[:, 1] + t[1]) / res)
        out[:, 2] = np.floor((hh - tol) / zb)
        return out

    def unique_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按 (x, y, z) 去重，返回 (每个唯一键的代表行, **全部**输入行的打包键)。

        用稠密标记数组代替 ``np.unique`` 的排序：射线体素全在关键帧 ``range_m`` 半径内，
        打包键的跨度约 100×100×17，一个几 MB 的 int32 标记数组就装得下，而排序要动
        每关键帧 7 万多个 int64。这是 ``_sync_ray`` 的主要开销（18.6 ms/关键帧里绝大部分）。
        倒序赋值 ⇒ 重复键留下**最小**下标，与 ``np.unique(return_index=True)`` 的
        "first occurrence" 语义一致（``RayClearTests`` 依赖这个确定性）。
        """
        lo = v.min(axis=0)
        span = v.max(axis=0) - lo + 1
        key = ((v[:, 0] - lo[0]) * span[1] + (v[:, 1] - lo[1])) * span[2] + (v[:, 2] - lo[2])
        size = int(span[0]) * int(span[1]) * int(span[2])
        if size <= (1 << 22):                       # 4M 格（16 MB int32）以内走稠密标记
            # 存下标+1：下标 0 与"这一格没被写过"在 0 初始化数组里长得一模一样，
            # 不加哨兵的话每个结果集都会漏掉第 0 行（300 组随机输入里 300 组不一致，
            # 当场抓住的，不是想出来的）。
            pos = np.zeros(size, np.int32)
            pos[key[::-1]] = np.arange(len(key), 0, -1, dtype=np.int32)
            return v[pos[np.flatnonzero(pos)] - 1], key
        _u, first = np.unique(key, return_index=True)
        return v[first], key

    o = (h > tol) & (h < top)
    hits = unique_rows(cells(rxy[o].astype(np.float64), h[o].astype(np.float64)))[0] if o.any() else empty
    sel = (h >= -tol) & (h < top)
    if not sel.any():
        return hits, empty
    exy, eh = rxy[sel].astype(np.float64), h[sel].astype(np.float64)
    q = np.column_stack([np.floor(exy / 0.1), np.floor(eh / 0.1)]).astype(np.int64)
    first = np.unique(unique_rows(q)[1], return_index=True)[1]
    exy, eh = exy[first], eh[first]
    r = np.hypot(exy[:, 0], exy[:, 1])
    n = np.maximum(0, np.floor((r - cfg.ray_stop_m) / cfg.ray_step_m).astype(np.int64))
    if not n.sum():
        return hits, empty
    idx = np.repeat(np.arange(len(r)), n)
    j = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n) + 1
    f = j * cfg.ray_step_m / r[idx]
    sh = cam_h + f * (eh[idx] - cam_h)
    ok = (sh > tol) & (sh < top)
    if not ok.any():
        return hits, empty
    # exy[idx[ok]] 先取一次再乘 f：写成 cells(exy[idx[ok]] * f[ok, None], ...) 时
    # 两列各自 gather 一遍同样的 6 万个下标，实测这一行就占 10.2 ms（_ray_voxels 的 77%）。
    xy = exy[idx[ok]] * f[ok, None]
    miss = unique_rows(cells(xy, sh[ok]))[0]
    if len(hits):
        both = np.vstack([hits, miss])
        key = unique_rows(both)[1]
        miss = miss[~np.isin(key[len(hits):], key[:len(hits)])]
    return hits, miss


def disparity_range_px(fx: float, *, base_fx: float = 202.5, base_px: int = 64) -> int:
    """按 fx 缩放的视差搜索范围（取 16 的倍数）。

    ``d = fx·B/Z`` ⇒ 同一深度下视差与 fx 成正比。换采集宽度（720→1440→2880，
    fx 202.5→405→810）时范围不跟着缩放，近场会被裁掉：64@fx202.5 覆盖到 Z≈0.40 m，
    仍用 64 的话边界退到 ≈0.80 / 1.60 m。默认参数下返回 64 —— 720×405 行为逐位不变。
    见 `Docs/全分辨率验证录制SOP（2026-10-05）.md`。
    """
    return max(16, int(16 * round(base_px * float(fx) / base_fx / 16.0)))


def make_sgbm(num_disparities: int = 64) -> Any:
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=int(num_disparities), blockSize=5,
                                 P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def stereo_disparity(left_gray: np.ndarray, right_gray: np.ndarray, matcher: Any = None) -> np.ndarray:
    """左目视差（像素，float32）；无效处 ≤ 0。"""
    m = matcher if matcher is not None else make_sgbm()
    return m.compute(left_gray, right_gray).astype(np.float32) / 16.0


def stereo_points(left_gray: np.ndarray, right_gray: np.ndarray, *, fx: float, cx: float,
                  cy: float, baseline_m: float, max_range_m: float = 3.5,
                  step: int = 2, matcher: Any = None, disp: np.ndarray | None = None) -> np.ndarray:
    """校正好的双目 → base 系点（N×3，追踪米）。光学系 z 前 x 右 y 下 → base x 前 y 左 z 上。
    ``disp`` 给了就不再算一遍视差（回环检测要同一张视差图）。"""
    if disp is None:
        disp = stereo_disparity(left_gray, right_gray, matcher)
    d = disp[::step, ::step]
    v, u = np.mgrid[0:disp.shape[0]:step, 0:disp.shape[1]:step]
    ok = d > 1.0
    z = fx * baseline_m / d[ok]
    keep = z <= max_range_m
    z = z[keep]
    x = (u[ok][keep] - cx) * z / fx
    y = (v[ok][keep] - cy) * z / fx
    return np.column_stack([z, -x, -y]).astype(np.float32)


# ``_kf_base`` 返回元组的下标 → 累加器属性名。``_sync_acc``（写入）与 ``_grow_acc``（搬迁）都按
# 这张表驱动：加一个分带只要在 _kf_base 里多返回一段、这里多一行，不必改两处加法。
_ACC_SLOTS: tuple[tuple[int, str], ...] = (
    (2, "_acc_g"), (3, "_acc_o"), (6, "_acc_near"), (7, "_acc_far"),
    (8, "_acc_on"), (9, "_acc_om"), (10, "_acc_gn"), (11, "_acc_omk"),
    (12, "_acc_b0"), (13, "_acc_b1"), (14, "_acc_b2"), (15, "_acc_b3"),
    (16, "_acc_sb_n"), (17, "_acc_sb_h"), (18, "_acc_sb_h2"),
    (19, "_acc_ob_h"), (20, "_acc_ob_h2"),
)
_ACC_NAMES: tuple[str, ...] = tuple(name for _slot, name in _ACC_SLOTS)


class KeyframeGridMapper:
    def __init__(self, cfg: MapperConfig | None = None) -> None:
        self.cfg = cfg or MapperConfig()
        self._pts: dict[int, np.ndarray] = {}       # 体素中心（关键帧 base 系，追踪米）
        self._cnt: dict[int, np.ndarray] = {}       # 每个体素里的原始点数
        self._pose: dict[int, np.ndarray] = {}
        self._osc: dict[int, float] = {}
        self._trail: dict[int, list[tuple[np.ndarray, np.ndarray, float | None]]] = {}
        self._trail_arr: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cam_h = self.cfg.cam_h_default
        self._cam_h_at = 0            # 上次改 cam_h 时的关键帧数（cam_h_min_kf 限流用）
        # 缓存：_rot[k] = (R·p 的 xy, R·p 的 z = 相对关键帧的高度, 点数, 算它用的 R)；
        # 只改平移时高度不变，分类也不变。
        # _base[k] = (ix, iy, n_ground, n_obst, 算它用的 t_xy, cam_h, n_near, n_far)；
        # 近/远看按点到关键帧相机的水平距离分桶（相机在 base 系原点，与平移无关），整格平移照常复用。
        self._rot: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._base: dict[int, tuple[Any, ...]] = {}
        self._hist: dict[int, np.ndarray] = {}
        lo, hi = self.cfg.cam_h_band
        self._hist_sum = np.zeros(int(round((hi - lo) / 0.01)), np.int64)
        # 全局格计数累加器（[ix − lo_x, iy − lo_y]）。每个关键帧记下它当前加进去的 (base, 整格平移)，
        # 变了就先减旧的再加新的：新关键帧只算自己，回环只挪整格，不重投点。
        self._acc_g: np.ndarray | None = None
        self._acc_o: np.ndarray | None = None
        self._acc_near: np.ndarray | None = None    # 近看（≤ cov_near_m）点数，与 _acc_g 同布局同记账
        self._acc_far: np.ndarray | None = None     # 远看点数；near+far>0 = 已观测格（覆盖原始数据）
        # 质量分层专用：障碍点按观测距离分带的票、近距地面证据、中距障碍的关键帧票数。
        # 与 _acc_g/_acc_o 同布局同记账（同样只在 (base, 整格平移) 变了时才动）。
        self._acc_on: np.ndarray | None = None      # 障碍点，观测距离 < q_near_m
        self._acc_om: np.ndarray | None = None      # 障碍点，q_near_m ≤ 距离 < q_mid_m
        self._acc_gn: np.ndarray | None = None      # 地面点，距离 < q_free_m（判回空地要的"这格是地板"证据）
        self._acc_omk: np.ndarray | None = None     # 有中距障碍票的关键帧数（本关键帧最多 +1）
        # 高度分带（旁路，见 MapperConfig.hi_bands）：b0 地面以下 / b1 头顶 2~hi_band_m /
        # b2 hi_band_m~hi_band_top_m / b3 再往上。**不进地面层判定**，只回答"这个高度有没有东西"。
        self._acc_b0: np.ndarray | None = None
        self._acc_b1: np.ndarray | None = None
        self._acc_b2: np.ndarray | None = None
        self._acc_b3: np.ndarray | None = None
        # 头顶 obst_top_m 之上的三个**可加矩**（旁路，见 surface_counts）：n / Σw·h / Σw·h²。
        # 众数不可加，矩可加——回环挪位时才能精确回退。
        self._acc_sb_n: np.ndarray | None = None
        self._acc_sb_h: np.ndarray | None = None
        self._acc_sb_h2: np.ndarray | None = None
        # 障碍带（ground_tol, obst_top）内每格的 Σw·h / Σw·h²。给"这里有没有一张连贯的
        # 水平面"用（见 ``_near_plane`` 与 ``ray_near_exempt``）。**不是** n_o 的副本：n_o 会被
        # 看穿清零削掉，而这里要的恰恰是那些被清零之后**本该留住**的格。
        self._acc_ob_h: np.ndarray | None = None
        self._acc_ob_h2: np.ndarray | None = None
        self._acc_lo = (0, 0)
        # 最近一次 rasterize 用的栅格↔累加器格偏移 (lx, ly)：栅格列 c ↔ 累加器 x = lx + c，
        # 栅格行 r ↔ 累加器 y = ly + (h-1-r)（栅格行 0 是最大 y）。离屏分析/工具要靠它把
        # 累加器里的分带票映射回栅格坐标，别在别处重推一遍 lo_i——推错一次整张分析就静默错位。
        self._grid_lo: tuple[int, int] | None = None
        self._last_grid: tuple[int, int] | None = None   # 上次 rasterize 的 (h, w)，供 band_grid 对齐
        self._acc_cam_h: float | None = None
        self._applied: dict[int, tuple[tuple, int, int]] = {}
        # 上次 rasterize 之后新增或换过位姿的关键帧；只有它们要重查缓存（1200 帧时逐帧查一遍要十几 ms）。
        self._dirty: set[int] = set()
        self._walked_cache: tuple[int, list[np.ndarray]] | None = None
        self._ver = 0                            # 关键帧/位姿/轨迹任一变了就 +1，walked() 按它缓存
        # 射线清除：[iz, iy − lo_y, ix − lo_x] 的打中/看穿关键帧数，与 _acc_* 同原点同大小。
        # _ray[k] = [算它用的 _rot 条目, t0, t0 下的格包围盒, 打中, 看穿, 编码布局]：只依赖朝向（HMD 给的，回环不改），
        # 每个关键帧一辈子只追一次线；cam_h 变了也不重追——逐帧地面修正已吸收掉它。
        self._ray: dict[int, list] = {}
        self._ray_applied: dict[int, tuple[tuple, int, int]] = {}
        self._ray_hit: np.ndarray | None = None
        self._ray_mis: np.ndarray | None = None
        self.ray_cleared_cells = 0

    def __len__(self) -> int:
        return len(self._pts)

    def add_keyframe(self, node_id: int, points_base: np.ndarray, pose_map_base: np.ndarray,
                     osc_dist_m: float | None = None) -> None:
        """``osc_dist_m``：该关键帧时刻的 OSC 累计路程（世界米）；给了才能门控走过的走廊。"""
        k = int(node_id)
        if k in self._pts:
            # 与 LoopCloser.add_keyframe 同口径的廉价防御：重复 id 会静默覆盖点云与位姿。
            # drop_points(k) 之后再 add_keyframe(k) 不算重复（点云已摘掉），refresh 走的正是这条路。
            raise ValueError("关键帧 id 必须递增")
        p = np.asarray(points_base, np.float32).reshape(-1, 3)
        p = p[np.hypot(p[:, 0], p[:, 1]) <= self.cfg.range_m]
        self._pts[k], self._cnt[k] = _voxelize(p, self.cfg.vox_xy_m, self.cfg.vox_z_m)
        self._pose[k] = np.array(pose_map_base, np.float64).reshape(4, 4)
        self._drop_cache(k)
        self._dirty.add(k)
        self._ver += 1
        if osc_dist_m is not None:
            self._osc[k] = float(osc_dist_m)

    def drop_points(self, node_id: int) -> None:
        """去掉关键帧的点云贡献，保留位姿、OSC 路程和挂在它上面的轨迹：目标锚点照样能 resolve，
        走过的折线不变。用于原地不动时新帧替换上一帧（同一视角重复叠加只会放大深度噪声）。"""
        k = int(node_id)
        if self._pts.pop(k, None) is None:
            return
        self._cnt.pop(k, None)
        # 把 cam_h 限流计数器夹回当前规模。背景（核查 issue #3-5 时实测澄清）：
        #   * 在线路径（nav_online.py:920-927）总是先 add_keyframe(k) 再 drop_points(k-1)，
        #     len(_pts) 单调不减，_cam_h_at <= len(_pts) 恒成立 ⇒ **线上不可达**，这里是 no-op；
        #   * 离线回放（tools/q_tier_ab.py:87-92）是「先摘 k-1、可能再摘 k、最后加 k」，
        #     len 反而会下降，此时 _cam_h_at 会越过 len(_pts)；
        #   * 越过之后的后果与外界报告**相反**：d = len - _cam_h_at 为负 ⇒ 恒 < 阈值 ⇒
        #     走 :478 的 return，cam_h **冻结不再更新**，而不是"更频繁地重算"。冻结会一直
        #     持续到 len 重新爬回 _cam_h_at + len//8，比设计值多等 (_cam_h_at - len) 帧。
        # 夹回之后等待帧数恢复成设计值。不改变 cam_h 的最终取值，只改变到达它的时机。
        self._cam_h_at = min(self._cam_h_at, len(self._pts))
        self._drop_cache(k)
        self._dirty.discard(k)
        self._ver += 1

    def _drop_cache(self, k: int) -> None:
        self._rot.pop(k, None)
        self._base.pop(k, None)
        h = self._hist.pop(k, None)
        if h is not None:
            self._hist_sum -= h

    def _rotated(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = self._pose[k][:3, :3]
        c = self._rot.get(k)
        if c is not None and (c[3] is R or np.array_equal(c[3], R)):
            return c
        self._drop_cache(k)
        q = self._pts[k] @ R.T.astype(np.float32)
        c = (q[:, :2].copy(), q[:, 2].copy(), self._cnt[k], R.copy())
        self._rot[k] = c
        lo, hi = self.cfg.cam_h_band
        band = (c[1] < -lo) & (c[1] > -hi)
        b = np.clip(((-c[1][band] - lo) / 0.01).astype(np.int64), 0, len(self._hist_sum) - 1)
        h = np.bincount(b, c[2][band], minlength=len(self._hist_sum)).astype(np.int64)
        self._hist[k] = h
        self._hist_sum += h
        return c

    def _update_cam_h(self) -> None:
        """按全带内点云的加权中位更新 ``cam_h``。有滞回 + 限流，见 ``MapperConfig`` 里的说明。"""
        tot = int(self._hist_sum.sum())
        if tot < 200:
            return
        # 先平滑再取中位：`_ground_offset` 用的就是同一套 ±2 桶平滑，这里保持一致。
        # 不平滑的话加权中位会在直方图相邻两桶之间跳，滞回带再窄也压不住。
        # 分位必须用**平滑后**自己的总量：5 桶卷积把总和放大了约 5 倍，拿原始 tot/2 去搜
        # 等于在采 10% 分位，中位会一路往下漂（踩过：1.756 → 1.465，还顺带改坏了地面判定）。
        sm = np.convolve(self._hist_sum.astype(np.float64), np.ones(5), "same")
        cw = np.cumsum(sm)
        if cw[-1] <= 0.0:
            return
        est = self.cfg.cam_h_band[0] + (int(np.searchsorted(cw, cw[-1] / 2.0)) + 0.5) * 0.01
        c = self.cfg
        if abs(est - self.cam_h) <= c.cam_h_hyst_m:
            return
        # 限流：cam_h 一变就要把**所有**关键帧重算一遍，代价随关键帧数线性涨。估计算得再稳也
        # 架不住在阈值附近反复触发，所以按"距上次变更至少过了当前规模的 1/8"来限：
        # 早期（几帧）几乎立刻就能跟上一个正确值，晚期每次重算之间隔得越来越开。
        # 这样重建次数是 O(log N)、总代价 O(N)，而不是原来那种线性翻转让单次卡到 1.6 s。
        # max(0, ...)：drop_points 已把 _cam_h_at 夹回当前规模，这里再加一层防御。
        # 注意方向：d 为负时是**不更新**（冻结 cam_h），不是"放行重算"——别照着
        # "负值 ⇒ 限流失效"的直觉改这里。
        if len(self._pts) - max(0, self._cam_h_at) < max(1, len(self._pts) // 8):
            return
        self.cam_h = est
        self._cam_h_at = len(self._pts)

    def _kf_base(self, k: int) -> tuple[np.ndarray, ...]:
        """关键帧 k 在"算它那一刻的平移 t0"下的格计数
        (ix, iy, n_ground, n_obst, t0, cam_h, n_near, n_far, o_near, o_mid, g_near_mid, o_mid_kf,
         b_below, b_lo, b_mid, b_hi, s_n, s_h, s_h2)。

        朝向或 cam_h 变了才重算；只有平移变了就按整格平移复用（见 ``_shift``）。
        n_near/n_far：近/远看（观测距离 ≤ cov_near_m）的点数，与 n_ground/n_obst 同格同权重，纯显示用。
        o_near/o_mid/g_near_mid/o_mid_kf：质量分层的票——障碍点按观测距离落带、近中距地面证据、
        以及"有多少个关键帧给这个格投过中距障碍票"（每帧最多 +1，减帧精确回退）。
        b_*：高度分带的点计数（旁路，见 ``MapperConfig.hi_bands``）；s_n/s_h/s_h2 是头顶
        ``obst_top_m`` 之上的三个可加矩（见 ``surface_counts``）。**注意** ``n_ground``/
        ``n_obst``/``n_near``/``n_far`` 的口径完全没变——分带与矩只是把原本被丢掉的高度捡回来。"""
        c = self.cfg
        cached = self._base.get(k)
        if cached is not None and cached[5] == self.cam_h:
            return cached
        t = self._pose[k][:2, 3].copy()
        rxy, rel, cnt, _R = self._rot[k]
        h = rel + np.float32(self.cam_h)
        h = h - np.float32(_ground_correction(h, cnt, rxy, c.ground_offset_max_m, c.ground_offset_min_range_m,
                                              c.ground_plane_max_deg, c.ground_plane_band_m))
        g = np.abs(h) <= c.ground_tol_m
        o = (h > c.ground_tol_m) & (h < c.obst_top_m)
        # 高度分带：把地面层判据之外的高度捡回来（-1 = 不属于任何带）。
        if c.hi_bands:
            hb = np.full(len(h), -1, np.int8)
            hb[h <= -c.ground_tol_m] = 0
            hb[h >= c.obst_top_m] = 1
            hb[h >= c.hi_band_m] = 2
            hb[h >= c.hi_band_top_m] = 3
        else:
            hb = np.full(len(h), -2, np.int8)
        sel = g | o | (hb >= 0)
        if not sel.any():
            z = np.zeros(0, np.int64)
            out = (z, z, z, z, t, self.cam_h, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z)
            self._base[k] = out
            return out
        xy = rxy[sel].astype(np.float64) + t
        ix = np.floor(xy[:, 0] / c.res_m).astype(np.int64)
        iy = np.floor(xy[:, 1] / c.res_m).astype(np.int64)
        # 关键帧只覆盖 range_m 半径的小方块：在局部稠密方块里 bincount 去重，比 np.unique（排序）快一个量级。
        x0, y0 = int(ix.min()), int(iy.min())
        bw = int(iy.max()) - y0 + 1
        flat = (ix - x0) * bw + (iy - y0)
        size = (int(ix.max()) - x0 + 1) * bw
        w = cnt[sel]
        sg, so, shb = g[sel], o[sel], hb[sel]
        hh = h[sel].astype(np.float64)
        # sel 变长了（多了高带点），但下面每个掩码都只在自己的集合里为真，
        # 所以 ng/no/nn/nf 的口径与改动前**逐格相同**。
        ng = np.bincount(flat, w * sg, minlength=size)
        no = np.bincount(flat, w * so, minlength=size)
        # 覆盖伴生：点到相机（base 系原点）的水平距离分近/远两桶，复用同一 flat 与权重。
        # 口径仍是**地面层那两类点**（g|o），不带高带点——覆盖度显示层的语义不动。
        r = np.hypot(rxy[sel, 0], rxy[sel, 1])
        go = sg | so
        near = r <= c.cov_near_m
        nn = np.bincount(flat, w * (go & near), minlength=size)
        nf = np.bincount(flat, w * (go & ~near), minlength=size)
        # 质量分层：同一批点再按 q_near_m/q_mid_m 落带。分带只动竖直方向的判据，xy/flat/权重全复用。
        if c.q_tiers:
            bn = r < c.q_near_m
            bm = ~bn & (r < c.q_mid_m)
            on = np.bincount(flat, w * (so & bn), minlength=size)
            om = np.bincount(flat, w * (so & bm), minlength=size)
            # 判回空地要的地面证据带（q_free_m，默认近带；调成 q_mid_m 就是离线那一版的近+中距）。
            gn = np.bincount(flat, w * (sg & (r < c.q_free_m)), minlength=size)
            omk = (om > 0).astype(np.float64)      # 本关键帧对每个格最多投一票，不是点数
        else:
            zf = np.zeros(size)
            on = om = gn = omk = zf
        if c.hi_bands:
            bands = tuple(np.bincount(flat, w * (shb == b), minlength=size) for b in range(4))
            # 头顶之上的面：三个可加矩。shb>=1 恰好就是 h>obst_top_m（band 1 的下界），与带边界无关。
            shb_hi = shb >= 1
            sbn = np.bincount(flat, w * shb_hi, minlength=size)
            sbh = np.bincount(flat, w * shb_hi * hh, minlength=size)
            sbh2 = np.bincount(flat, w * shb_hi * hh * hh, minlength=size)
        else:
            zf = np.zeros(size)
            bands = (zf, zf, zf, zf)
            sbn = sbh = sbh2 = zf
        # 障碍带内的两个矩。与头顶矩同一批点、同一套 bincount，只是掩码换成 so。
        # **无条件算**（不进 hi_bands 分支）：看穿豁免要的恰恰是被清零之后仍然存在的那部分证据。
        obh = np.bincount(flat, w * so * hh, minlength=size)
        obh2 = np.bincount(flat, w * so * hh * hh, minlength=size)
        nz = np.flatnonzero((ng > 0) | (no > 0)
                            | (bands[0] > 0) | (bands[1] > 0) | (bands[2] > 0) | (bands[3] > 0)
                            | (sbn > 0))
        out = (nz // bw + x0, nz % bw + y0, ng[nz], no[nz], t, self.cam_h, nn[nz], nf[nz],
               on[nz], om[nz], gn[nz], omk[nz],
               bands[0][nz], bands[1][nz], bands[2][nz], bands[3][nz],
               sbn[nz], sbh[nz], sbh2[nz],
               obh[nz], obh2[nz])
        self._base[k] = out
        return out

    def _shift(self, k: int, base: tuple) -> tuple[int, int]:
        """当前平移相对 base 的 t0 挪了多少整格。整格平移与逐点重投最多差一格，而逐点重投本身
        也要量化到格；回环（853 帧全体平移）从逐帧重投 ~1 s 变成只挪下标。"""
        d = (self._pose[k][:2, 3] - base[4]) / self.cfg.res_m
        return int(round(float(d[0]))), int(round(float(d[1])))

    def _sync_acc(self, keys: set[int]) -> set[int]:
        """全局累加器只处理 (base, 整格平移) 变了的关键帧：旧贡献负权、新贡献正权，一次 bincount。
        ``keys``：要重查的关键帧；cam_h 变了会清空重建，此时改查全部。返回实际查过的集合（射线同用）。"""
        if self._acc_cam_h != self.cam_h:
            for name in _ACC_NAMES:
                setattr(self, name, None)
            self._ray_hit = self._ray_mis = None
            self._applied.clear()
            self._ray_applied.clear()
            self._acc_cam_h = self.cam_h
            keys = set(self._pts)
        work: list[tuple[tuple, int, int, float]] = []
        for k in keys:
            base = self._kf_base(k)
            sx, sy = self._shift(k, base)
            prev = self._applied.get(k)
            if prev is not None and prev[0] is base and prev[1] == sx and prev[2] == sy:
                continue
            if prev is not None and len(prev[0][0]):
                work.append((*prev, -1.0))
            if len(base[0]):
                work.append((base, sx, sy, 1.0))
            self._applied[k] = (base, sx, sy)
        for k in [k for k in self._applied if k not in self._pts]:
            prev = self._applied.pop(k)
            if len(prev[0][0]):
                work.append((*prev, -1.0))
        if not work:
            return keys
        ix = np.concatenate([b[0] + sx for b, sx, _sy, _g in work])
        iy = np.concatenate([b[1] + sy for b, _sx, sy, _g in work])
        self._grow_acc(int(ix.min()), int(iy.min()), int(ix.max()), int(iy.max()))
        (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
        flat = (iy - y0) * W + (ix - x0)
        for slot, name in _ACC_SLOTS:
            acc = getattr(self, name)
            w = np.concatenate([b[slot] * sg for b, _sx, _sy, sg in work])
            acc += np.bincount(flat, w, minlength=H * W).reshape(H, W)
        return keys

    def _ray_z(self) -> int:
        c = self.cfg
        return int(math.ceil((c.obst_top_m - c.ground_tol_m) / c.ray_z_m))

    def _grow_acc(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        m = 64                                   # 每次多留 6.4 m，边走边长不必每帧重分配
        if self._acc_g is None:
            x0, y0 = xmin - m, ymin - m
            shape = (ymax + 1 + m - y0, xmax + 1 + m - x0)
            for name in _ACC_NAMES:
                setattr(self, name, np.zeros(shape))
            self._acc_lo = (x0, y0)
            if self.cfg.ray_clear:
                self._ray_hit = np.zeros((self._ray_z(), *shape), np.int32)
                self._ray_mis = np.zeros((self._ray_z(), *shape), np.int32)
            return
        (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
        if xmin >= x0 and ymin >= y0 and xmax < x0 + W and ymax < y0 + H:
            return
        nx0 = xmin - m if xmin < x0 else x0
        ny0 = ymin - m if ymin < y0 else y0
        nx1 = xmax + 1 + m if xmax >= x0 + W else x0 + W
        ny1 = ymax + 1 + m if ymax >= y0 + H else y0 + H
        for name in _ACC_NAMES + ("_ray_hit", "_ray_mis"):
            old = getattr(self, name)
            if old is None:
                continue
            new = np.zeros((*old.shape[:-2], ny1 - ny0, nx1 - nx0), old.dtype)
            new[..., y0 - ny0:y0 - ny0 + H, x0 - nx0:x0 - nx0 + W] = old
            setattr(self, name, new)
        self._acc_lo = (nx0, ny0)

    def _kf_ray(self, k: int) -> list:
        """关键帧 k 的射线体素；只依赖朝向，缓存到 _rot 条目换掉为止。新算的先存 K×3 原始体素
        （布局 None），累加器长够之后由 ``_encode_ray`` 换成一维下标。"""
        rot = self._rot[k]
        c = self._ray.get(k)
        if c is not None and c[0] is rot:
            return c
        cfg = self.cfg
        rxy, rel, cnt, _R = rot
        h = rel + np.float32(self.cam_h)
        corr, under = _ground_model(h, cnt, rxy, cfg.ground_offset_max_m, cfg.ground_offset_min_range_m,
                                    cfg.ground_plane_max_deg, cfg.ground_plane_band_m)
        h = h - np.float32(corr)
        t = self._pose[k][:2, 3].copy()
        hits, miss = _ray_voxels(rxy, h, t, self.cam_h - under, cfg)
        Z = self._ray_z()
        hits = hits[(hits[:, 2] >= 0) & (hits[:, 2] < Z)]
        miss = miss[(miss[:, 2] >= 0) & (miss[:, 2] < Z)]
        both = np.vstack([hits, miss])
        box = (int(both[:, 0].min()), int(both[:, 1].min()), int(both[:, 0].max()), int(both[:, 1].max()))             if len(both) else None
        c = [rot, t, box, hits, miss, None]
        self._ray[k] = c
        return c

    def _acc_layout(self) -> tuple[int, int, int, int]:
        (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
        return x0, y0, H, W

    def _encode_ray(self, c: list) -> None:
        """把射线体素存成当前累加器布局下、平移为 0 时的 int32 一维下标（4 B/体素），整格平移只是加常数。
        累加器长大换布局时按旧布局解码再编一次（很少发生）；_sync_ray 保证 t0 包围盒一直在累加器里，解码不越界。"""
        x0, y0, H, W = lay = self._acc_layout()
        if c[5] is None:
            for i in (3, 4):
                v = c[i]
                c[i] = ((v[:, 2] * H + v[:, 1] - y0) * W + v[:, 0] - x0).astype(np.int32)
            c[5] = lay
            return
        if c[5] == lay:
            return
        ox0, oy0, oH, oW = c[5]
        for i in (3, 4):
            f = c[i].astype(np.int64)
            iz, r = np.divmod(f, oH * oW)
            iy, ix = np.divmod(r, oW)
            c[i] = ((iz * H + iy + oy0 - y0) * W + ix + ox0 - x0).astype(np.int32)
        c[5] = lay

    def _sync_ray(self, keys: set[int]) -> None:
        """与 _sync_acc 同一套增量：只处理 (射线缓存, 整格平移) 变了的关键帧。必须在 _sync_acc 之后调，
        ``keys`` 用它的返回值（累加器重建过时是全部关键帧）。"""
        if not self.cfg.ray_clear or self._ray_hit is None:
            return
        res = self.cfg.res_m
        work: list[tuple[list, int, int, int]] = []
        for k in keys:
            if k not in self._pts:
                continue
            ray = self._kf_ray(k)
            d = (self._pose[k][:2, 3] - ray[1]) / res
            sx, sy = int(round(float(d[0]))), int(round(float(d[1])))
            prev = self._ray_applied.get(k)
            if prev is not None and prev[0] is ray and prev[1] == sx and prev[2] == sy:
                continue
            if prev is not None:
                work.append((*prev, -1))
            work.append((ray, sx, sy, 1))
            self._ray_applied[k] = (ray, sx, sy)
        for k in [k for k in self._ray_applied if k not in self._pts]:
            work.append((*self._ray_applied.pop(k), -1))
            self._ray.pop(k, None)
        if not work:
            return
        if len(work) > len(self._ray_applied):
            # 大半关键帧都挪了（回环）：清零后只加一遍，比逐个"减旧加新"少一半写入。
            self._ray_hit[...] = 0
            self._ray_mis[...] = 0
            work = [(*v, 1) for v in self._ray_applied.values()]
        # 平移后的包围盒要装得下；t0 的也要（编码下标以 t0 为准，换布局解码要求它在累加器内）。
        boxes = np.array([(r[2][0] + min(sx, 0), r[2][1] + min(sy, 0), r[2][2] + max(sx, 0), r[2][3] + max(sy, 0))
                          for r, sx, sy, g in work if g > 0 and r[2] is not None], np.int64).reshape(-1, 4)
        if len(boxes):
            self._grow_acc(int(boxes[:, 0].min()), int(boxes[:, 1].min()),
                           int(boxes[:, 2].max()), int(boxes[:, 3].max()))
        W = self._acc_g.shape[1]
        for r, _sx, _sy, _g in work:
            self._encode_ray(r)
        for i, acc in ((3, self._ray_hit), (4, self._ray_mis)):
            parts = [(r[i], sy * W + sx, g) for r, sx, sy, g in work if len(r[i])]
            if not parts:
                continue
            idx = np.concatenate([f + off for f, off, _g in parts])
            sg = np.repeat(np.array([g for _f, _o, g in parts], np.int32), [len(f) for f, _o, _g in parts])
            # 一维下标 np.add.at（numpy ≥2 已向量化）：比 np.unique 去重快一个量级，也不用分配整块三维 bincount。
            np.add.at(acc.reshape(-1), idx, sg)

    def _near_plane_parts(self) -> dict[str, np.ndarray]:
        """``_near_plane`` 的逐项判据（**累加器行序**）。返回各条的掩码，供归因用。"""
        c = self.cfg
        if self._acc_o is None or self._acc_ob_h is None:
            z = np.zeros((0, 0), bool)
            return {"pts": z, "near": z, "inside": z, "flat": z, "all": z}
        n_o = self._acc_o
        pts = n_o >= c.ray_plane_pts
        near = pts & (self._acc_on >= c.ray_plane_near * np.maximum(n_o, 1e-9))
        H, W = n_o.shape
        hi = np.full((H, W), -np.inf, np.float32)
        lo = np.full((H, W), np.inf, np.float32)
        cnt = np.zeros((H, W), np.int8)
        mean = np.where(pts, self._acc_ob_h / np.maximum(n_o, 1e-9), -np.inf)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys0, ys1 = max(0, dy), H + min(0, dy)
                xs0, xs1 = max(0, dx), W + min(0, dx)
                s = mean[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
                v = pts[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
                np.fmax(hi[ys0:ys1, xs0:xs1], np.where(v, s, -np.inf), out=hi[ys0:ys1, xs0:xs1])
                np.fmin(lo[ys0:ys1, xs0:xs1], np.where(v, s, np.inf), out=lo[ys0:ys1, xs0:xs1])
                cnt[ys0:ys1, xs0:xs1] += v
        # 三个变体在 044153 + 045615 上量过的结果（tools/ray_exempt_ab.py，
        # 括号里是**票面**比值"回收的真结构格 / 新封死的路径格"——不是判据）：
        #
        #   每格平均高度 + 邻居≥6      135/29 (4.7:1)    47/33  (1.4:1)   ← 三个里最不差
        #   众数层     + 邻居≥6         59/10 (5.9:1)    30/127 (0.2:1)   亏
        #   众数层     + 边缘感知邻居     70/16 (4.4:1)    44/145 (0.3:1)   亏
        #
        # ⚠️ 三个变体在 tools/ray_exempt_conn.py 的连通性上**全部净亏**，开关保持 False：
        # 后两个都是在修真问题（均值被立面+顶面混拉散、邻居≥6 误杀 0.4 m 细墙），结果更差：
        #   * 换成 ``_ray_hit`` 的众数层（``ray_z_m=0.10`` 已经切好层，argmax 零额外成本）
        #     确实把"卡在平面这一项"的漏放从 60~78% 降到 36~55%，但**同时放进大量幻影**：
        #     045615 回收几乎没涨（44 vs 47），路径封死从 33 涨到 145。
        #   * 邻居≥6 改成"只看数组边缘"确实收回部分细墙，但也是往放幻影那一侧走，127→145。
        # ``flat`` 和 ``inside`` 是一对**互相支撑**的误杀集合：均值版里 ``flat`` 顺手拦掉的
        # 幻影，正是众数版放进来的那批。只放松一个，另一个就顶上。
        #
        # **这两个门在两段录制上调不出一个共同好的版本，别再盲调。** 要继续先得有第三段录制
        # ——现在是在两点之间过拟合，分不清哪个门真的更好。归因看 tools/plane_gate_why.py。
        inside = pts & (cnt >= 6)
        flat = inside & ((hi - lo) <= c.ray_plane_tol)
        return {"pts": pts, "near": near, "inside": inside, "flat": flat, "all": near & flat}

    def _near_plane(self) -> np.ndarray:
        """障碍带里"有一张连贯水平面"的格（**累加器行序**，可以直接喂给 ``_ray_veto``）。

        这是 ``ray_near_exempt`` 的门。用票数当门已经实测是净亏（tools/ray_exempt_ab.py：
        044153 回收 195 但封死 129 个路径格，045615 回收 123 封死 1155）——``_acc_on > min_pts``
        把"近距看过一次"和"近距看清一张面"混成了一件事。

        ⚠️ **换成这个几何门之后仍然是净亏**，只是亏的方式变了：票面比值变成 4.7:1 / 1.4:1
        看着划算，但 tools/ray_exempt_conn.py 量连通性——044153 可达域 −1104 格、隔离出 1 块
        区域、4 对走过的路变得走不通、绕行 +13%；045615 可达域 −954 格、+3%。
        **格数比值与真实代价反向**（赔率最好看的那段损伤最重），所以别拿 ab 的比值定案。

        判据三项（逐项掩码见 ``_near_plane_parts``）：

          票够    ``n_o >= ray_plane_pts``。1.5 m 外真地面被抖进障碍带是必然的（δz=z²/25.5，
                  带宽 1.7 m），所以票数先要压过噪声地板。
          近距    近距票占障碍票 ≥ ``ray_plane_near``。1.5 m 处 δz 只有 9 cm，地面抖不出 1 m，
                  所以近距离成片出现的面不可能是散点。
          平面    3×3 邻域**每格平均高度**的极差 ≤ ``ray_plane_tol``。同一张面在邻域里高度
                  应当基本一致；散点会摊开。随机置换对照里这个判据 16×/30× 信噪比。

        用每格**平均**高度而不是众数：矩可加、增量账（回环挪位/撤帧）能精确回退，众数不行——
        和 ``surface_counts`` 选矩不选众数是同一个理由。代价是单格内的双峰会拉高均值，
        所以平面这一项只是必要条件，单靠它不能定案。
        """
        return self._near_plane_parts()["all"]

    def _ray_veto(self, walked: list[np.ndarray] | None) -> np.ndarray | None:
        """累加器坐标下"障碍被看穿清掉"的格：没有任何一层体素满足 看穿 < ray_beta × 打中。"""
        if not self.cfg.ray_clear or self._ray_hit is None:
            return None
        c = self.cfg
        (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
        veto = np.zeros((H, W), bool)
        # 只看本来就够障碍点数的格：整块体素逐个比，1200 帧的图每次要几十 ms。
        cand = np.flatnonzero(self._acc_o.reshape(-1) > c.min_pts - 0.5)
        if not len(cand):
            return veto
        hit = self._ray_hit.reshape(len(self._ray_hit), -1)[:, cand].astype(np.float32)
        mis = self._ray_mis.reshape(len(self._ray_mis), -1)[:, cand].astype(np.float32)
        if walked:
            line = np.zeros((H, W), np.uint8)
            k = 1.0 / (c.world_scale * c.res_m)
            for q in walked:
                p = np.column_stack([np.floor(q[:, 0] * k) - x0, np.floor(q[:, 1] * k) - y0]).astype(np.int32)
                cv2.polylines(line, [p.reshape(-1, 1, 2)], False, 1, 1)
            mis += line.reshape(-1)[cand].astype(np.float32) * np.float32(c.ray_walk_w)
        veto.reshape(-1)[cand] = ~((hit > 0) & (mis < c.ray_beta * hit)).any(axis=0)
        # 近距票豁免（默认关）。1.5 m 处 δz 只有 9 cm，一次近距打中本身就是可信的，不该让
        # **远距离**的"看穿"统计把它抹掉。现有判据要求整列**所有** z 层都不通过才清零，而一根
        # 从相机掠过矮墙顶、往墙后地面俯冲的射线会在墙顶那一层 0.80~0.90 m 处穿过去记一次看穿，
        # 于是真实矮墙被"自己上方的空间"判成透明。实测（tools/low_ceiling_nav.py，044153）：
        # 0.85 m 那一层 349 个有打中的格，**349 个全部**被看穿票压过。
        if c.ray_near_exempt:
            veto &= ~self._near_plane()
        return veto

    def add_trail(self, node_id: int, pose_map_frame: np.ndarray, osc_dist_m: float | None = None) -> None:
        """关键帧之间的逐帧位姿（地图系，追踪米）挂到最近的前一个关键帧上，存成相对位姿，
        回环后跟着关键帧走。关键帧 1 Hz 时一跳可达 4 m，只靠关键帧连线门控会被尺度误差误断。"""
        k = int(node_id)
        if k not in self._pose:
            return
        rel = np.linalg.inv(self._pose[k]) @ np.asarray(pose_map_frame, np.float64).reshape(4, 4)
        self._trail.setdefault(k, []).append((rel[:2, 3].copy(), rel[:2, :2].copy(),
                                              None if osc_dist_m is None else float(osc_dist_m)))
        self._trail_arr.pop(k, None)
        self._ver += 1

    def _trail_xy_d(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        c = self._trail_arr.get(k)
        if c is None:
            tr = self._trail.get(k, ())
            c = (np.array([t for t, _r, _d in tr], float).reshape(-1, 2),
                 np.array([np.nan if d is None else d for _t, _r, d in tr], float))
            self._trail_arr[k] = c
        return c

    def walked(self, *, gate_m: float = 0.15, gate_frac: float = 0.05,
               max_hop_m: float = 1.5) -> list[np.ndarray]:
        """关键帧 + 逐帧轨迹（按 id 顺序，当前位姿）→ 走过的折线，导航系世界米。

        双目看不到脚下 ~1.6 m 以内的地面，身体实际走过的地方只能靠这条链当通行证据。
        相邻两点的 SLAM 位移超过 OSC 位移 ×(1+gate_frac) + gate_m 就断开（位姿跳变不能画成走廊）；
        没有 OSC 路程时退回 max_hop_m（追踪米）硬门限。"""
        key = (self._ver, gate_m, gate_frac, max_hop_m)
        if self._walked_cache is not None and self._walked_cache[0] == key:
            return [q.copy() for q in self._walked_cache[1]]
        out = self._walked(gate_m, gate_frac, max_hop_m)
        self._walked_cache = (key, out)
        return [q.copy() for q in out]

    def _walked(self, gate_m: float, gate_frac: float, max_hop_m: float) -> list[np.ndarray]:
        s = self.cfg.world_scale
        keys = sorted(self._pose)
        if not keys:
            return []
        # 每个关键帧先放自己，再放挂在它上面的轨迹点；位姿批量套，1200 帧时逐帧 Python 循环要 15 ms。
        Ts = np.array([self._pose[k] for k in keys])
        tr = [self._trail_xy_d(k) for k in keys]
        n = np.array([1 + len(t[0]) for t in tr])
        owner = np.repeat(np.arange(len(keys)), n)
        rel = np.zeros((len(owner), 2))
        d = np.empty(len(owner))
        head = np.cumsum(n) - n
        d[head] = [self._osc.get(k, np.nan) for k in keys]
        tail = np.ones(len(owner), bool)
        tail[head] = False
        if tail.any():
            rel[tail] = np.concatenate([t[0] for t in tr])
            d[tail] = np.concatenate([t[1] for t in tr])
        p = (Ts[owner, :2, 3] + np.einsum("nij,nj->ni", Ts[owner, :2, :2], rel)) * s
        step = np.hypot(*(p[1:] - p[:-1]).T)
        dd = d[1:] - d[:-1]                        # 任一端没有 OSC 路程 → NaN → 退回硬门限
        limit = np.where(np.isnan(dd), max_hop_m * s, np.nan_to_num(dd) * (1 + gate_frac) + gate_m)
        cuts = np.flatnonzero(~(step <= limit)) + 1
        return [q for q in np.split(p, cuts) if len(q) >= 2]

    def update_poses(self, poses: dict[int, np.ndarray]) -> float:
        """回环优化后整体换位姿；返回关键帧的最大平移变化（追踪米），便于记录。"""
        moved = 0.0
        for k, T in poses.items():
            if k in self._pose:
                T = np.array(T, np.float64).reshape(4, 4)
                if np.array_equal(T, self._pose[k]):
                    continue
                moved = max(moved, float(np.linalg.norm(T[:3, 3] - self._pose[k][:3, 3])))
                self._pose[k] = T
                self._dirty.add(k)
                self._ver += 1
        return moved

    def pose(self, node_id: int) -> np.ndarray | None:
        """关键帧在**当前**地图系的位姿（回环修正后）；没有这个关键帧返回 None。

        跨会话 gauge 估计拿它当"新帧位姿"（``nav_prior``）：回环只改平移（朝向来自
        HMD，见 nav_online 模块 docstring），所以约束验证时的朝向与这里始终一致。
        """
        T = self._pose.get(int(node_id))
        return None if T is None else np.array(T, np.float64, copy=True)

    def anchor(self, xy_track: tuple[float, float]) -> tuple[int, tuple[float, float]] | None:
        """把地图系一点挂到最近的关键帧上（该关键帧水平系下的偏移）。回环后地图系会动，
        目标用 ``resolve`` 跟着关键帧走，而不是钉死在旧坐标上。"""
        if not self._pose:
            return None
        p = np.asarray(xy_track, float)
        k = min(self._pose, key=lambda i: float(np.hypot(*(self._pose[i][:2, 3] - p))))
        T = self._pose[k]
        yaw = math.atan2(T[1, 0], T[0, 0])
        d = p - T[:2, 3]
        c, s = math.cos(yaw), math.sin(yaw)
        return k, (c * d[0] + s * d[1], -s * d[0] + c * d[1])

    def resolve(self, anc: tuple[int, tuple[float, float]]) -> tuple[float, float] | None:
        k, (lx, ly) = anc
        T = self._pose.get(k)
        if T is None:
            return None
        yaw = math.atan2(T[1, 0], T[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        return float(T[0, 3] + c * lx - s * ly), float(T[1, 3] + s * lx + c * ly)

    def rasterize(self, extra_xy: np.ndarray | None = None) -> NavGrid:
        """当前全部关键帧 → NavGrid（未 build）。``extra_xy``：还要包进图里的点（追踪米，如当前位姿）。

        增量：全局累加器只更新变了的关键帧——新关键帧投一次；回环只改平移，按整格挪下标；
        朝向或 cam_h 变了才重投。输出只裁剪累加器，不再每次拼接全部关键帧的格。"""
        c = self.cfg
        dirty, self._dirty = {k for k in self._dirty if k in self._pts}, set()
        for k in dirty:
            self._rotated(k)
        self._update_cam_h()
        self._sync_ray(self._sync_acc(dirty))
        veto = self._ray_veto(self.walked()) if self.cfg.ray_clear else None
        pts = []
        if self._acc_g is not None:
            seen = (self._acc_g > 0.5) | (self._acc_o > 0.5)
            rows, cols = np.flatnonzero(seen.any(axis=1)), np.flatnonzero(seen.any(axis=0))
            if len(rows):
                x0, y0 = self._acc_lo
                pts.append(np.array([[x0 + cols[0], y0 + rows[0]], [x0 + cols[-1], y0 + rows[-1]]], float)
                           * c.res_m + 0.5 * c.res_m)
        anchors = [T[:2, 3] for T in self._pose.values()]
        if extra_xy is not None:
            anchors += list(np.asarray(extra_xy, float).reshape(-1, 2))
        if anchors:
            pts.append(np.array(anchors, float))
        allxy = np.vstack(pts) if pts else np.zeros((1, 2))
        # 原点对齐到 res 整数倍：图长大时旧格子编号含义不变，调试时两帧可以直接逐格比。
        lo_i = np.floor((allxy.min(axis=0) - c.pad_m) / c.res_m).astype(np.int64)
        lo = lo_i * c.res_m
        hi = allxy.max(axis=0) + c.pad_m
        w, h = (np.ceil((hi - lo) / c.res_m).astype(int) + 1)
        n_g, n_o, o_n, o_m, g_n, o_mk = (np.zeros((h, w)) for _ in range(6))
        if self._acc_g is not None:
            (x0, y0), (H, W) = self._acc_lo, self._acc_g.shape
            lx, ly = int(lo_i[0]), int(lo_i[1])
            self._grid_lo = (lx, ly)
            ox0, oy0 = max(lx, x0), max(ly, y0)
            ox1, oy1 = min(lx + w, x0 + W), min(ly + h, y0 + H)
            if ox1 > ox0 and oy1 > oy0:
                dv = (slice(oy0 - ly, oy1 - ly), slice(ox0 - lx, ox1 - lx))
                sv = (slice(oy0 - y0, oy1 - y0), slice(ox0 - x0, ox1 - x0))
                for dst, src in zip((n_g, n_o, o_n, o_m, g_n, o_mk),
                                    (self._acc_g, self._acc_o, self._acc_on, self._acc_om,
                                     self._acc_gn, self._acc_omk)):
                    dst[dv] = src[sv]
                if veto is not None:
                    v = veto[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
                    view = n_o[oy0 - ly:oy1 - ly, ox0 - lx:ox1 - lx]
                    self.ray_cleared_cells = int((v & (view > c.min_pts - 0.5)).sum())
                    view[v] = 0.0
        # 权重 = 原始点数，min_pts 语义不变；累加器是整数加减，用 ±0.5 比较免得 1e-12 级残差翻转。
        # 扁平口径（历史规则）：只数格内总票数，看不见"这些票是几米外打的"。
        occ = ((n_o > c.min_pts - 0.5) & (n_o >= c.occ_ground_ratio * n_g)).astype(np.uint8)
        # BORDER_CONSTANT：默认的 reflect 会把图外沿的障碍格镜像回填，边缘一圈白送 3×3 支撑。
        occ = ((cv2.filter2D(occ, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) >= 3)
               .astype(np.uint8))
        occ, restored, rejected = self._by_quality(occ, n_g, n_o, o_n, o_m, g_n, o_mk)
        g = np.full((h, w), UNK, np.uint8)
        g[(n_g > 0.5) & (occ == 0)] = FREE
        g[restored] = FREE       # 远场票被否证、而近距在同一格见过地面 ⇒ 判回空地
        g[occ == 1] = OCC
        # 上面那行"有地面观测且非障碍 ⇒ free"会把**所有**被降级的格顺手填成 free，包括那些
        # 根本没有近距地面证据的。必须显式压回 unknown：没有可信证据说它是地板，就不许当 free
        # （nav_grid 的红线"unknown 永不当 free"）。少了这一行，降级就等于清障，白做。
        g[rejected & ~restored] = UNK
        self._last_grid = (h, w)
        return NavGrid(g[::-1].copy(), GridMeta(c.res_m, (float(lo[0]), float(lo[1])), c.world_scale))

    def _by_quality(self, base_occ: np.ndarray, n_g: np.ndarray, n_o: np.ndarray,
                    o_n: np.ndarray, o_m: np.ndarray, g_n: np.ndarray, o_mk: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把"扁平计数判为障碍"的格按**观测距离**复核一遍。返回 (障碍, 降级回空地, 降级但无近距证据)。

        为什么必须分层：视差深度误差 δz = z²/(fx·b)·δd，面板口径 fx 202.5 / 基线 0.126 下
        1 像素 ≈ z²/25.5 m（1.5 m→9 cm、3 m→35 cm、5 m→98 cm），而障碍高度带 (0.3, 2.0) 有
        1.7 m 宽。所以 1.5 m 以外"真地面被抖进障碍带"是必然的：min_pts、3×3 多数、
        occ_ground_ratio 都只压密度，补不回那一层根本不存在的信息。这就是"走过的空地中间
        冒出孤岛障碍"的物理来源（Docs/RTAB-Map双目评估（2026-09-26）.md:128 同一个结论）。

        规则（口径取自 Docs/离线多视角融合v1-假障碍清除（2026-10-02）.md，该录制障碍占比
        31.4%→13.3%；与 ``ray_clear`` 机理正交，可叠加）：
          * 近带 <q_near_m：误差还在带边沿内，单帧即定案；**故意不带**地面压制条件，
            否则 occ_ground_ratio 会把真细障碍（桌腿、栏杆）一起误杀；
          * 中带 q_near~q_mid：默认须 ≥q_mid_kf 个关键帧一致 + 地面压制，压住量化倾斜与单帧鬼影；
            但"只被一个关键帧看到"也是常态（刚走近就看见了），所以留了一条**单帧例外**：
            票够多（≥q_solo_pts）且远带票占比 ≤q_far_tol 时单帧即定案——近/中距的票本来就可信，
            远带票才是噪声来源，票里没有远带成分就没必要等第二个关键帧；
          * 远带 >q_mid：没有定案权。纯远场票的格降级——**近距**在同一格见过地面 ⇒ FREE，
            否则 ⇒ UNK（诚实降级，不伪造 free：见 nav_grid 的"unknown 永不当 free"）。

        与离线那一版的一处收紧：降级判回 FREE 只认**近距**地面证据，不用中距。中距地面自己
        就是被抖出来的，拿它当"这格是地板"的证据，等于用噪声治噪声——实测会把只被一个关键帧
        看到的中距真墙判成 free（可走 ⇒ 规划直接穿墙）。近距地面才够格推翻一张障碍票。
        """
        if not self.cfg.q_tiers:
            z = np.zeros(base_occ.shape, bool)
            return base_occ, z, z
        c = self.cfg
        hi = c.min_pts - 0.5
        near = o_n > hi
        mid = (o_m > hi) & (o_mk >= c.q_mid_kf - 0.5) & (n_o >= c.occ_ground_ratio * n_g)
        # 单帧例外：票密集且不含远带成分 ⇒ 近/中距证据本身就够定案，不等第二个关键帧。
        far = np.maximum(n_o - o_n - o_m, 0.0)
        solo = (o_m > hi) & (n_o >= c.q_solo_pts) & (far <= c.q_far_tol * np.maximum(n_o, 1.0))
        # ``q_near_pts`` 把文档里那句"近带**故意不带**地面压制"真正接上。历史上它是死代码：
        # ``rasterize`` 的 base 规则先做了 (n_o > min_pts) & (n_o ≥ ratio·n_g)，而
        # ``occ = (base_occ > 0) & (...)`` 只能给已有障碍加分，救不回被地面压制杀掉的格。
        # 后果实测（tools/near_band_check.py，044153）：201 格票数是门槛 8 倍、99.5% 判成可走，
        # 而它们的形状（n_g 中位 130 / n_o 中位 26）正是文档点名要保护的细障碍——桌腿、栏杆、矮墙。
        # 0 = 关闭，此时 near_only 恒 False，**与改动前逐格相同**（见 tests 的 no-op 断言）。
        if c.q_near_pts > 0:
            near_pts = min(c.q_near_pts, c.min_pts) - 0.5
            near_only = (o_n > near_pts) & (n_o > near_pts)
        else:
            near_only = np.zeros_like(near)
        # base_occ > 0 而不是 base_occ：base_occ 是 uint8，& 出来的还是 uint8，而
        # (a) numpy 把 uint8 数组当**整数下标**不是布尔掩码——g[那个数组] 会去写第 0 行、第 1 行，
        #     一格都改不到还不报错；(b) uint8 上做 ~ 是按位取反（0→254），不是逻辑非。
        # 三个返回值一律真 bool，调用方拿它当掩码用才对。
        ok_rule = near | mid | solo | near_only
        occ = ((base_occ > 0) & ok_rule) | near_only
        rejected = (base_occ > 0) & ~ok_rule
        return occ, rejected & (g_n > 0.5), rejected

    def coverage_counts(self) -> dict[str, Any] | None:
        """覆盖伴生网格的只读快照（**仅建图线程可调**，HTTP 侧消费下游快照）。

        total = near + far > 0 即"已观测格"；near = 近看（≤ cov_near_m）点数。
        布局与 _acc_g 相同：原点 ``lo``（格下标）、分辨率 ``res_m``（追踪米）。
        数组是引用不拷贝——调用方不得持有多轮更新之间的引用做写比较。"""
        if self._acc_g is None or self._acc_near is None:
            return None
        return {"near": self._acc_near, "far": self._acc_far,
                "lo": self._acc_lo, "res_m": float(self.cfg.res_m),
                "cov_near_m": float(self.cfg.cov_near_m)}

    def _cut_acc(self, acc: np.ndarray) -> np.ndarray:
        """把一个累加器（无翻转行序）裁成 ``NavGrid`` 行序（行 0 = 最大 y）。

        换算照 ``rasterize`` 的切片式抄：``n_g[R] = acc[R + (ly - ay)]``，再 ``g[::-1]``
        ⇒ NavGrid 行 = ``h-1-R``。**漏掉这个翻转不会报错，只会让旁路数据上下镜像**，
        看着还挺像回事——本文件里已经栽过两次，抽出来是为了别有第三次。
        """
        if self._last_grid is None or self._grid_lo is None:
            return np.zeros((0, 0))
        h, w = self._last_grid
        (ax, ay) = self._acc_lo
        lx, ly = self._grid_lo
        dh, dw = ly - ay, lx - ax
        s_lo, s_hi = max(0, dh), min(acc.shape[0], h + dh)
        c_lo, c_hi = max(0, dw), min(acc.shape[1], w + dw)
        out = np.zeros((h, w), acc.dtype)
        if s_hi > s_lo and c_hi > c_lo:
            sub = acc[s_lo:s_hi, c_lo:c_hi]
            r0 = h - s_hi + dh                 # sub[-1] 翻转后落到 NavGrid 行 h-1-(s_hi-1+dh)
            out[r0:r0 + sub.shape[0], c_lo - dw:c_lo - dw + sub.shape[1]] = sub[::-1]
        return out

    def band_counts(self) -> dict[str, Any] | None:
        """高度分带占用的只读快照（**仅建图线程可调**；旁路产物，不喂导航）。

        每个格子带四个**点计数**（不是三态）：``below``（地面以下）、``lo``（头顶 obst_top_m
        ~hi_band_m）、``mid``、``hi``。计数 >0 只说明"那个高度看到过东西"，**不等于障碍**——
        楼板、天花板、横梁、栏杆、桥面全都会落进去。要区分得再做带内高度聚类（尚未做），
        所以这里只交原始计数，不替上层下结论。

        布局与 ``_acc_g`` 相同：原点 ``lo``（累加器格下标）、分辨率 ``res_m``（追踪米）。
        带边界是 h 的上界（h = 离地高）：below=−ground_tol、lo=obst_top_m、mid=hi_band_m、
        hi=hi_band_top_m，末带无上界。数组是引用不拷贝。
        """
        if not self.cfg.hi_bands or self._acc_g is None or self._acc_b1 is None:
            return None
        c = self.cfg
        return {"below": self._acc_b0, "lo": self._acc_b1, "mid": self._acc_b2, "hi": self._acc_b3,
                "lo_xy": self._acc_lo, "res_m": float(c.res_m),
                "edges_m": [float(-c.ground_tol_m), float(c.obst_top_m),
                            float(c.hi_band_m), float(c.hi_band_top_m)]}

    def band_grid(self) -> np.ndarray | None:
        """把分带计数按 NavGrid 布局裁好（4, h, w）。调用方按 ``res_m`` 乘回世界米。

        ⚠️ 只覆盖 NavGrid 图幅，而图幅是按 ``_acc_g``/``_acc_o`` 定的。真实场景总有地面，
        所以图幅罩得住全部数据；但**只有头顶点、没有地面点**的格会被裁掉——那种格要拿全量的话
        直接读 ``band_counts()`` 里的原始累加器（自带 ``lo_xy`` 原点）。
        """
        if not self.cfg.hi_bands or self._acc_b1 is None:
            return None
        return np.stack([self._cut_acc(a) for a in
                         (self._acc_b0, self._acc_b1, self._acc_b2, self._acc_b3)])

    def surface_counts(self) -> dict[str, Any] | None:
        """头顶 ``obst_top_m`` 之上那张面的三个**可加矩**（旁路，不喂导航）。

        每格带 ``n``（点权重和）、``sum_h``（Σ w·h）、``sum_h2``（Σ w·h²），
        由此得**均值高度** ``mean = sum_h/n`` 与**格内标准差** ``sd = sqrt(sum_h2/n − mean²)``。
        为什么存矩而不是众数：现有累加器全靠 ``acc += 贡献`` 增减（回环挪位要能精确回退），
        而**众数不可加**——两帧各投一批点，并集的众数不等于各自众数的平均。
        h 是**离地高**，与带边界无关：改 ``hi_band_m`` 不用迁移历史。

        ⚠️ 均值与本格众数的**中位只差约 1 cm**（044153 +0.010 m / 045615 +0.0125 m，
        `research/tools/height_accuracy_report.py` 实测），但**均值的尾部松一倍**：
        3x3 平面残差 p90，众数场 2.72 cm vs 均值场 5.20 cm；< 5 cm 的格 100% vs 89.5%。
        （旧稿写的"均值系统性偏高 0.17 m"是把 p90 = +0.138 m 的**尾部值**当成了中位偏差。）
        所以判「这里有没有一张平整的面」够用；判「面正好在 3.0 m、能不能站上去」不够——
        那要众数，也就是带内高度直方图，尚未做。
        详见 Docs/建图实测能力边界（2026-10-05）.md §二。
        """
        if not self.cfg.hi_bands or self._acc_sb_h is None:
            return None
        return {"n": self._acc_sb_n, "sum_h": self._acc_sb_h, "sum_h2": self._acc_sb_h2,
                "lo_xy": self._acc_lo, "res_m": float(self.cfg.res_m),
                "above_m": float(self.cfg.obst_top_m)}

    def surface_grid(self) -> dict[str, np.ndarray] | None:
        """``surface_counts`` 的矩按 NavGrid 布局裁好，另给出 mean_h / sd_h（追踪米）。"""
        sc = self.surface_counts()
        if sc is None:
            return None
        out = {k: self._cut_acc(sc[k]) for k in ("n", "sum_h", "sum_h2")}
        nz = out["n"] > 0.5
        n = np.where(nz, out["n"], 1.0)
        out["mean_h"] = np.where(nz, out["sum_h"] / n, np.nan)
        out["sd_h"] = np.where(nz, np.sqrt(np.maximum(out["sum_h2"] / n - out["mean_h"] ** 2, 0.0)),
                               np.nan)
        return out


@dataclass
class Frontier:
    goal_xy_m: tuple[float, float]      # 导航系世界米，保证是可走中心格、与起点同区
    path_dist_m: float                  # 起点沿中心区的测地距离
    size_m: float                       # 边界长度（格数 × 格宽）

    def to_dict(self) -> dict[str, Any]:
        return {"goal_xy_m": [round(self.goal_xy_m[0], 3), round(self.goal_xy_m[1], 3)],
                "path_dist_m": round(self.path_dist_m, 2), "size_m": round(self.size_m, 2)}


def _geodesic(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """中心区上的 8 邻域 Dijkstra（格为单位）；不可达为 inf。"""
    import heapq

    dist = np.full(mask.shape, np.inf, np.float32)
    dist[start] = 0.0
    pq = [(0.0, start)]
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    H, W = mask.shape
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc, w in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and d + w < dist[rr, cc]:
                dist[rr, cc] = d + w
                heapq.heappush(pq, (d + w, (rr, cc)))
    return dist


def frontiers(ng: NavGrid, start_xy: tuple[float, float], *, min_size_m: float = 0.5,
              chunk_m: float = 1.5, reach_m: float = 0.6, start_snap_m: float = 0.5,
              limit: int = 8) -> list[Frontier]:
    """探索目标：观测 free 里紧挨 unknown 的边界，目标格取**可走中心区**里离边界 ≤ reach_m、
    且与起点连通的格。目标本身永远不在 unknown 里——走过去以后前面的 unknown 才会被看清。

    视野扇形的边界通常整圈连成一个分量；按 chunk_m 的方块切段，每段各出一个目标，
    否则只剩离起点最近的那一段（往往是身后），前方的边界永远不会被选中。"""
    snapped = ng.snap_to_center(start_xy, start_snap_m)
    if snapped is None:
        return []
    s, _ = snapped
    cw = ng.meta.cell_world_m
    g = ng.grid
    unk = (g == UNK).astype(np.uint8)
    touch = cv2.dilate(unk, np.ones((3, 3), np.uint8)) > 0
    edge = (g == FREE) & touch
    n, lab = cv2.connectedComponents(edge.astype(np.uint8), connectivity=8)
    reach = ng.center & (ng.labels == ng.labels[s])
    geo = _geodesic(reach, s)
    k = max(1, int(math.ceil(reach_m / cw)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    sizes = np.bincount(lab.ravel(), minlength=n)
    # 段 = (分量, 方块)；太碎的段（方块切角剩下的几格）并不单独出目标。
    blk = max(1, int(round(chunk_m / cw)))
    rr_e, cc_e = np.nonzero(lab > 0)
    keys = lab[rr_e, cc_e].astype(np.int64) * 1_000_000 + (rr_e // blk) * 1000 + (cc_e // blk)
    best: dict[tuple[int, int], Frontier] = {}
    for key in np.unique(keys):
        comp = int(key // 1_000_000)
        sel = keys == key
        if sizes[comp] * cw < min_size_m or sel.sum() * cw < min(min_size_m, chunk_m / 3):
            continue
        m = np.zeros(g.shape, np.uint8)
        m[rr_e[sel], cc_e[sel]] = 1
        cand = (cv2.dilate(m, ker) > 0) & np.isfinite(geo)
        if not cand.any():
            continue
        rr, cc = np.nonzero(cand)
        j = int(np.argmin(geo[rr, cc]))
        rc = (int(rr[j]), int(cc[j]))
        f = Frontier(ng.to_world(rc), float(geo[rc]) * cw, float(sel.sum()) * cw)
        if rc not in best or f.size_m > best[rc].size_m:     # 两段落到同一个目标格只留一个
            best[rc] = f
    out = sorted(best.values(), key=lambda f: f.path_dist_m)
    return out[:limit]


class NavSession:
    """增量地图上的"去某处 / 探索"会话：每次地图更新（新关键帧或回环）都重新栅格化、
    重新规划、换一个新的跟随器。

    * 目标挂在最近关键帧上（``KeyframeGridMapper.anchor``），回环挪动地图系后目标跟着走；
    * 探索模式下目标是 frontier，每次更新重选，但 ``keep_m`` 内还有 frontier 就不换，免得来回摆；
    * 估计位姿 ``est`` 与 ``PathFollower.step`` 同格式（导航系世界米），由调用方给——在线建图时
      就是 OSC 门控过的当前 SLAM 位姿，回环后它和地图一起跳。
    """

    def __init__(self, mapper: KeyframeGridMapper, *, radius_m: float = 0.25,
                 keep_m: float = 1.0, follow_cfg: Any = None,
                 request_update: Callable[[], Any] | None = None,
                 prior: Callable[[], Any] | None = None) -> None:
        self.mapper = mapper
        self.radius_m = radius_m
        self.keep_m = keep_m
        self.follow_cfg = follow_cfg
        # 跨会话世界先验（P0.3b）：callable 返回已投影到本会话帧的 ``nav_prior.PriorOverlay``，
        # 没有 gauge / 过不了质量闸时返回 None —— 先验一格不许用（见 backend/nav_prior.py）。
        self.prior = prior
        # 给了就异步：跟随器要重规划 / 到达 frontier 时只发请求，由建图线程去算；
        # 不给（离线回放、测试）就当场 on_map_update。
        self.request_update = request_update
        self.ng: NavGrid | None = None
        self.mode = "idle"          # idle / goto / explore / done
        self._anchor: tuple[int, tuple[float, float]] | None = None
        self._goal_req: tuple[float, float] | None = None   # goto 的原始目标，建图线程下次更新时挂锚
        self._gen = 0               # 意图（goto/explore/cancel/到达）每变一次 +1；过期的规划结果丢弃
        self._awaiting = False      # 已请求异步重规划、结果还没回来
        self.follower: Any = None
        self.contract: Any = None
        self.last: dict[str, Any] = {}

    @property
    def _s(self) -> float:
        return self.mapper.cfg.world_scale

    def _resolve(self, anchor: tuple[int, tuple[float, float]] | None) -> tuple[float, float] | None:
        if anchor is None:
            return None
        p = self.mapper.resolve(anchor)
        return None if p is None else (p[0] * self._s, p[1] * self._s)

    def goal_xy(self) -> tuple[float, float] | None:
        """当前目标（导航系世界米），已按最新关键帧位姿解析；goto 还没挂锚时返回原始目标。"""
        if self._anchor is None:
            return self._goal_req
        return self._resolve(self._anchor)

    def _anchor_of(self, xy_world: tuple[float, float]) -> tuple[int, tuple[float, float]] | None:
        return self.mapper.anchor((xy_world[0] / self._s, xy_world[1] / self._s))

    def _intent(self, mode: str, anchor: Any = None, goal_req: tuple[float, float] | None = None) -> None:
        self.mode, self._anchor, self._goal_req, self.follower = mode, anchor, goal_req, None
        self._gen += 1
        self._awaiting = mode in ("goto", "explore")

    def goto(self, xy_world: tuple[float, float]) -> None:
        self._intent("goto", goal_req=(float(xy_world[0]), float(xy_world[1])))

    def explore(self) -> None:
        self._intent("explore")

    def cancel(self) -> None:
        self._intent("idle")

    def snapshot(self) -> dict[str, Any]:
        """``compute`` 需要的会话意图。调用方在锁里取，``compute`` 在锁外跑。"""
        return {"gen": self._gen, "mode": self.mode, "anchor": self._anchor, "goal_req": self._goal_req}

    def compute(self, est: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
        """重的那半：栅格化 + build + frontier/规划。不改会话字段（mapper 缓存除外），
        返回交给 ``apply``。mapper 同一时刻只能有一个线程在写/算。"""
        from .nav_follow import PathFollower

        mode, anchor = snap["mode"], snap["anchor"]
        here = est.get("xy") if est.get("state") == "localized" else None
        extra = None if here is None else np.array([[here[0] / self._s, here[1] / self._s]])
        # 先验视图（已投影到会话帧）：过了质量闸才有；四角并进裁剪范围，先验覆盖区
        # （本场还没走到的部分）也要在图里，goto 才可能规划到那儿。
        ovl = None if self.prior is None else self.prior()
        if ovl is not None:
            extra = ovl.corners if extra is None else np.vstack([extra, ovl.corners])
        ng = self.mapper.rasterize(extra)
        walked = self.mapper.walked()
        # 脚下 ~1.6 m 双目看不到，身体走过的走廊（OSC 门控）是当前位置可走的唯一证据。
        # 先验必须在 build 之前叠：可走区/中心区都从三态图算，晚一步等于没叠。
        prior_info = None if ovl is None else ovl.apply_into(ng, walked)
        stats = ng.build(radius_m=self.radius_m, walked=walked)
        res: dict[str, Any] = {"gen": snap["gen"], "ng": ng, "mode": mode, "anchor": anchor,
                               "contract": None, "follower": None}
        info: dict[str, Any] = {"mode": mode, "grid": stats}
        if prior_info is not None:
            info["prior"] = prior_info
        res["info"] = info
        if mode in ("idle", "done") or here is None:
            info["reason"] = "not_localized" if here is None else mode
            return res
        if mode == "goto" and anchor is None and snap["goal_req"] is not None:
            anchor = res["anchor"] = self._anchor_of(snap["goal_req"])
        if mode == "explore":
            fs = frontiers(ng, here)
            info["frontiers"] = len(fs)
            if not fs:
                res["mode"], res["anchor"] = "done", None
                info["reason"] = "no_frontier"
                return res
            cur = self._resolve(anchor)
            pick = fs[0]
            if cur is not None:
                near = [f for f in fs if math.hypot(f.goal_xy_m[0] - cur[0], f.goal_xy_m[1] - cur[1]) <= self.keep_m]
                if near:
                    pick = near[0]
            anchor = res["anchor"] = self._anchor_of(pick.goal_xy_m)
        goal = self._resolve(anchor)
        if goal is None:
            res["mode"] = "idle"
            info["reason"] = "goal_anchor_lost"
            return res
        c = ng.plan(here, goal)
        res["contract"] = c
        info.update(c.to_dict())
        if c.accepted:
            res["follower"] = PathFollower(c.waypoints_xy_m, self.follow_cfg)
        return res

    def apply(self, res: dict[str, Any]) -> dict[str, Any]:
        """轻的那半：换上新栅格；意图没被改过才换规划（算的时候用户又 goto/cancel 了就只换栅格）。"""
        self.ng = res["ng"]
        info = res["info"]
        self._awaiting = res["gen"] != self._gen
        if res["gen"] != self._gen:
            info = {**info, "reason": "superseded"}
            self.last = info
            return info
        self.mode, self._anchor, self.follower = res["mode"], res["anchor"], res["follower"]
        if res["anchor"] is not None:
            self._goal_req = None
        if res["contract"] is not None:
            self.contract = res["contract"]
        self.last = info
        return info

    def on_map_update(self, est: dict[str, Any]) -> dict[str, Any]:
        """新关键帧 / 回环之后调用（同步版）。返回本次规划结果（给日志和 LLM 看）。"""
        return self.apply(self.compute(est, self.snapshot()))

    def _need_update(self, est: dict[str, Any]) -> None:
        if self.request_update is None:
            self.on_map_update(est)
        else:
            self.follower, self._awaiting = None, True
            self.request_update()

    def step(self, est: dict[str, Any], *, local_stop: bool = False) -> dict[str, Any]:
        idle = {"forward": 0.0, "turn_rate": 0.0}
        if self.mode == "idle":
            return {"state": "idle", "reason": "no_goal", **idle}
        if self.mode == "done":
            return {"state": "explore_done", "reason": "no_frontier", **idle}
        if self.follower is None:
            if self._awaiting and self.ng is not None:
                return {"state": "wait_map", "reason": "replanning", **idle}
            if self.ng is None or est.get("state") != "localized":
                return {"state": "wait_map", "reason": self.last.get("reason", "no_map"), **idle}
            # 规划被拒（目标在 unknown / 不连通……）：等下一次地图更新，不猜。
            return {"state": "blocked", "reason": self.last.get("reason", "no_plan"), **idle}
        out = self.follower.step(est, local_stop=local_stop)
        if out["state"] == "replan":
            self._need_update(est)
            return {**out, "forward": 0.0, "turn_rate": 0.0}
        if out["state"] == "arrived":
            if self.mode == "explore":
                self._intent("explore")     # 到了就当看过，下一次更新选新的 frontier
                self._need_update(est)
                return {**out, "state": "frontier_reached"}
            self._intent("idle")
        return out
