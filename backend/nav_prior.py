# -*- coding: utf-8 -*-
"""世界先验的**固化产出**与**在线消费**（P0.3b 下一步②的最后一环）。

**先验是什么**：``research/tools/offline_fusion.py --write-prior`` 把世界里多场会话的
三态地图（各自按并树的**刚体分量**落到同一坐标系）融合成一张世界系先验，落
``<world>/prior/prior.npz + prior.json``（0.1 m/格，原子写；0=UNK / 1=FREE / 2=OCC）。
本模块负责**在线消费**：会话进行中跨会话约束攒够 ⇒ 用 ``estimate_gauge`` 在线估
"本场会话 → 世界系"的 gauge（带质量闸）⇒ 把先验投影进本场会话帧 ⇒ 叠进在线栅格。

**为什么必须先估 gauge**：在线 mapper 的栅格在**会话自身 map 系**（会话起点为原点），
先验在世界系（世界树的 root 会话 gauge）。两者的关系正是跨会话重定位要解的那个量，
会话开始时未知 ⇒ **没攒够够格的约束之前，先验一格都不许用**（这正是 P0 的意义所在）。

**叠加规则（宁可 unknown 不可伪造，与 nav_grid 红线一致）**：
* live 证据优先——先验只写 live 判 UNK 的格，live 已见的格先验一个都不动；
* 先验自己的 UNK 格保持 unknown（不伪造 free）；
* 先验 OCC **不写"身体走过的走廊"格**：走过就是活的通行证据，而 gauge 有残余误差时
  一个假障碍守在走廊上会把自己困住（start 不在中心区 ⇒ 规划直接拒绝）；
* 先验 FREE 会被 live 后续观测覆盖（同一规则的另一半）。

**质量闸**（任一不过就不注入，``status()["prior"]`` 里如实报原因）—— 一律只对 **IRLS 内点核心**
（= 真正参与 gauge 的那批约束）说话：约束数 ≥ ``min_constraints``、核心 ≥ ``min_inlier``、
核心占比 ≥ ``min_inlier_frac``、核心位置残差中位 ≤ ``max_pos_med_m``、旋转残差中位 ≤
``max_rot_med_deg``、核心跨度 ≥ ``min_core_span_frac``（别只在命中区间的一小片成立）、
前后半交叉验证 ≤ ``max_split_m`` / ``max_split_yaw_deg``。
另外先验所在坐标系的根（``base_sid``）必须与约束引用的旧会话表**同根**
（``XSessionTracker.table_base``）——**绝不混 gauge**，同源判据见
``Docs/P0.3b会话末自动采纳（2026-10-03）.md`` 与 ``Docs/离线多视角融合v1-假障碍清除``。
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .nav_grid import FREE, OCC, UNK, NavGrid
from .nav_xsession import estimate_gauge

__all__ = ["PriorConfig", "PriorMap", "PriorOverlay", "PriorConsumer",
           "write_prior", "load_prior", "prior_meta", "project_prior", "PRIOR_DIR"]

PRIOR_DIR = "prior"
PRIOR_NPZ = "prior.npz"
PRIOR_JSON = "prior.json"
# 1 = 首版（labels/origin/res/world_scale/base_sid/built_wall）。
PRIOR_SCHEMA = 1
# 先验标签编码（**与 nav_grid 的 FREE/OCC/UNK 数值不同**，转换只在本模块的一处做）。
L_UNK, L_FREE, L_OCC = 0, 1, 2


@dataclass
class PriorConfig:
    """在线先验消费（``OnlineNavConfig.prior``）。默认值 = 保守档，全部进 status 便于现场核。"""

    enabled: bool = True
    # gauge 质量闸。门槛与 ``XSessionConfig.align_min_constraints`` 同源（会话末采纳的口径），
    # 但这里更严一点：采纳错只是写错一张表，注入错是**当场**把障碍画到错的格上。
    min_constraints: int = 8      # 参与 gauge 的原始约束数下限
    min_inlier: int = 6           # IRLS 内点数下限
    # 内点占比下限（防"约束多但全是外点"）。
    # ⚠️ 2026-10-06 从 0.5 降到 0.2：**这条判据原来在否决好数据**。live 四场复现（052927/050848/
    # 035644/001523）显示，一场会话的约束池天然是"一整场都自洽的核心 + 局部形变的散兵"（DR 漂移
    # 2.8% × 路程，回环只修平移 ⇒ 地图局部扭曲 1–3 m），核心稳定在 8–59 条、残差中位 0.14–0.31 m，
    # 而占比只有 0.23–0.36。旧门槛让散兵**否决**核心（035644 的 58/161 被拒，而同一批数据离线
    # 采纳的留出中位是 0.326 m ⇒ 本来就该放行）。占比仍有意义——它挡的是"池子很大而核心极小"。
    min_inlier_frac: float = 0.2
    max_pos_med_m: float = 0.5    # 内点位置残差（中位）上限
    max_rot_med_deg: float = 5.0  # 内点旋转残差（中位）上限
    # 核心跨度：核心约束的 ``new_dist_m`` 跨度 / 全池跨度。核心若只挤在命中区间的一小段里，
    # 说明这个 gauge 只在那一小片成立（会话帧局部扭曲），别拿去铺整张先验图。
    min_core_span_frac: float = 0.5
    # 半样本交叉验证：两组子集各估一次 gauge，比它们把**同一批新帧**映射到哪儿。
    # 为什么不能只靠残差（2026-10-06 live 教训）：16 条约束就能拟合出一个**自洽但错**的刚体
    # 变换——池化残差 0.39 m / 1.8° 全过闸，而那个 gauge 的 yaw 是 2.4°，全场 161 条给的是
    # 8.6°（差 6°）。先验按它投影 ⇒ 现场看到"一张废了的地图"，约束攒多后闸门改判拒绝、
    # 地图自己恢复正常。两组独立估计对不上 = 这组数据还撑不起一个 gauge，一律不注入。
    # ⚠️ 2026-10-06 起这道闸与上面的残差闸一样**只在核心上做**（外点本来就不参与拟合，让它们
    # 再来一次否决是双重标准）；上面那条"核心要有跨度"接住了"只在局部自洽的核心"这个洞，
    # 事故场（16 条时核心 = 全池 ⇒ 前后半 1.007 m/4.59°）照旧被拒 —— 有回归测试钉住。
    max_split_m: float = 0.6          # 两组 gauge 映射同一批新帧的位置差（中位）上限
    max_split_yaw_deg: float = 2.5    # 两组 gauge 的 yaw 差上限

    # ---- 分片（tile）降级档（2026-10-06 新增）----
    # 为什么要有这一档：全局 gauge 是**一个刚体管整张图**，会话帧只要有一片局部形变
    # （DR 漂移 + 回环只修平移 ⇒ 地图局部扭曲 1–3 m），整张先验就被否决 —— 明明对准的
    # 那大半张图跟着一起丢。判据没错，错在**粒度**：该问的不是"整张图能不能用一个刚体
    # 对上"，而是"**哪几片**能对上"。⇒ 全局闸拒了之后，改用分片 gauge 兜底：每片只用
    # 它自己附近的约束估 gauge、自己过闸，**过闸的那几片才注入**；没过闸的片保持 unknown。
    # ⚠️ 它是**降级档不是替代档**：全局闸放行时不动（那说明整图一致，分片只会引入片间
    # 数值抖动）；全局拒了才启用。宁可少画，不画错（035644 那张废图的代价守着这条）。
    tiles_enabled: bool = True
    tile_size_m: float = 8.0          # 分片边长（会话帧追踪米）；片内先验格用该片的 gauge
    # 每片估 gauge 时用**半径**内的约束（> tile_size_m 保证相邻片的约束集高度重叠 ⇒
    # 片与片的 gauge 连续变化，不会在片界上把地图撕开）。
    tile_neighbor_m: float = 12.0
    tile_min_constraints: int = 6     # 片内（含邻域）约束数下限，不够 ⇒ 该片不注入
    tile_min_inlier: int = 4          # 片内 IRLS 内点下限
    tile_split_min: int = 8           # 片内约束 ≥ 这个数才补做前后半交叉验证

    # ---- free 注入开关（2026-10-06 新增）----
    # 为什么默认改 False：free 是**上千关键帧在各自漂移位姿下累积扫出来**的，先验渲染实证
    # （.tmp/_render_prior.py，3 场 1497 kf）free 呈放射状星芒、轨迹跨度 27–42 m（实际房间
    # 量级 ~15 m）——约束 gauge 残差合格（0.355 m）**不代表 free 不糊**：闸门量的是约束
    # 一致性，量不了"free 累积漂移"。注入糊 free = 往 unknown 区铺假可走 ⇒ 污染 frontier。
    # OCC 不受此影响：融合档验证过（已知内障碍 31→11~13、各场 OCC 落先验 OCC 85–98%），
    # 且 apply_into 已有 walked 走廊保护。⇒ 先验注入**默认只注障碍**，free 等离线把
    # free 质量（多场一致性阈值 / 更好位姿）修好再开。
    inject_free: bool = False


@dataclass
class PriorMap:
    """磁盘上的世界先验（只读）。``labels`` 行 = y、列 = x；格心 = origin + (i+0.5)·res。"""

    labels: np.ndarray      # (H, W) uint8，0/1/2
    origin_xy: np.ndarray   # (2,) 追踪米，世界系（格下标 0 的**左下角**）
    res_m: float            # 追踪米/格
    world_scale: float      # 产出时的世界尺度（与在线不一致 ⇒ 拒绝消费）
    base_sid: str           # 坐标系的根会话（= 并树 merged 的 base_sid）
    meta: dict[str, Any]    # prior.json 全文（血缘/统计）
    path: Path


def prior_dir(world_dir: str | Path) -> Path:
    return Path(world_dir) / PRIOR_DIR


def write_prior(world_dir: str | Path, labels: np.ndarray, origin_xy: Iterable[float], res_m: float,
                *, world_scale: float, base_sid: str, meta: dict[str, Any] | None = None) -> Path:
    """原子写世界先验（npz + json **成对**）。写完返回 npz 路径。

    顺序：两份都先写 ``.tmp``，再先 replace npz、后 replace json（json 最后落 = 提交点）。
    读到"新 npz + 旧 json"的窗口由 ``built_wall`` 成对校验挡掉（见 ``load_prior``）。
    """
    lab = np.asarray(labels, np.uint8)
    if lab.ndim != 2 or lab.size == 0:
        raise ValueError(f"labels 必须是 (H, W) 非空二维数组，收到 {lab.shape}")
    bad = sorted(set(np.unique(lab).tolist()) - {L_UNK, L_FREE, L_OCC})
    if bad:
        raise ValueError(f"labels 含非法取值 {bad}（合法 {L_UNK}/{L_FREE}/{L_OCC}）")
    res = float(res_m)
    if not (res > 0.0):
        raise ValueError(f"res_m 必须为正，收到 {res_m!r}")
    origin = np.asarray(origin_xy, np.float64).reshape(2)
    if not np.isfinite(origin).all():
        raise ValueError(f"origin_xy 含 NaN/Inf：{origin}")
    ws = float(world_scale)
    if not (ws > 0.0):
        raise ValueError(f"world_scale 必须为正，收到 {world_scale!r}")

    built = float(time.time())
    d = prior_dir(world_dir)
    d.mkdir(parents=True, exist_ok=True)
    npz_tmp, json_tmp = d / (PRIOR_NPZ + ".tmp"), d / (PRIOR_JSON + ".tmp")
    # np.savez 会给自己加 .npz 后缀（"prior.npz.tmp" → "prior.npz.tmp.npz"）⇒ 用文件句柄写。
    with open(npz_tmp, "wb") as f:
        np.savez_compressed(f, labels=lab, origin_xy=origin, res_m=np.float64(res),
                            world_scale=np.float64(ws), base_sid=str(base_sid),
                            built_wall=np.float64(built))
    doc = {"schema": PRIOR_SCHEMA, "built_wall": built, "base_sid": str(base_sid),
           "res_m": res, "origin_xy": [float(v) for v in origin], "world_scale": ws,
           "frame": "world_map_track_m", "shape": [int(lab.shape[0]), int(lab.shape[1])],
           "counts": {"free": int((lab == L_FREE).sum()), "occ": int((lab == L_OCC).sum()),
                      "unknown": int((lab == L_UNK).sum())},
           **dict(meta or {})}
    json_tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    npz_tmp.replace(d / PRIOR_NPZ)
    json_tmp.replace(d / PRIOR_JSON)
    return d / PRIOR_NPZ


def prior_meta(world_dir: str | Path) -> dict[str, Any] | None:
    """只读 ``prior.json`` 的血缘摘要（**不碰 npz**）：有没有先验、什么时候固化的、多少格。

    给管理页/面板显示"这个世界的先验在不在、是几号固化的"用 —— 加载整张图（npz，几 MB）
    只为说一句"有"，代价不成比例。读不到 / schema 不认识 ⇒ None（宁可说没有）。
    """
    try:
        doc = json.loads((prior_dir(world_dir) / PRIOR_JSON).read_text(encoding="utf-8"))
        if int(doc.get("schema", 0)) != PRIOR_SCHEMA:
            return None
        counts = dict(doc.get("counts") or {})
        built = doc.get("built_wall")
        return {"built_at": (time.strftime("%Y-%m-%d %H:%M", time.localtime(float(built)))
                             if built else None),
                "built_wall": built, "base_sid": doc.get("base_sid"),
                "free_cells": counts.get("free"), "occ_cells": counts.get("occ"),
                "sessions": [s.get("sid") for s in (doc.get("sessions") or [])]}
    except (OSError, ValueError, TypeError):
        return None


def load_prior(world_dir: str | Path) -> tuple[PriorMap | None, str]:
    """读世界先验；返回 ``(PriorMap | None, 原因)``（成功时原因为空串）。永不抛。

    缺文件 = ``no_prior``（正常态：还没固化过）；坏文件/对不上 = 如实报原因并拒用，
    **不猜**（宁可没有先验，不可拿一张错位图当先验）。
    """
    d = prior_dir(world_dir)
    npz_p, js_p = d / PRIOR_NPZ, d / PRIOR_JSON
    if not (npz_p.is_file() and js_p.is_file()):
        return None, "no_prior"
    try:
        doc = json.loads(js_p.read_text(encoding="utf-8"))
        if int(doc.get("schema", 0)) != PRIOR_SCHEMA:
            return None, "schema_mismatch"
        with np.load(npz_p) as z:
            labels = np.asarray(z["labels"], np.uint8)
            origin = np.asarray(z["origin_xy"], np.float64).reshape(2)
            res = float(z["res_m"])
            ws = float(z["world_scale"])
            base = str(z["base_sid"])
            built = float(z["built_wall"])
    except (OSError, KeyError, ValueError) as exc:
        return None, f"unreadable:{type(exc).__name__}"
    if abs(built - float(doc.get("built_wall", -1.0))) > 1e-6:
        return None, "pair_mismatch"          # 写入中途被读到：npz/json 不是同一代
    if labels.ndim != 2 or labels.size == 0:
        return None, "bad_labels"
    if not (res > 0.0) or not np.isfinite(origin).all() or not (ws > 0.0):
        return None, "bad_meta"
    if sorted(set(np.unique(labels).tolist()) - {L_UNK, L_FREE, L_OCC}):
        return None, "bad_labels"
    if not base:
        return None, "bad_meta"
    return PriorMap(labels=labels, origin_xy=origin, res_m=res, world_scale=ws,
                    base_sid=base, meta=doc, path=npz_p), ""


@dataclass
class PriorOverlay:
    """先验投影到**本场会话帧**的一次性视图（gauge 一变就得重建，见 ``project_prior``）。

    ``labels`` 只保留非 UNK 格（UNK 不携带信息）；``xy_session`` 是这些格的**格心**在
    会话帧的连续坐标——先验与在线栅格同分辨率也会因两系之间任意转角而错位，
    所以必须按格心重采样，不能按格下标平移。
    """

    labels: np.ndarray        # (M,) uint8，1=FREE / 2=OCC
    xy_session: np.ndarray    # (M, 2) float64，会话帧**追踪米**
    corners: np.ndarray       # (4, 2) 先验标注范围的投影四角（给栅格扩边用）
    stat: dict[str, Any]      # 血缘摘要 + 最近一次 apply_into 的注入统计
    #: 最近一次 ``apply_into`` 里**真写进栅格**的格掩码（1-D bool，长度 = ``labels``）。
    #: 没被写 = live 已有证据（先验让位）或被走廊规则挡下。整块替换、绝不就地改，
    #: 所以 HTTP 线程读引用是安全的；给 ``prior`` 图层区分"先验补的格"与"live 自己的格"用。
    applied_mask: np.ndarray | None = None

    def cells_of(self, ng: NavGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """投影格心 → ``ng`` 的 (row, col) 与在界掩码。**取整口径只此一处**，
        ``apply_into`` 与可视化图层共用，避免两处各推一遍推歪。"""
        m = ng.meta
        h, w = ng.grid.shape
        col = np.floor((self.xy_session[:, 0] - m.origin_xy_m[0]) / m.resolution_m).astype(np.int64)
        rfb = np.floor((self.xy_session[:, 1] - m.origin_xy_m[1]) / m.resolution_m).astype(np.int64)
        row = h - 1 - rfb
        inb = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        return row, col, inb

    def apply_into(self, ng: NavGrid, walked: Iterable[np.ndarray] = ()) -> dict[str, Any]:
        """把先验叠进在线栅格（**就地**改 ``ng.grid``，只写 live UNK 的格）。返回本次统计。"""
        h, w = ng.grid.shape
        row, col, inb = self.cells_of(ng)
        st = {"cells": int(len(self.labels)), "in_crop": int(inb.sum()),
              "out_crop": int((~inb).sum()), "applied_free": 0, "applied_occ": 0,
              "skipped_known": 0, "blocked_walked": 0}
        applied = np.zeros(len(self.labels), bool)
        if inb.any():
            tr, tc = row[inb], col[inb]
            lab = self.labels[inb]
            live = ng.grid[tr, tc]
            can = live == UNK
            free_sel = can & (lab == L_FREE)
            occ_sel = can & (lab == L_OCC)
            st["skipped_known"] = int((~can).sum())
            if occ_sel.any():
                # 走过走廊 = 活的通行证据，先验障碍不许盖（gauge 残余误差下守走廊会困住自己）
                wm = ng.walked_mask(walked, 0.20) if walked else None
                if wm is not None:
                    blocked = occ_sel & wm[tr, tc]
                    st["blocked_walked"] = int(blocked.sum())
                    occ_sel = occ_sel & ~wm[tr, tc]
            st["applied_free"] = int(free_sel.sum())
            st["applied_occ"] = int(occ_sel.sum())
            # 先 FREE 后 OCC：同一格被先验多格覆盖时**障碍赢**（保守侧）。
            ng.grid[tr[free_sel], tc[free_sel]] = FREE
            ng.grid[tr[occ_sel], tc[occ_sel]] = OCC
            idx = np.flatnonzero(inb)
            applied[idx[free_sel | occ_sel]] = True
        self.applied_mask = applied           # 整块替换，读者只会看到完整的一份
        self.stat["applied"] = st
        return st


def project_prior(prior: PriorMap, R2: np.ndarray, t2: np.ndarray,
                  *, stat: dict[str, Any] | None = None) -> PriorOverlay:
    """世界系先验 → 会话帧视图。``R2/t2``：xy 面 gauge，``p_world = R2 @ p_sess + t2``。

    逆变换逐格心做：``p_sess = R2.T @ (p_world − t2)``（行向量写法 ``(P − t2) @ R2``）。
    """
    lab = prior.labels
    nz = np.argwhere(lab != L_UNK)
    if not len(nz):
        empty = np.zeros((0, 2), np.float64)
        return PriorOverlay(labels=np.zeros(0, np.uint8), xy_session=empty,
                            corners=empty, stat=dict(stat or {}))
    rows, cols = nz[:, 0], nz[:, 1]
    x = prior.origin_xy[0] + (cols + 0.5) * prior.res_m
    y = prior.origin_xy[1] + (rows + 0.5) * prior.res_m
    P = np.column_stack([x, y])
    r2 = np.asarray(R2, np.float64).reshape(2, 2)
    t = np.asarray(t2, np.float64).reshape(2)
    sess = (P - t) @ r2
    x0 = prior.origin_xy[0] + float(cols.min()) * prior.res_m
    x1 = prior.origin_xy[0] + (float(cols.max()) + 1.0) * prior.res_m
    y0 = prior.origin_xy[1] + float(rows.min()) * prior.res_m
    y1 = prior.origin_xy[1] + (float(rows.max()) + 1.0) * prior.res_m
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    corners = (box - t) @ r2
    return PriorOverlay(labels=lab[rows, cols].astype(np.uint8), xy_session=sess,
                        corners=corners, stat=dict(stat or {}))


def project_prior_tiles(prior: PriorMap, tiles: dict[tuple[int, int], dict[str, Any]],
                        assign_R2: np.ndarray, assign_t2: np.ndarray,
                        *, tile_size_m: float,
                        stat: dict[str, Any] | None = None) -> PriorOverlay:
    """**分片**投影：每片先验格用**它自己那一片**的 gauge（全局闸拒了之后的降级档）。

    * ``tiles``：``{(i, j): {"R2": (2,2), "t2": (2,)}}``，i/j 是会话帧 ``tile_size_m`` 网格下标；
    * ``assign_R2/t2``：**归属判定**用的变换（取全局 gauge）——只用来决定"这一格归哪一片"，
      分片是米级尺度，归属判定对 gauge 的亚米级误差不敏感；真正画图用该片自己的 gauge；
    * 没落在任何过闸片里的格**直接丢掉**（保持 unknown），绝不拿别的片的 gauge 顶替。

    返回的是**一份** ``PriorOverlay``（各片投影完再拼）⇒ ``apply_into`` / 图层 / 测试全不用改。
    """
    lab = prior.labels
    nz = np.argwhere(lab != L_UNK)
    if not len(nz) or not tiles:
        empty = np.zeros((0, 2), np.float64)
        return PriorOverlay(labels=np.zeros(0, np.uint8), xy_session=empty,
                            corners=empty, stat=dict(stat or {}))
    rows, cols = nz[:, 0], nz[:, 1]
    P = np.column_stack([prior.origin_xy[0] + (cols + 0.5) * prior.res_m,
                         prior.origin_xy[1] + (rows + 0.5) * prior.res_m])
    aR = np.asarray(assign_R2, np.float64).reshape(2, 2)
    at = np.asarray(assign_t2, np.float64).reshape(2)
    sess_assign = (P - at) @ aR                            # 粗投影：只用于决定归属哪一片
    size = float(tile_size_m)
    ij = np.floor(sess_assign / size).astype(np.int64) if size > 0 else np.zeros_like(
        sess_assign, dtype=np.int64)
    sess = np.empty_like(P)
    used = np.zeros(len(P), bool)
    for (ti, tj), tg in tiles.items():
        m = (ij[:, 0] == int(ti)) & (ij[:, 1] == int(tj))
        if not m.any():
            continue
        sess[m] = (P[m] - np.asarray(tg["t2"], np.float64).reshape(2)) @ np.asarray(
            tg["R2"], np.float64).reshape(2, 2)
        used[m] = True
    if not used.any():
        empty = np.zeros((0, 2), np.float64)
        return PriorOverlay(labels=np.zeros(0, np.uint8), xy_session=empty,
                            corners=empty, stat=dict(stat or {}))
    x0 = prior.origin_xy[0] + float(cols.min()) * prior.res_m
    x1 = prior.origin_xy[0] + (float(cols.max()) + 1.0) * prior.res_m
    y0 = prior.origin_xy[1] + float(rows.min()) * prior.res_m
    y1 = prior.origin_xy[1] + (float(rows.max()) + 1.0) * prior.res_m
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    keep = np.flatnonzero(used)
    return PriorOverlay(labels=lab[rows, cols].astype(np.uint8)[keep], xy_session=sess[keep],
                        corners=(box - at) @ aR, stat=dict(stat or {}))


class PriorConsumer:
    """在线先验消费（**只允许建图线程**调 ``add_constraints`` / ``note_map_moved`` / ``overlay``）。

    ``status`` 可能来自 HTTP 线程，只读一份快照（小锁保护）。
    依赖注入（全部来自 nav_online，便于测试用假的）：
      * ``pose_of(k)``：本会话关键帧在当前地图系的位姿（回环修正后），没有返回 None；
      * ``pose_of_old(sid, kf)``：旧关键帧在**索引快照**系（= 其位姿表所在坐标系）的 (R, xy)；
      * ``table_base(sid)``：旧会话位姿表所在坐标系的根（merged → base_sid；raw → 自己）。
    """

    def __init__(self, cfg: PriorConfig, world_dir: str | Path, *, world_scale: float,
                 pose_of: Callable[[int], np.ndarray | None],
                 pose_of_old: Callable[[str, int], tuple[np.ndarray, np.ndarray] | None] | None = None,
                 table_base: Callable[[str], str | None] | None = None) -> None:
        self.cfg = cfg
        self.world_dir = Path(world_dir)
        self.world_scale = float(world_scale)
        self._pose_of = pose_of
        self._pose_of_old = pose_of_old
        self._table_base = table_base
        self._cons: list[dict[str, Any]] = []
        self._bases: dict[str, str | None] = {}
        self._dirty = True
        self._prior: PriorMap | None = None
        self._prior_reason = "loading"
        self._prior_tried = False
        self._gauge: dict[str, Any] | None = None
        self._split: dict[str, Any] | None = None     # 半样本交叉验证的差异统计
        self._core: list[dict[str, Any]] = []         # 本次估计的内点核心（= 真参与 gauge 的约束）
        self._core_span: float | None = None          # 核心在命中区间里覆盖的里程占比
        self._tiles: dict[str, Any] = {}              # 分片降级档摘要（片数 / 过闸数 / 拒绝原因计数）
        self._overlay: PriorOverlay | None = None
        self._lock = threading.Lock()
        self._st: dict[str, Any] = {"enabled": True, "state": "idle", "n_constraints": 0}

    # ---- 建图线程入口 ----
    def add_constraints(self, recs: list[dict[str, Any]]) -> None:
        """新确认的跨会话约束（``XSessionTracker.on_keyframe`` 的返回值）进池。"""
        if not recs:
            return
        self._cons.extend(recs)
        if len(self._cons) > 4096:               # 约束池上限：只留最近的（gauge 用不到陈年尾巴）
            del self._cons[:len(self._cons) - 4096]
        self._dirty = True

    def note_map_moved(self) -> None:
        """回环修正了全表位姿 ⇒ 会话帧内容变了，gauge 必须重估。"""
        self._dirty = True

    def overlay(self) -> PriorOverlay | None:
        """给 ``NavSession.compute`` 的钩子：返回当前可用的会话帧先验视图（或 None）。

        没有先验文件 / gauge 过不了闸 / 坐标系不同根 ⇒ None（一格都不注入）。
        """
        if self._dirty:
            self._dirty = False
            try:
                self._refresh()
            except Exception as exc:  # noqa: BLE001 - 先验坏了不许拖垮建图
                self._overlay = None
                self._publish(f"error:{type(exc).__name__}", None, None)
        return self._overlay

    # ---- 内部（建图线程）----
    def _refresh(self) -> None:
        # 先验文件只在会话内读一次（`_prior_tried`）：离线固化是会话之间的操作，
        # 会话跑着换先验会让"这一场到底用了哪张图"无法复述——宁可整场不用。
        if self._prior is None and not self._prior_tried:
            self._prior_tried = True
            p, why = load_prior(self.world_dir)
            if p is not None and abs(p.world_scale - self.world_scale) > 1e-6:
                # 换过 avatar / 身高 ⇒ 世界米尺度变了，会话帧与先验不同尺度，拒绝消费。
                p, why = None, "world_scale_mismatch"
            self._prior, self._prior_reason = p, (why or "")
        prior = self._prior
        if prior is None:
            # 没有先验文件 / 尺度对不上 / 读坏：一格都不注入，但 gauge 照算（供现场核对）。
            aligned = self._aligned(self._usable(None))
            self._gauge = self._estimate(aligned)
            self._set_core(aligned, self._gauge)
            self._overlay = None
            self._publish(self._prior_reason or "no_prior", self._gauge, None)
            return
        cons = self._usable(prior)
        # 池里有约束、但没一条与先验同根 ⇒ 坐标系对不上：如实报，不做任何估算（绝不混 gauge）。
        mismatch = self._table_base is not None and bool(self._cons) and not cons
        aligned = self._aligned(cons)
        g = None if mismatch else self._estimate(aligned)
        self._gauge = g
        self._split = None
        core = self._set_core(aligned, g)
        reason = "frame_mismatch" if mismatch else self._gate(g, aligned, core)
        # 半样本交叉验证：**总是算**（数字进 status，现场能看见"数据撑不撑得起 gauge"），
        # 但只在残差闸已放行时才让它定生死 —— 残差都不合格时原因要报更根本的那条。
        if not mismatch and g is not None and self._table_base is not None:
            self._split, split_reason = self._cross_check(core)
            if not reason:
                reason = split_reason
        self._overlay = None
        if reason:
            # 降级档：全局一个刚体对不上整张图时，改问"哪几片对得上"（见 _tiles_fallback）。
            # ⚠️ 坐标系不同根（mismatch）时不许走这条路 —— 那是"混 gauge"，不是"局部不准"。
            if not mismatch and self._tiles_fallback(prior, aligned, g):
                return
            self._publish(reason, g, prior)
            return
        self._tiles = {}
        r2 = np.asarray(g["R_G"], np.float64)[:2, :2]
        t2 = np.asarray(g["t_G"], np.float64)[:2]
        stat = {"base_sid": prior.base_sid, "built_wall": prior.meta.get("built_wall"),
                "path": str(prior.path), "cells_free": int((prior.labels == L_FREE).sum()),
                "cells_occ": int((prior.labels == L_OCC).sum())}
        self._overlay = self._maybe_strip_free(project_prior(prior, r2, t2, stat=stat))
        self._publish("ok", g, prior)

    def _tiles_fallback(self, prior: PriorMap, aligned: list[dict[str, Any]],
                        g: dict[str, Any]) -> bool:
        """全局闸拒了 ⇒ 改问"哪几片能对上"。过闸的那几片才注入，其余保持 unknown。

        返回 True = 已按分片注入（调用方直接 return）。**一格都过不了闸就返回 False**，
        交给调用方按原样报全局的拒绝原因 —— 分片是降级档，不是"总能画点什么"的借口。
        """
        self._tiles = {}
        if not self.cfg.tiles_enabled or g is None:
            return False
        if len(aligned) < int(self.cfg.tile_min_constraints):
            return False
        xy = self._new_xy(aligned)
        if xy is None or len(xy) != len(aligned):
            return False
        tiles, why = self._tile_gauges(aligned, xy)
        self._tiles = {"size_m": float(self.cfg.tile_size_m), "n": len(tiles) + sum(why.values()),
                       "ok": len(tiles), "rejects": why}
        if not tiles:
            return False
        r2 = np.asarray(g["R_G"], np.float64)[:2, :2]
        t2 = np.asarray(g["t_G"], np.float64)[:2]
        stat = {"base_sid": prior.base_sid, "built_wall": prior.meta.get("built_wall"),
                "path": str(prior.path), "cells_free": int((prior.labels == L_FREE).sum()),
                "cells_occ": int((prior.labels == L_OCC).sum()),
                "tiles": dict(self._tiles)}
        self._overlay = self._maybe_strip_free(
            project_prior_tiles(prior, tiles, r2, t2,
                                tile_size_m=float(self.cfg.tile_size_m), stat=stat))
        self._publish("ok_tiles", g, prior)
        return True

    def _maybe_strip_free(self, ovl: PriorOverlay | None) -> PriorOverlay | None:
        """``inject_free=False``（默认）⇒ 先验只注障碍，FREE 格降为 unknown（不画）。"""
        if ovl is None or self.cfg.inject_free:
            return ovl
        drop = ovl.labels == L_FREE
        if drop.any():
            ovl.labels[drop] = L_UNK
            ovl.stat["free_dropped"] = int(drop.sum())
        return ovl

    def _usable(self, prior: PriorMap | None) -> list[dict[str, Any]]:
        """只留"旧会话表坐标系根 == 先验坐标系根"的约束——绝不混 gauge。

        先验还没载入（或没有 table_base 注入，如测试）时不筛，估算照做（供 status 报数）。
        """
        if prior is None or self._table_base is None:
            return list(self._cons)
        out = []
        for c in self._cons:
            sid = str(c.get("old_sid", ""))
            if sid not in self._bases:
                try:
                    self._bases[sid] = self._table_base(sid)
                except Exception:  # noqa: BLE001 - 查不到根 = 不可用，不猜
                    self._bases[sid] = None
            if self._bases[sid] == prior.base_sid:
                out.append(c)
        return out

    def _aligned(self, cons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """只留"两侧位姿都查得到"的约束 —— ``_estimate`` / ``_cross_check`` / 核心掩码对齐用同一份，
        三处各筛一遍的话 ``g["inlier"]`` 就和约束列表错位了。"""
        out = []
        for c in cons:
            if self._pose_of(int(c["new_kf"])) is None:
                continue
            if self._pose_of_old is None or self._pose_of_old(
                    str(c["old_sid"]), int(c["old_kf"])) is None:
                continue
            out.append(c)
        return out

    def _estimate(self, cons: list[dict[str, Any]]) -> dict[str, Any] | None:
        """约束池 + 当前位姿 → gauge（``estimate_gauge``，口径与会话末采纳一致）。

        ``cons`` 必须是 ``_aligned`` 过的那份（每条都能查到两侧位姿）。
        """
        if len(cons) < 3:                                     # estimate_gauge 的硬下限
            return None
        R_old, p_old, R_ab, t_ab, R_new, p_new, w = [], [], [], [], [], [], []
        for c in cons:
            T_new = self._pose_of(int(c["new_kf"]))
            R_o, xy_o = self._pose_of_old(str(c["old_sid"]), int(c["old_kf"]))
            R_old.append(np.asarray(R_o, np.float64))
            # ⚠️ z 一律置 0：索引只存 xy，先验也是纯 xy 图；掺半个 3D 只会让 t_G 的 z
            # 分量去吸收"混合口径"，对 xy 面映射是纯污染（本模块只消费 R_G/t_G 的 xy 块）。
            p_old.append([float(xy_o[0]), float(xy_o[1]), 0.0])
            R_ab.append(np.asarray(c["R_ab"], np.float64))
            t_ab.append(np.asarray(c["t_ab"], np.float64))
            T = np.asarray(T_new, np.float64)
            R_new.append(T[:3, :3])
            p_new.append([float(T[0, 3]), float(T[1, 3]), 0.0])
            w.append(max(float(c.get("inliers", 1.0)), 1e-6))
        if len(p_new) < 3:
            return None
        return estimate_gauge(np.array(R_old), np.array(p_old), np.array(R_ab), np.array(t_ab),
                              np.array(R_new), np.array(p_new), np.array(w))

    def _set_core(self, aligned: list[dict[str, Any]], g: dict[str, Any] | None) -> list[dict[str, Any]]:
        """把这次估计的**内点核心**（= 真正参与 gauge 的那批约束）记下来给闸门/status 用。"""
        if g is None:
            self._core, self._core_span = [], None
            return self._core
        self._core = [c for c, k in zip(aligned, np.asarray(g["inlier"], bool).tolist()) if k]
        self._core_span = self._core_span_frac(aligned, self._core)
        return self._core

    @staticmethod
    def _core_span_frac(pool: list[dict[str, Any]], core: list[dict[str, Any]]) -> float | None:
        """核心在"命中区间"里覆盖的里程占比（核心 ``new_dist_m`` 跨度 / 全池跨度）。

        ``new_dist_m`` 缺失（老的 jsonl / 测试用假约束）⇒ None = 这条闸不适用（不猜）。
        """
        try:
            all_d = [float(c["new_dist_m"]) for c in pool]
            core_d = [float(c["new_dist_m"]) for c in core]
        except (KeyError, TypeError, ValueError):
            return None
        if len(all_d) < 3 or len(core_d) < 2:
            return None
        span = max(all_d) - min(all_d)
        if span <= 1e-6:
            return None
        return (max(core_d) - min(core_d)) / span

    def _gate(self, g: dict[str, Any] | None, pool: list[dict[str, Any]],
              core: list[dict[str, Any]]) -> str:
        """质量闸：返回空串 = 放行，否则是拒绝原因（进 status）。

        残差（``pos_med``/``rot_med``）本来就只统计内点；占比、跨度、前后半也一律只谈核心——
        外点已经被 IRLS 剔掉，不该再否决一次（见 ``PriorConfig.min_inlier_frac`` 的说明）。
        """
        c = self.cfg
        n = len(pool)
        if n < int(c.min_constraints):
            return "gauge_too_few"
        if g is None:
            return "gauge_degenerate"
        if g["n_inlier"] < int(c.min_inlier):
            return "gauge_few_inliers"
        if g["n_inlier"] < float(c.min_inlier_frac) * max(n, 1):
            return "gauge_inlier_frac"
        if g["pos_med"] > float(c.max_pos_med_m):
            return "gauge_pos_residual"
        if g["rot_med"] > float(c.max_rot_med_deg):
            return "gauge_rot_residual"
        if self._core_span is not None and self._core_span < float(c.min_core_span_frac):
            return "gauge_core_local"
        return ""

    def _cross_check(self, core: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
        """前后半交叉验证（**只在内点核心上做**）：``(两组差异统计 | None, 拒绝原因)``。

        **按时间前后切**（不是交错取），比较两半各估出的 gauge 把同一批新帧映射到哪儿：
        位置差中位 ≤ ``max_split_m``、yaw 差 ≤ ``max_split_yaw_deg`` 才放行。

        ⚠️ 为什么必须"前后切"而不是交错取（2026-10-06 live 教训）：那场的问题不是噪声，而是
        **会话帧相对世界在缓慢转动** —— gauge 的 yaw 随约束积累从 2.4° 漂到 8.6°。同一段时间
        里的两组子集**都偏同样的 6°**、交错对比照样一致（实测 split 0.375 m / 2.3°，放行），
        于是先验按错 6° 的变换画了满屏废图；而前后两半一比就差 6°，立刻露馅。
        ⇒ 这条闸测的是"**gauge 跟时间有关**"，正是漂移/形变的指纹。

        事故场在只喂 16 条时核心 = 全池，这道闸照旧拦下（split 1.007 m / 4.59°）；核心被外点
        稀释的场（052927）则不再被"外点自己造成的漂移"误杀 —— 那是两种不同的病。
        """
        half = len(core) // 2
        ga, gb = self._estimate(core[:half]), self._estimate(core[half:])
        if ga is None or gb is None:
            return None, "gauge_unstable"
        xy = self._new_xy(core)
        if xy is None or len(xy) < 3:
            return None, "gauge_unstable"
        pa = xy @ np.asarray(ga["R_G"], np.float64)[:2, :2].T + np.asarray(ga["t_G"], np.float64)[:2]
        pb = xy @ np.asarray(gb["R_G"], np.float64)[:2, :2].T + np.asarray(gb["t_G"], np.float64)[:2]
        split_m = float(np.median(np.linalg.norm(pa - pb, axis=1)))
        split_yaw = self._yaw_deg(ga) - self._yaw_deg(gb)
        split_yaw = abs((split_yaw + 180.0) % 360.0 - 180.0)
        st = {"split_m": round(split_m, 3), "split_yaw_deg": round(split_yaw, 2), "n_split": len(xy)}
        if split_m > float(self.cfg.max_split_m) or split_yaw > float(self.cfg.max_split_yaw_deg):
            return st, "gauge_unstable"
        return st, ""

    def _tile_gauges(self, aligned: list[dict[str, Any]], xy: np.ndarray
                     ) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, int]]:
        """**分片**估 gauge：每片只用自己附近的约束，各自过闸。返回 ``(过闸片, 拒绝原因计数)``。

        只在**全局闸已拒**时调用（降级档）。片心 = 约束所在片；每片取半径
        ``tile_neighbor_m`` 内的约束（半径 > 片边长 ⇒ 相邻片的约束集高度重叠 ⇒ 片间 gauge
        连续，不会在片界上把地图撕开）。

        片级闸只留"这一片撑不撑得起一个刚体"的三条（内点数 / 位置残差 / 旋转残差），
        **不做核心跨度**（局部本来就是局部）；约束够多时补做前后半交叉验证
        （会话帧随时间长转那种病在片内一样能被抓到）。
        """
        c = self.cfg
        size = float(c.tile_size_m)
        rad = float(c.tile_neighbor_m)
        out: dict[tuple[int, int], dict[str, Any]] = {}
        why: dict[str, int] = {}
        if size <= 0 or rad <= 0 or len(aligned) < 3:
            return out, why
        ij = np.floor(xy / size).astype(np.int64)
        for key in sorted({(int(a), int(b)) for a, b in ij.tolist()}):
            cx, cy = (key[0] + 0.5) * size, (key[1] + 0.5) * size
            sel = (np.abs(xy[:, 0] - cx) <= rad) & (np.abs(xy[:, 1] - cy) <= rad)
            sub = [cc for cc, k in zip(aligned, sel.tolist()) if k]
            if len(sub) < int(c.tile_min_constraints):
                why["few"] = why.get("few", 0) + 1
                continue
            g = self._estimate(sub)
            if g is None:
                why["degenerate"] = why.get("degenerate", 0) + 1
                continue
            if g["n_inlier"] < int(c.tile_min_inlier):
                why["few_inliers"] = why.get("few_inliers", 0) + 1
                continue
            if g["pos_med"] > float(c.max_pos_med_m):
                why["pos_residual"] = why.get("pos_residual", 0) + 1
                continue
            if g["rot_med"] > float(c.max_rot_med_deg):
                why["rot_residual"] = why.get("rot_residual", 0) + 1
                continue
            if len(sub) >= int(c.tile_split_min):
                core = [cc for cc, k in zip(sub, np.asarray(g["inlier"], bool).tolist()) if k]
                _, sreason = self._cross_check(core)
                if sreason:
                    why["unstable"] = why.get("unstable", 0) + 1
                    continue
            out[key] = {"R2": np.asarray(g["R_G"], np.float64)[:2, :2],
                        "t2": np.asarray(g["t_G"], np.float64)[:2]}
        return out, why

    @staticmethod
    def _yaw_deg(g: dict[str, Any]) -> float:
        R = np.asarray(g["R_G"], np.float64)
        return math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))

    def _new_xy(self, cons: list[dict[str, Any]]) -> np.ndarray | None:
        """这批约束里"新帧在会话帧"的 xy（与 ``_estimate`` 用的是同一批，顺序一致）。"""
        out = []
        for c in cons:
            T = self._pose_of(int(c["new_kf"]))
            if T is None:
                continue
            T = np.asarray(T, np.float64)
            out.append([float(T[0, 3]), float(T[1, 3])])
        return np.asarray(out, np.float64) if out else None

    def _publish(self, state: str, g: dict[str, Any] | None, prior: PriorMap | None) -> None:
        st: dict[str, Any] = {
            "enabled": True, "state": state, "n_constraints": len(self._cons),
            "world_scale": self.world_scale,
            "gauge": None if g is None else {
                "n": int(g["n"]), "n_inlier": int(g["n_inlier"]),
                "pos_med_m": round(float(g["pos_med"]), 3),
                "pos_p90_m": round(float(g["pos_p90"]), 3),
                "rot_med_deg": round(float(g["rot_med"]), 2),
                # R_dev 只是 |gauge 的 yaw|（两场会话共享 SteamVR 锚时 ≈0），信息量用，不是闸
                "r_dev_deg": round(float(g["R_dev_deg"]), 2)},
            # 内点核心（真正参与 gauge 的那批约束）：条数 + 覆盖命中区间的里程占比
            "core": {"n": len(self._core),
                     "span_frac": (None if self._core_span is None else round(self._core_span, 3))},
            # 半样本差异（两组子集各估一份 gauge 差多远）：池化残差好看但这里大 = 数据撑不起 gauge
            "split": (None if self._split is None else
                      {"m": self._split.get("split_m"), "yaw_deg": self._split.get("split_yaw_deg"),
                       "n": self._split.get("n_split")}),
            # 分片降级档（全局拒了才用）：片数 / 过闸片数 / 各片被拒的原因计数
            "tiles": dict(self._tiles),
            # inject_free=False 时被剥掉的 free 格数（现场要能看见"free 没注"这件事）
            "free_dropped": (None if self._overlay is None
                             else int(self._overlay.stat.get("free_dropped", 0))),
            "prior": None if prior is None else {
                "base_sid": prior.base_sid, "path": str(prior.path),
                "built_wall": prior.meta.get("built_wall"),
                "cells_free": int((prior.labels == L_FREE).sum()),
                "cells_occ": int((prior.labels == L_OCC).sum())},
            "reason": self._prior_reason if prior is None else "",
        }
        with self._lock:
            self._st = st

    def status(self) -> dict[str, Any]:
        with self._lock:
            st = dict(self._st)
        # apply_into 每次栅格化都会重写，status 拿最近一份（读多写一，值都是不可变小对象）。
        ovl = self._overlay
        if ovl is not None and st.get("state") in ("ok", "ok_tiles"):
            st["applied"] = dict(ovl.stat.get("applied") or {})
        return st

    def view(self) -> tuple["PriorOverlay | None", dict[str, Any]]:
        """HTTP/显示线程的只读入口：``(当前投影视图 | None, status)``。

        读的是**引用**（不在锁里）——``_overlay`` 与它的 ``applied_mask`` 在构造后都是
        整块替换、绝不就地改写，所以读者要么拿到上一份完整的、要么拿到新的一份，不会
        看到半个对象。视图只用于画图，**不许**拿它去改栅格（那是建图线程的 apply_into）。
        """
        return self._overlay, self.status()