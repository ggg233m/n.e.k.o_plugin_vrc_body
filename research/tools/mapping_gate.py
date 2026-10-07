# -*- coding: utf-8 -*-
r"""建图 golden 门：把"建图质量"变成一个会失败的门，而不是一个感觉。

    python research/tools/mapping_gate.py              # 比对基线，漂移则 exit 1
    python research/tools/mapping_gate.py --update     # 重写基线（**必须**人工看过 diff 再提交）
    python research/tools/mapping_gate.py --strict     # 录制缺失也算失败

放在 ``research/tools/`` 而不是根目录的 ``tools/``，有两个理由，都不是洁癖：

* ``tests/test_research_isolation.py`` 明确断言根目录 ``tools/`` **不该存在**，
  研究工具应并入 ``research/``。往那儿加文件等于把一个已知红灯越弄越红。
* ``pyproject.toml`` 的 ``[tool.neko.build] exclude_dirs`` 含 ``research/``
  **不含** ``tools/`` ⇒ 放在 ``tools/`` 会被打进**发行包**。
  本门是开发期校验，不该出现在用户的安装目录里。

为什么要有这个门
----------------
建图侧的改动一直在发生（``q_tiers`` 分层、``cam_h`` 滞回、射线清除、``ray_clear``/``ray_exempt``），
而每次改动的判据都是"量到的数字变好了"。这套数字散在 ``tools/q_tier_ab.py`` 等七八个
脚本里，口径各自略有不同，**没有一个能回答"这次改动有没有动到不该动的东西"**。

本门把一条真实录制喂进**在线 ``KeyframeGridMapper`` 本体**（不另写一份分类器，
否则离线实验固化的规则与在线实现漂移也无处藏——这是 ``q_tier_ab.py`` 的原话，
本门沿用），然后冻结五样东西：

1. **栅格指纹**：三态栅格的 SHA-256。任何一格的分类变化都会被抓到。
2. **几何与拓扑**：``NavGrid.build()`` 的全部返回量。
3. **走廊障碍**：轨迹两侧 0.5 m 内还剩几个障碍格——这是"能不能走过去"的直接度量
   （沿用 ``q_tier_ab.corridor_blockers`` 的口径，两者必须可比）。
4. **环内比例**：R∈{1.5,2.5,3.0} 的已知障碍%与 free%。**单独看会奖励"把真墙也删掉"**，
   所以必须和走廊障碍并排读。
5. **A\* 探针**：从轨迹上取固定下标的一组 (起点, 终点)，逐个跑 ``NavGrid.plan()``，
   冻结 ``accepted / reason / 折点数 / 折点指纹``。

第 5 项是本门与既有脚本最大的不同：既有脚本量的是**图**，
本门量的是**图被规划器实际消费之后的结果**。后续把地形代价接进 ``_cost()`` 时，
唯一该先问的问题是"路径变了吗"，而这件事只能从规划器输出上量。
基线文件 ``tools/mapping_gate_baseline.json`` 只存**指标与指纹**，不存栅格本身
（栅格几百 KB × 三场，进了版本库没人会去看 diff）。栅格要对比用
``--dump-grid`` 落到临时文件。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]     # research/tools/ → 仓库根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_grid import FREE, OCC, UNK, NavGrid          # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))          # research/tools/
from trail_geom import trail_cells, trail_mask                    # noqa: E402

BASELINE = Path(__file__).parent / "mapping_gate_baseline.json"
RECS = ("20260929_045615", "20261001_044120", "20261001_044153")
RINGS = (1.5, 2.5, 3.0)

# A* 探针的取点下标。写死是为了可复现；**改了这两个数基线必然失效**，
# 那是故意的——改它们等于换了一道题，必须重跑基线而不是绕过门。
PROBE_STARTS = (0, 40, 80, 120)
PROBE_GOAL_OFFSET = 60


def _sha(a: np.ndarray) -> str:
    """数组内容的 SHA-256。``tobytes()`` 前必须保证 C 连续且 dtype 固定，
    否则同一张图在不同机器/版本上会算出不同摘要，门就成了随机门。"""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_events(rec: Path) -> list[dict]:
    lines = (rec / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def replay(rec: Path, cfg: MapperConfig):
    """按录制事件顺序把关键帧喂进 mapper，末尾做一次最终栅格化。

    逐行沿用 ``tools/q_tier_ab.replay``：包括 ``refresh`` 时的 ``drop_points(k-1)``
    （原地不动补帧，上一帧让出）与在线同口径。**这里不要"顺手优化"**——
    这段顺序是复现线上行为的一部分，改了它门量的就不是线上的东西了。
    """
    events = load_events(rec)
    fp = rec / "final_poses.npz"
    if fp.exists():
        z = np.load(fp)
        pose_of = {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
    else:
        pose_of = {}
    cache: dict[int, np.ndarray] = {}

    def kf(k: int):
        if k not in cache:
            cache.clear()
            cache[k] = np.load(rec / "kf" / f"{k:06d}.npz")
        return cache[k]

    def pose(k: int) -> np.ndarray | None:
        if k in pose_of:
            return pose_of[k]
        try:
            return np.asarray(kf(k)["T_map"], np.float64)
        except (FileNotFoundError, KeyError, OSError):
            return None

    m = KeyframeGridMapper(cfg)
    trail: list[np.ndarray] = []
    n_refresh = 0
    for e in events:
        kind = e.get("kind")
        if kind == "kf":
            k = int(e["k"])
            T = pose(k)
            if T is None:
                continue
            pts = kf(k)["pts"].astype(np.float32)
            if e.get("refresh"):
                m.drop_points(k - 1)
                n_refresh += 1
            if k in m._pts:
                m.drop_points(k)
            m.add_keyframe(k, pts, T, e.get("osc"))
            trail.append(T[:2, 3])
        elif kind == "trail":
            k = int(e["k"])
            T = pose(k)
            if T is None:
                continue
            m.add_trail(k, T @ np.asarray(e["T_dr"], np.float64), e.get("osc"))
    ng = m.rasterize()
    return ng, trail, {"n_kf": len(m), "n_refresh": n_refresh}, m


def _trail_mask(ng: NavGrid, trail: list[np.ndarray], radius_m: float) -> np.ndarray:
    """轨迹附近的格。**实现已迁到 ``trail_geom.trail_mask``**（唯一实现，不许再各推一遍）。

    ⚠️ 2026-10-08 修掉一个一直量错地方的 bug：本函数以前写的是
    ``cv2.circle(m, (int((p[1]-oy)/r), int((p[0]-ox)/r)))`` —— 把 x 当成了行、又漏掉
    ``NavGrid`` 的上下翻转（行轴是 y 且自下而上，见 ``nav_grid.to_cell``）。
    后果不是报错，是**数了一片镜像区域**：044153 上与正确掩码的 IoU = 0.090，
    落在掩码里的障碍格 505 → 899。下面这几行保留成薄壳，只为让调用点不改名。
    """
    return trail_mask(ng, trail, radius_m)


def astar_probes(ng: NavGrid, trail: list[np.ndarray]) -> list[dict]:
    """固定的 (起点, 终点) 探针，冻结规划器**实际输出**。

    坐标口径：``NavGrid`` 的世界坐标是"追踪米 × world_scale"，而 trail 存的是
    原始 T_map 平面坐标，所以**必须先乘 world_scale** 再喂给 ``plan()``。
    与 ``q_tier_ab.render`` 同一个乘法。
    """
    out: list[dict] = []
    ws = ng.meta.world_scale
    n = len(trail)
    # 探针跨度按轨迹长度自适应。写死 PROBE_GOAL_OFFSET 的后果是：短录制
    # （044120 只有 40 多个轨迹点）会得到 0 个探针，那一场等于**完全没被门保护**
    # 却照样显示 [ok]。宁可探针位置随录制变，也不要出现"0 探针也算通过"。
    off = PROBE_GOAL_OFFSET if n >= PROBE_GOAL_OFFSET * 3 else max(2, n // 4)
    last = n - off - 1
    if last < 0:
        return out                      # 轨迹太短，连一个探针都摆不下
    stride = max(1, last // max(1, len(PROBE_STARTS)))
    for si in PROBE_STARTS:
        # 夹到 last 而不是 n-1：夹到 n-1 会让 gi=si+off 越界，
        # 探针被 continue 掉，于是长录制反而只剩 1 个探针（实测 397 点→1 个）。
        si = min(si * stride, last)
        gi = si + off
        if gi >= n:
            continue
        s = [float(trail[si][0] * ws), float(trail[si][1] * ws)]
        g = [float(trail[gi][0] * ws), float(trail[gi][1] * ws)]
        c = ng.plan(s, g)
        item = {"start": si, "goal": gi, "accepted": c.accepted, "reason": c.reason,
                "n_wp": len(c.waypoints_xy_m)}
        if c.accepted:
            pts = np.asarray(c.waypoints_xy_m, np.float64)
            item["len_m"] = round(c.length_m, 3)
            item["corridor_half_width_m"] = round(c.corridor_half_width_m, 4)
            item["start_snapped_m"] = round(c.start_snapped_m, 4)
            # 折点量化到 1 mm 再哈希：double 的最低位在跨平台重算时不可靠，
            # 直接哈希会把纯浮点末位抖动报成"路径变了"。
            item["wp_sha"] = _sha(np.round(pts * 1000.0).astype(np.int64))
        out.append(item)
    return out


def occ_shape(occ: np.ndarray, res_m: float, min_cells: int = 20) -> dict:
    """障碍的**形状**：边界格占比 + 局部厚度中位。回答"墙有没有变胖"。

    为什么门需要这一项：``corridor_blockers``、环比例、``build()`` 量的都是**多少格**，
    对"障碍被抹厚"几乎不敏感——墙从 2 格胖到 6 格时，只要它仍然堵在路上，格数变化会被
    "清掉了一些孤岛"抵消掉，指标看起来还可能变好。而墙厚是**能不能贴着走**的直接物理量。

    两个量的分工（合成对照见 ``_shape_selftest``）：
      * ``boundary_frac_pct``：4 邻域有缺失的格占比。薄墙/环 → ~100%；实心块 → 很低。
      * ``thickness_p50_m``：``2 × 中位(距离变换)``。薄墙 → 就是它自己的厚度；实心块 → 直径量级。

    ⚠️ 只用**≥min_cells 格的连通块**：碎屑（单帧量化打出的几个格）本身就是"全边界、厚度 0"，
    会把统计拉向"薄"，正好抵消掉我们要抓的"变胖"。**别用 PCA 伸长率代替**：房间四壁是一个
    闭环，主轴长宽比 ≈1，与实心块无法区分（本仓踩过，见 C 记录）。

    ⚠️ 该项对**分割阈值**敏感：``occ`` 必须是三态栅格里的 GiST OCC，不是原始票数图。
    """
    out = {"cells": int(occ.sum()), "components": 0, "in_blocks": 0,
           "boundary_frac_pct": 0.0, "thickness_p50_m": 0.0}
    if not out["cells"]:
        return out
    n, cc = cv2.connectedComponents(occ.astype(np.uint8), connectivity=8)
    out["components"] = int(n - 1)
    sizes = np.bincount(cc.ravel())
    keep = np.zeros_like(occ, bool)
    for i in np.flatnonzero(sizes >= min_cells):
        if i != 0:
            keep |= (cc == i)
    out["in_blocks"] = int(keep.sum())
    if not keep.any():
        return out
    k4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    nb = cv2.filter2D(keep.astype(np.uint8), -1, k4, borderType=cv2.BORDER_CONSTANT)
    out["boundary_frac_pct"] = round(100.0 * float((keep & (nb < 5)).sum()) / float(keep.sum()), 2)
    dt = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5)
    out["thickness_p50_m"] = round(2.0 * float(np.median(dt[keep])) * float(res_m), 4)
    return out


def _shape_selftest() -> None:
    """合成对照自检：**必须**能把薄墙/环与实心块分开，否则这个指标没有分辨力。

    本仓已有一次"自检的两个分支算同一个表达式"的教训——那种自检比没有还糟。所以这里
    断言的是**两组数字必须分开**，而不是"跑完不报错"。
    """
    wall = np.zeros((300, 300), bool)
    wall[150:152, 30:270] = True                       # 16 m 长、0.2 m 厚
    ring = np.zeros((300, 300), bool)
    ring[50:250, 50:52] = ring[50:250, 248:250] = True
    ring[50:52, 50:250] = ring[248:250, 50:250] = True  # 20×20 m 房间的四壁
    rr, cc_ = np.ogrid[:300, :300]
    disc = ((rr - 150) ** 2 + (cc_ - 150) ** 2) <= 45 ** 2
    w, r_, d = (occ_shape(m, 0.1) for m in (wall, ring, disc))
    assert abs(w["thickness_p50_m"] - 0.20) < 0.02, w
    assert abs(r_["thickness_p50_m"] - 0.20) < 0.02, r_
    assert w["boundary_frac_pct"] > 95.0 and r_["boundary_frac_pct"] > 95.0, (w, r_)
    assert d["thickness_p50_m"] > 2.0, d                # 实心盘直径 9 m，中位厚度 ~2.6 m
    assert d["boundary_frac_pct"] < 15.0, d
    # 分辨力：薄墙与实心块在两个量上都必须分开（同一个数就不是分辨力）
    assert w["thickness_p50_m"] < 0.5 < d["thickness_p50_m"]
    assert d["boundary_frac_pct"] < 50.0 < w["boundary_frac_pct"]
    # 碎屑不能主导：加一堆 1 格孤点，结论必须不变
    noisy = wall.copy()
    noisy[10, 10] = noisy[20, 20] = noisy[30, 30] = True
    assert abs(occ_shape(noisy, 0.1)["thickness_p50_m"] - 0.20) < 0.02
    print("occ_shape self-test: PASS  "
          f"(wall {w['boundary_frac_pct']}%/{w['thickness_p50_m']}m, "
          f"ring {r_['boundary_frac_pct']}%/{r_['thickness_p50_m']}m, "
          f"disc {d['boundary_frac_pct']}%/{d['thickness_p50_m']}m)")


def measure(ng: NavGrid, trail: list[np.ndarray], mapper: KeyframeGridMapper) -> dict:
    g = ng.grid
    occ, free = g == OCC, g == FREE
    out: dict = {
        "shape": list(g.shape),
        "resolution_m": ng.meta.resolution_m,
        "origin_xy_m": [round(v, 6) for v in ng.meta.origin_xy_m],
        "world_scale": ng.meta.world_scale,
        "grid_sha256": _sha(g),
        "occ_cells": int(occ.sum()),
        "free_cells": int(free.sum()),
        "unk_cells": int((g == UNK).sum()),
    }
    # walked 必须用 mapper 自己的 ``walked()``，不能拿 trail 顶替：
    # 它带 OSC 位姿跳变门控（gate_m/gate_frac），会把瞬移段切掉，而 trail 是原样点位。
    # NavGrid.build 会把 walked 无条件算作可走（前提是该格不是障碍），
    # 传空列表量出来的"可走中心面积"跟规划器真正吃的东西不是一回事。
    build = ng.build(radius_m=0.25, walked=mapper.walked())
    out["build"] = {k: build[k] for k in sorted(build)}
    out["corridor_blockers"] = int((_trail_mask(ng, trail, 0.5) & occ).sum())
    out["occ_shape"] = occ_shape(occ, ng.meta.resolution_m)
    for R in RINGS:
        ring = _trail_mask(ng, trail, R)
        n_o, n_f, n_all = int((occ & ring).sum()), int((free & ring).sum()), int(ring.sum())
        out[f"R{R}"] = {"known_obstacle_pct": round(100.0 * n_o / max(n_o + n_f, 1), 3),
                        "free_pct": round(100.0 * n_f / max(n_all, 1), 3)}
    out["astar_probes"] = astar_probes(ng, trail)
    return out


def recorded_config(rec: Path) -> tuple[MapperConfig, dict]:
    """按**录制当时**的配置回放，而不是当前默认值。

    为什么必须这样：``meta.json`` 里的 ``config.mapper`` 是采集时的快照。
    实测三场里 ``20260929_045615`` 录的是 ``range_m=3.0``，而当前默认是 ``5.0``，
    两者回放出来的走廊障碍是 **4439 vs 3618（差 23%）**。
    用默认值回放等于在**别的采集条件**下断言"建图没变"——那不是回归门，
    是拿新问题冒充旧问题。

    语义：**录制里有记录的键用它；录制里没有的新键（``q_tiers`` / ``hi_bands``
    之类后来加的）保持当前默认值**。否则三场老录制会因为新开关存在就被判漂移，
    门会立刻变成一片红然后被人加忽略标记——那是更坏的失败模式。
    """
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    blob = (meta.get("config") or {}).get("mapper") or {}
    known = {f.name for f in dataclasses.fields(MapperConfig)}
    used, ignored = {}, sorted(set(blob) - known)
    for k, v in blob.items():
        if k in known:
            used[k] = v
    started = meta.get("started_wall")
    return dataclasses.replace(MapperConfig(), **used), {
        "recorded_keys": len(used),
        "recorded_not_in_dataclass": ignored,
        "range_m": blob.get("range_m"),
        "started": (datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M")
                    if isinstance(started, (int, float)) else None),
    }


def recorded_sensors(rec: Path) -> dict:
    """从 ``events.jsonl`` 的 ``sensors`` 事件取**实测**内参与基线。

    这份数据本来就在（``backend/nav_online.py`` 感知线程启动时写一行），
    但它不在 ``meta.json`` 里，而 ``meta.json`` 才是大多数人（包括我一开始）
    会去看的地方。⇒ 这里把它读出来并固化进基线，让"这批数字是在什么镜头参数下
    产生的"变成门输出的一部分，而不是需要人去翻 jsonl 的知识。

    缺事件就返回空并由调用方报错——**不猜默认值**。按 cfg 里的名义值回填，
    等于把"没记"伪装成"记了"，那正是本项目约束第 1 条禁止的。
    """
    for line in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "sensors":
            # OpenVR 走 float32，202.5 会变成 202.49999849125743。规整到 4 位：
            # 显示层不需要 float32 噪声，但**基线里要留 4 位**——0.126 与 0.063
            # 的差别正在这一位上，规整到 1 位会把两者抹成一样。
            def _r(v, n=4):
                return round(float(v), n) if isinstance(v, (int, float)) else v
            return {"fx": _r(e.get("fx")), "cx": _r(e.get("cx"), 2), "cy": _r(e.get("cy"), 2),
                    "baseline_m": _r(e.get("baseline_m")),
                    "size": (e.get("info") or {}).get("size")}
    raise KeyError(f"{rec.name}: events.jsonl 里没有 sensors 事件，无法确定实测基线/内参")


def why_corridor(name: str, top: int = 10) -> dict:
    """把**当前**规则留下的走廊障碍拆成连通块，按票型指纹分类。

    与 ``tools/q_tier_why.py`` 问的是不同的问题：那个比"新旧规则"，
    这个问"**剩下**的还堵路的是什么"。现在 ``range_m=5`` + ``q_tiers=on``
    的代表样本是 044153（505 个走廊障碍），要判断的是这里面
    **真结构占多少、量化倾斜占多少**——决定下一步该改规则还是改感知。

    票型含义（沿用 ``MapperConfig`` 的 ``q_*`` 分带）：
        n_o  格内障碍原始点数        o_n  近距(<q_near_m)障碍票
        o_m  中距障碍票              o_mk 投过中距票的关键帧数
        g_n  近距地面票              远/障 远带票占比
    真结构指纹：``o_n`` 大或 ``o_mk≥2``；量化倾斜指纹：票几乎全在远带、
    ``o_n≈0``、``o_mk≤1``。
    """
    rec = ROOT / "navmesh_recordings" / name
    cfg, _info = recorded_config(rec)
    ng, trail, _st, mapper = replay(rec, cfg)
    corridor = _trail_mask(ng, trail, 0.5)
    occ = (ng.grid == OCC) & corridor
    out: dict = {"corridor_blockers": int(occ.sum())}
    if not occ.any():
        return out
    # ⚠️ 累加器→NavGrid 的行翻转逻辑**只从 q_tier_why.to_grid 取**，不重写。
    # 漏掉 g[::-1] 不会报错，只会让票型上下镜像、看着还挺像回事（该文件的原话）。
    from tools.q_tier_why import to_grid            # TODO: tools/ 迁进 research/ 后改指新路径
    n_o = to_grid(mapper._acc_o, mapper, ng)
    o_n = to_grid(mapper._acc_on, mapper, ng)
    o_m = to_grid(mapper._acc_om, mapper, ng)
    o_mk = to_grid(mapper._acc_omk, mapper, ng)
    far = np.maximum(n_o - o_n - o_m, 0.0)

    n, lab, stats, _c = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-area)[:top]
    blocks = []
    for r in order:
        cid, msk = int(r) + 1, lab == int(r) + 1
        no = n_o[msk]
        far_sh = float(np.median(far[msk] / np.maximum(no, 1.0)))
        near_sh = float(np.median((o_n[msk] + o_m[msk]) / np.maximum(no, 1.0)))
        blocks.append({
            "id": cid, "cells": int(area[r]),
            "n_o": round(float(np.median(no)), 1),
            "o_n": round(float(np.median(o_n[msk])), 1),
            "o_m": round(float(np.median(o_m[msk])), 1),
            "o_mk": round(float(np.median(o_mk[msk])), 1),
            "far_share": round(far_sh, 2), "near_share": round(near_sh, 2),
        })
    out["blocks"] = blocks
    out["n_blocks"] = int(n - 1)
    small = sum(b["cells"] for b in blocks if b["cells"] <= 9)
    out["cells_in_top_blocks"] = sum(b["cells"] for b in blocks)
    out["cells_in_small_blocks"] = small
    return out


def run(recs: list[str], dump_grid: Path | None) -> dict:
    out: dict = {}
    for name in recs:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.exists():
            print(f"[skip] {name}: 录制不存在（navmesh_recordings/ 不在版本库里）", file=sys.stderr)
            continue
        cfg, cfg_info = recorded_config(rec)
        sensors = recorded_sensors(rec)
        if not sensors.get("baseline_m"):
            raise KeyError(f"{name}: sensors 事件里没有 baseline_m")
        ng, trail, stats, mapper = replay(rec, cfg)
        out[name] = {"replay": stats, "config": cfg_info, "sensors": sensors,
                     **measure(ng, trail, mapper)}
        if dump_grid is not None:
            dump_grid.mkdir(parents=True, exist_ok=True)
            np.save(dump_grid / f"{name}_grid.npy", ng.grid)
            print(f"  栅格已落盘：{dump_grid / f'{name}_grid.npy'}")
    return out


# ---- 逐字段比对 ----
# 浮点一律按"容差内相等"判，不按字符串判。容差是**相对**的：绝对容差在不同
# 图幅尺寸下含义不同，相对容差才能跨录制复用。
REL_TOL = 1e-6


def _close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= REL_TOL * max(1.0, abs(float(a)), abs(float(b)))
        except (TypeError, ValueError):
            return a == b
    return a == b


def diff(cur: dict, base: dict, path: str = "") -> list[str]:
    """返回人类可读的漂移清单；空列表=逐位一致。"""
    if isinstance(base, dict) and isinstance(cur, dict):
        out: list[str] = []
        for k in sorted(set(base) | set(cur)):
            if k not in cur:
                out.append(f"{path}{k}: 基线有、当前缺")
            elif k not in base:
                out.append(f"{path}{k}: 当前新增（基线无）")
            else:
                out += diff(cur[k], base[k], f"{path}{k}.")
        return out
    if isinstance(base, list) and isinstance(cur, list):
        if len(base) != len(cur):
            return [f"{path[:-1]}: 长度 {len(base)} → {len(cur)}"]
        out = []
        for i, (b, c) in enumerate(zip(base, cur)):
            out += diff(c, b, f"{path}[{i}].")
        return out
    return [] if _close(cur, base) else [f"{path[:-1]}: {base!r} → {cur!r}"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="建图 golden 门")
    ap.add_argument("recs", nargs="*", help=f"录制名（默认 {', '.join(RECS)}）")
    ap.add_argument("--update", action="store_true", help="重写基线")
    ap.add_argument("--strict", action="store_true", help="录制缺失也算失败")
    ap.add_argument("--dump-grid", type=Path, default=None, help="把栅格落盘到该目录")
    ap.add_argument("--why", action="store_true",
                    help="只跑走廊障碍拆解：剩余堵路的里面真结构 vs 量化倾斜")
    ap.add_argument("--selftest", action="store_true",
                    help="只跑合成对照自检（形状指标必须有分辨力，否则门会给出假绿）")
    args = ap.parse_args(argv)

    if args.selftest:
        _shape_selftest()
        return 0

    recs = args.recs or list(RECS)

    if args.why:
        for name in recs:
            if not (ROOT / "navmesh_recordings" / name).exists():
                print(f"[skip] {name}", file=sys.stderr)
                continue
            d = why_corridor(name)
            print(f"\n=== {name}  走廊障碍 {d['corridor_blockers']} 格 / {d.get('n_blocks','?')} 块 ===")
            print(f"{'块#':>5} {'格数':>6} {'n_o':>7} {'o_n':>7} {'o_m':>7} {'o_mk':>5} "
                  f"{'远/障':>6} {'近/障':>6}")
            print("-" * 66)
            for b in d.get("blocks", []):
                print(f"{b['id']:>5} {b['cells']:>6} {b['n_o']:>7.1f} {b['o_n']:>7.1f} "
                      f"{b['o_m']:>7.1f} {b['o_mk']:>5.1f} {b['far_share']:>6.2f} "
                      f"{b['near_share']:>6.2f}")
            if d.get("blocks"):
                print(f"（仅列前 {len(d['blocks'])} 大块；"
                      f"其中 ≤9 格的小块合计 {d['cells_in_small_blocks']} 格）")
        return 0

    cur = run(recs, args.dump_grid)

    if args.update:
        BASELINE.write_text(json.dumps(cur, ensure_ascii=False, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"[update] 基线已写出：{BASELINE}（{len(cur)} 场）")
        for name, r in cur.items():
            print(f"  {name}: occ={r['occ_cells']} unk={r['unk_cells']} "
                  f"走廊障碍={r['corridor_blockers']} 区域={r['build']['regions']}")
        return 0

    if not BASELINE.exists():
        print(f"[fail] 基线不存在：{BASELINE}\n        先跑 python -m tools.mapping_gate --update",
              file=sys.stderr)
        return 2
    base = json.loads(BASELINE.read_text(encoding="utf-8"))

    failed = False
    for name in recs:
        if name not in base:
            print(f"[skip] {name}: 基线里没有这一场（新增录制需先 --update）")
            continue
        if name not in cur:
            msg = f"[fail] {name}: 录制缺失，无法比对"
            print(msg, file=sys.stderr)
            failed = failed or args.strict
            continue
        d = diff(cur[name], base[name])
        if d:
            failed = True
            print(f"[DRIFT] {name}: {len(d)} 处")
            for line in d[:40]:
                print(f"    {line}")
            if len(d) > 40:
                print(f"    ... 还有 {len(d) - 40} 处")
        else:
            r = cur[name]
            np_ = len(r["astar_probes"])
            tail = ""
            if np_ == 0:
                # 0 探针不是"通过"，是"这一场没被门保护"。必须喊出来，
                # 否则门会给出一片绿，用户以为路径受保护了。
                tail = "  [WARN] 无 A* 探针：该场未受路径保护"
            cfg_r = r["config"]
            sn = r["sensors"]
            print(f"[ok] {name}: 采于 {cfg_r['started']} range_m={cfg_r['range_m']} "
                  f"fx={sn['fx']} baseline={sn['baseline_m']}m "
                  f"occ={r['occ_cells']} 走廊障碍={r['corridor_blockers']} "
                  f"区域={r['build']['regions']} "
                  f"A*探针 {sum(1 for p in r['astar_probes'] if p['accepted'])}/{np_} 通过{tail}")
    print("\n" + ("[gate] 失败：有漂移" if failed else "[gate] 通过：与基线逐位一致"))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main(sys.argv[1:]))
