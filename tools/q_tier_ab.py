# -*- coding: utf-8 -*-
"""观测质量分层（``MapperConfig.q_tiers``）A/B：拿真实录制跑**在线 mapper 本体**。

    python -m tools.q_tier_ab [录制名 ...]        # 默认三场全跑

与 ``.tmp/_fusion_v1.py`` 的离线实验同一套度量口径（run5-7「已知内障碍%」+ 轨迹距离环
R∈{1.5,2.5,3.0}），但**不另写一份分类器**：直接把录制喂进 ``KeyframeGridMapper``，
跑 ``rasterize()``。这样量到的就是线上真正在跑的那条路径，离线实验固化的规则与在线
实现漂移也无处藏。

三组对照：
  * ``off``   —— q_tiers=False，历史扁平计数规则（本次改动前的线上行为）
  * ``on``    —— q_tiers=True，分层规则（本次改动）
  * ``doc``   —— 离线实验那一版的字面口径（降级判回 FREE 用近+中距地面），用来量
                 "把恢复条件收紧到只用近距地面"到底损失多少、安全性换回什么

外加两个安全代理指标（单看障碍%会奖励"把真墙也删掉"，必须并排看）：
  * ``downgraded_free_no_near_gnd``：被判回 FREE、但该格**没有**近距地面证据的格数。
    这些是"没有任何可信证据说它是地板"的扩张；收紧后的 ``on`` 组按构造恒为 0。
  * ``walkable_center_m2``：``build()`` 后的可走中心面积——导航真正吃的量。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_grid import FREE, OCC, UNK                       # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig   # noqa: E402

sys.path.insert(0, str(ROOT / "research" / "tools"))
from trail_geom import trail_mask                                 # noqa: E402

RINGS = (1.5, 2.5, 3.0)
RECS = ("20260929_045615", "20261001_044120", "20261001_044153")


def load_events(rec: Path) -> list[dict]:
    lines = (rec / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def replay(rec: Path, cfg: MapperConfig):
    """按录制事件顺序把关键帧喂进 mapper，末尾做一次最终栅格化。

    返回 ``(栅格, 轨迹, 统计, mapper)``——mapper 带着对齐好的分带票累加器，供离屏诊断使用。
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
    t0 = time.perf_counter()
    for e in events:
        kind = e.get("kind")
        if kind == "kf":
            k = int(e["k"])
            T = pose(k)
            if T is None:
                continue
            pts = kf(k)["pts"].astype(np.float32)
            if e.get("refresh"):
                m.drop_points(k - 1)      # 原地不动补帧：上一帧让出（与在线同口径）
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
    build_s = time.perf_counter() - t0
    ng = m.rasterize()
    return ng, trail, {"n_kf": len(m), "n_refresh": n_refresh, "replay_s": round(build_s, 1)}, m


def measure(ng, trail: list[np.ndarray], base: np.ndarray | None,
            walked: list[np.ndarray] | None = None) -> dict:
    g = ng.grid
    occ, free = g == OCC, g == FREE
    out: dict = {"occ_cells": int(occ.sum()), "free_cells": int(free.sum()),
                 "unk_cells": int((g == UNK).sum())}
    # walked 必须真传：NavGrid.build 会把它无条件算作可走（前提是那格不是障碍），
    # 传空列表量出来的"可走中心面积"跟规划器真正吃的东西不是一回事。
    build = ng.build(radius_m=0.25, walked=walked or [])
    out["walkable_center_m2"] = build["walkable_center_m2"]
    out["regions"] = build["regions"]
    out["corridor_blockers"] = corridor_blockers(ng, trail)


def ring_masks(ng, trail: list[np.ndarray], radius_m: float):
    """轨迹半径 ``radius_m`` 环内的格。**实现已迁到 ``trail_geom.trail_mask``**（唯一实现）。

    ⚠️ 2026-10-08：本函数原来是转置的（把 x 当行、漏掉 NavGrid 的上下翻转），与
    ``mapping_gate._trail_mask``、``precision_probe.corridor_blockers`` 是同一个错误。
    签名从 ``(shape, meta, trail, R)`` 改成 ``(ng, trail, R)``：算对朝向需要 ``to_cell``，
    而 ``shape`` + ``meta`` 拆开传正是让"自己再推一遍坐标"变得顺手的形状。
    """
    return trail_mask(ng, trail, radius_m)


def near_ground_mask(ng, trail: list[np.ndarray], band_m: float = 0.6) -> np.ndarray:
    """轨迹两侧 band_m 内的格——身体实际走过的地方，是"这确实是地板"的独立证据。

    用来给"判回 FREE"的格做抽检：落在走廊里的降级几乎必然是真假障碍修正，
    落在没人走过的角落里的降级则没有外部证据支撑。

    实现见 ``trail_geom.trail_mask``（2026-10-08 起唯一实现）。
    """
    return trail_mask(ng, trail, band_m)


def corridor_blockers(ng, trail: list[np.ndarray], half_m: float = 0.5) -> int:
    """走廊上还有几个障碍格——这才是"能不能走过去"的直接度量。

    走过的走廊本来就会被 ``NavGrid.build`` 的 ``walked`` 无条件算作可走（前提是那格不是障碍），
    所以障碍格才是唯一拦路的；降级成 unknown 的格并不挡身体已经走过的路。
    门宽按 avatar 半径留余量：走廊中线两侧 half_m 内仍算"卡在门上"。
    """
    return int((near_ground_mask(ng, trail, half_m) & (ng.grid == OCC)).sum())


def measure(ng, trail: list[np.ndarray], base: np.ndarray | None,
            walked: list[np.ndarray] | None = None) -> dict:
    g = ng.grid
    occ, free = g == OCC, g == FREE
    out: dict = {"occ_cells": int(occ.sum()), "free_cells": int(free.sum()),
                 "unk_cells": int((g == UNK).sum())}
    # walked 必须真传：NavGrid.build 会把它无条件算作可走（前提是那格不是障碍），
    # 传空列表量出来的"可走中心面积"跟规划器真正吃的东西不是一回事。
    build = ng.build(radius_m=0.25, walked=walked or [])
    out["walkable_center_m2"] = build["walkable_center_m2"]
    out["regions"] = build["regions"]
    out["corridor_blockers"] = corridor_blockers(ng, trail)
    for R in RINGS:
        ring = ring_masks(ng, trail, R)
        n_o, n_f = int((occ & ring).sum()), int((free & ring).sum())
        n_all = int(ring.sum())
        out[f"R{R}"] = {"known_obstacle_pct": round(100.0 * n_o / max(n_o + n_f, 1), 2),
                        "free_pct": round(100.0 * n_f / max(n_all, 1), 2),
                        "unknown_pct": round(100.0 * (n_all - n_o - n_f) / max(n_all, 1), 2)}
    if base is not None:
        on_corr = near_ground_mask(ng, trail)
        to_free = base & ~occ
        to_unk = base & (g == UNK)
        out["from_occ_to_free"] = int(to_free.sum())
        out["from_occ_to_unk"] = int(to_unk.sum())
        # 走廊内/外分开：走廊里的降级有"身体确实走过"作外部证据，走廊外的没有。
        out["freed_on_corridor"] = int((to_free & on_corr).sum())
        out["freed_off_corridor"] = int((to_free & ~on_corr).sum())
        out.update(components(base, occ))
    return out


def components(base: np.ndarray, occ: np.ndarray, min_big: int = 40) -> dict:
    """把 baseline 的障碍连通块按大小分组，看新规则保住了哪些。

    这是回答"删掉的是墙还是孤岛"最直接的判据：墙是**大块**连通体（连续观测几米，几十上百格），
    假障碍是**小岛**（单帧量化倾斜打出来的一撮，几个到几十格）。项目自己的实测也是这个分布
    （Docs/离线多视角融合v1:21 min 150 块障碍里 94 块是空地中间的孤岛）。

    每块记两件事：size（格数）、kept（该块在新规则下仍是障碍的比例）。
    期望：大块几乎全保住，小块基本被清掉。
    """
    n, lab, stats, _c = cv2.connectedComponentsWithStats(base.astype(np.uint8), connectivity=8)
    out: dict[str, float] = {}
    if n <= 1:
        return {"comp_big": 0, "comp_small": 0, "comp_big_kept_frac": 0.0,
                "comp_small_kept_frac": 0.0, "comp_big_cells": 0, "comp_small_cells": 0}
    area = stats[1:, cv2.CC_STAT_AREA]
    ids = np.arange(1, n)
    big, small = ids[area >= min_big], ids[area < min_big]
    flat_lab, flat_occ = lab.reshape(-1), occ.reshape(-1)
    for tag, sel in (("big", big), ("small", small)):
        if not len(sel):
            out[f"comp_{tag}"] = 0
            out[f"comp_{tag}_cells"] = 0
            out[f"comp_{tag}_kept_frac"] = 0.0
            continue
        mask = np.isin(flat_lab, sel)
        out[f"comp_{tag}"] = int(len(sel))
        out[f"comp_{tag}_cells"] = int(mask.sum())
        out[f"comp_{tag}_kept_frac"] = round(float(flat_occ[mask].mean()), 3)
    return out


def variants(rec: Path) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    trail: list[np.ndarray] = []
    res: dict[str, dict] = {}
    grids: dict[str, np.ndarray] = {}
    base_grid = None
    # on = 默认（q_free_m=1.5，只认近距地面，判回空地很克制）；on_mid = 判回空地放宽到
    # 近+中距（离线 v1 的字面口径）。两者对"走廊上还有几个障碍"几乎没差别，差的只是图外
    # 那圈可探索面积——所以要把这个差别量出来，而不是凭直觉选一个。
    for name, cfg in (("off", MapperConfig(q_tiers=False)),
                      ("on", MapperConfig(q_tiers=True)),
                      ("on_mid", MapperConfig(q_tiers=True, q_free_m=3.0))):
        ng, trail, stats, m = replay(rec, cfg)
        res[name] = measure(ng, trail, base_grid, m.walked())
        res[name].update(stats)
        grids[name] = ng.grid.copy()
        if name == "off":
            base_grid = ng.grid == OCC
    return res, grids, ng, trail


def render(grids: dict[str, np.ndarray], ng, trail: list[np.ndarray], dest: Path) -> None:
    """左=改动前（扁平计数），右=改动后（质量分层）。黑=障碍 白=空地 灰=未知 蓝=轨迹。"""
    tiles = []
    for name, title in (("off", "before  q_tiers=off (flat count rule)"),
                        ("on", "after  q_tiers=on  (quality layered)")):
        g = grids[name]
        img = np.where((g == UNK)[..., None], np.array([205, 205, 205]),
                       np.where((g == OCC)[..., None], np.array([25, 25, 25]),
                                np.array([250, 250, 250]))).astype(np.uint8)
        for p in trail:
            rc = ng.to_cell((p[0] * ng.meta.world_scale, p[1] * ng.meta.world_scale))
            if ng.inside(rc):
                cv2.circle(img, (rc[1], rc[0]), 1, (40, 130, 235), -1)
        sc = min(1.0, 760 / max(g.shape))
        if sc < 1.0:
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, 30, 8, 8, 8, cv2.BORDER_CONSTANT, value=(45, 45, 45))
        cv2.putText(img, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    canvas = np.full((max(t.shape[0] for t in tiles), sum(t.shape[1] for t in tiles) + 16, 3), 45, np.uint8)
    x = 0
    for t in tiles:
        canvas[:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 16
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), canvas)


def main(recs: list[str]) -> None:
    out: dict[str, dict] = {}
    for name in recs:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.exists():
            print(f"[skip] {name}: 录制不存在", file=sys.stderr)
            continue
        res, grids, ng, trail = variants(rec)
        out[name] = res
        render(grids, ng, trail, Path(__file__).parent / f"q_tier_{name}.png")
    dest = Path(__file__).parent / "q_tier_ab.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, r in out.items():
        print(f"\n=== {name} ===")
        for v in ("off", "on", "on_mid"):
            if v not in r:
                continue
            d = r[v]
            line = (f"  {v:7s} occ={d['occ_cells']:6d} free={d['free_cells']:6d} "
                    f"unk={d['unk_cells']:6d} 走廊障碍={d['corridor_blockers']:5d} "
                    f"中心区={d['walkable_center_m2']:7.1f}m2 区域={d['regions']}")
            if "from_occ_to_free" in d:
                line += (f"  障碍→空地={d['from_occ_to_free']:6d}"
                         f"(走廊内 {d['freed_on_corridor']}/外 {d['freed_off_corridor']})"
                         f" 障碍→未知={d['from_occ_to_unk']:6d}")
                line += (f"  块: 大{d['comp_big']}块(留{d['comp_big_kept_frac']:.0%})"
                         f"/小{d['comp_small']}块(留{d['comp_small_kept_frac']:.0%})")
            print(line)
            print("       环内已知障碍% " + "  ".join(
                f"R{R}={d[f'R{R}']['known_obstacle_pct']:.2f}" for R in RINGS)
                + "   free% " + "  ".join(f"R{R}={d[f'R{R}']['free_pct']:.1f}" for R in RINGS))
    print(f"\n写出：{dest}")


if __name__ == "__main__":
    try:                       # 控制台可能是 GBK（Windows 默认），别让一个上标符号毁掉整轮测量
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main(sys.argv[1:] or list(RECS))
