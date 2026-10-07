#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""用**当前**回环配置重放一场录制，并把结果写回该会话的位姿表。

为什么需要它
------------
会话的 ``poses.npz`` 是**录制当时那一次在线运行**的产物，带着当时的 ``LoopConfig``。
配置本身会演进（实测：``044153`` 录制时 ``loop.max_offset_m = 3.0``，代码默认早已是
``6.0``），于是老会话永久冻结在一个**已知更差**的位姿上：

* ``044153`` 用当时的配置只出 **225** 条回环，尾部 **169 kf / 99.5 m** 一条长回环都没有
  （占全场 45.6%）；今天配置在同一份数据上出 **257** 条，最长空洞降到 **4 kf**。
* 而下游（``pointcloud_view``、``offline_fusion``、世界树、先验）读的都是这份旧表
  ⇒ **旧伤被一路继承**。

本工具只换 ``T_map``，其余字段（``T_dr`` / ``dist_m`` / ``t_s`` / ``has_feat``）原样保留
—— 它们与录制档逐位一致（实测 T_dr 最大差 9.5e-07、dist_m 7.5e-06 m），没有理由重算。

⚠️ 两条安全闸（都通过才写）
--------------------------
1. **id 一一对应**：重放的关键帧 id 必须与 ``poses.npz`` 的 ``ids`` 完全相同；
2. **坐标系口径一致**：新旧 ``T_map`` 的旋转必须**逐位相同**（yaw 来自 HMD、回环不改朝向），
   且两者都应以首关键帧为原点。否则写进去会让整场跳到错误位置。

⚠️ 写了之后世界树/merged 还没变
-------------------------------
``session_pose_table`` 优先返回 ``xsession/merged_*.npz``，而 ``pose_table_rigid`` 是
"把原表刚体拟合到 merged"。所以换掉原表**立刻**就会改变点云（实测最终世界位置移动
中位 1.37 m），但要让**跨会话**一致，还得重跑 ``tools/xsession_align.py --world-tree``。
本工具会在结尾提示这一点。

用法
----
    # 干跑：只算不写，报出新旧差异与两条闸的结果
    python research/tools/rerun_session_poses.py --rec 20261001_044153 --world wrld_home-7cf435ea --dry-run

    # 落盘（会先自动备份 poses.npz 与 session.json）
    python research/tools/rerun_session_poses.py --rec 20261001_044153 --world wrld_home-7cf435ea --write

    # 用**录制当时**的配置重放（复现当年，用于自检工具本身）
    python research/tools/rerun_session_poses.py --rec ... --world ... --recorded-config --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_loop import LoopCloser                                   # noqa: E402
from research.tools.loop_replay import features_of, recorded_loop_config  # noqa: E402

S = 0.755


def replay(rec_dir: Path, cfg) -> tuple[list[int], dict[int, np.ndarray], list[tuple[int, int]],
                                        dict[str, Any]]:
    """按录制档的 kf 顺序跑 LoopCloser，返回 (ids, {k: T_map}, 回环对, 统计)。"""
    kfs = sorted(rec_dir.glob("kf/*.npz"), key=lambda p: int(p.stem))
    if not kfs:
        raise SystemExit(f"[fail] {rec_dir}/kf 是空的")
    closer = LoopCloser(cfg)
    ids: list[int] = []
    for f in kfs:
        k = int(f.stem)
        with np.load(f) as z:
            feat = features_of(z)
            T_dr = np.asarray(z["T_dr"], np.float64)
            dist_m = float(np.asarray(z["dist_m"]).reshape(-1)[0])
        closer.add_keyframe(k, T_dr, dist_m, feat)
        ids.append(k)
    poses = {k: closer.pose(k) for k in ids}
    st = closer.status()
    pairs = [(int(L.a), int(L.b)) for L in closer.loops]
    return ids, poses, pairs, {"loops": len(closer.loops),
                               "downweighted": int(st.get("loops_downweighted", 0)),
                               "rejects": st.get("rejects", {})}


def interval_union_cover(ids: list[int], loops: list[tuple[int, int]],
                         dist: dict[int, float], min_gap_m: float = 15.0) -> dict:
    """回环**区间并集**覆盖（端点计数会严重高估，见 Docs/回环触发率-供给复核 §三）。"""
    pos = {k: i for i, k in enumerate(ids)}
    n = len(ids)
    cov = np.zeros(n, bool)
    spans = 0
    for a, b in loops:
        if a in pos and b in pos and dist.get(b, 0.0) - dist.get(a, 0.0) >= min_gap_m:
            cov[pos[a]:pos[b] + 1] = True
            spans += 1
    run = best = 0
    best_i = 0
    for i, c in enumerate(cov):
        run = 0 if c else run + 1
        if run > best:
            best, best_i = run, i
    return {"n_loops_long": spans, "covered_kf": int(cov.sum()), "n_kf": n,
            "covered_frac": float(cov.mean()),
            "longest_gap_kf": int(best),
            "longest_gap_from": (ids[max(0, best_i - best + 1)] if best else None),
            "longest_gap_to": (ids[best_i] if best else None),
            "longest_gap_m": (float(dist[ids[best_i]] - dist[ids[max(0, best_i - best + 1)]])
                              if best else 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rec", required=True, help="录制目录名（navmesh_recordings/ 下）")
    ap.add_argument("--world", required=True, help="世界名（navmesh_memory/ 下）")
    ap.add_argument("--session", default=None, help="会话 sid（默认与 --rec 同名）")
    ap.add_argument("--recorded-config", action="store_true",
                    help="用录制当时的 LoopConfig 而不是当前默认（复现当年，用于自检）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="只算不写")
    g.add_argument("--write", action="store_true", help="备份后落盘")
    a = ap.parse_args()

    rec_dir = ROOT / "navmesh_recordings" / a.rec
    if not rec_dir.is_dir():
        raise SystemExit(f"[fail] 没有这个录制：{rec_dir}")
    sid = a.session or a.rec
    world_dir = ROOT / "navmesh_memory" / a.world
    sdir = world_dir / "sessions" / sid
    if not (sdir / "poses.npz").is_file():
        raise SystemExit(f"[fail] 没有这个会话的位姿表：{sdir / 'poses.npz'}")

    cfg, raw = recorded_loop_config(rec_dir, current=not a.recorded_config)
    print(f"[cfg  ] {'录制当时' if a.recorded_config else '**当前默认**'}；"
          f"max_offset_m={cfg.max_offset_m:g} search_max_m={cfg.search_max_m:g} "
          f"min_path_m={cfg.min_path_m:g} min_inliers={cfg.min_inliers} "
          f"bow_candidates={cfg.bow_candidates}")

    with np.load(sdir / "poses.npz") as z:
        s_ids = np.asarray(z["ids"], np.int64)
        old_T = np.asarray(z["T_map"], np.float64)
        keep = {k: np.asarray(z[k]) for k in z.files if k not in ("ids", "T_map")}
    old_meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))

    # 旧表的回环（从 events.jsonl 读，那是当时在线记的）
    ev = [json.loads(x) for x in
          (rec_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    old_loops = [(int(L["a"]), int(L["b"])) for e in ev if e.get("kind") == "kf"
                 for L in (e.get("loops") or [])]

    t0 = time.perf_counter()
    ids, poses, new_loops, st = replay(rec_dir, cfg)
    dt = time.perf_counter() - t0
    print(f"[replay] {len(ids)} 关键帧，{dt:.1f}s，回环 **{st['loops']}** 条"
          f"（降权 {st['downweighted']}）  拒绝 {st['rejects']}")

    # ---- 闸 1：id 一一对应 ----
    new_ids = np.asarray(ids, np.int64)
    if not np.array_equal(new_ids, s_ids):
        raise SystemExit(f"[fail] 闸1 不过：重放 id 与位姿表不一致"
                         f"（{len(new_ids)} vs {len(s_ids)}）")
    print(f"[闸1  ] id 一一对应 ✅（{len(ids)} 帧）")

    new_T = np.stack([np.asarray(poses[k], np.float64) for k in ids])

    # ---- 闸 2：坐标系口径 ----
    dR = float(max(np.abs(old_T[i][:3, :3] - new_T[i][:3, :3]).max() for i in range(len(ids))))
    if dR > 1e-9:
        raise SystemExit(f"[fail] 闸2 不过：新旧 T_map 的旋转不一致（最大差 {dR:.3e}）"
                         f"—— yaw 来自 HMD，回环不该改朝向；不同系不能直接覆盖")
    if np.linalg.norm(new_T[0][:2, 3]) > 1e-6 or np.linalg.norm(old_T[0][:2, 3]) > 1e-6:
        raise SystemExit("[fail] 闸2 不过：首帧平移不是原点 —— 两者不是同一个基准")
    print(f"[闸2  ] 旋转逐位相同（最大差 {dR:.1e}）、两者均以首帧为原点 ✅")

    # ---- 差异报告 ----
    po, pn = old_T[:, :2, 3], new_T[:, :2, 3]
    d = np.linalg.norm(po - pn, axis=1) * S
    dist = {k: float(v) for k, v in zip(ids, np.asarray(keep["dist_m"], np.float64))}
    cov_old = interval_union_cover(list(s_ids), old_loops, dist)
    cov_new = interval_union_cover(ids, new_loops, dist)

    print(f"\n[diff ] 逐帧世界位置：中位 {np.median(d):.3f} m  p90 {np.percentile(d,90):.3f}  "
          f"最大 {d.max():.3f} m")
    for tag, c, n in (("旧表（在线当时）", cov_old, len(old_loops)),
                      ("**新表（本次重放）**", cov_new, st["loops"])):
        print(f"[cover] {tag} 回环 {n} 条 ⇒ 长回环 {c['n_loops_long']} 条，"
              f"覆盖 {c['covered_kf']}/{c['n_kf']} kf ({c['covered_frac']*100:.1f}%)，"
              f"最长空洞 **{c['longest_gap_kf']} kf / {c['longest_gap_m']:.1f} m**"
              + (f"（kf {c['longest_gap_from']}→{c['longest_gap_to']}）" if c["longest_gap_kf"] else ""))

    if not a.write:
        print("\n[dry  ] 未写盘。加 --write 落盘。")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = ROOT / ".tmp" / f"rerun_session_poses_{stamp}"
    bak.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sdir / "poses.npz", bak / "poses.npz")
    shutil.copy2(sdir / "session.json", bak / "session.json")

    buf = io.BytesIO()
    np.savez_compressed(buf, ids=np.asarray(ids, np.int32), T_map=new_T.astype(np.float32),
                        **{k: v for k, v in keep.items()})
    tmp = sdir / "poses.npz.tmp"
    tmp.write_bytes(buf.getvalue())
    tmp.replace(sdir / "poses.npz")

    meta = dict(old_meta)
    meta["loops"] = int(st["loops"])
    meta["poses_written_wall"] = time.time()
    meta["rerun"] = {"by": "research/tools/rerun_session_poses.py",
                     "at_wall": time.time(),
                     "config": "current" if not a.recorded_config else "recorded",
                     "loops_before": int(old_meta.get("loops", -1)),
                     "backup": str(bak.relative_to(ROOT))}
    (sdir / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"\n[write] {sdir / 'poses.npz'}（T_map 已换，其余字段原样）")
    print(f"[write] {sdir / 'session.json'}：loops {old_meta.get('loops')} → {st['loops']}")
    print(f"[bak  ] {bak}")
    print("\n⚠️ 世界树/merged 还是旧的：要让**跨会话**一致，接着跑"
          "\n      python tools/xsession_align.py --world "
          f"{a.world} --world-tree")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())