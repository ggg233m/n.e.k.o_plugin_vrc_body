# -*- coding: utf-8 -*-
r"""把在线回环闭合（``nav_loop.LoopCloser``）接成**离线回放**，产出最终位姿。

    python research/tools/loop_replay.py 20261001_044153     # 有 final_poses.npz，用来自检
    python research/tools/loop_replay.py 20260929_045615     # 缺 final_poses.npz，这就是目的

**为什么要它**：`final_poses.npz` 只有**在线** `nav_online` 在 `close()` 时才写
（`backend/nav_online.py:248-255`），而 `mapping_gate.replay()` 走的 `KeyframeGridMapper`
**只累积点云、不做位姿优化** ⇒ 老录制（如 `045615`）永远拿不到最终位姿，
`drift_shape.py` 的位姿场证据（B/C/E）就无从谈起（详见
`Docs/漂移形态诊断（2026-10-05）.md` §五/§六）。

**为什么能重建**：`kf/<id>.npz` 存的就是 `KeyframeFeatures` 的全部字段
（`uv` / `des` / `xyz` / `des3d` / `K` / `size` / `eye_y`）+ `T_dr` + `dist_m`，
正好是 `LoopCloser.add_keyframe(k, T_dr, dist_m, feat)` 的入参。

**配置口径**：用**录制当时**的 `meta.json["config"]["loop"]`，不是当前默认值。
实测两场录制的 `max_offset_m = 3.0`，而当前代码默认已是 `6.0` —— 用默认值回放等于
换了采集条件（与 `mapping_gate.recorded_config` 同一条教训）。录制里没记的键保持当前默认。

⚠️ **自检优先**：凡是有 `final_poses.npz` 的录制，本工具会逐帧对比并报差异。
**自检不过就别信缺 final_poses 那场的产物。**
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_loop import KeyframeFeatures, LoopCloser, LoopConfig      # noqa: E402


def recorded_loop_config(rec: Path, current: bool = False) -> tuple[LoopConfig, dict]:
    """构造 ``LoopConfig``。

    ``current=False``（默认）：用**录制当时**的 ``meta.json["config"]["loop"]``，
    录制没记的键保持当前默认 —— 复现的是**当年那一次运行**。
    ``current=True``：除 ``world_scale``（数据口径，不是可调参数）外**全用当前默认** ——
    问的是"**今天的代码**在这份数据上还会不会出同样的问题"。
    """
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    cfg_all = meta.get("config", {})
    raw = dict(cfg_all.get("loop") or {})
    if "world_scale" not in raw and "world_scale" in cfg_all:
        raw["world_scale"] = cfg_all["world_scale"]
    if current:
        ws = cfg_all.get("world_scale", LoopConfig().world_scale)
        return LoopConfig(world_scale=float(ws)), {"world_scale": ws, "_current": True}
    valid = {f.name for f in fields(LoopConfig)}
    kw = {k: v for k, v in raw.items() if k in valid}
    return LoopConfig(**kw), raw


def features_of(z) -> KeyframeFeatures:
    size = tuple(int(v) for v in np.asarray(z["size"]).ravel()[:2])
    return KeyframeFeatures(uv=np.asarray(z["uv"], np.float32),
                            des=np.asarray(z["des"], np.uint8),
                            xyz=np.asarray(z["xyz"], np.float32),
                            des3d=np.asarray(z["des3d"], np.uint8),
                            K=np.asarray(z["K"], np.float64),
                            size=size, eye_y=float(z["eye_y"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="*", default=["20261001_044153"])
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "loop_replay")
    ap.add_argument("--current-config", action="store_true",
                    help="用当前默认 LoopConfig（只保留 world_scale）而不是录制当时的配置；"
                         "问的是『今天的代码在这份数据上还会不会同样出问题』")
    args = ap.parse_args()

    rc = 0
    for name in args.recs:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.exists():
            print(f"[skip] {name} 不存在", file=sys.stderr)
            rc = 1
            continue

        cfg, raw = recorded_loop_config(rec, current=args.current_config)
        events = [json.loads(x) for x in
                  (rec / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        kfs = [e for e in events if e.get("kind") == "kf"]
        print(f"\n=== {name} ===  关键帧事件 {len(kfs)}"
              f"   配置口径：{'**当前默认**' if args.current_config else '录制当时'}")
        print(f"  loop 配置：max_offset_m={cfg.max_offset_m:g} search_max_m={cfg.search_max_m:g} "
              f"min_path_m={cfg.min_path_m:g} bow_candidates={cfg.bow_candidates}"
              f"（当前默认 max_offset_m={LoopConfig().max_offset_m:g}）")

        closer = LoopCloser(cfg)
        n_acc = 0
        acc_pairs: set[tuple[int, int]] = set()
        for e in kfs:
            k = int(e["k"])
            with np.load(rec / "kf" / f"{k:06d}.npz") as z:
                feat = features_of(z)
                T_dr = np.asarray(z["T_dr"], np.float64)
                dist_m = float(z["dist_m"])
            for L in closer.add_keyframe(k, T_dr, dist_m, feat):
                n_acc += 1
                acc_pairs.add((int(L["a"]), int(L["b"])))

        rec_pairs = {(int(L["a"]), int(L["b"])) for e in kfs for L in (e.get("loops") or [])}
        print(f"  本次回环 {n_acc} 条；录制文件里记了 {len(rec_pairs)} 条")
        print(f"  一致 {len(acc_pairs & rec_pairs)}  只在本次 {len(acc_pairs - rec_pairs)}  "
              f"只在录制 {len(rec_pairs - acc_pairs)}")
        print(f"  拒绝统计：{dict(sorted(closer.rejects.items(), key=lambda kv: -kv[1]))}")

        poses = closer.poses()
        ids = sorted(poses)
        args.out.mkdir(parents=True, exist_ok=True)
        dest = args.out / f"{name}.npz"
        np.savez_compressed(dest, ids=np.array(ids),
                            T=np.stack([np.asarray(poses[k], float) for k in ids]))
        print(f"  产物：{dest}（{len(ids)} 帧）")

        fp = rec / "final_poses.npz"
        if fp.exists():
            with np.load(fp) as z:
                ref = {int(i): np.asarray(T, float) for i, T in zip(z["ids"], z["T"])}
            common = sorted(set(ids) & set(ref))
            d = np.array([np.linalg.norm(poses[k][:2, 3] - ref[k][:2, 3]) for k in common])
            print(f"  ── 自检 vs final_poses.npz（{len(common)} 帧）──")
            print(f"     平面位置差：中位 {np.median(d):.4f}   p90 {np.percentile(d,90):.4f}   "
                  f"最大 {d.max():.4f} （追踪米；×0.755 → 世界米）")
            ok = float(np.median(d)) < 1e-6 and float(d.max()) < 1e-6
            print("     ✅ **逐位一致**" if ok else
                  "     ⚠️ **不一致** —— 缺 final_poses 的场次不要用本工具的产物下结论")
        else:
            print("  ── 自检：该录制没有 final_poses.npz，**无法自检** ──")
            print("     结论只能放在「与有自检的场次同口径」这一前提下使用。")
            rc = 2
    return rc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
