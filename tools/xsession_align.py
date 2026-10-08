# -*- coding: utf-8 -*-
"""跨会话采纳入口：逐对采纳（P0.3a）/ 世界树 pass（P0 §洞4）。

核心逻辑在 ``backend/nav_xsession.align_into / align_world_tree``（在线 stop() 的自动采纳
也走前者），本脚本只是命令行入口。产物与 ``sessions/*/poses.npz`` 原表并存，不覆盖：
  * ``<world>/xsession/align_<new>_into_<old>.json`` / ``merged_<new>_into_<old>.npz``
  * ``<world>/xsession/world_tree.json``（--world-tree 的报告）

用法::

    python tools/xsession_align.py --world wrld_home-7cf435ea \\
        --new 20261001_044153 --old 20261001_013519
    python tools/xsession_align.py --world wrld_home-7cf435ea --world-tree --no-write
    python tools/xsession_align.py --world wrld_home-7cf435ea --world-tree --bridge-min 5

方法与两个已付学费的坑见 ``backend/nav_xsession.align_into`` docstring 与
``Docs/archive/P0.2跨会话位姿图合并v1（2026-10-03）.md``；世界树的选根/次序/质量闸见
``backend/nav_xsession.align_world_tree``。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.nav_xsession import XSessionConfig, align_into, align_world_tree  # noqa: E402


def _short(sid: str) -> str:
    return str(sid)[-6:]


def _print_pair(out: dict) -> None:
    print(f"[gauge ] R_dev={out['gauge']['R_dev_deg']}°  t_G={out['gauge']['t_G']}  "
          f"inlier {out['gauge']['n_inlier']}/{out['gauge']['n']}")
    print(f"[train ] n={out['train']['n']}  med={out['train']['med_m']} m  p90={out['train']['p90_m']} m")
    print(f"[holdout] n={out['holdout']['n']}  med={out['holdout']['med_m']} m  "
          f"p90={out['holdout']['p90_m']} m")
    print(f"[moved ] head_med={out['moved_m']['head_med']} m  tail_med={out['moved_m']['tail_med']} m  "
          f"max={out['moved_m']['max']} m")
    if out.get("merged"):
        print(f"[merged] {out['merged']}")
        print(f"[align ] {out['align']}")


def _print_tree(out: dict) -> None:
    print(f"[tree  ] 世界树：bridge_min={out['bridge_min']}  write={out['write']}"
          f"{'  force' if out['force'] else ''}  会话 {len(out['sessions'])}"
          f" · 分量 {len(out['components'])} · 落盘 {out['aligned']}"
          f" · 已在系 {out['skipped']} · 拒 {out['refused']}")
    for c in out["components"]:
        tree = "  ".join(f"{_short(t['sid'])}←{t['count']}" for t in c["tree"]) or "—"
        print(f"[comp  ] root={_short(c['root'])}  覆盖 {c['coverage']}  树: {tree}")
        b = c.get("bridge")
        if b:
            print(f"         缺口: {_short(b['new'])}→{_short(b['old'])} {b['count']} 条"
                  f"（{'够桥' if b['meets_min'] else '低于门槛'}）")
        for u in c.get("unreachable", []):
            e = u.get("best_edge") or {}
            print(f"         到不了根: {_short(u['sid'])}  最强边 "
                  f"{e.get('dir', '?')}={e.get('count')}（{_short(str(e.get('peer', '?')))}）")
        for o in c["ops"]:
            head = f"{_short(o['sid'])}←{_short(o['parent'])} {o['count']} 条"
            st = o["status"]
            pv = o.get("preview") or {}
            if st == "aligned":
                g, ho = pv.get("gauge") or {}, pv.get("holdout") or {}
                print(f"[align ] {head}  已落盘  R_dev={g.get('R_dev_deg')}°  "
                      f"留出 med={ho.get('med_m')} m")
            elif st == "already":
                print(f"[skip  ] {head}  已在该系（链新鲜）")
            elif st == "would_align":
                ho = pv.get("holdout") or {}
                print(f"[plan  ] {head}  可落盘（留出 n={ho.get('n')} med={ho.get('med_m')} m）")
            elif st == "would_refuse":
                print(f"[plan  ] {head}  不过闸：{o.get('gate')}")
            elif st == "refused_gate":
                print(f"[refuse] {head}  质量闸不过：{o.get('gate')}")
            elif st == "blocked_parent":
                print(f"[block ] {head}  父未落位，链断")
            elif st == "planned":
                print(f"[plan  ] {head}  计划内（父落盘后才会真算）")
            else:
                print(f"[{st}] {head}  {pv.get('reason') or pv.get('n') or ''}")
    print(f"[report] {out.get('report')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True, help="世界名（navmesh_memory/ 下）或绝对路径")
    ap.add_argument("--new", help="要被对齐的新会话 sid（逐对模式）")
    ap.add_argument("--old", help="作为世界系基准的旧会话 sid（逐对模式）")
    ap.add_argument("--world-tree", action="store_true",
                    help="世界树 pass：按约束图把够格的桥链式采纳进同一坐标系（不需要 --new/--old）")
    ap.add_argument("--bridge-min", type=int, default=None,
                    help="世界树：桥门槛（默认 align_min_constraints=8；调低=强制弱桥，有风险）")
    ap.add_argument("--root", default=None, help="世界树：强制根会话 sid")
    ap.add_argument("--force", action="store_true", help="世界树：跳过落盘质量闸（自行担责）")
    ap.add_argument("--tail-m", type=float, default=100.0)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--elastic", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-write", action="store_true",
                    help="只算不落盘（世界树模式 = 只出计划与预演，不写 merged）")
    args = ap.parse_args()

    wdir = Path(args.world)
    if not wdir.is_absolute():
        wdir = ROOT / "navmesh_memory" / wdir
    if args.world_tree:
        cfg = XSessionConfig(align_tail_m=args.tail_m, align_holdout_frac=args.holdout_frac,
                             align_elastic=args.elastic)
        out = align_world_tree(wdir, cfg, bridge_min=args.bridge_min, root=args.root,
                               write=not args.no_write, force=args.force)
        if not out.get("ok"):
            print(f"[fail] {out}")
            sys.exit(1)
        _print_tree(out)
        return
    if not (args.new and args.old):
        ap.error("逐对模式需要 --new 与 --old（或改用 --world-tree）")
    out = align_into(wdir, args.new, args.old, tail_m=args.tail_m,
                     holdout_frac=args.holdout_frac, elastic=args.elastic,
                     seed=args.seed, write=not args.no_write)
    if not out.get("ok"):
        print(f"[fail] {out}")
        sys.exit(1)
    _print_pair(out)


if __name__ == "__main__":
    main()