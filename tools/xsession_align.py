# -*- coding: utf-8 -*-
"""会话末跨会话采纳：把一个（新）会话对齐进另一个（旧）会话的世界系（P0.3a）。

核心逻辑在 ``backend/nav_xsession.align_into``（在线 stop() 的自动采纳也走它），
本脚本只是命令行入口。产物与 ``sessions/*/poses.npz`` 原表并存，不覆盖：
  * ``<world>/xsession/align_<new>_into_<old>.json``
  * ``<world>/xsession/merged_<new>_into_<old>.npz``

用法::

    python tools/xsession_align.py --world wrld_home-7cf435ea \
        --new 20261001_044153 --old 20261001_013519

方法与两个已付学费的坑见 ``backend/nav_xsession.align_into`` docstring 与
``Docs/P0.2跨会话位姿图合并v1（2026-10-03）.md``。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.nav_xsession import align_into  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True, help="世界名（navmesh_memory/ 下）或绝对路径")
    ap.add_argument("--new", required=True, help="要被对齐的新会话 sid")
    ap.add_argument("--old", required=True, help="作为世界系基准的旧会话 sid")
    ap.add_argument("--tail-m", type=float, default=100.0)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--elastic", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-write", action="store_true", help="只算不落盘")
    args = ap.parse_args()

    wdir = Path(args.world)
    if not wdir.is_absolute():
        wdir = ROOT / "navmesh_memory" / wdir
    out = align_into(wdir, args.new, args.old, tail_m=args.tail_m,
                     holdout_frac=args.holdout_frac, elastic=args.elastic,
                     seed=args.seed, write=not args.no_write)
    if not out.get("ok"):
        print(f"[fail] {out}")
        sys.exit(1)
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


if __name__ == "__main__":
    main()
