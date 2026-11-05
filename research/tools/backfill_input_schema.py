# -*- coding: utf-8 -*-
"""给已有 run 补写 input_schema.json（回填历史 checkpoint）。

为什么要回填：manifest 之前不存在，而 r8_nowp 这类 run 的训练口径可以从
args.json 精确还原（args.json 在 run 根目录，记录了全部训练开关）。
回填比留空更安全：留空会走 LEGACY_DEFAULTS（act_dim=2, drop_wp=False），
对 act_dim=3 且 --drop-wp 的 run 是完全错误的口径。

用法：python backfill_input_schema.py <run_dir> [<run_dir> ...]
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, "research/tools")
import input_schema


def backfill(run):
    run = Path(run)
    ap = run / "args.json"
    if not ap.exists():
        print("  SKIP %s: 没有 args.json" % run)
        return 0
    a = json.loads(ap.read_text(encoding="utf-8"))
    n = 0
    for d in sorted(p for p in run.iterdir() if p.is_dir() and (p / "head.pt").exists()):
        out = int(__import__("torch").load(d / "head.pt", map_location="cpu")["fc.weight"].shape[0])
        if out == 12 * 3:
            act_dim = 3
        elif out == 12 * 2:
            act_dim = 2
        else:
            print("  SKIP %s: head 输出 %d 无法解释" % (d.name, out))
            continue
        s = input_schema.from_args(type("A", (), a))
        s["act_dim"] = act_dim
        s["backfilled"] = True
        input_schema.save(d, s)
        print("  %-28s act_dim=%d drop_wp=%s quad=%s fmt=%s"
              % (run.name + "/" + d.name, act_dim, s["drop_wp"],
                 s["quad_feats"], s["prompt_format"]))
        n += 1
    return n


tot = 0
for r in sys.argv[1:]:
    print("[backfill] %s" % r)
    tot += backfill(r)
print("[done] 写出 %d 份 manifest" % tot)
