# -*- coding: utf-8 -*-
"""审计：哪些 run 的 adapter 带 _orig_mod 前缀（= 直接 load_adapter 会静默失效）。

带前缀的 run，凡是用裸 load_adapter() 评过的数，都是「未训练模型」的数。
"""
import sys, os
from pathlib import Path
sys.path.insert(0, "research/tools")
from safetensors import safe_open

root = Path(".venv_xpu/runs")
rows = []
for d in sorted(root.rglob("adapter")):
    st = d / "adapter_model.safetensors"
    if not st.exists():
        continue
    with safe_open(str(st), "pt") as f:
        ks = list(f.keys())
    n = sum(1 for k in ks if "._orig_mod." in k)
    run = d.parent.relative_to(root).as_posix()
    rows.append((run, len(ks), n, "PREFIXED" if n else "clean"))

w = max(len(r[0]) for r in rows) + 2
print("%-*s %6s %8s  %s" % (w, "run", "keys", "orig_mod", "verdict"))
for r in rows:
    print("%-*s %6d %8d  %s" % (w, r[0], r[1], r[2], r[3]))
print()
bad = [r[0] for r in rows if r[3] == "PREFIXED"]
print("带前缀的 run（%d 个）:" % len(bad))
for b in bad:
    print("   ", b)
print()
print("干净 run（%d 个）:" % (len(rows) - len(bad)))
for r in rows:
    if r[3] == "clean":
        print("   ", r[0])
