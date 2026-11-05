# -*- coding: utf-8 -*-
"""对照：旧 cls（只看平移）vs 新 cls2（走/转正交）。"""
import sys
from pathlib import Path
sys.path.insert(0, "research/tools")
import train_policy_lora as T
import policy_classify as C

s = T.build_index(Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far"), None)
s = C.relabel(s)
print("[总数] %d" % len(s))
print()
print("新 4 类分布（全部样本）:")
for k, v in C.summarize(s).items():
    print("   %-16s n=%6d  %5.1f%%" % (k, v["n"], v["pct"]))
print()
print("旧 cls -> 新 cls2 交叉表:")
print("   %-8s" % "", end="")
for k in C.CLASSES:
    print("%14s" % k, end="")
print("%10s" % "合计")
for o in ("idle", "move", "trans"):
    xs = [x for x in s if x["cls"] == o]
    if not xs:
        continue
    print("   %-8s" % o, end="")
    for k in C.CLASSES:
        print("%14d" % sum(1 for x in xs if x["cls2"] == k), end="")
    print("%10d" % len(xs))
print()
tr = [x for x in s if x["split"] == "train"]
print("train split 的新 4 类:")
for k, v in C.summarize(tr).items():
    print("   %-16s n=%6d  %5.1f%%" % (k, v["n"], v["pct"]))
