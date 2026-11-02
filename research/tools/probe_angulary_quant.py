# -*- coding: utf-8 -*-
"""AngularY 的量化与削顶影响量化（只读 G:/vrc_recording）。

用户指出的两个硬约束：
1. AngularY 是**量化参数**，满值 1024，超过削顶；
2. 量化步长有限（data-architect 实测 8 单位≈5.75°/s）。

本脚本量化这两条各自吃掉多少信息：
* 削顶率：|AngularY| == 1024 的占比（这些样本的真实角速度未知，只能当截断值）；
* 量化分辨率：相邻非零步长的分布，以及「真实角速度低于一个量化步长」时读到 0 的比例；
* 反向影响：若把它当连续标签，削顶段会被当成「刚好 737°/s」，实际可能更快。

**这决定了 omega 标签是「可用」「需裁剪」还是「不可用」。**

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_angulary_quant.py \
        --root G:/vrc_recording --out .venv_xpu/runs/angulary_quant
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FULL = 1024.0


def read_jsonl(p: Path):
    import json as _j
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _j.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="G:/vrc_recording")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    root = Path(a.root)
    sess = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] == "20")
    print("[scan] %d 场" % len(sess), flush=True)

    per = {}
    all_vals = []
    for s in sess:
        f = s / "avatar_params.jsonl"
        if not f.exists():
            continue
        v = []
        for r in read_jsonl(f):
            if r.get("name") == "AngularY":
                v.append(float(r["value"]))
        if not v:
            continue
        arr = np.asarray(v)
        all_vals.append(arr)
        clipped = int((np.abs(arr) >= FULL).sum())
        # 量化步长：相邻**不同值**之差（排除 0，因为正负号变化会产生假步长）
        u = np.unique(arr)
        steps = np.diff(u)
        steps = steps[steps > 0]
        per[s.name] = dict(
            n=int(arr.size),
            clip_count=clipped,
            clip_pct=round(100.0 * clipped / arr.size, 3),
            zero_pct=round(100.0 * float((arr == 0).mean()), 2),
            distinct=int(u.size),
            step_min=float(steps.min()) if steps.size else None,
            step_median=float(np.median(steps)) if steps.size else None,
            absmax=float(np.abs(arr).max()),
            p99_abs=float(np.percentile(np.abs(arr), 99)),
        )
        print("  %s n=%-6d clip=%6.2f%%  zero=%5.1f%%  distinct=%4d  step_med=%s"
              % (s.name, arr.size, per[s.name]["clip_pct"], per[s.name]["zero_pct"],
                 u.size, per[s.name]["step_median"]), flush=True)

    allv = np.concatenate(all_vals) if all_vals else np.zeros(0)
    n = int(allv.size)
    u = np.unique(allv)
    steps = np.diff(u)
    steps = steps[steps > 0]
    out = {
        "sessions": per,
        "total": dict(
            n=n,
            clip_count=int((np.abs(allv) >= FULL).sum()),
            clip_pct=round(100.0 * float((np.abs(allv) >= FULL).mean()), 3),
            near_clip_pct=round(100.0 * float((np.abs(allv) >= 0.95 * FULL).mean()), 3),
            zero_pct=round(100.0 * float((allv == 0).mean()), 2),
            absmax=float(np.abs(allv).max()),
            p50_abs=float(np.percentile(np.abs(allv), 50)),
            p90_abs=float(np.percentile(np.abs(allv), 90)),
            p99_abs=float(np.percentile(np.abs(allv), 99)),
            distinct=int(u.size),
            step_median=float(np.median(steps)) if steps.size else None,
            step_mode=int(np.bincount(steps.astype(int)).argmax()) if steps.size else None,
        ),
    }
    t = out["total"]
    print()
    print("合计 n=%d  |v|==1024 占比 %.3f%%   |v|>=0.95*1024 占比 %.3f%%"
          % (n, t["clip_pct"], t["near_clip_pct"]))
    print("  |v| p50=%.0f p90=%.0f p99=%.0f max=%.0f" %
          (t["p50_abs"], t["p90_abs"], t["p99_abs"], t["absmax"]))
    print("  量化步长 中位=%s 众数=%s  distinct=%d" %
          (t["step_median"], t["step_mode"], t["distinct"]))
    print("  ==0 占比 %.2f%%" % t["zero_pct"])

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "angulary_quant.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "angulary_quant.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
