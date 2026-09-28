"""按区域标注的区域内外相似度分布分析（离线，不接实时链路）。

配合 ``world_mapping_probe.py`` 的 go/no-go 验证：把录屏 1fps 抽帧后按肉眼
核验标注粗粒度区域（含重访归并），统计现有指纹在"区域内 vs 区域间"的
相似度分布，检验是否存在可行阈值（区域内 min > 区域间 max 才可分）。

区域标注基于 tools 作者对 65 帧的逐帧肉眼核验（2026-09-18 07-31-11.mkv）：
  A  t1-9   室内海报大厅        A2 t58-63 大厅重访（归并 A）
  B  t10-14 初音舞台            B2 t37-43 舞台重访（归并 B）
  C  t15-19 灯笼岩台            C2 t53-57 灯笼岩台回程（归并 C）
  D  t20-27 室外沙地广场        D2 t44-52 广场回程（归并 D）
  E  t28-32 舞蹈地板/花园边缘
  F  t33-36 花园深处
  M  t64-65 VRChat 菜单遮挡（剔除，不参与统计）

用法
----
  .venv/Scripts/python.exe research/tools/world_region_similarity.py \
      --frames-dir .tmp --pattern wm_f_ --json-out .tmp/world_region_sim.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.avatar_identity import (  # noqa: E402
    _context_descriptor,
    appearance_descriptor,
    _similarity,
)

RANGES = [
    ("A", range(1, 10)), ("B", range(10, 15)), ("C", range(15, 20)),
    ("D", range(20, 28)), ("E", range(28, 33)), ("F", range(33, 37)),
    ("B", range(37, 44)), ("D", range(44, 53)), ("C", range(53, 58)),
    ("A", range(58, 64)), (None, range(64, 66)),  # None = 菜单遮挡，剔除
]


def region_of(i: int) -> str | None:
    for label, rng in RANGES:
        if i in rng:
            return label
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, default=Path(".tmp"))
    parser.add_argument("--pattern", default="wm_f_")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import cv2  # noqa: F401  缺 cv2 时在此如实报错

    ctx: dict[int, object] = {}
    col: dict[int, object] = {}
    n_max = max(max(r) for _, r in RANGES)
    for i in range(1, n_max + 1):
        frame = cv2.imread(str(args.frames_dir / f"{args.pattern}{i:04d}.jpg"))
        if frame is None:
            print(f"[warn] 缺帧 {i}", file=sys.stderr)
            continue
        ctx[i] = _context_descriptor(frame)
        col[i] = appearance_descriptor(frame, (0.0, 0.0, 1.0, 1.0))

    def analyze(desc: dict[int, object], name: str) -> dict:
        same: dict[str, list[float]] = {}
        diff: dict[tuple[str, str], list[float]] = {}
        for a in desc:
            for b in desc:
                if b <= a:
                    continue
                la, lb = region_of(a), region_of(b)
                if la is None or lb is None:
                    continue
                s = _similarity(desc[a], desc[b])  # type: ignore[arg-type]
                if la == lb:
                    same.setdefault(la, []).append(s)
                else:
                    diff.setdefault((min(la, lb), max(la, lb)), []).append(s)

        def summary(v: list[float]) -> dict:
            v = sorted(v)
            return {"n": len(v), "min": v[0], "median": median(v),
                    "max": v[-1]}

        intra_all, inter_all = [], []
        for v in same.values():
            intra_all.extend(v)
        for v in diff.values():
            inter_all.extend(v)
        intra_all.sort()
        inter_all.sort()
        return {
            "intra_by_region": {k: summary(v) for k, v in sorted(same.items())},
            "inter_by_pair_top8_bottom3": [
                {"pair": list(k), **summary(v)}
                for k, v in sorted(diff.items(),
                                   key=lambda kv: -median(kv[1]))[:8]
                + sorted(diff.items(),
                         key=lambda kv: -median(kv[1]))[-3:]],
            "intra_min": intra_all[0],
            "inter_max": inter_all[-1],
            "separable": intra_all[0] > inter_all[-1],
        }

    report = {"labels": {str(i): region_of(i) for i in range(1, n_max + 1)},
              "context_4x8": analyze(ctx, "context"),
              "color_histogram": analyze(col, "color")}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False,
                                            indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
