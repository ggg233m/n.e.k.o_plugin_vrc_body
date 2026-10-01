# -*- coding: utf-8 -*-
"""从一份录制训练 ORB 词袋词汇树，供回环的**外观候选**使用（`backend/nav_bow.py`）。

用法::

    python tools/train_bow_vocab.py --rec 20261001_044153
    python tools/train_bow_vocab.py --rec 20261001_044153,20260929_045615 \
        --out models/bow_vocab.npz --branching 16 --depth 3

输出到 `models/bow_vocab.npz`；`backend/nav_loop.py` 的 `LoopConfig.bow_vocab`
默认指向这里，但 `bow_candidates` 默认 **0（关）**，要显式打开才有外观候选。

⚠️ **词汇树是世界相关的**：它把 ORB 描述子空间切成 4096 个词，切法由训练数据决定。
跨路线（同世界）迁移实测只掉 0.5~4.6 pt 召回；**跨世界未测**，换世界请重训。
这也是它不随包默认开启的原因之一。

训练只用**前若干帧**的描述子（默认前一半），这样尾部帧可以用来做"没见过的地方"的
召回测试，不会自己检自己。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import nav_bow  # noqa: E402


def load_descriptors(rec: Path, upto_frac: float, max_rows: int) -> np.ndarray:
    files = sorted(rec.glob("kf/*.npz"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"{rec} 里没有 kf/*.npz")
    n = max(1, int(len(files) * upto_frac))
    pool = []
    for f in files[:n]:
        d = np.load(f, allow_pickle=True)
        des = d.get("des")
        if des is not None and len(des):
            pool.append(np.asarray(des, dtype=np.uint8))
    if not pool:
        raise SystemExit(f"{rec} 前 {n} 帧没有描述子")
    A = np.concatenate(pool)
    if len(A) > max_rows:
        A = A[np.random.default_rng(0).choice(len(A), max_rows, replace=False)]
    return A


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True,
                    help="录制名或目录，逗号分隔多份（navmesh_recordings/<name>）")
    ap.add_argument("--out", default="models/bow_vocab.npz")
    ap.add_argument("--branching", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--upto-frac", type=float, default=0.5,
                    help="每份录制只用前这么多比例的关键帧训练（留后半做测试）")
    ap.add_argument("--max-rows", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = []
    for name in a.rec.split(","):
        name = name.strip()
        if not name:
            continue
        rec = Path(name) if Path(name).is_dir() else ROOT / "navmesh_recordings" / name
        if not rec.exists():
            raise SystemExit(f"找不到录制：{rec}")
        d = load_descriptors(rec, a.upto_frac, a.max_rows)
        print(f"  {rec.name}：{len(d)} 行描述子")
        pool.append(d)
    A = np.concatenate(pool)
    print(f"合计 {len(A)} 行；词汇树 {a.branching}^{a.depth} = {a.branching ** a.depth} 词")

    t0 = time.perf_counter()
    vocab = nav_bow.BowVocabulary(branching=a.branching, depth=a.depth,
                                  seed=a.seed).train(A, iters=a.iters)
    print(f"训练 {time.perf_counter() - t0:.1f}s")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(str(out))
    print(f"已写出 {out}（{out.stat().st_size / 1024:.0f} KB）")

    # 自检：词分布不要太集中（都落进少数几个词 ⇒ 检索没有判别力）
    w = vocab.transform(A[:5000])
    used = int(np.unique(w).size)
    top = float(np.bincount(w, minlength=vocab.n_words).max() / max(1, len(w)))
    print(f"自检：5000 行落在 {used}/{vocab.n_words} 个词上，最热词占比 {top:.1%}")
    if used < vocab.n_words * 0.2:
        print("⚠️ 词覆盖过低：判别力可能不够，试试 --iters 调大或换训练数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
