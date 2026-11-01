# -*- coding: utf-8 -*-
"""G1 实测：PolicyRuntime 端到端延迟（p50/p95/p99），按生产口径。

与 bench_policy_latency.py 的区别（那个测的是训练侧 batch，不是生产路径）：

* 输入是真机 1080x606 原始帧，走 to_policy_rgb + processor，与在线逐字一致；
* 测的是 PolicyRuntime.act() 本身，含预处理、前向、head、D2H；
* 可选 --graph 走 XPUGraph 双图重放，这是本轮要裁决的那条路。

G1 的两条门（方案 §1）：p95 <= 200 ms 且总占用 < 90%；块丢弃率 <= 1% 等价 p99 <= 300 ms。
本脚本只量延迟那一半，GPU 总占用要靠另测（本机没有 xpu-smi）。

用法::

    .venv_xpu/Scripts/python.exe research/tools/bench_policy_g1.py \
        --frames navmesh_recordings/20261031_001200/kf_img \
        --out .venv_xpu/runs/g1_graph.json --graph --iters 200
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def load_frames(pattern: str, limit: int):
    """读 1080x606 原始帧。不缩放 —— 生产给的就是全分辨率。"""
    from PIL import Image

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("no frames matched %r" % pattern)
    frames = []
    for p in paths[:limit]:
        frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
    h, w = frames[0].shape[:2]
    print("[frames] %d 张 %dx%d（源 %s）" % (len(frames), w, h, paths[0]), flush=True)
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--frames", default="navmesh_recordings/20261031_001200/kf_img")
    ap.add_argument("--frame-glob", default="*_L.png")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256])
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--graph", action="store_true", help="XPUGraph 双图重放（需 Level-Zero V2）")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="不合并 LoRA（只作对照）")
    ap.add_argument("--check-equiv", type=int, default=0,
                    help=">0：额外跑这么多拍，比图重放与 eager 的动作块 cos（方案 §6.1）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    import torch
    from policy_runtime import PolicyRuntime

    frames = load_frames(str(Path(a.frames) / a.frame_glob), a.max_frames)

    print("[runtime] graph=%s compile=%s merge=%s"
          % (a.graph, not a.no_compile, not a.no_merge), flush=True)
    t_setup0 = time.perf_counter()
    rt = PolicyRuntime(a.model, a.run, size=tuple(a.size), max_len=a.max_len,
                       compile=not a.no_compile, graph=a.graph, merge=not a.no_merge)
    setup_s = time.perf_counter() - t_setup0

    graph_ready = bool(getattr(rt, "graph_ready", False))

    def _dual_stats():
        if getattr(rt, "dual", None) is None:
            return None
        try:
            return rt.dual.stats()
        except Exception as exc:  # noqa: BLE001
            return {"stats_error": repr(exc)}

    print("[runtime] setup %.1fs graph_ready=%s dual=%s" % (setup_s, graph_ready, _dual_stats()),
          flush=True)

    # 生产口径：wp 恒在 lookahead 处（VRC_POLICY_WP_MIN=0.8 的效果），body 取行走中值。
    wp = (0.0, 0.8)
    body = (0.0, 2.8)

    def one(i: int) -> float:
        f = frames[i % len(frames)]
        prev = frames[(i - 1) % len(frames)]
        if rt.device == "xpu":
            torch.xpu.synchronize()
        t = time.perf_counter()
        rt.act(f, prev, wp, body)
        if rt.device == "xpu":
            torch.xpu.synchronize()
        return time.perf_counter() - t

    for i in range(a.warmup):
        one(i)
    if rt.device == "xpu":
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()

    times = []
    for i in range(a.iters):
        times.append(one(a.warmup + i))

    # 预处理单独量一遍：同一批帧的 to_policy_rgb + build_inputs（不含前向）
    pre = []
    for i in range(min(50, a.iters)):
        f = frames[i % len(frames)]
        p = frames[(i - 1) % len(frames)]
        if rt.device == "xpu":
            torch.xpu.synchronize()
        t = time.perf_counter()
        rt.build_inputs(f, p, wp, body)
        if rt.device == "xpu":
            torch.xpu.synchronize()
        pre.append(time.perf_counter() - t)

    # 计数必须在跑完之后取：setup 刚结束那会儿 replays 还是 0（这一步曾经把
    # 「图真的被 replay 了」误读成「图没生效」）。
    dual_stats = _dual_stats()

    # 图重放 vs eager 的等价性。**必须先证明这张图算的是同一件事**，
    # 否则「124 ms 过了门槛」只是在给一个错误的答案掐表（方案 §6.1 的 cos 门）。
    equiv = None
    if a.check_equiv > 0 and getattr(rt, "dual", None) is not None and rt.graph_ready:
        import torch as _t

        def _run(use_graph: bool, i: int):
            f = frames[i % len(frames)]
            p = frames[(i - 1) % len(frames)]
            rt.dual.ready = use_graph
            with _t.no_grad():
                out = rt.act(f, p, wp, body)
            rt.dual.ready = True
            return np.asarray(out["block"], dtype=np.float64).reshape(-1)

        cos_list, absmax_list = [], []
        for i in range(a.check_equiv):
            g = _run(True, i)
            e = _run(False, i)
            denom = (np.linalg.norm(g) * np.linalg.norm(e))
            cos_list.append(float(g @ e / denom) if denom > 0 else 1.0)
            absmax_list.append(float(np.abs(g - e).max()))
        equiv = dict(n=a.check_equiv,
                     cos_min=min(cos_list), cos_mean=float(np.mean(cos_list)),
                     absmax_max=max(absmax_list),
                     gate_cos_0_999=bool(min(cos_list) >= 0.999))

    peak = (torch.xpu.max_memory_allocated() / 2 ** 30) if rt.device == "xpu" else None
    ms = [x * 1000 for x in times]
    res = dict(
        tag=a.tag, device=rt.device, iters=a.iters, warmup=a.warmup,
        size=list(a.size), graph=bool(a.graph), graph_ready=graph_ready,
        compiled=not a.no_compile, merged=not a.no_merge,
        setup_seconds=round(setup_s, 1),
        act_ms=dict(p50=pct(ms, 50), p95=pct(ms, 95), p99=pct(ms, 99), max=max(ms),
                    mean=float(np.mean(ms))),
        preprocess_ms=dict(p50=pct(pre, 50) * 1000, p95=pct(pre, 95) * 1000),
        peak_gib=round(peak, 2) if peak else None,
        dual=dual_stats,
        equiv=equiv,
        gate=dict(p95_le_200=bool(pct(ms, 95) <= 200.0),
                  p99_le_300=bool(pct(ms, 99) <= 300.0)),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
