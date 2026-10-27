# -*- coding: utf-8 -*-
r"""建图线程单帧成本的组件级实测：落盘 / 回环 / 入图 / 栅格化。

为什么需要它：`map_backlog.py` 能从 mtime 反演出**总服务时间**（585 ms/帧），但反演不出
**它花在哪**。要判「该降分辨率还是该改批大小」必须拆开量，不能靠估。

三块各自独立可测：
1. **落盘** `NavRecorder.keyframe()` —— 1 个 npz + 2 张 PNG；`--width 720` 会先把原图
   下采样再写，用来量分辨率代价（2026-10-28 实测 1080 → 720 省 69 ms/帧，+106%）。
2. **回环** `LoopCloser.add_keyframe()` —— 含 BowIndex 查询，随帧数线性涨。
3. **入图** `KeyframeGridMapper.add_keyframe()` —— 纯累加，很便宜。

⚠️ **栅格化不在这里**：它由 `_mapping_loop` 按批调用 `NavSession.compute()`，成本取决于
**每批几帧**（固定开销 ~110 ms/次），见 `--batch` 那一节。

用法：
    python research/tools/kf_io_bench.py 20261026_184318
    python research/tools/kf_io_bench.py 20261026_184318 --width 720   # 量分辨率代价
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "tools"))

from backend.nav_loop import LoopCloser                                    # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig, NavSession  # noqa: E402
from backend.nav_online import SessionRecorder                            # noqa: E402
from loop_replay import features_of, recorded_loop_config                # noqa: E402


def bench_write(rec: Path, ks: list[int], width: int | None, reps: int = 2) -> float:
    """走**真实** `SessionRecorder.keyframe()` 落盘一次的墙钟（ms，中位）。

    `width` 非空时先把原图缩到该宽度再交给录制器 —— 除分辨率外**其余完全不变**，
    所以两者的差就是「采集宽度」这一个变量的代价。
    返回 0 表示该场没有 `kf_img/`。
    """
    if not (rec / "kf_img").is_dir():
        return 0.0
    tmp = Path(tempfile.mkdtemp())
    try:
        recorder = SessionRecorder(tmp, max_bytes=1e12, meta={"probe": True}, images=True)
        times: list[float] = []
        for _ in range(reps):
            for k in ks:
                with np.load(rec / "kf" / f"{k:06d}.npz") as z:
                    pts = np.asarray(z["pts"], np.float32)
                    T_dr = np.asarray(z["T_dr"], np.float64)
                    T_map = np.asarray(z["T_map"], np.float64)
                    dist = float(z["dist_m"])
                    feat = features_of(z)
                imgs = []
                for eye in "LR":
                    p = rec / "kf_img" / f"{k:06d}_{eye}.png"
                    if not p.exists():
                        return 0.0
                    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if width and im.shape[1] != width:
                        h = int(round(im.shape[0] * width / im.shape[1]))
                        im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
                    imgs.append(im)      # 已是 BGR（imread 惯例），录制器再 cvtColor 会反色
                t0 = time.perf_counter()
                if imgs[0].ndim == 3:
                    # kf_img 落盘是 RGB→BGR 转过的，读回来是 BGR；这里补一次 BGR→RGB
                    # 让喂进去的仍是 RGB，与在线路径一致（转置开销可忽略，不影响结论）。
                    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
                recorder.keyframe(k, pts, T_dr, T_map, dist, feat, images=tuple(imgs))
                times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(times))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec", help="录制目录名（navmesh_recordings/ 下）")
    ap.add_argument("--span", type=int, default=100, help="取多少帧做统计")
    ap.add_argument("--start", type=int, default=120, help="起始帧（避开开场 cam_h 未收敛段）")
    ap.add_argument("--width", type=int, default=None,
                    help="落盘前把原图缩到这个宽度（量分辨率代价；不写=原样）")
    ap.add_argument("--batch", nargs="+", type=int, default=None,
                    help="额外测「每批 N 帧后一次 compute」的代价，如 --batch 1 8")
    a = ap.parse_args()

    rec = ROOT / "navmesh_recordings" / a.rec
    if not rec.is_dir():
        raise SystemExit(f"[fail] 没有这个录制：{rec}")
    ks = [k for k in range(a.start, a.start + a.span)
          if (rec / "kf" / f"{k:06d}.npz").exists()]
    if len(ks) < 5:
        raise SystemExit(f"[fail] 可用帧太少（{len(ks)}）")

    cfg, _raw = recorded_loop_config(rec)
    closer = LoopCloser(cfg)
    mapper = KeyframeGridMapper(MapperConfig())

    def load(k: int):
        with np.load(rec / "kf" / f"{k:06d}.npz") as z:
            return (features_of(z), np.asarray(z["T_dr"], np.float64),
                    float(z["dist_m"]), np.asarray(z["pts"], np.float32))

    t_loop, t_map = [], []
    for k in ks:
        feat, T, d, pts = load(k)
        t0 = time.perf_counter()
        closer.add_keyframe(k, T, d, feat)
        t1 = time.perf_counter()
        mapper.add_keyframe(k, pts, closer.pose(k), osc_dist_m=d, t_s=float(k))
        t2 = time.perf_counter()
        t_loop.append((t1 - t0) * 1000.0)
        t_map.append((t2 - t1) * 1000.0)

    t_wr = bench_write(rec, ks[:12], a.width)

    print(f"\n=== {a.rec} ===  k={ks[0]}..{ks[-1]}（{len(ks)} 帧）")
    print(f"  LoopCloser.add_keyframe  中位 {np.median(t_loop):7.1f} ms   p90 {np.percentile(t_loop, 90):7.1f}")
    print(f"  mapper.add_keyframe      中位 {np.median(t_map):7.1f} ms   p90 {np.percentile(t_map, 90):7.1f}")
    if t_wr:
        tag = f"{a.width}px" if a.width else "原样"
        print(f"  SessionRecorder.keyframe 中位 {t_wr:7.1f} ms   （{tag}，1 npz + 2 PNG）")
    else:
        print("  SessionRecorder.keyframe 跳过（该场没有 kf_img/）")
    print(f"  ── 可归因小计 {np.median(t_loop) + np.median(t_map) + t_wr:.1f} ms")
    print("     其余归栅格化与锁/GIL（用 --batch 量，或用 map_backlog.py 反演总服务时间后相减）")

    if a.batch:
        print()
        print("  ── 每批 N 帧后的 compute（rasterize+build+plan）──")
        est = {"state": "localized", "xy": (0.0, 0.0), "yaw": 0.0}
        for n in a.batch:
            m2 = KeyframeGridMapper(MapperConfig())
            c2 = LoopCloser(cfg)
            sess = NavSession(m2, radius_m=0.25)
            for k in range(0, a.start):
                feat, T, d, pts = load(k)
                c2.add_keyframe(k, T, d, feat)
                m2.add_keyframe(k, pts, c2.pose(k), osc_dist_m=d, t_s=float(k))
            sess.compute(est, sess.snapshot())          # 预热
            times: list[float] = []
            k = a.start
            while k + n <= a.start + a.span:
                for j in range(k, k + n):
                    feat, T, d, pts = load(j)
                    c2.add_keyframe(j, T, d, feat)
                    m2.add_keyframe(j, pts, c2.pose(j), osc_dist_m=d, t_s=float(j))
                t0 = time.perf_counter()
                sess.compute(est, sess.snapshot())
                times.append((time.perf_counter() - t0) * 1000.0)
                k += n
            if times:
                print(f"     每批 {n:2d} 帧 -> {np.median(times):7.1f} ms/次")
        print("     判读：每批 1 帧的**固定开销**被按帧放大 ⇒ 平均服务时间上升，但单次最坏延迟下降。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())