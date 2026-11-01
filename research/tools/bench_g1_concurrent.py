# -*- coding: utf-8 -*-
"""G1 并发实测：策略头跑 **生产节拍** 的推理环，同时 LAS2 在子进程里持续出帧。

方案 §1 的 G1 门是「VRChat VR + LAS2 同跑时 p95 <= 200 ms 且 GPU 总占用 < 90%」，
R11-7 再加 p99 <= 300 ms（等价于块丢弃率 <= 1%）。之前所有数字都不是这个口径：

* latency_448_compiled.json —— 空 GPU，只有策略；
* latency_448_compiled_vrc.json —— VRChat 在跑，但 LAS2 没跑；
* R2（在线 164 ms）—— LAS2 在跑，但只记了中位，没有分位数，且是 idle 会话。

链路完全按生产接：LAS2 走 `make_stereo_backend(subprocess=True)`（与
nav_online 同一工厂、同一个 .venv_las 子进程），策略走 `PolicyRuntime.act`。
GPU 占用用 typeperf 采 Windows GPU Engine 计数器（3D 引擎求和），
因为 torch 只给得出「分配了多少」，给不出「用了多少」。

用法::

    .venv_xpu/Scripts/python.exe research/tools/bench_g1_concurrent.py \
        --graph --las --seconds 120 --out .venv_xpu/runs/g1_concurrent_graph.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT))


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


class GpuUtil:
    """typeperf 采 GPU Engine 利用率：3D + Compute 两类引擎求和。

    Compute 必须算进来 —— LAS2 就是跑在 Compute 引擎上的（实测占 83%）。
    只取 3D 会得出「GPU 空闲」的错误结论。
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._th = None
        self.error = None
        self.channels = 0

    def start(self):
        self._th = threading.Thread(target=self._run, daemon=True, name="gpu-util")
        self._th.start()

    def _run(self):
        import subprocess
        import tempfile

        csv = Path(tempfile.gettempdir()) / ("gpu_util_%d.csv" % int(time.time() * 1000))
        try:
            p = subprocess.Popen(
                ["typeperf", r"\GPU Engine(*)\Utilization Percentage",
                 "-sc", "100000", "-si", str(int(max(1, self.interval))), "-f", "CSV",
                 "-o", str(csv)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            while not self._stop.wait(self.interval):
                pass
            p.terminate()
            try:
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                p.kill()
            if csv.exists():
                lines = csv.read_text(errors="replace").splitlines()
                # ⚠️ 三个坑，全部实测踩过（每一个都让结果恒为 0%，看起来像「GPU 没干活」）：
                #   ① **只有一行表头**（lines[0]），lines[1:] 全是数据。
                #      早先按「两行表头」写，把第一行数据当表头，keep 永远为空。
                #   ② 计数器名是 engtype_3D（大写 D），小写比较才对得上。
                #   ③ **LAS2 跑在 engtype_compute，VRChat 跑在 engtype_3d**。
                #      只取 3d 时 LAS2 那 83% 完全看不见，求和恒 0。
                hdr = [h.lower() for h in (lines[0].split(",") if lines else [])]
                keep = [i for i, h in enumerate(hdr)
                        if "engtype_3d" in h or "engtype_compute" in h]
                self.channels = len(keep)
                for row in lines[1:]:
                    vals = row.split(",")
                    tot = 0.0
                    for i in keep:
                        if i < len(vals):
                            try:
                                tot += float(vals[i].strip('"') or 0)
                            except ValueError:
                                pass
                    self.samples.append(tot)
                csv.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)[:200]

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=25)

    def summary(self):
        if not self.samples:
            return {"error": self.error or "no samples"}
        s = self.samples
        return dict(n=len(s), channels=self.channels,
                    p50=pct(s, 50), p95=pct(s, 95), max=max(s),
                    mean=float(np.mean(s)))


def las2_load(left, right, stop_evt, out: dict, timeout_s: float = 5.0):
    """后台线程：持续 compute，与 nav_online 的 stereo worker 同一条路。"""
    from backend.stereo_backend import make_stereo_backend, close_las2

    t0 = time.perf_counter()
    try:
        be = make_stereo_backend("las2", subprocess=True, shape=left.shape[:2],
                                 fx=303.75, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        out["error"] = "spawn: %r" % (exc,)
        return
    out["spawn_s"] = round(time.perf_counter() - t0, 1)
    out["ready"] = True
    lat, busy, err = [], 0, 0
    while not stop_evt.is_set():
        t = time.perf_counter()
        try:
            be.compute(left, right)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "StereoBusy":
                busy += 1
            else:
                err += 1
            out["last_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            time.sleep(0.05)
            continue
        lat.append((time.perf_counter() - t) * 1000)
        # ⚠️ 必须**逐帧**发布计数：父进程靠 out["frames"] > 0 判断「负载已经上来」。
        # 只在退出时写一次的话，父进程永远等到超时，然后把一次成功的并发测
        # 误判成「LAS2 未出帧、数据无效」（实测踩过，浪费 7 min）。
        out["frames"] = len(lat)
        if len(lat) == 1:
            out["first_frame_s"] = round(time.perf_counter() - t0, 1)
    try:
        close_las2()
    except Exception:  # noqa: BLE001
        pass
    out["frames"] = len(lat)
    out["busy"] = busy
    out["errors"] = err
    if lat:
        out["las_ms"] = dict(p50=pct(lat, 50), p95=pct(lat, 95), max=max(lat))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--frames", default="navmesh_recordings/20261031_001200/kf_img")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--baseline-seconds", type=float, default=0.0,
                    help=">0：开表前先采同样秒数的「无策略环」基线 GPU 占用")
    ap.add_argument("--hz", type=float, default=1.0, help="生产节拍 VRC_POLICY_HZ")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--las", action="store_true", help="同时在子进程里跑 LAS2")
    ap.add_argument("--las-warm", type=float, default=420.0,
                    help="等 LAS2 出第一帧的上限秒数（首帧冷编译实测可到 5 min）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    import torch
    from PIL import Image
    from policy_runtime import PolicyRuntime

    paths = sorted(glob.glob(str(Path(a.frames) / "*_L.png")))[:40]
    frames = [np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in paths]
    rights = [np.asarray(Image.open(p.replace("_L.png", "_R.png")).convert("RGB"), np.uint8)
              for p in paths] if a.las else None

    print("[bench] 加载 PolicyRuntime graph=%s …" % a.graph, flush=True)
    rt = PolicyRuntime(a.model, a.run, size=(448, 256), max_len=192, graph=a.graph)
    print("[bench] graph_ready=%s" % getattr(rt, "graph_ready", False), flush=True)
    wp, body = (0.0, 0.8), (0.0, 2.8)

    stop_evt = threading.Event()
    las_out: dict = {}
    las_th = None
    if a.las and rights:
        las_th = threading.Thread(target=las2_load, args=(frames[0], rights[0], stop_evt, las_out),
                                  daemon=True, name="las2-load")
        las_th.start()
        # 不能死等固定秒数：首帧冷编译实测 >100 s（子进程要载权重 + 编 Triton 内核），
        # 睡不够就开表，等于测「LAS2 在编译时策略有多慢」，那不是 G1 要的口径。
        # 改为等第一帧真的算出来 —— las_out["frames"] > 0 才认为负载上来了。
        print("[bench] 等 LAS2 出第一帧（最多 %.0f s）…" % a.las_warm, flush=True)
        waited = 0.0
        while waited < a.las_warm and las_out.get("frames", 0) == 0 \
                and "error" not in las_out:
            time.sleep(1.0)
            waited += 1.0
        if las_out.get("frames", 0) == 0:
            print("[bench] ⚠️ LAS2 未出帧（busy=%s err=%s），本次并发数据无效"
                  % (las_out.get("busy"), las_out.get("last_error")), flush=True)
        else:
            print("[bench] LAS2 已出 %d 帧，开始计时" % las_out["frames"], flush=True)

    # 先跑一段**同样负载、只差策略环**的基线，用同一套计数器量 GPU 占用。
    # 为什么要它：Windows 的 GPU Engine 是「按进程×引擎」的利用率，3D 与
    # compute 可以同时忙，求和天然会超过 100%（实测 109%）。拿它去比方案里的
    # 「总占用 < 90%」是跨口径比较。同口径的前后差才是能说的数。
    util_base = GpuUtil(1.0)
    if a.baseline_seconds > 0:
        util_base.start()
        tb_end = time.perf_counter() + a.baseline_seconds
        while time.perf_counter() < tb_end:
            time.sleep(0.5)
        util_base.stop()

    util = GpuUtil(1.0)
    util.start()

    lat, n = [], 0
    t_end = time.perf_counter() + a.seconds
    period = 1.0 / max(0.1, a.hz)
    next_t = time.perf_counter()
    while time.perf_counter() < t_end:
        f = frames[n % len(frames)]
        p = frames[(n - 1) % len(frames)]
        if rt.device == "xpu":
            torch.xpu.synchronize()
        t = time.perf_counter()
        rt.act(f, p, wp, body)
        if rt.device == "xpu":
            torch.xpu.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
        n += 1
        next_t = max(next_t + period, time.perf_counter())
        time.sleep(max(0.0, next_t - time.perf_counter()))

    util.stop()
    stop_evt.set()
    if las_th is not None:
        las_th.join(timeout=20)

    gpus = util.summary()
    ms = lat
    res = dict(tag=a.tag, seconds=a.seconds, hz=a.hz, samples=len(ms),
               graph=bool(a.graph), graph_ready=bool(getattr(rt, "graph_ready", False)),
               las=a.las, las_stats=las_out or None,
               act_ms=dict(p50=pct(ms, 50), p95=pct(ms, 95), p99=pct(ms, 99), max=max(ms)),
               gpu_util_pct=gpus,
               gpu_util_baseline=(util_base.summary() if a.baseline_seconds > 0 else None),
               gate=dict(p95_le_200=bool(pct(ms, 95) <= 200.0),
                         p99_le_300=bool(pct(ms, 99) <= 300.0),
                         gpu_lt_90=bool(isinstance(gpus.get("p95"), float) and gpus["p95"] < 90.0)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
