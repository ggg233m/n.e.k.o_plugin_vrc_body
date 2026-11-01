# -*- coding: utf-8 -*-
"""在线 dry-run：真实 OpenVR 左目帧 → PolicyRuntime → 只记录动作块，不发任何 input_axes。

前 ``--walk-s`` 秒给 ``wp=(0, 0.8)``（应输出前进），之后给 ``wp=(0, 0)``（应被门控为零）。
本体速度暂用 ``(0,0)``（9001 可能被后端占用）；dry-run 不闭环，只验证运行环与输出合理性。

用法::

    G:\\vrc_recording\\vrc_model\\venv_xpu\\Scripts\\python.exe research/tools/dryrun_policy_live.py \\
        --run h:\\...\\.venv_xpu\\runs\\m0_s3_448\\best --seconds 120 --walk-s 60 \\
        --out h:\\...\\.venv_xpu\\runs\\dryrun.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_runner import PolicyRunner  # noqa: E402
from policy_runtime import PolicyRuntime  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3")
    ap.add_argument("--run", default=r"h:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.venv_xpu\runs\m0_s3_448\best")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--walk-s", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--graph", action="store_true",
                    help="切 Level-Zero V2 并用 XPUGraph 双图重放（需要 merge）")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args(argv)

    rt = PolicyRuntime(a.model, a.run, compile=not a.no_compile,
                       graph=a.graph, merge=not a.no_merge)
    print("[dryrun] runtime ready", flush=True)

    import openvr
    sys.path.insert(0, r"h:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
    from backend.openvr_mirror import MirrorEye, create_device, read_stereo
    openvr.init(openvr.VRApplication_Background)
    device, context = create_device()
    comp = openvr.VRCompositor()
    eyes = [MirrorEye(comp, device, context, e, gpu_downscale=(1080, 1))
            for e in (openvr.Eye_Left, openvr.Eye_Right)]
    print("[dryrun] mirror open", flush=True)

    t_start = time.perf_counter()

    def frame_source():
        pair = read_stereo(eyes[0], eyes[1])
        return None if pair is None else pair[0]

    def wp_provider():
        return (0.0, 0.8) if (time.perf_counter() - t_start) < a.walk_s else (0.0, 0.0)

    runner = PolicyRunner(rt, frame_source, wp_provider, lambda: (0.0, 0.0),
                          on_block=None, hz=a.hz, dry_run=True,
                          stale_s=0.8, log_path=a.out)
    runner.start()
    time.sleep(a.seconds)
    runner.stop()
    for m in eyes:
        m.close()

    rows = []
    with open(a.out, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    ok = [r for r in rows if "block" in r and "latency_ms" in r]
    lat = [r["latency_ms"] for r in ok]
    step0 = np.array([np.hypot(r["block"][0][0], r["block"][0][1]) for r in ok]) if ok else np.zeros(0)
    gated = np.array([bool(r.get("gated")) for r in ok]) if ok else np.zeros(0)
    summary = dict(seconds=a.seconds, hz=a.hz, samples=len(ok),
                   latency_ms=dict(p50=float(np.percentile(lat, 50)) if lat else None,
                                   p95=float(np.percentile(lat, 95)) if lat else None,
                                   max=float(max(lat)) if lat else None),
                   gated_frac=float(gated.mean()) if len(gated) else None,
                   step0_speed=dict(mean=float(step0.mean()) if len(step0) else None,
                                    p95=float(np.percentile(step0, 95)) if len(step0) else None,
                                    max=float(step0.max()) if len(step0) else None),
                   walk_s=a.walk_s, body_fixed=[0.0, 0.0])
    Path(a.out).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
