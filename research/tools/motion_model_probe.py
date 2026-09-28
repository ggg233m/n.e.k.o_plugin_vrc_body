# -*- coding: utf-8 -*-
"""Does a motion model need *acceleration*? Measure it instead of assuming.

Context: the plan for speeding up online SLAM is projection-guided descriptor
matching -- predict the next pose, project the local map, and only match
descriptors near the predicted pixel. That needs a *pose prediction*. The open
question this script settles is **which** predictor: zero-order hold (last pose),
constant velocity, or constant acceleration.

The answer decides whether the avatar's motion is smooth enough for a
second-order model to pay for itself, or whether motion is piecewise-constant
(steps) so that acceleration is an impulse and a second-order model is actively
harmful.

Two independent sources, because they answer different things:

* **Ground truth** (OSC speed + HMD yaw dead reckoning) -- what the camera
  really did. Errors are in metres.
* **Visual trajectory** (a retest run's ``trajectory.json``) -- what the SLAM
  reports frame to frame. Errors are in map units. This one includes estimator
  jitter, so it is the fair test for "would the predictor help tracking".

Usage:
    python research/tools/motion_model_probe.py --traj .tmp/slam_retest_fix7/trajectory.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / ".slam_probe" / "offline_probe" / "recorder" / "runs" / "20260920-233456"


def _load_retest():
    spec = importlib.util.spec_from_file_location(
        "neko_retest_for_motion", REPO / "research" / "tools" / "slam_metric_retest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neko_retest_for_motion"] = mod
    spec.loader.exec_module(mod)
    return mod


def _predict_errors(P: np.ndarray, mask: np.ndarray) -> dict:
    """Per-step prediction error for ZOH / CV / CA over a position series.

    ``mask`` marks the frames whose position is usable; a prediction is only
    scored when *every* frame it reads and the frame it predicts are all usable.
    """
    n = len(P)
    zoh, cv, ca = [], [], []
    for k in range(2, n - 1):
        if not (mask[k - 2] and mask[k - 1] and mask[k] and mask[k + 1]):
            continue
        p0, p1, p2, actual = P[k - 2], P[k - 1], P[k], P[k + 1]
        zoh.append(np.linalg.norm(p1 - actual))          # hold the previous pose
        cv.append(np.linalg.norm((2 * p1 - p0) - actual))  # p + (p - p_prev)
        ca.append(np.linalg.norm((2.5 * p2 - 2 * p1 + 0.5 * p0) - actual))
    out = {}
    for name, arr in (("zoh", zoh), ("cv", cv), ("ca", ca)):
        a = np.asarray(arr, dtype=float)
        out[name] = {
            "n": int(a.size),
            "mean": round(float(a.mean()), 4) if a.size else None,
            "median": round(float(np.median(a)), 4) if a.size else None,
            "p95": round(float(np.percentile(a, 95)), 4) if a.size else None,
        }
    if out["zoh"]["median"]:
        for name in ("cv", "ca"):
            out[name]["vs_zoh"] = round(
                out[name]["median"] / out["zoh"]["median"], 4)
    return out


def _osc_motion_stats(osc) -> dict:
    """Per-report speed and its frame-to-frame change, straight from the OSC log.

    ``cum_vx`` / ``cum_vz`` accumulate ``v * dt``, so differencing them recovers
    the per-report velocity without re-parsing the log.
    """
    t = np.asarray(osc.t, dtype=float)
    dt = np.diff(t)
    good = dt > 0
    vx = np.diff(np.asarray(osc.cum_vx, dtype=float)) / np.where(good, dt, np.nan)
    vz = np.diff(np.asarray(osc.cum_vz, dtype=float)) / np.where(good, dt, np.nan)
    speed = np.hypot(vx, vz)
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        return {"available": False}
    # Round to 3 dp: VRChat's velocity channel is quantised, so exact repeats are
    # meaningful and the repeat rate is the direct test for "piecewise constant".
    q = np.round(speed, 3)
    repeats = int(np.sum(np.diff(q) == 0.0))
    return {
        "available": True,
        "n_reports": int(speed.size),
        "speed_median": round(float(np.median(speed)), 4),
        "speed_p95": round(float(np.percentile(speed, 95)), 4),
        "speed_max": round(float(speed.max()), 4),
        "frac_reports_speed_unchanged": round(repeats / max(1, speed.size - 1), 4),
        "frac_speed_zero": round(float(np.mean(q == 0.0)), 4),
        "report_dt_median_s": round(float(np.median(dt[good])), 4) if good.any() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="运动模型该用几阶？用数据回答")
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--traj", default=str(REPO / ".tmp" / "slam_retest_fix7" / "trajectory.json"))
    ap.add_argument("--out", default=str(REPO / ".tmp" / "motion_model_probe.json"))
    a = ap.parse_args()

    rt = _load_retest()
    osc = rt.OscPath(a.run)
    hmd = rt.HmdYaw(a.run)

    traj = json.loads(Path(a.traj).read_text(encoding="utf-8"))
    tele = np.asarray([r["tele"] for r in traj], dtype=float)
    gt = rt.build_gt(osc, hmd, tele, -1, "zoh")
    P_gt = np.column_stack([gt["x"], gt["z"]])
    gt_valid = np.asarray(gt["valid"], dtype=bool)

    P_vis = np.asarray([[r["x"], r["z"]] for r in traj], dtype=float)
    vis_valid = np.asarray([r["state"] == "OK" for r in traj], dtype=bool)

    report = {
        "run": str(a.run),
        "traj": str(a.traj),
        "osc": _osc_motion_stats(osc),
        "gt_prediction_error_m": _predict_errors(P_gt, gt_valid),
        "visual_prediction_error_units": _predict_errors(P_vis, vis_valid),
    }
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print("运动模型该用几阶？（预测下一帧的误差，越小越好）")
    print("=" * 74)
    print("OSC 速度通道（变化驱动，静止/匀速都不发包）:")
    for k, v in report["osc"].items():
        print("  %-32s %s" % (k, v))
    print()
    for title, key, unit in (("真值（OSC+HMD 航位推算）", "gt_prediction_error_m", "m"),
                             ("视觉轨迹（SLAM 逐帧报告）", "visual_prediction_error_units", "单位")):
        d = report[key]
        print("%s —— 预测下一帧位置的中位误差 (%s):" % (title, unit))
        for name, label in (("zoh", "零阶保持（沿用上一帧位姿）"),
                            ("cv", "恒速（外推上一帧位移）"),
                            ("ca", "恒加速（二阶外推）")):
            e = d[name]
            extra = "" if "vs_zoh" not in e else "   vs ZOH ×%.3f" % e["vs_zoh"]
            print("  %-26s n=%-5d 中位 %-8s p95 %-8s%s"
                  % (label, e["n"], e["median"], e["p95"], extra))
        print()
    print("wrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
