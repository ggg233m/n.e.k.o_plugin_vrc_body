#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that every recording channel actually wrote data for a recorder run dir.

This is a READ-ONLY gate meant to run BEFORE a full controlled route recording.
It fails (non-zero exit) if any channel is missing or broken, so an operator does
not end up with "no action log" material like the historical run
(runs/20260920-233456 had no action_timeline.jsonl and a hard-coded square route).

Usage:
    python research/tools/check_recording_channels.py --run <run_dir>

Per-channel result is one of PASS / WARN / FAIL.
Exit code is 0 only when there are zero FAILs (WARN does not block).
Any "missing" is reported as FAIL -- a missing channel is never treated as OK.

The OSC freshness numbers (total / total_stop / max_gap / coverage_frac) are
imported read-only from recorder/run_motion.py::OscPath. The recorder code is
never modified by this tool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Locate the recorder package (READ-ONLY import of run_motion.OscPath).
_RECORDER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "recorder")
if os.path.isdir(_RECORDER_DIR):
    sys.path.insert(0, _RECORDER_DIR)

from run_motion import OscPath, load_anchor  # noqa: E402


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


class _Report:
    def __init__(self):
        self.lines = []
        self.n_fail = 0
        self.n_warn = 0

    def channel(self, name):
        self.lines.append("")
        self.lines.append("=== %s ===" % name)

    def row(self, status, label, detail=""):
        if status == FAIL:
            self.n_fail += 1
        elif status == WARN:
            self.n_warn += 1
        self.lines.append("  [%s] %s%s" % (status, label, ("  " + detail) if detail else ""))

    def text(self):
        return "\n".join(self.lines)


def _read_jsonl(path):
    """Yield parsed dicts from a jsonl file, skipping blank / malformed lines.

    Returns a list so callers can count and re-scan. Files are only read.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _count_velocity(path):
    """Count rows and VelocityX / VelocityZ packets in osc.jsonl."""
    n_total = 0
    n_vx = 0
    n_vz = 0
    t_vx = []
    t_vz = []
    for r in _read_jsonl(path):
        n_total += 1
        addr = r.get("addr", "")
        t = r.get("t")
        if addr.endswith("/VelocityX"):
            n_vx += 1
            if t is not None:
                t_vx.append(float(t))
        elif addr.endswith("/VelocityZ"):
            n_vz += 1
            if t is not None:
                t_vz.append(float(t))
    return n_total, n_vx, n_vz, t_vx, t_vz


def check_osc(run_dir, rep):
    rep.channel("osc.jsonl")
    path = os.path.join(run_dir, "osc.jsonl")
    if not os.path.exists(path):
        rep.row(FAIL, "osc.jsonl missing", "no OSC telemetry at all")
        return
    n_total, n_vx, n_vz, t_vx, t_vz = _count_velocity(path)
    rep.row(PASS if n_total else FAIL, "osc.jsonl rows", "%d" % n_total)
    vx_ok = n_vx > 0
    vz_ok = n_vz > 0
    rep.row(PASS if vx_ok else FAIL, "/VelocityX present", "%d packets" % n_vx)
    rep.row(PASS if vz_ok else FAIL, "/VelocityZ present", "%d packets" % n_vz)
    if not (vx_ok and vz_ok):
        return

    # Freshness via OscPath (read-only import).
    try:
        p = OscPath(run_dir)
    except Exception as e:
        rep.row(FAIL, "OscPath build failed", str(e))
        return

    if len(p.t):
        dur = float(p.t[-1] - p.t[0])
        rep.row(PASS, "coverage duration", "%.2f s" % dur)
        rep.row(PASS, "total distance (zoh)", "%.3f m" % p.total)
        rep.row(PASS, "total distance (stop)", "%.3f m" % p.total_stop)
        rep.row(PASS, "max gap", "%.3f s" % p.max_gap)
        cov = p.coverage_frac(p.t[0], p.t[-1])
        rep.row(PASS, "coverage_frac", "%.3f" % cov)
        fr = p.freshness(p.t[0], p.t[-1])
        rep.row(PASS, "freshness",
                "zoh=%.3f stop=%.3f spread=%.3f max_gap=%.3f cov=%.3f"
                % (fr["dist_zoh_m"], fr["dist_stop_m"], fr["dist_spread_m"],
                   fr["max_gap_s"], fr["coverage_frac"]))
        # First Velocity packet latency vs the video/telemetry anchor.
        anchor = load_anchor(run_dir)
        if anchor is not None:
            first_vel = min((min(t_vx) if t_vx else 1e18),
                            (min(t_vz) if t_vz else 1e18))
            latency = float(first_vel - anchor)
            rep.row(PASS, "first velocity latency",
                    "%.3f s (channel is NaN before this)" % latency)
        # Warnings: large ZOH extrapolation vs stop, or sparse coverage.
        if p.total_stop > 1.0 and (p.total - p.total_stop) > 0.5 * p.total_stop:
            rep.row(WARN, "large ZOH extrapolation",
                    "zoh-stop spread=%.3f m (verify movement vs stop)" % (p.total - p.total_stop))
        if dur > 5.0 and cov < 0.3:
            rep.row(WARN, "sparse OSC coverage",
                    "coverage_frac=%.3f (OK if subject was mostly stationary)" % cov)


def check_hmd(run_dir, rep):
    rep.channel("hmd_frames.jsonl")
    path = os.path.join(run_dir, "hmd_frames.jsonl")
    if not os.path.exists(path):
        rep.row(FAIL, "hmd_frames.jsonl missing", "no HMD pose channel")
        return
    n = 0
    has_rot = False
    ts = []
    for r in _read_jsonl(path):
        n += 1
        rot = (r.get("hmd") or {}).get("rotation_xyzw")
        if rot:
            has_rot = True
        if r.get("t") is not None:
            ts.append(float(r["t"]))
    rep.row(PASS if n else FAIL, "frames", "%d" % n)
    rep.row(PASS if has_rot else FAIL, "hmd.rotation_xyzw present", "")
    if len(ts) >= 2:
        rep.row(PASS, "frame span", "%.2f s" % (max(ts) - min(ts)))


def check_run_json(run_dir, rep):
    rep.channel("run.json (timebase)")
    path = os.path.join(run_dir, "run.json")
    if not os.path.exists(path):
        rep.row(FAIL, "run.json missing", "no run metadata")
        return
    try:
        with open(path, encoding="utf-8") as fh:
            rj = json.load(fh)
    except Exception as e:
        rep.row(FAIL, "run.json unreadable", str(e))
        return
    anchor = None
    for ev in rj.get("events", []) or []:
        if isinstance(ev, dict) and ev.get("obs_start_monotonic") is not None:
            anchor = float(ev["obs_start_monotonic"])
            break
    if anchor is None:
        anchor = rj.get("obs_start_monotonic")
    rep.row(PASS if anchor is not None else FAIL,
            "events[].obs_start_monotonic",
            "non-null (timebase OK)" if anchor is not None
            else "MISSING -- video/OSC/HMD cannot be aligned")


def check_action_timeline(run_dir, rep):
    rep.channel("action_timeline.jsonl")
    path = os.path.join(run_dir, "action_timeline.jsonl")
    if not os.path.exists(path):
        rep.row(FAIL, "action_timeline.jsonl missing",
                "historical pitfall: B/C arms get no route-history input")
        return
    rows = _read_jsonl(path)
    if not rows:
        rep.row(FAIL, "action_timeline.jsonl empty", "0 rows")
        return
    rep.row(PASS, "rows", "%d" % len(rows))
    required = ["monotonic_time", "frame_index", "frame_timestamp",
                "input_command", "actual_send_result"]
    missing_any = False
    for k in required:
        present = sum(1 for r in rows if k in r)
        if present < len(rows):
            missing_any = True
            rep.row(WARN, "field %s" % k, "%d/%d rows" % (present, len(rows)))
    if not missing_any:
        rep.row(PASS, "required fields present", " ".join(required))


def check_video(run_dir, rep):
    rep.channel("video (video_timebase)")
    path = os.path.join(run_dir, "run.json")
    if not os.path.exists(path):
        rep.row(FAIL, "run.json missing", "cannot read video_timebase")
        return
    try:
        with open(path, encoding="utf-8") as fh:
            rj = json.load(fh)
    except Exception as e:
        rep.row(FAIL, "run.json unreadable", str(e))
        return
    vt = rj.get("video_timebase") or {}
    off = vt.get("offset_s")
    if off is None:
        rep.row(FAIL, "video_timebase.offset_s missing",
                "video<->telemetry alignment broken")
    else:
        rep.row(PASS, "video_timebase.offset_s", "%.3f s" % float(off))


def run_check(run_dir):
    """Return (exit_code, report_text)."""
    rep = _Report()
    if not os.path.isdir(run_dir):
        rep.lines.append("ERROR: run dir not found: %s" % run_dir)
        return 2, rep.text()
    check_osc(run_dir, rep)
    check_hmd(run_dir, rep)
    check_run_json(run_dir, rep)
    check_action_timeline(run_dir, rep)
    check_video(run_dir, rep)
    rep.lines.append("")
    rep.lines.append("SUMMARY: %d FAIL, %d WARN" % (rep.n_fail, rep.n_warn))
    if rep.n_fail:
        rep.lines.append("RESULT: FAIL (exit 1)")
    elif rep.n_warn:
        rep.lines.append("RESULT: PASS with WARN (exit 0)")
    else:
        rep.lines.append("RESULT: PASS (exit 0)")
    return (1 if rep.n_fail else 0), rep.text()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-channel health check for a recorder run directory.")
    ap.add_argument("--run", required=True, help="recorder run directory")
    args = ap.parse_args(argv)
    code, text = run_check(args.run)
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
