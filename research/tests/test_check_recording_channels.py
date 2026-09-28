#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research/tools/check_recording_channels.py.

Builds a SYNTHETIC recorder run directory under .tmp (no real material) and
asserts the PASS/FAIL exit-code contract:

  * all channels present          -> exit 0
  * action_timeline.jsonl missing -> exit != 0 and reports FAIL
  * osc.jsonl missing VelocityX   -> FAIL (exit != 0)
"""
import json
import os
import sys
import unittest

# Make tools/ importable regardless of CWD.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, os.path.join(_ROOT, "research", "tools"))

import check_recording_channels as cc  # noqa: E402

_ANCHOR = 1000.0


def _write_run(run_dir, *, with_velocity_x=True, with_action=True,
               with_hmd_rot=True, with_anchor=True, with_offset=True):
    os.makedirs(run_dir, exist_ok=True)
    # run.json
    events = []
    if with_anchor:
        events.append({"event": "obs_start", "obs_start_monotonic": _ANCHOR})
    rj = {"events": events}
    if with_offset:
        rj["video_timebase"] = {"offset_s": 2.45}
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(rj, f)

    # osc.jsonl: interleaved VelocityX / VelocityZ packets from t=1001..1005
    with open(os.path.join(run_dir, "osc.jsonl"), "w", encoding="utf-8") as f:
        for i in range(10):
            t = _ANCHOR + 1.0 + i * 0.4
            if with_velocity_x:
                f.write(json.dumps({"t": t, "addr": "/avatar/parameters/VelocityX",
                                    "args": [0.05]}) + "\n")
            f.write(json.dumps({"t": t, "addr": "/avatar/parameters/VelocityZ",
                                "args": [0.66]}) + "\n")

    # hmd_frames.jsonl
    with open(os.path.join(run_dir, "hmd_frames.jsonl"), "w", encoding="utf-8") as f:
        for i in range(10):
            t = _ANCHOR + 1.0 + i * 0.4
            rot = [0.0, 0.0, 0.0, 1.0] if with_hmd_rot else None
            f.write(json.dumps({"t": t, "hmd": {"position": [0, 1.5, 0],
                                                "rotation_xyzw": rot}}) + "\n")

    # action_timeline.jsonl
    if with_action:
        with open(os.path.join(run_dir, "action_timeline.jsonl"), "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({
                    "monotonic_time": _ANCHOR + 1.0 + i * 0.5,
                    "frame_index": i * 30,
                    "frame_timestamp": _ANCHOR + 1.0 + i * 0.5,
                    "input_command": {"forward": 0.66},
                    "actual_send_result": "sent",
                }) + "\n")


class CheckChannelsTest(unittest.TestCase):

    def setUp(self):
        self.root = os.path.join(_ROOT, ".tmp", "check_channels_test")
        os.makedirs(self.root, exist_ok=True)
        self.run_dir = os.path.join(self.root, self._testMethodName)
        if os.path.isdir(self.run_dir):
            import shutil
            shutil.rmtree(self.run_dir)
        os.makedirs(self.run_dir)

    def test_all_present_exits_zero(self):
        _write_run(self.run_dir)
        code, text = cc.run_check(self.run_dir)
        self.assertEqual(code, 0, "expected PASS exit 0\n" + text)
        self.assertIn("RESULT: PASS", text)

    def test_missing_action_timeline_fails(self):
        _write_run(self.run_dir, with_action=False)
        code, text = cc.run_check(self.run_dir)
        self.assertNotEqual(code, 0, "missing action_timeline must fail\n" + text)
        self.assertIn("action_timeline.jsonl missing", text)
        self.assertIn("FAIL", text)

    def test_missing_velocity_x_fails(self):
        _write_run(self.run_dir, with_velocity_x=False)
        code, text = cc.run_check(self.run_dir)
        self.assertNotEqual(code, 0, "missing VelocityX must fail\n" + text)
        self.assertIn("/VelocityX present", text)
        self.assertIn("FAIL", text)

    def test_missing_hmd_rotation_fails(self):
        _write_run(self.run_dir, with_hmd_rot=False)
        code, text = cc.run_check(self.run_dir)
        self.assertNotEqual(code, 0, "missing HMD rotation must fail\n" + text)
        self.assertIn("hmd.rotation_xyzw present", text)

    def test_missing_anchor_fails(self):
        _write_run(self.run_dir, with_anchor=False)
        code, text = cc.run_check(self.run_dir)
        self.assertNotEqual(code, 0, "missing anchor must fail\n" + text)
        self.assertIn("obs_start_monotonic", text)
        self.assertIn("MISSING", text)

    def test_missing_offset_fails(self):
        _write_run(self.run_dir, with_offset=False)
        code, text = cc.run_check(self.run_dir)
        self.assertNotEqual(code, 0, "missing video offset must fail\n" + text)
        self.assertIn("video_timebase.offset_s", text)

    def test_run_dir_not_found_exits_two(self):
        code, text = cc.run_check(os.path.join(self.root, "does_not_exist"))
        self.assertEqual(code, 2)
        self.assertIn("not found", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
