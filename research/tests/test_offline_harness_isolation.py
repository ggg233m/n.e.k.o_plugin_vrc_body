#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolation guard for research/tools/offline_timebase_harness.py.

Incident (2026-09-21): the regression gate went RED with harness checks
1/4/6/7/8 failing (rows=6, dist=30.0, n_records=600, ...). Root cause was NOT
an API change in driver_log.py: the harness reused a FIXED shared workdir
(``.tmp/offline_harness``) and only cleaned it AFTER a run. ``ActionLogRecorder``
opens the JSONL in APPEND mode, so a leftover dir from an interrupted run made
the next run accumulate rows (x3 reproduced the exact failing numbers).

The fix cleans the workdir BEFORE the run as well. These tests pin that:

  * a pre-dirtied workdir must still produce a fully passing run
  * two consecutive runs in the SAME workdir must both exit 0

They use ``--no-real-artifacts`` (synthetic only) and a temp workdir, so they
are fast and independent of large .tmp material.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
_HARNESS = os.path.join(_ROOT, "research", "tools", "offline_timebase_harness.py")


def _run_harness(workdir=None):
    cmd = [sys.executable, _HARNESS, "--no-real-artifacts"]
    if workdir is not None:
        cmd += ["--workdir", workdir]
    proc = subprocess.run(
        cmd,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _popen_harness(workdir=None):
    cmd = [sys.executable, _HARNESS, "--no-real-artifacts"]
    if workdir is not None:
        cmd += ["--workdir", workdir]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding="utf-8", errors="replace")


def _seed_spurious_rows(workdir):
    """Pretend a previous interrupted run left a header + 3 action rows."""
    os.makedirs(workdir, exist_ok=True)
    lines = ['{"protocol_version": 1, "record": "timebase", "fps": 20.0, '
             '"anchor_monotonic": 1000.0}']
    for i in range(3):
        lines.append(
            '{"record": "action", "monotonic_time": %d.0, "frame_index": %d, '
            '"frame_timestamp": %d.0, "input_command": {"forward": 1.0, '
            '"strafe": 0.0}, "actual_send_result": "sent", "driver_ack": "none", '
            '"osc_velocity": {"vx": 0.0, "vy": 0.0, "vz": 1.0}, '
            '"turn_intent": {"yaw_delta": 0.0}}' % (1000 + i, i * 20, 1000 + i))
    with open(os.path.join(workdir, "c1.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class OfflineHarnessIsolationTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="harness_iso_")

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_pre_dirtied_workdir_still_passes(self):
        _seed_spurious_rows(self.work)
        rc, out = _run_harness(self.work)
        self.assertEqual(rc, 0, msg="dirty workdir caused a false failure:\n" + out)
        self.assertIn("passed=8", out)
        self.assertIn("failed=0", out)

    def test_repeated_runs_in_same_workdir_both_pass(self):
        rc1, out1 = _run_harness(self.work)
        rc2, out2 = _run_harness(self.work)
        self.assertEqual(rc1, 0, msg="run 1 failed:\n" + out1)
        self.assertEqual(rc2, 0, msg="run 2 failed (row accumulation):\n" + out2)
        self.assertIn("failed=0", out2)

    def test_two_concurrent_harnesses_both_pass(self):
        # Concurrency-safety proof: launch TWO harness processes at the SAME time
        # with NO --workdir (each must pick its own unique dir) and assert both
        # exit 0. If the workdir were shared this would clobber/append and fail.
        p1 = _popen_harness()
        p2 = _popen_harness()
        # communicate() (not wait()+stdout) is required: with stdout=PIPE the
        # attribute is a stream, and communicate also closes the pipes.
        out1, err1 = p1.communicate(timeout=300)
        out2, err2 = p2.communicate(timeout=300)
        rc1, rc2 = p1.returncode, p2.returncode
        self.assertEqual(rc1, 0,
                         msg="concurrent run 1 failed:\n" + (out1 or "") + (err1 or ""))
        self.assertEqual(rc2, 0,
                         msg="concurrent run 2 failed:\n" + (out2 or "") + (err2 or ""))
        self.assertIn("failed=0", out1 or "")
        self.assertIn("failed=0", out2 or "")

    def test_default_uses_unique_dir(self):
        # Two sequential default runs must both pass; each uses its own unique
        # dir so there is no cross-run sharing or accumulation.
        rc1, out1 = _run_harness()
        rc2, out2 = _run_harness()
        self.assertEqual(rc1, 0, msg="default run 1 failed:\n" + out1)
        self.assertEqual(rc2, 0, msg="default run 2 failed:\n" + out2)
        self.assertIn("passed=8", out1)
        self.assertIn("failed=0", out2)

    def test_explicit_workdir_supported(self):
        # A fresh explicit --workdir must pass...
        rc, out = _run_harness(self.work)
        self.assertEqual(rc, 0, msg="explicit workdir run failed:\n" + out)
        self.assertIn("passed=8", out)
        self.assertIn("failed=0", out)

        # ...and a PRE-EXISTING explicit dir must NOT be deleted by the run
        # (owns == False rule): it is given a clean slate but preserved.
        explicit = os.path.join(self.work, "explicit_pinned")
        os.makedirs(explicit, exist_ok=True)
        sentinel = os.path.join(explicit, "user_asset.txt")
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write("do-not-delete")
        rc2, out2 = _run_harness(explicit)
        self.assertEqual(rc2, 0, msg="pre-existing explicit workdir failed:\n" + out2)
        self.assertTrue(
            os.path.isdir(explicit),
            msg="explicit workdir was deleted despite being pre-existing")
        self.assertTrue(
            os.path.exists(sentinel),
            msg="user asset inside pre-existing explicit workdir was deleted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
