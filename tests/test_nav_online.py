from __future__ import annotations

import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend import nav_online
from neko_anyadance_body.backend.nav_online import (DeadReckoner, OnlineNavConfig, OnlineNavigator,
                                                    SessionRecorder, hmd_to_base_rotation, near_obstacle)

S = 0.755
CAM_H = 1.73


def hmd_yaw(deg: float) -> np.ndarray:
    """SteamVR 站立系（x 右 y 上 z 后）绕 +y 转 deg：从上往下看是逆时针（左转）。"""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def osc(ts: float, vx: float, vz: float) -> dict:
    return {"timestamp": ts, "velocity_x": vx, "velocity_z": vz}


class DeadReckonerTest(unittest.TestCase):
    def test_hmd_yaw_left_is_ccw_in_base(self) -> None:
        r = hmd_to_base_rotation(hmd_yaw(90.0))
        np.testing.assert_allclose(r[:2, 0], [0.0, 1.0], atol=1e-9)   # 前方转到 +y（左）

    def test_forward_walk_output_leads_by_velocity_times_lag(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.13)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        T = dr.update(1.0, r, [osc(0.0, 0.0, 1.0)])
        self.assertAlmostEqual(T[0, 3] * S, 1.13, places=6)   # 积分 1.0 + 1 m/s × 0.13 s
        self.assertAlmostEqual(T[1, 3], 0.0, places=9)
        self.assertAlmostEqual(dr.dist_m, 1.0, places=6)      # 路程不含超前量
        self.assertEqual(dr.samples, 1)                     # 重复样本不重复计

    def test_lag_compensation_survives_late_samples(self) -> None:
        # 旧实现把积分终点推到 now+lag，迟到样本又从 now+lag 起生效，两者抵消：换 lag 输出不变。
        r = hmd_to_base_rotation(np.eye(3))
        out = []
        for lag in (0.0, 0.2):
            dr = DeadReckoner(S, osc_lag_s=lag)
            dr.update(0.0, r, [osc(0.0, 0.0, 0.0)])
            dr.update(1.0, r, [osc(0.95, 0.0, 2.0)])       # 每个样本都晚 50 ms 才到
            out.append(dr.update(2.0, r, [osc(1.95, 0.0, 2.0)])[0, 3] * S)
        self.assertAlmostEqual(out[1] - out[0], 2.0 * 0.2, places=6)

    def test_stop_removes_lead(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.2)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        T = dr.update(1.0, r, [osc(1.0, 0.0, 0.0)])
        self.assertAlmostEqual(T[0, 3] * S, 1.0, places=6)

    def test_strafe_right_goes_to_negative_y(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 1.0, 0.0)])
        T = dr.update(0.5, r, [])
        self.assertAlmostEqual(T[1, 3] * S, -0.5, places=6)

    def test_heading_rotates_velocity(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(hmd_yaw(90.0))
        dr.update(0.0, r, [osc(0.0, 0.0, 2.0)])
        T = dr.update(1.0, r, [])
        np.testing.assert_allclose(T[:2, 3] * S, [0.0, 2.0], atol=1e-9)

    def test_velocity_change_mid_interval_and_late_sample_ignored(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(np.eye(3))
        dr.update(0.0, r, [osc(0.0, 0.0, 1.0)])
        dr.update(1.0, r, [osc(0.5, 0.0, 0.0)])            # 0.5 s 时停下
        T = dr.update(2.0, r, [osc(0.2, 0.0, 5.0)])        # 迟到样本：不回改、不生效
        self.assertAlmostEqual(T[0, 3] * S, 0.5, places=6)
        self.assertEqual(dr.samples, 2)

    def test_non_finite_velocity_skipped(self) -> None:
        dr = DeadReckoner(S, osc_lag_s=0.0)
        r = hmd_to_base_rotation(np.eye(3))
        T = dr.update(1.0, r, [osc(0.1, float("nan"), 1.0)])
        self.assertEqual(float(T[0, 3]), 0.0)
        self.assertEqual(dr.samples, 0)


def ground_and_wall(wall_x_track: float | None) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(0.3, 3.0, 0.05), np.arange(-1.5, 1.5, 0.05))
    pts = [np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -CAM_H)])]
    if wall_x_track is not None:
        wy, wz = np.meshgrid(np.arange(-0.5, 0.5, 0.03), np.arange(-CAM_H + 0.5, -CAM_H + 1.2, 0.05))
        pts.append(np.column_stack([np.full(wy.size, wall_x_track), wy.ravel(), wz.ravel()]))
    return np.vstack(pts).astype(np.float32)


class NearObstacleTest(unittest.TestCase):
    def kw(self) -> dict:
        return dict(cam_h=CAM_H, scale=S, ahead_m=0.7, half_width_m=0.25, ground_tol_m=0.3,
                    top_m=2.0, min_pts=40)

    def test_wall_ahead_triggers_ground_does_not(self) -> None:
        r = np.eye(3)
        self.assertFalse(near_obstacle(ground_and_wall(None), r, **self.kw())[0])
        hit, n = near_obstacle(ground_and_wall(0.5 / S), r, **self.kw())
        self.assertTrue(hit)
        self.assertGreaterEqual(n, 40)

    def test_detection_follows_head_heading(self) -> None:
        # 点在 base 系（跟头走），头左转 90° 后正前方仍是这些点；近的触发、远的不触发。
        r = hmd_to_base_rotation(hmd_yaw(90.0))
        self.assertTrue(near_obstacle(ground_and_wall(0.5 / S), r, **self.kw())[0])
        far = near_obstacle(ground_and_wall(1.5 / S), r, **self.kw())
        self.assertFalse(far[0])


class FakeSensors:
    fx, cx, cy, baseline_m = 202.5, 360.0, 202.5, 0.126

    def __init__(self) -> None:
        self.closed = False

    def open(self) -> dict:
        return {"size": [720, 405]}

    def hmd_rotation(self) -> np.ndarray:
        return np.eye(3)

    def read_stereo(self):
        z = np.zeros((4, 4), np.uint8)
        return z, z

    def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(self, armed: bool = True, vz: float = 0.0, record_root: Path | None = None) -> None:
        self.armed = armed
        self.vz = vz
        self.moves: list[tuple[float, int]] = []
        self.turns: list[float] = []
        self.stops = 0
        self.sensors = FakeSensors()
        cfg = OnlineNavConfig(map_min_interval_s=0.1, stereo_period_s=0.05, kf_max_age_s=0.3)
        self.nav = OnlineNavigator(
            motion_history=lambda: [osc(0.0, 0.0, self.vz)],
            send_move=lambda f, ms: self.moves.append((f, ms)),
            send_turn=lambda d: self.turns.append(d),
            stop_motion=self._stop,
            drive_block_reason=lambda: None if self.armed else "autonomy_not_armed",
            sensors_factory=lambda: self.sensors, cfg=cfg, record_root=record_root)

    def _stop(self) -> None:
        self.stops += 1


class ControlTickTest(unittest.TestCase):
    def prime(self, h: Harness, stereo_age: float = 0.0) -> None:
        now = time.monotonic()
        h.nav._pose, h.nav._pose_at = np.eye(4), now
        h.nav._stereo_at = now - stereo_age

    def test_idle_sends_nothing(self) -> None:
        h = Harness()
        self.prime(h)
        h.nav._control_tick(time.monotonic())
        self.assertEqual((h.moves, h.turns, h.stops), ([], [], 0))

    def test_not_armed_blocks_and_stops_once(self) -> None:
        h = Harness(armed=False)
        self.prime(h)
        h.nav.session.explore()
        h.nav._moving = True
        h.nav._control_tick(time.monotonic())
        h.nav._control_tick(time.monotonic())
        self.assertEqual(h.nav._drive_block, "autonomy_not_armed")
        self.assertEqual(h.stops, 1)
        self.assertEqual(h.moves, [])

    def test_stale_stereo_blocks(self) -> None:
        h = Harness()
        self.prime(h, stereo_age=5.0)
        h.nav.session.explore()
        h.nav._control_tick(time.monotonic())
        self.assertEqual(h.nav._drive_block, "stereo_stale")
        self.assertEqual(h.moves, [])

    def test_stale_pose_blocks(self) -> None:
        h = Harness()
        self.prime(h)
        h.nav._pose_at -= 5.0
        h.nav.session.explore()
        h.nav._control_tick(time.monotonic())
        self.assertEqual(h.nav._drive_block, "pose_stale")

    def test_follower_output_mapped_to_commands(self) -> None:
        h = Harness()
        self.prime(h)
        h.nav.session.explore()
        with mock.patch.object(h.nav.session, "step",
                               return_value={"state": "following", "forward": 0.6, "turn_rate": math.radians(30)}):
            h.nav._control_tick(time.monotonic())
        self.assertEqual(h.moves, [(0.6, 400)])
        self.assertAlmostEqual(h.turns[0], 3.0)          # 30°/s ÷ 10 Hz，正数 = 左转

    def test_goto_requires_running(self) -> None:
        self.assertEqual(Harness().nav.goto(1.0, 0.0)["reason"], "navmesh_not_running")
        self.assertFalse(Harness().nav.goto(float("nan"), 0.0)["ok"])


class ThreadsTest(unittest.TestCase):
    def test_slow_map_rebuild_does_not_block_status_or_control(self) -> None:
        # 实机：853 帧回环后一次重建 >1 s，旧实现握着 _nav_lock，status 请求 30 s 超时、控制线程停拍。
        h = Harness(armed=False, vz=0.5)
        real = nav_online.NavSession.compute
        gate = threading.Event()

        def slow(self, est, snap):
            gate.set()
            time.sleep(1.5)
            return real(self, est, snap)

        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)),                 mock.patch.object(nav_online.NavSession, "compute", slow):
            h.nav.start()
            try:
                self.assertTrue(gate.wait(5.0))
                t0 = time.perf_counter()
                h.nav.status()
                h.nav._control_tick(time.monotonic())
                self.assertTrue(h.nav.cancel()["ok"])
                self.assertLess(time.perf_counter() - t0, 0.3)
            finally:
                h.nav.stop()

    def test_end_to_end_keyframes_map_and_shutdown(self) -> None:
        h = Harness(armed=False, vz=0.5)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and h.nav.status()["map_updates"] < 2:
                    time.sleep(0.05)
                self.assertTrue(h.nav.explore()["ok"])
                time.sleep(0.3)
                st = h.nav.status()
            finally:
                h.nav.stop()
        self.assertGreaterEqual(st["keyframes"], 2)
        self.assertGreaterEqual(st["map_updates"], 2)
        self.assertEqual(st["pose_source"], "dead_reckoning+loop_closure")
        self.assertEqual(st["loop_closure"]["keyframes"], st["keyframes"])
        self.assertEqual(st["pose_state"], "localized")
        self.assertGreater(st["odometry_distance_m"], 0.0)
        self.assertEqual(st["drive_block"], "autonomy_not_armed")
        self.assertEqual(h.moves, [])
        self.assertEqual(st["errors"], [])
        self.assertIsNotNone(h.nav.grid_png())
        view, ng = h.nav.grid_view(), h.nav.session.ng
        self.assertEqual((view["rows"], view["cols"]), ng.grid.shape)
        # UI 用这套元数据把点击像素换成导航系世界米，必须与 NavGrid.to_world 一致。
        r, c = view["rows"] // 3, view["cols"] // 2
        o, res, s = view["origin_xy_track_m"], view["resolution_track_m"], view["world_scale"]
        ui_xy = ((o[0] + (c + 0.5) * res) * s, (o[1] + (view["rows"] - 1 - r + 0.5) * res) * s)
        self.assertEqual(tuple(round(v, 9) for v in ui_xy), tuple(round(v, 9) for v in ng.to_world((r, c))))
        self.assertEqual(ng.to_cell(ui_xy), (r, c))
        self.assertTrue(h.sensors.closed)
        self.assertFalse(h.nav.running)

    def test_external_hmd_yaw_snap_reported(self) -> None:
        h = Harness(armed=False)
        yaw = {"deg": 0.0}
        h.sensors.hmd_rotation = lambda: hmd_yaw(yaw["deg"])
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start()
            try:
                time.sleep(0.3)
                self.assertEqual(h.nav.status()["hmd_yaw_jumps"], [])
                yaw["deg"] = 90.0
                time.sleep(0.3)
                jumps = h.nav.status()["hmd_yaw_jumps"]
            finally:
                h.nav.stop()
        self.assertEqual(len(jumps), 1)
        self.assertAlmostEqual(jumps[0]["delta_deg"], 90.0, delta=0.5)

    def test_sensor_open_failure_reported(self) -> None:
        h = Harness()

        class Broken(FakeSensors):
            def open(self) -> dict:
                raise RuntimeError("SteamVR not running")

        h.nav._sensors_factory = Broken
        h.nav.start()
        time.sleep(0.2)
        st = h.nav.status()
        h.nav.stop()
        self.assertTrue(any("SteamVR not running" in e for e in st["errors"]))
        self.assertEqual(st["pose_state"], "unknown")


class RecorderTest(unittest.TestCase):
    def run_session(self, root: Path | None, record: bool) -> dict:
        h = Harness(armed=False, vz=0.5, record_root=root)
        with mock.patch.object(nav_online, "stereo_points", return_value=ground_and_wall(None)),                 mock.patch.object(nav_online, "stereo_disparity", return_value=np.zeros((4, 4), np.float32)):
            h.nav.start(record=record)
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and h.nav.status()["keyframes"] < 3:
                    time.sleep(0.05)
            finally:
                st = h.nav.stop()
        return st

    def test_default_off_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            st = self.run_session(Path(d), record=False)
            self.assertIsNone(st["recording"])
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_recording_has_everything_needed_to_replay(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            st = self.run_session(Path(d), record=True)
            rec = st["recording"]
            self.assertEqual(rec["stopped"], "closed")
            out = Path(rec["dir"])
            self.assertEqual(out.parent, Path(d))
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("mapper", meta["config"])
            ev = [json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            kfs = [e for e in ev if e["kind"] == "kf"]
            self.assertEqual(ev[0]["kind"], "sensors")
            self.assertAlmostEqual(ev[0]["baseline_m"], FakeSensors.baseline_m)
            # 录到的关键帧数与状态一致、id 连续，每个都有点云文件。
            self.assertEqual([e["k"] for e in kfs], list(range(st["keyframes"])))
            self.assertGreaterEqual(len(kfs), 3)
            for e in kfs:
                with np.load(out / "kf" / f"{e['k']:06d}.npz") as z:
                    self.assertEqual(z["pts"].shape, ground_and_wall(None).shape)
                    self.assertEqual(z["T_dr"].shape, (4, 4))
            hmd = (out / "hmd.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(hmd), len(kfs))
            self.assertEqual(np.array(json.loads(hmd[0])["R"]).shape, (3, 3))
            # 假 OSC 历史每拍都一样：同一时间戳只记一次。
            osc_lines = (out / "osc.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(osc_lines), 1)
            with np.load(out / "final_poses.npz") as final:
                self.assertEqual(list(final["ids"]), [e["k"] for e in kfs])

    def test_size_limit_stops_recording_not_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r = SessionRecorder(Path(d) / "s", max_bytes=2000, meta={"x": 1})
            for i in range(3):
                r.keyframe(i, np.zeros((500, 3), np.float32), np.eye(4), np.eye(4), 0.0, None)
            self.assertEqual(r.status()["stopped"], "size_limit")
            self.assertEqual(r.status()["keyframes"], 1)
            r.line("hmd", {"t": 0.0})
            r.close(None)
            self.assertFalse((Path(d) / "s" / "hmd.jsonl").exists())
            self.assertEqual(r.status()["stopped"], "size_limit")

    def test_record_without_root_reports_error(self) -> None:
        st = self.run_session(None, record=True)
        self.assertIsNone(st["recording"])
        self.assertTrue(any("record_root_not_configured" in e for e in st["errors"]))


if __name__ == "__main__":
    unittest.main()
