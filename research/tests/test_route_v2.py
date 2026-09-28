"""route_v2 集成测试：路线解析 + 动作日志落盘读回 + 绕行逻辑。

防"静默无效"三层断言的第二层（入口接线）+ 第三层（运行时写行）：
用假轴/假缓存驱动 RouteDriver 真跑一段路线，落盘后必须能被
driver_log.load_action_timeline 原样读回——格式漂移在这里红灯。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401  # 包别名：neko_anyadance_body

_RECORDER_DIR = (Path(__file__).resolve().parents[2]
                 / "research" / "recorder")
sys.path.insert(0, str(_RECORDER_DIR))

import route_v2  # noqa: E402
from neko_anyadance_body.driver_log import (  # noqa: E402
    ActionLogRecorder,
    VideoTimebase,
    load_action_timeline,
)

# 临时文件一律落仓库 .tmp（C: 紧张 + 杀软扫描 C:\Temp 新落 .jsonl 会跟
# rmtree 竞态，WinError 32）。清理用 ignore_errors——残留可弃。
_TMP_ROOT = Path(__file__).resolve().parents[2] / ".tmp" / "_rv2_tests"


def _fresh_dir() -> Path:
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=str(_TMP_ROOT)))


class FakeAxes:
    """记录每次 set 调用；tick 不发网络包。"""

    def __init__(self) -> None:
        self.send_errors = 0
        self.calls: list[tuple[float, float, float]] = []

    def set(self, vertical: float, look: float, horizontal: float = 0.0) -> None:
        self.calls.append((vertical, look, horizontal))

    def tick(self, force: bool = False) -> None:
        _ = force


class FakeCache:
    def __init__(self, speed=None) -> None:
        self._speed = speed

    def horizontal_speed(self):
        return self._speed

    def velocity_last_known(self):
        return (0.0, 0.0 if self._speed is None else self._speed)


def _make_driver(recorder, cache, events, **kw):
    defaults = dict(axes=FakeAxes(), cache=cache, events=events,
                    recorder=recorder, hmd=None, vertical=0.3, hz=50.0,
                    stall_speed=0.15, stall_ticks=2, strafe_value=0.5,
                    strafe_s=0.05, max_detours=1, hmd_turn_rate=180.0)
    defaults.update(kw)
    return route_v2.RouteDriver(**defaults)


class LoadRouteTest(unittest.TestCase):
    def _write(self, doc):
        d = _fresh_dir()
        self.addCleanup(shutil.rmtree, d, True)
        path = d / "route.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        return str(path)

    def test_valid_route_loads(self):
        path = self._write({"version": 1, "legs": [
            {"op": "beacon"},
            {"op": "compass_check", "deg": 90, "fwd_sec": 2.0},
            {"op": "forward", "sec": 4.0},
            {"op": "turn", "deg": 90},
        ]})
        doc = route_v2.load_route(path)
        self.assertEqual(len(doc["legs"]), 4)

    def test_unknown_op_rejected(self):
        path = self._write({"version": 1, "legs": [{"op": "fly"}]})
        with self.assertRaises(ValueError):
            route_v2.load_route(path)

    def test_missing_sec_rejected(self):
        path = self._write({"version": 1, "legs": [{"op": "forward"}]})
        with self.assertRaises(ValueError):
            route_v2.load_route(path)

    def test_bad_version_rejected(self):
        path = self._write({"version": 2, "legs": [{"op": "beacon"}]})
        with self.assertRaises(ValueError):
            route_v2.load_route(path)

    def test_empty_legs_rejected(self):
        path = self._write({"version": 1, "legs": []})
        with self.assertRaises(ValueError):
            route_v2.load_route(path)


class RouteRunIntegrationTest(unittest.TestCase):
    """真跑路线 → 真落盘 → 真加载器读回。"""

    def _run(self, legs, speed=None):
        events: list[dict] = []
        cache = FakeCache(speed)
        d = _fresh_dir()
        self.addCleanup(shutil.rmtree, d, True)
        path = str(d / "action_timeline.jsonl")
        tb = VideoTimebase(30.0)
        tb.begin()
        rec = ActionLogRecorder(path, tb)
        self.assertTrue(rec.start())
        driver = _make_driver(rec, cache, events)
        driver.run({"version": 1, "legs": legs})
        rec.stop()
        written = rec.written
        timebase, rows = load_action_timeline(path)
        return written, timebase, rows, events

    def test_rows_written_and_readable(self):
        written, timebase, rows, events = self._run([
            {"op": "beacon"},
            {"op": "compass_check", "deg": 90, "fwd_sec": 0.1},
            {"op": "forward", "sec": 0.1},
            {"op": "turn", "deg": 90},
        ])
        self.assertGreater(written, 0)
        self.assertEqual(written, len(rows))
        self.assertIsNotNone(timebase)
        self.assertIsNotNone(timebase.anchor)
        first = rows[0]
        self.assertEqual(first["record"], "action")
        self.assertEqual(first["goal_id"], "route_v2")
        self.assertEqual(first["turn_intent"]["source"], "hmd_pose")
        self.assertIn("input_command", first)
        self.assertIn("osc_velocity", first)
        self.assertEqual(first["actual_send_result"], "sent")
        # frame_index 单调不减、非负（时间基准没漂移）
        idx = [r["frame_index"] for r in rows]
        self.assertTrue(all(i >= 0 for i in idx))
        self.assertEqual(idx, sorted(idx))

    def test_turn_records_yaw_accumulation(self):
        _, _, rows, events = self._run([{"op": "turn", "deg": 90}])
        yaws = [r["turn_intent"]["hmd_yaw_deg"] for r in rows
                if r["turn_intent"]["source"] == "hmd_pose"]
        self.assertTrue(any(y > 89.0 for y in yaws),
                        "终点 yaw 应到达 +90°，实际序列尾=%s" % yaws[-3:])
        turn_ev = [e for e in events if e["event"] == "turn_hmd"]
        self.assertEqual(len(turn_ev), 1)
        self.assertEqual(turn_ev[0]["yaw_cmd_deg"], 90)

    def test_stall_triggers_detour_then_abandon(self):
        # speed=0.05 < stall_speed=0.15：每个 tick 都算低速。
        # stall_ticks=2 → 很快判撞；max_detours=1 → 第 2 次撞后弃腿。
        written, _, rows, events = self._run(
            [{"op": "forward", "sec": 0.2}], speed=0.05)
        stalls = [e for e in events if e["event"] == "stall"]
        detours = [e for e in events if e["event"] == "detour"]
        abandoned = [e for e in events if e["event"] == "leg_abandoned"]
        self.assertGreaterEqual(len(stalls), 2)
        self.assertEqual(len(detours), 1)
        self.assertEqual(len(abandoned), 1)
        # 绕行拍里 horizontal 必须非零
        strafe_rows = [r for r in rows
                       if abs(r["input_command"].get("horizontal", 0.0)) > 0.01]
        self.assertGreater(len(strafe_rows), 0)
        fwd_ev = [e for e in events if e["event"] == "forward"][0]
        self.assertTrue(fwd_ev["abandoned"])

    def test_no_stall_clean_forward(self):
        _, _, rows, events = self._run([{"op": "forward", "sec": 0.1}],
                                       speed=1.0)
        self.assertEqual([e for e in events if e["event"] == "stall"], [])
        fwd = [r for r in rows
               if r["input_command"].get("forward", 0.0) > 0.01]
        self.assertGreater(len(fwd), 0)

    def test_beacon_and_compass_events(self):
        _, _, _, events = self._run([
            {"op": "beacon"},
            {"op": "compass_check", "deg": 90, "fwd_sec": 0.05},
        ])
        kinds = {e["event"] for e in events}
        self.assertIn("beacon", kinds)
        self.assertIn("compass_check", kinds)
        compass = [e for e in events if e["event"] == "compass_check"][0]
        self.assertEqual(compass["yaw_cmd_deg"], 90)
        self.assertEqual(len(compass["turn_t"]), 2)
        self.assertEqual(len(compass["fwd_t"]), 2)

    def test_rows_skipped_when_recorder_none(self):
        """未锚定（recorder=None）时必须不写、不炸——降级纪律。"""
        events: list[dict] = []
        driver = _make_driver(None, FakeCache(1.0), events)
        driver.run({"version": 1, "legs": [{"op": "turn", "deg": 90}]})
        turn_ev = [e for e in events if e["event"] == "turn_hmd"]
        self.assertEqual(len(turn_ev), 1)


class YawQuatTest(unittest.TestCase):
    def test_zero_yaw_identity(self):
        self.assertEqual(route_v2.yaw_quat_xyzw(0.0), [0.0, 0.0, 0.0, 1.0])

    def test_quarter_turn(self):
        x, y, z, w = route_v2.yaw_quat_xyzw(3.141592653589793 / 2.0)
        self.assertAlmostEqual(abs(y), 0.7071067811865476, places=9)
        self.assertAlmostEqual(w, 0.7071067811865476, places=9)
        self.assertEqual(x, 0.0)
        self.assertEqual(z, 0.0)


def _command_event(seq=42, accepted=True, with_hmd=True):
    """按驱动契约构造一条 command_processed 数据报（JSON 字节）。"""
    devices = {}
    if with_hmd:
        devices["hmd"] = {"valid": True, "connected": True,
                          "pose": {"position": [0.1, 1.5, 0.2],
                                   "rotation_xyzw": [0.0, 0.3827, 0.0, 0.9239]}}
    payload = json.dumps({"version": 1, "devices": devices})
    return json.dumps({
        "version": 1, "event": "command_processed", "sequence": seq,
        "timestamp_ms": 1700000000123, "suppressed": 0, "detail": "ok",
        "source": {"host": "127.0.0.1", "port": 50000},
        "command": {"protocol": "pose_frame", "bytes": 200,
                    "accepted": accepted, "devices": list(devices),
                    "y_clamped": [], "payload": payload},
    }).encode("utf-8")


class HmdCaptureTest(unittest.TestCase):
    def test_parse_accepted_event(self):
        parsed = route_v2.parse_command_event(_command_event(seq=7))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["seq"], 7)
        self.assertEqual(parsed["driver_ts_ms"], 1700000000123)
        self.assertAlmostEqual(parsed["hmd"]["position"][1], 1.5)
        self.assertAlmostEqual(parsed["hmd"]["rotation_xyzw"][3], 0.9239)
        self.assertIn("hmd", parsed["all_device_poses"])

    def test_parse_rejected_event_returns_none(self):
        self.assertIsNone(route_v2.parse_command_event(
            _command_event(accepted=False)))

    def test_parse_wrong_event_returns_none(self):
        datagram = json.dumps({"version": 1, "event": "haptic_vibration",
                               "sequence": 1, "detail": "x",
                               "device": "left_controller",
                               "haptic": {"duration_seconds": 0.1,
                                          "frequency_hz": 160.0,
                                          "amplitude": 0.5}}).encode()
        self.assertIsNone(route_v2.parse_command_event(datagram))

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(route_v2.parse_command_event(b"\xff\xfe not json"))

    def test_parse_no_hmd_returns_none(self):
        self.assertIsNone(route_v2.parse_command_event(
            _command_event(with_hmd=False)))

    def test_ingest_writes_and_dedups(self):
        d = _fresh_dir()
        self.addCleanup(shutil.rmtree, d, True)
        path = d / "hmd_frames.jsonl"
        listener = route_v2.HmdCaptureListener(str(path))
        self.assertTrue(listener.start())   # 先 start：写盘依赖文件句柄
        try:
            self.assertTrue(listener._ingest(route_v2.parse_command_event(
                _command_event(seq=1))))
            self.assertFalse(listener._ingest(route_v2.parse_command_event(
                _command_event(seq=1))))    # 重复 seq：丢弃
            self.assertTrue(listener._ingest(route_v2.parse_command_event(
                _command_event(seq=2))))
        finally:
            listener.stop()
        self.assertEqual(listener.accepted, 3)
        self.assertEqual(listener.duplicates, 1)
        self.assertEqual(listener.last_seq, 2)
        self.assertEqual(listener.hmd_frames, 2)
        self.assertIsNone(listener._handle)
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["seq"] for r in rows], [1, 2])
        self.assertIn("t", rows[0])

    def test_ingest_before_start_drops(self):
        d = _fresh_dir()
        self.addCleanup(shutil.rmtree, d, True)
        listener = route_v2.HmdCaptureListener(str(d / "hmd_frames.jsonl"))
        self.assertFalse(listener._ingest(route_v2.parse_command_event(
            _command_event(seq=1))))
        self.assertEqual(listener.hmd_frames, 0)


class RecorderOutdirTest(unittest.TestCase):
    """run_id 按启动时间生成；同秒重跑绝不静默合并（追加 -2/-3）。"""

    def test_default_uses_startup_stamp(self):
        import record_stage1
        d = _fresh_dir()
        self.addCleanup(shutil.rmtree, d, True)
        outdir = record_stage1.resolve_outdir(str(d), "20260920-225555", None)
        self.assertEqual(Path(outdir).parent, d)
        self.assertEqual(Path(outdir).name, "20260920-225555")

    def test_collision_gets_suffix_never_merges(self):
        import record_stage1
        d = _fresh_dir()
        self.addCleanup(shutil.rmtree, d, True)
        first = record_stage1.resolve_outdir(str(d), "20260920-225555", None)
        Path(first).mkdir()
        second = record_stage1.resolve_outdir(str(d), "20260920-225555", None)
        self.assertNotEqual(first, second)
        self.assertEqual(Path(second).name, "20260920-225555-2")
        Path(second).mkdir()
        third = record_stage1.resolve_outdir(str(d), "20260920-225555", None)
        self.assertEqual(Path(third).name, "20260920-225555-3")

    def test_explicit_outdir_respected_as_is(self):
        import record_stage1
        d = _fresh_dir()
        self.addCleanup(shutil.rmtree, d, True)
        explicit = str(d / "myrun")
        self.assertEqual(record_stage1.resolve_outdir(str(d), "20260920-225555",
                                                      explicit), explicit)


if __name__ == "__main__":
    unittest.main()
