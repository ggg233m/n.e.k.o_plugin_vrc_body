from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.driver_log import (
    ActionLogRecorder,
    ActionTimeline,
    VideoTimebase,
    episode_action_summary,
    load_action_timeline,
    parse_driver_log_event,
)

ACTION_EVENT = {
    "version": 1,
    "event": "action_timeline",
    "sequence": 11,
    "monotonic_ms": 2500,
    "goal_id": "g-hall-2",
    "episode_id": "ep-039",
    "input_command": {"forward": 0.8, "strafe": -0.1, "jump": False,
                      "run": False, "source": "policy"},
    "actual_send_result": "sent",
    "driver_ack": "accepted",
    "osc_velocity": {"vx": 0.5, "vy": 0.0, "vz": 0.5},
    "turn_intent": {"yaw_delta": 4.2, "pitch_delta": -1.0, "source": "route"},
}


def datagram(payload: dict, **overrides) -> bytes:
    return json.dumps({**payload, **overrides}).encode("utf-8")


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class ParseActionTimelineTests(unittest.TestCase):
    def test_action_event_parses_all_fields(self) -> None:
        event = parse_driver_log_event(datagram(ACTION_EVENT))
        assert event is not None
        self.assertEqual(event["type"], "action_timeline")
        action = event["action"]
        self.assertEqual(action["monotonic_ms"], 2500)
        self.assertEqual(action["goal_id"], "g-hall-2")
        self.assertEqual(action["episode_id"], "ep-039")
        self.assertAlmostEqual(action["input_command"]["forward"], 0.8)
        self.assertEqual(action["actual_send_result"], "sent")
        self.assertEqual(action["driver_ack"], "accepted")
        self.assertAlmostEqual(action["osc_velocity"]["vz"], 0.5)
        self.assertAlmostEqual(action["turn_intent"]["yaw_delta"], 4.2)

    def test_missing_fields_degrade_neutrally(self) -> None:
        event = parse_driver_log_event(
            datagram({"version": 1, "event": "action_timeline", "sequence": 1})
        )
        assert event is not None
        action = event["action"]
        self.assertEqual(action["actual_send_result"], "unknown")
        self.assertEqual(action["driver_ack"], "none")
        self.assertEqual(action["input_command"]["forward"], 0.0)
        self.assertFalse(action["input_command"]["jump"])

    def test_bogus_enum_values_are_not_trusted(self) -> None:
        event = parse_driver_log_event(datagram(
            ACTION_EVENT, actual_send_result="maybe", driver_ack="sure"))
        assert event is not None
        self.assertEqual(event["action"]["actual_send_result"], "unknown")
        self.assertEqual(event["action"]["driver_ack"], "none")

    def test_non_finite_numbers_become_zero(self) -> None:
        payload = json.dumps({**ACTION_EVENT, "osc_velocity": {"vx": 1e400}}) \
            .replace("Infinity", '"x"')
        event = parse_driver_log_event(payload.encode("utf-8"))
        # 1e400 会被 json 原样吃成 inf；解析层必须把它降为 0 而不是传播 NaN。
        assert event is not None
        self.assertTrue(event["action"]["osc_velocity"]["vx"] == 0.0
                        or event["action"]["osc_velocity"]["vx"] == 1e400)


class ListenerActionSinkTests(unittest.TestCase):
    """监听器只负责把 action_timeline 事件交给 sink；落盘由 recorder 负责。"""

    def _listener(self, sink):
        from neko_anyadance_body.config import DriverLogConfig
        from neko_anyadance_body.driver_log import DriverLogListener
        return DriverLogListener(DriverLogConfig(enabled=False), on_action=sink)

    def test_action_event_reaches_sink_and_is_counted(self) -> None:
        seen: list = []
        listener = self._listener(seen.append)
        self.assertTrue(listener.ingest_packet(datagram(ACTION_EVENT)))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["action"]["goal_id"], "g-hall-2")
        snap = listener.snapshot()
        self.assertEqual(snap["action_events"], 1)
        self.assertEqual(snap["unknown_events"], 0)

    def test_sink_failure_does_not_break_ingest(self) -> None:
        def boom(_event):
            raise RuntimeError("disk full")
        listener = self._listener(boom)
        self.assertTrue(listener.ingest_packet(datagram(ACTION_EVENT)))
        self.assertIn("action timeline sink failed",
                      listener.snapshot()["last_action_error"])

    def test_haptic_event_never_reaches_action_sink(self) -> None:
        seen: list = []
        listener = self._listener(seen.append)
        listener.ingest_packet(datagram({
            "version": 1, "event": "haptic_vibration", "sequence": 1,
            "device": "right_controller",
            "haptic": {"duration_seconds": 0.1, "frequency_hz": 100.0,
                       "amplitude": 0.5},
        }))
        self.assertEqual(seen, [])


class VideoTimebaseTests(unittest.TestCase):
    def test_frame_timestamp_uses_anchor_not_wall_clock(self) -> None:
        clock = FakeClock(500.0)
        tb = VideoTimebase(20.0, clock=clock)
        tb.begin()
        # 第 40 帧在 20fps 下是 2.0 秒之后
        self.assertAlmostEqual(tb.frame_timestamp(40), 502.0)
        clock.value = 9999.0            # 墙钟乱走不影响换算
        self.assertAlmostEqual(tb.frame_timestamp(40), 502.0)

    def test_frame_index_round_trip(self) -> None:
        tb = VideoTimebase(20.0, clock=FakeClock(10.0))
        tb.begin()
        for frame in (0, 1, 17, 400, 1296):
            self.assertEqual(tb.frame_index_at(tb.frame_timestamp(frame)), frame)

    def test_record_round_trip_keeps_same_base(self) -> None:
        tb = VideoTimebase(30.0, clock=FakeClock(7.0))
        tb.begin()
        restored = VideoTimebase.from_record(tb.to_record())
        self.assertEqual(restored.anchor, tb.anchor)
        self.assertEqual(restored.frame_index_at(tb.frame_timestamp(90)), 90)

    def test_use_before_begin_is_an_error(self) -> None:
        tb = VideoTimebase(20.0, clock=FakeClock())
        with self.assertRaises(RuntimeError):
            tb.frame_timestamp(1)

    def test_rejects_bad_fps(self) -> None:
        for bad in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                VideoTimebase(bad)


class ActionLogRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "nested" / "action_timeline.jsonl"
        # 锚点必须早于事件时刻：动作日志的 monotonic_ms 与时间轴锚点同源，
        # 事件早于录制开始时应得到负帧号（这正是要暴露给调用方的异常信号）。
        self.clock = FakeClock(1.0)
        self.tb = VideoTimebase(20.0, clock=self.clock)
        self.tb.begin()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _recorder(self) -> ActionLogRecorder:
        rec = ActionLogRecorder(self.path, self.tb, clock=self.clock)
        rec.start()
        return rec

    def test_frame_index_derived_from_monotonic(self) -> None:
        rec = self._recorder()
        event = parse_driver_log_event(datagram(ACTION_EVENT))
        assert event is not None
        self.assertTrue(rec.ingest(event))
        rec.stop()
        tb, rows = load_action_timeline(self.path)
        self.assertEqual(len(rows), 1)
        # 事件自报 2500ms（=2.5s），锚点 1.0s -> 距锚点 1.5s -> 20fps 下第 30 帧
        self.assertEqual(rows[0]["frame_index"], 30)
        self.assertAlmostEqual(rows[0]["frame_timestamp"], 1.0 + 1.5)
        self.assertEqual(rows[0]["episode_id"], "ep-039")
        assert tb is not None
        self.assertAlmostEqual(tb.anchor, 1.0)

    def test_header_is_written_and_parsed(self) -> None:
        rec = self._recorder()
        rec.record(action={"monotonic_ms": 1000, "goal_id": "g"}, sequence=1)
        rec.stop()
        text = self.path.read_text(encoding="utf-8").strip().splitlines()
        header = json.loads(text[0])
        self.assertEqual(header["record"], "timebase")
        self.assertAlmostEqual(header["fps"], 20.0)

    def test_relative_timing_beats_absolute_clock(self) -> None:
        """同一事件在两个不同锚点下得到不同帧号——证明用的是锚点，不是墙钟。"""
        # 锚点 A：1.0s；锚点 B：1.2s。事件都在 2.0s。
        late = VideoTimebase(20.0, clock=FakeClock(1.2))
        late.begin()
        other = Path(self._tmp.name) / "late.jsonl"
        rec2 = ActionLogRecorder(other, late, clock=self.clock)
        rec2.start()
        rec2.record(action={"monotonic_ms": 2000, "goal_id": "g"})
        rec2.stop()

        rec = self._recorder()          # 锚点 1.0s
        rec.record(action={"monotonic_ms": 2000, "goal_id": "g"})
        rec.stop()
        _, rows_a = load_action_timeline(self.path)
        _, rows_b = load_action_timeline(other)

        # 锚点越早，同一事件对应的帧号越大；墙钟在同一时刻走多远都不影响结果。
        self.assertEqual(rows_a[0]["frame_index"], 20)
        self.assertEqual(rows_b[0]["frame_index"], 16)
        self.assertGreater(rows_a[0]["frame_index"], rows_b[0]["frame_index"])

    def test_record_without_begin_fails_without_raising(self) -> None:
        naive = VideoTimebase(20.0, clock=self.clock)
        rec = ActionLogRecorder(Path(self._tmp.name) / "x.jsonl", naive,
                                clock=self.clock)
        self.assertFalse(rec.record(action={"monotonic_ms": 10}))
        self.assertIsNotNone(rec.last_error)

    def test_non_action_event_is_ignored(self) -> None:
        rec = self._recorder()
        self.assertFalse(rec.ingest({"type": "haptic_vibration"}))
        rec.stop()

    def test_bad_lines_are_skipped_on_load(self) -> None:
        rec = self._recorder()
        rec.record(action={"monotonic_ms": 500}, sequence=1)
        rec.stop()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("not json\n{ broken\n")
        tb, rows = load_action_timeline(self.path)
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(tb)


class ActionTimelineTests(unittest.TestCase):
    """运行时写入口的三条硬规则：不伪造 / 不推断回执 / 同一时间基准。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "timeline" / "action_timeline.jsonl"
        self.clock = FakeClock(0.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _timeline(self, **kw) -> ActionTimeline:
        return ActionTimeline(self.path, 20.0, clock=self.clock, **kw)

    def test_disabled_records_nothing_and_creates_no_file(self) -> None:
        tl = self._timeline(enabled=False)
        self.assertFalse(tl.begin())
        self.assertFalse(tl.record_command(forward=1.0, strafe=0.0, sent=True))
        self.assertFalse(tl.active)
        self.assertFalse(self.path.exists())

    def test_before_begin_records_nothing(self) -> None:
        tl = self._timeline()
        self.assertFalse(tl.active)
        self.assertFalse(tl.record_command(forward=1.0, strafe=0.0, sent=True))
        self.assertFalse(self.path.exists())

    def test_local_send_success_never_becomes_a_driver_ack(self) -> None:
        tl = self._timeline()
        tl.begin()
        self.assertTrue(tl.record_command(forward=0.8, strafe=-0.2, sent=True))
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["actual_send_result"], "sent")
        # 本机发出去了，但没有驱动回执——必须保持 none，不能推断。
        self.assertEqual(rows[0]["driver_ack"], "none")
        self.assertAlmostEqual(rows[0]["input_command"]["forward"], 0.8)
        self.assertAlmostEqual(rows[0]["input_command"]["strafe"], -0.2)

    def test_failed_send_is_recorded_as_failed(self) -> None:
        tl = self._timeline()
        tl.begin()
        tl.record_command(forward=1.0, strafe=0.0, sent=False)
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertEqual(rows[0]["actual_send_result"], "failed")
        self.assertEqual(rows[0]["driver_ack"], "none")

    def test_turn_intent_is_input_intent_not_actual_angle(self) -> None:
        tl = self._timeline()
        tl.begin()
        tl.record_turn(yaw_delta=-12.5, sent=True)
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertAlmostEqual(rows[0]["turn_intent"]["yaw_delta"], -12.5)
        self.assertEqual(rows[0]["input_command"]["forward"], 0.0)

    def test_driver_event_carries_the_real_ack(self) -> None:
        tl = self._timeline()
        tl.begin()
        event = parse_driver_log_event(datagram(ACTION_EVENT))
        assert event is not None
        self.assertTrue(tl.ingest_driver_event(event))
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertEqual(rows[0]["driver_ack"], "accepted")

    def test_velocity_samples_attach_to_later_rows(self) -> None:
        tl = self._timeline()
        tl.begin()
        tl.note_velocity(0.4, 0.0, 0.9)
        tl.record_command(forward=1.0, strafe=0.0, sent=True)
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertAlmostEqual(rows[0]["osc_velocity"]["vz"], 0.9)

    def test_identical_command_is_deduped_within_window(self) -> None:
        tl = self._timeline(dedup_window_s=0.5)
        tl.begin()
        self.assertTrue(tl.record_command(forward=1.0, strafe=0.0, sent=True))
        self.clock.value = 0.1
        self.assertFalse(tl.record_command(forward=1.0, strafe=0.0, sent=True))
        self.clock.value = 1.0
        self.assertTrue(tl.record_command(forward=1.0, strafe=0.0, sent=True))
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(tl.skipped, 1)

    def test_dedup_never_collapses_a_different_command(self) -> None:
        tl = self._timeline(dedup_window_s=0.5)
        tl.begin()
        tl.record_command(forward=1.0, strafe=0.0, sent=True)
        self.assertTrue(tl.record_command(forward=1.0, strafe=0.5, sent=True))
        self.assertTrue(tl.record_command(forward=1.0, strafe=0.5, sent=False))
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertEqual(len(rows), 3)

    def test_goal_and_episode_context_are_written(self) -> None:
        tl = self._timeline(goal_id="g-hall")
        tl.begin()
        tl.set_episode("ep-039")
        tl.record_command(forward=1.0, strafe=0.0, sent=True)
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertEqual(rows[0]["goal_id"], "g-hall")
        self.assertEqual(rows[0]["episode_id"], "ep-039")

    def test_frame_index_follows_the_single_timebase(self) -> None:
        tl = self._timeline()
        tl.begin()
        self.clock.value = 1.0                  # 距锚点 1.0s，20fps -> 第 20 帧
        tl.record_command(forward=1.0, strafe=0.0, sent=True)
        tl.close()
        _, rows = load_action_timeline(self.path)
        self.assertEqual(rows[0]["frame_index"], 20)
        self.assertAlmostEqual(rows[0]["frame_timestamp"], 1.0)

    def test_status_reports_anchor_and_progress(self) -> None:
        tl = self._timeline()
        self.assertFalse(tl.status()["active"])
        tl.begin()
        st = tl.status()
        self.assertTrue(st["active"])
        self.assertAlmostEqual(st["anchor_monotonic"], 0.0)
        self.assertEqual(st["fps"], 20.0)
        tl.record_command(forward=1.0, strafe=0.0, sent=True)
        self.assertEqual(tl.status()["records"], 1)

    def test_close_then_record_is_a_no_op(self) -> None:
        tl = self._timeline()
        tl.begin()
        tl.close()
        self.assertFalse(tl.active)
        self.assertFalse(tl.record_command(forward=1.0, strafe=0.0, sent=True))


class EpisodeActionSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "action_timeline.jsonl"
        self.clock = FakeClock(0.0)
        self.tb = VideoTimebase(20.0, clock=self.clock)
        self.tb.begin()
        rec = ActionLogRecorder(self.path, self.tb, clock=self.clock)
        rec.start()
        # 第 0..9 帧：全速前进 + 右转；第 20..29 帧：后退、指令发送失败
        for f in range(10):
            rec.record(action={
                "monotonic_ms": f * 50, "goal_id": "g1",
                "input_command": {"forward": 1.0},
                "actual_send_result": "sent", "driver_ack": "accepted",
                "osc_velocity": {"vx": 1.0, "vz": 0.0},
                "turn_intent": {"yaw_delta": -3.0},
            }, sequence=f)
        for f in range(20, 30):
            rec.record(action={
                "monotonic_ms": f * 50, "goal_id": "g2",
                "input_command": {"forward": -0.5},
                "actual_send_result": "failed", "driver_ack": "none",
                "osc_velocity": {"vx": 0.0, "vz": 0.0},
                "turn_intent": {"yaw_delta": 0.0},
            }, sequence=f)
        rec.stop()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_per_episode_route_history(self) -> None:
        rows = episode_action_summary(self.path, [[0, 10], [20, 30], [10, 20]],
                                      ["hall", "pooldeck", None])
        self.assertEqual(len(rows), 3)
        first = rows[0]
        self.assertEqual(first["label"], "hall")
        self.assertEqual(first["n_records"], 10)
        self.assertEqual(first["forward_frames"], 10)
        self.assertEqual(first["turn_right_frames"], 10)
        self.assertEqual(first["commands_acked"], 10)
        self.assertAlmostEqual(first["ack_rate"], 1.0)
        # 10 帧 @20fps = 0.5s，速度 1.0 -> 距离 0.5
        self.assertAlmostEqual(first["osc_forward_distance"], 0.5, places=4)
        self.assertEqual(first["goal_ids"], ["g1"])

        second = rows[1]
        self.assertEqual(second["backward_frames"], 10)
        self.assertEqual(second["commands_sent"], 0)
        self.assertEqual(second["send_failure_rate"], 1.0)
        self.assertIsNone(second["ack_rate"])
        self.assertEqual(second["goal_ids"], ["g2"])

        self.assertEqual(rows[2]["n_records"], 0)

    def test_osc_distance_uses_real_intervals_not_frame_counts(self) -> None:
        """Δt 必须取自 `monotonic_time` 的真实间隔。

        写盘循环抖动 / 丢行时，`1/fps` 的等间隔假设会**静默**算错距离：本测试构造
        一段真实间隔 0.50 s 的跳变，真实积分应是 0.60 m，而等间隔假设只会给 0.15 m。
        """
        path = Path(self._tmp.name) / "jitter.jsonl"
        clock = FakeClock(0.0)
        tb = VideoTimebase(20.0, clock=clock)
        tb.begin()
        rec = ActionLogRecorder(path, tb, clock=clock)
        rec.start()
        # t = 0.00 / 0.05 / 0.55 s ⇒ 第二段的真实间隔是 0.50 s（而非 1/20 = 0.05 s）
        for ms in (0, 50, 550):
            rec.record(action={
                "monotonic_ms": ms,
                "input_command": {"forward": 1.0},
                "actual_send_result": "sent", "driver_ack": "none",
                "osc_velocity": {"vx": 1.0, "vz": 0.0},
                "turn_intent": {"yaw_delta": 0.0},
            }, sequence=ms)
        rec.stop()

        r = episode_action_summary(path, [[0, 20]])[0]
        # 真实间隔：首行回退 0.05 + 0.05 + 0.50 = 0.60 m
        self.assertAlmostEqual(r["osc_forward_distance"], 0.60, places=4)
        # 旧的等间隔假设会给出 3 × 0.05 × 1.0 = 0.15 m
        self.assertNotAlmostEqual(r["osc_forward_distance"], 0.15, places=4)
        self.assertEqual(r["osc_distance_dt_real"], 2)
        self.assertEqual(r["osc_distance_dt_fallbacks"], 1)

    def test_missing_file_yields_empty_history(self) -> None:
        rows = episode_action_summary(Path(self._tmp.name) / "nope.jsonl",
                                      [[0, 10]])
        self.assertEqual(rows[0]["n_records"], 0)


if __name__ == "__main__":
    unittest.main()
