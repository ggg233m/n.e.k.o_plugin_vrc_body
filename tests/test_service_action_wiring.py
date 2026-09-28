"""证明 `service.py` 的动作时间轴接线**真的会写行**（而不是文档写了、代码没接）。

为什么要单独测：`ActionTimeline` 自身的单元测试只能证明"数据结构对"，
不能证明"运行链路把它接上了"。上一轮就吃过这个亏——`route_history_seed_edges()`
有定义、有单测、有文档引用，却**没有任何调用点**，静默无效。

这里不去构造完整的 :class:`BackendService`（构造函数会预热 ONNX Runtime、建采集源，
在无 GPU/无屏环境下不可靠），而是把三个钩子方法**绑定到一个只带它们真正用到的
属性的轻量桩上**：`scheduler` / `config.input.primary` / `action_timeline` / `osc`。
这样测的就是**方法体本身**——也就是生产路径上真正执行的那段代码。
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.service import BackendService
from neko_anyadance_body.config import PluginConfig
from neko_anyadance_body.driver_log import ActionTimeline, load_action_timeline

REPO = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, value: float = 1000.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def tick(self, seconds: float) -> None:
        self.value += float(seconds)


class FakeScheduler:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls: list[tuple[str, dict]] = []

    def submit(self, kind: str, params=None):
        self.calls.append((kind, dict(params or {})))
        return {"accepted": self.accepted, "kind": kind}


class FakeOsc:
    def __init__(self, feedback) -> None:
        self._feedback = feedback
        self.calls = 0

    def motion_feedback(self):
        self.calls += 1
        return dict(self._feedback)


class ActionWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "nested" / "action_timeline.jsonl"
        self.clock = FakeClock()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ---- helpers -------------------------------------------------------
    def _timeline(self) -> ActionTimeline:
        tl = ActionTimeline(self.path, 20.0, clock=self.clock)
        self.assertTrue(tl.begin(), "ActionTimeline.begin() failed")
        return tl

    def _stub(self, *, timeline=None, scheduler=None, osc=None,
              primary: str = "anyadance"):
        return SimpleNamespace(
            config=SimpleNamespace(input=SimpleNamespace(primary=primary)),
            scheduler=FakeScheduler() if scheduler is None else scheduler,
            action_timeline=timeline,
            osc=osc,
        )

    def _rows(self):
        _, rows = load_action_timeline(self.path)
        return rows

    # ---- axes ----------------------------------------------------------
    def test_axes_writes_forward_strafe_and_send_result(self) -> None:
        tl = self._timeline()
        stub = self._stub(timeline=tl)
        ok = BackendService._navigator_send_axes(stub, "both", -0.25, 0.75, 200)
        self.assertTrue(ok)
        self.assertEqual(stub.scheduler.calls[0][0], "input_axes")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["input_command"]["forward"], 0.75)   # forward = y
        self.assertAlmostEqual(rows[0]["input_command"]["strafe"], -0.25)   # strafe  = x
        self.assertEqual(rows[0]["actual_send_result"], "sent")
        self.assertEqual(rows[0]["driver_ack"], "none")     # 本机发出 ≠ 驱动回执

    def test_rejected_submit_is_recorded_as_failed(self) -> None:
        tl = self._timeline()
        stub = self._stub(timeline=tl, scheduler=FakeScheduler(accepted=False))
        ok = BackendService._navigator_send_axes(stub, "both", 0.0, 0.5, 150)
        self.assertFalse(ok)
        self.assertEqual(self._rows()[0]["actual_send_result"], "failed")

    def test_non_anyadance_primary_writes_nothing(self) -> None:
        tl = self._timeline()
        stub = self._stub(timeline=tl, primary="osc")
        self.assertFalse(BackendService._navigator_send_axes(stub, "both", 0.0, 1.0, 100))
        self.assertEqual(self._rows(), [])
        self.assertEqual(stub.scheduler.calls, [])

    # ---- turn ----------------------------------------------------------
    def test_turn_records_intent_not_actual_angle(self) -> None:
        tl = self._timeline()
        stub = self._stub(timeline=tl)
        self.assertTrue(BackendService._navigator_send_turn(stub, 4.5))
        self.assertEqual(stub.scheduler.calls[0], ("turn", {"correction_deg": 4.5}))
        row = self._rows()[0]
        self.assertAlmostEqual(row["turn_intent"]["yaw_delta"], 4.5)
        self.assertEqual(row["actual_send_result"], "sent")
        self.assertEqual(row["driver_ack"], "none")

    # ---- motion feedback ----------------------------------------------
    def test_motion_feedback_attaches_velocity_to_later_rows(self) -> None:
        tl = self._timeline()
        osc = FakeOsc({"available": True, "velocity_x": 0.5, "velocity_z": -0.25})
        stub = self._stub(timeline=tl, osc=osc)
        fb = BackendService._navigator_motion_feedback(stub)
        self.assertTrue(fb["available"])
        self.assertEqual(osc.calls, 1)
        self.clock.tick(1.0)
        BackendService._navigator_send_axes(stub, "both", 0.0, 1.0, 200)
        row = self._rows()[0]
        self.assertAlmostEqual(row["osc_velocity"]["vx"], 0.5)
        self.assertAlmostEqual(row["osc_velocity"]["vz"], -0.25)

    def test_motion_unavailable_does_not_zero_the_velocity(self) -> None:
        """读不到速度时必须保留上一份，绝不把"不可观测"写成零速度。"""
        tl = self._timeline()
        stub = self._stub(
            timeline=tl,
            osc=FakeOsc({"available": True, "velocity_x": 0.5, "velocity_z": 0.5}))
        BackendService._navigator_motion_feedback(stub)
        stub.osc = FakeOsc({"available": False})
        self.clock.tick(1.0)
        BackendService._navigator_motion_feedback(stub)
        self.clock.tick(1.0)
        BackendService._navigator_send_turn(stub, 3.0)
        row = self._rows()[0]
        self.assertAlmostEqual(row["osc_velocity"]["vx"], 0.5)
        self.assertAlmostEqual(row["osc_velocity"]["vz"], 0.5)

    # ---- disabled ------------------------------------------------------
    def test_disabled_timeline_hooks_are_noops(self) -> None:
        """没开时间轴时钩子必须照常返回、不抛错、不建文件。"""
        stub = self._stub(timeline=None)
        self.assertTrue(BackendService._navigator_send_axes(stub, "both", 0.0, 1.0, 100))
        self.assertTrue(BackendService._navigator_send_turn(stub, 1.0))
        self.assertFalse(self.path.exists())

    def test_worker_without_osc_does_not_raise(self) -> None:
        stub = self._stub(timeline=None, osc=None)
        fb = BackendService._navigator_motion_feedback(stub)
        self.assertFalse(fb["available"])
        self.assertEqual(fb["reason"], "osc_unavailable")


class ConfigConstructionTests(unittest.TestCase):
    """`from_mapping` 传给 `DriverLogConfig(...)` 的每个关键字都必须真的存在。

    这条守着一个**已经发生过一次**的 P0：`from_mapping` 里加了
    `action_timeline_fps=...`，但 dataclass 里漏了同名字段 ⇒
    `DriverLogConfig.__init__() got an unexpected keyword argument` ⇒
    **整个 BackendService 都构造不出来**（插件根本起不来）。而当时 60 条单测
    + 12 条回归断言全绿——因为它们没有一个真的去构造配置。
    """

    def test_default_config_constructs(self) -> None:
        cfg = PluginConfig.from_mapping({})
        self.assertEqual(cfg.driver_log.action_timeline_fps, 20.0)
        self.assertFalse(cfg.driver_log.action_timeline_enabled)

    def test_all_action_timeline_keys_are_accepted(self) -> None:
        cfg = PluginConfig.from_mapping({"driver_log": {
            "action_timeline_enabled": True,
            "action_timeline_path": "x/at.jsonl",
            "action_timeline_fps": 30.0,
        }})
        self.assertTrue(cfg.driver_log.action_timeline_enabled)
        self.assertEqual(cfg.driver_log.action_timeline_path, "x/at.jsonl")
        self.assertEqual(cfg.driver_log.action_timeline_fps, 30.0)

    def test_out_of_range_fps_is_rejected_not_silently_accepted(self) -> None:
        for bad in (0.0, 999.0, float("nan")):
            with self.subTest(fps=bad):
                with self.assertRaises(ValueError):
                    PluginConfig.from_mapping(
                        {"driver_log": {"action_timeline_fps": bad}})


class PcModeRuntimeTests(unittest.TestCase):
    """PC 模式：把**真实 BackendService** 起起来，验证动作时间轴真的落盘。

    这是三层防线里最外层的"真的能用"测试：走
    `PluginConfig.from_mapping` → `BackendService.__init__` → `start()` 的完整真实
    路径（不是假桩），再调用 service 自己的导航钩子，看 `start()` 亲手创建的
    `ActionTimeline` 有没有写行。
    """

    def test_real_service_writes_action_timeline_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            at = root / "action_timeline.jsonl"
            config = {
                # 只留动作时间轴：测的是接线本身，不是本机有没有 VRChat / VMC / 摄像头。
                "vmc_idle": {"enabled": False, "manage_host_output": False},
                "vrchat_osc": {"enabled": False},
                "driver_log": {"enabled": False,
                               "action_timeline_enabled": True,
                               "action_timeline_path": str(at),
                               "action_timeline_fps": 20.0},
                "vision": {"enabled": False, "source": "none"},
                "input": {"primary": "anyadance"},
            }
            service = BackendService(config, str(REPO),
                                     state_dir=str(root / "state"), dry_run=True)
            service.start()
            try:
                status = service.action_timeline_status()
                self.assertTrue(status["enabled"], "时间轴未在真实 start() 中启用")
                self.assertTrue(status["active"], "时间轴未锚定")
                self.assertEqual(status["fps"], 20.0)

                # 尚未写过行时允许录像侧重锚
                self.assertTrue(service.action_timeline_begin()["ok"])

                service._navigator_send_axes("both", -0.3, 0.85, 200)
                time.sleep(0.25)
                service._navigator_send_turn(6.5)
                time.sleep(0.25)

                after = service.action_timeline_status()
                self.assertGreaterEqual(after["records"], 2,
                                        f"真实服务未落盘：{after}")
                self.assertIsNone(after["last_error"])
            finally:
                service.stop()

            timebase, rows = load_action_timeline(at)
            self.assertIsNotNone(timebase, "未写出时间基准头")
            self.assertEqual(timebase.fps, 20.0)
            self.assertGreaterEqual(len(rows), 2)
            first, second = rows[0], rows[1]
            self.assertAlmostEqual(first["input_command"]["forward"], 0.85)
            self.assertAlmostEqual(first["input_command"]["strafe"], -0.3)
            self.assertIn(first["actual_send_result"], ("sent", "failed"))
            # 本机绝不自称拿到了驱动回执
            self.assertEqual(first["driver_ack"], "none")
            self.assertAlmostEqual(second["turn_intent"]["yaw_delta"], 6.5)
            # 帧号必须按唯一时间基准推进（0.25 s @20fps ≈ 5 帧）
            self.assertGreater(second["frame_index"], first["frame_index"])


if __name__ == "__main__":
    unittest.main()
