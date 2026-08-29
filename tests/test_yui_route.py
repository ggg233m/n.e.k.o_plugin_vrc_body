"""宿主连续路线编排测试；该能力不得进入 LLM 工具面。"""

from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from yui_npc_controller.runtime.host_route import YuiContinuousRouteRunner


class RecordingRouteAdapter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def go_to_xyz(self, x, z, *, yaw=None, speed_mps=None):
        self.calls.append({"x": x, "z": z, "yaw": yaw, "speed_mps": speed_mps})
        index = len(self.calls)
        return {
            "status": "accepted",
            "op_id": f"1193046:{index}:ABCD",
            "midi_sent": True,
        }


class RouteSession:
    def __init__(self, speeds: list[float]) -> None:
        self.speeds = list(speeds)

    def wait_for_npc_near(self, x, z, distance_m, timeout_s, *, operation_id=None):
        return {"pos": [x, 0.0, z], "speed": self.speeds.pop(0)}

    def wait_for_operation(self, operation_id, timeout_s):
        segment = int(operation_id.split(":")[1])
        if segment < 3:
            return {"op_id": operation_id, "status": "cancelled", "reason": "replaced"}
        return {"op_id": operation_id, "status": "succeeded", "result": "arrived"}


class YuiContinuousRouteTests(unittest.TestCase):
    def test_route_prequeues_points_and_requires_replaced_lifecycle(self) -> None:
        adapter = RecordingRouteAdapter()
        runner = YuiContinuousRouteRunner(adapter, RouteSession([1.3, 1.1]))
        result = runner.run(
            [
                {"x": -2.0, "z": 0.0},
                {"x": 0.0, "z": 2.0},
                {"x": 2.0, "z": 0.0, "yaw": 180.0},
            ],
            speed_mps=1.5,
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["seamless"])
        self.assertEqual(result["handoff_speeds_mps"], [1.3, 1.1])
        self.assertEqual(len(adapter.calls), 3)
        self.assertEqual(adapter.calls[-1]["yaw"], 180.0)

    def test_zero_speed_handoff_is_reported_as_visual_continuity_failure(self) -> None:
        adapter = RecordingRouteAdapter()
        runner = YuiContinuousRouteRunner(adapter, RouteSession([0.0, 1.0]))
        result = runner.run(
            [{"x": -2.0, "z": 0.0}, {"x": 0.0, "z": 2.0}, {"x": 2.0, "z": 0.0}],
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "visual_continuity_lost")
        self.assertFalse(result["seamless"])

    def test_invalid_route_sends_nothing(self) -> None:
        adapter = RecordingRouteAdapter()
        runner = YuiContinuousRouteRunner(adapter, RouteSession([]))
        result = runner.run([])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "invalid_param")
        self.assertEqual(adapter.calls, [])


if __name__ == "__main__":
    unittest.main()
