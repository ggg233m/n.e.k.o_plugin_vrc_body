from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.cognition import (
    CognitionRuntime,
    SkillPlanner,
    StateEstimator,
)


class CognitionTests(unittest.TestCase):
    def test_state_estimator_tracks_confidence_and_staleness(self) -> None:
        now = [10.0]
        estimator = StateEstimator(clock=lambda: now[0], stale_after_s=1.0)
        fresh = estimator.ingest(
            "yolo",
            "detection",
            {"label": "avatar"},
            confidence=0.9,
            frame_id="frame-1",
        )
        self.assertEqual(fresh["mode"], "nominal")
        self.assertEqual(fresh["sources"]["yolo"]["observation_count"], 1)
        self.assertEqual(fresh["observations"][0]["frame_id"], "frame-1")

        now[0] = 10.5
        degraded = estimator.ingest("vlm", "scene", {"text": "uncertain"}, confidence=0.2)
        self.assertEqual(degraded["mode"], "degraded")
        self.assertIn("low_confidence_observation", degraded["uncertainties"])

        now[0] = 12.0
        stale = estimator.snapshot()
        self.assertEqual(stale["mode"], "unknown")
        self.assertIn("no_recent_cognition_observation", stale["uncertainties"])

    def test_planner_normalizes_single_action_and_rejects_empty_goal(self) -> None:
        now = [1.0]
        planner = SkillPlanner(clock=lambda: now[0])
        planned = planner.plan({
            "goal": "raise hand",
            "action": "arm_pose",
            "params": {"side": "right", "elevation_deg": 110},
            "timeout_s": 2.0,
            "expected": ["right_arm_changed"],
            "abort_if": ["world_changed"],
        })
        planned_data = planned.to_dict()
        self.assertEqual(planned_data["status"], "planned")
        self.assertEqual(planned_data["steps"][0]["action"], "arm_pose")
        self.assertEqual(planned_data["steps"][0]["params"]["side"], "right")
        self.assertEqual(planned_data["steps"][0]["timeout_s"], 2.0)
        self.assertEqual(planned_data["steps"][0]["expected"], ["right_arm_changed"])
        self.assertEqual(planned_data["steps"][0]["abort_if"], ["world_changed"])

        blocked = planner.plan({"goal": "look around"}).to_dict()
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("action", blocked["reason"])

        malformed = planner.plan({
            "steps": [{"action": "wave", "params": {}, "expected": "done"}],
        }).to_dict()
        self.assertEqual(malformed["status"], "blocked")
        self.assertIn("string arrays", malformed["reason"])
        malformed_item = planner.plan({
            "steps": [{"action": "wave", "params": {}, "expected": [1]}],
        }).to_dict()
        self.assertEqual(malformed_item["status"], "blocked")

    def test_cognition_records_rejection_and_requests_replan(self) -> None:
        now = [2.0]
        runtime = CognitionRuntime(
            lambda: {"body": {"state": "idle"}},
            clock=lambda: now[0],
        )
        plan = runtime.plan({"action": "gesture", "params": {"name": "wave"}})
        self.assertEqual(plan["status"], "planned")
        runtime.record_action(
            "gesture",
            {
                "accepted": False,
                "action_id": "a-1",
                "state": "idle",
                "reason": "body output is disabled",
            },
        )
        snapshot = runtime.snapshot()
        self.assertTrue(snapshot["replan_required"])
        self.assertEqual(snapshot["replan_reason"], "execution_rejected")
        self.assertEqual(snapshot["metrics"]["action_count"], 1)

        now[0] = 3.0
        feedback = runtime.feedback({"type": "world_changed", "data": {"entity": "button"}})
        self.assertEqual(feedback["replan_reason"], "world_changed")


if __name__ == "__main__":
    unittest.main()
