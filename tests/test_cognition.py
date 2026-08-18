from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.backend.cognition import (
    CognitionRuntime,
    SkillPlanner,
    StateEstimator,
    WorldPrecondition,
    WorldPreconditionGate,
)


class CognitionTests(unittest.TestCase):
    def test_world_precondition_gate_reports_actionable_failures(self) -> None:
        gate = WorldPreconditionGate()
        world = {
            "available": True,
            "entities": [{
                "id": "yolo:cup:7",
                "label": "cup",
                "confidence": 0.72,
                "state": "grabbable",
                "source": ["yolo"],
                "age_ms": 120.0,
                "visible": True,
            }],
            "events": [{
                "type": "hand_near_target",
                "target_id": "yolo:cup:7",
                "confidence": 0.8,
                "source": ["mediapipe"],
                "age_ms": 80.0,
            }],
            "status": {"last_observation_age_ms": 120.0},
        }
        passed = gate.evaluate([
            {
                "kind": "entity_visible",
                "entity_id": "yolo:cup:7",
                "label": "cup",
                "state": "grabbable",
                "source": "yolo",
                "min_confidence": 0.7,
                "max_age_ms": 200,
            },
            {
                "kind": "event_recent",
                "event_type": "hand_near_target",
                "target_id": "yolo:cup:7",
                "source": "mediapipe",
                "min_confidence": 0.75,
                "max_age_ms": 100,
            },
        ], world)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["checked"], 2)

        low_confidence = gate.evaluate([{
            "kind": "entity_visible",
            "entity_id": "yolo:cup:7",
            "min_confidence": 0.9,
        }], world)
        self.assertFalse(low_confidence["passed"])
        self.assertEqual(low_confidence["failures"][0]["code"], "entity_low_confidence")

        too_old = gate.evaluate([{
            "kind": "entity_visible",
            "entity_id": "yolo:cup:7",
            "max_age_ms": 100,
        }], world)
        self.assertEqual(too_old["failures"][0]["code"], "entity_observation_too_old")

        missing = gate.evaluate([{
            "kind": "entity_visible",
            "entity_id": "yolo:missing:1",
        }], world)
        self.assertEqual(missing["failures"][0]["code"], "entity_not_visible")

        default_confidence = gate.evaluate([{
            "kind": "entity_visible",
            "entity_id": "yolo:cup:7",
        }], {
            **world,
            "entities": [{**world["entities"][0], "confidence": 0.49}],
        })
        self.assertEqual(
            default_confidence["failures"][0]["code"],
            "entity_low_confidence",
        )

        invalid = gate.evaluate("not-an-array", world)
        self.assertEqual(invalid["failures"][0]["code"], "invalid_world_precondition")

        for malformed_condition in (
            {
                "kind": "entity_visible",
                "entity_id": "yolo:cup:7",
                "min_confidnce": 0.9,
            },
            {
                "kind": "entity_visible",
                "entity_id": "yolo:cup:7",
                "min_confidence": False,
            },
            {
                "kind": "entity_visible",
                "type": "event_recent",
                "entity_id": "yolo:cup:7",
            },
            {
                "kind": "entity_visible",
                "entity_id": "yolo:cup:7",
                "id": "yolo:cup:8",
            },
        ):
            malformed = gate.evaluate([malformed_condition], world)
            self.assertFalse(malformed["passed"])
            self.assertEqual(
                malformed["failures"][0]["code"],
                "invalid_world_precondition",
            )

        malformed_dataclass = gate.evaluate([
            WorldPrecondition(
                kind="entity_visible",
                entity_id="yolo:cup:7",
                min_confidence=-1.0,
            )
        ], world)
        self.assertFalse(malformed_dataclass["passed"])
        self.assertEqual(
            malformed_dataclass["failures"][0]["code"],
            "invalid_world_precondition",
        )

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

        malformed_precondition = planner.plan({
            "action": "reach_and_grab",
            "params": {"side": "right"},
            "preconditions": [{
                "kind": "entity_visible",
                "entity_id": "yolo:cup:7",
                "max_age_mss": 500,
            }],
        }).to_dict()
        self.assertEqual(malformed_precondition["status"], "blocked")
        self.assertTrue(malformed_precondition["replan_required"])
        self.assertEqual(
            malformed_precondition["precondition_check"]["failures"][0]["code"],
            "invalid_world_precondition",
        )

        conflicting_alias = planner.plan({
            "action": "reach_and_grab",
            "params": {"side": "right"},
            "preconditions": None,
            "world_preconditions": [{
                "kind": "entity_visible",
                "entity_id": "yolo:cup:7",
            }],
        }).to_dict()
        self.assertEqual(conflicting_alias["status"], "blocked")
        self.assertEqual(
            conflicting_alias["precondition_check"]["failures"][0]["code"],
            "invalid_world_precondition",
        )

        multi_step = planner.plan({
            "steps": [
                {
                    "action": "reach_and_grab",
                    "params": {"side": "right"},
                    "preconditions": [{
                        "kind": "entity_visible",
                        "entity_id": "yolo:cup:7",
                        "min_confidence": 0.8,
                    }],
                },
                {"action": "hand", "params": {"side": "right", "pose": "grip"}},
            ],
        }).to_dict()
        self.assertEqual(multi_step["status"], "planned")
        self.assertEqual(len(multi_step["steps"]), 2)
        self.assertEqual(
            multi_step["steps"][0]["preconditions"][0]["entity_id"],
            "yolo:cup:7",
        )

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

    def test_cognition_blocks_plan_until_first_step_preconditions_pass(self) -> None:
        world = {
            "available": False,
            "entities": [],
            "events": [],
            "status": {"last_observation_age_ms": None},
        }
        runtime = CognitionRuntime(
            lambda: {"body": {"state": "idle"}},
            world_provider=lambda: world,
            clock=lambda: 5.0,
        )
        goal = {
            "action": "reach_and_grab",
            "params": {"side": "right"},
            "preconditions": [{
                "kind": "entity_visible",
                "entity_id": "yolo:cup:7",
                "min_confidence": 0.7,
                "max_age_ms": 500,
            }],
        }
        blocked = runtime.plan(goal)
        self.assertEqual(blocked["status"], "blocked")
        self.assertTrue(blocked["replan_required"])
        self.assertEqual(
            blocked["precondition_check"]["failures"][0]["code"],
            "entity_not_visible",
        )
        self.assertEqual(runtime.snapshot()["replan_reason"], "world_precondition_failed")

        world.update({
            "available": True,
            "entities": [{
                "id": "yolo:cup:7",
                "label": "cup",
                "confidence": 0.9,
                "source": ["yolo"],
                "age_ms": 20.0,
                "visible": True,
            }],
        })
        planned = runtime.plan(goal)
        self.assertEqual(planned["status"], "planned")
        self.assertTrue(planned["precondition_check"]["passed"])


if __name__ == "__main__":
    unittest.main()
