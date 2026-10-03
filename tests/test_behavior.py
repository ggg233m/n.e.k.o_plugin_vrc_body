from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401
from neko_anyadance_body.behavior import (
    BehaviorStateMachine,
    EXPRESSION_INTENTS,
    expression_admission,
    resolve_expression,
)


class BehaviorStateMachineTests(unittest.TestCase):
    def test_semantic_intent_resolves_defaults_and_alternating_side(self) -> None:
        resolved = resolve_expression(
            "explain",
            side="auto",
            intensity=None,
            duration_ms=None,
            alternate_side="left",
        )
        self.assertIn("explain", EXPRESSION_INTENTS)
        self.assertEqual(resolved["gesture"], "explain")
        self.assertEqual(resolved["side"], "left")
        self.assertEqual(resolved["energy"], 0.40)
        self.assertEqual(resolved["duration_ms"], 1500)

    def test_unknown_intent_raises_a_readable_value_error(self) -> None:
        """未知 intent 必须显式报错并列出合法值，不能是 KeyError。

        上游只做 str(params.get("intent") or "")，VLM 编出来的 intent 或外部调用
        都可能不合法；静默降级成某个默认表情会把「编错了」伪装成「角色莫名叹气」。
        """
        with self.assertRaises(ValueError) as ctx:
            resolve_expression("不存在的意图", side="auto", intensity=None,
                               duration_ms=None, alternate_side="left")
        message = str(ctx.exception)
        self.assertIn("unknown expression intent", message)
        self.assertNotIn("KeyError", message)
        for known in EXPRESSION_INTENTS:
            self.assertIn(known, message)
        # 空串同样要报，不能落到某个默认 profile 上
        with self.assertRaises(ValueError):
            resolve_expression("", side="auto", intensity=None,
                               duration_ms=None, alternate_side="left")

    def test_state_machine_tracks_layers_transition_and_bounded_history(self) -> None:
        machine = BehaviorStateMachine(history_size=4)
        machine.activate_base(action_id="pose", kind="arm_pose", now=1.0, params={"side": "right"})
        machine.activate_overlay(action_id="expression", kind="express", now=1.1, params={"gesture": "nod"})
        snapshot = machine.snapshot(runtime_state="moving", now=1.5)
        self.assertEqual(snapshot["mode"], "posing")
        self.assertEqual(snapshot["base"]["priority"], 80)
        self.assertEqual(snapshot["overlays"][0]["mode"], "expressing")
        self.assertEqual(snapshot["active_layers"], ["base", "expression"])
        self.assertEqual(snapshot["transition"]["strategy"], "current_pose_snapshot_crossfade")

        machine.finish_overlay(action_id="expression", now=2.0, outcome="completed")
        machine.finish_base(action_id="pose", now=2.1, outcome="completed")
        for index in range(6):
            machine.activate_base(action_id=str(index), kind="gesture", now=3.0 + index)
            machine.finish_base(action_id=str(index), now=3.5 + index, outcome="completed")
        self.assertLessEqual(len(machine.snapshot(runtime_state="idle", now=10.0)["history"]), 4)

    def test_full_body_expression_is_blocked_during_clip_but_head_motion_is_allowed(self) -> None:
        snapshot = {
            "state": "moving",
            "behavior": {"base": {"mode": "clip"}},
        }
        self.assertFalse(expression_admission(snapshot, "explain")[0])
        self.assertTrue(expression_admission(snapshot, "nod")[0])


if __name__ == "__main__":
    unittest.main()
