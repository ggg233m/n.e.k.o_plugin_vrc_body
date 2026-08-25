from __future__ import annotations

import ast
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HostedUiTests(unittest.TestCase):
    def test_manifest_declares_hosted_debug_panel(self) -> None:
        with (ROOT / "plugin.toml").open("rb") as handle:
            manifest = tomllib.load(handle)
        self.assertEqual(manifest["plugin"]["version"], "0.13.21")
        self.assertTrue(manifest["plugin"]["ui"]["enabled"])
        panel = manifest["plugin"]["ui"]["panel"][0]
        self.assertEqual(panel["id"], "debug")
        self.assertEqual(panel["entry"], "ui/panel.tsx")
        self.assertEqual(panel["context"], "debug_dashboard")
        self.assertEqual(panel["permissions"], ["state:read", "action:call"])

    def test_backend_exposes_context_and_one_bounded_debug_action(self) -> None:
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_DEBUG_COMMAND_NAMES"
        }
        commands = set(assignments["_DEBUG_COMMAND_NAMES"])
        self.assertIn("body_stop", commands)
        self.assertIn("body_play_clip", commands)
        self.assertIn("body_express", commands)
        self.assertIn("body_avatar_parameter", commands)
        self.assertIn("body_vrchat_input", commands)
        self.assertIn("body_locomotion", commands)
        self.assertIn("body_turn", commands)
        self.assertIn("body_stop_movement", commands)
        self.assertIn("body_chatbox", commands)
        self.assertIn("observe_vrchat_world", commands)
        self.assertIn("navigate_vrchat_world", commands)
        self.assertIn("vrc_scan_surroundings", commands)
        self.assertIn("vrc_wander_step", commands)

        plugin_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NekoAnyadanceBodyPlugin"
        )
        methods = {node.name: node for node in plugin_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        context_decorators = ast.unparse(methods["debug_dashboard_context"]).splitlines()[0]
        action_source = ast.unparse(methods["debug_command"])
        context_source = ast.unparse(methods["debug_dashboard_context"])
        list_source = ast.unparse(methods["body_list_clips"])
        play_source = ast.unparse(methods["body_play_clip"])
        self.assertIn("ui.context", context_decorators)
        self.assertIn("ui.action", action_source)
        self.assertIn("plugin_entry", action_source)
        self.assertIn("观察与导航", action_source)
        self.assertIn("manual_arm_required", action_source)
        self.assertIn(".catalog()", context_source)
        self.assertIn("driver_log", context_source)
        self.assertIn("world_bridge", context_source)
        self.assertIn("semantic_push_rejected", context_source)
        self.assertIn("_world_bridge_thread", context_source)
        self.assertNotIn(".list()", context_source)
        self.assertIn("asyncio.to_thread", list_source)
        self.assertIn("asyncio.to_thread", play_source)

        observe_source = ast.unparse(methods["observe_vrchat_world"])
        navigate_source = ast.unparse(methods["navigate_vrchat_world"])
        cancel_source = ast.unparse(methods["_replace_cancelled_semantic_push"])
        navigation_outcome_source = ast.unparse(methods["_push_navigation_outcome"])
        autonomy_goal_source = ast.unparse(methods["vrc_autonomy_goal"])
        wander_step_source = ast.unparse(methods["vrc_wander_step"])
        world_loop_source = ast.unparse(methods["_world_context_loop_run"])
        semantic_text_source = ast.unparse(methods["_semantic_request_text"])
        semantic_push_source = ast.unparse(methods["_push_passive_semantic_parts"])
        self.assertIn("plugin_entry", observe_source)
        self.assertIn("当前 VRChat 视觉检测", observe_source)
        self.assertIn("plugin_entry", navigate_source)
        self.assertIn("manual_arm_required", navigate_source)
        self.assertIn("autonomy.intent", navigate_source)
        self.assertIn("unsupported_spatial_navigation", navigate_source)
        self.assertIn("'depart'", autonomy_goal_source)
        self.assertIn("'wander'", autonomy_goal_source)
        self.assertIn("_semantic_request_id", wander_step_source)
        self.assertIn("autonomy.wander_step", wander_step_source)
        self.assertNotIn("target_id", wander_step_source)
        self.assertIn("_execution_result", action_source)
        execution_result_source = ast.unparse(methods["_execution_result"])
        self.assertIn("Err", execution_result_source)
        startup_source = ast.unparse(methods["on_startup"])
        self.assertIn("_register_agent_entries", startup_source)
        register_source = ast.unparse(methods["_register_agent_entries"])
        self.assertIn("agent_scan_vrchat_surroundings", register_source)
        self.assertIn("register_dynamic_entry", register_source)
        self.assertIn("neko_anyadance_body.semantic.latest", cancel_source)
        self.assertIn("被动语义任务已取消", cancel_source)
        self.assertIn("movement_not_started", cancel_source)
        self.assertIn("ai_behavior='respond'", cancel_source)
        self.assertIn("outcome_sequence", navigation_outcome_source)
        self.assertIn("ai_behavior='respond'", navigation_outcome_source)
        self.assertIn("_fetch_frame_image_part", navigation_outcome_source)
        self.assertIn("execution_summary", navigation_outcome_source)
        self.assertIn("world_observation_verified", navigation_outcome_source)
        self.assertIn("_push_navigation_outcome", world_loop_source)
        self.assertIn("agent_wander_direction_unresolved", semantic_text_source)
        self.assertIn("vrc_wander_step", semantic_text_source)
        self.assertIn("semantic_wake", world_loop_source)
        self.assertIn("ai_behavior='respond' if wake else 'read'", semantic_push_source)

    def test_panel_uses_only_hosted_ui_runtime_and_covers_controls(self) -> None:
        source = (ROOT / "ui" / "panel.tsx").read_text(encoding="utf-8")
        self.assertIn('from "@neko/plugin-ui"', source)
        self.assertNotIn("dangerouslySetInnerHTML", source)
        self.assertNotIn("fetch(", source)
        self.assertIn("未索引", source)
        for command in (
            "body_enable",
            "body_stop",
            "body_reset",
            "body_arm_pose",
            "body_hand",
            "body_reach_and_grab",
            "body_gesture",
            "body_express",
            "body_play_clip",
            "body_avatar_parameter",
            "body_vrchat_input",
        ):
            self.assertIn(f'run("{command}"', source)
        self.assertIn("wrist_pitch_deg: wristPitch", source)
        self.assertIn("wrist_yaw_deg: wristYaw", source)
        self.assertIn("wrist_roll_deg: wristRoll", source)
        gesture_options = source.split("const gestureOptions = [", 1)[1].split("]", 1)[0]
        for invalid_name in ("idle", "pose", "stretch", "playful"):
            self.assertNotIn(f'value: "{invalid_name}"', gesture_options)
        self.assertNotIn('key: "udpConnection"', source)
        self.assertIn('label="身体动作意图"', source)
        self.assertIn("请求身体语义动作", source)


if __name__ == "__main__":
    unittest.main()
