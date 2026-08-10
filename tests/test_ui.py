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
        self.assertEqual(manifest["plugin"]["version"], "0.13.9")
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
        self.assertIn(".catalog()", context_source)
        self.assertNotIn(".list()", context_source)
        self.assertIn("asyncio.to_thread", list_source)
        self.assertIn("asyncio.to_thread", play_source)

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
