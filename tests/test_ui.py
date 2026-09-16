from __future__ import annotations

import ast
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HostedUiTests(unittest.TestCase):
    def _manifest(self) -> dict:
        with (ROOT / "plugin.toml").open("rb") as handle:
            return tomllib.load(handle)

    def _command_tuple(self, name: str) -> list[str]:
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        return next(
            list(ast.literal_eval(node.value))
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        )

    def test_manifest_keeps_debug_entry_discoverable_without_import(self) -> None:
        # 安装信息刷新不执行装饰器，清单必须独立提供同一份有界入口参数。
        manifest = self._manifest()
        entry = next(item for item in manifest["plugin"]["entries"] if item["id"] == "debug_command")
        commands = self._command_tuple("_DEBUG_COMMAND_NAMES")
        schema = entry["input_schema"]
        self.assertEqual(schema["required"], ["command"])
        self.assertEqual(schema["properties"]["command"]["enum"], commands)
        self.assertEqual(schema["properties"]["arguments"], {"type": "object", "default": {}})

    def test_capability_switches_are_panel_only_and_hidden_from_agent(self) -> None:
        # 这条测试锁的是一个安全边界，不只是形状：debug_command 是 Agent 与面板
        # 共用的入口，按 getattr 派发且拿不到调用方身份。开关留在那张表里，Agent
        # 就能自己给自己授权自主移动，plugin.toml 的 manual_arm = true 会被绕过。
        # metadata.agent_auto=False 是宿主真正会执行的拦截（见宿主
        # brain/task_executor.py::_agent_visible_plugin_entries 与 _find_plugin_entry）。
        switches = self._command_tuple("_PANEL_SWITCH_NAMES")
        debug_commands = self._command_tuple("_DEBUG_COMMAND_NAMES")
        self.assertEqual(
            set(switches),
            {
                "body_enable",
                "body_disable",
                "body_reset",
                "vrc_autonomy_arm",
                "vrc_autonomy_disarm",
                "vrc_vision_start",
                "vrc_vision_stop",
            },
        )
        self.assertEqual(set(switches) & set(debug_commands), set())

        manifest = self._manifest()
        entries = {item["id"]: item for item in manifest["plugin"]["entries"]}
        self.assertEqual(entries["panel_command"]["input_schema"]["properties"]["command"]["enum"], switches)
        self.assertIs(entries["panel_command"]["metadata"]["agent_auto"], False)
        # 共用入口绝不能反过来带上 agent_auto=false，否则 Agent 连动作都调不到。
        self.assertNotIn("metadata", entries["debug_command"])

        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        plugin_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NekoAnyadanceBodyPlugin"
        )
        methods = {node.name: node for node in plugin_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        panel_source = ast.unparse(methods["panel_command"])
        self.assertIn("agent_auto", panel_source)
        self.assertIn("_PANEL_SWITCH_NAMES", panel_source)
        self.assertIn("ui.action", panel_source)
        # 急停留在共用表、解除急停只在面板：Agent 能踩刹车，不能自己松刹车。
        self.assertIn("body_stop", debug_commands)
        self.assertIn("body_reset", switches)
        # 读取刻意留给 Agent，否则它无从知道能力关着，也就没法提示用户去面板打开。
        self.assertIn("vrc_autonomy_status", debug_commands)
        self.assertIn("vrc_vision_status", debug_commands)

    def test_manifest_declares_hosted_debug_panel(self) -> None:
        manifest = self._manifest()
        self.assertEqual(manifest["plugin"]["version"], "0.13.24")
        self.assertTrue(manifest["plugin"]["ui"]["enabled"])
        panel = manifest["plugin"]["ui"]["panel"][0]
        self.assertEqual(panel["id"], "debug")
        self.assertEqual(panel["entry"], "ui/panel.tsx")
        self.assertEqual(panel["context"], "debug_dashboard")
        self.assertEqual(panel["permissions"], ["state:read", "action:call"])

    def test_backend_exposes_context_and_one_bounded_debug_action(self) -> None:
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        commands = set(self._command_tuple("_DEBUG_COMMAND_NAMES"))
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
        # 两个入口只是白名单不同，派发体共用；断言跟着搬到共用体上。
        dispatch_source = ast.unparse(methods["_run_bounded_command"])
        context_source = ast.unparse(methods["debug_dashboard_context"])
        list_source = ast.unparse(methods["body_list_clips"])
        play_source = ast.unparse(methods["body_play_clip"])
        self.assertIn("ui.context", context_decorators)
        self.assertIn("ui.action", action_source)
        self.assertIn("plugin_entry", action_source)
        self.assertIn("观察与导航", action_source)
        self.assertIn("manual_arm_required", action_source)
        self.assertIn("_DEBUG_COMMAND_NAMES", action_source)
        # 校验必须按传入的白名单做，不能退回去查全集——否则分表形同虚设。
        self.assertIn("allowed", dispatch_source)
        self.assertIn("normalized not in allowed", dispatch_source)
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
        self.assertIn("_execution_result", dispatch_source)
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
        # 动作类命令走共用入口。
        for command in (
            "body_stop",
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
        # 开关类必须走 panel_command，且渲染成滑块而不是按钮——按钮看不出当前
        # 状态，开关滑块的 checked 直接绑定后端真值。
        self.assertIn('item.id === "panel_command"', source)
        self.assertIn('toggle("body_enable", "body_disable")', source)
        self.assertIn('toggle("vrc_autonomy_arm", "vrc_autonomy_disarm")', source)
        self.assertIn('toggle("vrc_vision_start", "vrc_vision_stop")', source)
        self.assertIn('runSwitch("body_reset"', source)
        for switch in (
            "body_enable",
            "body_disable",
            "vrc_autonomy_arm",
            "vrc_autonomy_disarm",
            "vrc_vision_start",
            "vrc_vision_stop",
            "body_reset",
        ):
            self.assertNotIn(f'run("{switch}"', source)
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
