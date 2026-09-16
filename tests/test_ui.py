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
                "body_freeze",
                "llm_freeze_allow",
                "llm_freeze_deny",
                "vrc_autonomy_arm",
                "vrc_autonomy_disarm",
                "vrc_vision_start",
                "vrc_vision_stop",
                "body_chatbox_relay_enable",
                "body_chatbox_relay_disable",
                "vmc_recalibrate",
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
        # 急停留在共用表，解除按「谁锁的谁能解」分流：模型自己踩的刹车由
        # body_stop(scope="unfreeze") 自己松开，面板急停与故障闩锁仍只认 body_reset。
        self.assertIn("body_stop", debug_commands)
        self.assertIn("body_reset", switches)
        # 但「用户下的急停」必须有一条模型走不到的入口。共用入口拿不到调用方身份，
        # 参数里的 source 是模型也能填的字符串；panel_command 的 agent_auto=False
        # 才是宿主真会执行的拦截，所以面板急停走 body_freeze 而不是 body_stop。
        self.assertIn("body_freeze", switches)
        self.assertNotIn("body_freeze", debug_commands)
        # 「是否允许模型急停」同理：这条权限本身要是模型改得动，开关就白设了。
        self.assertIn("llm_freeze_allow", switches)
        self.assertIn("llm_freeze_deny", switches)
        self.assertNotIn("llm_freeze_allow", debug_commands)
        self.assertNotIn("llm_freeze_deny", debug_commands)
        # 重新校准要救的故障（零点锁错，动作一直是歪的）只有人眼看得出来：中转不
        # 报错、帧也在正常收，模型手里没有任何能判断「歪没歪」的信号。给它这个入口
        # 只会让它在看不见的状态上乱按，而每按一次角色都可能停在最后一帧。
        self.assertIn("vmc_recalibrate", switches)
        self.assertNotIn("vmc_recalibrate", debug_commands)
        # 读取刻意留给 Agent，否则它无从知道能力关着，也就没法提示用户去面板打开。
        # 三份状态已经并进 body_status(include=…)，共用表里只剩这一个入口。
        self.assertIn("body_status", debug_commands)
        self.assertNotIn("vrc_autonomy_status", debug_commands)
        self.assertNotIn("vrc_vision_status", debug_commands)

    def test_manifest_declares_hosted_debug_panel(self) -> None:
        manifest = self._manifest()
        self.assertEqual(manifest["plugin"]["version"], "0.13.30")
        self.assertTrue(manifest["plugin"]["ui"]["enabled"])
        panel = manifest["plugin"]["ui"]["panel"][0]
        self.assertEqual(panel["id"], "debug")
        self.assertEqual(panel["entry"], "ui/panel.tsx")
        self.assertEqual(panel["context"], "debug_dashboard")
        self.assertEqual(panel["permissions"], ["state:read", "action:call"])

    def test_debug_context_version_reads_from_manifest(self) -> None:
        # 面板副标题按清单显示当前版本。写死字面量的那份已经漏了七个版本没跟上
        # （显示 0.13.22，清单是 0.13.29）——版本号有两处真值时，漏掉的永远是
        # 没人看的那处。这条测试锁住这一对，将来发版时只要这里挂了，就意味着
        # _plugin_version() 要么返回了过期值，要么读不到清单。
        manifest = self._manifest()
        expected_version = manifest["plugin"]["version"]
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        plugin_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NekoAnyadanceBodyPlugin"
        )
        methods = {node.name: node for node in plugin_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        context_source = ast.unparse(methods["debug_dashboard_context"])
        # 上下文里必须调用 _plugin_version()，而不是写死字面量或拿别的变量凑。
        self.assertIn("_plugin_version()", context_source)
        self.assertNotIn('"0.13.22"', context_source)
        self.assertNotIn('"0.13.23"', context_source)
        # 再确认辅助函数真的读得到清单里那个值。插件模块本身导入宿主 SDK，测试
        # 环境里导不进来（其余插件级断言也因此都是 AST 的），所以这里把函数源码
        # 单独取出来执行，并按真实模块路径设 __file__。
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_plugin_version"
        )
        namespace: dict = {"__file__": str(ROOT / "__init__.py")}
        exec(
            "from functools import lru_cache\nfrom pathlib import Path\nimport tomllib\n"
            + ast.unparse(helper),
            namespace,
        )
        self.assertEqual(namespace["_plugin_version"](), expected_version)

    def test_backend_exposes_context_and_one_bounded_debug_action(self) -> None:
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        commands = set(self._command_tuple("_DEBUG_COMMAND_NAMES"))
        self.assertIn("body_stop", commands)
        self.assertIn("body_status", commands)
        self.assertIn("body_express", commands)
        self.assertIn("body_vrchat_input", commands)
        self.assertIn("body_turn", commands)
        self.assertIn("body_chatbox", commands)
        self.assertIn("world_observe", commands)
        self.assertIn("navigate_vrchat_world", commands)
        self.assertIn("vrc_wander_step", commands)
        # 合并掉的同义入口不能再回到白名单里——一件事只留一个入口是这轮收敛的前提。
        for retired in (
            "body_awareness",
            "body_stop_movement",
            "body_cancel",
            "body_move_hand",
            "body_sequence",
            "body_locomotion",
            "body_play_clip",
            "body_list_clips",
            "body_avatar_parameter",
            "vrc_autonomy_status",
            "vrc_autonomy_stop",
            "vrc_vision_status",
            "vrc_wander_route",
            "vrc_scan_surroundings",
            "vrc_controller_input",
            "vrc_menu_navigate",
            "vrc_jump",
            "observe_vrchat_world",
        ):
            self.assertNotIn(retired, commands)

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
        status_source = ast.unparse(methods["body_status"])
        stop_source = ast.unparse(methods["body_stop"])
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
        # 合并后的读取入口必须真的把三段都取到，否则 include 只是个空壳参数。
        self.assertIn("autonomy.snapshot", status_source)
        self.assertIn("_vision.perception", status_source)
        self.assertIn("_body_snapshot", status_source)
        # 合并后的停止入口同理：撤目标和清轴都要在，且撤目标必须排在清轴之前。
        goal_cleanup_source = ast.unparse(methods["_stop_autonomy_goal"])
        self.assertIn("autonomy.stop", goal_cleanup_source)
        self.assertIn("_stop_autonomy_goal", stop_source)
        self.assertIn("stop_movement", stop_source)
        self.assertLess(stop_source.index("_stop_autonomy_goal"), stop_source.index("stop_movement"))
        # 急停必须能被下急停的人自己解除：模型有权进入 freeze，就得有权离开，
        # 否则它踩一脚刹车就把自己锁死，只能干等用户去点面板。
        unfreeze_source = ast.unparse(methods["_unfreeze"])
        self.assertIn("'unfreeze'", stop_source)
        self.assertIn("_freeze_owner", stop_source)
        self.assertIn("_unfreeze", stop_source)
        # 但「谁锁的谁能解」要真的判来源，而且默认拒绝：故障闩锁的 owner 是 None，
        # 面板急停是 "panel"，两者都只认面板复位。写成 == "panel" 就反了。
        self.assertIn("_freeze_owner != 'llm'", unfreeze_source)
        self.assertIn("stopped_latched", unfreeze_source)
        self.assertIn("'reset'", unfreeze_source)
        # 急停要停「所有动作」，所以 freeze 这一支也得撤掉自主目标：闩锁只让导航器的
        # 指令发不出去，不会让它收手，否则它一路重试到自己超时、目标还挂在 body_status。
        self.assertIn("goal_cleanup", stop_source)
        # 但撤目标是尽力而为的副作用，不能参与成败判定——后端连不上时身体其实已经停住
        # 了，算进去会把一次成功的急停报成失败，调用方于是重发一次急停。
        self.assertIn("{'scope', 'goal_cleanup'}", stop_source)
        # 「允许模型急停」是面板开关，所以 freeze 这一支必须真的读那个标志，而且只
        # 拦模型这一侧——面板自己那颗按钮不受开关影响，否则关掉之后用户也停不下来。
        self.assertIn("_llm_freeze_allowed", stop_source)
        self.assertIn("normalized_source != 'panel'", stop_source)
        # 来源参数必须带下划线前缀，并由共用派发体剥掉：那里把 arguments 原样展开成
        # 关键字参数且拿不到调用方身份，留一个可填的 source 等于让模型冒充用户急停，
        # 既绕过开关，又把自己锁进只有面板能解的那把锁里。
        self.assertIn("str(_source or '')", stop_source)
        self.assertIn("startswith('_')", dispatch_source)
        freeze_source = ast.unparse(methods["body_freeze"])
        self.assertIn("_source='panel'", freeze_source)
        # 开关只管「能不能进 freeze」。顺手封掉 unfreeze 会把模型困在自己下的那次
        # 急停里，反而要用户多点一次面板才能救——那比放它自己解开更糟。
        deny_source = ast.unparse(methods["llm_freeze_deny"])
        self.assertIn("_llm_freeze_allowed = False", deny_source)
        self.assertNotIn("_freeze_owner", deny_source)
        self.assertIn("'llm_freeze'", context_source)
        # 重新校准默认走宿主 T Pose；退路必须是显式布尔参数，不能靠字符串真值。
        # 它决定的是「等一次权威静止姿势」还是「拿下一个动画帧当零点」，用
        # "false" 这种非空字符串误判成 True，救援就变成了第二次锁错。
        recalibrate_source = ast.unparse(methods["vmc_recalibrate"])
        self.assertIn("_boolean('accept_current_pose'", recalibrate_source)
        self.assertIn("recalibrate", recalibrate_source)

        world_observe_source = ast.unparse(methods["world_observe"])
        navigate_source = ast.unparse(methods["navigate_vrchat_world"])
        cancel_source = ast.unparse(methods["_replace_cancelled_semantic_push"])
        navigation_outcome_source = ast.unparse(methods["_push_navigation_outcome"])
        autonomy_goal_source = ast.unparse(methods["vrc_autonomy_goal"])
        wander_step_source = ast.unparse(methods["vrc_wander_step"])
        world_loop_source = ast.unparse(methods["_world_context_loop_run"])
        semantic_text_source = ast.unparse(methods["_semantic_request_text"])
        semantic_push_source = ast.unparse(methods["_push_passive_semantic_parts"])
        self.assertIn("llm_tool", world_observe_source)
        self.assertIn("plugin_entry", navigate_source)
        self.assertIn("manual_arm_required", navigate_source)
        self.assertIn("autonomy.intent", navigate_source)
        self.assertIn("unsupported_spatial_navigation", navigate_source)
        self.assertIn("'depart'", autonomy_goal_source)
        self.assertIn("'wander'", autonomy_goal_source)
        # 方向未定的闲逛必须改走 /autonomy/intent：submit_goal 硬性要求 turn_deg。
        self.assertIn("autonomy.intent", autonomy_goal_source)
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
            "body_vrchat_input",
        ):
            self.assertIn(f'run("{command}"', source)
        # 面板的「停止自主目标」走合并后的 body_stop；「立即急停」则必须离开这条
        # 共用路——参数里的 source 是模型也能填的字符串，靠它区分来源等于没区分。
        # 面板急停改走只有面板能分派的 body_freeze（panel_command，agent_auto=false）。
        self.assertIn('runSwitch("body_freeze")', source)
        self.assertNotIn('source: "panel"', source)
        self.assertIn('run("body_stop", { scope: "navigation" })', source)
        self.assertNotIn('run("vrc_autonomy_stop"', source)
        # 开关类必须走 panel_command，且渲染成滑块而不是按钮——按钮看不出当前
        # 状态，开关滑块的 checked 直接绑定后端真值。
        self.assertIn('item.id === "panel_command"', source)
        self.assertIn('toggle("body_enable", "body_disable")', source)
        self.assertIn('toggle("vrc_autonomy_arm", "vrc_autonomy_disarm")', source)
        self.assertIn('toggle("vrc_vision_start", "vrc_vision_stop")', source)
        self.assertIn('toggle("llm_freeze_allow", "llm_freeze_deny")', source)
        # 聊天框转发是隐私开关：关掉它等于停止把她说的话广播给周围玩家，
        # 所以必须走 panel_command，Agent 不能自行开关。
        self.assertIn('toggle("body_chatbox_relay_enable", "body_chatbox_relay_disable")', source)
        # 默认允许，所以只有显式 false 才算关闭：写成 Boolean(...) 会让旧版本
        # 后端（没有 permissions 段）把「默认允许」显示成「已禁止」。
        self.assertIn("permissions.llm_freeze !== false", source)
        self.assertIn('runSwitch("body_reset"', source)
        # 重新校准要救的是「动作一直是歪的」，而那个故障不产生任何错误状态——
        # 帧照收、last_error 是空的。所以按钮必须常驻，不能藏在某个条件后面。
        self.assertIn('runSwitch("vmc_recalibrate")', source)
        # 严格路径失败时必须有出口：否则用户被留在「角色停在最后一帧」里，
        # 只能重启后端。
        self.assertIn("accept_current_pose: true", source)
        self.assertIn("calibrationNeedsFallback", source)
        for switch in (
            "body_enable",
            "body_disable",
            "vrc_autonomy_arm",
            "vrc_autonomy_disarm",
            "vrc_vision_start",
            "vrc_vision_stop",
            "body_reset",
            "body_freeze",
            "llm_freeze_allow",
            "llm_freeze_deny",
            "body_chatbox_relay_enable",
            "body_chatbox_relay_disable",
            "vmc_recalibrate",
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
