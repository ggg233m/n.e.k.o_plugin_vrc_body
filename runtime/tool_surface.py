"""YUI v1.1 面向模型的唯一工具定义和分发层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .yui_adapter import YuiSemanticAdapter
from .yui_session import YuiSessionState


def _schema_error(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> str | None:
    """校验本项目工具使用的 JSON Schema 子集，服务端不能依赖 MCP 客户端代验。"""

    expected = schema.get("type")
    object_keywords = {"properties", "required", "additionalProperties"}
    if expected == "object" or any(key in schema for key in object_keywords):
        if not isinstance(value, Mapping):
            return f"{path} 必须是对象"
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return f"{path}.{key} 为必填字段"
        if schema.get("additionalProperties") is False:
            unknown = sorted(str(key) for key in value if key not in properties)
            if unknown:
                return f"{path} 包含未知字段: {', '.join(unknown)}"
        for key, item in value.items():
            item_schema = properties.get(key)
            if item_schema is None:
                continue
            error = _schema_error(item, item_schema, f"{path}.{key}")
            if error is not None:
                return error
    elif expected == "string":
        if not isinstance(value, str):
            return f"{path} 必须是字符串"
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return f"{path} 长度不能小于 {schema['minLength']}"
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return f"{path} 长度不能大于 {schema['maxLength']}"
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path} 必须是整数"
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{path} 必须是数字"
    elif expected == "boolean":
        if not isinstance(value, bool):
            return f"{path} 必须是布尔值"

    if "enum" in schema and value not in schema["enum"]:
        return f"{path} 不在允许值中"
    if "minimum" in schema and value < schema["minimum"]:
        return f"{path} 不能小于 {schema['minimum']}"
    if "maximum" in schema and value > schema["maximum"]:
        return f"{path} 不能大于 {schema['maximum']}"

    one_of = schema.get("oneOf")
    if one_of is not None:
        matches = sum(_schema_error(value, branch, path) is None for branch in one_of)
        if matches != 1:
            return f"{path} 必须且只能匹配一种目标形式"
    any_of = schema.get("anyOf")
    if any_of is not None and not any(
        _schema_error(value, branch, path) is None for branch in any_of
    ):
        return f"{path} 不匹配任一允许形式"
    denied = schema.get("not")
    if denied is not None and _schema_error(value, denied, path) is None:
        return f"{path} 包含互斥字段"
    return None


def _object_schema(
    properties: dict[str, Any] | None = None,
    *,
    required: list[str] | None = None,
    one_of: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    if one_of:
        schema["oneOf"] = one_of
    return schema


@dataclass(frozen=True, slots=True)
class YuiToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    timeout_s: float

    def as_mcp_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class YuiToolSurface:
    """N.E.K.O 与 MCP 共享的动态 capability 工具面。"""

    def __init__(
        self,
        adapter: YuiSemanticAdapter,
        session: YuiSessionState,
        *,
        host_arm_authorized: bool = False,
        free_coordinate_navigation: bool = False,
        include_player_names: bool = False,
        enable_wander_tool: bool = False,
        command_deadline_s: float = 5.0,
    ) -> None:
        self.adapter = adapter
        self.session = session
        self.host_arm_authorized = bool(host_arm_authorized)
        self.free_coordinate_navigation = bool(free_coordinate_navigation)
        self.include_player_names = bool(include_player_names)
        self.enable_wander_tool = bool(enable_wander_tool)
        self.command_deadline_s = max(0.1, float(command_deadline_s))

    def definitions(self) -> list[YuiToolDefinition]:
        if self.session.session <= 0:
            return []
        normal_timeout = self.command_deadline_s + 2.0
        composite_timeout = self.command_deadline_s * 3.0 + 10.0
        definitions = [
            YuiToolDefinition(
                "npc.observe",
                "读取 YUI 世界已确认的控制态、能力、NPC、玩家槽位、近期感知与语义目录；不猜测缺失事实。",
                _object_schema(),
                10.0,
            ),
            YuiToolDefinition(
                "npc.estop",
                "通过最高优先级 MIDI 通道立即锁存急停。只在危险、失控或明确要求急停时调用。",
                _object_schema(
                    {"reason": {"type": "string", "minLength": 1, "maxLength": 160}},
                    required=["reason"],
                ),
                10.0,
            ),
        ]
        if not self.host_arm_authorized or not self.session.host_arm_authorized:
            return definitions
        if self.session.control_state == "safe_idle":
            definitions.append(
                YuiToolDefinition(
                    "npc.arm",
                    "宿主已授权当前 session 时，将 NPC 从 safe_idle 切入 external；不会由其他工具自动调用。",
                    _object_schema(),
                    normal_timeout,
                )
            )
            return definitions
        if self.session.control_state not in {"external", "moving", "action"}:
            return definitions

        caps = set(self.session.capabilities)
        operation_tools = self.session.operation_lifecycle
        speed_schema: dict[str, Any] = {"type": "number", "minimum": 0}
        if self.session.max_speed_mps is not None:
            speed_schema["maximum"] = self.session.max_speed_mps

        if operation_tools and {"goto", "navmesh", "anchors"} <= caps:
            anchor_schema: dict[str, Any] = {"type": "string"}
            anchors = self.session.observe()["semantic_keys"].get("anchor", [])
            if anchors:
                anchor_schema["enum"] = anchors
            definitions.append(
                YuiToolDefinition(
                    "npc.go_to",
                    "前往 Inspector 明确发布的语义锚点；不得复制或猜测锚点坐标。",
                    _object_schema(
                        {"anchor_key": anchor_schema, "speed_mps": speed_schema},
                        required=["anchor_key"],
                    ),
                    normal_timeout,
                )
            )
        if (
            operation_tools
            and self.free_coordinate_navigation
            and {"goto", "navmesh"} <= caps
        ):
            definitions.append(
                YuiToolDefinition(
                    "npc.go_to_xyz",
                    "前往宿主明确启用后的世界 X/Z 坐标；发送前严格校验 activity bounds。",
                    _object_schema(
                        {
                            "x": {"type": "number"},
                            "z": {"type": "number"},
                            "yaw": {"type": "number"},
                            "speed_mps": speed_schema,
                        },
                        required=["x", "z"],
                    ),
                    normal_timeout,
                )
            )
        if operation_tools and {"follow", "navmesh"} <= caps:
            definitions.append(
                YuiToolDefinition(
                    "npc.follow",
                    "跟随当前观察结果中的 player_slot；槽位失效时不得改猜其他玩家。",
                    _object_schema(
                        {"player_slot": {"type": "integer", "minimum": 0, "maximum": 63}},
                        required=["player_slot"],
                    ),
                    self.command_deadline_s * 2.0 + 2.0,
                )
            )
        if operation_tools:
            definitions.append(
                YuiToolDefinition(
                    "npc.look_at",
                    "注视一个 player_slot，或注视 activity bounds 内的 XYZ 坐标；两种目标必须二选一。",
                    _object_schema(
                        {
                            "player_slot": {"type": "integer", "minimum": 0, "maximum": 63},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "duration_ms": {"type": "integer", "minimum": 0, "maximum": 127000},
                        },
                        required=["duration_ms"],
                        one_of=[
                            {"required": ["player_slot"], "not": {"anyOf": [{"required": ["x"]}, {"required": ["y"]}, {"required": ["z"]}]}},
                            {"required": ["x", "y", "z"], "not": {"required": ["player_slot"]}},
                        ],
                    ),
                    normal_timeout,
                )
            )
        if operation_tools and "actions" in caps:
            action_schema: dict[str, Any] = {"type": "string"}
            actions = self.session.observe()["semantic_keys"].get("action", [])
            if actions:
                action_schema["enum"] = actions
            definitions.append(
                YuiToolDefinition(
                    "npc.act",
                    "播放世界动作目录中的语义动作；需要目标时使用已观察到的 player_slot。",
                    _object_schema(
                        {
                            "action_key": action_schema,
                            "player_slot": {"type": "integer", "minimum": 0, "maximum": 63},
                            "loop": {"type": "boolean", "default": False},
                        },
                        required=["action_key"],
                    ),
                    self.command_deadline_s * 2.0 + 2.0,
                )
            )
        if operation_tools and "expressions" in caps:
            expression_schema: dict[str, Any] = {"type": "string"}
            expressions = self.session.observe()["semantic_keys"].get("expression", [])
            if expressions:
                expression_schema["enum"] = expressions
            definitions.append(
                YuiToolDefinition(
                    "npc.set_expression",
                    "设置世界表情目录中的语义表情；duration_ms=0 表示持续。",
                    _object_schema(
                        {
                            "expression_key": expression_schema,
                            "duration_ms": {"type": "integer", "minimum": 0, "maximum": 127000},
                        },
                        required=["expression_key", "duration_ms"],
                    ),
                    normal_timeout,
                )
            )
        if "text_utf8" in caps:
            say_properties: dict[str, Any] = {
                "text": {"type": "string", "minLength": 1},
                "estimated_delay_ms": {"type": "integer", "minimum": 0, "maximum": 12700},
                "duration_ms": {"type": "integer", "minimum": 250, "maximum": 31750},
            }
            if "actions" in caps:
                action_keys = self.session.observe()["semantic_keys"].get("action", [])
                action_key_schema: dict[str, Any] = {"type": "string"}
                if action_keys:
                    action_key_schema["enum"] = action_keys
                say_properties["action_key"] = action_key_schema
            definitions.append(
                YuiToolDefinition(
                    "npc.say",
                    "提交 1..384 UTF-8 字节文本；可关联同次确认的动作，只有 voice_stream 发布时才发送语音 cue。",
                    _object_schema(say_properties, required=["text"]),
                    composite_timeout,
                )
            )
        if operation_tools and self.enable_wander_tool and {"wander", "navmesh"} <= caps:
            definitions.append(
                YuiToolDefinition(
                    "npc.wander",
                    "启动 Inspector 预配置的 waypoint 循环巡逻；不接受方向、距离或任意坐标参数。",
                    _object_schema(),
                    normal_timeout,
                )
            )
        definitions.append(
            YuiToolDefinition(
                "npc.stop",
                "停止全部、移动或动作领域；不会解除 ESTOP。",
                _object_schema(
                    {
                        "scope": {
                            "type": "string",
                            "enum": ["all", "movement", "action"],
                            "default": "all",
                        }
                    }
                ),
                normal_timeout,
            )
        )
        return definitions

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        available = {definition.name: definition for definition in self.definitions()}
        definition = available.get(name)
        if definition is None:
            return {
                "status": "failed",
                "error": "tool_unavailable",
                "detail": f"工具 {name!r} 在当前 capability/控制态下不可用",
                "midi_sent": False,
            }
        if arguments is not None and not isinstance(arguments, Mapping):
            return {
                "status": "failed",
                "error": "invalid_arguments",
                "detail": "arguments 必须是对象",
                "midi_sent": False,
            }
        values = dict(arguments or {})
        validation_error = _schema_error(values, definition.input_schema)
        if validation_error is not None:
            return {
                "status": "failed",
                "error": "invalid_arguments",
                "detail": validation_error,
                "midi_sent": False,
            }
        if name == "npc.observe":
            return self.adapter.observe(include_player_names=self.include_player_names)
        if name == "npc.arm":
            return self.adapter.arm()
        if name == "npc.go_to":
            return self.adapter.go_to(**values)
        if name == "npc.go_to_xyz":
            return self.adapter.go_to_xyz(**values)
        if name == "npc.follow":
            return self.adapter.follow(**values)
        if name == "npc.look_at":
            return self.adapter.look_at(**values)
        if name == "npc.act":
            return self.adapter.act(**values)
        if name == "npc.set_expression":
            return self.adapter.set_expression(**values)
        if name == "npc.say":
            return self.adapter.say(**values)
        if name == "npc.wander":
            return self.adapter.wander()
        if name == "npc.stop":
            return self.adapter.stop(**values)
        if name == "npc.estop":
            return self.adapter.estop(**values)
        return {
            "status": "failed",
            "error": "tool_unavailable",
            "detail": f"未实现工具 {name!r}",
            "midi_sent": False,
        }


__all__ = ["YuiToolDefinition", "YuiToolSurface"]
