"""LLM 工具的集中元数据定义。"""

WORLD_PRECONDITIONS = {
    "type": "array",
    "minItems": 1,
    "maxItems": 16,
    "description": "执行前必须由最新世界状态满足的条件；字段或阈值非法时动作会被拒绝。",
    "items": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["world_available", "entity_visible", "event_recent"],
            },
            "entity_id": {"type": "string", "minLength": 1, "maxLength": 96},
            "event_type": {"type": "string", "minLength": 1, "maxLength": 64},
            "target_id": {"type": "string", "minLength": 1, "maxLength": 96},
            "source": {"type": "string", "minLength": 1, "maxLength": 48},
            "label": {"type": "string", "minLength": 1, "maxLength": 64},
            "state": {"type": "string", "minLength": 1, "maxLength": 64},
            "min_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "max_age_ms": {"type": "number", "minimum": 0, "maximum": 60000},
        },
        "required": ["kind"],
        "additionalProperties": False,
    },
}

BODY_ENABLE = {
    "name": "body_enable",
    "description": "显式启用 AnyaDance 身体姿态输出。启用后从标准 T Pose 开始以 60 Hz 输出。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_DISABLE = {
    "name": "body_disable",
    "description": "平滑回到标准 T Pose，发送安全保持帧后停止 AnyaDance 输出。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_ARM_POSE = {
    "name": "body_arm_pose",
    "description": "用抬升角和方位角设置任意方向的手臂姿态；手会随手臂方向旋转，再叠加掌心和手腕偏移。姿态保持到下一条命令。",
    "parameters": {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["left", "right", "both"]},
            "elevation_deg": {"type": "number", "minimum": 0, "maximum": 180},
            "azimuth_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0, "description": "0 向前，90 向右，-90 向左，180 向后"},
            "plane": {"type": "string", "enum": ["front", "side"], "description": "旧版兼容参数；仅在未提供 azimuth_deg 时使用"},
            "reach": {"type": "number", "minimum": 0.3, "maximum": 1.0, "default": 0.9},
            "palm": {"type": "string", "enum": ["neutral", "forward", "down", "inward"], "default": "neutral"},
            "wrist_pitch_deg": {"type": "number", "minimum": -90, "maximum": 90, "default": 0},
            "wrist_yaw_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0},
            "wrist_roll_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 600},
        },
        "required": ["side", "elevation_deg"],
    },
}

BODY_MOVE_HAND = {
    "name": "body_move_hand",
    "description": "把一只手移动到 HMD、胸口或髋部锚点附近；手会随肩到目标的方向旋转，再叠加掌心和手腕偏移。",
    "parameters": {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["left", "right"]},
            "relative_to": {"type": "string", "enum": ["hmd", "chest", "hip"], "default": "chest"},
            "x_m": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "y_m": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "z_m": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "palm": {"type": "string", "enum": ["neutral", "forward", "down", "inward"], "default": "neutral"},
            "wrist_pitch_deg": {"type": "number", "minimum": -90, "maximum": 90, "default": 0},
            "wrist_yaw_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0},
            "wrist_roll_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 600},
        },
        "required": ["side", "x_m", "y_m", "z_m"],
    },
}

BODY_HAND = {
    "name": "body_hand",
    "description": "设置一只或双手的开掌、握拳、抓握或指向手势。grip 会同时发送 VR 控制器握持输入。",
    "parameters": {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["left", "right", "both"]},
            "pose": {"type": "string", "enum": ["open", "fist", "grip", "point"]},
            "strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 1},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 300},
        },
        "required": ["side", "pose"],
    },
}

BODY_REACH_AND_GRAB = {
    "name": "body_reach_and_grab",
    "description": "向局部语义目标伸手并在最后阶段握持。视觉目标动作应携带 world_observe 返回实体的 preconditions；只能确认 grip 已触发，不能确认实际拿到 VRChat 物体。",
    "parameters": {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["left", "right"]},
            "height": {"type": "string", "enum": ["waist", "chest", "head"]},
            "direction": {"type": "string", "enum": ["forward", "inward", "outward"], "default": "forward"},
            "distance_m": {"type": "number", "minimum": 0.15, "maximum": 0.70, "default": 0.35},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 700},
            "preconditions": WORLD_PRECONDITIONS,
        },
        "required": ["side", "height"],
    },
}

BODY_GESTURE = {
    "name": "body_gesture",
    "description": "播放受控短手势并恢复动作前姿态。支持挥手、点头、鞠躬、摇头、耸肩、思考、指向、招手靠近、鼓掌、惊讶、安慰和叹气。",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": [
                    "wave", "nod", "bow", "shake_head", "shrug", "think",
                    "point", "beckon", "clap", "surprise", "comfort", "sigh",
                ],
            },
            "side": {"type": "string", "enum": ["left", "right", "both"], "default": "right"},
            "intensity": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
        },
        "required": ["name"],
    },
}

BODY_EXPRESS = {
    "name": "body_express",
    "description": "按语义意图请求自然表达动作。状态机优先从真实 VMD 动作目录选片，无匹配时回退到程序化覆盖层，并保护正在播放的高优先级全身动作。",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "greet", "agree", "disagree", "explain", "present", "think",
                    "celebrate", "question", "emphasize", "beckon", "comfort",
                    "apologize", "surprise", "shrug", "clap", "laugh", "sigh",
                    "idle", "pose", "stretch", "playful",
                ],
            },
            "side": {"type": "string", "enum": ["auto", "left", "right", "both"], "default": "auto"},
            "intensity": {"type": "number", "minimum": 0, "maximum": 1, "description": "省略时使用该意图的自然默认强度"},
            "duration_ms": {"type": "integer", "minimum": 500, "maximum": 5000, "description": "省略时使用该意图的自然默认时长"},
        },
        "required": ["intent"],
    },
}

BODY_SEQUENCE = {
    "name": "body_sequence",
    "description": "异步执行由 arm_pose、hand、move_hand、gesture 和 wait 组成的动作序列。最多 16 步、4 次循环、总时长 30 秒。",
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["arm_pose", "hand", "move_hand", "gesture", "wait"]},
                        "side": {"type": "string"},
                        "elevation_deg": {"type": "number"},
                        "azimuth_deg": {"type": "number"},
                        "reach": {"type": "number"},
                        "palm": {"type": "string"},
                        "wrist_pitch_deg": {"type": "number"},
                        "wrist_yaw_deg": {"type": "number"},
                        "wrist_roll_deg": {"type": "number"},
                        "pose": {"type": "string"},
                        "strength": {"type": "number"},
                        "relative_to": {"type": "string"},
                        "x_m": {"type": "number"},
                        "y_m": {"type": "number"},
                        "z_m": {"type": "number"},
                        "name": {"type": "string"},
                        "intensity": {"type": "number"},
                        "duration_ms": {"type": "integer"},
                    },
                    "required": ["type"],
                },
            },
            "loop_count": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1},
        },
        "required": ["steps"],
    },
}

BODY_CANCEL = {
    "name": "body_cancel",
    "description": "取消当前动作或指定 action_id 的当前动作，停在已经到达的合法姿态。",
    "parameters": {
        "type": "object",
        "properties": {"action_id": {"type": "string", "description": "省略时取消任意当前动作"}},
        "required": [],
    },
}

BODY_LIST_CLIPS = {
    "name": "body_list_clips",
    "description": "列出 motions 白名单目录内可播放的 AnyaDance .nya 预制动作及无效文件。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_PLAY_CLIP = {
    "name": "body_play_clip",
    "description": "按逻辑名称播放 motions 目录内的 .nya 预制动作，支持速度、有限循环、HMD 锚定、过渡和结束恢复。",
    "parameters": {
        "type": "object",
        "properties": {
            "clip_name": {"type": "string", "description": "不含路径或扩展名的预制动作名称，支持中文"},
            "speed": {"type": "number", "minimum": 0.25, "maximum": 3.0, "default": 1.0},
            "loop_count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
            "transition_ms": {"type": "integer", "minimum": 0, "maximum": 5000, "default": 400},
            "anchor": {"type": "boolean", "default": True, "description": "把片段第一帧 HMD 的 X/Z 对齐到当前姿态"},
            "restore_after": {"type": "boolean", "default": False, "description": "播放结束后回到播放前姿态"},
        },
        "required": ["clip_name"],
    },
}

BODY_AVATAR_PARAMETER = {
    "name": "body_avatar_parameter",
    "description": "通过 VRChat OSC 设置当前 Avatar 的 Bool、Int 或 Float 参数。参数必须已存在于该 Avatar；UDP 发送成功不代表 VRChat 已应用。",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 128},
            "value": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "integer", "minimum": -2147483648, "maximum": 2147483647},
                    {"type": "number"},
                ]
            },
        },
        "required": ["name", "value"],
    },
}

BODY_VRCHAT_INPUT = {
    "name": "body_vrchat_input",
    "description": "通过 AnyaDance 虚拟 Index 控制器优先发送一次左/右手 Grab、Use 或 Drop 输入；没有可用驱动时回退到 VRChat OSC，并自动释放按钮。无法确认 Pickup 结果。",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["grab", "use", "drop"]},
            "side": {"type": "string", "enum": ["left", "right"]},
            "hold_ms": {"type": "integer", "minimum": 20, "maximum": 1000, "default": 100},
        },
        "required": ["action", "side"],
    },
}

VRC_CONTROLLER_INPUT = {
    "name": "vrc_controller_input",
    "description": "直接设置 AnyaDance 虚拟 Index 控制器的摇杆或按钮。输入经过有限时长和范围保护，返回的是本机调度接受结果，不代表 VRChat 已执行绑定动作。",
    "parameters": {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["left", "right"]},
            "control": {"type": "string", "enum": ["stick", "trigger", "grip", "menu", "a", "b"]},
            "x": {"type": "number", "minimum": -1, "maximum": 1},
            "y": {"type": "number", "minimum": -1, "maximum": 1},
            "pressed": {"type": "boolean", "default": True},
            "value": {"type": "number", "minimum": 0, "maximum": 1, "default": 1},
            "duration_ms": {"type": "integer", "minimum": 20, "maximum": 10000, "default": 250},
        },
        "required": ["side", "control"],
    },
}

VRC_MENU_NAVIGATE = {
    "name": "vrc_menu_navigate",
    "description": "用右侧虚拟 Index 摇杆短暂导航 VRChat 快捷菜单；x/y 为 -1 到 1，超时自动回中。",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
            "y": {"type": "number", "minimum": -1, "maximum": 1, "default": 0},
            "duration_ms": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 250},
        },
        "required": [],
    },
}

VRC_JUMP = {
    "name": "vrc_jump",
    "description": "通过右侧虚拟 Index A 键发送一次跳跃脉冲。无法确认 VRChat 当前绑定是否使用 A 键跳跃。",
    "parameters": {
        "type": "object",
        "properties": {"hold_ms": {"type": "integer", "minimum": 20, "maximum": 1000, "default": 100}},
        "required": [],
    },
}

VRC_AUTONOMY_STATUS = {
    "name": "vrc_autonomy_status",
    "description": "读取 VRChat 自主控制授权、降级原因、当前目标和世界 revision。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VRC_AUTONOMY_GOAL = {
    "name": "vrc_autonomy_goal",
    "description": "提交一个受安全策略约束的当前实例自主目标；必须先手动 arm。",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "minLength": 1, "maxLength": 256},
            "kind": {"type": "string", "enum": ["explore", "approach", "follow", "interact", "socialize"]},
        },
        "required": ["goal"],
    },
}

VRC_AUTONOMY_STOP = {
    "name": "vrc_autonomy_stop",
    "description": "停止自主目标并释放 AnyaDance 控制器输入。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_STOP = {
    "name": "body_stop",
    "description": "最高优先级急停：冻结当前合法姿态、释放所有输入、清空队列并锁定后续动作。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_RESET = {
    "name": "body_reset",
    "description": "解除停止或非致命故障并平滑恢复标准 T Pose。输出未启用时不会自动启用。",
    "parameters": {
        "type": "object",
        "properties": {"duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 600}},
        "required": [],
    },
}

BODY_STATUS = {
    "name": "body_status",
    "description": "读取 AnyaDance 发送状态、当前动作、手臂和手部状态、安全锁定、队列及调度指标。UDP 本身没有响应；启用并收到驱动遥测时，driver_log 可确认 AnyaDance 是否实际处理了命令。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_AWARENESS = {
    "name": "body_awareness",
    "description": "读取 LLM 可理解的实时身体自知：当前/上一动作、切换关系、进度与剩余时间，以及双臂、双手和头部的语义姿态。连续动作、切换动作或回答当前在做什么之前应先调用。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

WORLD_OBSERVE = {
    "name": "world_observe",
    "description": "读取最近的视觉世界状态。结果来自可选的 VRChat 画面检测器/VLM，包含目标、事件、置信度和不确定性；没有新观测时必须按 unknown 处理，不能把空结果当成世界为空。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VRC_VISION_STATUS = {
    "name": "vrc_vision_status",
    "description": "读取本地视觉采集器、检测器和世界状态的运行情况；没有帧或检测器时必须按 unknown 处理。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VRC_VISION_START = {
    "name": "vrc_vision_start",
    "description": "启动或重启独立 VRChat 屏幕捕获与本地感知线程。只启动观察，不会启用身体输出或自主移动。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VRC_VISION_STOP = {
    "name": "vrc_vision_stop",
    "description": "停止独立 VRChat 屏幕捕获、释放捕获句柄并停止世界上下文推送；不会关闭 AnyaDance 后端。",
    "parameters": {
        "type": "object",
        "properties": {"reason": {"type": "string", "maxLength": 160}},
        "required": [],
    },
}

VRC_VISION_FRAME = {
    "name": "vrc_vision_frame",
    "description": (
        "取最近一帧 VRChat 画面来亲眼看看。适合确认检测器没有识别出的东西："
        "对方是谁、菜单开着没、界面上写了什么。看到的一切都是画面猜测，只能用来理解，"
        "不能写进 world_state，也不能拿来满足 body_reach_and_grab 的 preconditions——"
        "那条路必须用 world_observe 给出的 entity_id 与置信度。画面过期或采集已停止时"
        "返回 available=false，此时按看不见处理，不要沿用上一次看到的内容。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "max_age_ms": {
                "type": "integer",
                "minimum": 250,
                "maximum": 30000,
                "default": 3000,
                "description": "可接受的画面陈旧上限；超过则返回 available=false 而不是给旧画面。",
            }
        },
        "required": [],
    },
}

BODY_LOCOMOTION = {
    "name": "body_locomotion",
    "description": "优先通过 AnyaDance 虚拟 Index 左摇杆实现移动，没有可用驱动时回退到 VRChat OSC；直到超时或下一次调用。前后左右对应游戏摇杆输入：forward=1.0, backward=-1.0, left=-1.0, right=1.0；可同时设置斜向移动。",
    "parameters": {
        "type": "object",
        "properties": {
            "vertical": {"type": "number", "minimum": -1.0, "maximum": 1.0, "default": 0, "description": "前后轴：1.0=前进，-1.0=后退"},
            "horizontal": {"type": "number", "minimum": -1.0, "maximum": 1.0, "default": 0, "description": "左右轴：-1.0=左移，1.0=右移"},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 10000, "default": 1000, "description": "持续时间；超时后自动归零"},
        },
        "required": [],
    },
}

BODY_TURN = {
    "name": "body_turn",
    "description": "优先通过 AnyaDance 虚拟 Index 右摇杆水平轴实现转身，没有可用驱动时回退到 VRChat OSC；正值向右转，负值向左转。",
    "parameters": {
        "type": "object",
        "properties": {
            "horizontal": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "转身速度：-1.0=最快左转，1.0=最快右转"},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 10000, "default": 500, "description": "持续时间；超时后自动归零"},
        },
        "required": ["horizontal"],
    },
}

BODY_STOP_MOVEMENT = {
    "name": "body_stop_movement",
    "description": "立即停止所有移动和转身轴，将所有 locomotion 轴归零。不影响手臂姿态和手部动作。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_CHATBOX = {
    "name": "body_chatbox",
    "description": "通过 VRChat OSC /chatbox/input 发送文本到聊天框，周围玩家可见。文本限制 144 字符。immediate=true 时立即显示；false 时仅在打字时显示。",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 144},
            "immediate": {"type": "boolean", "default": True, "description": "true=立即显示，false=仅打字时显示"},
        },
        "required": ["text"],
    },
}
