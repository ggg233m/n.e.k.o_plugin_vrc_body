"""Central LLM tool metadata."""

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
    "description": "向局部语义目标伸手并在最后阶段握持。只能确认 grip 已触发，不能确认实际拿到 VRChat 物体。",
    "parameters": {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["left", "right"]},
            "height": {"type": "string", "enum": ["waist", "chest", "head"]},
            "direction": {"type": "string", "enum": ["forward", "inward", "outward"], "default": "forward"},
            "distance_m": {"type": "number", "minimum": 0.15, "maximum": 0.70, "default": 0.35},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 700},
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
    "description": "通过 VRChat OSC 安全脉冲一次左/右手 Grab、Use 或 Drop 输入，并自动释放按钮。仅 VR 模式下的部分输入有效，无法确认 Pickup 结果。",
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
    "description": "读取 AnyaDance 发送状态、当前动作、手臂和手部状态、安全锁定、队列及调度指标。UDP 无法确认接收端连接。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BODY_AWARENESS = {
    "name": "body_awareness",
    "description": "读取 LLM 可理解的实时身体自知：当前/上一动作、切换关系、进度与剩余时间，以及双臂、双手和头部的语义姿态。连续动作、切换动作或回答当前在做什么之前应先调用。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
