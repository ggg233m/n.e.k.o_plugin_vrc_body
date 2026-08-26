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
    "description": (
        "通过 VRChat OSC /input/Jump 发送一次跳跃脉冲；这是语义地址，与用户的按键绑定无关。"
        "accepted=true 只代表本机发送成功，不代表角色真的离地（当前世界可能禁跳，或人卡在低天花板下）。"
    ),
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
    "description": (
        "提交一个受安全策略约束的当前实例自主目标；必须先手动 arm。"
        "approach/approach_observe/follow/interact/socialize 只能锁定人物：优先提交同一次叠框画面中的 target_ref（T1/T2）"
        "和 frame_revision，由后端解析并锁定稳定 ID；不要复制长 target_id。"
        "多个候选目标必须先用带 overlay 的 vrc_vision_frame 让多模态模型选择。"
        "本机视觉只追踪人形，海报、屏幕、家具这类静态物体无法作为导航目标。"
        "用户要求接近这类物体时，改用 kind=\"wander\" 提交方位角闲逛：先看最新画面，"
        "估计相对方位填入 constraints.turn_deg（正数左转、负数右转；目标在画面右侧填负值，"
        "例如右前方填 -20，限 ±45°，更偏就先 body_turn 转过去），"
        "不提交 target_id/target_ref/selector。这条路只朝那个方向走一段（不会在物体前自动停下），"
        "所以不要说成走到它面前，要如实说朝那边走走看。"
        "explore 可以用 selector 描述要搜索的语义目标，并用 constraints 限制本地执行器；"
        "本地 Explorer 找到目标后只会将其保持在视野中央，不会自动接近。"
        "收到主模型闲逛路线任务时改用 vrc_wander_step，只提交 left/forward/right，不能调用 vrc_semantic_commit。"
        "该路段停止后会带新画面再次唤醒 LLM，导航器不会自行选择下一方向。"
        "应把 world_observe.decision_context.through_revision 原样写入 based_on_revision。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "minLength": 1, "maxLength": 256},
            "kind": {
                "type": "string",
                "enum": [
                    "explore", "wander", "depart",
                    "approach", "approach_observe", "follow", "interact", "socialize",
                ],
                "description": (
                    "先分清目标是人还是物。用户说“过去看看”指的是人时使用 approach_observe："
                    "后端一次完成朝向、接近、停稳和观察，不要拆成多个 approach/观察调用。"
                    "目标是海报、屏幕、家具这类静态物体时绝不能用 approach/approach_observe——"
                    "本机视觉只追踪人形，锁不住它们，提交了也只会走到一半报 target_lost；"
                    "这种情况一律用 wander 带 turn_deg 一段一段走过去。"
                    "用户要求离开当前观察点时使用 depart；用户说随便走走、逛逛，"
                    "或用“走吧”确认刚提出的闲逛时，先看最新画面，再用 wander 提交一条带 turn_deg 的短路段。"
                ),
            },
            "target_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 96,
                "description": "兼容内部调用的稳定实体 ID；主 LLM不要复制它，优先使用 target_ref。",
            },
            "target_ref": {
                "type": "string",
                "pattern": "^T[1-9][0-9]*$",
                "maxLength": 8,
                "description": "从同次 vrc_vision_frame(overlay=true) 画面中选出的短编号，例如 T2。",
            },
            "frame_revision": {
                "type": "integer",
                "minimum": 0,
                "description": "产生 target_ref 的同次叠框画面 revision；必须和 target_ref 一起原样提交。",
            },
            "selector": {
                "type": "object",
                "description": "仅供 explore 搜索使用的语义选择器；它不是实体 id。",
                "properties": {
                    "semantic_type": {
                        "type": "string",
                        "enum": ["npc", "player", "avatar", "person", "humanoid", "object"],
                    },
                    "label": {"type": "string", "minLength": 1, "maxLength": 64},
                    "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "additionalProperties": False,
            },
            "constraints": {
                "type": "object",
                "description": "本地执行器必须兑现的有界搜索或有限行为约束。",
                "properties": {
                    "max_duration_s": {"type": "number", "minimum": 1.0, "maximum": 600.0},
                    "max_scan_turns": {"type": "integer", "minimum": 1, "maximum": 32},
                    "max_forward_axis": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                    "settle_seconds": {"type": "number", "minimum": 0.2, "maximum": 3.0},
                    "observe_seconds": {"type": "number", "minimum": 0.5, "maximum": 10.0},
                    "turn_deg": {
                        "type": "number",
                        "minimum": -45.0,
                        "maximum": 45.0,
                        "description": (
                            "wander 必填：相对当前朝向的转角。正数左转、负数右转、0 直行——"
                            "目标在画面右侧就填负值（例：右前方填 -20），在左侧填正值。"
                        ),
                    },
                    "direction_scores": {
                        "type": "object",
                        "description": (
                            "可选的方向偏好，不是通行概率；键可用 left/forward/right "
                            "或角度字符串，分数范围 0~1。后端只记录并回报，不替 LLM 选路。"
                        ),
                        "maxProperties": 16,
                        "additionalProperties": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                },
                "additionalProperties": False,
            },
            "based_on_revision": {
                "type": "integer",
                "minimum": 0,
                "description": "生成此目标时看到的世界 revision；迟到决策会在执行前重新校验。",
            },
        },
        "required": ["goal"],
    },
}

VRC_WANDER_STEP = {
    "name": "vrc_wander_step",
    "description": (
        "完成当前 `[VRChat 主模型闲逛路线任务]`：根据该任务附带的最新画面选择"
        " left、forward 或 right，并启动一条最多三秒的短路段。工具只绑定插件刚注入"
        "当前会话的待决路线请求，不接受 target_id、target_ref 或人物选择器，因此不会"
        "把方向误绑定为某个角色。没有待决任务、任务已替换/过期或请求不匹配时会拒绝，"
        "不能用它接近或跟随人物。只有 accepted=true 才代表路段已经开始。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "forward", "right"],
                "description": "主 LLM 从任务配对画面选择的短路线：左前、正前或右前。",
            },
        },
        "required": ["direction"],
        "additionalProperties": False,
    },
}

VRC_WANDER_ROUTE = {
    "name": "vrc_wander_route",
    "description": (
        "在方向尚未决定时开始一次闲逛：把最新配对画面作为"
        " `[VRChat 主模型闲逛路线任务]` 交回给你选路。"
        "用户说“随便走走/去逛逛/往前走一小段”而你还没有看图决定方向时用它。"
        "本次调用不会移动，返回 accepted=false 且 reason_code=wander_direction_pending"
        "才是成功建立路线任务，此时不得声称已经出发；紧接着必须用 vrc_wander_step"
        "选择 left/forward/right 才真正开始移动。"
        "若你已经看过画面并确定了相对角度，可以直接用 vrc_autonomy_goal 提交 wander"
        "并填 constraints.turn_deg，不需要本工具。"
        "两条路都走本地闭环：会读取速度回传、检测撞墙、有限绕行并记录死路方向；"
        "绝不能改用 body_locomotion 这类开环遥控代替。"
        "需要用户先在插件面板手动启用自主控制。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "maxLength": 256,
                "description": "用户的原始意图描述，例如“往前走一小段”。",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}

VRC_AUTONOMY_STOP = {
    "name": "vrc_autonomy_stop",
    "description": "停止自主目标并释放 AnyaDance 控制器输入。",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VRC_SEMANTIC_COMMIT = {
    "name": "vrc_semantic_commit",
    "description": (
        "提交当前主多模态 LLM 对被动 VRChat 语义任务的结构化分类。"
        "必须原样使用任务消息中的 request_id 和 frame_revision，并且每个任务只调用一次；"
        "这是和当前用户聊天同一回合内的附带工作，不要为它另起回答。"
        "已有检测框优先复制 candidates 中的完整 target_id；只有检测器漏框时才填写归一化 bbox。"
        "海报、屏幕和镜像要如实标为 poster/screen/mirror，不要为了让导航通过就谎报成人物。"
        "这类静态物体默认被自主搜索当作干扰物过滤，但用户点名要去看它时是可以导航的："
        "在 vrc_autonomy_goal 的 selector.semantic_type 里填同一个类别即可放行。"
        "任务 reason=agent_navigation_target_unresolved 时，只提交用户所指的唯一真实目标；"
        "提交后端会自动续接一次有限导航，不要再发第二次移动命令。"
        "只有返回 pending_navigation.accepted=true 才代表移动已经开始。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "frame_revision": {"type": "integer", "minimum": 0},
            "entities": {
                "type": "array",
                "maxItems": 32,
                "description": "画面中的语义分类；确认没有可分类候选时提交空数组。",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_id": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 96,
                            "description": "本次任务 candidates 中的完整稳定 ID。",
                        },
                        "bbox": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "description": "仅漏检目标使用：[left, top, right, bottom] 归一化坐标。",
                        },
                        "semantic_type": {
                            "type": "string",
                            "enum": [
                                "npc", "player", "avatar", "person", "humanoid", "object",
                                "poster", "screen", "mirror", "unknown"
                            ],
                        },
                        "label": {"type": "string", "minLength": 1, "maxLength": 64},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["semantic_type", "label", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["request_id", "frame_revision", "entities"],
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
    "description": (
        "读取最近的视觉世界状态，以及主 LLM 尚未确认消费的内存 revision 决策上下文。"
        "decision_context 会把重复位置更新压成每个实体的 first/latest 轨迹，并保留离散事件；"
        "生成自主目标时把 through_revision 写入 based_on_revision，只有目标被接受后才会确认消费。"
        "结果来自可选的 VRChat 画面检测器/VLM；没有新观测时必须按 unknown 处理，不能把空结果当成世界为空。"
        "其中 traversability_prediction 是独立的连续帧光流几何预测，不是实体或地图；"
        "只有 predicted_blocked 才可能触发当前路线的安全停车，unknown 不能当作畅通。"
        "traversability_prediction.ground_extent 是单帧地面可见范围，advisory_only；"
        "它只给方向之间的相对开阔度排序，不触发停车，extent_ratio 是画面跨度比例而非米制。"
    ),
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
        "返回 available=false，此时按看不见处理，不要沿用上一次看到的内容。overlay=true"
        "时图中 T1/T2 与结果 overlay.candidates 一一对应；选择后把短 target_ref 和同次"
        "frame_revision 交给 vrc_autonomy_goal，稳定 ID 由后端解析。"
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
            },
            "overlay": {
                "type": "boolean",
                "default": False,
                "description": (
                    "叠加检测框，用于对照「检测器看到的」与「画面里实际有的」——"
                    "例如确认某个高分实体圈的是真人还是墙上的立绘。框与 JPEG 来自同一次"
                    "本地检测，并以单槽内存对象配对；overlay.paired=true 才能使用。没有"
                    "同帧配对时返回 drawn=false，不会拿旧图叠最新世界。叠了框也不改变"
                    "性质：画面结论仍然只是低置信视觉猜测，不能写进 world_state。成功时"
                    "结果返回不含长稳定 ID 的 overlay.candidates 和 frame_revision。"
                ),
            },
        },
        "required": [],
    },
}

BODY_LOCOMOTION = {
    "name": "body_locomotion",
    "description": (
        "开环遥控摇杆：直接通过 VRChat OSC 推移动轴，保持到超时或下一次调用。"
        "前后左右对应游戏摇杆输入：forward=1.0, backward=-1.0, left=-1.0, right=1.0；"
        "可同时设置斜向移动。移动方向相对角色当前朝向，不是世界方向。"
        "它不读速度回传、不检测撞墙、不会绕行、也不记录死路方向，accepted=true 只代表"
        "本机发送成功，不能证明角色真的移动了。因此只用于用户明确要求的一次性微调"
        "（挪一点、退半步、对齐位置）。"
        "凡是“走走/逛逛/往前走/过去/离开”这类导航意图一律改用闭环："
        "已看图定好方向用 vrc_autonomy_goal(kind=\"wander\") 并填 constraints.turn_deg，"
        "尚未定方向用 vrc_wander_route 取回路线任务后再 vrc_wander_step。"
    ),
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
    "description": (
        "转身：直接旋转虚拟 HMD 的朝向。符号与 wander 的 turn_deg 一致——"
        "正值向左转，负值向右转（例：用户说“向右转”填 horizontal=-0.5）。"
        "不走摇杆——VR 模式下 VRChat 的右摇杆转向不可靠，照样会回 accepted=true 却不动。"
        "转身同时就是转视角：转完之后画面朝向变了，body_locomotion 的前后左右也随之改变，"
        "所以「先转向再前进」是改变行进方向的正确做法。"
        "accepted=true 只代表本机发送成功，要确认真的转了得用 vrc_vision_frame 看画面。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "horizontal": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "转身速度：1.0=最快左转，-1.0=最快右转（正左负右，与 turn_deg 同）"},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 10000, "default": 500, "description": "持续时间；超时后自动归零"},
        },
        "required": ["horizontal"],
    },
}

VRC_SCAN_SURROUNDINGS = {
    "name": "vrc_scan_surroundings",
    "description": (
        "按用户明确要求让 VRChat 视角原地完整转一圈，并等待本地转向调度结束。"
        "completed=true 只证明一整圈转向在本地调度器中完成，不证明已经看清沿途物体；"
        "visual_inspection_complete=false 时不得声称没有任务道具、暗格或遮挡痕迹。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right"],
                "default": "right",
                "description": "转圈方向。",
            },
        },
        "required": [],
        "additionalProperties": False,
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
