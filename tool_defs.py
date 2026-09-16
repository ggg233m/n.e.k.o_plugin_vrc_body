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
    "description": (
        "设置手臂姿态，姿态保持到下一条命令。"
        "mode=polar 用抬升角和方位角指向任意方向（要 side + elevation_deg）；"
        "mode=anchor 把一只手放到 HMD、胸口或髋部锚点附近的偏移处（要 side + x_m/y_m/z_m）。"
        "两种模式下手都会随手臂方向旋转，再叠加掌心和手腕偏移。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["polar", "anchor"], "default": "polar", "description": "polar=角度指向；anchor=相对身体锚点的位置"},
            "side": {"type": "string", "enum": ["left", "right", "both"], "description": "anchor 模式只接受 left/right"},
            "elevation_deg": {"type": "number", "minimum": 0, "maximum": 180, "description": "polar 模式必填"},
            "azimuth_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0, "description": "polar：0 向前，90 向右，-90 向左，180 向后"},
            "plane": {"type": "string", "enum": ["front", "side"], "description": "旧版兼容参数；仅在未提供 azimuth_deg 时使用"},
            "reach": {"type": "number", "minimum": 0.3, "maximum": 1.0, "default": 0.9},
            "relative_to": {"type": "string", "enum": ["hmd", "chest", "hip"], "default": "chest", "description": "anchor 模式的参考锚点"},
            "x_m": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "anchor 模式必填"},
            "y_m": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "anchor 模式必填"},
            "z_m": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "anchor 模式必填"},
            "palm": {"type": "string", "enum": ["neutral", "forward", "down", "inward"], "default": "neutral"},
            "wrist_pitch_deg": {"type": "number", "minimum": -90, "maximum": 90, "default": 0},
            "wrist_yaw_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0},
            "wrist_roll_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0},
            "duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 600},
        },
        "required": ["side"],
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
    "description": (
        "停下来。用 scope 选停哪一层，默认 all（先撤自主目标再清移动轴，这是"
        "用户喊「停下/别动」时要的那个）。"
        "navigation=只取消正在跑的自主目标；axes=只把移动与转向轴归零；"
        "action=取消当前手臂动作，停在已经到达的合法姿态；"
        "freeze=最高优先级急停，撤目标、冻结姿态并锁定后续动作与转向；"
        "unfreeze=解除你自己刚才那次 freeze 并恢复 T Pose——"
        "用户在面板按下的急停和故障闩锁解不开，那种要请用户去面板复位。"
        "只清轴不撤目标的话导航器会在下一帧把她推回去，所以别单用 axes。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["all", "navigation", "axes", "action", "freeze", "unfreeze"],
                "default": "all",
                "description": "停哪一层；不确定就用 all。",
            },
            "reason": {
                "type": "string",
                "maxLength": 160,
                "default": "autonomy_stop",
                "description": "撤销自主目标时记录的原因。",
            },
        },
        "required": [],
    },
}

BODY_RESET = {
    "name": "body_reset",
    "description": (
        "面板专用：解除任意来源的急停或非致命故障并平滑恢复标准 T Pose"
        "（模型自己下的急停用 body_stop(scope=\"unfreeze\")）。输出未启用时不会自动启用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {"duration_ms": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 600}},
        "required": [],
    },
}

BODY_STATUS = {
    "name": "body_status",
    "description": (
        "读取实时状态。默认三段全给：body=当前/上一动作、切换关系、进度与剩余时间、"
        "双臂双手和头部的语义姿态、安全锁定与驱动投递确认；"
        "autonomy=自主移动授权、降级原因、当前目标和世界 revision；"
        "vision=采集器与检测器运行情况。"
        "连续动作、切换动作、回答当前在做什么，或者要解释「为什么没动」之前先调用它——"
        "没动通常是 autonomy.armed=false，这时应如实请用户去调试台启用，不要自己重试。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "include": {
                "type": "array",
                "items": {"type": "string", "enum": ["body", "autonomy", "vision"]},
                "maxItems": 3,
                "description": "只读其中几段；省略则三段全读。",
            },
        },
        "required": [],
    },
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
        "不能写进 world_state，也不能替代 world_observe 给出的 entity_id 与置信度去锁定"
        "移动目标。画面过期或采集已停止时"
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

BODY_TURN = {
    "name": "body_turn",
    "description": (
        "转身：直接旋转虚拟 HMD 的朝向，按角度给。符号与 wander 的 turn_deg 一致——"
        "正数左转、负数右转（用户说“向右转 30 度”填 degrees=-30）。"
        "不走摇杆——VR 模式下 VRChat 的右摇杆转向不可靠，照样会回 accepted=true 却不动。"
        "转身同时就是转视角：转完之后画面朝向变了，wander 的前进方向也随之改变，"
        "所以「先转向再前进」是改变行进方向的正确做法。"
        "要原地看一圈就填 ±360 并置 wait_complete=true；"
        "此时 completed=true 只证明转向调度完成，"
        "visual_inspection_complete 始终为 false——转的过程中没有逐方位看图，"
        "不得据此声称沿途没有道具、暗格或遮挡痕迹。"
        "accepted=true 只代表本机发送成功，要确认真的转了得用 vrc_vision_frame 看画面。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "degrees": {
                "type": "number",
                "minimum": -360.0,
                "maximum": 360.0,
                "description": "相对当前朝向的转角，正左负右。整圈填 360 或 -360。",
            },
            "wait_complete": {
                "type": "boolean",
                "default": False,
                "description": "等待本地转向调度结束并回报 completed；整圈扫视时用。",
            },
        },
        "required": ["degrees"],
    },
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
