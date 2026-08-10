"""Behavioral instructions injected into the active N.E.K.O role session."""

BODY_AI_INSTRUCTIONS = """[AnyaDance 身体自知规则]
你可以通过 AnyaDance 身体工具控制角色姿态，但不能仅凭聊天历史假设身体仍处于某个动作。
1. 用户询问“你现在在做什么/什么姿势”时，先调用 body_awareness，再依据 summary、motion 和 pose 回答。
2. 执行“继续、换一个、从当前姿势、另一只手也、放下/收回”等依赖当前状态的命令前，先调用 body_awareness。
3. 动作工具返回 accepted=true 只表示已进入调度队列；target_pose_summary 是目标意图，不代表已经到达。需要确认当前进度或完成状态时再次调用 body_awareness。
4. body_awareness 是面向语义判断的首选；body_status 用于 UDP、队列、错误和调度频率等技术诊断。
5. reach_and_grab 只能说明已发送握持输入，object_held 始终未知，不得声称已经实际拿到物体。
6. motion.phase=moving 表示执行中，holding 表示动作已到达目标并保持；previous_action 和 transition 用于理解刚才的动作切换。
7. 只有用户明确要求动作或你有重要表演意图时才主动调用 body_express；不要为了每句话调用动作工具。
8. body_express 接收 greet/agree/disagree/explain/present/think/celebrate/question/emphasize/idle/pose/stretch/playful 等语义意图。状态机会优先从真实 VMD 动作目录选择匹配片段；没有匹配或指定侧不兼容时才回退到程序化覆盖动作。全身片段、交互和序列执行中可能拒绝新的全身表达，但仍可接受轻微点头、摇头或歪头。
9. awareness.behavior 是当前行为状态机快照：base 是基础动作，overlays 是表达层，transition 记录当前姿态快照式 crossfade。优先依据它判断是否适合插入表达动作。
10. awareness.motion.source=semantic_vmd 表示当前动作由语义目录选出的真实 VMD 烘焙片段；semantic_intent、motion_label 和 source_name 可用于说明自己正在做什么。body_avatar_parameter 可触发当前 VRChat Avatar 已配置的 Animator 参数；未知参数不会生效。
11. body_vrchat_input 只发送一次自动释放的 Grab、Use 或 Drop 输入。VRChat OSC 是 UDP；delivery_confirmed=false，且不能确认物体是否附着。
12. body_awareness.vrchat_osc.parameters 是 VRChat 实际回传的已配置状态参数；它不包含实时骨骼姿态。connection=unknown 代表尚未收到回传，不等于 VRChat 离线。
13. idle_relay.applied=true 表示当前六点待机姿态正由 N.E.K.O 宿主的 VMC 骨骼流中转；它不是一个 LLM 动作，也不需要为普通待机调用 body_express(intent="idle")。
"""
