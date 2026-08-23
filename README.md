## 此项目正在开发中功能尚未完善

- [开发进度](ROADMAP.md)
- [独立后端](backend/README.md)


# N.E.K.O VRC Body

这是一个独立的 N.E.K.O 插件：待机时把宿主公开的 VMC 骨骼流转换为 AnyaDance 六点姿态，显式动作时由 LLM 通过受控语义工具接管；VRChat 移动和 Index 控制器输入优先走 AnyaDance 虚拟设备，Avatar 参数、聊天和兼容回退继续走 VRChat OSC。

## 姿态能力

- 用 `0–180°` 抬升角和 `-180–180°` 方位角控制左手、右手或双手；0° 方位向前，90° 向右，-90° 向左。
- 手的基础旋转会自动跟随肩到手的方向，再用 pitch/yaw/roll 和掌心预设叠加局部偏移。
- 把手移动到相对 HMD、胸口或髋部的局部 XYZ 目标。
- 开掌、握拳、抓握和指向。
- 向腰部、胸部或头部高度伸手，并在轨迹末段发送 grip。
- 挥手、点头、鞠躬、急停、复位和显式启停输出。
- 用 `body_express` 提交“问候、同意、解释、思考、庆祝”等语义意图，由分层状态机选择自然表达动作。
- 用 `body_sequence` 组合最多 16 步动作，支持等待、1–4 次循环与 action ID 取消。
- 枚举并播放白名单 `motions/` 目录中的 AnyaDance `.nya` 预制动作。
- 用 `body_awareness` 读取 LLM 可理解的当前/上一动作、切换关系、完成状态、剩余时间和实际语义姿态。
- 用 `body_avatar_parameter` 触发当前 Avatar 已配置的 Bool、Int 或 Float Animator 参数。
- 用 `body_vrchat_input` 安全脉冲 Grab、Use 或 Drop，并自动发送释放值。
- 用 `body_locomotion`、`body_turn` 发送有时限的 VRChat 移动/转身轴值，或用 `body_stop_movement` 立即归零。
- 用 `body_chatbox` 发送附近玩家可见的 VRChat 聊天框文本（最多 144 字符）。
- 用 `vrc_controller_input`、`vrc_menu_navigate` 和 `vrc_jump` 发送有限时长的虚拟 Index 摇杆、按钮和跳跃脉冲。
- 用 `vrc_autonomy_status`、`vrc_autonomy_goal`、`vrc_autonomy_stop` 管理当前实例内的自主目标；授权必须从调试面板或 `/autonomy/arm` 手动启用。
- 监听 VRChat 的 Avatar 切换和参数回传，把白名单动作状态加入 `body_status` 与 `body_awareness`。
- 在 `idle` 状态监听 N.E.K.O VMC 2.0 OSC，完成 Humanoid FK 后中转头、双手、髋和双脚六点姿态。
- 单一发送线程以配置的 120 Hz（默认）向 `127.0.0.1:39570` 发送完整 UDP 帧，控制器叠加与六点姿态共用同一帧。

“拿东西”只表示手移动到语义目标并发送握持输入。VRChat OSC 不提供通用物体位置或 Pickup 附着确认，因此插件始终把 `object_held` 报告为 `unknown`。

## 安全与运行约束

1. 插件默认不自动启动；启动后仍处于 `disabled`，必须显式调用 `body_enable`。
2. `body_stop` 会冻结当前合法姿态、释放所有控制器输入，并锁定后续动作；调用 `body_reset` 才能恢复。
3. UDP 协议本身没有响应或发送者身份。启用并收到 AnyaDance 驱动遥测时，`body_status.driver_log` 和 `body_awareness.driver_delivery` 可以确认驱动实际处理了命令；遥测不可用时只能确认本地发送成功。
4. 插件假设接管期间 AnyaDance UI 不再向 39570 发送。驱动遥测发现其他活跃来源时会报告冲突、解除自主授权并释放输入；否则为 `unsupported` 或 `detected_unattributed`。
5. AnyaDance 虚拟驱动可能影响真实 SteamVR 设备追踪。实机测试应从私人 VRChat 实例、小幅度和低速度动作开始。
6. OSC 同样使用 UDP。`delivery_confirmed=false` 只表示本机完成发送；只有收到 9001 回传时 `connection` 才显示 `detected`，没有回传时为 `unknown`，不能据此断言 VRChat 离线。
7. 默认只有一个程序能独占监听 `127.0.0.1:9001`。若已有 OSC 路由器占用该端口，应修改 `listen_port`，并让路由器或 VRChat 向新端口转发。
8. VMC 待机中转默认独占监听 `127.0.0.1:39539`。N.E.K.O 的 VMC 输出必须启用并指向该端口；若端口已被其他 VMC 接收器占用，应修改双方端口或使用 OSC 路由器分流。

## 身体自知

插件启动时会向 LLM 注入身体工具使用规则：聊天历史不是可靠的实时姿态来源；回答当前动作，以及执行“继续、换一个、另一只手也”等相对命令前，应先调用 `body_awareness`。

`body_awareness` 提供：

- `motion`：当前动作来源、阶段、进度、已用/剩余时间及是否已经到达目标。
- `previous_action` 和 `transition`：上一动作的结束原因，以及刚从什么动作切换到什么动作。
- `pose`：由当前输出帧反推的双臂抬升角、方向、伸展、双手姿态/握持和头部旋转。
- `summary`：可直接用于自然语言判断的中文摘要。

每个成功受理的动作还会返回 `target_pose_summary`。其中 `completion_confirmed=false`，明确表示这只是目标意图；动作实际是否完成仍以 `body_awareness.motion` 为准。

## 分层动作状态机

调度器把动作分为安全层、基础动作层、手部层和表达层，并在 `body_status.behavior` 与 `body_awareness.behavior` 中公开当前模式、活动层、优先级、上一基础动作、切换策略和最近决策。

优先关系为：急停/复位/禁用 > 抓取交互 > 序列 > 精确姿态 > 手动预制片段 > 显式手势 > 语义 VMD/程序化表达 > VMC 待机基础层。精确工具属于显式控制，仍可替换当前动作；`body_express` 会先查询 `motions/catalog.json`，命中时播放低优先级真实 VMD，未命中时才使用程序化覆盖层。舞蹈、序列、抓取或完整手势移动期间，状态机会拒绝新的全身表达，但允许点头、摇头和歪头等头部表达。

插件复用了 N.E.K.O 3D 动画系统的三项策略：从当前实际输出姿态开始切换、默认约 0.4 秒 crossfade、四元数始终走最短旋转路径。Python 插件不能直接导入浏览器内的 `THREE.AnimationMixer`、VRM Humanoid 或 MMD IK/Grant Solver；这里用六点姿态快照与相对覆盖层实现等价的调度语义，不声称具备完整骨架混合。

语义表达示例：

```text
body_express(intent="greet")
body_express(intent="explain", side="auto", intensity=0.45)
body_express(intent="agree")
body_express(intent="celebrate", side="both", intensity=0.7)
```

支持 `greet | agree | disagree | explain | present | think | celebrate | question | emphasize | beckon | comfort | apologize | surprise | shrug | clap | laugh | sigh | idle | pose | stretch | playful`。省略侧别、强度或时长时使用动作目录的自然默认值；同一意图有多个 VMD 时会轮换并参考强度选择。

`body_awareness.vrchat_osc` 只报告 VRChat 实际发回的 Avatar ID 和配置参数，不提供实时骨骼角度，也不能确认 Pickup 是否附着。默认关注 `NEKO_Action`、`NEKO_ActionActive`、`NEKO_ActionPhase` 和 `NEKO_Holding`；这些参数需要先在 Avatar 的 Expression Parameters/Animator 中创建并驱动。

白名单还包含六个 VRChat 内置 Avatar 参数（`VelocityX/Y/Z`、`AngularY`、`Upright`、`Grounded`），`body_awareness.vrchat_osc.motion` 由它们算出实测移动反馈——这是全仓库唯一能说明“我是不是真的动了”的回传，所有工具的 `accepted=true` 都只代表本机 UDP 发送成功。

> ✅ **参数名已实机验证（2026-08-23）。** 实测 `VelocityX/Y/Z`、`AngularY`、`Grounded` 均会回传，且 `VelocityX/Z` 是 **avatar 本地坐标系**——转 90° 后仍是 Z 主导，因此 `velocity_z` 直接就是前进分量，不需要先按 HMD yaw 旋转。该 Avatar 实测跑满速度为 `2.6667 m/s`。`VelocityX/Z` 只有角色移动时才回传，因此每条速度记录只是短时样本；静止沉默是正常现象，旧的 0 或移动速度超过时限后都会变成 `motion.available=false` / `velocity_feedback_quiet`，不会被伪装成当前零速度。导航器只接受本次前进命令之后的新样本，并在起步阶段保留 450 ms 宽限。

`motion` 除标量速度外还导出 `velocity_x` / `velocity_z` 原始轴值，以及按水平模长归一的
`forward_ratio` / `slip_ratio`。这两个比值用于区分「畅通」与「撞墙」：VRChat 的角色控制器
会把移动向量投影到墙面上，所以斜撞墙时前进分量塌陷、侧滑分量抬起，而速度模长未必变小——
`hypot` 之后的标量看不出这个差别。`forward_ratio ≈ 1` 是畅通；明显小于 1 且 `slip_ratio`
变大表示正贴着墙滑行，滑行方向即可通行方向；`horizontal_speed_mps` 塌到接近 0 则是正面墙
或墙角。水平速度低于静止阈值时两个比值为 `None` 而不是 `0.0`——`0.0` 会被读成「正对着墙」，
与「站着不动」是完全不同的结论。

## 独立后端进程

身体运行时采用一个粗粒度的本机后端进程，而不是把每个动作拆成独立服务。Hosted 插件只保留 LLM 工具参数校验、UI 和 IPC 适配；后端进程统一持有 `BodyScheduler`、AnyaDance UDP 输出、VMC idle 中转、VRChat OSC、驱动遥测、动作加载和世界状态。两者通过带随机令牌的 loopback HTTP 通信，后端不可用时插件进入安全的 `backend_unavailable` 状态。

## 视觉世界状态（第一阶段）

插件新增 `world_observe` 工具和 revision 增量世界桥；后端目录内的 `backend/world_state.py` / `backend/vision.py` 提供状态层、DXcam/MSS 桌面镜像采集、可插拔 OpenVINO/VLM worker。它们不进入 AnyaDance 的 120 Hz 调度线程，也不替代宿主 VMC 待机中转。模型包和 OpenAI-compatible VLM 由部署环境提供，缺少依赖时明确降级为 unavailable；启用采集但没有模型时只运行 capture-only 诊断，不发布猜测实体。视觉后端可以发布带 `confidence`、`source`、`age_ms`、`ttl_ms` 和 `unknown` 不确定性的目标与事件；LLM 读取不到新观测时不得把空结果当成“场景为空”。

当前版本默认不加载第三方模型，但已经提供 DXcam/MSS、OpenVINO 和 OpenAI-compatible VLM 的可选适配器；未配置依赖时 `world_observe` 会明确返回 `available=false`，不会伪造世界状态。后端的可移植边界、启动方式和适配说明见 `backend/README.md`。

桌面镜像采集会自动探测 DXGI 的 GPU/显示输出，并在失败时逐个尝试 MSS 物理显示器；
`/perception` 会保留每个候选输出的错误，便于区分权限、显卡和 BitBlt 问题。可在
`[vision]` 中设置 `monitor_index`、`dxcam_device_idx`、`dxcam_output_idx`，或把
`dxcam_backend` 固定为 `dxgi`/`winrt` 进行排查。

要启用接近 OBS 窗口捕获的 Windows Graphics Capture 路径，可在后端使用的 Python
环境中安装可选组件：

```powershell
python -m pip install --user "dxcam[winrt]"
```

安装后保持 `dxcam_backend = "auto"`；DXGI 被拒绝时会自动切换到 WinRT。该依赖仍是
可选的，未安装时插件会继续使用 DXGI/MSS，并在 `/perception` 报告缺失原因。

`[vision]` 的 `window_title` 非空时（随插件部署的配置为 `"VRChat"`），采集区域取该
窗口的屏幕矩形而不是整块显示器。窗口矩形不是一次性的：窗口被拖动、改分辨率或全屏
切换后，启动时解析的坐标就会一直抓错位置，而 DXcam/MSS 的区域在构造时固定、没有
改区域的接口。因此后端按 `window_track_interval_ms`（默认 5000 ms）重新解析，只在
矩形真的变化时重建采集源；窗口暂时找不到时（最小化、Alt-Tab）保留上一次的矩形，
不回退全屏——把整个桌面喂给检测器比暂时抓一块过期区域危险得多。设为 0 可恢复
"只在启动时解析一次"的旧行为。

本地检测器还会丢弃过小的框，宽高分别设阈值（`min_box_width_ratio` 默认 0.8%、
`min_box_height_ratio` 默认 2%，`min_box_ratio` 为两轴共同回退值）。实测真实帧里出现
过 27×27 px 的高分假阳性，它们会被跟踪器当成真人并污染导航器的距离闭环。宽阈值
更松是因为站立的人本来就高而窄，共用一个阈值时总是宽度先卡，会把房间对面的人一并
裁掉；两条边仍然都要过关，墙缝和 UI 边框这类细长误检照样被挡住。

检测器只回答"有几个人、在哪个方位"。要确认对方是谁、菜单开着没、界面上写了什么，
`vrc_vision_frame` 会把最近一帧降采样后的 JPEG 注入当前回合；被判定为社交相关的主动
唤醒推送也会附带一张。图不能走工具返回值——那只是一个 JSON 值，模型看不到里面的
base64——所以两条路都通过 `push_message` 的 image part 交给宿主注入。

画面只用于理解，**不进入 `world_state`**：从像素得出的一切都是低置信视觉猜测，不能
当作实体、事件或位置的来源，也不能用来满足 `body_reach_and_grab` 的 `preconditions`；
那条路仍然只认 `world_observe` 给出的 `entity_id` 与置信度。采集停止、还没有帧或画面
超龄时明确返回 `available=false`，不会退而求其次给一张旧图。

拉图有每分钟上限（`frame_max_per_minute` 默认 10，滑动窗口）。一张 960 px 的 JPEG 进
上下文是十万字符量级的 base64，agent 在循环里每回合拉一张足以把会话挤爆。唤醒配的图
不占这个预算，它自己受 12 s 的最小唤醒间隔约束。

视觉捕获也可以独立于身体控制链路启停：调用后端 `POST /vision/stop` 会释放当前
DXcam/WinRT/MSS 句柄并解除自主导航，`POST /vision/start` 会创建全新的
`FrameSource` 后恢复采集；不会复用已经关闭的捕获对象。插件客户端提供对应的
`client.vision.stop()` / `client.vision.start()` 方法。配置为外部 source 时，停止后
需要重新注入外部采集器，不能直接重用旧对象。

自主目标不会直接把主 LLM 接入身体控制线程。后端的 `LocalNavigator` 会在会话手动
授权后，以约 10 Hz 根据新鲜视觉实体生成受限的短摇杆脉冲；目标不可见、方位或距离
未知、观测过期或世界带不确定性时立即释放输入。主 LLM 只负责目标和行为选择，
AnyaDance 调度器仍以 120 Hz 发送最新姿态与控制器状态。定向目标必须携带最新世界
快照中的精确 `target_id`。方位角和接近度在本地做轻量时序平滑，单帧低置信或漏检最多
宽限 300 ms；世界不确定、整体观测过期或宽限到期仍会硬停车。本地 person 检测会把易变的 `openvino:track:N` 映射为当前
后端会话内稳定的 `avatar:session:<token>:<N>`，目标短暂离开视野并被分配新轨迹后，
可通过检测框外观重新绑定；原轨迹 ID 保留在实体的 `attributes.track_entity_id` 中
用于诊断。锁定目标消失时不会回退选择同标签的海报、镜像或其他玩家。每个新视觉 revision
都把比例转向修正换算为“当前实际 yaw + 修正”的绝对目标，可以在上一段平滑转身仍执行时
连续重定向，不再等待收尾，也不会把相对角度叠加成超调。

这个稳定 ID 不是 VRChat 的 `usr_`/`avtr_`，后端重启后会变化，也无法可靠区分使用
完全相同 Avatar 的玩家或镜中像；有歧义时系统宁可分配新 ID，也不把两个可见目标合并。

世界日志若以后启用，只能作为低置信度辅助来源。玩家实体应使用
`vrchat:player:<user_id>` 这样的稳定 ID；收到 `player_left` 时，适配器必须在同一
批次提交 `remove_entity_ids` 和离开事件，并用 `remove_source` 限制删除范围。这样
玩家退出后不会残留幽灵实体，也不会误删视觉检测器发布的同名对象；后端还会用接收
水位拒绝保留窗口内迟到的旧帧，避免删除后幽灵复活。换世界时只清理日志来源的玩家实体；
若适配器会积压超长离线队列，应提高视觉水位上限。
当前仓库不接入世界日志解析器或 Contact 总线，也不安装动作生成模型；自主运行时只
管理手动授权、世界新鲜度和安全释放，不自动执行好友、邀请或世界切换。模型通过独立
detector 或 sidecar 接口接入，并由配置显式启用，不把未启用功能的状态塞进插件主流程。

`world_memory.persist_world` 只保存可跨进程复用的世界事实。逐帧检测轨迹、
`avatar:session:*`、synthetic 实体以及显式标记 `attributes.memory_scope="observation"`
的外部观测不会写盘；旧版本已经保存的此类实体会在下次启动时自动清除。持久化负载
使用稳定格式，内容没有变化时不会随 10 Hz 感知循环重复写文件。

## N.E.K.O VMC 待机中转

插件启动后会通过 N.E.K.O 公开的 VMC REST 控制面自动启用输出，并将目标设为 `127.0.0.1:39539`；随后请求一次短暂 T Pose，以 VRM 原始静止姿势校准手腕朝向与手指弯曲零点。插件关闭时恢复宿主原来的启用状态、目标和频率。插件接收 `/VMC/Ext/Root/Pos` 与 `/VMC/Ext/Bone/Pos`，把局部 Humanoid 骨骼变换经过 FK、坐标系还原和身高标定后转换为 AnyaDance 六设备帧：

```toml
[vmc_idle]
enabled = true
listen_host = "127.0.0.1"
listen_port = 39539
allowed_sender = "127.0.0.1"
stale_after_ms = 500
manage_host_output = true
host_api_url = "http://127.0.0.1:48911"
host_api_timeout_seconds = 3.0
host_output_host = "127.0.0.1"
host_send_rate_hz = 60
```

只有调度器处于 `idle` 且 VMC 帧未过期时才应用中转。精确动作、VMD、序列、保持姿态和急停不会被 VMC 覆盖；动作结束回到 `idle` 后自动恢复最新宿主姿态。`body_status.idle_relay` 和 `body_awareness.idle_relay` 会报告监听状态、帧龄和当前是否正在应用。

## VRChat OSC

在 VRChat Action Menu 中启用 `Options > OSC > Enabled`。插件默认向 `127.0.0.1:9000` 发送，并在 `127.0.0.1:9001` 监听回传：

```toml
[vrchat_osc]
enabled = true
send_host = "127.0.0.1"
send_port = 9000
listen_host = "127.0.0.1"
listen_port = 9001
allowed_sender = "127.0.0.1"
input_pulse_ms = 100
parameter_cache_size = 256
# 前四个是本插件自己驱动的动作状态参数；后六个是 VRChat 内置 Avatar 参数，
# 用于确认角色是否真的在移动（已实机验证会回传，收不到时报 available=false）。
awareness_parameters = [
  "NEKO_Action", "NEKO_ActionActive", "NEKO_ActionPhase", "NEKO_Holding",
  "VelocityX", "VelocityY", "VelocityZ", "AngularY", "Upright", "Grounded",
]
```

参数工具示例：

```text
body_avatar_parameter(name="NEKO_Action", value=2)
body_avatar_parameter(name="NEKO_ActionActive", value=true)
body_vrchat_input(action="grab", side="right", hold_ms=100)
body_locomotion(vertical=1.0, horizontal=0.0, duration_ms=1000)
body_turn(horizontal=-0.5, duration_ms=500)
body_stop_movement()
body_chatbox(text="你好", immediate=true)
```

传入不存在的 Avatar 参数时，OSC 数据报仍可能成功发送，但 VRChat 不会产生对应动作。VRChat 的移动轴必须使用 -1..1 的浮点数并在结束时归零；本插件对每次移动/转身设置 100–10000 ms 的自动归零超时，`body_stop_movement` 会同时归零三条轴。`accepted=true` 只代表本机 UDP 发送成功，不代表 VRChat 已移动或转身。`body_reach_and_grab` 会在动作最后 15% 自动安排同侧 `/input/GrabLeft` 或 `/input/GrabRight` 脉冲；动作被替换时守卫会阻止尚未发生的按下，急停、取消、复位、禁用和插件关闭都会清空定时输入并发送释放值。聊天框不是私密通道，附近玩家可能看到。

## 调试 UI

插件提供 N.E.K.O 原生 Hosted TSX 调试面板。在插件管理器中选择 `N.E.K.O AnyaDance Body`，点击“打开面板”即可使用。面板不启动额外 Web 服务，也不加载第三方前端依赖，只拥有 `state:read` 和 `action:call` 权限。

面板包括：

- AnyaDance 120 Hz 发送频率、丢帧、控制器释放延迟、数据包和当前动作进度。
- VRChat OSC 9000/9001 状态、Avatar ID、收发计数、错误和参数回传。
- 启用、平滑禁用、复位与急停。
- 手臂角度/方位/伸展/掌心调节，手型、程序化手势和伸手抓取测试。
- 语义表达意图、状态机模式、活动层和优先级保护调试。
- `.nya` 动作选择、速度、循环、切换时间与结束恢复。
- Avatar Parameter 以及 Grab/Use/Drop 输入测试。
- 每秒自动刷新、身体自知/OSC JSON 快照和最近调试命令日志。

调试面板调用的是与 LLM 工具相同的校验和调度实现，因此仍受启用状态、范围、安全锁定和动作时长限制。急停后必须点击“复位 T Pose”解除锁定。

面板每秒刷新时只扫描动作文件名、大小和修改时间，不读取或解析 `.nya` 正文。新动作在首次播放或显式调用 `body_list_clips` 时由后台线程完成完整校验；首次解析期间命令按钮会显示执行中，但状态刷新和其他插件事件不会被同步文件读取占住。解析结果按文件签名缓存，文件内容变更后会自动失效。当前最多保留两个已解析动作，便于重复播放和动作切换。

## 安装与验证

插件没有第三方 Python 运行时依赖。使用 N.E.K.O 仓库自带 CLI 对此目录执行检查、构建和安装：

```powershell
cd H:\AI\neko-music\N.E.K.O
python -m plugin.neko_plugin_cli.cli check H:\AI\neko-music\vrc\neko_anyadance_body
python -m plugin.neko_plugin_cli.cli build H:\AI\neko-music\vrc\neko_anyadance_body
```

若系统 Python 缺少 N.E.K.O 宿主依赖，请使用该仓库配置的 `uv run python` 执行同样命令。

## 建议调用

```text
body_enable()
body_arm_pose(side="right", elevation_deg=130, azimuth_deg=-25, reach=0.9, wrist_roll_deg=45)
body_move_hand(side="right", relative_to="chest", x_m=0.30, y_m=0.10, z_m=-0.45, palm="down")
body_hand(side="right", pose="grip", strength=1.0)
body_avatar_parameter(name="NEKO_Action", value=1)
body_vrchat_input(action="use", side="right")
body_awareness()
body_stop()
body_reset()
body_disable()
```

角度是插件定义的语义手臂角度，不是真实肩关节测量值。肩膀位置由 HMD 和身体配置估算，肩肘最终由 VRChat IK 求解。

## 预制动作

仓库不再分发预制 `.nya` 动作和 `motions/catalog.json`。这些文件可能包含第三方 VMD/PMX 素材的派生数据，是否可以再分发取决于各自授权；因此安装包默认只包含动作加载器，首次安装后动作目录可能为空。

如果你拥有已授权的 AnyaDance `.nya` 文件，把它们放入插件配置目录下的 `motions/` 目录即可。`catalog.json` 是可选的：存在时，`body_express` 会按意图、侧别和强度选择目录中的真实动作；不存在或没有匹配项时，会回退到程序化表达动作。

动作目录的文件名就是调用时使用的逻辑名称（不含 `.nya` 扩展名），支持中文名称，但不接受绝对路径、子目录或 Windows 非法文件名字符。

调用示例：

```text
body_list_clips()
body_play_clip(clip_name="my_greeting", speed=1.0, loop_count=1, restore_after=true)
body_play_clip(clip_name="my_idle", speed=1.0, loop_count=1, restore_after=false)
```

播放默认把片段第一帧 HMD 的 X/Z 对齐到当前姿态；`body_stop` 和 `body_cancel` 可随时中断。

`vmd_bake.py` 提供离线烘焙命令。它让 Blender/MMD Tools 完成真实骨架的 FK、IK 和约束求值，再复用 AnyaDance 的六设备重定向算法生成 `.nya`，不会在插件实时调度线程中运行：

```text
python vmd_bake.py --vmd <动作.vmd> --model <模型.pmx> --blender <blender.exe> --export-script <blender_export_mmd.py> --mmd-tools <mmd_tools目录> --target-height 1.50 --body-width-scale 1.0 --leg-length-scale 1.0 --hand-reach-scale 1.22 --hip-height-offset 0.0 --output motions/<名称>.nya
```

比例参数用于离线适配目标 Avatar：`target-height` 控制总体身高，`body-width-scale` 调整手脚相对身体中心的横向宽度，`leg-length-scale` 改变上半身相对脚底的高度，`hand-reach-scale` 控制手掌目标相对肩部的伸展，`hip-height-offset` 做最后的髋部高度微调。默认值等价于 AnyaDance 原始重定向。

VMD 和 PMX/PMD 都是独立的第三方作品；导入前应确认各自授权。插件的烘焙工具不赋予素材的复制或再分发权。

解析器限制单文件 64 MiB、18000 帧和 300 秒，并检查时间线、六设备完整性、有限数值、四元数、坐标及手指范围。设备 Y 略高于安全上限时会与 AnyaDance 一致钳制到上限；循环展开后的播放时间同样不得超过 300 秒。

大型动作在面板中最初会标为“未索引”。首次播放需要真实的 JSON 解码和逐帧安全校验时间；完成后会显示帧数/时长并进入内存缓存，之后再次播放无需重复解析。目录自动刷新不会触发这项重活。

动作序列示例：

```json
{
  "steps": [
    {"type": "arm_pose", "side": "right", "elevation_deg": 120, "azimuth_deg": 0, "duration_ms": 500},
    {"type": "move_hand", "side": "right", "relative_to": "chest", "x_m": 0.3, "y_m": 0.0, "z_m": -0.45, "duration_ms": 400},
    {"type": "hand", "side": "right", "pose": "grip", "strength": 1.0, "duration_ms": 250},
    {"type": "wait", "duration_ms": 800}
  ],
  "loop_count": 1
}
```
