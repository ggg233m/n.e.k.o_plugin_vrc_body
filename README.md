## 此项目正在开发中功能尚未完善

- [N.E.K.O 宿主一键安装包说明](INSTALL.md)：导入 `.neko-plugin` 即可安装模型和运行依赖，不需要手动解压或运行 pip。
- 🔴 **[读文档前先读这个：Docs 索引与权威性](Docs/README.md)** —— `Docs/` 下**多数是日期化实验记录**，描述的是**当时**的代码，彼此存在互相冲突的陈述。该索引给出权威矩阵、冲突登记表与三层检索协议；**任何文档与代码不符时以代码为准**。
- [开发进度](ROADMAP.md)：方向、能力阶梯与下一步。
- [独立后端](backend/README.md)：后端进程、控制台页面、采集与在线建图接口。插件侧与后端侧的**职责分工**见正文同名小节。
- [发行包重建](packaging/README.md)：vendor 同步、profile-safe 打包与装后验证。


# N.E.K.O VRC Body

这是一个独立的 N.E.K.O 插件：待机时把宿主公开的 VMC 骨骼流转换为 AnyaDance 六点姿态，显式动作时由 LLM 通过受控语义工具接管；VRChat 移动和 Index 控制器输入优先走 AnyaDance 虚拟设备，Avatar 参数、聊天和兼容回退继续走 VRChat OSC。

## 姿态能力

- 用 `0–180°` 抬升角和 `-180–180°` 方位角控制左手、右手或双手；0° 方位向前，90° 向右，-90° 向左。
- 手的基础旋转会自动跟随肩到手的方向，再用 pitch/yaw/roll 和掌心预设叠加局部偏移。
- 把手移动到相对 HMD、胸口或髋部的局部 XYZ 目标。
- 开掌、握拳、抓握和指向；`body_gesture` 另可播放十二种受控短手势并在结束后恢复动作前姿态。
- 向腰部、胸部或头部高度伸手，并在轨迹末段发送 grip。
- 用 `body_express` 提交“问候、同意、解释、思考、庆祝”等语义意图，由分层状态机选择自然表达动作。
- `motions/` 目录中的白名单 AnyaDance `.nya` 预制动作由 `body_express` 的状态机自动选片，不再单独暴露枚举/播放入口。
- 用 `body_status` 读取 LLM 可理解的当前/上一动作、切换关系、完成状态、剩余时间和实际语义姿态；`include` 可只取 `body`/`autonomy`/`vision` 中的某几段。
- 用 `body_vrchat_input` 安全脉冲 Grab、Use 或 Drop，并自动发送释放值。
- 用 `body_turn` 发送有时限的 VRChat 转身轴值，或用 `body_stop` 分层停下（`scope=all/navigation/axes/action/freeze/unfreeze`）。
- 用 `body_chatbox` 发送附近玩家可见的 VRChat 聊天框文本（最多 144 字符）。
- 把 N.E.K.O 对话里**角色说出口的那句**自动转发到 VRChat 聊天框（`[chatbox_relay]`，默认开启）。只转 `proactive_reply`，不转承载用户原话的 `proactive_instruction`。
- 用 `body_status(include=["autonomy"])`、`vrc_autonomy_goal`、`vrc_wander_step`、`body_stop(scope="navigation")` 管理当前实例内的自主目标；授权必须从调试面板或 `/autonomy/arm` 手动启用。
- 向 N.E.K.O 后台 Agent 暴露正式的“观察当前 VRChat 世界”和“寻找或走向 VRChat 目标”入口；自然语言里的“找 NPC / 走过去 / 跟着它”会进入同一个安全导航接口，而不是被误判成仅有身体姿态能力。唯一语义目标可自动绑定稳定 ID，多个候选仍必须交回主 LLM 选择。
- 用 `body_turn(degrees=360, wait_complete=true)` 执行并校验一次 360° 原地转向；它只证明转向完成，不会把沿途未送入 VLM 的画面伪装成“已经检查”。普通 Agent 的拒绝结果会提升为 failed run，避免 `accepted=false` 被宿主误说成动作完成。
- 监听 VRChat 的 Avatar 切换和参数回传，把白名单动作状态加入 `body_status`。
- 在 `idle` 状态监听 N.E.K.O VMC 2.0 OSC，完成 Humanoid FK 后中转头、双手、髋和双脚六点姿态。
- 单一发送线程以配置的发送频率向 `127.0.0.1:39570` 发送完整 UDP 帧，控制器叠加与六点姿态共用同一帧。`config.py` 的 `[anyadance].rate_hz` 默认 60 Hz，**随插件部署的 `plugin.toml` 设为 120 Hz**。

> 本地避障是 2.5D 增量栅格（需双目深度可用、且本地导航已启用），只覆盖走过的地方，不承诺跨遮挡推理，“绕到墙后”会明确返回 `unsupported_spatial_navigation`。它只管**局部安全**；整图建图与跨会话记忆是另一条链路，见下方「在线建图与跨会话记忆」。

“拿东西”只表示手移动到语义目标并发送握持输入。VRChat OSC 不提供通用物体位置或 Pickup 附着确认，因此插件始终把 `object_held` 报告为 `unknown`。

## 安全与运行约束

1. 插件默认不自动启动；启动后仍处于 `disabled`。**`body_enable` / `body_disable` / `body_reset` 是面板专属入口**
   （`panel_command`，`metadata.agent_auto=false`，宿主保证 Agent 看不到也分派不了）——
   **agent 调不到，需要用户在调试面板点「启用」**。其余身体操作走 `debug_command` 通道，agent 可用。
2. `body_stop(scope="freeze")` 会撤掉当前自主目标、冻结当前合法姿态、释放所有控制器输入，并锁定后续动作与转向（`turn` 与 `NORMAL`/`INPUT` 两组命令一起被闩锁挡在 `submit()` 外，因此导航器的朝向修正也进不来）。撤目标是尽力而为的副作用，后端不可用时不影响急停本身报成功。解除按「谁锁的谁能解」分流：LLM 自己下的急停可以由它自己调用 `body_stop(scope="unfreeze")` 恢复 T Pose（否则它有权进入一个自己无权离开的状态）；面板「立即急停」按钮下的急停和故障闩锁只能由用户点击面板的「复位 T Pose」解除。
3. 调试面板的「允许模型急停」开关决定 LLM 能不能自己下 `scope="freeze"`，默认允许（急停是降权，多数时候让它能停下更安全）。关闭后模型只剩 `scope="all"` 这类普通停车，freeze 会被明确拒绝而不是悄悄降级——降级会让它把返回值当成「已经急停」。开关只管进入 freeze，不影响它解除自己已经下过的那次急停，也不影响面板自己的急停按钮。面板急停走的是只有面板能分派的 `body_freeze`（`panel_command`，`metadata.agent_auto=false`），不是带来源参数的共用入口：共用入口拿不到调用方身份，参数里的来源字符串模型自己也能填。
4. UDP 协议本身没有响应或发送者身份。启用并收到 AnyaDance 驱动遥测时，`body_status.driver_log` 和 `body_status` 的 `body.driver_delivery` 可以确认驱动实际处理了命令；遥测不可用时只能确认本地发送成功。
5. 插件假设接管期间 AnyaDance UI 不再向 39570 发送。驱动遥测发现其他活跃来源时会报告冲突、解除自主授权并释放输入；否则为 `unsupported` 或 `detected_unattributed`。
6. AnyaDance 虚拟驱动可能影响真实 SteamVR 设备追踪。实机测试应从私人 VRChat 实例、小幅度和低速度动作开始。
7. OSC 同样使用 UDP。`delivery_confirmed=false` 只表示本机完成发送；只有收到 9001 回传时 `connection` 才显示 `detected`，没有回传时为 `unknown`，不能据此断言 VRChat 离线。
8. 默认只有一个程序能独占监听 `127.0.0.1:9001`。若已有 OSC 路由器占用该端口，应修改 `listen_port`，并让路由器或 VRChat 向新端口转发。
9. VMC 待机中转默认独占监听 `127.0.0.1:39539`。N.E.K.O 的 VMC 输出必须启用并指向该端口；若端口已被其他 VMC 接收器占用，应修改双方端口或使用 OSC 路由器分流。
10. 聊天框转发默认开启，会把角色说出口的话广播给 VRChat 里周围所有玩家。它只转发 `proactive_reply`（宿主确认提交过的回复），绝不转发 `proactive_instruction`——后者承载用户的原话，属于输入而不是发言。关闭面板的「聊天框转发」开关（`panel_command`，Agent 不可自动分派）后不再发送任何对话文本；重新开启不会补发关闭期间的历史。`delivery_confirmed=false` 同样适用于这条链路：`accepted=true` 只表示本机 OSC 发送成功，不代表 VRChat 已显示。

## 身体自知

插件启动时会向 LLM 注入身体工具使用规则：聊天历史不是可靠的实时姿态来源；回答当前动作，以及执行“继续、换一个、另一只手也”等相对命令前，应先调用 `body_status`。

`body_status` 的 `body` 段提供：

- `motion`：当前动作来源、阶段、进度、已用/剩余时间及是否已经到达目标。
- `previous_action` 和 `transition`：上一动作的结束原因，以及刚从什么动作切换到什么动作。
- `pose`：由当前输出帧反推的双臂抬升角、方向、伸展、双手姿态/握持和头部旋转。
- `summary`：可直接用于自然语言判断的中文摘要。

每个成功受理的动作还会返回 `target_pose_summary`。其中 `completion_confirmed=false`，明确表示这只是目标意图；动作实际是否完成仍以 `body_status` 的 `body.motion` 为准。

## 分层动作状态机

调度器把动作分为安全层、基础动作层、手部层和表达层，并在 `body_status` 的 `body.behavior` 中公开当前模式、活动层、优先级、上一基础动作、切换策略和最近决策。

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

`body_status` 的 `body.vrchat_osc` 只报告 VRChat 实际发回的 Avatar ID 和配置参数，不提供实时骨骼角度，也不能确认 Pickup 是否附着。默认关注 `NEKO_Action`、`NEKO_ActionActive`、`NEKO_ActionPhase` 和 `NEKO_Holding`；这些参数需要先在 Avatar 的 Expression Parameters/Animator 中创建并驱动。

白名单还包含六个 VRChat 内置 Avatar 参数（`VelocityX/Y/Z`、`AngularY`、`Upright`、`Grounded`），`body_status` 的 `body.vrchat_osc.motion` 由它们算出实测移动反馈——这是全仓库唯一能说明“我是不是真的动了”的回传，所有工具的 `accepted=true` 都只代表本机 UDP 发送成功。

> ✅ **参数名已实机验证（2026-08-23）。** 实测 `VelocityX/Y/Z`、`AngularY`、`Grounded` 均会回传，且 `VelocityX/Z` 是 **avatar 本地坐标系**——转 90° 后仍是 Z 主导，因此 `velocity_z` 直接就是前进分量，不需要先按 HMD yaw 旋转。⚠️ **跑满速度不在本 README 给数**：仓库自己登记着一组未解决的冲突（2.67 vs 4.0 m/s，见 `Docs/CONFLICTS.md` C11），裁决是**两个都不许单独用来换算距离**，需要一次标定跑速的实测。`VelocityX/Z` 只有角色移动时才回传，因此每条速度记录只是短时样本；静止沉默是正常现象，旧的 0 或移动速度超过时限后都会变成 `motion.available=false` / `velocity_feedback_quiet`，OSC 层不会把它伪装成当前零速度。当前 Avatar 已成功回传过 X/Z 后，导航器会确认反馈能力；前进超过 450 ms 起步宽限仍然沉默时，才将沉默作为零速度累计，默认连续 4 tick 后进入绕行。Avatar 切换会清空能力确认。

`motion` 除标量速度外还导出 `velocity_x` / `velocity_z` 原始轴值，以及按水平模长归一的
`forward_ratio` / `slip_ratio`。这两个比值用于区分「畅通」与「撞墙」：VRChat 的角色控制器
会把移动向量投影到墙面上，所以斜撞墙时前进分量塌陷、侧滑分量抬起，而速度模长未必变小——
`hypot` 之后的标量看不出这个差别。`forward_ratio ≈ 1` 是畅通；明显小于 1 且 `slip_ratio`
变大表示正贴着墙滑行，滑行方向即可通行方向；`horizontal_speed_mps` 塌到接近 0 则是正面墙
或墙角。水平速度低于静止阈值时两个比值为 `None` 而不是 `0.0`——`0.0` 会被读成「正对着墙」，
与「站着不动」是完全不同的结论。

## 插件侧与后端侧的分工

身体运行时采用一个粗粒度的本机后端进程，而不是把每个动作拆成独立服务。代码在同一个仓库、后端也直接复用插件包里的模块，但**运行时的持有者是后端进程**：插件启动时用一次性随机令牌拉起后端子进程（loopback HTTP），拿到的是一组**远程兼容代理**（`RemoteScheduler` / `RemoteOsc` / `RemoteVision` …）——同名、同方法，真实对象在另一边。后端不可用时这组代理全部置空，插件进入安全的 `backend_unavailable` 状态：工具如实报不可用，而不是假装成功。

| 谁 | 负责什么 |
|---|---|
| **Hosted 插件**<br>（宿主进程内）<br>`__init__.py`、`tool_defs.py`、`ui/panel.tsx`、`chatbox_relay.py`、`frame_budget.py`、`instructions.py` | LLM 工具的 schema 与参数校验；两个入口的注册与权限（`panel_command` 的面板专属开关、`debug_command` 的 agent 可用命令）；系统提示注入；Hosted TSX 调试面板；拉起并监督后端进程、维护 IPC；把后端的世界 delta 与取证图片转成宿主的 `push_message`；聊天框转发（轮询宿主对话存储）；取图预算（`frame_max_per_minute`） |
| **独立后端**<br>（自己的进程）<br>`backend/service.py`、`process.py`、`vision.py`、`world_state.py`、`navigator.py`、`nav_*.py` | `BodyScheduler` 与 AnyaDance UDP 发送（120 Hz）、VMC 待机中转、VRChat OSC 收发、AnyaDance 驱动遥测、控制器输入、动作片段加载；视觉采集与本地检测；世界状态与持久化；自主导航（`LocalNavigator`）；在线建图与跨会话记忆；Web 控制台与全部 HTTP 接口 |

判断依据是**生命周期，不是文件位置**：需要跨插件重载存活、或要长期跑 120 Hz 与端口监听的东西放后端；要用宿主 API（`push_message`、对话存储、调试面板、工具注册）的只能在插件侧。所以 `scheduler.py` / `osc.py` / `vmc_idle.py` / `driver_log.py` / `host_vmc.py` / `nya.py` 这些模块虽然躺在插件包里，运行时却由对面的 `backend/adapters.py` 实例化 —— **换宿主项目时只需要替换这一个适配器**，`service.py` 与视觉/世界/建图各模块都不用动。

### 独立后端的控制台

后端自带 Web 控制台，不依赖 N.E.K.O 宿主即可查看状态和配置视觉/VLM、
授权自治并提交目标：

```powershell
.\.venv\Scripts\python.exe backend\process.py --standalone --open-ui
```

控制台页面：

- `/`：原控制台——状态、视觉/VLM 配置、自主授权与目标提交、独立运行覆盖配置。
- `/navmesh`：在线建图的实况驾驶页（`goto` 需点击后确认，驾驶另需自主武装）。
- `/navmesh/memory`：按世界浏览关键帧记忆（缩略图、钉住、标签、删除、配额清理）。
- `/coverage`：覆盖俯视页（观测计数/近看质量、洞分类），只读。

普通配置保存在被 Git 忽略的 `backend.settings.json`，只有点击保存时才写一次；
API key 只从环境变量读取。完整说明见 [独立后端](backend/README.md#脱离-neko-运行)。

## 视觉世界状态

插件提供 `world_observe` 工具和 revision 增量世界桥；后端目录内的 `backend/world_state.py` / `backend/vision.py` 提供状态层、DXcam/MSS/WGC 采集与可插拔 OpenVINO/VLM worker。它们不进入 AnyaDance 的 120 Hz 调度线程，也不替代宿主 VMC 待机中转。模型包和 OpenAI-compatible VLM 由部署环境提供，缺少依赖时明确降级为 unavailable；启用采集但没有模型时只运行 capture-only 诊断，不发布猜测实体。视觉后端可以发布带 `confidence`、`source`、`age_ms`、`ttl_ms` 和 `unknown` 不确定性的目标与事件；LLM 读取不到新观测时不得把空结果当成“场景为空”。

### 模型

发行配置从插件内加载三个模型：`models/person_detect_v1.3_s/` 的人物检测 ONNX 与标签文件；`models/osnet_x0_25_msmt17/` 的人物重识别（ReID）嵌入 ONNX，`identity_reid_enabled` 默认开启，模型缺失或 onnxruntime 不可用时自动回落颜色直方图（此时生效阈值换成 `identity_reid_similarity`）；以及 `models/bow_vocab.npz` 的 ORB 词袋词表，离线训练、供在线建图的回环使用（见下方「在线建图与跨会话记忆」）。

各目录都带原始模型说明、参考阈值和 SHA-256 清单，详见 [模型说明](models/person_detect_v1.3_s/README.md)。模型路径相对于插件配置目录解析；**复制插件时应保留整个 `models/` 目录**。DXcam/MSS、OpenVINO 和 OpenAI-compatible VLM 仍使用可选适配器；缺少依赖时明确报告不可用。

### 采集路径

`[vision]` 的 `capture` **代码默认是 `"desktop_mirror"`**（`config.py` 的 `VisionConfig.capture`）；**随插件部署的 `plugin.toml` 设为 `"wgc"`**，按窗口捕获 `window_title` 指定的窗口（部署值为 `"VRChat"`）。Windows.Graphics.Capture 从 DWM 取该窗口自己的合成内容，压在上面的窗口不参与，因此别的窗口盖住 VRChat 时画面依然是游戏本身，也不需要窗口矩形——捕获项跟着窗口走，拖动和改分辨率都不必重新解析坐标。唯一仍然抓不到的情形是窗口最小化：DWM 不再为它合成，此时暂停观测。需要 Win10 1903+。

**改成 `capture = "desktop_mirror"`（或 `"dxcam"`/`"mss"`）才退到桌面镜像**：采集区域取窗口的屏幕矩形而不是整块显示器。这条路径抓的是**合成后的桌面**，VRChat 被别的窗口压住时采集照样“成功”，拿到的却是上层窗口的像素——那正是遮挡检测必须存在的原因（实测同一时刻两者相关系数仅 0.065）。窗口矩形也不是一次性的：窗口被拖动、改分辨率或全屏切换后，启动时解析的坐标就会一直抓错位置，而 DXcam/MSS 的区域在构造时固定、没有改区域的接口。因此后端按 `window_track_interval_ms`（默认 5000 ms）重新解析，只在矩形真的变化时重建采集源；窗口暂时找不到时（最小化、Alt-Tab）保留上一次的矩形，不回退全屏——把整个桌面喂给检测器比暂时抓一块过期区域危险得多。设为 0 可恢复“只在启动时解析一次”的旧行为。该参数对 `wgc` 无效。

桌面镜像采集会自动探测 DXGI 的 GPU/显示输出，并在失败时逐个尝试 MSS 物理显示器；`/perception` 会保留每个候选输出的错误，便于区分权限、显卡和 BitBlt 问题。可在 `[vision]` 中设置 `monitor_index`、`dxcam_device_idx`、`dxcam_output_idx`，或把 `dxcam_backend` 固定为 `dxgi`/`winrt` 进行排查。要启用接近 OBS 窗口捕获的 Windows Graphics Capture 路径，可在后端使用的 Python 环境中安装可选组件：

```powershell
python -m pip install --user "dxcam[winrt]"
```

安装后保持 `dxcam_backend = "auto"`；DXGI 被拒绝时会自动切换到 WinRT。该依赖仍是可选的，未安装时插件会继续使用 DXGI/MSS，并在 `/perception` 报告缺失原因。

### 检测与不确定度

本地检测器还会丢弃过小的框，宽高分别设阈值（`min_box_width_ratio` 默认 0.8%、`min_box_height_ratio` 默认 2%，`min_box_ratio` 为两轴共同回退值）。实测真实帧里出现过 27×27 px 的高分假阳性，它们会被跟踪器当成真人并污染导航器的距离闭环。宽阈值更松是因为站立的人本来就高而窄，共用一个阈值时总是宽度先卡，会把房间对面的人一并裁掉；两条边仍然都要过关，墙缝和 UI 边框这类细长误检照样被挡住。

检测器只回答“有几个人、在哪个方位”。要判断前方几何是否可能正在接近障碍，独立的 `VisionWorker` 还会从连续帧计算短期 `traversability_prediction`；它只输出分扇区的发散风险、接触时间估计、置信度或 `unknown`，不创建实体、不提供米制距离，也不写入持久化世界。转向、特征不足或画面跳变时必须按 `unknown` 处理。

### 给主 LLM 的画面与语义

要确认对方是谁、菜单开着没、界面上写了什么，`vrc_vision_frame` 会把最近一帧降采样后的 JPEG 注入当前回合；被判定为社交相关的主动唤醒推送也会附带一张。图不能走工具返回值——那只是一个 JSON 值，模型看不到里面的 base64——所以两条路都通过 `push_message` 的 image part 交给宿主注入。

画面只用于理解，**不进入 `world_state`**：从像素得出的一切都是低置信视觉猜测，不能当作实体、事件或位置的来源，也不能用来满足 `body_reach_and_grab` 的 `preconditions`；那条路仍然只认 `world_observe` 给出的 `entity_id` 与置信度。采集停止、还没有帧或画面超龄时明确返回 `available=false`，不会退而求其次给一张旧图。

拉图有每分钟上限（`frame_max_per_minute` 默认 10，滑动窗口）。一张 960 px 的 JPEG 进上下文是十万字符量级的 base64，agent 在循环里每回合拉一张足以把会话挤爆。唤醒配的图不占这个预算，它自己受 12 s 的最小唤醒间隔约束。

`semantic_backend="main_llm"` 时，后端不会再为语义识别另起一次模型请求。带 selector 的目标尚未解析时，它把同帧 JPEG、检测候选和 revision 放入一个内存单槽；插件用 `ai_behavior="read"` 将这项被动任务并入当前或下一次正常用户对话。主多模态 LLM 在完成聊天和画面理解的同一回合调用 `vrc_semantic_commit`，后端校验 request ID/revision 后只缓存分类，下一帧本地检测再把分类附到稳定 ID 的当前位置。pending 任务不重复注入，默认至少间隔 `semantic_main_llm_min_interval_s=12` 秒才允许产生下一张，并且新任务覆盖宿主尚未消费的旧图；因此锁定后的导航阶段不再消耗图片 token。

这条桥接不会让主 LLM 逐帧观看：普通位置变化仍只进入内存 revision 账本，重要社交事件沿用原有的限速主动唤醒，用户明确询问当前画面时仍可调用 `vrc_vision_frame`。海报、屏幕和镜像可以分别提交为 `poster`、`screen`、`mirror`，作为负语义缓存避免反复把同一干扰物当作 NPC。独立后端脱离 N.E.K.O 运行时没有“当前主 LLM”，应在面板中切换到 `openai_compatible`。

### 采集启停

视觉捕获也可以独立于身体控制链路启停：调用后端 `POST /vision/stop` 会释放当前 DXcam/WinRT/MSS 句柄并解除自主导航，`POST /vision/start` 会创建全新的 `FrameSource` 后恢复采集；不会复用已经关闭的捕获对象。插件客户端提供对应的 `client.vision.stop()` / `client.vision.start()` 方法。配置为外部 source 时，停止后需要重新注入外部采集器，不能直接重用旧对象。

### 跟踪与目标锁定

自主目标不会直接把主 LLM 接入身体控制线程。后端的 `LocalNavigator` 会在会话手动授权后，以约 10 Hz 根据新鲜视觉实体生成受限的短摇杆脉冲；目标不可见、方位或距离未知、观测过期或世界带不确定性时立即释放输入。主 LLM 只负责目标和行为选择，AnyaDance 调度器仍以 120 Hz 发送最新姿态与控制器状态。任何朝视觉实体移动的目标都必须携带由 LLM 从最新世界快照中选出的精确 `target_id`；没有 ID 的探索请求只观察，不移动。对于 wander，连续帧几何预测只作为主 LLM 已选方向的安全门：高置信 `predicted_blocked` 会停车并把原因交回主 LLM，不会在本地偷偷改选路线；`unknown` 绝不等于空旷。本地导航不会按文字、标签或置信度自动锁定实体。方位角和接近度在本地做轻量时序平滑，单帧低置信或漏检最多宽限 300 ms；高速巡航进入宽限时会立即降到最低油门，把最坏盲走距离限制在约 0.2 m。世界不确定、整体观测过期或宽限到期仍会硬停车。

本地 person 检测会把易变的 `openvino:track:N` 映射为当前后端会话内稳定的 `avatar:session:<token>:<N>`，目标短暂离开视野并被分配新轨迹后，可通过最多 6 个多视角外观原型重新绑定；外观候选打平时，再用剔除人物框后的低分辨率背景指纹、15 秒内的短时几何和轨迹稳定度消歧。原轨迹 ID 保留在实体的 `attributes.track_entity_id` 中用于诊断。锁定目标消失时不会回退选择同标签的海报、镜像或其他玩家。每个新视觉 revision 都把比例转向修正换算为“当前实际 yaw + 修正”的绝对目标，可以在上一段平滑转身仍执行时连续重定向，不再等待收尾，也不会把相对角度叠加成超调。

当画面中有多个 person 候选时，主多模态 LLM 先调用 `world_observe`，再调用 `vrc_vision_frame(overlay=true)`。叠框会显示仅对该帧有效的 `T1`、`T2`，工具结果的 `overlay.candidates` 同步给出每个 T 编号对应的完整 `target_id`。LLM 用画面排除看起来像海报、屏幕或镜像的候选，再把映射中的完整 ID 提交给自主导航；`T1` 本身绝不能作为 `target_id`。JPEG、该帧检测实体和 revision 在检测完成后原子地覆盖同一个内存单槽；不保留帧历史、不创建临时图片，也不增加世界存储写入。只有 `overlay.paired=true` 才会画框，因此不会再拿缓存旧图叠最新世界。叠框失败或候选为空时不会发起移动。

这个稳定 ID 不是 VRChat 的 `usr_`/`avtr_`，后端重启后会变化，也无法可靠区分使用完全相同 Avatar 的玩家或镜中像；两个目标同时可见时绝不合并，历史模板歧义只有在背景、短时几何或长期稳定度证据足够明显时才复用，否则仍分配新 ID。

### 世界日志与持久化边界

世界日志若以后启用，只能作为低置信度辅助来源。玩家实体应使用 `vrchat:player:<user_id>` 这样的稳定 ID；收到 `player_left` 时，适配器必须在同一批次提交 `remove_entity_ids` 和离开事件，并用 `remove_source` 限制删除范围。这样玩家退出后不会残留幽灵实体，也不会误删视觉检测器发布的同名对象；后端还会用接收水位拒绝保留窗口内迟到的旧帧，避免删除后幽灵复活。换世界时只清理日志来源的玩家实体；若适配器会积压超长离线队列，应提高视觉水位上限。当前仓库不接入世界日志解析器或 Contact 总线，也不安装动作生成模型；自主运行时只管理手动授权、世界新鲜度和安全释放，不自动执行好友、邀请或世界切换。模型通过独立 detector 或 sidecar 接口接入，并由配置显式启用，不把未启用功能的状态塞进插件主流程。

`world_memory.persist_world` 只保存可跨进程复用的世界事实。逐帧检测轨迹、`avatar:session:*`、synthetic 实体以及显式标记 `attributes.memory_scope="observation"` 的外部观测不会写盘；旧版本已经保存的此类实体会在下次启动时自动清除。持久化负载使用稳定格式，内容没有变化时不会随 10 Hz 感知循环重复写文件。**这是“当前世界有哪些实体”的存储，与下面按世界分区的关键帧记忆是两套东西**，不要互相套用。

## 在线建图与跨会话记忆

上面那条 2.5D 增量栅格只管**局部安全**；这条链路是**显式开启**的整图建图：边走边建一张三态栅格，并把“来过的地方”按世界分区记住，供下一次会话认出来。它**默认不占用 SteamVR**——必须显式 `start`（控制台 `/navmesh` 的按钮，或 `POST /worldmodel/navmesh/start`）才打开 OpenVR 会话。

- **位姿**：OSC `VelocityX/Z` 零阶保持积分 × HMD 朝向（航位推算 20 Hz），再由 `nav_loop.LoopCloser` 做回环（ORB + 双目 PnP，**只修平移**；朝向始终来自 HMD）。两次回环之间是纯航位推算，实测漂移约 2.8% 路程。
- **地图**：`nav_mapping.KeyframeGridMapper` 按关键帧融合双目点云，成 free / occupied / unknown 三态。**unknown 永远不当可通行**；free 只来自看见的地面，不做自由空间射线追踪。障碍判定按观测距离分层——视差深度误差随距离平方增长，1.5 m 以外的抖动会把真地面推进障碍带。
- **控制**：`goto`（地图系 `x/y`）、`explore`（前沿探索）、`cancel`。未授权、双目或位姿过期、已有其它自主目标在执行时一律不动；双目断了就停（没有近距急停不能盲走）。
- **坐标**：地图系是**会话自建**的（原点 = 会话起点，追踪米），**不是 VRChat 世界坐标**；对外 `goto`/`status` 一律是世界米 = 追踪米 × `world_scale`。写长度时必须标明是**追踪米**还是**世界米**，这是登记在案的口径规矩。
- **它是控制台面，不是 LLM 工具面**：驾驶入口是后端控制台的 `/navmesh`。Agent 侧的自主移动仍走 `vrc_autonomy_goal` 与本地导航器，插件调试面板里没有任何建图界面。

### 跨会话记忆：她认得以前来过的地方

- **检索默认开**（`nav_xsession`）：新会话的每个关键帧都去历史会话里查“这是不是我来过的地方”，验证通过就记一条**跨会话约束**（写到世界目录的 jsonl）。验证语义与回环一致（内点/覆盖率/重投影门槛），索引两侧统一用“深度 ≤ 8 m”的描述子子集。
- **记忆按世界分区**落在后端状态目录 `<state_dir>/navmesh_memory/`：只存位姿整表、地图侧重定位特征和彩色缩略图，**不存点云**（约 219 KB/帧 vs 17 KB/帧）。回环会整体改位姿，逐帧写的位姿会过期，因此位姿只整表写：周期性 + 会话结束。超配额从最旧的未钉会话删起，**钉住的会话永不自动删除**；管理页 `/navmesh/memory`。
- **外观候选**：回环原本只按“当前位置附近的帧”找候选，而位置本身就是漂移量——漂得多了真配对会排到候选之外。因此另加一路 ORB 词袋（`models/bow_vocab.npz`，离线训练、随包发布）做**外观**候选，与位置候选取并集。词表缺失时回环照常跑，只是没有外观候选。
- **世界先验**（可选）：`research/tools/offline_fusion.py --write-prior` 把同一世界里多场会话的栅格融合成一张世界系先验。在线会话要**先攒够跨会话约束、在线估出 gauge 并过质量闸**才允许注入；过不了闸一格都不注入，默认只注障碍、不注 free。闸门判定与原因都在 `status()["prior"]` 里。

### 配置

```toml
[navmesh]
world_scale = 0.755            # 追踪米 → 世界米。只改这里；mapper/loop 里的同名值是占位，会被它覆盖
expected_baseline_m = 0.126    # 启动自检的期望双目基线；0 = 不检查
baseline_tol_m = 0.01
loop_closure = true
xsession_world_tree = true     # 会话末自动并树；关掉就得手动跑 research/tools/xsession_align.py --world <世界目录> --world-tree

[navmesh.mapper]
range_m = 5.0

[navmesh.prior]                # 世界先验的在线消费闸门：过不了闸一格都不注入
enabled = true
min_constraints = 8
max_pos_med_m = 0.5
min_inlier_frac = 0.2          # 0.2 不是 0.5：整场自洽的核心占比实测只有 0.23–0.36
min_core_span_frac = 0.5

[navmesh.memory]               # 按世界分区的关键帧记忆
enabled = true
max_total_mb = 4096
max_world_mb = 1024
max_sessions_per_world = 30
max_session_mb = 512
min_session_keyframes = 20
pose_flush_s = 10
```

`world_scale` 是唯一真值源，而且**随 avatar 缩放**：在线会话开头按本场眼高自动乘（会话中途缩放偏离 >1% 且稳住 2 s 会重开会话——旧图按旧尺度建，不能续用）。换一个不同身高的 avatar 要重标基准值。

`expected_baseline_m` 只做启动自检：自编 AnyaDance 驱动给 0.126，SteamVR 注册的发行版会静默回到 0.063（深度噪声翻倍、跨会话记忆不能混用）。**改它不改 `world_scale`**（它同时进分子分母，会约掉）。

录制与记忆目录都在后端状态目录下（`navmesh_recordings/`、`navmesh_memory/`），可用环境变量 `VRC_NAVMESH_RECORD_DIR` / `VRC_NAVMESH_MEMORY_DIR` 覆盖；**录制默认关闭**，且两者都不写插件安装目录。

接口清单见 [独立后端](backend/README.md) 的「在线 navmesh」一节；架构定位与实测边界见 `Docs/README.md` §二 权威矩阵与 `Docs/FACTS.md` §五。

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

只有调度器处于 `idle` 且 VMC 帧未过期时才应用中转。精确动作、VMD、序列、保持姿态和急停不会被 VMC 覆盖；动作结束回到 `idle` 后自动恢复最新宿主姿态。急停解除也走这条路：`body_reset` 与 `body_stop(scope="unfreeze")` 提交的 `reset` 以 `completion="idle"` 收尾，滑回 T Pose 后状态变回 `idle`，VMC 待机中继随即自动接管，无需另外恢复。`body_status` 的 `body.idle_relay` 会报告监听状态、帧龄和当前是否正在应用。

### 重新校准（动作一直是歪的时候）

零点一旦锁在错误的姿势上，之后每一帧都带着同样的偏移：动作看起来永远别扭，但中转不报任何错误——帧照常接收、`last_error` 是空的，除了人眼看着不对没有别的信号。调试面板的「重新校准 VMC」按钮就是给这一刻用的，它走 `panel_command`（`metadata.agent_auto=false`），模型看不到也分派不了：这个故障它没有任何办法判断，给它入口只会让它在看不见的状态上乱按。

默认走宿主 T Pose 握手，等一次权威静止姿势再重建基准。这条路要求 `vmc_idle.manage_host_output = true` 且校准线程在跑；条件不满足时后端**原样拒绝而不是先清掉基准**——清了没人重建的话，角色会从「歪着能动」变成「正着不动」，比按之前更糟。

宿主给不出 T Pose（没有托管输出、REST 接口不通、握手反复超时）时，面板会亮出「用当前姿势作基准」：不等 T Pose，拿下一个完整帧直接当零点。基准准不准取决于按下那一刻的姿势，所以不做默认，但它必须存在——否则严格路径失败就没有出口，只能重启后端。校准进行中面板会显示已拒绝的帧数，因为那段时间普通帧全被拒、角色停在最后一帧。

宿主暂停、页面冻结或 VMC 输出重新握手时会发 `/VMC/Ext/OK = 0`，插件随即作废六点基准并停止应用中转（VRChat 角色停在最后一帧）。恢复后**不会**用彼时的动画姿势重新自锁手腕朝向与手指弯曲零点——那会让整套骨骼永久错位；插件改为自动重新请求一次 T Pose，拿到权威静止姿势后才重建基准。帧间隔超过 `stale_after_ms` 时丢弃卡顿前残留的半帧，避免与恢复后的新帧拼成跨卡顿的混合帧。校准状态见 `body_status.idle_relay.calibration` 的 `needs_recalibration`、`baseline_established` 与 `rejected_frames`。

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

参数工具示例（`body_turn` / `body_stop` / `body_chatbox` 是独立工具；
`body_vrchat_input` 走 `debug_command` 通道）：

```text
debug_command(command="body_vrchat_input", arguments={action="grab", side="right", hold_ms=100})
body_turn(degrees=-45)
body_stop(scope="axes")
body_chatbox(text="你好", immediate=true)
```

传入不存在的 Avatar 参数时，OSC 数据报仍可能成功发送，但 VRChat 不会产生对应动作。VRChat 的移动轴必须使用 -1..1 的浮点数并在结束时归零；本插件对每次移动/转身设置 100–10000 ms 的自动归零超时，`body_stop(scope="axes")` 会同时归零三条轴（但只清轴不撤自主目标，导航器下一帧还会把她推回去——用户喊“停下”要用默认的 `scope="all"`）。`accepted=true` 只代表本机 UDP 发送成功，不代表 VRChat 已移动或转身。`body_reach_and_grab` 会在动作最后 15% 自动安排同侧 `/input/GrabLeft` 或 `/input/GrabRight` 脉冲；动作被替换时守卫会阻止尚未发生的按下，急停、取消、复位、禁用和插件关闭都会清空定时输入并发送释放值。聊天框不是私密通道，附近玩家可能看到。

## 调试 UI

插件提供 N.E.K.O 原生 Hosted TSX 调试面板。在插件管理器中选择 `N.E.K.O AnyaDance Body`，点击“打开面板”即可使用。面板不启动额外 Web 服务，也不加载第三方前端依赖，只拥有 `state:read` 和 `action:call` 权限。

面板包括：

- AnyaDance 120 Hz 发送频率、丢帧、控制器释放延迟、数据包和当前动作进度。
- VRChat OSC 9000/9001 状态、Avatar ID、收发计数、错误和参数回传。
- 启用、平滑禁用、复位与急停，以及「允许模型急停」权限开关。
- 「重新校准 VMC」与失败时的「用当前姿势作基准」退路，附当前基准代际与已拒绝帧数。
- 手臂角度/方位/伸展/掌心调节，手型、程序化手势和伸手抓取测试。
- 语义表达意图、状态机模式、活动层和优先级保护调试。
- `.nya` 动作目录诊断（可用/已索引/待解析/缓存/元数据错误）；选片交给 `body_express` 的状态机，面板不再手点片段。
- Grab/Use/Drop 输入脉冲测试。
- 每秒自动刷新、身体状态/OSC JSON 快照和最近调试命令日志。

**建图与记忆不在这里**：这个面板只驱动身体与视觉开关；地图、点云和关键帧记忆的界面在独立后端控制台（`/navmesh`、`/navmesh/memory`、`/coverage`），见上方「在线建图与跨会话记忆」。

调试面板调用的是与 LLM 工具相同的校验和调度实现，因此仍受启用状态、范围、安全锁定和动作时长限制。面板按下的急停必须点击“复位 T Pose”解除锁定；LLM 自己下的急停由它自己 `body_stop(scope="unfreeze")` 解除。能力开关（身体输出、复位、立即急停、允许模型急停、自主授权、视觉采集、重新校准 VMC）走独立的 `panel_command` 入口，`metadata.agent_auto=false` 由宿主保证 Agent 既看不到也分派不了。

面板每秒刷新时只扫描动作文件名、大小和修改时间，不读取或解析 `.nya` 正文。新动作在首次播放时由后台线程完成完整校验；首次解析期间命令按钮会显示执行中，但状态刷新和其他插件事件不会被同步文件读取占住。解析结果按文件签名缓存，文件内容变更后会自动失效。当前最多保留两个已解析动作，便于重复播放和动作切换。

## 安装与验证

发行包（`dist/neko_anyadance_body-<版本>-windows-x64-py311-profile-safe.neko-plugin`，当前版本 `0.13.30`）自带模型与第三方 Python 依赖，**使用者不需要手动装依赖或运行 pip**；导入方式见 [N.E.K.O 宿主一键安装包说明](INSTALL.md)。包内 `DEPENDENCIES.json` 列出实际依赖版本，各依赖的许可保留在 `vendor/*.dist-info/`，包外 `.sha256` 用于校验下载完整性。

开发期用 N.E.K.O 仓库自带 CLI 对此目录执行检查与构建：

```powershell
cd H:\AI\neko-music\N.E.K.O
python -m plugin.neko_plugin_cli.cli check H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body
python -m plugin.neko_plugin_cli.cli build H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body
```

> ⚠️ 路径是**源码目录**，不是安装后的插件目录 `neko_anyadance_body`（那是 `plugin.toml` 的 `id`）。

若系统 Python 缺少 N.E.K.O 宿主依赖，请使用该仓库配置的 `uv run python` 执行同样命令。

**要出正式发行包，走 [发行包重建](packaging/README.md)**：先把 `uv.lock` 同步进本项目 `vendor/`（`uv export` 出锁文件，再 `uv pip install --target vendor --require-hashes`），然后跑 `packaging/build_neko.py --host <N.E.K.O源码根目录>`（`build.bat` 是它的本机一键封装）。只交付文件名带 `-profile-safe` 的包：它移除打包器自动生成、无业务参数的默认 profile 并重算 payload 哈希，避免旧宿主同名 profile 归属不明时导入失败。构建后可用 `packaging/verify_runtime.py` 在干净解释器里验证安装结果（不启采集、不输出控制指令）。换 Python 小版本或平台必须重建 `vendor/`，不要混用不同 ABI 的 `.pyd`。

## 建议调用

`body_status`、`body_stop`、`body_turn`、`body_chatbox`、`body_express`、`body_gesture` 等是**独立 LLM 工具**。
手臂/手部/抓取类没有独立 schema，走**共用入口** `debug_command(command="…", arguments={…})`：

```text
body_status()
body_status(include=["body", "autonomy"])
body_stop()
body_gesture(name="wave")
debug_command(command="body_arm_pose", arguments={mode="polar", side="right", elevation_deg=130, azimuth_deg=-25, reach=0.9, wrist_roll_deg=45})
debug_command(command="body_arm_pose", arguments={mode="anchor", side="right", relative_to="chest", x_m=0.30, y_m=0.10, z_m=-0.45, palm="down"})
debug_command(command="body_hand", arguments={side="right", pose="grip", strength=1.0})
debug_command(command="body_vrchat_input", arguments={action="use", side="right"})
```

`vrc_wander_step` 只在插件刚注入 `[VRChat 主模型闲逛路线任务]` 时有意义：它把本次选择绑定到那条待决路线请求，不接受 `target_id`，也不能用来接近或跟随人物。

⚠️ **`body_enable` / `body_disable` / `body_reset` agent 调不到** —— 它们是 `panel_command`
（`agent_auto=false`），需用户在调试面板点击。参见 `Docs/CONFLICTS.md` C8 / C17。

角度是插件定义的语义手臂角度，不是真实肩关节测量值。肩膀位置由 HMD 和身体配置估算，肩肘最终由 VRChat IK 求解。

## 预制动作

仓库不再分发预制 `.nya` 动作和 `motions/catalog.json`。这些文件可能包含第三方 VMD/PMX 素材的派生数据，是否可以再分发取决于各自授权；因此安装包默认只包含动作加载器，首次安装后动作目录可能为空。

如果你拥有已授权的 AnyaDance `.nya` 文件，把它们放入插件配置目录下的 `motions/` 目录即可。`catalog.json` 是可选的：存在时，`body_express` 会按意图、侧别和强度选择目录中的真实动作；不存在或没有匹配项时，会回退到程序化表达动作。

动作目录的文件名就是调用时使用的逻辑名称（不含 `.nya` 扩展名），支持中文名称，但不接受绝对路径、子目录或 Windows 非法文件名字符。

动作选片不再有独立的枚举/播放工具：`body_express` 命中 `motions/catalog.json` 时直接播放对应片段，面板只保留目录诊断。播放默认把片段第一帧 HMD 的 X/Z 对齐到当前姿态；`body_stop()` 与 `body_stop(scope="action")` 可随时中断。

`vmd_bake.py` 提供离线烘焙命令。它让 Blender/MMD Tools 完成真实骨架的 FK、IK 和约束求值，再复用 AnyaDance 的六设备重定向算法生成 `.nya`，不会在插件实时调度线程中运行：

```text
python vmd_bake.py --vmd <动作.vmd> --model <模型.pmx> --blender <blender.exe> --export-script <blender_export_mmd.py> --mmd-tools <mmd_tools目录> --target-height 1.50 --body-width-scale 1.0 --leg-length-scale 1.0 --hand-reach-scale 1.22 --hip-height-offset 0.0 --output motions/<名称>.nya
```

比例参数用于离线适配目标 Avatar：`target-height` 控制总体身高，`body-width-scale` 调整手脚相对身体中心的横向宽度，`leg-length-scale` 改变上半身相对脚底的高度，`hand-reach-scale` 控制手掌目标相对肩部的伸展，`hip-height-offset` 做最后的髋部高度微调。默认值等价于 AnyaDance 原始重定向。

VMD 和 PMX/PMD 都是独立的第三方作品；导入前应确认各自授权。插件的烘焙工具不赋予素材的复制或再分发权。

解析器限制单文件 64 MiB、18000 帧和 300 秒，并检查时间线、六设备完整性、有限数值、四元数、坐标及手指范围。设备 Y 略高于安全上限时会与 AnyaDance 一致钳制到上限；循环展开后的播放时间同样不得超过 300 秒。

大型动作在面板中最初会标为“未索引”。首次播放需要真实的 JSON 解码和逐帧安全校验时间；完成后会显示帧数/时长并进入内存缓存，之后再次播放无需重复解析。目录自动刷新不会触发这项重活。

调度器内部仍保留多步动作序列这一层（优先级高于精确姿态、低于抓取交互），但它不再有对外入口：`body_sequence` 工具和后端路由都已移除，组合动作改由 `body_express` 的状态机在片段层完成。

## 致谢

本项目的开发与完善离不开以下贡献者的支持：

- [LyaQanYi](https://github.com/LyaQanYi) —— 提供 Token 赞助
- [Kivo-bash](https://github.com/Kivo-bash) —— 提供 Token 赞助
- [Neilzzw](https://github.com/Neilzzw) —— 参与模型与地图制作
- [miaoluoxue](https://github.com/miaoluoxue) —— 早期开发
