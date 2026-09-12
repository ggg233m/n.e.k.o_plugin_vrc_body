# YUI NPC Controller

## 主动搭话与输入陪伴

世界聊天框按 T 打开，Enter 或发送按钮提交，Esc 收起；显示最近 40 条本地发言和 NPC 字幕页，支持滚轮翻阅。
界面独立显示连接状态。新世界通过 `player.chat_activity` 上报 `slot/pid/input_seq/state/idle_ms`，
其中 state 为 open、active、closed 或 submitted，正文仍仅在提交时发送；Owner 验证实际网络发送者、会话和顺序。
打开输入框立即抢占自主移动，5 秒续报、15 秒租约、120 秒无输入操作到期；手动关闭后等候 10 秒。
提交自动关框不解除陪伴，最后一页回复结束后默认等待 60 秒，继续输入可延长。其他玩家仅开框不能抢占已有对话。
`sys.chat_input_ready.activity_version=1` 表示支持输入检测；旧世界仍可在提交后启动陪伴，面板显示能力缺失。

`autonomy.proactive_chat_enabled` 默认开启。社交空闲至少 30 秒后，在活动边界选择同区、8 米内且世界确认可达的玩家；
最久未互动者优先，距离作为次级排序。全局间隔 120 秒、同一玩家 300 秒，无回应后对该玩家冷却 600 秒。
必须收到接近计划的成功终态才排队开场，玩家输入可以取消未发送开场；主聊天模型只生成一句对白，动作意图调用保持静默。
玩家区域及 NavMesh 可达性由世界 `player.pose` 补充，旧世界未发布时不会猜测主动搭话目标。

**鼠标平台差异：** ClientSim 安装编辑器适配器 `NekoChatClientSimInput`，打开聊天框时通过模拟器原生输入流程释放系统光标并停止视角旋转，关闭时恢复。
它不修改 SDK，也不会被打包进世界。真实 VRChat 的 Udon 不开放系统鼠标锁定接口，不能自动复制这一效果；
世界端保留 VRCUiShape 原生点击、滚动与输入交互，Canvas 使用 Default 层。此前的面板内虚拟指针已移除。

## 宿主管理面板

在宿主插件管理中打开本插件的“YUI 控制面板”，可查看状态、连接/断开、开始/暂停自主，
以及调整字幕、玩家聊天、陪伴锁和意图模型开关。面板默认每 5 秒串行刷新，可关闭；隐藏或保存设置时暂停自动刷新。
保存只更新当前配置档的改动字段，点击“重载配置”后生效；重载会中断当前控制。
“动作辅助模型”显示本次运行的调用、成功、失败和实际 HTTP 请求次数，以及最近一次模型输出、时间、耗时和校验结果。
统计包含手动模型测试，启动或重载后清零；兼容格式重试单独计入 HTTP 请求次数。
最近 20 次调用按时间倒序保留在内存，每次输出限长 8,000 字符，展开可查看时间、耗时、输出和校验结果。
采纳状态按调用标识关联自主控制器的实际事件，区分等待活动边界、已采纳、被替代、过期、暂停、完成和执行失败；模型测试明确标记不执行。重载后记录清空。
“当前动作进度”显示自主活动序号、语义目标、区域、行为图正在运行的节点与停留倒计时；并行动作分别列出，终态只依据执行器结果。
Token 用量显示每次调用及本次运行的输入、输出和总量，只累计接口 usage 中实际返回的非负整数。未报告显示“未提供”；已报告用量的重试或失败响应也计入累计，重载后清零。
模型调用记录、配置缺失和动作计划结果提供中文故障原因与处理建议；“原始错误代码”可展开查看。同名超时按模型请求和动作执行区分，未知错误保留代码，不推测原因。
所有面板入口仅供人工使用，不注册给 LLM；模型原文不进入普通状态或日志，面板提供只写不回显的密钥输入。

独立实现 YUI NPC v1.1/v1.2/v1.3 Python 后端、N.E.K.O 插件和 stdio MCP。v1.1、v1.2 保持冻结，v1.3 只在世界明确发布 capability 后增加语义定位、相对移动和连续区域探索。协议事实只来自
`Docs/Protocols/`，不会导入 AnyDance、YOLO、视觉导航或旧 `BackendService`。
0.5.0 增加可选的宿主常驻自主循环；0.5.1 增加独立意图模型和连续生活片段；
0.5.3 默认只读当前角色落盘的近期聊天：既用于生成更自然的观察、转身、附近闲逛与兴趣回访，也把主 LLM 新完成的回答分页显示到 NPC 头顶。0.5.4 增加每位玩家本地独立、按键唤起且跟随视野的世界内输入 UI；只有玩家显式发送才触发宿主主 LLM。0.5.5 修复重复相同问答时落盘修订未变化的问题。0.5.6 补读宿主已有的主动回复记录，使世界输入触发的主 LLM 回答无需等待落盘即可显示在头顶。0.5.7 将头顶回答改为 15–30 秒自适应阅读时长，并把正式世界气泡限制为 3.8 米宽的自动换行布局。0.5.8 修复回车结束编辑未绑定发送事件的问题，并在输入期间锁定本地玩家移动、退出时可靠释放。0.5.9 修复 ClientSim/世界重启后宿主仍复用旧 session、导致输入提交被拒绝的问题；检测到新世界或 `not_handshaken` 后会清理旧控制链路并自动重新 DISCOVER。0.5.10 在输入期间额外锁定姿态并禁用跳跃，头顶气泡改用动态中文字体和头骨锚定，避免缺字与随视角漂移。0.5.11 将回复改为始终位于 NPC 朝玩家一面的环绕浮空描边对白，移除头顶名称，并按统一服务器时间在客户端本地逐字弹出，不增加逐字网络流量。0.5.12 将硬盘记忆的动作上下文与对白显示拆开，只显示最新一条 AI 正文，隐藏存储时间戳和分页标记，并让新回答覆盖尚未播放的旧页。0.5.13 将对白中心改为稳定躯干锚点，并使用水平朝向平滑、微小死区和固定字符宽度消除位置与逐字排版抖动；连接 ACK、会话事件或目录同步超时后会保留物理 MIDI 端口、销毁旧 transport/session 并由常驻线程自动重建，热重载和人工断开使用 30 秒无发送交接窗口，手动连接失败也会恢复后台重连。0.5.14 增加玩家聊天陪伴锁：发言时立即停止自主远行，近处持续注视、超过 3 米才以步行接近，并在最后一页对白真实消失后继续停留 15 秒。
视觉阶段尚未实现，也不会采集画面。

## 控制边界

- 通用配置启动时只跟随 VRChat 日志，不打开 MIDI、不连接；只有配置档同时显式设置
  `yui.autonomy.enabled=true` 与 `auto_connect=true` 时才自动等待并连接兼容世界。
- 这是只在地图 NavMesh/activity bounds 内活动的世界 NPC。连接完成 DISCOVER 后，
  宿主内部可靠切入 `external`，模型不需要也看不到 `npc.arm`。
- `CLEAR_ESTOP` 只存在于宿主入口 `yui_clear_estop`，不是 LLM/MCP 工具。
- session 重建、watchdog、ESTOP、driver 离开或 owner 错误仍会立即停止控制；恢复时
  由宿主重新连接，或由操作者执行 `yui_clear_estop`。
- `free_coordinate_navigation`、玩家姓名、可选 `npc.wander` 和自主循环均默认关闭。
- 本地驱动锁会阻止 N.E.K.O 与独立 MCP 同时打开同一个 MIDI 端口；世界端
  session/driver/ownership 仍是最终权威。

模型工具名固定为：

```text
npc.observe  npc.go_to  npc.go_to_xyz  npc.follow
npc.look_at  npc.act  npc.set_expression  npc.say
npc.stop     npc.estop  npc.wander（可选）
```

v1.2 世界在同时发布 `world_map`、`semantic_navigation` 和
`operation_lifecycle` 后再动态增加：

```text
npc.world_query  npc.navigate  npc.approach  npc.orbit  npc.explore
npc.execute_plan  npc.plan_status  npc.plan_cancel
```

v1.3 世界再发布 `region_localization` 与 `local_navigation` 后增加
`npc.move_relative`；同名 `npc.explore` 自动改用 Unity 单 operation 连续探索。
模型只看到当前区域、楼层、最近 Anchor 的相对距离/方位，不看到 NPC、Region、
Anchor 或 NavMesh 的绝对坐标。开启 `yui.autonomy` 后，空闲时由宿主规则循环自行
驻足、沿 `route_edge` 游览、探索 Region，并响应触摸/挥手/凝视/靠近事件；不需要
LLM 持续调用工具。启用 `yui.autonomy.intent_model` 后，独立 API 只生成结构化心境
和 2～4 项生活活动；可选聊天上下文只读 `memory/<当前猫娘>/recent.json`，不会
扩展宿主总线、回写聊天，也不会触发聊天气泡或 TTS。

`npc.execute_plan` 只接受冻结的受限 JSON 图，由 Python 后台执行器持有
`plan_id` 和 operation 证据；不存在 `npc.plan_step`。`npc.observe` 最多投影
8 项附近语义事实，v1.2/v1.3 均不向模型返回绝对世界坐标。

其中 `npc.go_to_xyz`、`npc.wander` 和 operation 类工具按配置/capability 动态隐藏。
冻结的 `LOOK_AT` 玩家命令没有时长寄存器，因此玩家注视只接受
`duration_ms=0`（持续），使用 `npc.stop` 清除；坐标注视支持 0..127000ms。

## N.E.K.O 安装

最终插件目录：

```text
N.E.K.O/plugin/plugins/yui_npc_controller/
```

从 N.E.K.O 仓库执行：

```powershell
# 按本机实际位置填写；不要把这些路径写进插件配置。
$YuiProject = "<path-to-yui_npc_controller>"
$NekoProject = "<path-to-N.E.K.O>"
$Python311 = "<path-to-python-3.11.exe>"
$NekoPython = Join-Path $NekoProject ".venv\Scripts\python.exe"
$PluginPackage = Join-Path $YuiProject "dist\yui_npc_controller.neko-plugin"
$PluginsRoot = Join-Path $NekoProject "plugin\plugins"

Set-Location $NekoProject

& $NekoPython -m plugin.neko_plugin_cli sync `
  --python $Python311 `
  $YuiProject

& $NekoPython -m plugin.neko_plugin_cli check $YuiProject

& $NekoPython -m plugin.neko_plugin_cli build `
  $YuiProject `
  --out $PluginPackage

& $NekoPython -m plugin.neko_plugin_cli install `
  $PluginPackage `
  --plugins-root $PluginsRoot
```

`sync` 只更新插件自身的 `vendor/`，不会把插件安装进 N.E.K.O。必须再执行
`build` 与 `install`；安装成功时 CLI 会同时验证包内 payload 哈希。

宿主入口为 `yui_connect`、`yui_clear_estop`、`yui_disconnect`、`yui_status`、
`yui_reload_config`、`yui_autonomy_start/pause/status`、
`yui_autonomy_intent_probe`；它们不会注册为 LLM 工具。
同一宿主进程内重复调用 `yui_connect` 会复用已经完整握手的 session，不重放
DISCOVER 目录；需要重新声明 ownership 时应先 `yui_disconnect` 再连接。目录只在
全部页面到齐后重建一次动态工具面，高频 `npc.state`/心跳 ACK 不会重复注册工具。

`npc.navigate`、`npc.orbit`、`npc.move_relative`、`npc.explore` 和 `npc.execute_plan` 都只等待命令 ACK，
随后立即返回 `accepted + plan_id`；长行为在后台继续执行。宿主不得把工具调用本身
阻塞到动作结束，应通过 `npc.plan_status` 读取终态证据。

## 聊天记忆驱动的自然自主行为（0.5.2）

自主循环和 LLM 工具共享同一个行为图调度器，但计划来源不同。显式工具自动取消并
停止自主来源计划；自主计划永远不能抢占显式计划。普通显式命令只有在对应 plan 或
operation 收到 `succeeded/failed/cancelled/unknown` 终态后，才开始计算默认 8 秒恢复
延迟；`npc.stop`、`npc.estop`、watchdog 和人工断开保持暂停，必须人工启动或重新连接。

规则候选只来自世界发布的 Anchor、Region 和 `route_edge`，维护近期访问历史、
10 分钟路线签名冷却与最长 5 分钟的指数失败退避。动作混合由意图模型决定，不再
追逐固定移动比例；规则层只在长时间无移动时防止永久站立。普通跨区至少间隔 180 秒，
每个生活片段最多跨区一次；强度不低于 0.7 的有效兴趣可消费一次路线冷却覆盖。
独立意图模型在启动、完整聊天轮次更新、片段边界、重要社交事件及 3～6 分钟保底周期
异步生成生活片段；失败或超时不会中断规则循环。普通动作意图请求不发送
`ai_behavior="respond"` 消息；主动搭话通过独立开关控制，并在真实靠近完成后调用主聊天入口。第一阶段的 `AutonomyStimulusProvider` 固定使用
无操作实现，不采图。

模型请求会把当前世界的目标、动作、标签与玩家槽位写入结构化输出约束，
并提供可直接复制的引用列表。输出校验失败时最多纠正一次，沿用同一目录快照和
原请求总时限；纠正后仍不合法就回退规则，不猜测或替换未知目标。
一次逻辑调用可能包含 schema 兼容回退及输出纠正，实际 HTTP 次数和 Token 分别累计。

已采纳片段的兴趣、避让目标和活动偏好会保留到有效期结束，等待下一片段时继续影响
目标选择；安静偏好延长停留，活跃偏好使用附近小范围移动。偏好延续至少间隔 30 秒，
与原片段共用最多一次跨区限制，仍须通过路径、能力和失败退避检查。
暂停、人工接管、进入对话陪伴或切换角色会清除旧偏好。面板显示当前模型目标、
剩余有效期，以及当前活动来自模型片段、偏好延续还是规则。

NEKO Home 配置档可显式添加：

```toml
[yui.autonomy]
enabled = true
auto_connect = true
decision_interval_s = 1
resume_delay_s = 8
dwell_range_s = [8, 20]
explore_range_s = [15, 35]
social_cooldown_s = 60
llm_inspiration_range_s = [180, 360]

[yui.autonomy.intent_model]
enabled = true
endpoint = ""
model = "gemini-3.7-flash"
api_key_env = "TEST_API"
timeout_s = 20
min_interval_s = 30
temperature = 0.7
max_output_tokens = 700

[yui.autonomy.intent_model.chat_context]
enabled = true
source = "recent_file"
max_turns = 6
max_chars = 6000
poll_interval_s = 1
max_file_bytes = 2097152
```

代码不内置任何中转站地址。`endpoint` 留空或省略时意图模型不会启用（`configured()`
为假，只报 `not_configured`），规则循环照常运行；要接入 OpenAI 兼容中转站，在这里
填写完整的 chat/completions 地址即可，必须是 `https://`。密钥可在插件面板输入；也可使用备用环境变量 `TEST_API`
环境变量中；设置后必须重启 N.E.K.O。人工入口
`yui_autonomy_intent_probe` 只验证认证、模型、JSON schema 与脱敏记忆状态，不会
显示聊天正文，也不会应用返回的活动。

## 主对话头顶显示（0.5.7）

`yui.chat_bridge` 默认开启。普通用户回合仍由独立只读提供器轮询当前角色的
`memory/<当前猫娘>/recent.json`；世界输入通过 `push_message(..., ai_behavior="respond")`
形成的主动回合不会立即落盘，因此只读补充宿主现有 `conversations` 存储里的
`proactive_reply`。没有新增总线字段，也没有修改宿主核心。每个新完成的主 LLM
回答通过既有 `TEXT_UTF8` 原子事务显示在 NPC 的 `BubbleText`。超过单笔 384 UTF-8
字节的回答会安全分页，默认最多 4 页。短句约 10 秒，长页按每秒 8 字加 6 秒
估算，总时长最多 25 秒；显式配置更长最短时长仍会尊重该设置。世界内 `BubbleText` 使用 3.8 米最大宽度和自动换行，
避免长回答横跨场景。

这条投影不调用 `push_message`，不会再次触发主模型，也不会触发 TTS。人工入口
`yui_chat_bridge_status` 只报告文件、修订、主动回复记录计数、排队页数和脱敏错误，不显示聊天正文，
且不注册给 LLM。

VRChat 当前没有向 Udon 或 OSC 暴露其他玩家原生聊天框正文，因此本项目不读取原生
聊天框，也不扫描进程内存或使用 OCR。0.5.4 改用世界内自定义输入：面板平时完全
隐藏；桌面端按 `T` 呼出跟随式输入条，`Enter` 发送、`Esc` 关闭。`T` 可在
`NekoNpcChatInput.openKey` 中修改；没有复用 VRChat 原生 `Y`，避免两个聊天框同时弹出。输入框打开时会临时锁定本地玩家移动，关闭、发送或对象禁用时立即释放。提交经带参数 Udon
网络事件仅送往 NPC 当前 owner/driver，再产生唯一 `player.chat_submit` 日志；宿主验证
session、slot、长度、重复和每玩家 2 秒冷却后，先以 `visibility=[]`、
`ai_behavior="read"` 隐藏完整世界上下文，再以 `visibility=["chat"]`、
`ai_behavior="respond"` 显示玩家原话并触发唯一一次主 LLM 回答。普通动作、触摸及错误事件不能进入该通道；主动开场使用独立的 `YUI_PROACTIVE_CHAT_REQUEST`，不会伪造玩家发言。

安装或更新正式世界 UI 使用 `NEKO/YUI Formal/Chat Input/Install Or Update`，随后运行
`NEKO/YUI Formal/Chat Input/Validate`。宿主入口 `yui_player_chat_status` 只显示就绪、
计数和脱敏错误，不保存或回显玩家正文。

## Unity 源码

`unity/Assets/NEKO/` 保存与本 Python 核心配套的 YUI
UdonSharp 源码及稳定 `.meta`。它只包含闭环所需的 NPC、EyeCam 和生成器脚本，
不包含旧玩家雷达、实验 Camera Dolly、场景文件或测试场专用对象。

导入现有 Unity 工程时，保留目录结构和 `.meta`：

```powershell
$YuiProject = "<path-to-yui_npc_controller>"
$UnityProject = "<path-to-unity-project>"
$UnityAssets = Join-Path $UnityProject "Assets"

Copy-Item -LiteralPath (Join-Path $YuiProject "unity\Assets\NEKO") `
  -Destination $UnityAssets -Recurse -Force
Copy-Item -LiteralPath (Join-Path $YuiProject "unity\Assets\NEKO.meta") `
  -Destination (Join-Path $UnityAssets "NEKO.meta") -Force
```

模型、材质、Animator 资产、NavMesh 和场景引用仍由目标世界维护，不能用测试场
场景覆盖。导入后必须让 Unity 完成 UdonSharp 编译，再按协议验收流程测试。
火柴盒验收场可在 Unity 菜单选择
`NEKO > YUI NPC > 3 Configure Matchbox v1.3 + Validate`；该操作生成上下层、
楼梯、中央障碍、三个 Region 定位体积和语义目录，并静态验证完整 NavMesh 路径与绕行圆周。

## 连续路线（仅宿主/测试）

`runtime.host_route.YuiContinuousRouteRunner` 只是 v1.1 宿主回归工具，用冻结协议内的多条 `GOTO_XZ`
编排连续坐标路线：NPC 进入中间点预切半径时发送下一点，前一操作必须按规范回报
`cancelled/replaced`，最后一段才回报 `succeeded`。它会记录每次交接的
`npc.state.speed`；没有终态证据时返回 `unknown` 并要求快照取证。该接口不注册为
LLM 工具，且仍受 `free_coordinate_navigation`、activity bounds 和 capability 门控。
v1.2 `npc.orbit` 不使用该路线器，只发送一条 `ORBIT_ENTITY`，连续切点在 Unity 同一 operation 内完成。

## 独立 MCP

首次在源码环境中使用标准可编辑安装注册正式包名（不会安装或升级依赖）：

```powershell
$YuiProject = "<path-to-yui_npc_controller>"
$Python311 = "<path-to-python-3.11.exe>"
& $Python311 -m pip install --no-deps -e $YuiProject
```

不带 `--connect` 启动时只跟随日志且工具列表为空：

```powershell
$YuiProject = "<path-to-yui_npc_controller>"
$Python311 = "<path-to-python-3.11.exe>"
Set-Location (Split-Path $YuiProject -Parent)
& $Python311 -m yui_npc_controller.mcp_server
```

操作者明确连接；连接完成后模型可直接操作地图 NPC：

```powershell
& $Python311 -m yui_npc_controller.mcp_server `
  --connect --midi NEKO_MIDI `
  --free-coordinate-navigation --enable-wander-tool
```

MCP 使用一行一个 JSON-RPC 消息的 stdio 传输，不引入额外运行依赖。

## 测试

聊天诊断日志使用 `YUI_DIAG` 前缀，写入宿主插件日志，并在 `yui_status.diagnostics`
保留最近 64 条。`runtime_id` 区分插件启动实例；`session/event_session/submit_seq`
关联玩家提交，`reply_serial/transfer_sequence` 关联已读到的回复与世界文本。
`chat.host_submit_result=accepted` 仅表示宿主接受提交，不能当作主模型回复完成。
依次检查 `chat.received → chat.validated → chat.host_submit_result → reply.queued →
reply.display_result → world.event(npc.text_cleared)`；`chat.rejected.reason` 给出拒绝原因。
会话重连记录 `connection.attempt/result/closed/world_state/ack_failed`。重复状态最多
每 30 秒记录一次，变化立即记录；不输出聊天正文、玩家姓名、密钥或异常响应正文。

```powershell
$YuiProject = "<path-to-yui_npc_controller>"
$Python311 = "<path-to-python-3.11.exe>"
Set-Location (Split-Path $YuiProject -Parent)
& $Python311 -m pytest -q $YuiProject
```

测试同时校验 v1.1 冻结常量和 82 条向量、v1.2/v1.3 扩展向量、编码器、日志投影、安全门、行为图、动态工具面、
MCP 工具隔离和本地单驱动锁。Unity/VRChat 行为仍需使用相同向量做 Editor、
ClientSim 和真实双客户端验收。


## ARDY 可选动作系统

接入源码、独立动作服务和 Unity 执行桥已合入。ARDY 默认关闭，字幕不驱动身体动作。
启动方式、计划和协议文档见 [ARDY 文档索引](Docs/ARDY/README.md)；旧迁移结果归入历史记录。


### 换机部署状态（2026-09-13）

普通YUI插件无需ARDY模型；按上方宿主CLI流程重新sync依赖、build与install，配置新机器的NEKO_MIDI端口、世界及日志位置。不要复制旧虚拟环境或个人令牌。profiles.toml不再启用test配置；使用通用默认配置，默认日志路径留空并自动跟随VRChat日志。profiles/test.toml仅留在本机，不参与Git提交或安装包。

可选ARDY后台需要单独准备轻量接入包，并由使用者下载官方ARDY源码及模型，详见[轻量接入说明](Docs/ARDY/PORTABLE.md)。ARDY补丁和应用工具已纳入 [integrations/ardy-patches](integrations/ardy-patches/README.md)。完整portable安装工具和图重放运行工具仍由接入包提供，仅克隆仓库不等于安装完整后端。

本仓库 integrations/start-motion-service.ps1 已转发至接入包启动器；源码仓库开发调试用 -KitRoot 指定接入包目录。日常使用直接双击接入包的 启动NPC动作.cmd。验收脚本使用指定接入包的环境和当前验收Python，模型导入工具通过Unity文件选择器获取本机模型包。XPU在本机换目录、新环境生成验证通过，CUDA与第二台电脑尚未实测。Unity集成文件也不等于完整家园工程，需原工程及对应SDK。

Git只保留人工验收报告；原始证据和逐次result/summary/manifest留在本地。宿主构建规则单独排除Docs/ARDY等开发资料，安装包仅用于运行；忽略规则不会删除本地证据。


当前ARDY接入仍依赖相对官方基线的补丁，不支持直接换成未经修改的官方安装。轻量包会应用这些补丁；XPU图重放、合并NF4文本编码与连续窗口处理包含本地改动。CUDA关闭图重放也未完成纯官方版验证，不能据此宣称零修改兼容。

ARDY日常启动：双击接入工具包的 `启动NPC动作.cmd`，等待就绪，在宿主插件面板启用ARDY并保存、重载。无需令牌或宿主命令行；宿主与动作服务运行在同一台电脑。

聊天动作：开启自主生活、独立意图模型、聊天记忆辅助动作及ARDY后，新聊天快照触发原地情绪动作更新；聊天中通常每10秒刷新一次；旧请求未返回时最多等待30秒再刷新，避免周期刷新反复丢弃在途结果。模型请求保持单并发、待发队列只留最新上下文，最短请求间隔2秒。ARDY持续执行最新意图，不绑定字幕和语音时长。30秒未收到有效意图时回退基础姿态；暂停、离开聊天或显式接管后旧结果失效。实际延迟还包括聊天落盘轮询、模型推理和动作提交。

插件面板的世界连接与模型连接可设置MIDI、控制码、VRChat/Unity日志、ARDY地址、模型API/名称/密钥及请求参数。保存后重载配置生效。API Key存于宿主本机配置（不是加密凭据库），面板不回显；留空保留，勾选清除后保存可移除。不要提交或分享包含密钥的本机配置。
