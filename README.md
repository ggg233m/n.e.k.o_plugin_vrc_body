# YUI NPC Controller

独立实现 YUI NPC v1.1/v1.2/v1.3 Python 后端、N.E.K.O 插件和 stdio MCP。v1.1、v1.2 保持冻结，v1.3 只在世界明确发布 capability 后增加语义定位、相对移动和连续区域探索。协议事实只来自
`Docs/Protocols/`，不会导入 AnyDance、YOLO、视觉导航或旧 `BackendService`。
0.5.0 增加可选的宿主常驻自主循环；0.5.1 增加独立意图模型和连续生活片段；
0.5.3 默认只读当前角色落盘的近期聊天：既用于生成更自然的观察、转身、附近闲逛与兴趣回访，也把主 LLM 新完成的回答分页显示到 NPC 头顶。0.5.4 增加每位玩家本地独立、按键唤起且跟随视野的世界内输入 UI；只有玩家显式发送才触发宿主主 LLM。0.5.5 修复重复相同问答时落盘修订未变化的问题。0.5.6 补读宿主已有的主动回复记录，使世界输入触发的主 LLM 回答无需等待落盘即可显示在头顶。0.5.7 将头顶回答改为 15–30 秒自适应阅读时长，并把正式世界气泡限制为 3.8 米宽的自动换行布局。0.5.8 修复回车结束编辑未绑定发送事件的问题，并在输入期间锁定本地玩家移动、退出时可靠释放。0.5.9 修复 ClientSim/世界重启后宿主仍复用旧 session、导致输入提交被拒绝的问题；检测到新世界或 `not_handshaken` 后会清理旧控制链路并自动重新 DISCOVER。0.5.10 在输入期间额外锁定姿态并禁用跳跃，头顶气泡改用动态中文字体和头骨锚定，避免缺字与随视角漂移。
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
异步生成生活片段；失败或超时不会中断规则循环。YUI 不再发送任何
`ai_behavior="respond"` 自主消息。第一阶段的 `AutonomyStimulusProvider` 固定使用
无操作实现，不采图。

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
填写完整的 chat/completions 地址即可，必须是 `https://`。密钥只允许设置在 `TEST_API`
环境变量中；设置后必须重启 N.E.K.O。人工入口
`yui_autonomy_intent_probe` 只验证认证、模型、JSON schema 与脱敏记忆状态，不会
显示聊天正文，也不会应用返回的活动。

## 主对话头顶显示（0.5.7）

`yui.chat_bridge` 默认开启。普通用户回合仍由独立只读提供器轮询当前角色的
`memory/<当前猫娘>/recent.json`；世界输入通过 `push_message(..., ai_behavior="respond")`
形成的主动回合不会立即落盘，因此只读补充宿主现有 `conversations` 存储里的
`proactive_reply`。没有新增总线字段，也没有修改宿主核心。每个新完成的主 LLM
回答通过既有 `TEXT_UTF8` 原子事务显示在 NPC 的 `BubbleText`。超过单笔 384 UTF-8
字节的回答会安全分页，默认最多 4 页。每页至少保留 15 秒，并按可见字符数
自动延长到最多 30 秒；世界内 `BubbleText` 使用 3.8 米最大宽度和自动换行，
避免长回答横跨场景。

这条投影不调用 `push_message`，不会再次触发主模型，也不会触发 TTS。人工入口
`yui_chat_bridge_status` 只报告文件、修订、主动回复记录计数、排队页数和脱敏错误，不显示聊天正文，
且不注册给 LLM。

VRChat 当前没有向 Udon 或 OSC 暴露其他玩家原生聊天框正文，因此本项目不读取原生
聊天框，也不扫描进程内存或使用 OCR。0.5.4 改用世界内自定义输入：面板平时完全
隐藏；桌面端按 `T` 呼出跟随式输入条，`Enter` 发送、`Esc` 关闭。`T` 可在
`NekoNpcChatInput.openKey` 中修改；没有复用 VRChat 原生 `Y`，避免两个聊天框同时弹出。输入框打开时会临时锁定本地玩家移动，关闭、发送或对象禁用时立即释放。提交经带参数 Udon
网络事件仅送往 NPC 当前 owner/driver，再产生唯一 `player.chat_submit` 日志；宿主验证
session、slot、长度、重复和每玩家 2 秒冷却后，以 `visibility=["chat"]`、
`ai_behavior="respond"` 触发主 LLM。自主、触摸、社交及错误事件仍不能进入该通道。

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

```powershell
$YuiProject = "<path-to-yui_npc_controller>"
$Python311 = "<path-to-python-3.11.exe>"
Set-Location (Split-Path $YuiProject -Parent)
& $Python311 -m pytest -q $YuiProject
```

测试同时校验 v1.1 冻结常量和 82 条向量、v1.2/v1.3 扩展向量、编码器、日志投影、安全门、行为图、动态工具面、
MCP 工具隔离和本地单驱动锁。Unity/VRChat 行为仍需使用相同向量做 Editor、
ClientSim 和真实双客户端验收。
