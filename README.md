# YUI NPC Controller

独立实现 YUI NPC v1.1/v1.2 Python 后端、N.E.K.O 插件和 stdio MCP。v1.1 字节级冻结，v1.2 只在世界明确发布 capability 后增加语义地图和后台行为图。协议事实只来自
`Docs/Protocols/`，不会导入 AnyDance、YOLO、视觉导航或旧 `BackendService`。

## 控制边界

- 启动只跟随 VRChat 日志，不打开 MIDI、不连接；仍由操作者显式调用宿主连接入口。
- 这是只在地图 NavMesh/activity bounds 内活动的世界 NPC。连接完成 DISCOVER 后，
  宿主内部可靠切入 `external`，模型不需要也看不到 `npc.arm`。
- `CLEAR_ESTOP` 只存在于宿主入口 `yui_clear_estop`，不是 LLM/MCP 工具。
- session 重建、watchdog、ESTOP、driver 离开或 owner 错误仍会立即停止控制；恢复时
  由宿主重新连接，或由操作者执行 `yui_clear_estop`。
- `free_coordinate_navigation`、玩家姓名和可选 `npc.wander` 均默认关闭。
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

`npc.execute_plan` 只接受冻结的受限 JSON 图，由 Python 后台执行器持有
`plan_id` 和 operation 证据；不存在 `npc.plan_step`。`npc.observe` 最多投影
8 项附近语义事实，v1.2 不向模型返回绝对世界坐标。

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

宿主入口为 `yui_connect`、`yui_clear_estop`、
`yui_disconnect`、`yui_status` 和 `yui_reload_config`；它们不会注册为 LLM 工具。
同一宿主进程内重复调用 `yui_connect` 会复用已经完整握手的 session，不重放
DISCOVER 目录；需要重新声明 ownership 时应先 `yui_disconnect` 再连接。目录只在
全部页面到齐后重建一次动态工具面，高频 `npc.state`/心跳 ACK 不会重复注册工具。

`npc.navigate`、`npc.orbit`、`npc.explore` 和 `npc.execute_plan` 都只等待命令 ACK，
随后立即返回 `accepted + plan_id`；长行为在后台继续执行。宿主不得把工具调用本身
阻塞到动作结束，应通过 `npc.plan_status` 读取终态证据。

## Unity 源码

`unity/Assets/NEKO/` 保存与本 Python 核心配套、已在火柴盒工程验证的 YUI
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
`NEKO > YUI NPC > 3 Configure Matchbox v1.2 + Validate`；该操作生成上下层、
楼梯、中央障碍和语义目录，并静态验证完整 NavMesh 路径与绕行圆周。

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

测试同时校验 v1.1 冻结常量和 82 条向量、v1.2 扩展向量、编码器、日志投影、安全门、行为图、动态工具面、
MCP 工具隔离和本地单驱动锁。Unity/VRChat 行为仍需使用相同向量做 Editor、
ClientSim 和真实双客户端验收。
