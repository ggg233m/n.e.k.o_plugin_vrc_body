# YUI NPC Controller

独立实现 YUI NPC v1.1 Python 后端、N.E.K.O 插件和 stdio MCP。协议事实只来自
`Docs/Protocols/`，不会导入 AnyDance、YOLO、视觉导航或旧 `BackendService`。

## 安全边界

- 启动只跟随 VRChat 日志，不打开 MIDI、不连接、不授权、不 ARM。
- `DISCOVER` 只建立 session；连接绝不自动调用 `npc.arm`。
- `host_arm_authorized` 是当前 session 的宿主人工门，LLM 无法修改。
- `CLEAR_ESTOP` 只存在于宿主入口 `yui_clear_estop`，不是 LLM/MCP 工具。
- session 重建、watchdog、ESTOP、driver 离开或 owner 错误会撤销当前授权。
- `free_coordinate_navigation`、玩家姓名和可选 `npc.wander` 均默认关闭。
- 本地驱动锁会阻止 N.E.K.O 与独立 MCP 同时打开同一个 MIDI 端口；世界端
  session/driver/ownership 仍是最终权威。

模型工具名固定为：

```text
npc.observe  npc.arm  npc.go_to  npc.go_to_xyz  npc.follow
npc.look_at  npc.act  npc.set_expression  npc.say
npc.stop     npc.estop  npc.wander（可选）
```

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

& $NekoPython -m plugin.neko_plugin_cli check $YuiProject

& $NekoPython -m plugin.neko_plugin_cli sync `
  --python $Python311 `
  $YuiProject

& $NekoPython -m plugin.neko_plugin_cli build `
  $YuiProject `
  --out $PluginPackage

& $NekoPython -m plugin.neko_plugin_cli install `
  $PluginPackage `
  --plugins-root $PluginsRoot
```

`sync` 只更新插件自身的 `vendor/`，不会把插件安装进 N.E.K.O。必须再执行
`build` 与 `install`；安装成功时 CLI 会同时验证包内 payload 哈希。

宿主入口为 `yui_connect`、`yui_authorize_arm`、`yui_clear_estop`、
`yui_disconnect`、`yui_status` 和 `yui_reload_config`；它们不会注册为 LLM 工具。

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

## 连续路线（仅宿主/测试）

`runtime.host_route.YuiContinuousRouteRunner` 用冻结协议内的多条 `GOTO_XZ`
编排连续坐标路线：NPC 进入中间点预切半径时发送下一点，前一操作必须按规范回报
`cancelled/replaced`，最后一段才回报 `succeeded`。它会记录每次交接的
`npc.state.speed`；没有终态证据时返回 `unknown` 并要求快照取证。该接口不注册为
LLM 工具，且仍受 `free_coordinate_navigation`、activity bounds 和 capability 门控。

## 独立 MCP

不带 `--connect` 启动时只跟随日志且工具列表为空：

```powershell
$YuiProject = "<path-to-yui_npc_controller>"
$Python311 = "<path-to-python-3.11.exe>"
Set-Location (Split-Path $YuiProject -Parent)
& $Python311 -m yui_npc_controller.mcp_server
```

操作者明确连接并授权当前 session（仍需 Agent 单独调用 `npc.arm`）：

```powershell
& $Python311 -m yui_npc_controller.mcp_server `
  --connect --midi NEKO_MIDI --host-arm-authorized `
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

测试同时校验冻结常量、82 条向量文档、编码器、日志投影、安全门、动态工具面、
MCP 工具隔离和本地单驱动锁。Unity/VRChat 行为仍需使用相同向量做 Editor、
ClientSim 和真实双客户端验收。
