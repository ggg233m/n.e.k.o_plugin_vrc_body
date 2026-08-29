# YUI NPC Python 后端兼容层 v1.1

本文说明仓库中对应《YUI NPC Unity 接口规范 v1.1》的 Python 后端实现。它只负责宿主侧协议，不创建 Unity、VCC、Udon 或场景资产。

## 当前实现范围

- `runtime/yui_protocol.py`：冻结常量、逐命令参数约束、CRC16/CCITT-FALSE、请求哈希、可靠命令帧、上半身流、UTF-8 文本事务和量化编解码。命令约束从冻结 JSON 缓存读取，非法参数不会进入 MIDI 发送层。
- `runtime/yui_session.py`：解析 `[NEKO]` JSON 行，维护会话、catalog、玩家、NPC、语音、文本、操作生命周期、日志 gap 和 ACK 关联。
- `runtime/yui_transport.py`：全局序号、单普通未决命令、2 秒 ACK、原帧一次重发、5 秒 unknown、独立心跳/ESTOP 以及带宽限流。
- `runtime/yui_adapter.py`：只向上层暴露 `semantic_key`、`anchor_key` 和 `player_slot`；默认关闭自由坐标导航，且不会自动 arm。
- `runtime/yui_log.py`：只读跟随最新 VRChat `output_log`，只吸收带 `[NEKO]` 标记且已经换行提交的协议行，并处理残行补写、轮转和截断。
- `__init__.py`：N.E.K.O 生命周期、配置、连接和动态 LLM 工具入口；不包含协议编码细节。

冻结测试向量由 `tests/test_yui_protocol.py`、`tests/test_yui_session.py` 和 `tests/test_yui_adapter.py` 覆盖。实现不会把模拟执行结果当成世界 ACK；上行日志仍是状态与结果的唯一权威来源。

## 运行模式隔离

YUI LLM 控制与现有 AnyDance/YOLO 控制是两套互斥方案，不是可以叠加的数据源或降级路径：

- `yui_llm` 模式只启动 `yui_protocol/session/transport/adapter/log` 及其独立 LLM 工具运行时。
- `anydance_yolo` 模式继续使用现有身体、视觉、检测和 `BackendService` 链路。
- 同一宿主进程或角色控制会话只能选择其中一种模式；两个插件的 manifest 已声明双向 `conflicts = true`，同时启用会由宿主拒绝。
- YUI 不导入 `BackendService`、AnyDance scheduler、vision、local perception、YOLO detector 或其 world state。
- 两种模式之间不共享 observation、目标 ID、动作队列、arm 状态或失败回退；切换模式必须先完整停止当前控制器，再建立新 session。

YUI 使用独立的 N.E.K.O 插件入口和动态 LLM 工具，不注册到现有 AnyDance/YOLO HTTP/IPC 工具面。

## 最小接入

实时 MIDI 输出依赖 `mido` 与 `python-rtmidi`，协议编解码和日志解析本身不依赖它们。按 N.E.K.O 插件规范同步到插件私有 `vendor/`：

```powershell
neko-plugin sync <yui_npc_controller目录>
```

宿主应先启动日志跟随器，再建立 MIDI 发送端和语义适配器：

```python
from yui_npc_controller.runtime.yui_adapter import YuiSemanticAdapter
from yui_npc_controller.runtime.yui_log import YuiOutputLogTailer
from yui_npc_controller.runtime.yui_session import YuiSessionState
from yui_npc_controller.runtime.yui_transport import MidoOutputSink, YuiReliableTransport

session = YuiSessionState()
logs = YuiOutputLogTailer(session)
sink = MidoOutputSink("NEKO_MIDI")
transport = YuiReliableTransport(sink, session)
adapter = YuiSemanticAdapter(transport, session)

logs.start()
result = adapter.connect(claim_code=1234)
```

`connect()` 会阻塞等待对应 DISCOVER ACK；成功后才启动独立心跳。要进入 external，宿主必须先完成真实的人工授权，再显式调用：

```python
adapter.authorize_arm(True)
arm_result = adapter.arm()
```

不得用 LLM 自己的决定代替 `authorize_arm(True)`。driver 离开、session 重建、watchdog 或 ESTOP 都会撤销宿主授权。

退出时应按持有顺序关闭资源：

```python
transport.stop_heartbeat()
logs.stop()
transport.close()
```

## 适配器安全边界

- `go_to()` 只接受世界发布的 `anchor_key`；`go_to_xyz()` 默认禁用。
- `follow()` 串行发送 `SET_TARGET` 和 `SET_MODE`，不会把两条可靠命令并发在途。
- `operation_lifecycle` 未发布时，长操作工具不会被当成可向 LLM 暴露的能力，`active_ops` 固定为空且非权威。
- `observe()` 默认不返回玩家显示名；只有宿主明确传入 `include_player_names=True` 才会包含。
- ACK 丢失且原帧重发后仍无匹配结果时返回 `unknown`，并要求重建 session；不会猜测成功。
- ESTOP 不占普通命令槽，可走任意 MIDI channel；清除 ESTOP 的人工安全流程不由此适配器自动执行。

## N.E.K.O 工具暴露

- 插件启动时只跟随日志，不会自动连接或发送 MIDI。
- `yui_connect` 完成 DISCOVER 后，先动态注册 `yui_npc_observe` 与 `yui_npc_estop`。
- 只有实际配置把 `host_arm_authorized` 设为 `true`，且当前世界处于 `safe_idle`，才注册 `yui_npc_arm`。
- arm 成功且世界发布相应 capability 后，才注册导航、跟随、动作、文本和停止工具。
- 工具参数中的 anchor/action 枚举来自当前世界 catalog；自由坐标工具仍需额外配置开关。

## 尚未包含

- 没有修改现有 `BackendService` 的 HTTP/IPC API；按互斥约束，YUI 也不会接入该 AnyDance/YOLO 运行入口。
- 没有安装或配置虚拟 MIDI 设备，也没有更改 VRChat 本地设置。
- 没有在真实双客户端世界中完成互操作测试；目前验证边界是冻结常量、82 个协议向量及 Python 单元测试。
