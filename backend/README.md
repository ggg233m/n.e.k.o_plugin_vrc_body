# 独立后端

本目录是身体控制器的粗粒度运行时边界，刻意设计为一个后端进程，而不是多个
微服务。

## 内容

- `service.py` 持有长期运行的调度器、AnyaDance UDP 输出、VMC 待机中转、
  VRChat OSC、驱动遥测、动作片段加载和世界状态。
- `process.py` 通过带认证的本机回环 HTTP 暴露服务。它支持 JSON/TOML 配置文件，
  可用 `--offline`/`--dry-run` 关闭 VMC、VRChat OSC、驱动遥测和 AnyaDance UDP，
  直接启动开发后端；完全省略配置时也默认使用该安全模式。
- `client.py` 是插件侧的轻量 IPC 客户端和兼容代理层。
- `debug_cli.py` 是不依赖 SDK 的实时调试命令行，可以读取状态、注入世界观测、
  提交动作、创建高层计划、注入反馈和停止后端。
- `cognition.py` 保存有界的观测、状态新鲜度/置信度、严格计划 JSON 和执行反馈；
  它不进入 AnyaDance 的 120 Hz 控制线程，主 LLM 通过世界 delta 桥接接收主动摘要。
- `autonomy.py` 只管理手动授权、会话 TTL、世界新鲜度和急停释放；它不会绕过
  调度器盲目移动，也不会自动执行好友、邀请或世界切换。
- `adapters.py` 是唯一的宿主项目集成接缝，负责把后端映射到当前项目的调度器、
  VMC、OSC、遥测、配置和动作片段库。
- `vision.py` 与 `world_state.py` 是无模型依赖的感知状态基础模块；其中
  `FrameSource`/`FrameDetector`/`VisionWorker` 组成后端内采集接缝。`DesktopMirrorFrameSource`
  会优先尝试 DXcam、再回退 MSS；`OpenVinoLocalDetector` 和
  `OpenAICompatibleSemanticBackend` 是可插拔的 YOLOX/depth/OCR/VLM 接缝，不会在缺少
  依赖时伪造检测。外部 detector 只需实现 `status()` 与
  `observe(frame, now=...)`，再通过 `BackendService.attach_vision()` 注入。
  worker 使用有界 latest-frame 队列，掉帧优先于堆积，不进入 120 Hz 控制线程。
- `MssFrameSource`/`DxcamFrameSource` 是可选桌面采集器，依赖懒加载；没有捕获库或
  Pillow/numpy
  时后端仍可启动并报告 `available=false`。不要从兄弟插件项目导入截图服务，避免
  把 SDK 依赖带回独立后端。

要将此后端适配到其他项目，请复制本目录，替换 `adapters.py` 中的调度器、VMC、
OSC 和遥测映射，并保持进程协议与世界状态模块不变。进程入口会动态发现所在的
项目包，不会导入 N.E.K.O SDK。

## 实时开发

从项目根目录启动一个不连接宿主的开发后端。动作仍会进入调度器，但 UDP 只会
进入 dry-run 计数器，不会发到网络：

```powershell
python backend/process.py --config-file plugin.toml --offline --port 48912 --token dev
```

另一个终端可以直接查看或驱动它：

```powershell
python backend/debug_cli.py --port 48912 --token dev snapshot
python backend/debug_cli.py --port 48912 --token dev ingest --json '{"source":"yolo","entities":[{"id":"button","label":"button","confidence":0.92}]}'
python backend/debug_cli.py --port 48912 --token dev plan --json '{"goal":"举右手","action":"arm_pose","params":{"side":"right","elevation_deg":110}}'
python backend/debug_cli.py --port 48912 --token dev cognition
python backend/debug_cli.py --port 48912 --token dev feedback --json '{"type":"world_changed","data":{"reason":"new_avatar"}}'
python backend/debug_cli.py --port 48912 --token dev action --kind enable --json '{}'
python backend/debug_cli.py --port 48912 --token dev shutdown
```

在没有 N.E.K.O 宿主时，后端也可以直接作为临时宿主使用。`debug_cli.py` 提供
VRChat OSC 的轻量控制入口；它不会自动启用身体输出，第一次使用仍需显式发送
`action --kind enable`：

```powershell
python backend/debug_cli.py --port 48912 --token dev action --kind enable --json '{}'
python backend/debug_cli.py --port 48912 --token dev locomotion --vertical 0.35 --duration-ms 600
python backend/debug_cli.py --port 48912 --token dev turn --horizontal -0.5 --duration-ms 500
python backend/debug_cli.py --port 48912 --token dev input --action grab --side right --hold-ms 100
python backend/debug_cli.py --port 48912 --token dev parameter --name NEKO_Action --value 1
python backend/debug_cli.py --port 48912 --token dev chatbox --text '你好'
python backend/debug_cli.py --port 48912 --token dev stop-movement
python backend/debug_cli.py --port 48912 --token dev controller --side left --control stick --y 0.35 --duration-ms 600
python backend/debug_cli.py --port 48912 --token dev autonomy-arm
python backend/debug_cli.py --port 48912 --token dev autonomy-goal --goal "探索附近的入口"
python backend/debug_cli.py --port 48912 --token dev autonomy-stop
```

需要同时更新多个控制量时，可使用最多 8 条命令的批量接口；同一轴在批次内以后
出现的值为准，轴命令失败时会自动归零：

```powershell
python backend/debug_cli.py --port 48912 --token dev batch --json '{"commands":[{"kind":"locomotion","vertical":0.35,"horizontal":0.0,"duration_ms":600},{"kind":"turn","horizontal":-0.4,"duration_ms":400}]}'
```

`parameter --value` 使用 JSON 标量解析，因此 `true`、`1` 和 `0.5` 会分别作为
Bool、Int 和 Float 发送；`chatbox` 默认立即显示，使用 `--deferred` 可改为仅在
输入时显示。所有移动轴都有时限，结束时会自动归零；`stop-movement` 和
`cancel-inputs` 可用于手动释放状态。

## 低延迟路径

Hosted 插件的动作、移动和 OSC 调用使用后端的持久 HTTP/1.1 控制连接，避免每次
命令重新建立回环 TCP 连接；后端快照/意识状态从 120 Hz 身体控制线程中降到约 10 Hz
发布，不会让状态深拷贝阻塞姿态帧。`debug_cli.py` 的单次命令仍适合人工调试，不
适合高频循环；高频调用可以使用常驻 JSON-lines 控制会话：

```powershell
python backend/debug_cli.py --port 48912 --token dev shell
```

然后逐行发送 JSON（每行都会立即返回一行 JSON）：

```json
{"path":"/osc/locomotion","payload":{"vertical":0.35,"horizontal":0.0,"duration_ms":600}}
{"path":"/osc/turn","payload":{"horizontal":-0.4,"duration_ms":400}}
{"path":"/osc/stop_movement","payload":{}}
```

输入 `quit` 或 `exit` 结束会话。

AnyaDance 虚拟 Index 输入使用 `POST /input/axes`、`POST /input/button` 和
`POST /input/release`；同一 UDP 发送线程将控制器叠加到 VMC/动作帧，旧输入采用
latest-wins，轴和按钮到期自动释放。`GET /autonomy`、`POST /autonomy/arm`、
`POST /autonomy/disarm`、`POST /autonomy/goal`、`POST /autonomy/stop` 管理手动授权；
`GET /world/delta?after_revision=N&wait_ms=250` 用 revision 长轮询世界变化。

HTTP 端点对应为 `GET /cognition`、`POST /cognition/plan` 和
`POST /cognition/feedback`。计划只做校验和记录，不会绕过现有安全调度器直接执行；
执行仍必须经过 `/action` 或插件工具。

`POST /world/ingest` 仍支持完整状态回包；高频外部发布者可设置
`"ack_only": true`，只得到 source/frame_id、实体/事件计数和 observation_count，
以及 revision、完整的本批次删除 ID 和删除计数，避免重复传输完整世界快照。
`revision` 与这批变更对应，便于调用方确认删除确认和快照属于同一提交。后端内置
worker 直接写入 `WorldStateStore`，不经过 HTTP。

生命周期来源可以在同一批次提交明确的实体删除：

```json
{
  "source": "vrchat_log",
  "events": [{
    "type": "player_left",
    "target_id": "vrchat:player:usr_123",
    "confidence": 1.0
  }],
  "remove_entity_ids": ["vrchat:player:usr_123"],
  "remove_source": "vrchat_log",
  "ack_only": true
}
```

删除只接受发布者明确给出的 ID；普通检测漏帧不会自动删除实体。带
`remove_entity_ids` 时必须同时提供非空 `remove_source`，它是删除命令的来源所有者
校验，不是可省略的提示字段；`remove_source` 不能单独触发整来源清理。实体
`source` 数组的第一项是 canonical owner，后续项只能表示辅助证据。删除和离开
事件在同一锁区间提交，响应中的 `removed_entity_ids` 和 `revision` 可用于确认
没有幽灵玩家。若世界日志
适配器运行在后端进程内，换世界时可调用
`WorldStateStore.remove_entities_by_source("vrchat_log", prefix="vrchat:player:")`；
若它是独立进程，则应维护自己的玩家 ID 集合并逐个提交删除，不能通过 HTTP 请求
一个无范围的“清空来源”操作。
水位表有界保留（默认至少覆盖一个完整删除批次）；若适配器存在很长的离线队列，
应提高 `WorldStateStore(lifecycle_watermark_limit=...)`，不要依赖过期的旧帧恢复实体。
来源/换世界清理还会移除该来源对应范围内的旧事件，避免上一世界的
`player_joined`/Contact 类事件继续满足 `event_recent` 门禁。

外部 detector 应为每个持续跟踪目标使用稳定 ID。新 detector 应显式调用
`stable_track_entity_id(source, track_id)` 生成不受类别抖动影响的
`{source}:track:{track_id}`；状态层对缺少 `id` 的旧调用仍兼容使用
`stable_entity_id(source, label, track_id)`，但它只适合类别不可变的来源。不要在
未启用 tracking 时按帧生成随机 UUID。

直接接入外部世界日志前必须做字段翻译：日志适配器要把事件映射为 `type`、稳定的
`target_id`、canonical `source`，并在 `player_left` 同批提供 `remove_entity_ids`；
旧版 `{event, detail, at_unix}` 结构不能直接提交，否则会被当作 `unknown` 事件，
也不会触发玩家删除。

配置段默认关闭；随插件部署的 `plugin.toml` 将 AnyaDance 发送频率设为 120 Hz，并
打开虚拟控制器主路由（OSC 仍是回退）：

```toml
[vision]
enabled = false
source = "none" # none / mss / external
capture = "desktop_mirror"
local_backend = "openvino"
semantic_backend = "openai_compatible"
semantic_max_per_minute = 30
interval_ms = 100
queue_size = 1
lifecycle_watermark_limit = 4096
```

配置只描述 worker，不下载或加载模型。OpenVINO 模型包、VLM endpoint 和 API key
由部署环境提供（VLM endpoint 可用 `VRC_VLM_ENDPOINT`、模型用 `VRC_VLM_MODEL`），
没有依赖时状态会明确标记为 unavailable。

当前核心不内置 Contact 事件总线、Autonomy 反射规划、浏览器视觉桥或动作生成
模型，也不会为这些未启用功能在插件里维护隐式状态。将来接入时应各自实现独立
适配器/sidecar，并通过配置显式启用；不能改变身体调度器和世界状态协议的默认行为。

## 世界状态动作门禁

计划步骤与 `POST /action` 都可以显式声明 `preconditions`。未声明时保持原有动作
行为；声明后，认知层会在规划第一步和实际提交前分别检查一次。支持的条件为：

- `world_available`：存在未过期的世界观测，可用 `max_age_ms` 进一步限制新鲜度；
- `entity_visible`：按稳定 `entity_id` 检查可见性，并可约束 `source`、`label`、
  `state`、`min_confidence` 和 `max_age_ms`；
- `event_recent`：按 `event_type` 检查近期事件，并可约束 `target_id`、`source`、
  `min_confidence` 和 `max_age_ms`。

`entity_visible` 和 `event_recent` 未显式填写 `min_confidence` 时默认使用 `0.5`；
`event_recent` 未填写 `max_age_ms` 时默认使用 `2000` 毫秒。实体过期仍以各实体自己的
`ttl_s` 为准。紧急/安全控制动作 `stop`、`disable`、`reset`、`cancel` 永远不会被
视觉门禁阻止；若错误地携带了前置条件，响应会标记
`precondition_check.bypassed: true`。

检测器发布的实体 ID 应采用 `{source}:{track_id}` 或
`{source}:{class}:{track_id}`，并在连续帧之间保持稳定。一个带视觉门禁的动作请求示例：

```json
{
  "kind": "reach_and_grab",
  "params": {"side": "right"},
  "preconditions": [
    {
      "kind": "entity_visible",
      "entity_id": "yolo:track:7",
      "source": "yolo",
      "min_confidence": 0.8,
      "max_age_ms": 500
    }
  ]
}
```

失败响应包含 `reason_code: "world_precondition_failed"`、`replan_required: true`
及逐项 `precondition_check.failures`。门禁只读取世界状态快照，不进入或阻塞 120 Hz
身体调度线程。多步计划在创建时只预检当前第一步；后续每一步仍须在提交执行时携带
自身的 `preconditions`，以免把前一步尚未产生的世界变化误判为失败。
