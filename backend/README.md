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
  它不进入 AnyaDance 的 60 Hz 控制线程，未来的 LLM 只需要生成同一计划格式。
- `adapters.py` 是唯一的宿主项目集成接缝，负责把后端映射到当前项目的调度器、
  VMC、OSC、遥测、配置和动作片段库。
- `vision.py` 与 `world_state.py` 是无模型依赖的感知状态基础模块；其中
  `FrameSource`/`FrameDetector`/`VisionWorker` 组成后端内采集接缝。当前不内置
  YOLO/MediaPipe 模型，未来 detector 只需实现 `status()` 与
  `observe(frame, now=...)`，再通过 `BackendService.attach_vision()` 注入。
  worker 使用有界 latest-frame 队列，掉帧优先于堆积，不进入 60 Hz 控制线程。
- `MssFrameSource` 是可选的纯 mss 桌面采集器，依赖懒加载；没有 mss/Pillow/numpy
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

HTTP 端点对应为 `GET /cognition`、`POST /cognition/plan` 和
`POST /cognition/feedback`。计划只做校验和记录，不会绕过现有安全调度器直接执行；
执行仍必须经过 `/action` 或插件工具。

`POST /world/ingest` 仍支持完整状态回包；高频外部发布者可设置
`"ack_only": true`，只得到 source/frame_id、实体/事件计数和 observation_count，
避免重复传输完整世界快照。后端内置 worker 直接写入 `WorldStateStore`，不经过 HTTP。

外部 detector 应为每个持续跟踪目标使用稳定 ID，推荐调用
`stable_entity_id(source, label, track_id)` 生成
`{source}:{label}:{track_id}`；不要在未启用 tracking 时按帧生成随机 UUID。

配置段默认关闭：

```toml
[vision]
enabled = false
source = "none" # none / mss / external
interval_ms = 100
queue_size = 1
```

配置只描述 worker，不下载或加载模型。YOLO 后端交付后再把它作为外部
`FrameDetector` 接入；当前仓库不会安装 Ultralytics、Torch 或模型权重。

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
      "entity_id": "yolo:cup:7",
      "source": "yolo",
      "min_confidence": 0.8,
      "max_age_ms": 500
    }
  ]
}
```

失败响应包含 `reason_code: "world_precondition_failed"`、`replan_required: true`
及逐项 `precondition_check.failures`。门禁只读取世界状态快照，不进入或阻塞 60 Hz
身体调度线程。多步计划在创建时只预检当前第一步；后续每一步仍须在提交执行时携带
自身的 `preconditions`，以免把前一步尚未产生的世界变化误判为失败。
