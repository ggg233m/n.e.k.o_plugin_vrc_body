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
  会自动探测 DXGI 适配器/输出，失败后按物理显示器回退 MSS；DXcam 还会在可用时尝试
  WinRT 后端。每个候选输出的错误会出现在 `/perception` 的 `source.backends` 和
  `candidate_errors` 中，不再把所有失败压缩成一个 BitBlt 错误。配置 `window_title`
  后，`WindowTrackedFrameSource` 会按 `window_track_interval_ms` 重新解析窗口矩形，
  只在矩形真的变化时重建内部采集源（DXcam/MSS 的区域在构造时固定，没有改区域的
  接口）；窗口暂时找不到时保留上一次的矩形，不回退全屏。真正的本地检测器是
  `local_perception.OpenVinoLocalDetector`；`vision.OpenVinoLocalDetector` 只是注入式
  `infer` 的适配壳，自己不加载任何图，其 `status()` 因此不声明模型列表。
  `OpenAICompatibleSemanticBackend` 是可插拔的 VLM 接缝，不会在缺少
  依赖时伪造检测。外部 detector 只需实现 `status()` 与
  `observe(frame, now=...)`，再通过 `BackendService.attach_vision()` 注入。
  worker 使用有界 latest-frame 队列，掉帧优先于堆积，不进入 120 Hz 控制线程。
- `MssFrameSource`/`DxcamFrameSource` 是可选桌面采集器，依赖懒加载；没有捕获库或
  Pillow/numpy
  时后端仍可启动并报告 `available=false`。不要从兄弟插件项目导入截图服务，避免
  把 SDK 依赖带回独立后端。

## 采集环境（真机实测，2026-08-21）

后端必须跑在仓库根目录的 `.venv`（Python 3.11.11）。该环境里
**`onnxruntime` 有、`mss` 和 `cv2` 没有**，因此：

- `DesktopMirrorFrameSource` 名义上是「DXcam 失败回退 MSS」，但在这个环境里
  **实际只有 DXcam 一条腿**——MSS 那一路恒为 `ModuleNotFoundError`。
  排查采集问题时不要假设有回退兜底。
- 系统 python（scoop 3.11.9）能力正好相反：有 `mss`/`cv2`、没有 `onnxruntime`。
  用它跑探针脚本会得到和后端**不一样**的结论，务必用
  `.venv/Scripts/python.exe` 复现。

**不要"顺手补装 mss 恢复双后端"。** 实测过：窗口被拖出屏幕边缘时 MSS 确实能出帧，
但越界的那部分是**纯黑填充**（`mean=0.00 / std=0.00 / unique=1`），等于把假像素喂给
检测器，还会顺带稀释 `apparent_height`。这种"回退成功"比直接失败更危险，正是
「不可违反的约束」第 1 条要防的东西。真正的修法是把窗口矩形夹到虚拟桌面内
（`find_window_region` 已实现，并通过 `window_clamped` / `window_clamped_px` 报告
夹掉了多少）。

⚠️ `process.py` 的 `--offline`/`--dry-run` 会强制 `vision.enabled=False`
（见 `process.py` 中 offline 分支）。用它启动后 `frames_captured` 恒为 0——那是
视觉被关掉，不是采集坏了。验证采集时**不能带 `--offline`**。

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
python backend/debug_cli.py --port 48912 --token dev autonomy-goal --kind explore --goal "寻找 NPC" --selector-json '{"semantic_type":"npc","min_confidence":0.7}' --constraints-json '{"max_scan_turns":16,"max_forward_axis":0.3}' --based-on-revision 42
python backend/debug_cli.py --port 48912 --token dev autonomy-goal --kind explore --target-id "vision:door:1" --goal "探索这个入口"
python backend/debug_cli.py --port 48912 --token dev autonomy-goal --kind follow --target-id "avatar:session:abcd1234:7" --goal "跟随这个玩家"
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

自主目标被接受后，`LocalNavigator` 以约 10 Hz 运行在后端本地。它只接受新鲜、可见、
置信度足够且带方位提示的世界实体；每次只发送 220 ms 左右的受限摇杆脉冲。单帧低置信
或漏检最多复用最后可靠观测 300 ms，宽限到期、观测过期、世界不确定、会话解除或后端
停止时释放输入。它不会调用 LLM、
等待 VLM，也不会在没有目标方位时盲目向前走。`GET /snapshot`、`GET /perception`
和 `GET /autonomy` 的 `navigation` 字段会报告当前决策、脉冲计数和停止原因。
任何朝视觉实体移动的目标都必须提供由 LLM 从最新世界快照中选出的精确
`target_id`；无 ID 的探索请求只停留在观察/规划阶段。本地导航不按 goal 文本、实体标签或
置信度自动选目标，实体暂时消失时也不会回退到同标签目标。目标方位在同侧做 EMA 平滑，明显跨过
中心时立即采用新方向。生产转向发送器把每个新 revision 的比例修正换成“当前 yaw + 修正”
的绝对目标，上一段尚未结束也能安全重定向；同一 revision 仍只发送一次。

OpenVINO 的 person/player/avatar 检测在短期 IoU 跟踪之上启用了会话级外观重识别。
对外实体 ID 是 `avatar:session:<session_token>:<number>`；底层易变的
`openvino:track:<number>` 放在 `attributes.track_entity_id`，重新进入视野后可据此确认
是否发生换轨。导航必须使用实体主 `id`，不要锁定诊断用的 `track_entity_id`。
身份特征只在内存保留，默认 30 分钟，后端重启后重新编号；它不是 VRChat
`usr_`/`avtr_`，相同 Avatar、镜像和大幅换装/视角仍可能产生歧义。

多人场景下，先读取 `world_observe`，再调用 `vrc_vision_frame` 并设置
`overlay=true`。图上的 `T1/T2` 是单帧临时编号，同次工具结果会在
`overlay.candidates` 中返回它们到完整 `target_id` 的映射。多模态 LLM 只负责从本次
候选中排除看起来像海报、屏幕、镜像或无法判断的目标；提交 `vrc_autonomy_goal` 时必须
使用映射中的完整 ID，不能提交 T 编号。JPEG、同帧检测实体和 revision 只在内存中的
单槽对象保存，下一组直接覆盖；不写临时图片、不保留历史、不增加世界持久化次数。
`overlay.paired!=true`、`drawn=false`、候选为空或 `skew_warning=true` 时保持停止并重新观察。

### 卡墙判据（movement_stalled）

检测器只看画面，永远不会报告「前面有堵墙」；VRChat 内置 Velocity 参数是唯一能区分
「正在前进」和「顶着墙推摇杆」的回传。`VrchatOscBridge.motion_feedback()` 把这些内置
参数汇总成 `body_awareness.vrchat_osc.motion`（`GET /snapshot` 的 `vrchat_osc.motion`
也带同一份读数）。`VelocityX/Z` 只有角色移动时才回传，所以导航器只接受本次前进命令
之后的新样本，并给起步保留 450 ms；静止或停包会显示 `velocity_feedback_quiet`，OSC 层
不会把历史 0 伪装成当前速度。当前 Avatar 的 X/Z 都曾成功回传后，导航器会设置
`capability_confirmed=true`：此后前进超过起步宽限仍然沉默，就按零速度累计。连续默认
4 tick 沉默或实测水平速度低于 `stall_speed_mps` 时停车，`navigation.last_decision.reason` 变成
`movement_stalled`。水平速度用 `hypot(VelocityX, VelocityZ)`，不含 `VelocityY`——否则
「贴着墙往下滑」会被读成「正在前进」。

判定会闩锁：停下之后速度未知，靠速度自己解不开。导航器会先按有限预算尝试后退/沿
滑行方向转身，预算用尽才把当前实体记为暂时不可达并交还高层决策。
`navigation.stall.detectable=false` 且 `capability_confirmed=false` 表示当前 Avatar 从未
证明支持水平速度回传，「卡墙」这件事根本无法被观测到，**不是「没卡」**；此时判据
失效并放行。Avatar 切换会清空确认状态，不会继承上一个 Avatar 的能力。

> ⚠️ 已实测确认（2026-08-23）：内置参数名有效，`VelocityX/Z` 是 avatar 本地系，且
> 只在角色移动时回传。该 avatar 跑满速度为 `2.6667 m/s`，`y = 0.28`
> 实测约为 `0.8 m/s`，`y = 0.30` 实测约为 `0.8889 m/s`。当前导航在目标框达到
> 停止尺寸的 80% 前使用 `y = 0.60` 巡航，最后 20% 再线性降到 `y = 0.25`；
> 不同 Avatar/世界的阈值仍需真机会话复核。
> 若目标短暂漏检并进入 300 ms 视觉宽限，前进轴会立即限制为 `y = 0.25`，不会
> 按 `y = 0.60` 沿旧画面继续盲走；重新获得新鲜观测后才恢复巡航。
>
> 另外 `motion` 现在还导出 `velocity_x` / `velocity_z` 与 `forward_ratio` / `slip_ratio`。
> 比起只看 `horizontal_speed_mps` 是否塌到 0，这两个比值能区分「正面墙」和「斜撞墙正在
> 滑行」——后者速度模长未必小，但前进分量已经塌了，而滑行方向就是可通行方向。

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

视觉采集可以在后端运行期间独立启停，不需要重启 AnyaDance/OSC 控制链路：
以下示例假定后端使用 `vision.enabled = true` 的配置启动；`--offline` 会按设计强制
关闭视觉，调用 `start` 时会返回配置禁用原因。

```powershell
# 停止并释放 DXcam/WinRT/MSS 句柄，同时解除自主导航授权
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:48912/vision/stop `
  -Headers @{"X-Neko-Backend-Token"="dev"} -Body '{"reason":"manual_stop"}' `
  -ContentType 'application/json'

# 重新创建 FrameSource 并启动 worker（不会复用已关闭的 source）
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:48912/vision/start `
  -Headers @{"X-Neko-Backend-Token"="dev"} -Body '{}' `
  -ContentType 'application/json'
```

插件侧对应 `client.vision.stop()` 与 `client.vision.start()`。`start` 返回的
`worker` 状态包含实际后端和错误；配置为 `source = "external"` 时，必须先通过
`BackendService.attach_vision()` 注入新的 source，不能把已经停止的外部对象再次使用。
停止视觉会立即解除自主授权，避免没有新画面时继续导航；身体/OSC 后端保持运行。

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

内置 OpenVINO detector 会进一步把 Avatar 类轨迹交给 `AvatarIdentityRegistry`：同一
track 直接延续身份，并为每个身份有界保留最多 6 个正面/侧面/背面外观原型；新 track
优先按最接近的历史视角匹配。多个旧模板同样相似时，优先比较剔除全部人物框后的
4x8 低分辨率背景指纹；背景仍不明确时，仅在 15 秒内检测框几何明显连续，或某个模板的
稳定观测数至少达到其他候选 3 倍时复用。其余歧义仍分配新 ID，同一帧不允许两个轨迹
占用同一身份。外观不可提取、功能关闭或类别不是 Avatar 时，会安全降级为上述
`{source}:track:{track_id}`。`identity_reid` 状态中的 `ambiguous_reused_count`、
`context_reidentified_count`、`geometry_reidentified_count`、
`established_reidentified_count`、`appearance_prototype_count` 和
`context_prototype_count` 可用于区分真正重识别与模板增殖。

视觉轨迹和会话身份只参与当前进程的导航，不进入 `world_memory.json`。外部 detector
发布逐帧观测时应在实体 attributes 中设置 `memory_scope="observation"`；状态层也会
兼容识别旧版 `*:track:*`、`avatar:session:*` 和已知本地检测来源。`GET /perception`
的 `memory.transient_entities_persisted=false` 表示此边界生效，
`memory.persistence_write_count` 可用于确认没有随每个视觉帧重复写盘。

直接接入外部世界日志前必须做字段翻译：日志适配器要把事件映射为 `type`、稳定的
`target_id`、canonical `source`，并在 `player_left` 同批提供 `remove_entity_ids`；
旧版 `{event, detail, at_unix}` 结构不能直接提交，否则会被当作 `unknown` 事件，
也不会触发玩家删除。

配置段默认关闭；随插件部署的 `plugin.toml` 将 AnyaDance 发送频率设为 120 Hz，并
打开虚拟控制器主路由（OSC 仍是回退）：

```toml
[vision]
enabled = false
source = "none" # none / mss / dxcam / desktop_mirror / external
capture = "desktop_mirror"
local_backend = "openvino"
model_path = "models/yolox.xml" # 可选 XML/ONNX 路径，相对于配置目录
labels_path = "models/labels.txt" # 可选；留空时使用 COCO 名称
device = "AUTO" # AUTO / GPU / CPU，取决于已安装的 OpenVINO 插件
onnxruntime_cuda = "auto" # auto / prefer / disabled；只对 ONNX 模型生效
onnxruntime_cuda_device_id = 0 # NVIDIA CUDA device_id
fallback_backend = "none" # 显式设为 "opencv_hog" 可启用降级的仅人形模式
confidence_threshold = 0.35
input_width = 640
input_height = 640
horizontal_fov_deg = 90.0
max_detections = 64
# 检测框宽/高占画面的最小比例，用于滤掉几十像素级的高分假阳性。0 关闭对应轴。
# min_box_ratio 是两轴的共同回退值；只写它时两轴同值（旧配置行为不变）。
min_box_ratio = 0.02
min_box_width_ratio = 0.008
min_box_height_ratio = 0.02
semantic_backend = "openai_compatible"
semantic_max_per_minute = 30
# 给 agent 看的单槽帧缓存；与 world_state 无关，不产生实体也不产生事件。
frame_cache_interval_s = 1.0
frame_max_width = 960 # 降采样宽度；0 表示不缩放
frame_jpeg_quality = 70
frame_max_per_minute = 10 # agent 主动拉图的滑动窗口上限；0 表示禁止拉图
# -1 自动探测；MSS 的 0 是虚拟桌面，物理显示器从 1 开始。
monitor_index = -1
# -1 自动探测 DXGI 设备/输出；也可以填固定索引排查多 GPU 环境。
dxcam_device_idx = -1
dxcam_output_idx = -1
dxcam_backend = "auto" # auto / dxgi / winrt
interval_ms = 100
queue_size = 1
lifecycle_watermark_limit = 4096
# 非空时只采集该标题窗口的屏幕矩形，而不是整块显示器；仅限 Windows。
window_title = ""
# 重新解析窗口矩形的间隔（毫秒）；0 表示只在启动时解析一次。
window_track_interval_ms = 5000
```

配置只描述 worker，不下载模型。启用视觉后，即使模型暂缺也可以用
`capture_only=true` 运行采集诊断；这时世界仍是 unknown，不会产生实体。配置了
`model_path` 后，后端会在独立视觉 worker 中加载模型。ONNX 的自动优先级为
OpenVINO NPU/GPU → ONNX Runtime CUDA（可选）→ OpenVINO CPU → ONNX Runtime
CPU → OpenCV DNN。`onnxruntime_cuda = "auto"` 只负责探测，不构成硬依赖；设为
`prefer` 可让 CUDA 抢在 OpenVINO NPU/GPU 前（适合 Intel 核显 + NVIDIA 独显）；环境里
没有 `onnxruntime-gpu`、CUDA/cuDNN 动态库不匹配或 Session 创建失败时，原因会出现在
`/perception` 的 `onnxruntime.cuda_error`，然后继续 CPU 回落。设置为 `disabled` 可
完全跳过探测。常见 YOLO/SSD 输出会被归一化为带稳定 track ID 的实体；
`attributes.bearing_deg` 和屏幕几何关系可以供本地导航使用，但没有深度模型时距离
仍然是 unknown。OpenVINO 模型包、VLM endpoint 和 API key 由部署环境提供（VLM
endpoint 可用 `VRC_VLM_ENDPOINT`、模型用 `VRC_VLM_MODEL`），没有运行时或模型时
`/perception` 会明确报告 `available=false`，不会注入占位实体。若暂时没有模型，
只能显式设置 `fallback_backend = "opencv_hog"`；该路径仅检测行人并标记
`degraded=true`，不识别玩家身份，也不产生通用物体或距离结论。

检测结果在进入跟踪器之前会先过一道最小尺寸过滤。实测真实帧里出现过 27×27 px 的
0.9 分框：跟踪器会给它分配实体 ID，导航器再用 `apparent_height` 反推距离，于是把
噪点当成一个站在很远处的人——比漏检更糟，因为它会主动驱动动作。

宽高分开设阈值（`min_box_width_ratio` 默认 0.8%、`min_box_height_ratio` 默认 2%），
因为站立的人在画面里是高而窄的：共用一个 2% 时总是宽度先卡，1920 宽下要求最小宽
38 px，按人体长宽比反推，能进入世界的最小 `apparent_height` 已经有 7%~11%，而导航
器的目标是 0.55——房间对面的人会和噪点一起被裁掉。放松宽阈值让高度成为主判据；
两条边仍然都要过关，所以墙缝和 UI 边框这类细长误检照样被挡住。

`min_box_ratio` 是两轴的共同回退值，只设置它时两轴同值，旧配置行为不变。排查漏检
时可以把相应的值设为 0 关闭该轴的过滤。当前生效值会出现在 `/perception` 的检测器
状态里（`min_box_width_ratio` / `min_box_height_ratio`）。

### 给 agent 看的帧（`GET /vision/frame`）

检测器只回答「有几个人、在哪个方位」。要确认对方是谁、菜单开着没、界面上写了什么，
需要让 agent 亲眼看画面。这条路径和 `world_state` 完全分离：帧不产生实体、不产生
事件，也**不能**用来满足 `body_reach_and_grab` 的 `preconditions`——那条路仍然只认
检测器给出的 `entity_id` 与置信度。从像素得出的结论一律是低置信视觉猜测。

`VisionRuntime` 持有一个单槽最新帧缓存，由采集 worker 自己的消费线程在
`process_frame` 里填充，位置在 `_release_frame` 之前（句柄一旦释放就编码不了了），
并按 `frame_cache_interval_s` 限流。存最新一帧而不是队列，是因为过期的画面比没有
画面更危险：agent 会把它当成现在。

`GET /vision/frame?max_age_ms=N` 返回 `data_base64` 与 `age_ms`；采集已停止、还没有
帧、或缓存超过 `max_age_ms` 时返回 `available=false` 并给出 `reason`
（`capture_stopped` / `no_frame_cached` / `frame_stale`），不会退而求其次给旧画面。
编码失败只记进 `frame_cache.last_error` 并让这次拉取报不可用，绝不打断采集——看不到
图是降级，掉帧才是故障。

运行时层面 `max_age_ms <= 0` 表示不限龄，这是留给内部调用方的逃生口；`BackendService`
和 LLM 工具都把下限抬到 250 ms，否则「要最新的画面」写成 0 反而会拿到最旧的一张。

Windows 上若 DXGI 返回 `0x80070005`，可安装 `dxcam[winrt]` 启用合成器捕获回退：

```powershell
python -m pip install --user "dxcam[winrt]"
```

这条路径与 OBS 的 Windows Graphics Capture 更接近，不需要把后端提升为管理员；
如果目标窗口属于受保护内容，仍可能不可捕获，状态会保留在 `source.backends`。

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
