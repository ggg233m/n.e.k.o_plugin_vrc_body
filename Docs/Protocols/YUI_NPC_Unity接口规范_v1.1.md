# YUI 世界原生 NPC · Unity/Udon 开发接口规范

**规范版本：1.1（冻结）　Wire 版本：1　发布日期：2026-08-29　**

本文件是 YUI NPC 世界侧的实现与验收契约。负责 Unity 的开发者只依赖本文件、同目录的常量 JSON 和测试向量 JSON，即可实现完整 Prefab；Python 开发者依赖同一材料实现编码器、日志解析器和模拟器。本文中的“必须”“不得”“应”是规范性要求，“建议”才允许等价实现。

协议数值的机器可读事实源是 [`YUI_NPC_ProtocolConstants_v1.1.json`](./YUI_NPC_ProtocolConstants_v1.1.json)，示例与一致性输入是 [`YUI_NPC_TestVectors_v1.1.json`](./YUI_NPC_TestVectors_v1.1.json)。三份文件版本必须同时更新。Downloads 中的 v1.0 仅作历史参考，不得作为实现依据。

***

## 1. 交付范围与非目标

### 1.1 固定运行范围

| 项     | v1.1 冻结值                                                                    |
| ----- | --------------------------------------------------------------------------- |
| Unity | `2022.3.22f1`                                                               |
| 工程    | 独立 VCC Worlds 仓库；创建时选最新稳定 Worlds SDK，并在 `Packages/vpm-manifest.json` 锁定精确版本 |
| 平台    | PC                                                                          |
| 实例    | 私人实例，通常 2–3 人                                                               |
| NPC   | 单个，稳定标识 `yui`                                                               |
| 控制端   | driver 客户端本机的 N.E.K.O Python 进程                                             |
| 实现语言  | UdonSharp 或 Udon Graph 均可；对外行为必须完全相同                                        |

VRChat 官方当前指定 Unity `2022.3.22f1`；不得因 Unity Hub 的升级提示擅自换版本：[Current Unity Version](https://creators.vrchat.com/sdk/upgrade/current-unity-version/)。

### 1.2 Unity 开发者必须交付

1. 可直接拖入场景的 `YUI_NPC.prefab`，包含本规范定义的所有子对象、引用和 Udon 行为。
2. 已转为 Unity Humanoid 的 YUI 模型，移除 VRM 运行时不受 VRChat Worlds 支持的组件。
3. Animator Controller、Avatar Mask、完整实现 §10.2 冻结的 16 个核心语义动作、表情目录和静态文案目录。
4. NavMeshAgent、用于演示与验收的已烘焙 NavMesh、环境碰撞层和动态障碍配置。
5. 玩家触摸区、感知与射线、名牌和 UTF-8 文案气泡。
6. 固定公网直播流的 AVPro 播放器、NPC 空间音源与状态遥测。
7. 根对象、上身参数和持久状态的三类网络同步，以及统一 driver 所有权处理。
8. `YUI_NPC_Demo.unity` 验收场景、Inspector 配置记录、Build & Test 操作说明和两客户端验收结果。

YUI VRM、动作原素材及授权由交付方提供；Unity 开发者负责 Humanoid 转换、动作重定向、Animator 分层、裁剪和 PC 世界构建。

### 1.3 明确不包含

v1.1 不包含多 NPC、公共房高人数优化、全身骨骼实时流、精确嘴型、Persistence、运行时自由加载动画、Udon 本地社交决策、NPC 作为真实 VRCPlayer、任意外部代码执行或任意 URL 控制。

外部控制的含义是：Python 是**决策权威**，Udon 是**执行与安全权威**，VRChat 是**网络所有权权威**。Python 可在动作目录、低维上身姿态、移动、注视、表情、文本和语音提示的白名单范围内控制 NPC；不能绕过 ESTOP、NavMesh、世界边界、动作目录和 VRChat 网络限制。

***

## 2. 架构与职责边界

```text
N.E.K.O / Python
  ├─ loopMIDI: NEKO_MIDI
  │    ├─ Ch0 可靠命令
  │    ├─ Ch1 名义 20 Hz 上身参数
  │    └─ Ch2 UTF-8 文本载荷
  └─ tail VRChat output_log，查找任意位置的 [NEKO]{JSON}
                         │
                         ▼
driver 的 VRChat 客户端 ── Udon 执行/安全 ── VRChat 网络同步 ── 旁观客户端
                         │
                         └─ AVPro 固定公网直播流（只加载一次）
```

| 层         | 唯一职责                                       |
| --------- | ------------------------------------------ |
| Python    | 意图、人格、社交决策、命令排队、MIDI 限速、ACK 重试、TTS 和直播网关   |
| Udon      | 解码、鉴权门、幂等、状态机、NavMesh 移动、Animator、感知、安全、遥测 |
| VRChat 网络 | owner 写入、远端同步、晚加入者状态恢复                     |
| 直播网关      | 稳定 URL 上持续输出静音或 TTS 音频；接受唯一允许离开本机的数据类别     |

三条全局规则：

- **不伪造**：没有观测到的值用 JSON `null`，不得估计后冒充测量。
- **ESTOP 最高优先且锁存**：任何通道的 NoteOn `0x7F` 都必须立即处理；只有 `CLEAR_ESTOP` 能解除。
- **隐私边界**：仅生成后的 TTS 音频可上传到公网语音网关。玩家感知、日志、画面、显示名、playerId 和任何身份信息不得上传。

***

## 3. 规范性 Prefab 结构

类名是建议名，GameObject 分工和组件关系是强制契约。

```text
YUI_NPC                         Prefab/移动根；总控、VRCObjectSync、NavMeshAgent、根胶囊
├── Model                       Humanoid Animator；SkinnedMeshRenderer
│   ├── EyeAnchor               绑定 Head bone 的眼位锚点
│   └── AudioAnchor             绑定 Head/Chest 之间的空间音源锚点
│       ├── AudioSource
│       └── VRCSpatialAudioSource
├── UpperBodySync               Continuous 同步 UdonBehaviour
├── StateSync                   Manual 同步 UdonBehaviour
├── Runtime
│   ├── MidiRouter              VRC Midi Listener + 命令寄存器/幂等缓存
│   ├── SafetyController        状态机、心跳、ESTOP
│   ├── Locomotion              NavMesh 目标、跟随、卡住检测、局部射线
│   ├── ActionController        动作目录、Animator、动作生命周期
│   ├── UpperBodyController     8 参数流、约束、断流淡出
│   ├── TextController          7-bit 解包、CRC、气泡和同步
│   ├── Perception              玩家槽位、pose、社交几何、射线
│   ├── Telemetry               唯一业务 Debug.Log 出口
│   └── VoiceController         固定 URL AVPro 状态与 SPEECH_CUE
├── TouchZones
│   ├── Head                    Trigger Collider
│   ├── Cheek                   Trigger Collider
│   ├── HandL                   Trigger Collider
│   ├── HandR                   Trigger Collider
│   └── Torso                   Trigger Collider
├── UI
│   ├── Nameplate               World Space Canvas
│   └── SpeechBubble            UTF-8 文本；面向本地玩家
└── Voice
    └── AVProVideoPlayer        固定 VRCUrl；音频输出指向 AudioAnchor/AudioSource
```

TouchZones、UI、Voice 和 Model 都必须是移动根 YUI\_NPC 的后代，随根 Transform 一起移动；不得把触摸区或音源留在 RootSync 之外。本文后续的“RootSync”指 YUI\_NPC 根上的 VRCObjectSync 逻辑组件，不要求另建同名子对象。

Head/Cheek/HandL/HandR/Torso 五个 TouchZone 必须分别跟随对应 Humanoid bone（Cheek 跟 Head，Torso 跟 Chest）；可作为骨骼子对象或用当前 SDK allowlist 内的静态约束实现。Collider 本身为 trigger，不挂 Rigidbody，不参与 NavMesh 障碍。

### 3.1 组件输入、输出与失败状态

| 组件               | 输入                                   | 输出                           | 生命周期              | 必须处理的失败                           |
| ---------------- | ------------------------------------ | ---------------------------- | ----------------- | --------------------------------- |
| MidiRouter       | MIDI NoteOn/CC                       | 原子命令快照、上身帧、文本字节              | 世界加载到离开           | 未知通道、缺帧、保留位、重复 seq、seq 冲突         |
| SafetyController | DISCOVER、心跳、控制命令、owner 事件            | 控制状态、任务取消                    | 全生命周期             | 心跳超时、driver 离开、ownership 丢失、ESTOP |
| Locomotion       | goto/follow/wander/turn              | NavMeshAgent、arrived/blocked | 仅 driver owner 执行 | 越界、目标不在 NavMesh、无路径、卡住、局部障碍       |
| ActionController | 动作/表情命令、上身目标                         | Animator、动作事件                | 模型初始化后            | 目录无条目、优先级冲突、不可打断、状态缺失             |
| TextController   | TEXT\_BEGIN、CC29、COMMIT              | 气泡文本、Manual 同步               | session 内         | 超长、超时、长度/CRC/UTF-8 错误             |
| Perception       | VRCPlayerApi、Trigger、Physics.Raycast | 玩家、触摸、社交、射线日志                | driver 握手后        | 玩家离开、无效 player、无命中                |
| Telemetry        | 所有业务事件                               | `[NEKO]` 单行 JSON             | 全生命周期             | 20 行/s、950 UTF-8 字节、低优先级丢弃        |
| VoiceController  | 固定 VRCUrl、AVPro 回调、SPEECH\_CUE       | 空间音频、voice 日志                | 每次世界加载只打开一次       | URL 不可用、AVPro 错误、未播放时 cue         |

任何组件不得直接输出 `[NEKO]` 业务日志，必须调用 Telemetry。Unity/SDK 自身的诊断日志不受此限制。

***

## 4. 坐标、身份与基础类型

### 4.1 坐标

- 位置使用 Unity 世界坐标，单位米；日志保留到 3 位小数。
- yaw：`0° = +Z`，俯视顺时针为正，归一化到 `[0,360)`。
- `brg` 是所有 JSON 日志与 snapshot 中 `bearing` 的唯一字段名，单位为度：以 NPC **当前正前方**为 `0°`，俯视顺时针（NPC 右侧）为正，逆时针（NPC 左侧）为负，归一化到 `(-180,180]`。不得按事件类型改变符号。
- Python 把相对方位转为 TURN\_TO 绝对朝向时必须使用 `absolute_yaw_deg=wrap360(npc_yaw_deg+brg_deg)`。例如 NPC 当前 `yaw=350°`，目标在右前方 `brg=+20°`，则绝对 yaw 为 `10°`、TURN\_TO `P0=455`；`brg=-20°` 明确表示左前方，不能解释为右前方。
- `wireBoundsMin/Max` 是位置量化使用的三维 AABB；`activityBoundsMin/Max` 是 NPC 实际获准活动/注视的三维 AABB。wire bounds 必须在每个轴的两侧至少比 activity bounds 多 1.0m，且严格包含后者。
- `GOTO_XZ` 只改 X/Z，Y 由 NavMesh 决定；`LOOK_AT_XYZ` 使用完整 XYZ。两条命令先按 wire bounds 解码，再用 activity bounds 校验。
- 解码位置越出 activity bounds 必须拒绝并回 `target_out_of_bounds`，不得静默 clamp。activity bounds 内但在 `0.5m` 内找不到 NavMesh 点时回 `target_not_on_navmesh`。双边界使越界值在 wire 上可表示，因此该错误可由 Unity 独立测试，而不是只依赖 Python 预检查。

位置 14-bit 编码（下式 min/max 固定取 wire bounds 对应轴）：

```text
q = floor(((value - min) / (max - min)) * 16383 + 0.5)
value = min + (q / 16383) * (max - min)
```

yaw 14-bit 编码：

```text
q = floor((wrap360(yaw) / 360) * 16384 + 0.5) mod 16384
yaw = q * 360 / 16384
```

速度：`q=floor(speed/maxSpeed*127+0.5)`，解码 `speed=q*maxSpeed/127`。语义输入超出 `[0,maxSpeed]` 时发送端必须拒绝；世界侧也必须以 `invalid_param` 拒绝不可能的组合。

### 4.2 session、driver、pid 与 slot

- `session` 是 Python 在 DISCOVER 中选择的 28-bit 非零整数：`session=P0+(P1<<14)`。`0` 只用于未握手日志。
- `pid` 是 `VRCPlayerApi.playerId`，只在当前实例有效；日志中的 pid 与 slot 都以 session 为作用域。
- 世界侧维护 `0..63` slot 池，玩家加入时分配最小空闲 slot，离开时立即释放并发 `player.leave`。`127` 表示“无玩家/清除”。
- `world_id` 是 Inspector 必填稳定字符串；不得用可能本地化或重名的场景显示名替代。
- `driverClaimCode` 是 `0..16383` 的防误操作码；它不是密码。`driverDisplayName` 是可选便利门，非空时必须精确匹配。成功 DISCOVER 后，以同步的 `driverPid` 和三个同步对象的 owner 为运行期权威，不再用显示名判断。

DISCOVER 的 claim 顺序固定为：先校验 claim code 与非空显示名门；若当前 driverPid 指向仍在场的其他玩家，则回 `session_conflict`；若 driverPid 为空、指向本地玩家，或原 driver 已离开，才允许尝试取得三个对象 ownership。显示名或 claim code 不符回 `driver_auth_failed`，ownership 不完整回 `ownership_failed`。

失败 DISCOVER 不安装请求中的 session，不清旧状态/缓存/slot。失败 ACK 公共头的 session 与 state 使用失败前当前值；尚未握手时分别为 0 和 unhandshaken。只有成功 ACK 才把请求 session 写入公共头并发布 session/hello/catalog。

私人实例与 claim code 只能防误触，不能抵御恶意客户端；本规范不宣称 MIDI 控制具备安全认证。

***

## 5. 安全状态机

### 5.1 状态

| 数值 | 状态             | 含义                                                                      |
| -: | -------------- | ----------------------------------------------------------------------- |
|  0 | `unhandshaken` | 尚未接受 DISCOVER；仅 DISCOVER、SNAPSHOT\_REQUEST（回 not\_handshaken）和 ESTOP 有效 |
|  1 | `safe_idle`    | 已握手但未授权外部执行，或看门狗/driver 离开/清除 ESTOP 后的安全待机                              |
|  2 | `external`     | 外部控制已启用，当前无移动和主动作                                                       |
|  3 | `moving`       | goto/follow/wander/turn 正在执行；允许上身动作叠加                                   |
|  4 | `action`       | 无移动时有动作，或阻断移动的全身动作正在执行                                                  |
|  5 | `estop`        | 锁存急停；NavMesh、动作、注视、表情和上身流均冻结/清空                                         |

同一时刻有移动和允许移动的上身动作时，主状态报告 `moving`，动作细节由 `action_id/action_seq` 单独报告。状态优先级固定为：`estop > unhandshaken > safe_idle > 阻断移动的 action > moving > action > external`。

### 5.2 唯一转移规则

| 事件                            | 前态                                | 后态                      | 副作用                                                         |
| ----------------------------- | --------------------------------- | ----------------------- | ----------------------------------------------------------- |
| 首次有效 DISCOVER                 | unhandshaken                      | safe\_idle              | 建 session、claim driver、获取三个对象 ownership、发 ACK/hello/catalog |
| 新 session DISCOVER            | 任意非 estop                         | safe\_idle              | 取消移动/动作/文本，清幂等缓存与 slot 映射后重建                                |
| 新 session DISCOVER            | estop                             | estop                   | 重建 session，但保持 ESTOP 锁存                                     |
| SET\_CONTROL\_MODE external   | safe\_idle/external               | external                | 不恢复旧任务                                                      |
| SET\_CONTROL\_MODE safe\_idle | external/moving/action/safe\_idle | safe\_idle              | 取消移动、动作、注视、文本传输并清除气泡                                        |
| 有效移动命令                        | external/moving/action            | moving                  | 按 §11 的动作冲突规则处理                                             |
| 动作开始                          | external/action/moving            | action 或保持 moving       | 取决于 layer/movement                                          |
| STOP                          | safe\_idle/external/moving/action | external 或保持 safe\_idle | 正常停止移动并取消动作，不解除 ESTOP                                       |
| HEARTBEAT 超过 3 秒未到            | external/moving/action            | safe\_idle              | 取消任务并发 `sys.watchdog`；恢复心跳不恢复任务                             |
| 任意 NoteOn 0x7F                | 任意                                | estop                   | 下一次 Update 前停止 agent，清任务、动作、文本传输和气泡；锁存                      |
| CLEAR\_ESTOP                  | estop                             | safe\_idle              | 解除锁存；必须另发 SET\_CONTROL\_MODE external 才能继续                  |
| driver 离开或 owner 丢失           | 任意非 estop                         | safe\_idle              | 停止本地执行、driverPid=-1；不得自动选新 driver                           |

ESTOP 后发送 SET\_MODE、STOP、DISCOVER 或 SET\_CONTROL\_MODE 都不得解除锁存。除 HEARTBEAT、DISCOVER、SNAPSHOT\_REQUEST、CLEAR\_ESTOP 和 ESTOP 外，estop 中的命令回 `estop_latched`。

***

## 6. MIDI 物理层与可靠命令

VRChat 实时 MIDI 需要 `VRC Midi Listener`，并在组件中显式启用 NoteOn 与 ControlChange；运行时可用 `--midi=NEKO_MIDI` 指定设备。官方事件的 channel、number、value/velocity 都是 `0..127`：[Realtime MIDI](https://creators.vrchat.com/worlds/udon/midi/realtime-midi/)、[MIDI Events](https://creators.vrchat.com/worlds/udon/midi/)。

### 6.1 通道

| 0-based channel | 用途                        |
| --------------: | ------------------------- |
|               0 | 可靠命令：CC 写 P0–P5，NoteOn 提交 |
|               1 | 名义 20 Hz 上身语义流            |
|               2 | 动态 UTF-8 文本载荷             |

NoteOff 一律忽略。除 ESTOP 外，其他 NoteOn 只在指定通道生效。

Channel 0 的 cmdId `0x00` 与 `0x17..0x7E` 保留，当前收到时按 `unknown_cmd` 处理；Channel 1 除 CC20..27、NoteOn 0x40 和 ESTOP 外全部忽略；Channel 2 除 CC29 和 ESTOP 外全部忽略。被忽略的非协议事件不产生 ACK，只计入对应的 invalid\_events，且同类诊断最多 1Hz。

### 6.2 P0–P5 与提交

| 参数 | 高 7 位 | 低 7 位 | 值           |
| -- | ----: | ----: | ----------- |
| P0 |  CC20 |  CC21 | `hi*128+lo` |
| P1 |  CC22 |  CC23 | 同上          |
| P2 |  CC24 |  CC25 | 同上          |
| P3 |  CC26 |     — | 0..127      |
| P4 |  CC27 |     — | 0..127      |
| P5 |  CC28 |     — | 0..127      |

每条 Channel 0 可靠命令必须依次写完 CC `20,21,22,23,24,25,26,27,28`，包括值为零和该命令未使用的寄存器，然后发送 `NoteOn(number=cmdId, velocity=seq)`。不得依赖上一命令残留值。`seq=1..127` 循环；velocity=0 时只有 ESTOP 仍执行且不回 ACK，其他命令必须忽略且不执行。

### 6.3 请求哈希与幂等

`request_hash` 是 CRC-16/CCITT-FALSE，输入字节严格为：

```text
cmdId, seq, P0_hi, P0_lo, P1_hi, P1_lo, P2_hi, P2_lo, P3, P4, P5
```

多字节寄存器在哈希中按高 7 位、低 7 位顺序；CRC 参数为 poly `0x1021`、init `0xFFFF`、RefIn/RefOut false、XorOut `0x0000`。日志写 4 位大写十六进制。

处理顺序固定为：先识别并执行 ESTOP；其余命令先确定 session（DISCOVER 使用请求中的 session，其他命令使用当前 session）并计算 hash，再做重复/seq 冲突判断，最后才做 driver、状态和参数校验并执行。因此同 seq 不同 hash 始终回 `seq_conflict`，不会被后续 `reserved_bits` 等错误覆盖。ESTOP 不读取任何通道的寄存器，ACK hash 以 P0–P5 全 0 的规范帧计算。

幂等键固定为 `session + seq + cmdId + request_hash`：

- 5 秒窗口内完全相同的请求不得再次产生移动、动作、文本重置等副作用，必须重放相同结果的 ACK，且 `replayed=true`。
- 同一 session、同一 seq 在 5 秒窗口内出现不同 cmdId 或不同 hash，拒绝并回 `seq_conflict`。
- 普通可靠命令在 2 秒未收 ACK 时才可原样重发；超过 5 秒仍未成功时不得继续用旧 seq，应重建 session。普通命令等待 ACK 或重发期间，§6.4 的 HEARTBEAT 保活通道必须继续工作。
- HEARTBEAT 每次必须从同一全局 seq 分配器取得新 seq，名义每 1 秒发送一次，不得重发旧 HEARTBEAT。其 ACK 与 `sys.pong` 只用于链路诊断；漏收 HEARTBEAT 的 ACK 不得阻塞下一次 HEARTBEAT，也不得占用普通命令的未决槽。未决普通命令保留原 seq；中间若已发送若干新 seq 的 HEARTBEAT，2 秒后仍允许用该普通命令的旧 seq 与原 hash 重发，接收端按幂等缓存重放 ACK，不得因 seq 数值倒退而拒绝。
- 新 session 清空缓存。ESTOP 的锁存不因清缓存解除。

### 6.4 优先级与预算

MIDI 任意滑动 1 秒不得超过 200 条。正常调度硬上限为 199 条，并始终为 ESTOP 保留第 200 条；ESTOP 不得依赖正常队列先腾位。

优先级固定为 `ESTOP > HEARTBEAT > 普通可靠命令 > 文本载荷 > 上身流`。一条上身帧固定 9 条，一条可靠命令固定 10 条。上身流名义为 20 帧/s：滑动 1 秒内有 0 或 1 条可靠命令时最多 20 帧；有 2 条时必须丢弃该窗最晚的一条未发送上身帧，最多 19 帧。因此三种上限分别为 `180/190/191` 条。文本传输时上身流暂停并淡回中立，文本 CC29 最多 159 条/s，与 4 条可靠命令合计 `159+40=199`。任何正常路径都不得达到 200。

可靠命令还有独立硬上限，HEARTBEAT 计入总数：上身流 active 时最多 2 条/s；文本事务期间最多 4 条/s；其余时间最多 4 条/s。合规 Python 发送端必须为 1Hz HEARTBEAT 预留其中一条，因此普通可靠命令对应最多为 `1/3/3` 条/s；HEARTBEAT 的调度延迟不得超过 250ms。只有普通可靠命令受 `max_in_flight=1` 限制：Python 必须先等该普通命令 ACK 或 2 秒超时后才能发送下一条普通命令；ESTOP 与 HEARTBEAT 均不占此槽。普通命令超时重发时仍保持 1Hz HEARTBEAT，漏收 HEARTBEAT ACK 不触发重发。超限命令不得进入 MIDI 发送队列。该限制同时保证最坏情况下 ACK、操作生命周期和 5Hz state/2Hz pose/1Hz stream stats 能落在 20 行/s 遥测预算内。

Unity 的 3 秒看门狗只由通过 session/driver/owner 校验且实际收到的 HEARTBEAT 刷新。`npc.state`、`sys.pong`、其他 ACK 或其他业务日志都不得作为隐性心跳；Python 侧收到这些日志也不得假定 Unity 看门狗已刷新。因而普通命令 ACK 丢失时仍由独立 HEARTBEAT 保持 external，不会在 2 秒重发窗口内误降级 safe idle。

***

## 7. Channel 0 命令表

表中未列出的寄存器必须为 0。任何非零保留位回 `reserved_bits`；参数域不合法回 `invalid_param`。成功 ACK 只表示世界侧已接受并开始/完成即时操作，不代表移动已经到达或动作自然结束。

|     ID | 名称                 | 参数                                                                        | 确定行为                                                                                                                                            |
| -----: | ------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `0x01` | SET\_MODE          | P3: 0 idle / 1 follow / 2 goto / 3 wander                                 | idle 停移动；follow/goto/wander 分别要求自身 capability 且共同要求 navmesh，缺任一项按 §7.4 回 `unsupported_capability`；能力具备后才检查目标/waypoint |
| `0x02` | GOTO\_XZ           | P0 X，P1 Z，P2 yaw，P3 speed，P4 bit0 hasYaw                                  | 同时要求 goto+navmesh；完整校验后才写目标并进入 moving；P4 bit0=0 时 P2 必须为 0；越界拒绝，绝不 clamp，不允许直线降级 |
| `0x03` | SET\_SPEED         | P3 speed                                                                  | 更新后续与当前移动速度；0 表示暂停但保留目标                                                                                                                         |
| `0x04` | TURN\_TO           | P0 yaw                                                                    | agent 原地转向，最大 180°/s；误差 ≤2° 后回 external                                                                                                         |
| `0x05` | LOOK\_AT           | P3 slot，127 清除                                                            | 注视指定玩家，不移动；未知 slot 回 `slot_unknown`                                                                                                             |
| `0x06` | PLAY\_ANIM         | P0 actionSeq，P3 actionId，P4 bit0 loop                                     | actionSeq 1..16383；actionId 不在目录回 `action_not_found`；loop 仅在目录 `loopable=true` 时允许；冲突规则见 §10                                                 |
| `0x07` | STOP               | 全 0                                                                       | 正常停止移动、动作、注视与表情覆盖，清当前目标，回 external；safe\_idle 中为成功 no-op；estop 中回 `estop_latched`                                                               |
| `0x08` | TEXT\_PRESET       | P3 presetId，127 清空；P4 秒，0 默认 5                                            | 显示静态目录文案；127 清除静态或动态气泡且 P4 必须为 0；不存在回 `text_preset_not_found`                                                                                   |
| `0x09` | RAY\_SCAN          | P3 0 八向 / 1 前向 7 射线                                                       | 回 `env.ray`；其他值拒绝                                                                                                                               |
| `0x0A` | SET\_RATE          | P3 state 档；P4 pose 档                                                      | state 档 0/1/2/3→0/1/5/10Hz；pose 档 0/1/2→0/1/2Hz                                                                                                 |
| `0x0B` | HEARTBEAT          | 全 0                                                                       | 有效接收即更新看门狗并回 ACK 后发 `sys.pong`；每次使用新 seq、不得重发；ACK 仅供诊断，不占普通命令未决槽，也不会自动恢复任务                                                                         |
| `0x0C` | DISCOVER           | P0 session low14；P1 high14；P2 claim code                                  | claim driver、建/重建 session，回 ACK、`sys.session`、hello 和全部 catalog；同请求重放上述响应但不重置状态                                                                 |
| `0x0D` | CLEAR\_ESTOP       | 全 0                                                                       | 仅 estop 中有效；解除后进入 safe\_idle，不恢复任务                                                                                                              |
| `0x0E` | STOP\_ACTION       | P0 actionSeq，0=当前动作                                                       | 显式取消匹配动作，即使目录标为不可打断；没有匹配动作时成功 no-op                                                                                                             |
| `0x0F` | SNAPSHOT\_REQUEST  | 全 0                                                                       | 发布 snapshot 时 ACK 后输出 §14.8 完整 snapshot；未发布时按 §7.4 回 `unsupported_capability`，不输出任何 part |
| `0x10` | SET\_TARGET        | P3 slot，127 清除                                                            | 设置 follow 默认目标；不立即移动；未知 slot 拒绝且保留旧目标                                                                                                           |
| `0x11` | LOOK\_AT\_XYZ      | P0 X、P1 Y、P2 Z；P3 weight；P4 秒，0 持续；P5 bit0 body assist                    | 注视空间点；weight=q/127；到期淡回；越界拒绝                                                                                                                    |
| `0x12` | SET\_EXPRESSION    | P3 expressionId，127 清除；P4 weight；P5 秒，0 持续                                | 在 Face 层交叉淡化；127 时 P4/P5 必须为 0；不存在回 `expression_not_found`                                                                                      |
| `0x13` | TEXT\_BEGIN        | P0 transferSeq；P1 字节长；P2 CRC low14；P3 CRC high2；P4 秒                      | P1 为 0 或大于 384 时专门回 `text_too_long`；否则建立唯一文本事务并暂停上身流；细节见 §9                                                                              |
| `0x14` | TEXT\_COMMIT       | 与 BEGIN 完全相同                                                              | 校验长度、CRC、严格 UTF-8；成功后原子显示并同步                                                                                                                    |
| `0x15` | SPEECH\_CUE        | P0 speechSeq；P1 textSeq/0；P2 actionSeq/0；P3 delay*100ms；P4 duration*250ms | P4 必须 1..127；仅关联与遥测，不生成音频、不启动动作；播放器非 playing 时拒绝                                                                                                |
| `0x16` | SET\_CONTROL\_MODE | P3 0 safe\_idle / 1 external                                              | 进入安全待机或启用外部执行；不能解除 ESTOP                                                                                                                        |
| `0x7F` | ESTOP              | 无参数，任意通道                                                                  | 立即锁存。velocity 1..127 时回 ACK；velocity 0 仍执行但不回 ACK                                                                                               |

`GOTO_XZ` 的 P4 只有 bit0 有效，因此 v1.0 中超出 14-bit 范围的 yaw 哨兵已彻底删除。

### 7.1 DISCOVER 顺序

有效新 session 的输出顺序必须是：`npc.ack` → `sys.session` → `sys.hello` → 各类 `sys.catalog`。catalog 顺序固定为 `action`、`expression`、`text_preset`、`anchor`，每类按 id 升序，每行一个 item。若 20 行/s 预算不足，后续行排队，不得乱序。

同 session、相同 DISCOVER 重试时，ACK 标 `replayed=true`，然后重新输出 hello 和完整 catalog；这属于响应重放，不得重置动作或移动。

当前 driver 用新 seq 对同一 session 发 DISCOVER 时视为主动刷新：不重置运行状态，依次输出成功 ACK、`sys.session{previous_session==new_session,reset:false}`、hello 和完整 catalog。只有 session 数值变化时才执行任务/slot/幂等缓存重建。

首次或新 session DISCOVER 在 catalog 之后，还必须按 slot 升序为当前实例内每位有效玩家补发一次 `player.join`（包括 driver），让 Python 重建 session 作用域内的 slot 映射。重复 DISCOVER 只重放 hello/catalog，不重复 player.join；需要完整当前玩家表时使用 SNAPSHOT\_REQUEST。

SET\_MODE 的目标保留规则固定如下：idle 停止 agent，但保留最近有效 GOTO 目标和 SET\_TARGET 玩家；follow/goto 是对这两个已保存目标的显式恢复命令；STOP 和 session 重建清除两类目标。进入 safe\_idle 不自动恢复任何任务，但保留目标，重新 external 后仍须显式 SET\_MODE 才能使用。turn 执行期间 `npc.state.state="moving"`、`mode="idle"`。

### 7.2 命令 × 状态权限矩阵

下表是唯一权限源。`✓` 表示继续做能力、参数和资源校验；`NH` 回 `not_handshaken`；`IS` 回 `invalid_state`；`EL` 回 `estop_latched`。STOP 在 safe\_idle 为成功 no-op。DISCOVER 在 estop 中不得清除锁存。

| 命令 | unhandshaken | safe\_idle | external | moving | action | estop |
|---|---:|---:|---:|---:|---:|---:|
| SET\_MODE | NH | IS | ✓ | ✓ | ✓ | EL |
| GOTO\_XZ | NH | IS | ✓ | ✓ | ✓ | EL |
| SET\_SPEED | NH | IS | ✓ | ✓ | ✓ | EL |
| TURN\_TO | NH | IS | ✓ | ✓ | ✓ | EL |
| LOOK\_AT | NH | IS | ✓ | ✓ | ✓ | EL |
| PLAY\_ANIM | NH | IS | ✓ | ✓ | ✓ | EL |
| STOP | NH | ✓ | ✓ | ✓ | ✓ | EL |
| TEXT\_PRESET | NH | IS | ✓ | ✓ | ✓ | EL |
| RAY\_SCAN | NH | ✓ | ✓ | ✓ | ✓ | EL |
| SET\_RATE | NH | ✓ | ✓ | ✓ | ✓ | EL |
| HEARTBEAT | NH | ✓ | ✓ | ✓ | ✓ | ✓ |
| DISCOVER | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CLEAR\_ESTOP | NH | IS | IS | IS | IS | ✓ |
| STOP\_ACTION | NH | IS | ✓ | ✓ | ✓ | EL |
| SNAPSHOT\_REQUEST | NH | ✓ | ✓ | ✓ | ✓ | ✓ |
| SET\_TARGET | NH | IS | ✓ | ✓ | ✓ | EL |
| LOOK\_AT\_XYZ | NH | IS | ✓ | ✓ | ✓ | EL |
| SET\_EXPRESSION | NH | IS | ✓ | ✓ | ✓ | EL |
| TEXT\_BEGIN | NH | IS | ✓ | ✓ | ✓ | EL |
| TEXT\_COMMIT | NH | IS | ✓ | ✓ | ✓ | EL |
| SPEECH\_CUE | NH | IS | ✓ | ✓ | ✓ | EL |
| SET\_CONTROL\_MODE | NH | ✓ | ✓ | ✓ | ✓ | EL |
| ESTOP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 7.3 错误判定优先级

同一输入同时违反多条规则时，必须按以下顺序返回第一项；开发者不得调整：

1. 任意通道 ESTOP 立即执行，跳过其余步骤。
2. 非 ESTOP 的 velocity=0 静默忽略。
3. 确定 session、计算 request hash、处理幂等重放或 `seq_conflict`。
4. 未定义 cmdId 回 `unknown_cmd`。
5. DISCOVER 单独执行 claim code、显示名、session 冲突和 ownership 校验；完成后不进入后续通用步骤。
6. 无 session 回 `not_handshaken`。
7. 本地玩家不等于 driverPid 回 `not_driver`。
8. driver 未拥有三个同步对象回 `not_owner`。
9. estop 中不在白名单的命令回 `estop_latched`。
10. 按 §7.2 回 `invalid_state`。
11. 未发布能力回 `unsupported_capability`。
12. 保留寄存器或 bit 非零回 `reserved_bits`。
13. TEXT\_BEGIN 的 P1 不在 1..384 时回专用 `text_too_long`。
14. 其余数值范围或字段组合错误回 `invalid_param`。
15. slot、action、expression、text 等引用错误回对应资源错误。
16. NavMesh、voice 等运行资源失败回对应运行错误。
17. 所有检查通过后才允许改变状态并回成功 ACK。

幂等重放只重放先前的 ok/err/detail，不重复执行；重放 ACK 的 `state` 必须取重放时的当前状态，避免 ESTOP 后仍报告旧状态。

### 7.4 capability 缺失时的唯一行为

`sys.hello.caps/cap_bits` 是能力是否发布的唯一权威。Python 必须先读能力表；因此“能力未发布所以不产生日志”和“能力已发布但当前没有事件”可以确定地区分。状态权限为 `✓` 只表示该命令在该状态可进入后续校验，不绕过本表的 capability 门。

| bit | capability | 入口或输出 | 未发布时的强制行为 |
| ---: | --- | --- | --- |
| 0 | `goto` | GOTO\_XZ；SET\_MODE goto；`npc.go_to/_xyz` | 两条 wire 入口回 `unsupported_capability`；适配层不暴露对应工具；不得直线移动 |
| 1 | `follow` | SET\_MODE follow；`npc.follow` | SET\_MODE 回 `unsupported_capability`；SET\_TARGET 可保存 slot 但不得启动跟随；适配层不暴露工具 |
| 2 | `wander` | SET\_MODE wander | 回 `unsupported_capability`；不得把缺 waypoint 泛化成 idle 或随机游走 |
| 3 | `actions` | PLAY\_ANIM、STOP\_ACTION；action catalog | 两命令都回 `unsupported_capability`；action catalog 必须为空；不触发 Animator 动作 |
| 4 | `expressions` | SET\_EXPRESSION；expression catalog | 命令回 `unsupported_capability`；expression catalog 必须为空；保持 neutral |
| 5 | `text_preset` | TEXT\_PRESET；text\_preset catalog | 命令回 `unsupported_capability`；text\_preset catalog 必须为空 |
| 6 | `text_utf8` | TEXT\_BEGIN/COMMIT；Channel 2 CC29 | BEGIN/COMMIT 回 `unsupported_capability`；CC29 静默丢弃，不建立事务、不发错误 |
| 7 | `upper_body_stream` | Channel 1 CC20..27 + commit；stream stats | 整帧静默丢弃，不更新 Animator/Continuous 同步/统计，不增加 invalid\_events，不发 `sys.err` 或 `npc.stream_stats` |
| 8 | `ray_scan` | RAY\_SCAN；`env.ray` | 命令回 `unsupported_capability`；不得发 `env.ray` |
| 9 | `touch` | TouchZones；`player.touch` | TouchZone collider/回调必须禁用；不得发 `player.touch`，不影响普通玩家 slot 管理 |
| 10 | `player_pose` | 周期 `player.pose` | 不启动周期 pose 定时器，不发 `player.pose`；若 snapshot 能力存在，snapshot players section 仍按固定 schema 返回 |
| 11 | `voice_stream` | SPEECH\_CUE；AVPro；voice 日志 | SPEECH\_CUE 回 `unsupported_capability`；不 LoadURL、不发周期/状态 voice 日志；snapshot voice 固定为 disabled/null/false/null |
| 12 | `snapshot` | SNAPSHOT\_REQUEST | 命令回 `unsupported_capability` 且不发任何 `sys.snapshot`；§7.2 的 `✓` 不是“无条件可用” |
| 13 | `navmesh` | GOTO\_XZ；SET\_MODE follow/goto/wander | 上述入口即使自身能力存在也回 `unsupported_capability`；禁止直线移动或改回 `no_path`。TURN\_TO 不依赖 navmesh |
| 14 | `social_signals` | `social.gaze/wave/approach` | 不运行社交检测器、不发任何 `social.*`；不得影响 touch/player.pose |
| 15 | `anchors` | anchor catalog；`npc.go_to` | DISCOVER 仍按固定顺序发一页空 anchor catalog，hello count=0；适配层不暴露 `npc.go_to`，不得猜测未配置地点 |
| 16 | `operation_lifecycle` | `npc.operation_*`；active\_ops | 命令和领域事件照常执行，但不发任何 `npc.operation_*`；`npc.state` 与 snapshot npc 的 `active_ops` 字段仍必填且固定为空数组 |

同一命令依赖多个能力时，只要任一项未发布就回一次 `unsupported_capability`，不继续参数、目录或运行资源校验。特别地：GOTO 依赖 `goto+navmesh`，follow 依赖 `follow+navmesh`，wander 依赖 `wander+navmesh`；缺 navmesh 从不回 `no_path`。所有 17 项的禁用行为都必须由 Constants JSON 指定的独立向量覆盖。

***

## 8. Channel 1：名义 20 Hz 上身语义流

### 8.1 帧格式

发送端每帧依次发送 Channel 1 的 CC20..CC27，再发送 `NoteOn(number=0x40, velocity=streamSeq)`。`streamSeq=1..127` 循环，0 禁用。接收端只在 NoteOn 时原子采样 8 个值；session 首帧前若任一 CC 未收到，必须丢弃该帧、令 `invalid_events += 1`，并发 `sys.err{err:"stream_incomplete",code:28,source:"stream",fatal:false,related_seq:streamSeq}`。同类 `sys.err` 受 5 秒冷却限制，但每个不完整帧都必须累计到 `npc.stream_stats.invalid_events`。

| CC | 参数                  |         范围 | 中立 | 正方向        |
| -: | ------------------- | ---------: | -: | ---------- |
| 20 | head yaw            |   -60..60° | 0° | 向 NPC 右侧转头 |
| 21 | head pitch          |   -35..35° | 0° | 抬头         |
| 22 | torso yaw           |   -35..35° | 0° | 向 NPC 右侧扭身 |
| 23 | torso pitch         |   -20..20° | 0° | 后仰         |
| 24 | left arm elevation  |    0..160° | 0° | 从自然下垂抬臂    |
| 25 | left arm azimuth    | -120..120° | 0° | 向 NPC 右侧   |
| 26 | right arm elevation |    0..160° | 0° | 从自然下垂抬臂    |
| 27 | right arm azimuth   | -120..120° | 0° | 向 NPC 右侧   |

对有正负范围的值，`q=64` 必须精确表示 0：

```text
value <= 0: q = floor(64 + value/abs(min)*64 + 0.5)
value > 0:  q = 64 + floor(value/max*63 + 0.5)
q <= 64: value = (q-64)/64*abs(min)
q > 64:  value = (q-64)/63*max
```

手臂 elevation 使用普通 `0..127 ↔ 0..160°` 线性量化。中立帧是 `[64,64,64,64,0,64,0,64]`。

### 8.2 应用、安全与网络

- 20 Hz 是 driver 本机无拥塞时的名义输入频率；滑动 1 秒内出现两条可靠命令时按 §6.4 降为最多 19 帧。这不是 VRChat 远端网络频率保证。
- 连续 250ms 没有有效 commit，8 个值必须在 200ms 内线性淡回中立；不得保持最后姿态。
- estop、safe\_idle、未握手、全身阻断动作或文本事务期间不应用流；文本期间发送端也必须停发。
- 头、躯干和手臂优先通过 Animator BlendTree 与预制 Humanoid additive clips 实现；如使用其他静态约束组件，必须确认在当前 Worlds SDK allowlist 内。不得在运行时载入骨骼数据或绕开 Humanoid 限制。
- `UpperBodySync` 以 Continuous 同步 8 个归一化 float、`streamSeq` 和 `active`。远端做 100ms 指数/阻尼平滑；不自行补帧为“真实观测”。
- 每秒输出一次 `npc.stream_stats`；序号跨越时按 1..127 环计算丢帧，完全重复 seq 计 duplicate、不重复应用。

骨骼仲裁固定为：UpperBody 目录动作激活时由动作拥有 Spine/Chest/Shoulder/Arm，流中的 torso 与 arm 六值继续接收但不应用；动作结束后，最后一帧年龄 <250ms 才在 100ms 内恢复，否则保持中立。Head/Neck 不在 UpperBody 动作 AvatarMask 中：有 LOOK\_AT/LOOK\_AT\_XYZ 时由注视目标拥有 Head/Neck；无注视时由流的 head yaw/pitch 拥有。LOOK\_AT\_XYZ 的 body-assist bit=1 且没有 UpperBody/FullBody 动作时，可拥有 torso yaw/pitch，并把角度限制在流定义范围内。FullBody 动作期间暂停全部 8 个流值和注视骨骼，Face blendshape 表情仍可播放。

***

## 9. Channel 2：动态 UTF-8 文本

### 9.1 事务

1. Python 计算严格 UTF-8 字节，长度必须 `1..384`。Udon 仍须独立检查；TEXT\_BEGIN 的 P1=0 或 P1>384 固定回 `text_too_long`，不得泛化为 `invalid_param`。
2. 计算 CRC-16/CCITT-FALSE；`P2=crc & 0x3FFF`，`P3=(crc>>14)&0x03`。
3. 发送 TEXT\_BEGIN；收到成功 ACK 后暂停上身流。
4. 把 UTF-8 字节按 §9.2 打包，经 Channel 2 CC29 发送。
5. 发送参数完全相同的 TEXT\_COMMIT。成功 ACK 后文本原子显示并写入 StateSync。

`transferSeq=1..16383`，在同一 session 内不得复用。BEGIN 已有事务时回 `transfer_busy`；完全相同的 BEGIN 命令重试只重放 ACK，不清空缓冲。已经成功 COMMIT 过的 transferSeq 再次用于新的 BEGIN 时回 `transfer_seq_mismatch`，即使元数据和文本相同也不重新显示。5 秒内未成功 COMMIT 时取消事务、淡回上身中立并发 `sys.err{text_timeout}`。

COMMIT 依次校验：事务存在 → transferSeq 相同 → BEGIN/COMMIT 元数据相同 → 收到的原始长度 → CRC → 严格 UTF-8。失败不显示部分文本，错误分别为 `transfer_missing`、`transfer_seq_mismatch`、`invalid_param`、`length_mismatch`、`crc_mismatch`、`invalid_utf8`。除 `transfer_missing` 外，任一失败 COMMIT 都立即销毁当前事务并允许新的 BEGIN；发送端修正后必须换新 transferSeq，不得向旧事务补发。

成功 COMMIT 的日志顺序固定为 `npc.ack` 后 `npc.text_displayed`；ACK 排队成功前不得先发 displayed。

P4 为展示秒数，0 表示 5 秒，1..127 为实际秒数。到期清空气泡并同步；后来的有效文本替换当前文本。日志和网络中不得截断 UTF-8 码点。

### 9.2 标准 MIDI 7-bit packing

每组最多 7 个原始字节 `b0..b6`：

```text
msb = Σ(((bi >> 7) & 1) << i)
发送 CC29(msb)
按顺序发送 CC29(bi & 0x7F)
```

最后一组按实际字节数结束，不补零。接收端由 BEGIN 的原始字节长度推导每组长度与总 packed 长度 `rawLength + ceil(rawLength/7)`。非 CC29 的 Channel 2 CC 忽略并计入流统计 invalid\_events。

没有活动 TEXT\_BEGIN 时收到 CC29，或已经收到推导出的 packed 总长度后仍收到 CC29，都必须忽略并增加 invalid\_events；不得把这些值带入下一次事务。若事务结束前缺字节，COMMIT 回 `length_mismatch`。

***

## 10. Animator 与动作契约

### 10.1 固定层

| 层顺序 | 名称              | 模式                    | 要求                                               |
| --: | --------------- | --------------------- | ------------------------------------------------ |
|   0 | Base Locomotion | Override              | Idle/Walk/Run BlendTree，由 `Speed` 驱动             |
|   1 | UpperBody       | Override + AvatarMask | 目录 upper\_body 动作与 torso/arm 六参数；不改根、腿、Neck、Head |
|   2 | FullBody        | Override              | 全身动作；目录 movement 必须为 block                       |
|   3 | FaceAndLook     | Override/Additive     | Head/Neck 的流/注视，以及表情 blendshape；不改根和四肢           |

Animator 必须 `Apply Root Motion=false`。根移动只由 owner 的 NavMeshAgent/转向控制器产生。建议参数名：`Speed(float)`、`ActionId(int)`、`ActionSeq(int)`、`ActionLoop(bool)`、`ExpressionId(int)`、`ExpressionWeight(float)`、`Estop(bool)` 和 8 个上身 float；若内部改名，目录与行为必须等价。

### 10.2 action catalog

每个 action 条目必须完整提供：

```json
{
  "id": 3,
  "name": "wave",
  "semantic_key": "greet_wave",
  "description_zh": "向目标友好挥手",
  "intent_tags": ["greeting", "friendly"],
  "target_required": "player",
  "speech_compatible": true,
  "layer": "upper_body",
  "duration_ms": 1800,
  "loopable": false,
  "movement": "allow",
  "priority": 40,
  "interruptible": true,
  "fade_in_ms": 120,
  "fade_out_ms": 180
}
```

约束：id `0..126` 唯一；name 与 semantic_key 都匹配 `/^[a-z0-9_]{1,32}$/`；`description_zh` 为 1..80 UTF-8 字节；intent_tags 为 1..8 个同样格式的字符串；target_required 只能为 `none/player/point`；speech_compatible 为 bool。所有 action 的 duration 都为 `1..600000ms`，循环动作填写单周期时长；priority `0..100`；淡入淡出 `0..5000ms`。目录时长必须与重定向后有效 clip 时长相差不超过 20ms，非循环自然完成以 Animator normalizedTime 首次达到 1 为准。`full_body` 必须 `movement=block`。

id 0..15 是 LLM 可长期依赖的核心语义，Unity 开发者可以更换动作素材，但不得改 id 或 semantic_key：

| id | semantic_key | 最低语义 |
|---:|---|---|
| 0 | greet | 普通问候姿态 |
| 1 | agree_nod | 点头同意 |
| 2 | disagree_shake_head | 摇头否定 |
| 3 | greet_wave | 友好挥手 |
| 4 | apologize_bow | 道歉/致意鞠躬 |
| 5 | explain | 解释性手势 |
| 6 | think | 思考姿态 |
| 7 | celebrate | 庆祝 |
| 8 | listen | 倾听姿态 |
| 9 | confused | 困惑 |
| 10 | point_left | 指向左侧 |
| 11 | point_right | 指向右侧 |
| 12 | point_forward | 指向前方 |
| 13 | shrug | 耸肩/不确定 |
| 14 | laugh | 开心笑的身体动作 |
| 15 | comfort | 安慰性姿态 |

扩展动作只能使用 id 16..126。完整 Prefab 必须实现上述 16 项；`sys.catalog` 仍上报实际执行元数据，让 Python 检查素材时长和兼容性。

expression 核心 id 固定为 `0 neutral / 1 happy / 2 sad / 3 surprised`，扩展只能用 4..126。SET\_EXPRESSION 的 127 清除最终淡化到 id 0。

### 10.3 唯一冲突规则

1. 没有当前动作时直接开始。
2. 新动作 priority 高于当前，或相等且当前 `interruptible=true`：先发旧动作 cancelled/replaced，再开始新动作。
3. 新动作 priority 更低：回 `action_busy`，不改变当前动作。
4. 当前 `interruptible=false`：任何 PLAY\_ANIM 替换都回 `action_busy`；STOP、STOP\_ACTION、safe idle、watchdog 与 ESTOP 仍可取消。
5. 开始 `movement=block` 动作时停止并保留当前移动目标；动作自然结束后不自动恢复移动，回 external。
6. 移动命令遇到 `movement=block` 动作：当前动作可打断则以 reason `movement` 取消后移动；不可打断则命令回 `action_busy`。
7. loop 请求只在条目 `loopable=true` 时有效，否则 `invalid_param`。
8. 同一 session 内重复 actionSeq 且 actionId/loop 不同，回 `action_seq_conflict`。

同一 session 内 actionSeq 不得代表第二次动作实例。已见过的 actionSeq 若 actionId/loop 相同，PLAY\_ANIM 成功 ACK 但为 no-op，不重新触发 started；元数据不同才回 `action_seq_conflict`。需要再次播放相同动作必须分配新 actionSeq。

每次真实开始、自然结束和取消必须分别发 `npc.action_started`、`npc.action_finished`、`npc.action_cancelled`。StateSync 持久同步 `actionId/actionSeq/loop/actionStartedAtServerMs`；晚加入者按服务器时间恢复归一化进度。动作不可伪装成从头新开始。

PLAY\_ANIM 的输出顺序固定为成功 `npc.ack`、`npc.operation_started`、`npc.action_started`；STOP\_ACTION/STOP 等显式取消的输出顺序固定为成功 ACK、`npc.action_cancelled`、`npc.operation_cancelled`。被新动作替换时无需额外停止命令：PLAY\_ANIM ACK 后先发旧动作 cancelled/operation\_cancelled，再发新动作 operation\_started/action\_started。

***

## 11. NavMesh 移动、注视与感知

VRChat Worlds 支持 Unity AI Navigation、NavMesh Surface、动态障碍和 Off-Mesh Link；v1.0“Udon 无 NavMesh”的前提作废。使用默认 Agent Type，不使用官方说明当前无法正确应用的自定义 Agent Type：[AI Navigation](https://creators.vrchat.com/worlds/udon/ai-navigation/)。

### 11.1 NavMeshAgent 固定配置

| 字段                         |                       值 |
| -------------------------- | ----------------------: |
| Agent Type                 |                 Default |
| Radius                     |                   0.25m |
| Height                     |                   1.60m |
| Base Offset                |                       0 |
| Speed                      |         最大 2.0m/s，由命令下调 |
| Acceleration               |                 4.0m/s² |
| Angular Speed              |                  180°/s |
| Stopping Distance          |                   0.30m |
| Auto Braking               |                    true |
| Update Position / Rotation | owner=true，remote=false |

GOTO 的处理顺序固定为：按 wire bounds 解码 → activity bounds 校验 → `NavMesh.SamplePosition` 半径 0.5m → 计算路径 → 仅完整路径接受 → 写目标和 ACK。任一步失败不得改变旧目标。到目标平面距离 ≤0.30m 且速度 <0.05m/s 时发 arrived；hasYaw 时再转到误差 ≤2° 后才发 arrived。

移动期若 1.5 秒净位移 <0.05m 且期望速度 >0.1m/s，或前向 `0°,±25°` 的 1.0m 环境射线连续两帧命中，停止并发 blocked（`stuck` 或 `local_obstacle`）。NavMesh 是寻路权威，射线只作局部安全门，不替代路径规划。

follow 每 0.5 秒把有效目标玩家位置采样到 NavMesh，目标距离 ≤1.2m 时停止。目标离开时发 `npc.blocked{reason:"target_left"}` 并回 external。

wander 是 `SET_MODE P3=3` 启动的无限 operation，不另占 cmdId。只有 Inspector `wanderWaypoints` 至少 2 项且发布 capability bit2 时才可启动，否则回 `unsupported_capability`。每次启动固定从 index 0 开始，按数组顺序行走；到最后一项后回到 index 0，不随机、不选择最近点、不跳过不可达点，也不进行社交决策。每到一项发 `npc.arrived{final:false,waypoint_index:<index>}` 后立即计算下一项；某项不可达按普通 no_path/blocked 终止。wander 永不自然 completed，只能被 STOP、模式替换、安全状态、看门狗、ESTOP 或 session 重建取消；STOP 的取消 reason 固定为 `explicit_stop`。

### 11.2 玩家、触摸、社交与 ray

- 玩家雷达 0.5 秒更新一次；pose 输出频率由 SET\_RATE 控制。
- TouchZones 使用普通 trigger collider 和能提供 `VRCPlayerApi` 的玩家 Trigger 事件，不用 VRCContact 代替身份来源。部位固定为 `head/cheek/handL/handR/torso`，同 slot+part 的 enter/exit 防抖 0.5 秒。
- `social.gaze`：头 forward 与指向 NPC 的夹角 <15°、距离 <3m 连续 2 秒为 on；超出 20° 或 3.5m 连续 0.5 秒为 off。
- `social.wave`：手高于头 0.1m，水平方向在 1.5 秒内至少两次换向且峰峰值 ≥0.25m；触发后同玩家冷却 3 秒。
- `social.approach`：距离 <5m、朝 NPC 径向速度 >0.3m/s 连续 0.5 秒，仅在进入条件时发一次，离开 5.5m 后重新武装。
- ray 起点为 NPC 根 + `(0,0.9,0)`，最大 8m，只命中 Inspector `environmentMask`，忽略玩家与自身。bearing 同样使用 §4.1 的右正 `brg` 约定；mode0 顺序为 `[0,45,90,135,180,-135,-90,-45]`，mode1 为 `[-30,-20,-10,0,10,20,30]`。无命中为 `-1`。

***

## 12. 网络同步与所有权

VRChat 同步变量由 owner 写入；Manual 需要显式 RequestSerialization，Continuous 自动发送，晚加入者会收到最新同步值：[Network Variables](https://creators.vrchat.com/worlds/udon/networking/variables/)、[Object Ownership](https://creators.vrchat.com/worlds/udon/networking/ownership/)。不得写“至少 10Hz 保证”之类超出平台承诺的要求。

| 对象            | 同步方式          | 字段/职责                                                                                                                             |
| ------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| RootSync      | VRCObjectSync | 根位置与旋转；owner 的 NavMeshAgent 驱动，remote 只消费同步                                                                                       |
| UpperBodySync | Continuous    | 8 个归一化 float、streamSeq、active；remote 平滑应用                                                                                         |
| StateSync     | Manual        | driverPid、session、controlState、locomotionMode、estop、actionId/Seq/loop/startServerMs、expression 状态、气泡文本/transferSeq/到期 server time |

所有 `*ServerMs` 字段使用 `Networking.GetServerTimeInMilliseconds()` 的有符号 32-bit 值；elapsed 必须用 unchecked 32-bit 差值处理回绕，结果再 clamp 到非负合理窗口。循环动作以 `elapsed % duration_ms` 对齐；非循环动作若远端收到时已超期，直接显示结束态，不回写 owner、不生成业务生命周期日志。文本到期采用相同回绕规则。

成功 DISCOVER 时，本地玩家必须对三个对象逐个 `Networking.SetOwner`，全部确认 owner 后才回成功 ACK；任一失败回 `ownership_failed` 并保持 safe idle。ownership 改变后，只有同步 `driverPid` 对应玩家且确为三个对象 owner 的客户端能执行 MIDI、NavMesh、业务日志和 Manual 写入。

非 owner 必须禁用 NavMeshAgent 的位置/旋转更新，只插值 VRCObjectSync 与同步状态。OnDeserialization 中 actionSeq 未变时不得重触发动作；变化时按 actionStartedAtServerMs 对齐。driver 离开后新 owner 只负责把状态置 safe idle、driverPid=-1 并同步，不得自动成为 driver。

StateSync 每次状态、动作、表情、文本或 driver 变化时 RequestSerialization 一次；同一帧多个变化合并成一次。网络拥塞时等待 `OnPostSerialization`/平台允许的下一次发送，不在 Update 中无界重试。VRChat 网络带宽与 Continuous 序列化大小存在平台限制，设计必须保持对象小而独立：[Networking Specs & Tricks](https://creators.vrchat.com/worlds/udon/networking/network-details/)。

***

## 13. 固定公网语音流

Unity 侧不做 TTS、不托管网关、不接收动态 URL。`fixedStreamUrl` 是 Inspector 的 `VRCUrl`，每次世界对象生命周期只调用一次 AVPro LoadURL；后续 SPEECH\_CUE 不能触发重载、seek 或换 URL。AVPro 支持直播流，但 VRChat 不保证低于 2 秒延迟：[Video Players](https://creators.vrchat.com/worlds/udon/video-players/)。自建域名若不在允许列表，测试账户必须显式允许不受信任 URL；这属于部署前置条件。

播放器状态机固定为：

```text
disabled（URL 空，仅开发诊断）
  或 loading --OnVideoReady--> ready --OnVideoStart--> playing
loading/ready/playing --OnVideoError--> error
playing --OnVideoEnd/Stop--> stopped
```

每次状态变化发 `voice.state`。完整交付中 URL 不得为空，`voice_stream` capability 必须存在。AudioSource 绑定 AudioAnchor，Spatial Blend=1，VRCSpatialAudioSource Gain=10，Near=0，Far=12m，Volumetric Radius=0；最终响度不得削波，若素材需要可下调 AudioSource volume。

SPEECH\_CUE 只记录外部预测：speechSeq、关联 text transferSeq、关联 actionSeq、预计延迟和时长。P1/P2 是不要求当前存在的关联编号，Unity 原样记录，不据此补发文本或动作。它不生成音频、不启动动作、不判断实际声音内容。speechSeq 在 session 内以一次 cue 为单位；相同 speechSeq 与相同元数据再次提交时成功 ACK 但不重复发 `voice.cue`，元数据不同回 `speech_seq_conflict`。播放器不是 `playing` 时回 `voice_unavailable`。

语音发布门槛由网关联调负责：两客户端、30 条发声，端到端 P95 <2000ms，且无 >2000ms 卡顿；否则**阻止语音功能发布**，不降级为“可能较慢”。Unity 侧仅需固定 URL 可播放、状态可测和空间化正确。

***

## 14. 上行日志与精确 JSON schema

### 14.1 公共格式

```text
[任意 VRChat 日志前缀][NEKO]{单行紧凑 JSON}
```

Python 必须查找一行中任意位置的第一个 `[NEKO]`，其后到行尾作为 JSON；不得假定标记在行首。JSON UTF-8 最多 950 字节，不含 `[NEKO]`；禁止换行、NaN 和 Infinity。

每条业务日志都必须包含以下公共头，顺序建议一致但解析不得依赖键顺序：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":42,"t":12.345,"type":"npc.state"}
```

| 字段        | 类型与范围             | 规则                                 |
| --------- | ----------------- | ---------------------------------- |
| v         | int，固定 1          | 破坏 wire 解析才升级                      |
| spec      | string，固定 `1.1`   | 语义版本                               |
| session   | int 0..268435455  | 未握手为 0，其余为当前 DISCOVER session      |
| world\_id | string 1..64      | Inspector 稳定值                      |
| npc       | string，固定 `yui`   | v1 单 NPC                           |
| log\_seq  | int 1..2147483647 | 每次世界加载从 1 递增；普通事件最大 2147483646，2147483647 专用于回绕标记；session 重建不清零 |
| t         | number ≥0         | `Time.timeSinceLevelLoad` 秒，3 位小数  |
| type      | string            | 只能取本文定义值；未来新增类型需保持向后兼容             |

字段值未知时用 JSON `null`；可选字段只有在下文明确写“条件字段”时才可省略。所有数值单位固定，不允许另加单位字符串。

Telemetry 在事件创建时、预算过滤和合并之前分配 `log_seq`。之后被丢弃的事件仍消耗该序号；多个周期事件合并成一行时，除实际输出的那个事件外，每个被压掉的原事件各消耗一个序号并计为 dropped。Python 因而可以用 gap 判断日志不完整，不得把 gap 当作解析器错误。

Telemetry 在任意滑动 1 秒最多发 20 行。优先级：ACK/ESTOP/内部错误 > session/operation/action/arrived/blocked/player join/leave > touch/social/ray/voice > state/pose/stream stats。预算满时先丢周期日志；关键事件进入 64 项有界队列并跨统计窗按原顺序排空，周期项可合并，operation/action 终态不得合并。对符合 §6.4 可靠命令速率、“至多 1 个普通未决命令”并持续使用独立 HEARTBEAT 保活通道的发送端，ACK 与 ESTOP 状态不得丢失。

Unity 按滑动 1 秒对 Channel 0 完整命令帧执行入口限流：上身流 active 时 2 帧/s，文本事务期间 4 帧/s，其余 4 帧/s；HEARTBEAT 计入总数。发送端的 `1/3/3` 普通命令限制用于给 HEARTBEAT 预留容量，Unity 入口仍按总 `2/4/4` 校验。超额帧不得执行、不得进入幂等缓存，也不视为可靠命令；每个 5 秒冷却窗只发一次 `sys.err{err:"rate_limited",source:"midi",related_seq:<首个超额帧seq>}`，其余仅累计内部计数。该规则确保恶意或失控发送端不能借失败 ACK 淹没日志。相同其他内部错误 5 秒冷却；已进入可靠处理的命令失败 ACK 不冷却。

丢行报告 `sys.telemetry` 不得丢弃，但连续报告可合并；在下一可用日志预算中输出：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":101,"t":12.5,"type":"sys.telemetry","dropped_since_last_report":1,"dropped_total":7,"first_dropped_log_seq":100,"last_dropped_log_seq":100,"wrap_count":0}
```

五个业务字段全部必填。没有新丢行时不得周期性发送；报告输出成功后才清零 since-last 计数。`dropped_total` 在本次世界对象生命周期内单调递增，session 重建不清零，并写入 snapshot session section。

普通事件只使用 log\_seq `1..2147483646`。下一次分配将越界时，必须先输出不可丢弃、不可合并的回绕标记，且暂停后续业务日志直到该标记真正写出：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":2147483647,"t":99.0,"type":"sys.log_wrap","previous_log_seq":2147483646,"next_log_seq":1,"wrap_count":1}
```

随后下一业务事件使用 log\_seq=1。`wrap_count` 每次世界加载从 0 开始并在标记中递增；Python 收到连续的 sys.log\_wrap 后必须把下一条 1 视为正常回绕，而不是巨大倒退。标记缺失时，倒退仍视为日志不完整。

### 14.2 ACK 与 NPC 状态

`npc.ack`：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":42,"t":12.345,"type":"npc.ack","seq":10,"cmd_id":2,"cmd":"GOTO_XZ","request_hash":"D34C","ok":true,"replayed":false,"state":"moving"}
```

必填业务字段：`seq:int 1..127`、`cmd_id:int`、`cmd:string`、`request_hash:string /^[0-9A-F]{4}$/`、`ok:bool`、`replayed:bool`、`state:control_state string`。`ok=false` 时必须增加 `err:error code string`；`detail` 是条件字段，最多 160 UTF-8 字节，不得含身份信息。未知 cmd 的 `cmd` 固定为 `UNKNOWN`。

`npc.state`：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":43,"t":12.5,"type":"npc.state","pos":[0.0,0.0,1.0],"yaw":0.0,"vel":[0.0,0.5],"speed":0.5,"state":"moving","mode":"goto","grounded":true,"target_slot":null,"action_id":null,"action_seq":null,"expression_id":null,"estop":false,"active_ops":["1193046:10:D34C"]}
```

所有字段必填。`pos` 为 XYZ，`vel` 为 XZ；目标/动作/表情不存在时为 null。grounded 取 NavMeshAgent 是否位于 NavMesh；未知时为 null。active_ops 按 operation 开始时间升序，最多 4 项，无操作时为空数组；未发布 `operation_lifecycle` 时该字段仍存在且固定为 `[]`。

移动事件：

- `npc.arrived`：`op_id:string`、`request_seq:int`、`request_hash:string`、`pos:[x,y,z]`、`yaw:number`、`error_m:number`、`final:bool`、`waypoint_index:int|null`。GOTO 到达时 final=true 且 waypoint\_index=null；wander 每个 waypoint 到达时 final=false 且填 index，不结束 operation。
- `npc.blocked`：`op_id:string|null`、`request_seq:int|null`、`request_hash:string|null`、`pos`、`reason:stuck|no_path|local_obstacle|target_left`、`target:[x,y,z]|null`。

#### 14.2.1 通用操作生命周期

GOTO、follow、wander、turn、look、action 和 expression 都是长操作。其稳定关联键固定为 `op_id="{session}:{request_seq}:{request_hash}"`。成功 ACK 后立即发 started；有自然终点的操作完成时发 completed；wander 没有自然终点；被 STOP、替换、看门狗、ESTOP、session 重建或玩家离开终止时发 cancelled。未通过 ACK 校验的命令不创建 operation。

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":44,"t":12.346,"type":"npc.operation_started","op_id":"1193046:10:D34C","kind":"goto","request_seq":10,"request_hash":"D34C","expected_end_ms":null}
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":45,"t":15.0,"type":"npc.operation_completed","op_id":"1193046:10:D34C","kind":"goto","request_seq":10,"request_hash":"D34C","elapsed_ms":2654,"result":"arrived"}
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":46,"t":13.0,"type":"npc.operation_cancelled","op_id":"1193046:10:D34C","kind":"goto","request_seq":10,"request_hash":"D34C","elapsed_ms":654,"reason":"explicit_stop"}
```

started 的 `expected_end_ms` 对已知时长 action/expression/look 为相对开始的预计毫秒，持续操作为 null。completed 的 result 只能是 `arrived/turned/expired/natural_end/cleared`；cancelled reason 只能是 `explicit_stop/replaced/movement/control_safe_idle/watchdog/estop/session_reset/player_left/stuck/local_obstacle/target_left`。同一 op_id 恰好一个 started，随后最多一个 completed 或 cancelled。

输出顺序固定为 ACK → operation_started → 领域 started；结束时领域 finished/arrived 先发，operation_completed 后发；取消时领域 cancelled/blocked 先发，operation_cancelled 后发。`npc.state` 和 snapshot 的 npc section 必须增加 `active_ops:[op_id...]`，按开始时间升序，最多 4 项。未发布 `operation_lifecycle` 时，仅省略三个通用 operation 日志，领域日志仍按其自身 capability 输出；active_ops 按 §7.4 固定为空。

### 14.3 动作生命周期 schema

三个类型公共必填字段相同：`op_id:string`、`request_seq:int`、`request_hash:string`、`action_id:int`、`action_seq:int 1..16383`、`semantic_key:string`、`name:string`、`layer:upper_body|full_body`。

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":50,"t":20.0,"type":"npc.action_started","op_id":"1193046:30:717D","request_seq":30,"request_hash":"717D","action_id":3,"action_seq":77,"semantic_key":"greet_wave","name":"wave","layer":"upper_body","loop":false,"started_at_server_ms":123456789}
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":51,"t":21.8,"type":"npc.action_finished","op_id":"1193046:30:717D","request_seq":30,"request_hash":"717D","action_id":3,"action_seq":77,"semantic_key":"greet_wave","name":"wave","layer":"upper_body","elapsed_ms":1800}
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":52,"t":20.7,"type":"npc.action_cancelled","op_id":"1193046:30:717D","request_seq":30,"request_hash":"717D","action_id":3,"action_seq":77,"semantic_key":"greet_wave","name":"wave","layer":"upper_body","elapsed_ms":700,"reason":"explicit_stop"}
```

started 额外必填 `loop:bool`、`started_at_server_ms:int`；finished 额外必填 `elapsed_ms:int`；cancelled 额外必填 `elapsed_ms:int`、`reason`，reason 只能取常量 JSON 的 `action_cancel_reason`。

### 14.4 上身流与文本 schema

`npc.stream_stats` 每 1000ms 一个统计窗，全部字段必填：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":60,"t":30.0,"type":"npc.stream_stats","interval_ms":1000,"received_frames":20,"applied_frames":20,"dropped_frames":0,"duplicate_frames":0,"invalid_events":0,"last_seq":42,"age_ms":18,"active":true,"suspended_for_text":false,"faded_to_neutral":false}
```

没有收到过帧时 `last_seq:null`、`age_ms:null`；其他计数为非负 int。一次 commit 只要至少一个参数实际驱动骨骼就计 applied\_frames；动作仲裁导致 8 项全未应用时只计 received。`active` 表示当前至少一个流参数正在应用，不等于最近收到过；`faded_to_neutral` 表示本统计窗内完成过断流淡出。`invalid_events` 汇总 Channel 1/2 的非协议事件、缺帧、文本事务外载荷和文本超额载荷；命令参数错误不计入此字段，因为已有失败 ACK。

文本事件：

- `npc.text_displayed`：`transfer_seq:int`、`utf8_bytes:int`、`crc16:string 4位大写HEX`、`display_seconds:int`、`text:string`。
- `npc.text_cleared`：`transfer_seq:int`、`reason:expired|replaced|explicit|control_safe_idle|watchdog|estop|session_reset`。

文本最多 384 UTF-8 字节，因此单行仍须在 950 字节预算内；若公共头加文本超限，`npc.text_displayed.text` 改为 `null` 并增加 `text_omitted:true`，但 `utf8_bytes/crc16` 必须保留。此规则只影响日志，不得截断气泡或网络文本。

### 14.5 玩家、触摸、社交与射线 schema

- `player.join`/`player.leave`：`pid:int`、`name:string`、`slot:int 0..63`、`count:int`。leave 中 name 是离开前缓存值。
- `player.pose`：`batch_seq:int 1..2147483647`、`page:int 1..pages`、`pages:int`、`players:array`，每页最多 4 项；项为 `{"slot":0,"pid":1,"d":1.25,"brg":-30.0,"yaw":180.0,"vr":true}`。未知 yaw/vr 为 null。
- `touch.enter`/`touch.exit`：`slot:int`、`pid:int`、`name:string`、`part:head|cheek|handL|handR|torso`。
- `social.wave`：`slot:int`、`brg:number`、`hand:left|right`。
- `social.gaze`：`slot:int`、`on:bool`、`d:number`、`brg:number`。
- `social.approach`：`slot:int`、`d:number`、`brg:number`、`radial_speed:number`。

上述 `player.pose`、`social.wave/gaze/approach` 的 `brg` 全部使用 §4.1 的同一坐标约定。Python 若要据此生成绝对朝向，只能使用 §4.1 的加法与 `wrap360`，不得另设 `turn_deg` 符号或按工具临时取反。
- `env.ray`：`request_seq:int`、`mode:0|1`、`bearings:[number]`、`d:[number]`；两个数组长度分别为 8 或 7，无命中值为 -1。

### 14.6 系统握手与 catalog schema

`sys.boot` 只在世界侧初始化完成时发一次，session=0，业务字段为 `ready:bool`。Python 不依赖这条日志，必须发 DISCOVER。

`sys.session`：`previous_session:int`、`new_session:int`、`driver_pid:int`、`reset:bool`、`estop_preserved:bool`。

`sys.hello` 的业务 schema：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_test_yui","npc":"yui","log_seq":3,"t":0.1,"type":"sys.hello","world_name":"YUI Test","wire_bounds":[-12.0,-1.0,-22.0,12.0,5.0,22.0],"activity_bounds":[-10.0,0.0,-20.0,10.0,4.0,20.0],"max_speed":2.0,"watchdog_ms":3000,"caps":["goto","follow","actions","expressions","text_preset","text_utf8","upper_body_stream","ray_scan","touch","player_pose","voice_stream","snapshot","navmesh","social_signals","anchors","operation_lifecycle"],"cap_bits":131067,"catalog_rev":1,"catalog_counts":{"action":16,"expression":4,"text_preset":4,"anchor":3}}
```

所有字段必填。wire_bounds 和 activity_bounds 顺序均为 minX,minY,minZ,maxX,maxY,maxZ。caps 按 capability bit 升序；`cap_bits=Σ(1<<bit)`。wander 只有配置至少 2 个 waypoint 时才发布；完整交付必须发布 anchors 与 operation_lifecycle。

`sys.catalog` 每行严格一个 item：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_test_yui","npc":"yui","log_seq":4,"t":0.15,"type":"sys.catalog","catalog_rev":1,"kind":"action","page":1,"pages":16,"items":[{"id":3,"name":"wave","semantic_key":"greet_wave","description_zh":"向目标友好挥手","intent_tags":["greeting","friendly"],"target_required":"player","speech_compatible":true,"layer":"upper_body","duration_ms":1800,"loopable":false,"movement":"allow","priority":40,"interruptible":true,"fade_in_ms":120,"fade_out_ms":180}]}
```

`catalog_rev` 为 `1..2147483647` 的 Inspector 整数，目录任何字段变化都必须递增。`kind=action` 时 item 必须是 §10.2 的完整对象；`kind=expression` 时 item 为 `{"id":0,"name":"neutral","semantic_key":"neutral","description_zh":"中立表情","default_duration_ms":0,"fade_ms":150}`，其中 duration `0..600000ms`、fade `0..5000ms`；`kind=text_preset` 时 item 为 `{"id":0,"name":"hello","text":"你好","default_display_seconds":5}`，其中 text 为 1..384 UTF-8 字节、秒数 1..127。

`kind=anchor` 时 item schema 固定为：

```json
{"id":0,"semantic_key":"stage_center","description_zh":"舞台中央","pos":[0.0,0.0,4.0],"yaw":180.0,"has_yaw":true,"arrival_radius":0.3,"tags":["stage","social"]}
```

anchor id 为 0..126 且唯一；semantic_key 匹配 `/^[a-z0-9_]{1,32}$/`；description_zh 为 1..80 UTF-8 字节；pos 必须位于 activity bounds 且能以 0.5m 半径采样到 NavMesh；yaw 为 `[0,360)`，has_yaw=false 时 yaw 必须为 0；arrival_radius 为 0.1..2.0m；tags 为 1..8 个语义字符串。完整 Prefab 至少提供 3 个 anchor。Python 选择 anchor 后仍使用现有 GOTO_XZ 编码其 pos/yaw，不新增 MIDI 命令。

四类 id 都为 `0..126` 且各 kind 内唯一，name/semantic_key 按各自 schema 校验。空目录用 `page=1,pages=1,items=[]`，但完整交付不得出现 action、expression 或 anchor 空目录。

`sys.pong`：`seq:int`、`watchdog_age_ms:int`。`sys.watchdog`：`elapsed_ms:int`、`previous_state:string`。

### 14.7 voice schema

`voice.state` 全部字段必填；不记录 URL：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":70,"t":40.0,"type":"voice.state","state":"playing","reason":"video_start","error_code":null,"url_loaded":true}
```

state 只能是 `disabled/loading/ready/playing/error/stopped`；reason 只能是 `world_start/video_ready/video_start/video_error/video_end/video_stop`；非错误时 error\_code 为 null，错误时为 AVPro/VRChat 回调枚举的稳定字符串，未知枚举写 `UNKNOWN_<int>`，不得伪造含义。

`voice.cue`：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":71,"t":40.2,"type":"voice.cue","speech_seq":91,"text_transfer_seq":7,"action_seq":77,"estimated_delay_ms":800,"duration_ms":6000}
```

五个业务字段全部必填，未关联的 text/action 用 0。cue 是预测关联，不表示声音已真正到达。

### 14.8 完整 snapshot schema

SNAPSHOT\_REQUEST 输出一组 `sys.snapshot`，公共业务字段为：`snapshot_seq`（请求 seq）、`part`、`parts`、`section`、`data`。part 从 1 开始连续，顺序固定为 session、npc、players（slot 升序，每页最多 4 人，空也发一页）、voice、text。catalog 内容不属于运行期状态，session section 的 `catalog_rev/counts` 是完整性引用；需要内容时重发 DISCOVER。

各 section 的 data 必须是：

- `session`：`driver_pid:int|null`、`control_state:string`、`watchdog_age_ms:int|null`、`estop:bool`、`catalog_rev:int`、`catalog_counts:object`、`caps:array`、`telemetry_dropped_total:int`、`log_wrap_count:int`。
- `npc`：与 `npc.state` 相同（含 active_ops），再增加 `target_pos:[x,y,z]|null`、`action_started_at_server_ms:int|null`、`text_transfer_seq:int|null`。
- `players`：`{"page":1,"pages":1,"players":[...]}`，每项是 player.pose item 再增加必填 `name:string`，因此 snapshot 可独立重建 slot↔pid↔name 映射。
- `voice`：`{"state":string,"error_code":string|null,"url_loaded":bool,"last_speech_seq":int|null}`。
- `text`：`{"transfer_seq":int|null,"utf8_bytes":int,"crc16":string|null,"display_until_server_ms":int|null,"text":string|null}`。

每个 part 的 `parts` 必须相同，且只有收到 `1..parts` 全部 part 才算 snapshot 完成。session 在输出期间变化时，立即终止旧组；新 session 的请求方必须重试。

只要发布 `snapshot`，五类 section 及字段就不得因其他 capability 缺失而删减：缺 `player_pose` 仍返回 players，缺 `voice_stream` 时 voice data 固定为 `{"state":"disabled","error_code":null,"url_loaded":false,"last_speech_seq":null}`，缺 `operation_lifecycle` 时 npc.active_ops 固定为 `[]`。未发布 `snapshot` 时则按 §7.4 拒绝命令，不输出部分 snapshot。

### 14.9 错误 schema 与冻结错误码

命令可预期失败只发 `npc.ack ok=false`。内部、异步或无对应命令的失败发：

```json
{"v":1,"spec":"1.1","session":1193046,"world_id":"wrld_example","npc":"yui","log_seq":80,"t":50.0,"type":"sys.err","err":"text_timeout","code":27,"source":"text","fatal":false,"related_seq":7,"detail":"transfer timed out"}
```

必填字段：`err:string`、`code:int`、`source:midi|safety|locomotion|action|stream|text|perception|network|voice|telemetry`、`fatal:bool`、`related_seq:int|null`、`detail:string|null`。err 与 code 必须严格匹配下表；detail 最多 160 UTF-8 字节。

| code | err                      | 含义                                             |
| ---: | ------------------------ | ---------------------------------------------- |
|    1 | unknown\_cmd             | cmdId 未定义                                      |
|    3 | not\_handshaken          | 尚无有效 session                                   |
|    4 | not\_driver              | 本地玩家不是已 claim driver                           |
|    5 | not\_owner               | driver 未持有全部同步对象                               |
|    6 | invalid\_state           | 命令在当前状态无定义                                     |
|    7 | estop\_latched           | ESTOP 锁存阻止命令                                   |
|    8 | invalid\_param           | 数值、组合或元数据非法                                    |
|    9 | reserved\_bits           | 保留位/寄存器非零                                      |
|   10 | target\_out\_of\_bounds  | 目标越出三维 AABB                                    |
|   11 | target\_not\_on\_navmesh | 0.5m 内无可用 NavMesh 点                            |
|   12 | no\_path                 | NavMesh 无完整路径                                  |
|   13 | target\_missing          | follow/goto 缺有效目标                              |
|   14 | slot\_unknown            | slot 未分配或已离开                                   |
|   15 | unsupported\_capability  | Prefab 未发布该能力                                  |
|   16 | action\_not\_found       | actionId 不在目录                                  |
|   17 | action\_busy             | 动作优先级/不可打断规则拒绝                                 |
|   18 | expression\_not\_found   | expressionId 不在目录                              |
|   19 | text\_preset\_not\_found | presetId 不在目录                                  |
|   20 | transfer\_busy           | 已有不同文本事务                                       |
|   21 | transfer\_missing        | COMMIT 时没有事务                                   |
|   22 | transfer\_seq\_mismatch  | BEGIN/COMMIT transferSeq 不同，或复用已完成 transferSeq |
|   23 | text\_too\_long          | UTF-8 长度 >384 或为 0                             |
|   24 | length\_mismatch         | 收到的原始字节数不等于声明                                  |
|   25 | crc\_mismatch            | CRC16 不一致                                      |
|   26 | invalid\_utf8            | 严格 UTF-8 解码失败                                  |
|   27 | text\_timeout            | 5 秒内未提交                                        |
|   28 | stream\_incomplete       | session 首帧缺少一个或多个 CC                           |
|   29 | seq\_conflict            | 5 秒内同 seq 不同请求                                 |
|   30 | voice\_unavailable       | 固定流当前不是 playing                                |
|   31 | ownership\_failed        | 无法取得全部同步对象 owner                               |
|   32 | session\_conflict        | 非当前 driver 尝试替换活跃 session                      |
|   33 | rate\_limited            | Channel 0 完整帧超出入口硬速率；异步 `sys.err`，超额帧不产生 ACK |
|   34 | internal\_error          | 未归类内部异常                                        |
|   35 | driver\_auth\_failed     | claim code 或非空显示名门失败                           |
|   36 | action\_seq\_conflict    | actionSeq 对应不同动作元数据                            |
|   37 | speech\_seq\_conflict    | speechSeq 对应不同 cue 元数据                         |
|   38 | catalog\_invalid         | 目录重复、字段缺失或 Animator 状态不可达                      |

错误码 2 在 v1.1 起保留，不得发送、不得复用于其他含义；其余未列入本表的 err 字符串也不得发送。新增错误必须升级配套三文件的小版本。版本不兼容由 Python 检查 hello 的 v/spec，不对应 Udon 命令错误，因为当前 DISCOVER wire 中没有版本字段。

***

## 15. Inspector 配置表

| 路径/字段                      | 类型           | 默认/要求              | 验收               |
| -------------------------- | ------------ | ------------------ | ---------------- |
| YUI\_NPC.worldId           | string       | 必填，1..64           | 与 hello 一致       |
| YUI\_NPC.npcId             | string       | 固定 `yui`           | 不可编辑或校验失败        |
| YUI\_NPC.driverDisplayName | string       | 发布时必填；本地自动化可空      | 非空精确匹配           |
| YUI\_NPC.driverClaimCode   | int          | 必填 0..16383        | 与 DISCOVER P2 一致 |
| YUI\_NPC.wireBoundsMin/Max | Vector3      | 必填，三轴 min\<max     | hello 与量化测试一致    |
| YUI\_NPC.activityBoundsMin/Max | Vector3   | 必填，严格位于 wire 内且各侧留 1m | 越界向量可被拒绝 |
| RootSync.maxSpeed          | float        | 2.0m/s             | 不得运行期超过          |
| RootSync.acceleration      | float        | 4.0m/s²            | Agent 一致         |
| RootSync.angularSpeed      | float        | 180°/s             | Agent 一致         |
| RootSync.stoppingDistance  | float        | 0.30m              | 到达误差通过           |
| Locomotion.followDistance  | float        | 1.20m              | 跟随停止正确           |
| Locomotion.environmentMask | LayerMask    | 必填                 | 不含 Player/NPC 层  |
| Locomotion.wanderWaypoints | Transform\[] | 可空；发布 wander 时至少 2 | 空时不发布 capability |
| Telemetry.stateHz/poseHz   | int          | 5/2                | SET\_RATE 可覆盖    |
| Telemetry.logBudget        | int          | 固定 20              | Inspector 不得设高   |
| Safety.watchdogMs          | int          | 固定 3000            | 不得改语义            |
| Text.maxUtf8Bytes          | int          | 固定 384             | 不得改语义            |
| Voice.fixedStreamUrl       | VRCUrl       | 完整交付必填             | 世界加载仅调用一次        |
| Voice AudioSource          | reference    | 必填                 | 绑定 AudioAnchor   |
| Action catalog             | array        | §10.2 核心 id 0..15 全部存在 | 语义键、Animator 状态和元数据自检通过 |
| Expression catalog         | array        | 核心 id 0..3 全部存在    | 语义键与 Animator 参数自检通过 |
| Text preset catalog        | array        | 可空                 | catalog 正确       |
| Anchor catalog             | array        | ≥3，id/semantic\_key 唯一 | 位于 activity bounds 且可采样到 NavMesh |
| VRC Midi Listener          | component    | NoteOn+CC active   | 指向 MidiRouter    |

Prefab `Start` 必须先做引用、四类目录、双 bounds、同步对象和 Animator 状态自检。阻塞性错误发 `sys.err{catalog_invalid/internal_error,fatal:true}`，保持 unhandshaken，DISCOVER 回对应错误。

***

## 16. Build & Test 与两客户端验收

### 16.1 操作步骤

1. 用 VCC 建 Worlds 工程，确认 Unity `2022.3.22f1`；把实际稳定 SDK 精确版本提交到 manifest/lock 文件。
2. 导入模型与动作，完成 Humanoid 重定向；确认 Animator Apply Root Motion 关闭、四层顺序和 AvatarMask。
3. 将 Prefab 放入 `YUI_NPC_Demo.unity`，设置 worldId、driver、claim code、wire/activity bounds、固定 VRCUrl、环境层、anchor 和 waypoint。
4. 为地面/墙体/家具加 collider，在默认 Agent Type 上烘焙 NavMesh；打开 Navigation 视图检查 0.25m agent 可达区。
5. 安装 loopMIDI 并创建 `NEKO_MIDI`；Editor 测试在 VRChat SDK MIDI Utility 选择该设备。客户端使用启动参数 `--midi=NEKO_MIDI`。
6. VRC Midi Listener 只启用 NoteOn 与 ControlChange，Behaviour 指向 MidiRouter；不得手工添加运行期 `VRCMidiHandler`。
7. 先在 Editor/ClientSim 跑无网络状态机与测试向量，再用 Build & Test 启动两个客户端。只有 driver 客户端连接发送端；第二客户端仅旁观。
8. 两客户端都允许固定流域名（若需要 Untrusted URLs），验证空间音频与播放器状态。

### 16.2 必须通过的验收

1. JSON 两附件可解析；命令 id、CC、范围、CRC 和错误码与本文一致。
2. DISCOVER 后按固定顺序收到 ACK/session/hello/catalog；重复 DISCOVER 不重置状态。
3. GOTO wire 最小、中点、最大和无 yaw 测试向量逐项一致；wire 极值因越出 activity bounds 被拒绝且旧目标不变，中点和无 yaw 正常执行。
4. 重发同请求只 replay ACK；同 seq 不同 hash 回 seq\_conflict。
5. 停心跳 3 秒进入 safe idle；恢复心跳不恢复任务。
6. 移动中 ESTOP 在下一 Update 前停住；SET\_MODE/STOP/SET\_CONTROL 均不能解锁；CLEAR 后为 safe idle。
7. NavMesh 空旷 GOTO 落点误差 ≤0.30m；墙后能绕行或明确 no\_path/blocked，不穿墙。
8. 8 参数最小/中立/最大帧与测试向量一致；断流 250ms 后在 200ms 淡回中立。
9. 文本 `你好，YUI 👋` 的 UTF-8、packing、CRC、显示、同步与向量逐字节一致；错误 CRC 不显示。
10. 16 个核心动作的 id、semantic\_key、语义元数据和 Animator 状态逐项一致；动作与通用 operation 的 started/finished/cancelled 各出现一次且 actionSeq/opId 对账；晚加入者看到正确进度，不重复触发。
11. 玩家 join/leave 的 slot 映射及时作废；pose 分页不超过 4 人/行；触摸 identity 正确。
12. SNAPSHOT 的 part 连续、sections 完整、运行状态与周期日志一致。
13. 第二客户端移动平滑，控制/ESTOP/动作/文本晚加入状态正确；remote 不执行 MIDI。
14. AVPro 每次世界加载只 LoadURL 一次，ready/playing/error 状态日志正确，AudioSource 空间化来自 NPC。
15. 名义 20 Hz 上身 + 1 Hz HEARTBEAT 为 190 msg/s；同一滑窗再有一条普通可靠命令时上身最多 19 帧且合计 ≤191；文本载荷 159 + HEARTBEAT 一条 + 普通命令三条 =199；任一正常路径 ≤199，ESTOP 后 ≤200。
16. 全部 23 条命令逐状态核对 §7.2 矩阵；构造多重非法请求，确认只返回 §7.3 中优先级最高的错误且没有副作用。
17. 压力下 `[NEKO]` 任意滑动 1 秒 ≤20 行，单个 JSON ≤950 UTF-8 字节；合规发送端关键 ACK 不丢；超额命令不执行且 `rate_limited` 每 5 秒最多一条。
18. Constants JSON 中全部 37 个有效错误码都必须至少被一个独立向量触发，错误名、数值、判定阶段和副作用与 §7.3/§14 一致；数值 2 仅为保留位，不得触发或复用。
19. wander capability 关闭时明确拒绝；开启时从 waypoint 0 按数组循环，经过最后一项回 0，STOP 只产生 cancelled、不产生 completed。
20. 人工压满遥测窗并丢弃周期事件，确认被丢事件消耗 log\_seq、下一条出现 gap 且 sys.telemetry 计数一致；在 2147483646 后输出 sys.log\_wrap，再从 1 继续。
21. DISCOVER 返回至少 3 个有效语义 anchor；Python 可仅凭 semantic\_key 解析并走到每个 anchor，LLM 层不接触坐标量化或 MIDI。
22. TURN、LOOK、expression、移动与动作都产生唯一 opId，并最终进入 completed/cancelled；snapshot 的 active\_ops 与生命周期日志一致。
23. 三附件中的成功与失败向量全部通过；失败向量不得改变旧目标、动作、文本、控制态或同步状态（ESTOP/CLEAR 的规定状态变化除外）。
24. 两客户端 30 条公网语音端到端 P95 <2 秒且无 >2 秒卡顿；失败则语音发布被阻断。
25. 审查世界代码：除生成 TTS 音频经外部网关外，没有上传玩家感知、日志、画面或身份的路径。
26. 运行 `bearing_right_positive_to_turn_q14` 向量：`npc_yaw=350°`、`brg=+20°` 必须得到绝对 yaw `10°` 和 TURN\_TO `P0=455`；若使用减法、左正号或漏掉回绕，向量必须失败。
27. 在未设置 `free_coordinate_navigation` 的全新配置中，LLM 工具 schema 不得包含 `npc.go_to_xyz`；anchor 只能解析 Inspector 已发布项，wander 不得接受方向/距离参数。
28. Constants JSON 的 17 项 `capability_contracts.*.disabled_test_vector` 必须全部存在并通过；其中命令门返回 `unsupported_capability`，无 ACK 上行门严格零日志/零状态变化，navmesh 不得降级直线移动或误回 `no_path`。
29. 运行 `heartbeat_bypasses_normal_in_flight`：普通命令 ACK 丢失并在 2 秒原样重发期间，1Hz HEARTBEAT 必须使用新 seq 持续发送；3 秒看门狗不得触发，控制态保持 external，普通重发只重放 ACK。
30. 运行 `set_control_external` 的语义适配断言：`npc.arm` 只展开为一次 SET\_CONTROL\_MODE P3=1；任何移动、动作或复合工具不得自动插入 arm，宿主授权被清除后不得发送该命令。
31. 运行 `operation_lifecycle_capability_disabled`：`npc.observe` 必须同时返回 `operation_lifecycle=false`、`active_ops_authoritative=false` 和 `active_ops=[]`，字段不得省略，模型不得用该空数组推断任务完成。

测试报告必须记录 Unity、Worlds SDK、UdonSharp（若使用）、场景 commit、Prefab commit、两客户端账号/角色说明、测试时间和每项证据。不能用“肉眼大致正常”代替协议向量结果。

***

## 17. 面向 LLM 的语义控制边界

LLM 不得直接生成 cmdId、MIDI 事件、seq、CRC、量化值、actionId、expressionId 或玩家 pid。Python 必须在协议编解码器之上提供语义适配层：启动时用 DISCOVER 获取 catalog，以 `semantic_key` 建索引；玩家只向模型暴露 session 内 slot、相对距离/方位和经授权的称呼；地点优先暴露 anchor。这样模型表达意图，确定性代码负责范围、状态、安全和传输。

### 17.1 最小工具面

工具名和输入语义固定如下；具体函数调用框架可变，但不得让模型越过这些字段写 wire 值。

| 工具 | 输入 | Python 确定性展开 |
| --- | --- | --- |
| `npc.observe` | 无 | 必要时 SNAPSHOT，返回控制态、caps、operation\_lifecycle、active\_ops\_authoritative、active\_ops、可见玩家 slot、最近感知、voice 状态和可用语义键 |
| `npc.arm` | 无 | 当前 session 已获宿主授权且处于 safe\_idle 时发送 SET\_CONTROL\_MODE P3=1；成功后为 external |
| `npc.go_to` | `anchor_key`，可选 `speed_mps` | anchor→GOTO\_XZ；不得让模型复制 anchor 坐标 |
| `npc.go_to_xyz` | `x,z`，可选 `yaw,speed_mps` | **默认关闭**；仅当宿主显式配置 `free_coordinate_navigation=true` 后才可暴露。启用时先在 Python 校验 activity bounds，再 GOTO\_XZ；只供明确空间推理场景 |
| `npc.follow` | `player_slot` | SET\_TARGET 成功后 SET\_MODE follow |
| `npc.look_at` | `player_slot` 或 `x,y,z` 二选一，`duration_ms` | player\_slot→LOOK\_AT，坐标→LOOK\_AT\_XYZ；若宿主另做“转身朝向该玩家”复合工具，绝对 yaw 必须按 §4.1 由最新 `npc_yaw+brg` 计算 |
| `npc.act` | `action_key`，可选 `player_slot,loop` | 校验 catalog 与 target\_required；需要玩家时先 SET\_TARGET，再 PLAY\_ANIM |
| `npc.set_expression` | `expression_key,duration_ms` | semantic\_key→SET\_EXPRESSION |
| `npc.say` | `text`，可选 `action_key,estimated_delay_ms,duration_ms` | 文本事务；有外部语音时再发 SPEECH\_CUE，动作关联必须使用同次已确认 actionSeq |
| `npc.stop` | `scope=all|movement|action` | all→STOP，movement→SET\_MODE idle，action→STOP\_ACTION |
| `npc.estop` | `reason` | 直接走最高优先级 ESTOP；不得等待普通队列 |

`npc.arm` 是进入 external 的唯一 LLM 工具，不得在 `npc.go_to`、`npc.follow`、`npc.act` 或任何其他工具前自动插入。宿主必须先为当前 session 设置 `host_arm_authorized=true`，适配层才能向模型暴露该工具；LLM 本身不能设置此标志。该授权在 session 重建、看门狗触发、ESTOP、driver 离开或 owner 丢失时立即清除。调用时若已在 external/moving/action，适配层不发 MIDI，直接返回 `succeeded` 并带 `already_external=true`；未握手、estop 或未获宿主授权时返回适配层本地失败 `arm_not_authorized`，不得发送命令。其他工具在 safe\_idle 中不得代为 arm，只能返回可恢复提示 `requires_arm`。

`CLEAR_ESTOP` 默认不作为 LLM 工具暴露，只能由明确的人类操作或宿主安全策略批准；批准后仍只进入 safe idle，不自动恢复 external。`free_coordinate_navigation` 缺失时必须按 `false` 处理；禁用时不得把 `npc.go_to_xyz` 放入模型工具 schema，模型只能通过 anchor 或玩家 slot 指定目标。

anchor 只表示 Unity 开发者在 Inspector 中预先配置、随世界发布的静态命名地点。它解决“前往已命名地点”，不提供视觉目标检测、动态物体跟踪或任意物体导航；例如要支持“去右边那块大手表”，世界侧必须为该手表附近预置合法 anchor。未配置的物体不得由 Python 猜测为 anchor。

wander 只表示 §10.3 的预置 waypoint 循环巡逻，不是相对位移或几何推理工具，不接受方向、距离或 `turn_deg`。它不能用于“向某方向走一段”或“分段接近静态物体”。最小工具面默认不暴露 wander；部署方若额外暴露 `npc.wander`，它只能无参数展开为 `SET_MODE P3=3`，停止仍使用 `npc.stop(scope=movement)`。

未发布 `operation_lifecycle` 时，Python 不得等待不存在的 `npc.operation_*`，也不得把始终为空的 `active_ops` 当成“操作已经完成”。最小 LLM 工具面必须隐藏会创建 operation 的 go\_to/go\_to\_xyz/follow/look\_at/act/set\_expression；宿主直接调用 wire 命令仍按 §7.4 执行，但只能依据领域终态日志判断结果，没有领域终态证据时最终状态必须为 `unknown`。

### 17.2 统一结果与闭环

每次非 observe 调用由 Python 返回并持续更新同一个结果对象：

```json
{"request_id":"host-uuid","status":"accepted","op_id":"1193046:10:D34C","kind":"goto","semantic_key":"stage_center","wire_seq":10,"error":null}
```

`status` 只能是 `accepted/running/succeeded/cancelled/failed/unknown`。成功 ACK 得到 accepted；收到 operation\_started 得到 running；operation\_completed 得到 succeeded；operation\_cancelled 得到 cancelled；失败 ACK 得到 failed 并原样保留 wire error code。即时命令没有 operation 时，成功 ACK 直接得到 succeeded。若普通命令 ACK 超时，Python 先按 §6.3 原样重发，同时不得暂停独立 HEARTBEAT；仍无法确定时请求 snapshot，只要没有足够证据证明执行结果就必须返回 unknown，不得向 LLM 伪报成功。

复合工具按子命令串行执行，任一子命令失败即停止后续步骤，并返回失败子步骤。例如 `npc.act` 的 SET\_TARGET 失败时不得发送 PLAY\_ANIM。`npc.say` 的文本成功只代表气泡已经提交；voice cue 仍是预计关联，不能被解释为玩家已经听见声音。

### 17.3 提供给模型的观察摘要

适配层每轮只提供最新 snapshot 加自上轮以来的领域事件，不把原始高频日志灌入上下文。最小摘要必须包含：`session/world_id/control_state/estop`、完整 `caps` 字符串数组、`operation_lifecycle` 布尔值、`active_ops_authoritative` 布尔值、始终存在的 `active_ops` 数组、NPC 位置与朝向、当前动作/表情/文本、玩家 `slot,d,brg,vr`、最近 touch/social 事件、voice state、action/expression/anchor 的 semantic\_key。`operation_lifecycle` 严格等于 `caps` 是否包含同名项，`active_ops_authoritative` 严格等于 `operation_lifecycle`；能力未发布时两者均为 false 且 `active_ops=[]`，模型不得把该空数组解释为“没有任务正在运行”。真实显示名默认不提供；只有宿主明确授权时才映射为称呼。

```json
{"control_state":"external","caps":["goto","operation_lifecycle"],"operation_lifecycle":true,"active_ops_authoritative":true,"active_ops":[]}
```

模型提出当前状态不允许、目录不存在或需要缺失目标的意图时，适配层应在发送 MIDI 前拒绝并给出可恢复提示，例如“当前为 safe\_idle，需先获准 external”或“动作需要 player\_slot”。世界侧矩阵和错误码仍是最终权威，Python 预检查不得替代 ACK。

***

## 18. 版本兼容规则

- `v` 只在 JSON/MIDI 物理 wire 发生破坏性变化时升级；`spec` 在命令语义、范围或状态机变化时升级。
- Python 遇到未知 JSON 字段必须忽略；未知 `type` 可记录但不得让解析器崩溃。
- Udon 遇未知 cmdId 回 `unknown_cmd`，不执行。
- Python 只有在 hello 的 `v=1` 且 spec 主版本为 `1` 时才可 SET\_CONTROL\_MODE external；否则只允许 DISCOVER、HEARTBEAT、SNAPSHOT\_REQUEST 和 ESTOP。
- capability 缺失行为必须逐项遵守 §7.4；只有该表明确标为“静默丢弃/不输出”的无 ACK 上行入口可以静默，命令入口不得以 no-op 冒充成功。
- v1.1 冻结后，任何常量修改必须同时修改 Markdown、Constants JSON、TestVectors JSON，并重新跑一致性验证。

***

## 19. 官方依据与实现注意

- [VRChat 当前 Unity 版本](https://creators.vrchat.com/sdk/upgrade/current-unity-version/)
- [Udon Realtime MIDI 与](https://creators.vrchat.com/worlds/udon/midi/realtime-midi/) [`--midi=`](https://creators.vrchat.com/worlds/udon/midi/realtime-midi/)
- [MIDI 事件取值范围](https://creators.vrchat.com/worlds/udon/midi/)
- [Udon AI Navigation / NavMesh](https://creators.vrchat.com/worlds/udon/ai-navigation/)
- [Network Variables 与晚加入者](https://creators.vrchat.com/worlds/udon/networking/variables/)
- [Object Ownership](https://creators.vrchat.com/worlds/udon/networking/ownership/)
- [Networking Specs & Tricks](https://creators.vrchat.com/worlds/udon/networking/network-details/)
- [AVPro / Video Players](https://creators.vrchat.com/worlds/udon/video-players/)
- [Debugging Udon Projects / output log](https://creators.vrchat.com/worlds/udon/debugging-udon-projects/)

本规范刻意不把平台未承诺的实时性写成保证：driver 本机 MIDI 上身流名义可以按 20 Hz 输入，并在命令预算冲突时按 §6.4 降为 19 帧；远端 Continuous 同步频率与 AVPro 直播延迟均受 VRChat 和网络条件约束。所有发布结论必须来自两客户端实测。
