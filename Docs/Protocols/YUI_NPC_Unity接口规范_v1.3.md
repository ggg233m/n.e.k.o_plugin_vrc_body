# YUI NPC Unity 接口规范 v1.3（冻结）

本文件是 v1.2 的向后兼容增量。wire 版本仍为 1；v1.1、v1.2 的命令、目录、日志和测试向量字节不变。v1.3 只在世界明确发布相应 capability 时启用。

## 1. Capability

- bit 19 `region_localization`：Unity 发布当前区域、楼层和最近 Anchor 的语义定位事实。
- bit 20 `local_navigation`：Unity 接受相对移动，并在一个 operation 内持续探索 Region。

缺少 capability 时，对应 wire 命令返回 `unsupported_capability`，适配层不得暴露 `npc.move_relative`。`npc.explore` 在缺少 `local_navigation` 的 v1.2 世界维持 Anchor 遍历行为。

## 2. 命令

### 2.1 `0x19 MOVE_RELATIVE`

- P0=`distance_mm`，250..10000。
- P1=0。
- P2=`bearing_q14`；0°前、90°右、180°后、270°左，相对命令接收时 NPC 朝向。
- P3=0。
- P4=`speed_q7`。
- P5 bit0=`face_travel`，bit1=`allow_shorter`，bits2..6=0。

命令创建 `kind=move_relative` 的 operation。目标方位必须严格保持，不得为绕障碍擅自左右偏转。完整距离无完整 NavMesh 路径时，`allow_shorter=true` 依次检查原距离的 100%、80%、60%、40%、20%，低于 0.25 米的候选跳过；首个仍位于 NavMesh 且完整路径可达的候选成为目标。无候选返回 `no_path`。`face_travel=false` 时保留既有注视/朝向控制。

### 2.2 `0x1A EXPLORE_REGION`

- P0=`duration_deciseconds`，10..6000，即 1..600 秒。
- P1=0，P2=0。
- P3=`region_id`，0..126。
- P4=`speed_q7`。
- P5：0=`unvisited`，1=`patrol`，bits1..6=0。

目标 Region 必须存在且 `explorable=true`，否则不得启动 operation。命令创建 `kind=explore` 的单个 operation；整个期限内由 Unity 连续 retarget，中间目标不得创建新的 ACK 或 operation。

`unvisited` 在目标 Region 体积内生成最多 12 个确定性候选。候选必须位于目标 Region、成功投影到 NavMesh 且从当前位置具有完整路径；从有效候选中选择与最近 8 个历史目标的最小距离最大者。连续三轮无候选后以 `no_path` 失败。`patrol` 按 Region 内 Anchor id 升序循环，只采用完整路径可达的 Anchor。

到目标约 0.4 米时提前切换下一目标，`NavMeshAgent.autoBraking=false`。只在期限结束、取消、ESTOP 或失败时制动。

## 3. Region 体积与定位

Region 目录增加 `explorable:boolean`。一个 Region 可由 Inspector 明确绑定一个或多个体积 Transform；其本地单位立方体定义体积，允许平移、缩放和 Y 轴旋转，X/Z 倾斜为静态校验错误。

NPC 同时命中多个体积时，依次选择：较高 `priority`、较小世界体积、较小 `region_id`。没有命中时发布 `localized=false`，禁止用最近 Anchor 冒充区域。最近 Anchor 可以独立存在，但只发布语义键、距离 `d` 和相对方位 `brg`。

`npc.state` 的 location 投影为：

```json
{"localized":true,"region_key":"ground_floor","floor_label":"一楼","nearest_anchor":{"semantic_key":"spawn","d":1.2,"brg":35.0}}
```

不得在 LLM 可见的 `npc.observe` 或 `npc.world_query` 中发布 NPC、Region、Anchor、候选点或 NavMesh 的绝对坐标。

## 4. 生命周期与安全

`MOVE_RELATIVE` 和 `EXPLORE_REGION` 遵循 v1.2 的可靠命令、状态权限、单移动域和 operation lifecycle。session 重建、watchdog、ESTOP、所有权丢失、驱动者离开、断开连接或 `replace_active=true` 的新移动计划都会立即取消当前相对移动/探索并制动。

v1.3 新增 `npc.operation_failed` 终态事件，字段与既有 operation 关联字段一致，并携带冻结错误名 `err` 与 `elapsed_ms`。运行期 `no_path`、`stuck` 等执行失败必须使用该事件；人工停止、替换和安全边界变化仍使用 `npc.operation_cancelled`。

空闲状态不得自动探索。`GOTO_XZ`/`npc.go_to_xyz` 仍是默认关闭的宿主诊断接口，不向主 LLM 开放。

## 5. 火柴盒验收目录

- `ground_floor`：可探索。
- `stairway`：仅通行，不可探索。
- `upper_floor`：可探索。

验收只在火柴盒世界进行，本规范不授权迁移或修改正式世界。
