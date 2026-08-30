# YUI NPC Agent 行为图 v1.2（冻结）

本文件与 `YUI_NPC_ProtocolConstants_v1.2.json` 共同定义 Python 侧的确定性行为编排。v1.1 工具和 wire 均保持不变。

## 1. 图结构

提交对象必须包含 `entry:string` 与 `nodes:array`。每个节点包含唯一的 `id` 和 `type`；引用只能指向同一图中的节点。任意回边均非法，循环只能由 `repeat` 表达。

- 控制节点：`sequence.children`、`selector.children`、`parallel.children/join`、`repeat.child/count`、`retry.child/max_attempts/delay_ms`、`timeout.child/timeout_ms`、`condition.predicate`。
- 行为节点：`navigate`、`approach`、`follow`、`orbit`、`explore`、`look_at`、`act`、`set_expression`、`say`、`wait`、`stop`。
- `parallel.join` 只能是 `all` 或 `race`。同一并行节点最多一个移动域子树；`movement=block` 动作不能与移动并行。
- `selector` 按顺序执行，首个成功即成功；`retry` 最多三次；`unknown` 永远不是成功。

### 1.1 控制节点字段

| type | 必填字段 | 可选字段/约束 |
|---|---|---|
| `sequence/selector` | `children` | 1..64 个子节点 |
| `parallel` | `children` | 1..4 个子节点；`join=all/race`，默认 `all` |
| `repeat` | `child,count` | `count=1..10` |
| `retry` | `child,max_attempts` | `max_attempts=1..3`；`delay_ms=0..5000` |
| `timeout` | `child,timeout_ms` | `timeout_ms=1..60000` |
| `condition` | `predicate` | 只允许白名单谓词 |

### 1.2 行为节点字段

| type | 必填字段 | 可选字段/约束 |
|---|---|---|
| `navigate` | `target_key` | `speed_mps` |
| `approach` | `player_slot` | `distance_m=0.5..5.0`、`speed_mps`、`face_target` |
| `follow` | `player_slot,duration_ms` | `duration_ms=1..60000`、`speed_mps` |
| `orbit` | `target_key` | `radius_m=0.25..5.0`、`laps=1..3`、`direction=cw/ccw`、`speed_mps`、`face_target` |
| `explore` | `region_key,duration_ms` | `duration_ms=1..600000`、`strategy=unvisited/patrol`、`speed_mps` |
| `look_at` | `player_slot,duration_ms` | `duration_ms=1..60000` |
| `act` | `action_key` | `player_slot`、`loop` |
| `set_expression` | `expression_key` | `duration_ms=0..60000` |
| `say` | `text` | 1..384 UTF-8 字节；`duration_ms`、`estimated_delay_ms`、`action_key` |
| `wait` | `duration_ms` | `duration_ms=1..60000` |
| `stop` | 无 | `scope=all/movement/action` |

### 1.3 条件谓词

`player_present`、`player_distance`、`event_seen`、`node_status`、`control_state`、`estop`、`elapsed_ms` 是唯一合法的 `predicate.type`。谓词只能读取当前 session 的结构化事实，不得包含代码、JSON 路径表达式或运行时自然语言判断。

## 2. 安全上限

每图最多 64 节点、深度 8、并行子节点 4、循环 10 次、单次等待 60 秒、总时长 10 分钟。条件只允许冻结常量列出的白名单谓词，不执行代码、JSONLogic 或自然语言回调。

## 3. 生命周期

计划状态只允许 `accepted/running/succeeded/cancelled/failed/unknown`。计划绑定提交时的 session、catalog revision 和 driver pid；任一发生变化，或出现 watchdog、ESTOP、所有权丢失、驱动者离开、显式断开时，计划立即取消。默认拒绝新的移动计划；`replace_active=true` 会先停止旧计划持有的控制域，再启动新计划。

`npc.plan_cancel` 会停止计划实际占用的控制域。`npc.stop` 会取消与 scope 相交的运行中和排队中计划；`npc.estop` 取消所有计划。LLM 不得获得 `npc.plan_step`。会话内只保留最近 16 个计划，不跨进程恢复。

## 4. 世界事实

静态目标只能使用世界目录明确发布的全局唯一 `semantic_key`；动态玩家只能使用当前 session 的 `player_slot`。`npc.observe` 和 `npc.world_query` 不向模型公开绝对坐标，玩家姓名也默认隐藏。
