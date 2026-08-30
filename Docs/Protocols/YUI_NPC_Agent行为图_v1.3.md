# YUI NPC Agent 行为图 v1.3（冻结）

本文件只定义 v1.3 对 `YUI_NPC_Agent行为图_v1.2.md` 的增量。v1.1、v1.2 文件和既有节点语义保持不变。

## 1. 新增移动节点

行为图新增 `move_relative` 叶节点：

| 字段 | 要求 |
|---|---|
| `bearing_deg` | 必填有限数值，按 360° 归一化；0°前、90°右、180°后、270°左 |
| `distance_m` | 必填，0.25..10.0 米 |
| `speed_mps` | 可选，使用既有速度量化规则 |
| `face_travel` | 可选布尔值，默认 `true` |
| `allow_shorter` | 可选布尔值，默认 `true` |

`move_relative` 占用移动域。它只能编译为一次 `MOVE_RELATIVE`，不得由 Python 展开成若干绝对坐标命令。Unity 必须保持请求方位；`allow_shorter=true` 时，只能按 100%、80%、60%、40%、20% 依次缩短，且最终距离不得小于 0.25 米。

## 2. 探索版本选择

`explore` 节点的字段保持 v1.2 不变。会话同时具备 `world_map`、`semantic_navigation` 和 `local_navigation` 时，它编译为一次 `EXPLORE_REGION`；缺少 `local_navigation` 时保持 v1.2 的 Anchor 遍历实现。探索只能由明确的 `npc.explore` 或行为图 `explore` 节点触发，空闲状态不得自动开始探索。

目标 Region 必须存在且 `explorable=true`。`unvisited` 与 `patrol` 分别编码为 P5=0 和 P5=1。计划仍遵循 v1.2 的单移动域、替换、取消、ESTOP、session 绑定和 `unknown` 传播规则。

## 3. 模型可见世界

`npc.observe.location` 只允许包含 `localized`、`region_key`、`floor_label` 和最近 Anchor 的 `semantic_key/d/brg`。定位事实来自 Unity 发布的 Region 体积命中结果；`localized=false` 时不得用最近 Anchor 推断 Region。主 LLM 不得看到 NPC、Region、Anchor 或 NavMesh 的绝对坐标。
