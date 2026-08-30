# YUI NPC Unity 接口规范 v1.2（冻结扩展）

v1.2 完整继承并保持 `YUI_NPC_Unity接口规范_v1.1.md`。本文件只规定新增事实；冲突时 v1.2 新增项仅对 `spec:"1.2"` 世界生效。

## 1. 新增 capability

- bit 17 `world_map`：发布 `region/entity/route_edge` 三类目录。
- bit 18 `semantic_navigation`：接受 `GOTO_ANCHOR/ORBIT_ENTITY`。

缺少 capability 时必须按 v1.1 §7.4 返回 `unsupported_capability`，不得回退到坐标猜测、直线移动或静默 no-op。

## 2. 新增命令

| id | 命令 | 参数 | 语义 |
|---:|---|---|---|
| `0x17` | `GOTO_ANCHOR` | P3 anchor id；P4 speed q7；其余 0 | 使用目录 Anchor 的完整 XYZ、yaw、arrival radius 计算完整 NavMesh 路径 |
| `0x18` | `ORBIT_ENTITY` | P0 radius mm 250..5000；P3 entity id；P4 speed q7；P5 bit0 ccw、bits1..2 laps-1、bit3 face target | Unity 在单个 movement operation 中连续完成 1..3 圈 |

两条命令与 `GOTO_XZ` 使用相同状态权限和 movement lane。不存在的目录 id 返回 `target_missing`。`ORBIT_ENTITY` 必须在 Unity 内连续 retarget，中间点关闭自动制动；只允许最终点、取消、安全状态或失败制动。

## 3. 目录

- `region`：`id,semantic_key,description_zh,tags,floor_label,entry_anchor_id`。
- `entity`：`id,semantic_key,description_zh,tags,region_key,center,approach_anchor_id,orbitable,orbit_min_radius,orbit_max_radius`。
- `route_edge`：`id,from_anchor_id,to_anchor_id,bidirectional,traversal,region_key`，其中 traversal 仅为 `walk/stairs/ramp`。
- v1.2 Anchor 追加 `region_key`。

Anchor、Region、Entity 的 semantic_key 必须全局唯一。内部目录坐标供确定性后端计算相对距离；面向 LLM 的投影必须删除绝对坐标。

## 4. 操作与日志

`GOTO_ANCHOR` 的 operation kind 为 `goto`，到达时 `npc.arrived` 追加 `semantic_key`。`ORBIT_ENTITY` 的 operation kind 为 `orbit`，自然完成结果为 `natural_end`。所有日志继续遵守 20 行/滑动秒和单 JSON 950 UTF-8 字节限制。
