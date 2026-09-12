# 全地图闲逛测试

实际走通 17/22 个语义目标；共 51 条目标记录，其中 30 次实际目标请求。

Unity主场景ClientSim，固定巡游路线，真实ARDY与MIDI。仅以世界 operation_completed 计为走通；失败记录全部保留。

| ID | 目标 | 区域 | 导航 | 实际结果 |
|---|---|---|---|---|
| 0 | hall_center | hall | PathComplete | 已走通；端口已停止、unsupported_ground、npc.operation_completed |
| 1 | bed_side | se_nook | PathComplete | 已走通；端口已停止、execution_unavailable、npc.stream_path_unconfirmed、npc.operation_completed |
| 2 | rocks_view | hall | PathComplete | 已走通；npc.operation_completed |
| 3 | north_door | north_walk | PathComplete | 已走通；npc.operation_completed |
| 4 | north_photo | north_walk | PathComplete | 已走通；npc.operation_completed |
| 5 | north_corner | north_walk | PathComplete | 未走通；pelvis_correction_limit |
| 6 | west_door | west_corridor | PathComplete | 已走通；execution_unavailable、npc.operation_completed |
| 7 | corridor_south | west_corridor | PathComplete | 已走通；execution_unavailable、npc.operation_completed |
| 8 | tv_sofa | tv_lounge | PathComplete | 已走通；execution_unavailable、npc.operation_completed |
| 9 | tv_rug_center | tv_lounge | PathComplete | 已走通；execution_unavailable、npc.operation_completed |
| 10 | tv_photo_wall | tv_lounge | PathComplete | 已走通；execution_unavailable、npc.operation_completed |
| 11 | gallery_west | south_gallery | PathComplete | 已走通；execution_unavailable、npc.operation_completed |
| 12 | gallery_center | south_gallery | PathComplete | 已走通；execution_unavailable、not_attempted_after_execution_failure、unsupported_ground、npc.operation_completed |
| 13 | gallery_bench | south_gallery | PathComplete | 已走通；npc.operation_completed |
| 14 | deck_lamp | east_deck | PathComplete | 已走通；npc.operation_completed |
| 15 | deck_photo_wall | east_deck | PathComplete | 已走通；npc.operation_completed |
| 16 | deck_east_end | east_deck | PathComplete | 已走通；npc.operation_completed |
| 17 | nook_rug | se_nook | PathComplete | 未走通；npc.stream_path_unconfirmed、not_attempted_after_execution_failure、execution_unavailable、端口已停止 |
| 18 | nook_photo_wall | se_nook | PathComplete | 已走通；端口已停止、not_attempted_after_execution_failure、npc.stream_path_unconfirmed、npc.operation_completed |
| 19 | cinema_aisle | cinema | PathComplete | 未走通；execution_unavailable、not_attempted_after_execution_failure、npc.stream_path_unconfirmed |
| 20 | cinema_front | cinema | PathComplete | 未走通；execution_unavailable、not_attempted_after_execution_failure、npc.stream_path_unconfirmed |
| 21 | media_lounge | cinema | PathComplete | 未走通；execution_unavailable、not_attempted_after_execution_failure、npc.stream_path_unconfirmed |

轨迹图：绿色为至少一次世界确认到达的目标，红色为未走通目标；线条为实际采样的楼上位置，不是规划路径。

![实际轨迹](map.svg)

已知限制：小房间地毯有路径点地面射线未命中；楼下路线超过平地高度限制。路径拒绝没有明确回执，会触发超时并影响后续请求。

北走廊拐角实测触发pelvis_correction_limit：需要12.15799cm骨盆下降，超过12cm上限。此为脚部接触/重定向约束失败，不是端口带宽故障。后续不可用目标没有获得实际行走验证。

## 行走摇摆诊断

175501巡游中采得923个有效移动样本：ARDY输入躯干倾角P95约18.68°、最大32.34°；Unity最终躯干P95约18.86°、最大32.23°。最终躯干横向倾斜范围约−29.65°至+14.19°，同期Animator身体写入采样为0。

因此这段明显摇摆已经出现在收到的生成姿态中，不能归因于Unity插值单独放大或Animator同时写入。当前巡游提示词包含looking around calmly；没有完成相同路径、种子和提示词的对照，尚不能断言是哪项生成条件导致。输入姿态指标来自MIDI解码后的姿态，不等同原始浮点模型输出。

本轮只增加测试与采样，未修改正式步态算法，也没有宣称摇摆已修复。脚部约束失败与地面支撑拒绝还需分别修复。

不能把本次覆盖测试当成全系统质量通过。详细回执见各运行目录。
