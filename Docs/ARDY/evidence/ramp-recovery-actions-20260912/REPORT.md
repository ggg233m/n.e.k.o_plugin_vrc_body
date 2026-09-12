# 主场景斜面、恢复与动作实测

本报告区分模型输出、世界执行回执与动作质量。环境为 Unity 主场景 SAVE / ClientSim、真实 ARDY、真实 MIDI；没有启动 VRChat，也没有活动 NEKO 宿主或多人验收。保存前的未提交场景和源码备份在 `H:/AI/NVidia/integrations/migration-backups/ramp-action-20260912-2020`。

## 已实现的修复

- 两段楼梯及影院三层台面使用显式标注的无渲染斜面，原楼梯渲染网格保留。影院斜面覆盖台面宽度，避免横向跨步踩出狭窄代理碰撞区；旧 Cube.005 阶梯 MeshCollider 关闭。
- 原始模型骨骼/骨盆插值不再混入上次 IK 修正。按真实腿长计算必要的支撑下降补偿；下降立即满足支撑，释放补偿平滑。补偿受腿长30%及25cm双重上限控制，当前约20.4cm；足端25cm修正上限、可达性和碰撞检查仍保留。此项不是鞋底滑动验收。
- `pose_surface_recovery_v1` 仅与持续协议能力同时发布。世界记录最多4个、30秒有效的失败点；宿主只凭对应世界回执恢复已接受的路径目标，新会话重新寻路，不重放旧帧。同一显式目标最多三次物理故障恢复；足端修正/可达性和骨盆支撑修正失败也可在安全姿态重新生成，但不记成环境障碍，耗尽时清理待恢复目标；新意图及急停清除旧目标。
- 绕行在世界侧寻找先退离、再侧绕的路径，检查NavMesh、活动边界、局部支撑和身体净空；找不到时返回明确 `safe_detour_unavailable`。候选有界，只在失败后请求路径时运行，不增加主模型工具或逐帧MIDI字段。
- 行走保留提前刹停距离，已知障碍旁允许安全退出。额外预警范围内的地面命中不当作墙；低矮地面圆角命中不作为提前预警依据，实际步长内的碰撞仍阻止，不增加逐帧查询。故障点只在重新寻路时复核，支撑及净空恢复后清除过时标记。
- 修复新路径已接受但尚未播放就断流时丢失目标的问题；完成/被替换/急停及陌生回执不会复活任务。三维运动根、飞行碰撞、空中安全落地尚未实现，不能用跳跃绕过上述检查。

## 动作组结果

213317：待机、挥手、鞠躬得到世界完成回执，已检查对应观察画面。4秒任务实际区间约4.009、3.937、4.009秒。待机稳定段头部相邻采样最大约0.42度；截图观察影响帧率，该段约19Hz有效采样，不冒充正式30Hz或性能验收。

下蹲和跳跃触发 `trajectory_height`。三组后空翻请求中，两组在执行前分别遇到租约丢失和缓冲欠载，一组模型未来输出出现倒立，但执行器因高度范围停止、实际倒立采样为0。因此没有一组完整世界后空翻通过，不能发布跳跃/空翻能力，不能作为脱困兜底。生成出的未来倒立帧也不代表已安全执行。原始模型、骨骼采样、时序审计和逐帧画面均保留。

短暂无支撑测试以一帧屏蔽支撑层模拟检测失效，随后恢复原LayerMask；这不等于真实地面空洞、动态平台或跳跃跨越测试。永久无路应明确失败并允许新意图，不保证任意缺口都能跨过。

## 全部运行记录

编号为当天时间，原始数据位于同名 `continuous-world-20260912-编号` 子目录。旧候选失败与注入错误均保留，不用最后成功轮次覆盖。

| 运行 | 探针状态 | 原始结果摘要 |
|---|---|---|
| 202313 | partial | 19:world_completion_timeout；12:npc.operation_completed |
| 202607 | partial | 19:npc.operation_cancelled，恢复1次；12:execution_unavailable |
| 203244 | partial | 19:npc.operation_cancelled，恢复2次；20:execution_unavailable；21:not_attempted_after_execution_failure；12:not_attempted_after_execution_failure |
| 203719 | partial | 19:npc.operation_cancelled；20:execution_unavailable；21:not_attempted_after_execution_failure；12:not_attempted_after_execution_failure |
| 204313 | partial | 19:npc.operation_completed，恢复1次；20:npc.operation_cancelled；21:execution_unavailable；12:not_attempted_after_execution_failure |
| 205431 | partial | 19:npc.operation_completed，恢复1次；20:npc.operation_completed；21:world_completion_timeout；12:npc.operation_cancelled |
| 210219 | succeeded | 19:npc.operation_completed，恢复1次；12:npc.operation_completed |
| 210610 | succeeded | 无效测试：故障观察器被安全检查拒绝，未注入障碍；13:npc.operation_completed；12:npc.operation_completed |
| 210726 | partial | 13:safe_detour_unavailable，恢复2次；12:端口已停止，恢复1次 |
| 211421 | failed | 13:safe_detour_unavailable；12:npc.operation_completed；fault_recovery_not_observed |
| 211817 | failed | 13:safe_detour_unavailable；12:npc.operation_completed；fault_recovery_not_observed |
| 213037 | failed | 0:无终态；surface_test_not_started |
| 213157 | partial | 0:npc.operation_completed，恢复1次；12:safe_detour_unavailable |
| 213317 | partial | idle:npc.operation_completed；wave:npc.operation_completed；bow:npc.operation_completed；crouch:trajectory_height；jump:trajectory_height；backflip_seed12:pose_lease_lost；backflip_seed13:trajectory_height；backflip_seed14:stream_underrun |
| 213719 | partial | 0:npc.operation_completed，恢复1次；12:safe_detour_unavailable，恢复1次 |
| 214203 | partial | 0:npc.operation_completed，恢复1次；12:safe_detour_unavailable，恢复1次 |
| 214721 | succeeded | 0:npc.operation_completed，恢复1次；12:npc.operation_completed |
| 214919 | failed | 无效测试：测试注入LayerMask类型错误，导致Udon异常；0:world_completion_timeout；12:execution_unavailable；19:not_attempted_after_execution_failure；12:not_attempted_after_execution_failure；fault_recovery_not_observed |
| 215307 | failed | 0:safe_detour_unavailable；12:safe_detour_unavailable；19:npc.operation_completed，恢复1次；12:safe_detour_unavailable；fault_recovery_not_observed |
| 220004 | partial | 0:npc.operation_completed，恢复1次；12:npc.operation_completed；19:npc.operation_completed；12:npc.operation_cancelled |
| 220346 | succeeded | 12:npc.operation_completed；19:npc.operation_completed；12:npc.operation_completed |
| 220805 | succeeded | 0:npc.operation_completed，恢复1次；12:npc.operation_completed |
| 220920 | failed | 0:safe_detour_unavailable；12:npc.operation_completed；fault_recovery_not_observed |
| 221643 | succeeded | 0:npc.operation_completed，恢复1次；12:npc.operation_completed |
| 221804 | succeeded | 卡顿恢复：1.7039999999979045秒；npc.operation_completed |
| 221922 | succeeded | 已请求切换=100 |
| 222312 | failed | world_completion_missing |
| 222426 | succeeded | 卡顿恢复：None秒； |

## 验收边界

220346的行走审计还记录到个别相邻采样骨盆位移约17.9cm、移动段头部变化最高约18.9度；需要继续做同视角视觉与接触质量核验，不能用路线走完代替自然度通过。`action-audit.json`保留这些数据。

当前不宣称动作系统完美、全地图所有路径稳定、鞋底滑动达标、30分钟稳定性通过或完整后空翻通过。上/下楼完成仅证明已测路线通过，不等于所有影院区域或楼层组合通过。碰撞代理为共享世界碰撞；玩家使用新斜面的体验尚未另行验收。

Python完整回归440 passed、1 skipped、301 subtests passed；追加恢复预算测试后，20项针对性回归通过。最终Udon编译、资源同步和场景视觉引用核验另见交付清单。旧余额采样210219及之前部分轮次采用不同时间基准，不能使用其按operation切片的旋转统计；213157以后已用Time.timeSinceLevelLoad对齐世界回执。
