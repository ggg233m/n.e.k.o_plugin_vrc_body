# 【历史】Bug：在线轨迹"被拉回起点" —— 多 segment 坐标系未对齐（2026-09-23）

> **一句话**：这不是回退，是**上一层修好之后才暴露出来的第二层问题**。
> 上一轮的"搬点公式少一次求逆"已生效（丢跟踪显著减少、`reloc_rejected_weak` 从 1181 冻结降到 **52 且不再增长**、
> `tracking_state` 回到 `tracking`），但 `slam_optimize` 仍在 **中位 3.67 s / p90 6.24 s**，
> 远超断档阈值 `max_timestamp_gap_s = 2.5`，于是建图被反复**断段**（`new_segments = 8`）；
> **每个新 segment 的坐标系都从自己的原点重新建立**，而前端把它们画在同一个"自建坐标系"里，
> 视觉上就是**轨迹跑远后突然跳回起点**。

---

## 一、现象

用户原话：**"丢追还有，不过确实减少了，不过现在丢追是，线路直接被拉回起点。"**

截图标题：`关键帧相机轨迹俯视图（自建坐标 X/Z；不是墙体或障碍物地图）`。
画面是一条细长的真实行进轨迹（两条几乎平行的纵向轨道），外加**大量跨越整个图面的长蓝线**，
下方还有一簇互相重叠的点。截图副本：`.tmp/_traj_pullback_screenshot.png`。

---

## 二、决定性数据（2026-09-23 23:57 在线实例，PID 28796，`127.0.0.1:48912`）

### 2.1 关键帧位姿序列

来源：在线图 `relocalization_map_live.json`（90 个关键帧，9 个 segment）。
相机中心 = `-R^T t`（`kf.R/t` 是 world->cam）。距离是**与上一个关键帧**的水平距离。

| kid | seg | frame | x | y | z | d(prev) |
|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 0.00 | 0.00 | 0.00 | — |
| 1 | 0 | 98 | -0.52 | -0.19 | 0.83 | 0.981 |
| … | 0 | … | … | | … | 稳定前进，0.2–3.5 |
| 36 | 0 | 289 | 3.92 | -3.98 | 20.80 | 1.171 |
| **37** | **0** | **298** | **0.00** | **0.00** | **0.00** | **21.164** ← 跳回原点 |
| 38 | 1 | 302 | -0.00 | -0.23 | 0.97 | — |
| 41 | 2 | 315 | 10.00 | -4.06 | 21.77 | **23.732** ← 段内跳 |
| 42 | 2 | 322 | 0.00 | 0.00 | 0.00 | 23.953 |
| 44 | 3 | 324 | 6.03 | -2.41 | 11.75 | 13.612 |
| 45 | 3 | 330 | 0.68 | -2.17 | 5.79 | 8.002 |
| 46 | 3 | 332 | 0.00 | 0.00 | 0.00 | 5.834 |
| 48 | 4 | 334 | 8.03 | -3.98 | 20.24 | **25.406** |
| 49 | 4 | 335 | 0.00 | 0.00 | 0.00 | 21.775 |
| 58–91 | 8 | 368–617 | X∈[-4.05,-2.60] | | Z∈[-1.37,0.26] | 多 < 0.5 |

### 2.2 每段汇总

```
seg 0  n=38  kid  0..37   X[-1.54,  5.32]  Z[ 0.00, 20.80]   ← 唯一一段真实轨迹
seg 1  n=2   kid 38..39                     Z[ 0.00,  0.97]
seg 2  n=3   kid 40..42   X[ 0.00, 10.00]  Z[-0.19, 21.77]   ← 段内自己跳 23.7
seg 3  n=4   kid 43..46   X[-1.00,  6.03]  Z[ 0.00, 11.75]   ← 段内跳 13.6 / 8.0 / 5.8
seg 4  n=3   kid 47..49   X[ 0.00,  8.03]  Z[-4.39, 20.24]   ← 段内跳 25.4 / 21.8
seg 5  n=2   kid 50..51
seg 6  n=2   kid 52..53
seg 7  n=4   kid 54..57
seg 8  n=34  kid 58..91   X[-4.05, -2.60]  Z[-1.37,  0.26]   ← 稳定，但 1.5 m 内打转
```

⇒ **seg 1–7 是"断段风暴"**：每段只攒下 2–4 个关键帧就再次断开。
⇒ 全表见 `.tmp/_traj_live.out`。

### 2.3 阶段耗时（`/worldmodel/mapping` → `profile`，在线实测，单位 ms）

| 阶段 | median | p90 | count |
|---|---|---|---|
| **`slam_optimize`** | **3671.392** | **6243.246** | 29 |
| `live_mapping_process` | 648.481 | 2660.969 | 683 |
| `slam_process` | 476.5 | 2078.0 | 608 |
| `map_stream_save` | 434.024 | 645.982 | 78 |
| `slam_relocalize` | 263.292 | 763.585 | 113 |
| `slam_track_local` | 127.897 | 346.899 | 393 |
| `slam_init` | 94.305 | 131.871 | 103 |
| `slam_start_segment` | 56.276 | 98.135 | 53 |
| `slam_keyframe` | 48.766 | 72.472 | 72 |
| `slam_associate` | 21.703 | 47.857 | 384 |
| `slam_detect` | 11.071 | 13.98 | 609 |
| `slam_loop` | 0.184 | 83.181 | 384 |

### 2.4 其它状态

- `stats`：`frames 609, tracks_ok 384, tracks_lost 5, keyframes 33, map_points 5370,
  loop_candidates 280, loop_closures 30, relocalizations 25, init_frames 103,
  odom_edges_downweighted 8, jumps_rejected 4, reloc_pending 36,
  reloc_rejected_weak 52, new_segments 8, points_moved_by_optimization 232697`
- `last_reset_reason = timestamp_gap_reset`，`max_timestamp_gap_s = 2.5`
- `mapping.segment_id = 8`，`keyframe_count = 90`，`map_point_count = 12671`
- `frame_rate = 2.207 Hz`（`measured`）
- `nav_map`：**`segment_id = 0`（与 `mapping.segment_id = 8` 不一致）**、`node_count 90`、
  `edge_count 89`、`passable 0` / `unknown 89`、`state = preview_ready`
- `metric_calibration`：`scale_validated = false`、`movement_basis_validated = false`、
  `scale_reason = "velocity_unavailable"`、`scale_samples = 0`
- 全量快照：`.tmp/_mapping_now.json`

---

## 三、根因链（当前最佳推断）

```
slam_optimize 中位 3.67 s / p90 6.24 s            ← 待查，见第四节 P1
   │  消费线程被卡住 > max_timestamp_gap_s = 2.5
   ▼
timestamp_gap_reset → mark_tracking_gap → LOST
   │
   ▼
断段：new_segments = 8（seg 1–7 每段只攒 2–4 个关键帧）
   │  每个新 segment 的坐标系从**自己的原点**重新建立（gauge 重置）
   ▼
各段之间没有任何"段间相对位姿"落盘；前端把 9 个 segment 的关键帧
画在**同一个"自建坐标系"**里
   │
   ▼
① 轨迹跑远后突然跳回原点（kid 36 → 37，距离 21.164）
② 段内也跳（seg 2 / 3 / 4 段内最大 25.406）
③ 段 1–7 只攒 2–4 个 kf（断段风暴）
④ 上游截图里"两条平行轨道 + 大量长蓝线" = 真实轨迹 + 跨段连线
```

**与上一轮修复的关系（重要）**：
上一轮修的是"搬点公式少一次求逆"，它让地图几何不再被撕碎。修复生效的直接证据：

| 指标 | 修复前 | 现在 |
|---|---|---|
| `reloc_rejected_weak` | 945 → 1181（每帧一次、持续暴涨、冻结） | **52，不再增长** |
| `tracks_lost` | 11（冻结） | 5 |
| `tracking_state` | 一直 `lost` | **`tracking`** |
| `slam_relocalize` median | 638 ms | 263 ms |
| SLAM 是否冻结 | frame 459 后位姿逐位相同 | 正常更新（`last_result.frame = 608`） |

⇒ **本 bug 是被它掩盖的第二层问题，不是回退。**

---

## 四、待排查假设（按优先级）

### P1 · `slam_optimize` 为什么还要 3.67 s？（最高优先）

上一轮把稠密 Hessian 换成 scipy 稀疏直接解，离线在 dof=1434 上测得 **1769 ms → 18 ms**，
但**在线实测中位仍是 3671 ms**。必须先分清"解图"和"搬点"各占多少：

- 现在的 `slam_optimize` 埋点**同时包住** `optimize_map()`（解位姿图）与
  `_optimize_and_correct()` 的搬点循环。搬点要对 **12671 个点**做 Python 级矩阵乘，
  再把结果回写到 **90 个关键帧**的 `pts3d`（`points_moved_by_optimization` 已累计 **232697**）。
  ⇒ **建议第一步：把埋点拆成 `slam_optimize_graph` / `slam_optimize_repair`**，再决定优化谁。
- 确认 scipy **真的在用**（`_have_scipy` / `_DENSE_SOLVER`），不要静默回退到稠密解。
- 注意 `loop_closures = 30` ⇒ 位姿图**真的在改位姿**了（不再是上一轮那种"残差恒 0"的空跑），
  所以 `_optimize_and_correct` 的搬点路径现在是**真实执行**的。

### P2 · 断档判据与优化耗时的耦合

`max_timestamp_gap_s = 2.5` 用的是"**被处理帧之间**的采集时间差"。只要单次 `slam_process`
超过 2.5 s（p90 已 2078 ms，`live_mapping_process` p90 2661 ms），就必然被误判为断档。
候选方案（择一或组合）：
1. 把 optimize 移到消费线程之外（异步/后台线程），消费线程只做跟踪；
2. 断档判据改成"图片采集时间戳之差"而不是"处理间隔"；
3. 优化分帧增量执行（每次只跑少量迭代，摊平尖峰）。

### P3 · 多 segment 的坐标系对齐

现在每个 segment 独立 gauge，段间没有任何相对位姿。需要决定并落地：
- （a）断段后**必须重定位回原坐标系**才算续上。`relocalizations = 25` 说明重定位**有成功**，
  但显然没有把新段的位姿映射回旧段坐标系；
- （b）或落盘时给出每个 segment 的 `frame_id` / `T_segment_to_world`，让前端**分段绘制**、
  段间**不连边**（至少不要画出误导性的长边）。
当前两者都没做：段间不共享坐标系，前端又按 `kid` 顺序连线。

### P4 · `mark_tracking_gap` 之后位姿是否归零

`kid 37`（仍标着 `segment_id = 0`）的位姿恰好是 `(0.00, 0.00, 0.00)`，而它前一个关键帧在 20.8 m 外。
需要确认 `mark_tracking_gap` / reset 路径是否把 `cur_R/cur_t` 重设为单位阵，
且这一帧**仍被当作关键帧写入了旧 segment**。若是，至少要保证"重置后的帧归入新 segment"，
不要污染旧段。

### P5 · `nav_map.segment_id = 0` 与 `mapping.segment_id = 8` 不一致

导航图可能还挂在旧段上（`passable_edge_count = 0`，全 unknown，因为
`scale_validated = false`、`scale_reason = velocity_unavailable`）。
这会让"路线被拉回起点"在导航层被放大（规划路径挂在错的 segment 上）。

### P6 · 上一轮文档 3.7 的"次级隐患"现在会真实触发

上一轮记录：同一个物理点只被"最新观测它的那个关键帧"的 delta 搬走 ⇒ 有回环时仍非刚性。
现在 `loop_closures = 30`、位姿真的在变 ⇒ 这条**从"隐患"升级为"可能正在发生"**。
严格解 = 让点云跟随**单一全局 SE(3)**，或做真正的 BA。

#### P6 补证（2026-09-24 00:2x，实测 —— 已从"可能"变为"已成立"）

**（a）合成探针已复现**（`.tmp/_p6_nonrigid_probe.py` → `.tmp/_p6_nonrigid_probe.out`）
两个关键帧共同观测同一批点；只对"最新观测者"kf1 施加修正、kf0 保持不动（模拟节点 0 软锚定的真实情形）：

| 场景 | kf0 自投影（位姿未动） | kf1 自投影（最新观测者） | 点最大位移 |
|---|---|---|---|
| 无修正（控制） | 0.00 px | 0.00 px | 0.000 m |
| 6° / 7 cm | **30.20 px** | 0.00 px | 0.781 m |
| 6° / 39 cm | **39.18 px** | 0.00 px | 1.006 m |

⇒ 位姿**没动**的那个关键帧，几何被**另一个**关键帧的 delta 拖走；误差随修正量单调增长。
控制组 0/0 证明探针本身可信（不是夹具假象）。

**（b）在线图正在发生**（`runs/20260920-233456/relocalization_map_live.json`，
154 kf / 16395 pts，mtime 09-24 00:18；脚本 `.tmp/_selfproj.py` → `.tmp/_selfproj_live.out`）

```
selfproj_med: p10=0.40  med=4.69  p90=55.42  max=229.39   frac_kf<2px=0.171
```

- 同一口径对比修复前：med **258 px** / `frac<8px` 0.102 ⇒ 少一次求逆的修复**确实把主症状压下去了**（258 → 4.69）。
- 但验收不变量是 **med < 2 px**（本文档 §5.5），现在只有 **17.1%** 的关键帧达标，p90 55 px、max 229 px。
- 分布与机制高度吻合：**kid 0–6（t = 1.3–207.9 s，首次优化之前产生）全部 ≤ 1.89 px、`frac<2px` = 1.00**；
  kid 7 起立刻掉到 30–177 px；kid 37–54（断段风暴产生的新 segment，几乎不被优化）又回到 ~0.3 px、`frac<2px` = 1.00；
  长尾 kid 60–153 稳定在 3–8 px 的新平台。
  ⇒ **"首次优化之前的 kf 干净、之后全部带上误差"**，正是"点只被最新观测者的 delta 搬"的指纹。

**（c）验收门盲区（必须补，否则 P6 永远不会被拦）**
`tests/test_slam_point_repair.py` 只建 **1 个关键帧**（所有点的 `observations` 恒为 `[(0, i)]`）
⇒ "最新观测者"规则在该夹具下**恒等价于正确**，P6 **结构性不可能被它抓到**。它全绿 ≠ P6 不存在。
建议补一条"两个 kf 共同观测同一点、且两者位姿修正量不同"的断言：`_optimize_and_correct` 后**每个** kf
的自投影都必须 < 2 px（当前代码会红）。

---

## 五、验收口径

1. **`slam_optimize` median < 100 ms**（至少 < 断档阈值的一半），p90 < 500 ms。
2. **`new_segments` 在一次正常录制里不增长**（除用户主动重开）；`last_reset_reason`
   不再是 `timestamp_gap_reset`。
3. **每个 segment 的轨迹在俯视图里连续、不自交、不跳回原点**；`d(prev)` 不出现 > 2 m
   的孤立尖峰（真实步速下关键帧间距应是几十 cm）。
4. **段间断开处**：要么有"已重定位回原坐标系"的可验证证据（`T_segment_to_world`），
   要么前端分段绘制、段间不连边。
5. **地图几何不变量不得退化**：每个关键帧的有限 `pts3d` 用它自己的 (R,t) 投回自己的 kps，
   中位 **< 2 px**（`tests/test_slam_point_repair.py` 守着这条）。

---

## 六、复现与取证

**在线（后端在跑时）**——鉴权**必须用 header**，查询串 `?token=` 只对 `/vision/mjpeg` 放行：

```bash
curl -H "X-Neko-Backend-Token: dev" http://127.0.0.1:48912/worldmodel/mapping
curl -H "X-Neko-Backend-Token: dev" http://127.0.0.1:48912/snapshot
```

关注字段：`profile.slam_optimize.{median_ms,p90_ms,count}`、`stats.new_segments`、
`last_reset_reason`、`segment_id`、`nav_map.segment_id`、`metric_calibration.scale_validated`。

**轨迹复现**（本次所用脚本，输出 `.tmp/_traj_live.out`）：

```python
import json, numpy as np
p = ".slam_probe/offline_probe/recorder/runs/20260920-233456/relocalization_map_live.json"
m = json.load(open(p, encoding="utf-8"))
for k in sorted(m["keyframes"], key=lambda k: k["kid"]):
    R = np.array(k["R"], float); t = np.array(k["t"], float).reshape(3)
    c = -R.T @ t                       # R,t 是 world->cam，相机中心 = -R^T t
    print(k["kid"], k["segment_id"], k["frame_idx"], round(c[0], 2), round(c[2], 2))
```

**证据文件**

- `.tmp/_mapping_now.json` —— 本次 `/worldmodel/mapping` 全量快照
- `.tmp/_traj_live.out` —— 90 个关键帧的 segment / 坐标 / 相邻距离全表
- `.tmp/_traj_pullback_screenshot.png` —— 用户截图副本
- `Docs/丢跟踪根因-图优化撕碎地图几何（2026-09-23）.md` —— **上一轮已修的那个 bug，先读它**
- `tests/test_slam_point_repair.py` —— 上一轮的回归门（必须保持全绿）

---

## 七、一句话交接

**先做 P1**（把 `slam_optimize` 埋点拆成"解图 / 搬点"两段，看 3.67 s 到底花在哪），
**再做 P2**（断开"优化耗时 → 2.5 s 断档 → 断段"的耦合）。
P1/P2 解决后 `new_segments` 应停止增长，"线路被拉回起点"会随之消失。
P3/P4/P5 是同一现象的另外几种暴露方式，可以并行确认，但不要抢在 P1/P2 之前动手。
