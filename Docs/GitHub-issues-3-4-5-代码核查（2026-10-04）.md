# GitHub issues #3/#4/#5 代码核查（2026-10-04）

来源：`https://github.com/ggg233m/n.e.k.o_plugin_vrc_body/issues`（#3/#4/#5，2026-10-03 由 LyaQanYi 提交，均 open；#1 已于 08-16 关闭，不在本次范围）。
方法：**不看结论看代码**。每条都回到当前工作树定位实际行号并复核；能算的用数值脚本算。报告里给的行号与 issue 有偏移（仓库在推进），下文以**当前**行号为准。

## 结论速览

| # | 位置（当前行号） | 判定 | 严重度 | 备注 |
|---|---|---|---|---|
| 3-1 | `backend/nav_mapping.py:900` | ✅ 成立 | **中-高** | ⛔ 已被用户自行实现并提交（见下） |
| 3-2 | `backend/nav_mapping.py:660-678` | ❌ **误报** | — | 数值验证往返一致，见下 |
| 3-3 | `backend/nav_online.py:158` | ✅ 成立 | 低-中 | `kf_defer_max_s=1.0` 超时路径 |
| 3-4 | `backend/nav_grid.py:252` | ✅ 成立 | 低 | L1/Bresenham 一行修 |
| 3-5 | `backend/nav_mapping.py:456` | ⚠️ **部分成立** | 低 | ⚠️ **报告把后果说反了**，见下 |
| 4-1 | `backend/traversability.py:324` | ✅ 成立 | 中 | 转向后首帧伪信号 |
| 4-2 | `backend/traversability.py:428` | ✅ 成立 | 低 | **不会导致撞墙**，见下 |
| 4-3 | `backend/vision.py:2126` | ✅ 成立 | 中 | 与代码注释意图相反 |
| 5-1 | `driver_log.py:685` | ✅ 成立 | 低 | `or 0` 假零 |
| 5-2 | `driver_log.py:398` | ✅ 成立 | 中 | 代码自己的注释就是这么给 action sink 做的 |
| 5-3 | `driver_log.py:982` | ✅ 成立 | 低 | 仅外部/旧 JSONL 触发 |
| 5-4 | `behavior.py:89` | ✅ 成立 | 中 | `service.py:3081` 只 `str()` 不校验 |
| 5-5 | `osc.py:319-330` | ✅ 成立 | 低 | FD 泄漏 |
| 5-6 | `osc.py:1027` | ✅ 成立 | 低 | 假零短路 |
| 5-7 | `motion.py:403/410/421/511` | ⚠️ **条件成立** | 低 | 默认 base 帧 HMD=identity，前后乘等价 |

**14/15 成立，1 条误报（3-2）。** 报告质量总体很高——没有一条是"看着像 bug 其实不是"的那种水货，唯一的误报出在指针/布局这类最容易脑补的地方。

---

## Issue #3 导航建图

### 3-1 ✅ `occ_ground_ratio` 早于近带豁免 —— 真，且违反明文设计

`nav_mapping.py:900`：

```python
occ = ((n_o > c.min_pts - 0.5) & (n_o >= c.occ_ground_ratio * n_g)).astype(np.uint8)
occ = ((cv2.filter2D(occ, -1, np.ones((3,3), np.float32), borderType=cv2.BORDER_CONSTANT) >= 3)
       .astype(np.uint8))
occ, restored, rejected = self._by_quality(occ, ...)
```

而 `_by_quality` 的 docstring（`:929-930`）写得很死：

> 近带 <q_near_m：误差还在带边沿内，单帧即定案；**故意不带**地面压制条件，否则 occ_ground_ratio 会把真细障碍（桌腿、栏杆）一起误杀

实现里 `near = o_n > hi`（`:947`）确实不带 ratio——但它的输入 `base_occ` **在进函数前已经被 ratio 清零了**（`:900`），`near` 救不回来。等于文档承诺的近距豁免从来没生效过。

修的时候要注意：`:900` 和 `_by_quality` 之间还夹着 **3×3 多数滤波**。所以不能简单地"把 ratio 挪进 `_by_quality`"，否则近带票会绕过 3×3 支撑要求，单格噪声直接复活。建议：

- 方案 A（稳）：`:900` 处额外算一张 `occ_near = (n_o > hi) & (o_n > hi)`，**它自己也过一遍 3×3**，然后 `_by_quality` 里 `occ = ((base_occ>0) & (near|mid|solo)) | (occ_near_3x3)`。
- 方案 B（激进，改动大）：`base_occ` 只按 `n_o > min_pts-0.5` 生成，ratio 下沉到 `_by_quality` 里按带施加；同时 3×3 也要按带拆。

⚠️ 这条会**改变建图输出**（近距细障碍变多），按仓库约定属于"改判据"，落地必须重跑 `tools/regression_place_identity.py` 并补一条近距桌腿/栏杆的用例。

### 3-2 ❌ `_encode_ray` 编解码索引公式不一致 —— **误报**

指控：初始编码 z 步长是 `H*W`，解码用 `divmod(f, oH*oW)`，扩容后 `oH != H` ⇒ 重编码错位。

实际不成立。关键点：**初始编码结束时 `c[5] = lay`（`:668`）**，也就是说"编码当时用的布局"被记下来了；后续解码用的 `oH*oW` 正是**当时**的 `H*W`，而不是新的。解码出的 `(iz, iy, ix)` 是相对旧原点的，重编码那一行 `:677` 又用 `+ oy0 - y0` 把原点差补回去，所以 z 步长的一致性是成立的。

数值验证（项目 `.venv`，两次连续扩容）：

```
重编码 == 新布局直接编码: True
往返一致: True
二次扩容一致: True
```

（脚本：随机 400 个体素，`lay1=(4,4,32,32) → lay2=(2,2,48,48) → lay3=(-3,-5,80,90)`，比较"重编码结果"与"用新布局从原始绝对下标直接编码"的结果。）

指控者把"扩容后 oH != H"当成了错误条件，但那恰恰是正确条件——解码必须用旧布局、编码必须用新布局。**这条不改。**

### 3-3 ✅ defer 超时路径产出 `"refresh"` —— 真

`nav_online.py:158`：

```python
if abs(yaw_rate_dps) > c.kf_defer_dps:
    if self.defer_since is None: self.defer_since = t
    if t - self.defer_since < c.kf_defer_max_s and d < c.kf_defer_max_m:
        self.deferred += 1
        return None
kind = "refresh" if (not moved and d < c.kf_still_m and dyaw < c.kf_still_deg) else "new"
```

`moved` 是 defer 之前算的。defer 因 `kf_defer_max_s=1.0` 超时退出时，若净位移/净转角都没攒够（`d < kf_still_m` 且 `dyaw < kf_still_deg`），就用**仍在快速旋转期间拍的帧**去 refresh 上一关键帧。`kf_defer_dps` 默认下只有"原地快速摆动 ≥1 s"才踩到，不算高频，但确实与设计意图（defer 就是为了躲开转动帧）相反。修法：`_take` 前判断 `defer_since is not None` ⇒ 强制 `"new"`。

### 3-4 ✅ `_line_ok` 斜线采样不足 —— 真（低），但真凶是 `round()` 不是步数

`nav_grid.py:252`：`n = max(|dr|,|dc|)+1`，`np.linspace(...).round()`。近轴向的斜线会漏格。

⚠️ **实施时（2026-10-04）发现只加 L1 步数不够**。数值验证（随机 4000 组 `dr/dc∈[0,60]`）：

| 方案 | 漏格 |
|---|---|
| 现状 `max+round` | 2519/3000 |
| 只加 L1 步数 + `round` | **2537/3000**（反而略升） |
| L1 + `floor` | **0** |

`round()` 取的是**最近**格而不是线段**所在**格；`to_cell()` 本身就是 floor 口径，只有 floor 与之自洽。典型反例 `(5,5)→(9,6)` 真穿过 `(8,5)`，`L1+round` 仍漏。已按 `L1 + floor`（+1e-9 防端点浮点回退）落地。

采样点数上限从 `L+1` 变 `2L+1`；`plan()` 受 `map_min_interval_s=0.3` 限流约 3 Hz，`L≈100~300` ⇒ 无性能问题。

### 3-5 ⚠️ `_update_cam_h` 限流被 `drop_points` 击穿 —— 真但**后果被说反了**，且线上不可达

`nav_mapping.py:470`（核查时为 456）：`if len(self._pts) - self._cam_h_at < max(1, len(self._pts)//8): return`。

**报告说**：`drop_points` 让 `len(_pts)` 缩到 `_cam_h_at` 以下 ⇒ 限流检查变负数 ⇒ cam_h 立即触发 ⇒ "比 O(log N) 更频繁的全量重算"。

**实测（2026-10-04 实施时复核）三点更正：**

1. **方向是反的。** 差值为负时 `-7 < max(1, N//8)` 恒真 ⇒ 走 `return`，cam_h **不再更新（冻结）**，不是"更频繁重算"。冻结会一直持续到 `len(_pts)` 重新爬回 `_cam_h_at + len//8`，比设计值多等 `(_cam_h_at - len)` 帧。
2. **线上不可达。** 在线路径（`nav_online.py:920-927`）总是先 `add_keyframe(k)` 再 `drop_points(k-1)`，`len(_pts)` 单调不减 ⇒ `_cam_h_at <= len(_pts)` 恒成立。
3. **只有离线回放能触发。** `tools/q_tier_ab.py:87-92` / `q_tier_bench.py:84-86` 是"先摘 k-1、可能再摘 k、最后加 k"，`len` 反而会下降。

已落地（commit `a7beb8b`）：`drop_points` 内 `self._cam_h_at = min(self._cam_h_at, len(self._pts))`，`:470` 再加一层 `max(0, ...)` 防御并注明方向。对生产是 no-op，对离线回放把等待帧数恢复成设计值。

> 教训：别照抄外部报告的因果链。这条如果按"更频繁重算"去理解，会得出完全相反的修复方向。

---

## 实施状态（2026-10-04 当晚）

| 项 | 状态 | commit |
|---|---|---|
| 3-1 | ✅ 已实现，⛔ 默认关 | 两版都实测净亏：票数门 `q_near_pts`（`fd8a8bd`，默认 0）+ 几何门 `ray_near_exempt`（默认 False，见下 §3-1 续） |
| 3-2 | ⛔ 出局 | 误报，不改 |
| 3-4 | ✅ 已修 | `da27e03` |
| 3-5 | ✅ 已修（含方向更正） | `a7beb8b` |
| 5-5 / 5-6 | ✅ 已修 | `ba7700c` |
| 5-1 / 5-2 / 5-3 | ✅ 已修 | `e419220` |
| 5-4 | ✅ 已修（抛 ValueError） | `27f8460` |
| 5-7 | ✅ 已修（全链路成对改） | `37a5b3b` |
| 3-3 | ✅ 已修 | `7d11336` 超时兜底后强制 `new` |
| 4-1 / 4-2 | ✅ 已修 | `cc04252` 转向帧不播种 + 全 unknown 报 unknown |
| 4-3 | ✅ 已修 | `bd960de` 空间轮 / 外观轮拆成两轮 |

至此 15 条指控全部结案：14 条成立（13 条已修 + 3-1 已实现但默认关）、1 条误报（3-2）。

### §3-1 续：从票数门换到几何门，再用连通性判——结论两次反转

票数门（`q_near_pts`）实测几乎不回收（044153 上 202 个漏放只回来 1 个），因为 `ray_clear` 的看穿清零在 `_by_quality` **上游**就把 `n_o` 削了，票根本走不到那一层。于是换了条路：**几何门** `ray_near_exempt` + `_near_plane`——判据不是票数而是"这里有没有一张连贯的近距水平面"（票够 ≥20、近距票占比 ≥0.5、3×3 邻域每格**平均**高度极差 ≤0.15），靠新增的 `_acc_ob_h/_acc_ob_h2`（障碍带内 Σw·h、Σw·h²，无条件算、不进 `hi_bands` 分支）来判。

`tools/ray_exempt_ab.py` 报"回收的真结构格 / 新封死的路径格"：

| | 044153（391 帧） | 045615（3014 帧） |
|---|---|---|
| 票数门 | 回收 1 格（净亏） | 回收 123 / 封死 1155（9:1 亏） |
| 几何门 | 135 / 29 = **4.7:1** | 47 / 33 = **1.4:1** |

两段都 >1，看着划算。但**这个比值不能定案**：分子在路径外、分母在路径上，两者价值不等价；而且它数的是格数，代价却取决于**那几格卡在什么位置**。于是加了第三把尺子 `tools/ray_exempt_conn.py`（连通分量数 / 起点可达格 / 相邻点对 plan 成功率 / 绕行长度中位数）：

| | 044153 | 045615 |
|---|---|---|
| 交换比 | 4.7:1 | 1.4:1 |
| 连通分量 | 28→29（**+1**） | 34→36（**+2**） |
| 起点可达格 | 40472→39368（**−1104**） | 51439→50485（**−954**） |
| 相邻点对断裂 | **3 对**（28/38→25/38） | 0（20/37→20/37） |
| 长程点对断裂 | **1 对**（31/39→30/39） | 0（24/39→24/39） |
| 长程长度中位数 | 7.14→8.04 m（**+13%**） | 6.46→6.66 m（+3%） |

🎯 **交换比与真实代价反向**：赔率最好看的那段（4.7:1）损伤最重（4 对走过且基线能通的路断裂、可达域 −1104、绕行 +13%），赔率接近 1 那段（1.4:1）反而轻。044153 上那 29 格不是"29 个格子"，而是掐断了 3 条实际走过的通路——障碍卡在窄处时一格就够。

⇒ `ray_near_exempt` 保持 **False**；三个变体在连通性上**全部**净亏，代码里"留最不差的那个"不等于"该打开"。判据从此以 `ray_exempt_conn.py` 为准，`ray_exempt_ab.py` 的比值只作"这一刀动了哪些格"的规模感。

⚠️ 口径陷阱：exempt2 与 exempt3 是同一份数据，只把交换比分母从**百分点**改成**绝对格数**，结论就从 −113.8× 翻成 4.7:1。改法本身对（pp 的分母是全部路径格 ≈2438，把"多封死 29 格"放大成"误判率翻 88%"），但足以让同一份数据给出相反结论——对外引用前先能自辩。

**全量回归**：`.tmp/_run_tests.py` 872 条跑、FAIL=2（两条均与本次改动无交集，见下）；`research/tools/regression_place_identity.py` EXIT=0。

⚠️ `TOTAL_FAIL=2` 是**旧脚本手写清单漏跑**掩盖下来的既有问题（本次把脚本改成自动发现后才浮出来，9 个 commit 均未触及 `process.py` / `tools/` / `research/`）：

| 模块 | 失败 | 性质 |
|---|---|---|
| `test_research_isolation` | `(ROOT/"tools").exists()` 为真 | 仓库卫生：`tools/` 未并入 `research/`。20+ 个脚本仍在旧位置（含本次用过的 `q_tier_bench.py`）。要修是搬家，不是改代码。 |
| `test_standalone_ui` | `backend/process.py:329` `self.server.service._http_client_addr = ...` → `AttributeError` | 测试桩把 service 换成 `object()`，而 `do_POST` 无条件给它挂属性。生产对象是真对象所以不炸；要么桩补全、要么 `do_POST` 加 `hasattr` 守卫。 |

两者都不在 15 条指控范围内，**未动**，留待另开任务。

### 修完后回头看：三条被推翻 / 收紧的结论

1. **3-5 后果说反了**（详见该节）：差值变负时走 `return`，cam_h **冻结**而非"更频繁重算"；且线上路径先 add 再 drop、`len(_pts)` 单调不减，根本不可达。
2. **3-4 真凶是 `round()` 不是步数**：随机 4000 组实测漏格——现状 2519/3000，只加 L1 步数仍有 2537/3000（反例 `(5,5)→(9,6)` 仍漏 `(8,5)`），`L1+floor` 才归零。`round` 取"最近"格，`to_cell()` 是 floor 口径。
3. **4-2 影响面比报告更小**：`navigator._traversability_guard_reason` 只读 `sectors`、只对 `predicted_blocked` 反应，顶层 state 报不报 clear 不进决策，只影响上报口径。

---

## Issue #4 感知 / 光流

### 4-1 ✅ 转向帧先进 `_previous_gray` —— 真

`traversability.py:324-325` 无条件赋值，然后 `:343 if turning: return _unknown(...)`、`:345 if moving is not True: ...`。转向期间的每一帧都被写进参考帧，转向结束后第一帧前向帧拿"最后一帧旋转模糊帧"算光流，全局旋转在各扇区叠加伪发散率 ⇒ 假 `predicted_blocked` / 假 `predicted_clear`。**已修（`cc04252`）**：把播种拆成"三类路径"——`warmup`（没有参考帧，本帧必须播种，否则永远返回 warmup）与 `frame_gap`（参考帧失效，必须重新播种，否则永久卡在 frame_gap）**保留播种**；只有 `turning` / `motion_gate_unknown` 两条**跳过播种**（转向结束后拿到的参考帧仍是旧的清晰帧）。⚠️ 报告建议的"把赋值挪到所有门控之后"会把前两条也挪走，直接把模块卡死——这是本次唯一一处**不能照报告写**的地方。

### 4-2 ✅ 全 unknown ⇒ `predicted_clear` —— 真，但**不会撞墙**

`traversability.py:428`：`overall_state = "predicted_blocked" if blocked_sector_count else "predicted_clear"`，且返回 `available=True`。低纹理场景所有扇区 `unknown` 时顶层报"畅通"，与仓库"unknown 永不当 free"的口径冲突，该修。

但要给严重度**降级**，理由查过调用方：`navigator._traversability_guard_reason`（`navigator.py:2292-2335`）**只读 `sectors`，且只对 `state == "predicted_blocked"` 反应**，`candidates` 为空就 `return None`（不拦截）。所以顶层 `state` 失真不会让导航器"以为畅通而撞墙"——它本来就是"没证据就不拦"。影响面是 API/面板上报口径，不是安全。

**已修（`cc04252`）**：`overall_state` 改为三级——有 blocked 扇区 ⇒ `predicted_blocked`；否则还有 unknown 扇区 ⇒ `unknown`；否则 `predicted_clear`。`:437-438` 的 `"turning": False, "moving": True` **没改**：那两处其实是对的（`:343`/`:345` 已把 turning / 非 moving 过滤掉了，能走到这里的必然是前向帧），报告说它是失真属误判。

### 4-3 ✅ 外观相似度覆盖已过 IoU 的空间匹配 —— 真，与注释相反

`vision.py:2126`：`if score <= best_score and item.get("descriptor") is not None`。候选 A 的 IoU=0.60 先过线（阈值 0.55），`best_score=0.60`；候选 B 的 IoU=0.0 ⇒ `score=0.0 <= 0.60` ⇒ 仍去算外观，`appearance=0.94` ⇒ `score=0.94` 反超 ⇒ 覆盖掉正确的空间匹配。

而 `:2118-2120` 的注释写的是"ID 注册表短暂重建时**先尝试严格同屏框重叠；仍不匹配才计算** 8x8 外观描述子"——"仍不匹配"= IoU 全军覆没，不是"当前这一个没匹配上"。

**已修（`bd960de`）**：没采用核查阶段写的 `if score == 0.0 and ...`。那个最小改法只堵住了"当前候选 IoU=0"这一个入口，结果仍依赖 dict 迭代顺序：远处撞脸候选排在前面时先以外观 1.0 占位，同屏候选轮到自己时它自己的外观分救不回来，照样被挤掉。改成真正的**两轮**——第一轮只比框（IoU≥0.55 取最大），第一轮一个都没选上才进第二轮算外观（阈值 0.94 不变）。两种插入顺序都必须选同屏候选，用例已锁。

顺带收敛了开销：外观描述子从"每个候选都可能触发一次 JPEG 编码"变成"缺帧时最多一次、编码失败即跳出"。

---

## Issue #5 OSC / 驱动日志 / 动作 / 行为

### 5-1 ✅ `monotonic_ms=0` —— 真（假零）

`driver_log.py:685-686`：`reported = action.get("monotonic_ms") or 0`；`event_time = reported/1000.0 if reported else now`。驱动自报 0（录制起始帧）时落到 `now`，`frame_index_at` 用错基准 ⇒ 帧号错。改成 `is None` 判断即可。

### 5-2 ✅ HMD sink 错误被立刻清零 —— 真，且是"自己人漏了"

`driver_log.py:465` 在 `_apply_event_locked` 里写 `self._last_error = f"HMD telemetry sink failed: {exc}"`，但 `ingest_packet` 在 `:398` 每次成功解码后无条件 `self._last_error = None` ⇒ HMD 的持续性故障在 `snapshot()` 里永远看不到。

最有力的证据是**代码自己的注释**（`:288-290`）：`_last_action_error` 就是为了这件事才从 `_last_error` 里分出来的——同一个坑，action sink 补了（`:483`），HMD sink 漏了（`:465`）。照抄一遍即可。

### 5-3 ✅ `episode_action_summary` 直接 key 访问 —— 真（低）

`driver_log.py:982/984` 用 `r["input_command"]` / `r["turn_intent"]`。`load_action_timeline`（`:899`）确实接受任意 `record=="action"` 的 dict（`:921-922`），坏行只跳过 JSON 解析错误、不校验字段。本仓库 `record()`（`:701/705`）恒定写这两个字段，所以**只有外部/旧版 JSONL 会触发**。改 `.get(..., {})` 零风险。

### 5-4 ✅ `resolve_expression` 未知 intent 抛 KeyError —— 真

`behavior.py:89`：`profile = EXPRESSION_PROFILES[intent]`。上游 `service.py:3081` 只做 `intent = str(params.get("intent") or "")`，**没有**拿 `EXPRESSION_INTENTS`（`behavior.py:71`）校验；VMD 分支（`:3091`）未命中或 `prefer_vmd_expressions=False` 时直接掉进 `resolve_expression`。VLM 返回的 intent 或外部调用会让它崩。改成显式校验并抛 `ValueError`（或回退）都行，但**要选一个口径**并同步 `tests/test_behavior.py`。

### 5-5 ✅ `bind()` 失败泄漏 socket FD —— 真（低）

`osc.py:319-330`：`receiver` 是局部变量，`bind()` 抛 `OSError` 时只置 `self._receive_socket = None`，从不 `close()`。重试循环下每次泄漏一个 FD。加 `try/finally` 或 `except` 里 `receiver.close()`。

### 5-6 ✅ `max_age_ms=0` 假零短路 —— 真（低）

`osc.py:1027`：`if value_age_ms is None or (limit_ms and value_age_ms > limit_ms)`。`limit_ms = max(0, int(max_age_ms))`（`:927`）**明确允许 0**，而 docstring（`:921-922`）承诺"样本超过 `max_age_ms` 后统一返回 `available=false`"——传 0 应该是什么都不认，实际变成什么都不拒。改 `if value_age_ms is None or limit_ms == 0 or value_age_ms > limit_ms`。

### 5-7 ⚠️ 手势头部旋转用世界系 —— 逻辑成立，但默认路径不可观测

`motion.py:403/410/421/511` 用前乘 `quat_multiply(axis_angle(world_X, pitch), hmd.rotation)`。`quat_multiply(a,b) = a⊗b`（`motion.py:56`，标准 Hamilton 积），前乘的轴在**世界系**解释 ⇒ 有 yaw 时点头变侧倾。语义上应该是 HMD 本地轴 ⇒ 后乘。

**但要如实降级**：`start = self._frame.clone()`（`scheduler.py:1107`），而中性帧的 HMD 是 `IDENTITY_QUAT`（`model.py:98`）；base 为 identity 时前乘与后乘**结果完全相同**。play-space yaw 是在输出阶段统一叠的（`scheduler.py:895 _apply_play_space_yaw`，`quat_multiply(rotation, device.rotation)`），在手势合成之后。所以只有当 `_frame` 带上真实 HMD 朝向时才可观测——`scheduler.py:1039`（外部帧赋值）和 `:1385`（clip 采样结果）是两条可能的来源。

结论：**改**（默认路径零风险、语义更对），但别指望改完能看到区别；`expression_motion.py:65/66/69/70/73/74/80/81/222/238` 是同一套前乘写法，要改就一起改，别只改一半造成同仓库两套约定。

---

## 建议的落地批次

- **批次 1（零/低风险，不改判据）**：3-4、3-5、4-1、4-2、5-1、5-2、5-3、5-5、5-6、5-7（含 `expression_motion.py` 同套写法）。全部是"让实现回到它自己注释说的样子"，不改变任何现有语义。
- **批次 2（改语义，需回归）**：3-1（近带豁免真正生效 ⇒ 近距细障碍变多）、4-3（外观兜底收紧 ⇒ 串号减少但匹配率可能降）、5-4（需先定"校验失败抛 ValueError 还是回退"的口径）。
- **不修**：3-2（误报）。

按仓库 binding：3-1 属于改判据，落地前必须重跑 `tools/regression_place_identity.py`（EXIT=0）并补近距细障碍用例；任何参数门槛改动都要同步这份文档的结论。
