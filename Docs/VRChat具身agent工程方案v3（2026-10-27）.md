# VRChat 具身 agent 工程方案 v3（2026-10-27，拓扑 A）

> **状态**：待评审（round 10 已改；评审更正记录见附录 A），取代 v2（`.tmp/_plans.txt` L4411 起）。v2 中本文没有改动的部分（头 A 的主干与输入、头 B 的离线片段库、技能卡格式）继续有效，这里不再重复。动作块长度、丢弃规则和 RTC 的用法本文改了（§1、§9.1、§13）。
> **依据**：§1 是 2026-10-27 在 A770 上的实测。文中引用的函数、参数和默认值都已对照 `backend/` 源码核对。没核对到或没测过的项标「未测」或「待 M0」。
> **范围**：只写选型、理由、接口约定和验收门，不含实现代码。
> **硬约束**（用户原话）：「绝对不能出现误删，点云也不能出错」。本方案新增的模块都不删点，也不改建图路径。唯一会影响建图的是 §3 的头部俯仰，为此单设门 G2。
> **拓扑 A**：本地 A770 16 GB（与 VRChat VR、LAS2 共用）+ 云 API，用户 2026-10-26 选定。慢脑的地点记忆图格式见 §10.3。

## 0. 相对 v2 的更正

| v2 的写法 | 更正 | 见 |
|---|---|---|
| 头 A 80–150 ms、3 Hz | 编译后 224² 单帧 p50 约 150 ms（ViT 46 + LM 103），p95 约 190 ms。这是 GPU 空闲时测的 | §1 |
| 3 Hz 时 GPU 占用 39–63% | 按「延迟 × 频率」算，224²@3 Hz 是 45%，加基线 48.6% 共 94%；@2 Hz 是 30%，共 79%。基线改为 2 Hz | §1 |
| 显存 1.3 GB | bf16 推理峰值分配实测 1.67 GiB，其中权重 853M × 2 B = 1.59 GiB。预算按 2.0–2.5 GB | §1 |
| 现有录制有 5 Hz 彩色流 | 没有。只有关键帧原图 `kf_img/`，`record_images` 默认关，关键帧间隔中位 0.16–0.45 s 且不等距。改为新增定频的 policy_obs 流。**所以现有录制的 osc 标签（15 场剔挂机后 16.4k 个 5 Hz 窗口）没有配对的图，训不了头 A**；带图的只有关键帧缩略图和原图 4,291 个，92% 是移动。M0 的主训练集要先新录（§9.2） | §5、§9 |
| 接管检测用 `GetManualControl()` | 这个接口不存在。OSC 回读也分不出玩家和策略（同形）。改为显式接管按钮（M0）+ 期望/回读速度失配（M1）；考场另用 Udon 日志 | §8 |
| 动作块 10 步 × 1 s；距观测 > 0.5 s、位移 > 0.3 m、转过 > 15° 即丢弃（按 ≥3 Hz 设计） | 头 A 训 12 步 × 0.1 s；2 Hz 下用到前 8 步 = 0.8 s（周期 0.5 s + p95 延迟 0.19 s + 迟到余量），1 Hz 降级时用整块 1.2 s，不重训。新块到即替换，丢弃阈值「距观测 > 0.8 s」（1 Hz 为 1.3 s），M0 不需要 RTC。位移/转角两条在行走速度下每块必触发，M0 不用，M1 改为期望/回读失配 | §1、§9.1 |
| 训练吞吐 150 样本/s | 按实际输入（单帧 45–49 视觉 token + 70 文本）重算 | §11 |
| 云端靠裁剪降本 | 百炼官方把图像折成 token 计费（每 32×32 像素 1 token），不按张。一次关键帧调用约 ¥0.002–0.003，其中输出 token 占大头 | §2 |
| patch 14 | Qwen3.5 视觉 patch 16、merge 2。224² 是 196 patch → 49 token；448² 是 784 patch → 196 token。`temporal_patch_size = 2`（config.json 实测），两帧沿时间轴打成一个 patch，token 数和单帧相同；两帧当两张图喂才翻倍（448² 是 392 token） | §1 |
| LoRA 25M | 按真实模块形状数（12 种线性层），r=32 加在全部线性层是 21.65M，不含 MLP 是 11.03M | §11 |

## 1. 实测基线与 GPU 预算（头 A 的设计依据）

**测法**：A770，GPU 空闲（VRChat 未开），Qwen3.5-0.8B（GameScaling 的底座，结构相同），bf16，单次 prefill，`use_cache=False`，像素随机，每档 20 次取中位。脚本在 `.tmp/qwen35_{breakdown,trisolve,compile}.py`。

| 档位 | ViT p50 / p95 | LM p50 / p95 | 合计 p50 / p95 | 说明 |
|---|---|---|---|---|
| transformers 原样（torch 回退） | 91–96 ms | 651 / — ms（224²）、711 ms（448²） | 0.75–0.81 s | 18 层线性注意力占 LM 的 91–98% |
| + trisolve（自写，数值等价 cos 0.99996） | 108 / 124 ms | 257 / 274 ms（224²） | 0.36 / 0.40 s | 逐行前代 63 次改成一次三角求解 |
| + torch.compile（default） | **46 / 55 ms（224²）**、55 / 62 ms（448²） | **103 / 134 ms（224²）**、164 / 180 ms（448²） | **0.15 / 0.19 s（224²）**、0.22 / 0.24 s（448²） | 与 eager 的 cos 0.99989；首次编译 525–631 s |
| + compile（reduce-overhead） | 46 / 52 ms（224²） | 124 / 148 ms（224²） | 0.17 / 0.20 s | XPU 上 cudagraphs 被跳过（日志 "multiple devices"），没有收益 |

- 合计 p95 是 ViT p95 + LM p95 的保守和（两段各自的 p95 不一定落在同一次调用，真实合计 p95 只会更小）。

- 文本按 70 token 算（航点 + 本体状态）。224² 共 119 token，448² 共 266 token。
- **头 A 只吃左目**，输入是「当前帧 + 0.4 s 前那一帧」，沿 `temporal_patch_size = 2` 的时间轴打成一个 patch（沿用 v2「左眼当前帧加前一帧」；v2 写 0.5 s，这里取 0.4 s 是为了落在 policy_obs 的 5 Hz 网格上）。token 数和单帧相同，上表就是这个口径：bench 的像素张量已经是 `3 × T × P × P`（T = 2）、`grid_thw = [1, gh, gw]`，等于两帧都算了，ViT 和 LM 都不多花。
  - **不用双目两张图**：两张图各 49 token，ViT p95 55 × 2 + LM 134 × 168/119 ≈ 299 ms，2 Hz 占 60%，加基线 48.6% 是 108.6%（p50 也约 96%），G1 必挂。右目只进 LAS2 测距和 policy_obs 存档，不进头 A。
  - 两帧中较旧的一帧取上一个 5 Hz 拍的缓存，不另外采集；会话开头不足 0.4 s 时两帧都填当前帧。
- fla 的 Triton kernel 在 A770 上编译失败（`chunk_scaled_dot_kkt_fwd_kernel` 需要 333 KB scratch，硬件上限 256 KB），这条路已排除。transformers 本来就只在 CUDA 上启用 fla。
- 推理峰值分配 1.67 GiB（全部权重 853M 参数 × 2 B = 1.59 GiB）。显存预算按 2.0–2.5 GB 留，包含动作头和激活。
- **编译缓存要搬出 C 盘**：首次编译 9–10 min，C 盘只剩 3.9 GB（2026-10-27 实测）。M0 把 inductor/Triton 缓存目录指到 H:，启动时像 LAS2 一样后台预热，预热期间控制环按「无策略」处理。

**GPU 预算**（占用 ≈ 单次延迟 × 频率，取 p95 延迟）：

| 配置 | 头 A p50 / p95 占用 | + 基线 48.6%（VRChat 33.4 + LAS2 15.4，实测 2026-10-27 A770） | 判断 |
|---|---|---|---|
| 224² @ 3 Hz | 45% / 57% | p95 时 105% | 超预算，不选 |
| **224² @ 2 Hz（基线）** | **30% / 38%** | **p50 79% / p95 87%** | 留给关键帧异步模块和争用约 13% |
| 448² @ 2 Hz（升档候选） | 44% / 48% | p95 时 97% | **升档后总占用仍 ≤ 85% 才开**：448² 自身 p95 占 48%，基线要 ≤ 37% 才够；按现基线 48.6% 不开 |

- 三行都按上表合计延迟算：224² 是 150 / 189 ms（ViT 46 / 55 + LM 103 / 134），448² 是 219 / 242 ms（ViT 55 / 62 + LM 164 / 180），× 频率得占用。

**2 Hz 为什么够**：头 A **训 12 步 × 0.1 s = 1.2 s** 的块（v2 是 10 步 × 1 s，按 ≥3 Hz 设计），2 Hz 下只用到前 8 步（索引 0–7，0.8 s）。**新块一到就替换旧块**，控制环 10 Hz 按 `(now − t_obs)/0.1 s` 在当前块里取步。稳态下新块在 t_obs + 延迟（p95 0.19 s）到达，下一块在 t_obs + 0.69 s 到达，所以实际用到的步索引是 1–6；下一块迟到时用到索引 7，到 0.8 s 丢弃。索引 8–11 只在 1 Hz 降级时用（见下）。块尾之后**沿用最后一步**（零阶保持），不外推。
- **丢弃阈值**：`now − t_obs > 0.8 s`（2 Hz）或 `> 1.3 s`（1 Hz 降级）时丢弃当前块并停车。v2 §5 的另外三条里只留「急停清空」。「观测后位移 > 0.3 m」「转过 > 15°」M0 不用：行走 1.1 m/s 时 0.8 s 内位移约 0.88 m，每块必触发，等于每块都丢。M1 由 §8.1 路径 B（期望/回读失配）接替。
- **为什么不用 1 s 块 + 「超块长才换」**：那样新块 0.5 s 就到了却要闲置 0.5 s，最老一步是 0.5 + 0.19 + 0.5 = 1.19 s，延迟翻倍。块长略大于推理周期时，相邻块的预测时段只重叠一个延迟窗，换块跳变在 M0 用执行器的限加速度消掉，不需要 RTC（见 §13）。
- **1 Hz 降级**：同一个 12 步头，不重训。下一块在 t_obs + 1.19 s 到达，所以用到整块（索引 1–11，块长 1.2 s ≥ 1.19 s），丢弃阈值改为 1.3 s（= 1.0 + 0.19 + 0.11），1.2–1.3 s 沿用最后一步。块长若只训 10 步 × 1 s，1.0–1.19 s 要靠零阶保持撑 0.19 s，迟到时就是 0.3 s，所以取 12 步。原来 7 步块直接搬到 1 Hz 会在 0.8 s 丢弃、1.19 s 才来新块，每周期停车约 0.4 s。
- **训练成本**：索引 8–11（0.8–1.2 s）在 2 Hz 下不执行，但仍有 osc 标签（§5.2），loss 照算，只多 4 × 2 个输出。

**门 G1（M0，必须在 VRChat VR + LAS2 同跑时测）**：
- 头 A 224²@2 Hz（左目两帧时间打包，同上）连跑 10 min，**单次推理 p95 ≤ 200 ms 且 GPU 总占用 < 90%**（与上面预算表同口径：头 A 占用 = p95 × 2 Hz，加同场实测基线），块丢弃率（`now − t_obs > 0.8 s`）≤ 1%。
  - 200 ms 对应 40% + 48.6% = 88.6%，预算表的 189 ms 是 87%。**两个条件都要过**：基线高于 48.6% 时，p95 ≤ 200 ms 也可能超 90%。原写的「p95 ≤ 250 ms」是 50% + 48.6% = 98.6%，按它过了 G1，G4 的「GPU 总占用 < 90%」必然不过，已改。
  - **余量只有 3 个百分点**（87% 对 90%）。189 ms 是 GPU 空闲时测的，同跑 VR + LAS2 时头 A 自身延迟大概率会涨（争用），基线 48.6% 虽是同跑实测，两者叠加没测过。**通过概率按「中等」计**，M0 排期按「可能要走降级 ①」留时间。降级 ① 因 12 步块不用重训，代价只是改执行频率和丢弃阈值。
- 位姿频率（现 14.0 Hz）和双目频率（现 7.8 Hz）相对同场不开头 A 的读数，掉幅 ≤ 10%。
- VRChat 帧时间 p95 增幅 ≤ 10%（读 SteamVR 帧计时，读法在 M0 定）。
- 显存总占用 ≤ 14 GB。

**不过门时的降级顺序**：① 1 Hz（头 A 占用 p95 19%，加基线 67.6%），用整块 12 步、丢弃阈值 1.3 s（见上），同时把地图内导航交还几何控制器（`nav_follow`），头 A 只管地图外；② 只留 ViT（46 ms）加一个约 10M 的小动作头，放弃 LM 的游戏预训练；③ 拓扑 B（加 N 卡，需用户拍板）。

## 2. 选型、云端计费与许可证门（P1-7）

| 槽位 | 选型 | 跑在哪 | 许可证（核对来源） | 状态 |
|---|---|---|---|---|
| 快策略主干 | GameScaling-0.8B（Qwen3.5） | A770，torch XPU + compile | Apache-2.0（repo LICENSE；HF 模型卡 M0 复核） | 未接 |
| 关键帧语义 + 开放词汇框 | qwen3.6-flash | 云 | 服务条款（不随插件分发） | 未接 |
| 技能整理、难例规划 | qwen3.6-plus | 云 | 同上 | 未接 |
| 慢脑 | N.E.K.O AGENT_MODEL | 云 | 宿主已有 | 已有 |
| 双目 | LAS2 | A770 子进程 | MIT | 已接（默认 las2，未提交） |
| 地点检索 | MegaLoc | A770 关键帧异步 | MIT | **仓库里还没有**，M3 接 |
| 人物检测 | person_detect_v1.3_s（ONNX） | OpenVINO | **待定**：Ultralytics 训练导出，AGPL 主张未解（`THIRD_PARTY_NOTICES.md` L15）。M0 如果判 AGPL，替换为 Apache-2.0 的 **RT-DETR**（640×640 ONNX，OpenVINO fp16，未测）或 **YOLOX-s**（同许可证，但精度比 RT-DETR 低） | 已有 |

**云端计费**（百炼官方，华北2 北京，2026-10-27 查）：
- qwen3.6-flash 输入 ¥1.2/M token、输出 ¥7.2/M token（单次 ≤256K），Batch 半价，上下文缓存命中按输入价的 10%。
- qwen3.6-plus 输入 ¥2/M、输出 ¥12/M。
- 图像折算成 token：每 32×32 像素 1 token，外加 2 个边界 token。448² 是 198 token，224² 是 51 token。
- v2 写的「¥0.6/张」在官方价目里找不到，撤回。
- 一次关键帧调用（448² 一张 + 约 300 token 提示，输出约 300 token 的 JSON）≈ ¥0.003，输出占 3/4。
- 结论：钱不再是约束。约束是云端往返延迟（慢脑实测约 5 s）和限流。调用频率按「新地点才调」控制，不按钱控制。

**开放词汇检测不用 YOLOE**：GitHub 页面标 AGPL-3.0，README 写明代码基于 ultralytics，与本仓库的 Apache-2.0 分发冲突。替代方案是在关键帧语义那次调用里让 qwen3.6-flash 一并返回物体框（Qwen VL 的 grounding 输出）。这样不多花钱，也少一个本地运行时。云端断开时只记几何，不做开放词汇。本地备选 OWLv2（Apache-2.0）在 A770 上未测，M3 不过门时再评估。

**门 G0（许可证，M0 第一天）**：
- 每个随插件分发或运行时下载的组件，都要在 `THIRD_PARTY_NOTICES.md` 里记下许可证、来源链接和核对日期。
- 与 Apache-2.0 分发不兼容的组件，要么替换，要么由用户签字接受。person_detect 的去留在这一步定（替换候选见表格）。
- VRChat 服务条款：自主控制只在私人实例或自建考场世界里跑，公开实例只开影子模式。
- **隐私掩膜**：上云关键帧（§10.2）前在**本地**用 person_detect（或替代品）生成人物+名牌 bbox 掩膜，涂黑后再上传。如果本地检测不可用（AGPL 未解且无替代），改为 bbox-based masking（在 cognition.py 里按视锥几何裁剪画面下 1/4 和远端 > 5 m，只传中近景）。

## 3. 俯仰视野与交互射线（P0-1）

**问题**：scheduler.py 只实现 yaw 控制（scheduler 的 `turn` 命令 + `_apply_play_space_yaw` 只绕 Y 轴旋转全部设备位姿），HMD 没有外部俯仰命令。motion.py 能读 `_quat_to_euler_deg` 的瞬态 pitch 快照，但快照不是命令。motion.py 的 `nod` 手势里有 24° × sin² 的 X 轴旋转（L406–410），所以 **HMD 设备本身能写 pitch**，只是 scheduler 外部接口不暴露。`arm_pose_target(elevation_deg, azimuth_deg, reach)` 和 `directed_wrist_rotation` 是手臂朝向原语，但没有组合成 `hand_aim` 命令。

在头无俯仰的约束下，以下三类任务不可达：
1. **look_at / scan / aim_controller**：多数物体（门、台阶、物品、天花板、蹲着的人）在高低错开的世界坐标下，没法让头和朝向同时指向目标。门框会从摄像头离开。
2. **press/interact**：交互半径 3.5 m，但没有射线和准星。M2 要求「注视中心附近的对象时命中率 ≥0.7」，只有平视不会漏。
3. **VLA 俯仰不耦合策略**：策略输入是 ViT 图像特征（无 focal length、无绝对位姿），推理靠泛化，但训练只有「平视移动」的先验，不会从图像里逆出俯仰 DoF。

**方案**：加两个命令。M0 接 turn_and_look 的 pitch，M2 前接 hand_ray_aim。

### 3.1 turn_and_look（pitch 分支）

在 scheduler.py 新增命令 `turn_and_look(yaw_deg, pitch_deg)`。yaw 复用现有 `_apply_play_space_yaw`，pitch 只改 HMD 设备 `rotation`（motion.py 的 `nod` 手势 L406–410 有 sample：`axis_angle((1.0, 0.0, 0.0), pitch)`），**不调运动学重算**——双臂、髋、脚位置按现有 `_compose` 逻辑从 base 继承，不因 HMD pitch 变化而变化。如果门 G2 测出俯仰头时双臂扭曲，再补运动学约束（M0 实测后定）。
- **范围：−30 ~ +70°**（抬头看天 70°、低头近处 −30°），超限 clamp。
- **闩锁门与 `turn` 完全一致**：`disabled` 下放行，`LATCHED_STATES`（`stopped_latched` / `fault_latched` / `shutdown`）下拒绝。依据是 scheduler.py L143–151 的注释和 `submit` 的门（L337–338 `kind == "turn" and state in LATCHED_STATES`）：急停的不变量是「不动」，不只是「不走」，所以闩锁期间连原地转向都挡。pitch 虽然不改 play space yaw、不产生位移，但它改 HMD 朝向，同样违反「不动」，不开例外。`disabled` 放行的理由与 `turn` 相同：宿主不在线时开 body 输出会把角色拽成 T Pose，导航必须能在输出关闭时转头（`test_turn_works_while_body_output_is_disabled`）。
- **闩锁时 pitch 保持当前值**，不自动回零（回零本身也是一次运动）。解锁后由下一条 `turn_and_look` 指定新值。M0 给 `turn_and_look` 补一条同名的闩锁单测。
- **pitch 不进 OSC**。AnyaDance 没有 `/input/LookVertical`，也不该走仲裁器。

### 3.2 hand_ray_aim（交互射线）

M2 再加 `hand_ray_aim(target_xyz, hand='right')`。它调 motion.py `directed_wrist_rotation` 算从肩到目标点的向量，然后逆运动学填 `arm_pose_target(elevation_deg, azimuth_deg, reach)`，这样手柄射线能指向门把手，看头再俯仰到门框。

- 在 interactor.py `try_interact` 里优先查几何射线能不能命中（ray_cast `Interact` layer，现在只有半径，M2 改），能命中就选那个，不能则退回距离过滤器和全场扫描的候选。
- 手势不能跟命令重叠，所以控制环在发 `hand_ray_aim` + `input_button` 前，先检查 scheduler 当前没有 `gesture` / `play_clip` / `sequence` 等会控制手臂的命令在执行（读 `status()["motion"]`，`motion` 为 `None` 时才发），避免手势和射线瞄准冲突。

**门 G2（head pitch 不能破坏现有建图，M0 按优先级）**：
1. 平地行走，记录时不开策略，回放 nav_online 的 2 Hz 建图。回放前手工 patch `scheduler._apply_turn_and_look` 按 0 / +30 / −15° 的 pitch 周期切换，建图正常完成、无死帧。
2. 用 pitch 的录制（策略下发 pitch −20 ~ +40°）开 `record_images=True`，把关键帧左右原图离线跑两个后端（`research/tools/las2_bench.py` 同口径载模型）。⚠️ **不能数 LAS2 的「视差 > 0」**：LAS2 是 soft-argmax 回归（`las2_model.py` L80–82），每个像素都输出 `[0, 192)` 里的一个值，没有无效标记，这个计数几乎恒为 100%，测不出退化（同 §10.4）。改测三项：
   - **SGBM 对照组**（有无效标记：`stereo_disparity` 输出 ≤ 0 即无效，nav_mapping.py L394–397）：对同一批帧跑 `make_sgbm(disparity_range_px(fx))`，统计有效像素占比。pitch 帧对比同路径平视基线（pitch = 0）衰减 ≤ 15%。
   - **LAS2 对 SGBM 的一致性**：取两者都落在 0.5–3.5 追踪米、且 SGBM 有效的像素，深度相对差 `|z_las − z_sgbm| / z_sgbm` 的中位。pitch 帧比平视基线高出不超过 1.5 倍。LAS2 在看不清的地方不会报无效，只会和 SGBM 分歧，所以分歧才是能测的量。
   - **近场像素占比**（z < max(0.5, 1.2 × z_min) 追踪米，§10.4 同口径）：低头时地面本来就近，这一项会自然上升，**只记录不设门**。它和第一项一起用来判断，有效像素衰减是因为「看到地面」还是因为「图像退化」。
3. 一次在线建图走 10 m，pitch 随机 −20 ~ +40°，可走面积不小于同路径关键帧 pitch=0 时的 85%。

## 4. 与现有世界模型/记忆模块的整合（P0-2）

### 4.1 现状梳理（仓库已有）

| 模块 | 职责 | 接口 | 全局状态 |
|---|---|---|---|
| world_model.py | 世界标识（world_key，用户手动输入），长期身份持久化 | `WorldModel.set_world(world_key, world_name)` / `.identity()` | `<state_dir>/world_identity.json` |
| nav_memory.py | 关键帧/位姿存盘，按 world_key 子目录 | `NavMemoryStore.begin/end`（会话级）+ `SessionWriter.keyframe/poses`（写帧与位姿表）| `<state_dir>/navmesh_memory/`（`VRC_NAVMESH_MEMORY_DIR` 可覆盖，service.py L453–454） |
| nav_prior.py | 世界先验（离线融合产出的 FREE/OCC 栅格）的在线消费：攒跨会话约束估 gauge，过质量闸才注入当前地图；全局闸不过时退分片档 | 写：`write_prior(world_dir, labels, ...)`（`research/tools/offline_fusion.py --write-prior`）；读：`PriorConsumer`（nav_online.py L1264 构造，`session.prior = overlay`） | `<world>/prior/prior.npz` + `prior.json`（成对原子写）；状态在 `status()["prior"]` |
| nav_xsession.py（75KB，活跃）| 跨会话地点检索（BoW + 几何验证）、gauge 估计、自动采纳 | `XSessionTracker(cfg, world_dir, sid)` → `.on_keyframe` / `.write_back` | `<world>/sessions/<sid>/` 索引快照 |

- **世界键来源**：VRChat OSC / OSCQuery 拿不到世界标识（官方地址表只有 `/avatar/*`, `/input/*`, `/chatbox/*`, `/tracking/*`），world_key **只能靠用户手动输入**（world_model.py docstring 第 2 条）。用户通过 `/worldmodel` 接口（`service.world_model_set_world`，L2160）设置。`wrld_` 开头视为稳定 ID（`manual_id`，转小写）；其余视为世界名（`manual_name`，原样保留，有同名风险，`world_conflict_risk=true`）。
- **记忆目录名**：nav_memory.py `world_dir_name()` = `sanitize_world_key(key)[:80]` + `-` + SHA-1 前 8 位。sanitize 把 `[A-Za-z0-9._-]` 以外的字符换成 `_` 再去掉首尾，所以中文被吃掉、靠哈希区分。例 "Just B Club（社畜版）" → `Just_B_Club-76ff55ab`。
- **gauge 质量闸**（阈值在 nav_prior.py `PriorConfig`（L58 起，全局闸参数在前、`tile_*` 分片参数在 `max_split_*` 之后），判定在 `_gate` L624–645 和 `_cross_check` L647 起）：有两个**口径不同**的位置门，不能合成一句：
  - **残差门** `max_pos_med_m = 0.5`：IRLS 内点核心的位置残差中位，量的是「这个 gauge 拟合得多紧」。同组还有约束数 ≥ 8、内点 ≥ 6、内点占比 ≥ 0.2、旋转残差中位 ≤ 5°、核心跨度 ≥ 0.5。
  - **前后半交叉门** `max_split_m = 0.6` / `max_split_yaw_deg = 2.5°`：按时间切成前后两半各估一个 gauge，比它们把**同一批新帧**映射到哪儿，量的是「gauge 是否随时间漂」。它挡的是残差全过但 yaw 错 6° 的自洽错解（2026-10-06 事故）。
  - 全部通过才注入；全局闸拒了退分片档（`tile_*`，片内约束 ≥ 6、内点 ≥ 4，约束 ≥ 8 才补做前后半）。另要求先验的 `base_sid` 与约束引用的旧会话表同根。世界目录下没有 `prior/` 时什么都不注入。

### 4.2 地点+物体记忆的上游与同步（接 §10 慢脑的地点记忆图）

- **地点子图在 cognition.py**：世界有唯一的地点子图，格式见 §10.3。地点节点的 `pose` 是**写它的那个会话**的视觉 SLAM 坐标下的位姿（与 keyframe 的 `T_map` 同系）；跨会话载入时旧节点先标 `stale`，对齐后才可导航（见下「跨会话载入」）。
- **物体的 `pose` 也在 T_map 系**：关键帧检测到物体时写下 `T_map @ T_body @ T_cam @ depth_point`。没有全局对齐。
- **回环闭合后不追溯**：loop closure 会改回环前关键帧的 `T_map`，但已记录的地点和物体 pose 不重算，停留在旧坐标系里。误差与回环闭合前的 DR 漂移同阶（约 0.45 m，见 memory `online-navmesh-wiring.md` run6），最大可达米级（045615 的 2.3 m 是极端情况）。**这是故意的**：人的空间记忆本来就不是实时重投的全局坐标，地点之间的连通性才是关键。
- **关键帧写盘接口**：`NavMemoryStore.begin(identity, meta)` 返回 `SessionWriter`，然后 `SessionWriter.keyframe(k, feat, thumb, refresh)` 写特征和缩略图，`SessionWriter.poses(table)` 写位姿表，会话结束时 `NavMemoryStore.end(table, status)`。关键帧原图（kf_img）在录制模块写（`record_images=True` 时），不走 NavMemoryStore。
- **持久化计划**：
  - **地点记忆图持久化在 cognition.py**，格式见 §10.3。M3 在 `cognition.py` 里加 `save_graph({nodes, edges})` 和 `load_graph() -> dict`，写到 `<world_key>/graph_<timestamp>.json`（与关键帧同世界目录，但不归 `NavMemoryStore` 管）。约每 30 s 或每 10 节点增量存一次（navigator.py 判断）。
  - **坐标系按节点记，不按快照记**：快照是累积的，B 场保存的图里同时有 A 场写的节点和 B 场新写的节点，所以「快照 sid」说明不了节点在哪个系。每个节点带 `pose_sid`（写入该节点的会话，nav_memory 的 `session_id`），`pose` 是**那个会话的原始** `T_map` 系。
  - **落盘时 `pose` / `pose_sid` 保持原值**：本会话里落位算出的当前系坐标只在内存里用，不写回。写回等于把一次 gauge 误差固化进去，下一场再叠一层。代价是 `pose_sid` 那场的会话一旦被删，该组就永远没有约束：M3 让 `NavMemoryStore.prune()` 跳过仍被地点图节点引用的会话（复用 pinned 判断，nav_memory.py L529）。
  - 物体不存，只存地点。关键帧里物体框的 JSON 本地不存，只在远程调用时记在那次调用的返回缓存里（15 min TTL），缓存见 §10.2。
- **跨会话载入（坐标系对齐，M3）**：新会话的 `T_map` 是 DR 从新起点建的，原点和朝向基准都与上一场不同。旧节点的 `pose` 原样放进新系，偏差就是两场之间的 gauge 差（米级），`goto_place` 会把 agent 送到错位置。现有代码**不会替我们做这件事**：nav_xsession 在 v1 范围内「确认的跨会话对只写约束 jsonl，不碰位姿图」（nav_xsession.py docstring L23–25）；nav_prior 的 `PriorConsumer` 虽然在线估 gauge，但只用来把先验**栅格**投影进当前地图（`project_prior` L300），不碰地点图。所以采用「**先载拓扑、命中后再落位姿**」：
  1. **起会话只载边与标签**：`load_graph()` 读最新快照，节点的 `summary` / `visit_count` / `frontiers` 和 `edges` 照常载入，但旧节点的 `pose` 一律标 `stale`（各自保留 `pose_sid`），不参与 `goto_place` 的目标解算，也不画进当前地图。慢脑可以照常按名字推理「厨房在客厅隔壁」，但导航目标只能落在已对齐的节点上。本会话新写的节点 `pose_sid` = 当前 sid，直接是 `live`。
  2. **按 `pose_sid` 分组，每组各估一次 gauge**：旧节点按 `pose_sid` 分组。对每组 s，只取**引用 s 的**跨会话约束（`XSessionTracker.on_keyframe` 返回的记录里 `old_sid == s`，nav_xsession.py L1321–1322；nav_online.py L1942–1944 已在建图线程拿到），调 `estimate_gauge`（nav_xsession.py L469，输入旧/新 kf 位姿 + 相对位姿 `R_ab/t_ab`，IRLS 剔外点，返回 `R_G/t_G` 与残差统计）。gauge 方向是新→旧（`p_old ≈ R_G @ p_new + t_G`），所以该组节点进当前系用逆变换 `p_new = R_G^T @ (p_old − t_G)`。**逐组落位，组与组互不影响**：A 组过闸不会把 B 组一起搬走，反之亦然。
  3. **闸门逐组判，沿用 nav_prior 的两个口径**（§4.1）：残差门（约束 ≥ 8、内点 ≥ 6、`pos_med` ≤ 0.5 m、`rot_med` ≤ 5°）和前后半交叉门（`max_split_m` 0.6 / 2.5°）都过，才把**这一组**落位。不过闸、或根本没有引用该 sid 的约束（这场没重访那一场看过的地方，或那场不在索引里），该组就一直 `stale`，**宁可不用也不送错地方**。
  4. **坐标系根逐组检查**：s 组节点的系是 s 的**原始** `T_map`，而 `XSessionTracker.old_pose`（nav_xsession.py L1191）返回的是索引快照系（merged 优先）。s 被并过树时两者不同根，所以每组先查 `table_base(s)`（L1208）：返回 s 自己（raw）才能用这组约束；返回别的 base_sid 或 None，这组保持 `stale`，不猜。与 `PriorConsumer._usable` 的「绝不混 gauge」同一原则。
  5. **回环后逐组重估**：当前会话回环会改新帧位姿（与 `PriorConsumer.note_map_moved`，nav_prior.py L425、nav_online.py L1960 同一时机），每个已落位的组都要用新位姿重估一次，重估不过闸的组退回 `stale`。
  - 否决的方案「载入即用 gauge 变换」：起会话那一刻还没有跨会话约束，gauge 无从估起；且 gauge 不过闸时没有退路。本方案在约束到来之前同样可用（只是不导航到旧节点）。
  - 地点记忆不自洽修图：跨会话落位后，旧节点与本会话新建的同地点节点**不合并**（§10.3）。

**整合检查清单**：
- world_key 从 `WorldModel.identity()["world_key"]` 取（用户手动设置后存在 `<state_dir>/world_identity.json`，启动时恢复），不从 OSC 推断。
- 在 service.py 把建图跑起来之前（M0 在 `_start_nav_online` 之前），调 `cognition.py` 加载该 world_key 的最新 graph。
- nav_prior 旧代码保留（gauge 门+overlay），和策略的地点记忆图不互相干涉（overlay 改 rasterize 前的点云，地点记忆改寻路目标，两条路径正交）。

## 5. 时间基准与训练数据（P0-3、P0-4）

### 5.1 现状核查

| 时钟 | 域 | 备注 |
|---|---|---|
| OSC 速度接收 | `time.perf_counter` | osc.py:324，与 DR 积分同域 |
| HMD/tracker 位姿采样 | 录制时 = 双目 while 循环的 tick = `perf_counter()`；.jsonl 的 `t` 字段 | `nav_online.py`L1576 `rec.line("hmd", {"t": tick, "R": ...})` |
| 关键帧时刻 `t_s` | `tick`，与 HMD 同域 | `nav_online.py`L1623 `t_s = tick` |
| osc.jsonl | ZOH（`{"t": perf_counter(), "vx": ..., "vz": ...}`），触发时刻而非积分端点 | `nav_online.py`L1578 |
| 视频 | 无；M0 再加（计划见 §5.2） | 现在录制没有视频流 |

- **关键帧与 osc 时刻对齐**：`t_s` 直接取 tick，`osc.jsonl` 的每行 `t` 也是 `perf_counter()` 触发时刻。二者同域，配对只需按时间戳插值。
- **没有额外的、真机 VRChat 产生的动作时间线**：现录制没有地面真值标签（真实身体朝向、当前指令）。
- `offline_timebase_harness.py` 在 `research/tools/` 下存在（合成数据固定时间基准的 9 条 contract）。

### 5.2 数据流设计（M0）

| 流 | 频率 / 触发 | 时刻 | 格式 | 用途 |
|---|---|---|---|---|
| policy_obs | 5 Hz（位姿线程，每次采集后判断） | ZOH `"t": tick` | .jsonl `{"t": float, "img_L": b64(jpeg), "img_R": b64(jpeg)}` | 训练/离线标注（头 A 只用 `img_L`，`img_R` 存档） |
| kf_img | 不等距（关键帧，中位 0.2–0.5 s 按场次） | `t_s` | `kf_img/<id>_L.png` + poses.npz 的行 | 云端语义/Re-ID 调用 + MegaLoc 检索 |
| osc.jsonl | 触发即记（VRChat 发 `/avatar/parameters/*`） | ZOH `perf_counter()` | `{"t": float, "vx": float, "vz": float}` | DR 积分 + 训练标签 ZOH 扩展 |
| hmd.jsonl | 20 Hz target（记录期间 14.0 Hz） | 双目 while 的 tick | `{"t": float, "R": [[...]]}` | 地面真值朝向 |
| devices.jsonl | 关键帧触发 | `t_s` | `{"t": float, "devs": [{serial, class, R, p}]}` | 手柄位姿（交互射线） |
| policy_labels | 离线生成，每个 policy_obs 一条 | policy_obs `t` | `{"t": float, "act": [[vx, vz] × 12], "mask": [0/1 × 12], "cls": "move/trans/idle", "wp": [dx, dz]}` | 头 A 的 12 步监督（M0–M1 来自 osc，M2 起来自 Udon，见下） |

- **policy_obs.jsonl 新增于 M0**：在 `nav_online.py` **位姿线程**（`_perception_loop`）加 5 Hz 门（上次发送 + 0.2 s），每次采集 `read_stereo_rgb()` 后判断是否到拍，到了就把**本拍的双目 RGB** 按 224² / 448² 缩放后压 JPEG q90，base64 进 `.line("policy_obs", {"t": tick, "img_L": ..., "img_R": ...})`。这样 policy_obs 的 `t` 与 hmd/devices 的 `t` 都是同一个 `tick`（`perf_counter()`），同域对齐。
  - **存储分辨率（G6 定）**：224² JPEG q90 约 28 KB/对 ≈ 140 KB/s（5 Hz），448² JPEG q90 约 85 KB/对 ≈ 425 KB/s。224² 训练够用，只在**升档后总占用仍 ≤ 85%** 时升档 448²（条件见 §1 预算表；按现基线不开）。
  - 只算 policy_obs 一项：5 Hz × 224² JPEG q90 ≈ 140 KB/s（8.4 MB/min）× 8.5 min ≈ 71 MB/场。80 min（M0）≈ 0.67 GB，35 h ≈ 140 KB/s × 126,000 s ≈ **17.6 GB**。录制总量（含 kf_img、kf/ 点云）见 §9.2，只按 10 h 预算（G6）。
- **关键帧彩色**：现录制 `kf_img/` 是关键帧决策时镜像 API `read_stereo_rgb()` 拍的，`record_images=True` 时写。时刻 = `t_s`（关键帧决策那个 tick）。与 policy_obs 的关系是"同一副本的不同抽样"（5 Hz 等距 vs 关键帧不等距）。
- **头 A 的 12 步标签（M0–M1，osc 速度）**：头 A 训 12 步（§1），**12 步都要有监督**，不能只用 `t_obs` 那一拍的瞬时速度（那样后 11 步学不到东西）。
  - **定义**：第 k 步（k = 0…11）的标签 = osc.jsonl 在 `t_obs + k×0.1 + osc_lag_s` 处的 ZOH 值（≤ 该时刻的最后一条 `vx, vz`）。`osc_lag_s = 0.18`（nav_online.py L84，OSC 速度比画面晚到约 0.18 s）——不加这项，标签整体晚 2 步。存成 `policy_labels.jsonl`（格式见上表），离线生成，与 policy_obs 按 `t` 一一对应。单位是 VelocityX/Z 原值（m/s，角色局部系，k = 1.0，memory `osc-speed-scale`），执行时怎么变成摇杆量见 §9.1。
  - **会话末尾**：`t_obs + k×0.1 + 0.18` 超过录制末尾（hmd.jsonl 最后一条 `t`）的步，mask 置 0，loss 不计。**不能补零**：补零会被学成「临近结束就刹车」。开头 `t_obs + 0.18` 早于第一条 osc 的样本整条丢弃。
  - **ZOH 跨长空窗是对的**：VelocityX/Z 只在变化时发，站着不动几十秒没有一条，ZOH 出的 0 就是真值。只有「最后一条速度 > 0.1，之后 > 2.5 s 没有新消息」才可疑（丢了停下的 0 包，或者 VRChat 卡住），窗口碰到这种空窗时整条 mask。2.5 s 与 DR 的 `osc_zoh_max_s` 同口径（nav_online.py L92–96：实测要外推非零速度的空档全部 ≤ 2.03 s）。
  - **挂机段整段剔除（新增）**：osc 静默 > 30 s **且** HMD 旋转矩阵 > 10 s 逐字节不变的窗口不进训练集（人离开了，零速不是「策略决定停」）。真站着的人 HMD 会微动，不受影响。045615 的「静止窗口」几乎全是这一类：OSC 只覆盖前约 1037 s，之后 3,818 s HMD 旋转完全不变（memory `drift-anomaly-045615-verdict`），所以只剔这一段，不整场丢——剔完后它仍是移动窗口最多的一场。
  - **站立与过渡样本要保留，但按类别重采样**（下一条）。原写的「站立样本不去重也不丢弃」在挂机段剔除后仍成立，但不能按自然比例喂 MSE 回归头。
  - **实测（K = 12，剔挂机后，`.tmp/_label_mix.py`，15 场）**：navmesh_recordings/ 在本轮评审期间有一场（202630）的目录消失，实测按 **15 场**（202630 不在；033237_f_mv 副本不计），5 Hz 网格 35,978 个窗口：
    - 挂机段 18,968 个（**52.7%**），几乎全在 045615（18,940 个）；
    - 剩 16,409 个（54.7 min）：静止（12 步全 ≤ 0.1 m/s）32.0%、**移动（第 0 步 > 0.1）47.4%，7,781 个 ≈ 25.9 min**、过渡 20.5%；
    - 移动窗口来源前三：045615 22.7%、033237 14.1%、184318 12.8%，没有一场独大；
    - mask 口径（剔挂机后 17,004 个，含开头未对上 osc 的）：开头 506、末尾 89、跨可疑空窗 194（1.1%，集中在 210847 和 044153），12 步 mask 全 1 占 **95.4%**；
    - 旧口径（K = 7、不剔挂机）的 37.5k 窗口里 67% 是全零，评审 round 10 指出，已作废。
  - **⚠️ 这批窗口大多没有图**（§0）：现有录制没有 5 Hz 图像流，带图的只有关键帧原图 `kf_img/`（5 场 23.5 min）和 `navmesh_memory` 的 320×180 缩略图，配上 osc 标签共 **4,291 个样本，92.2% 是移动**（关键帧按运动触发，比例和 5 Hz 网格正好反过来）。所以 M0 训练集**以新录 policy_obs 为主**（§9.2），旧关键帧样本只作辅助，且必须和新录数据一起按类别重采样，否则会倒向「永远在走」。
  - **新代码**：标签生成器是 M0 新写的离线脚本（`research/tools/` 下），仓库里没有现成实现。`offline_timebase_harness.py` 只验证合成数据的帧号/时间戳，不做 osc ZOH。
- **防「输出 0」塌缩（M0，评审 round 10）**：MSE 回归头在零速占多数时会学成「原地不动」，G1/G3 都测不到。三条一起上：
  - **类别标签**：每个窗口打 `cls`——`move`（第 0 步 > 0.1 m/s）、`trans`（其余有非零步的）、`idle`（12 步全 ≤ 0.1）。按类别重采样到 **move : trans : idle ≈ 2 : 1 : 1**（每 epoch 内），不改 loss 权重。
  - **航点条件**：输入的局部航点（§10.1 的 70 token 文本里）离线由录制轨迹回算：取 `t_obs` 处位置沿真实轨迹前进 `lookahead_m = 0.8 m`（`nav_follow.FollowConfig` 同值）的点，转到机体系 `(dx, dz)` 写进 `wp`；轨迹剩余不足 0.8 m 时取终点。站着的窗口航点距离 ≈ 0，「停」就成了输入可推出的，而不是要模型从图里猜的。位置用离线回环后的 `final_poses`（没有的场次用 DR）。v2 的方案 1「离线重放调 `nav_follow`」改为这一条：v2 说的 `_compute_local_goal` 在 `nav_follow.py` 里不存在，等价逻辑是 `PathFollower._point_at(progress + lookahead_m)`。
  - **门 G7（见下）**：用两条朴素基线卡住塌缩。
- **头 A 的 12 步标签（M2 起，Udon）**：同样按 12 拍窗口取（Udon 是真值，不加 `osc_lag_s`）。**口径写死为「记位姿、离线差分」**，标签不用 `GetVelocity()`（它只用于下面的时间对齐）：
  - **Udon 日志每 0.1 s 一行**：`[NEKO_GT] <t_rt> <x> <z> <yaw_deg>`。`t_rt` = `Time.realtimeSinceStartup`；`x, z` = `VRCPlayerApi.GetPosition()` 的水平分量（世界米）；`yaw_deg` = `GetRotation()` 的 yaw（玩家胶囊体朝向）。
  - **为什么不用 `GetVelocity()`**：它返回**世界系**速度，要转到机体系还得配一个朝向，口径上并不省事；而且它是物理胶囊体速度，在斜坡、台阶、被推挤时与摇杆意图不同。位姿差分只需一个位置、一个朝向，两者同一行、同一时刻采样，不存在速度和朝向错位。
  - **朝向用玩家胶囊体 yaw，不用 HMD**：`GetRotation()` 是 VRChat 移动用的身体朝向；六点追踪下 HMD 可以独立转头，HMD yaw 与移动方向可差几十度。VRChat 的 `/input/Horizontal|Vertical` 正是相对这个朝向解释的，所以标签要转到同一个系。
  - **机体系速度**：第 k 步（k = 0…11）用第 k 行和第 k+1 行差分：`Δp = p[k+1] − p[k]`（世界米，水平），`ψ = yaw[k]`，`v_body = R(−ψ) · Δp / Δt`，其中 `Δt = t_rt[k+1] − t_rt[k]`（实测值，不假设 0.1 s）。轴约定与 osc.jsonl 的 `vx`（右）/ `vz`（前）一致，M2 写生成器时用一段直走 + 一段侧移的录制钉住符号。
  - **窗口尾部要第 13 个位姿**：12 步速度要 13 个位姿（`p[0..12]`），所以从 `t_obs` 起取 13 行。会话末尾不够 13 行的，缺几步就把对应 mask 置 0，同 osc 标签规则，不补零。
  - **单位**：Udon 的位置是世界米，osc.jsonl 的 `vx/vz` 是 VRChat 移动参数（M0 标签的单位，前向尺度 k ≈ 0.95–1.0，memory `flow-fwd-k`）。两者混训前要换算到同一单位：M2 开工先用一段同时有 osc 与 Udon 的录制，拟合 `|v_udon| ≈ c · |v_osc|` 的比例 `c`，记进生成器配置。没拟合出 `c`（残差 > 10%）前，M2 标签不和 M0 标签混用。这个 `c` 与 §9.1 执行侧的「m/s → 摇杆量」标定是同一类问题，但方向相反、分开存：`c` 只管标签，摇杆曲线只管执行。
  - Udon 日志单行约 50 字节，10 Hz 下每小时约 1.8 MB，不影响 output_log 的体积。
  - ⚠️ **时间轴不能直接对齐**：`output_log.txt` 每行的时间戳是**墙钟、精度 1 s**（实测 `2026.10.26 18:07:33 Debug - [Behaviour] ...`），而 policy_obs 的 `t` 是 `perf_counter()`。所以 Udon 那一行**必须自带亚秒时间**（`Time.realtimeSinceStartup`，打进日志正文），再估一个常数偏移对到 `perf_counter` 域：Udon 同时记玩家速度（`VRCPlayerApi.GetVelocity()`，**仅用于对齐**，不做标签），与 osc.jsonl 的速度做互相关取峰值，残差要求 ≤ 0.05 s（未测）。对不上的场次不用。
  - 标签覆盖率（policy_obs 帧里 12 步全部有 Udon 值的比例）≥ 90%（G3）。
- **M0 不用考场**：M0–M1 只用现有世界录制（家和社畜 Club），标签只有 osc 速度。M2 起接 Udon 地面真值。

### 5.3 数据切分（P0-4）

- **按世界分**：family home、Club、考场三个场各占一个桶。Club 是 hold-out（M6 泛化评估用）。
  - ⚠️ **前置：现在还分不了**。现有录制的 `meta.json` 都没有 world 字段，要先人工补一份「场次 → 世界」对照表；Club 的 hold-out 有几场，补完表才知道，可能是零场（零场时 Club hold-out 改到 M1 起补录）。见 §9.2「场次口径」。
- **按会话切**：home 的多次录制分为 train / val，每次录制内的帧不跨组（同一会话的关键帧在空间上强相关，跨组会漏泄）。
- **会话内时间切**：每段录制的末 20% 标为该会话的 val 子集（训练时跳过），用于检测过拟合。全程 val 子集 ≈ 总数的 35%（0.2 × home sessions + Club 整场）。

**门 G3（数据冻结，M0 训练前）**：
- 每段录制的 policy_obs、osc、hmd、devices 四个流的 `t` 跨度一致（相差 ≤0.5 s），行数与时刻单调递增，无 NaN 时刻。
- policy_labels 中 12 步 mask 全为 1 的样本占比 ≥ 95%（剔挂机段之后算）。按 §5.2 定义实测，现有 15 场是 95.4%（剔挂机后 17,004 个，扣掉开头 506、末尾 89、跨可疑空窗 194；202630 录制目录在评审期间消失，不在此实测）。逐场低于 80% 的录制不进训练集：K = 12 下 194119（24.9%，OSC 晚开）、200039（76.3%）、210847（76.5%）三场剔除，合计 706 个窗口，剩 12 场、97.5%。（K = 7 时 210847 是 80.5%、压线保留，块长改 12 步后跨可疑空窗的窗口变多，掉到线下。）
- 新录的 policy_obs 同口径过这条门；另加 **policy_obs 与 osc 的覆盖**：每场 policy_obs 帧里能配上 osc 标签（不在挂机段、开头已有 osc）的 ≥ 90%。
- Udon 真值文件 M2 之前可以没有。M2 起要求考场录制的 12 步全有真值的比例 ≥ 90%，且 Udon↔perf_counter 偏移残差 ≤ 0.05 s。

**门 G7（防塌缩，M0 训练后、上线前，评审 round 10 新增）**：在 hold-out（§5.3 会话内末 20%）上，按 §5.2 的 `cls` 分组评估回归头，和两条朴素基线比。基线数字是现有 15 场 osc 标签在剔挂机后的实测（`.tmp/_g7_base.py`，同一批数据 G3 过滤后 12 场）：
- **两条基线**：「全零」（永远输出 0）和「抄当前速度」（12 步都填 `t_obs` 时刻的 osc 速度，本体状态里本来就有）。移动窗口上每维 MSE（m²/s²）：全零 3.10，抄当前 2.98（12 步平均），只看 2 Hz 实际执行的前 8 步是 **全零 3.31 / 抄当前 2.56**（逐步：全零递减 3.73→2.65，抄当前递增 0.78→4.02；抄当前在前 5 步赢、第 5 步起反超，因为后面的步要预测「会不会停/转」，抄当前做不到）。
- **门槛**：
  1. move 类前 8 步 MSE **< 抄当前基线**（< 2.56 量级，按 hold-out 自身重算基线）。这条直接卡住「输出 0」：全零在这里是 3.31，比抄当前差 29%。
  2. move 类「走起来」召回（第 0 步预测 |v| > 0.1 的比例）≥ 90%；抄当前基线是 88.6%（移动窗口里 88.6% 的 `t_obs` 时刻 osc 本来就 > 0.1，起步靠「抄当前」无法走出这个比例）。
  3. idle 类「停得住」召回（前 8 步预测 |v| 全 < 0.1）≥ 90%；抄当前基线是 94.1%。两条一起防「永远走」和「永远停」。
  4. trans 类（第 0 步停、后面会动）只记不设门：抄当前在这里只有 18.0% 能「起步」，这一类要靠航点条件，M0 看它有没有比 18% 高。
- **在线复核（G1 的 10 min 同跑里顺带记）**：航点距离 > 0.5 m 而执行器发出 0 的控制拍占比，只记不设门；M1 起超过 10% 时回查重采样比例。

## 6. 双运行时接入（P1-5）

| 模块 | 输入输出 | 编译目标 | 接口 | 精度对齐门 |
|---|---|---|---|---|
| 头 A（策略） | 图像 bf16 → 动作 fp32 | TorchScript（JIT trace 不够）+ torch.compile XPU | `policy.forward(images, text)` | cos ≥ 0.999（hidden） |
| LAS2 | 灰度 uint8 → 视差 int16 | LAS2 自带二进制（nvrtc 的 PTX）| SHM + sem | disp cos ≥ 0.999，留出在 memory |
| MegaLoc（M3） | 描述子 + query → 候选 | OpenVINO fp16 | 待 M3 查文档 | cos ≥ 0.90，IoU ≥ 0.80 |

### 6.1 策略（头 A）的导出与校验

- **torch.compile 路径不导出**：XPU 的 AOT inductor 未测，默认 JIT 模式下 Triton 缓存与 `compile` 内部已优化的 kernels 直接绑在 Python 进程里。保持 JIT 运行，由 OpenVINO 的 C++ 保底（如果 OpenVINO 性能在 M3 实测下也够，可退到 OpenVINO）。
- **输出校验**：训练完每个 LoRA 后，`torch.export` + `export.save`（保存输入/输出张量和权重 sha1），然后载入并跑同一个固定输入，输出 cos ≥ 0.999。
- **预热**：启动时先喂一批固定张量（单次前向），Triton 编译在此时完成。预热期间控制环接「没有策略」分支。

### 6.2 LAS2

已接（`stereo_backend.py`，`las2_model.py`），timeout 0.5 s / 心跳 60 s，handshake 600 s。fp16 视差误差留出在 memory `las2-stereo-a770.md`。

### 6.3 MegaLoc（M3）

仓库里还没有。计划在 M3 接入，对接 `nav_memory.py` 和 `nav_xsession.py`。OpenVINO 接口待 M3 查 IR 和 API 文档。接入后补精度留出和预热脚本。

**门 G4（双运行时冲突，M0）**：
- 头 A 和 LAS2 同时跑的积压测试（双目频率 ≥7 Hz，LAS2 滞后 p95 ≤ 150 ms，GPU 总占用 < 90%）。
- 编译缓存迁移到 H 盘后，启动时预热脚本在 60 s 内完成，C 盘用量 < 500 MB。

## 7. 进程边界与故障恢复（P1-6）

### 7.1 拓扑

| 进程 | 职责 | 子进程 | 接口 | 故障处理 |
|---|---|---|---|---|
| `backend/process.py` （主） | 建图、慢脑、控制环、状态 | stereo worker（las2 / sgbm 二选一） | `service.py` 的 HTTP API | 唯一入口，它挂就全停 |
| stereo worker | 双目深度 | 无 | SHM + NamedSemaphore，timeout 0.5 s，handshake 600 s | 超时降级为 `StereoBusy`，建图丢帧；心跳断判死，重启子进程 |
| （云端） | 慢脑调用 | 云 | HTTP | 断线时 `cognition.py` 返回空，慢脑调用超时后（10 s），控制环和导航器继续跑：① 关键帧语义不调云端，地点记忆图冻结（不新增节点，已有节点不变）；② 高层目标队列清空，导航器退回几何模式（只用 `nav_follow` 跟随航点，不调慢脑规划）；③ 技能整理和难例规划不触发。云端恢复后下一次关键帧语义调用成功，地点记忆恢复写入 |

- **stereo worker 在专用子进程**：手工 fork、`SharedMemory` + `NamedSemaphore`，超时 0.5 s，心跳 60 s，600 s 启动门槛（LAS2 首次编译）。手工 fork 是因为 Triton/PyTorch XPU 自己管设备初始化，multiprocessing.Pool 会冲突（已在 las2_model.py 里验证）。
- **timeout 口径**：连续 3 帧超时判「子进程死」，此时父进程杀子进程并重启，期间丢的帧数进 telemetry。
- **心跳**：父进程每帧都查子进程是否还在，子进程也每帧查父进程是否还在，一方发现对方死了就自己退出。

### 7.2 全局停车

- **stop_all_axes()**（osc.py L792）：归零所有 OSC 移动轴（包括手动和策略发的），并清除过期标记。
- **控制环每轮头**先调 `stop_all_axes()`（如果策略会发 axes 且当前块不是 locomotion 块），然后执行策略块。这样策略和接管不会争用同一个轴。
- **急停路径**：`scheduler` 进入 `stopped_latched` / `fault_latched` 状态时，阻塞 `turn` 和 `INPUT_COMMANDS`（scheduler.py L142/L150 LATCHED_STATES）。`service.py` 的 `stop_movement`（L2908）同时调 `osc.stop_all_axes()`（L2926）和 `scheduler.submit("turn", {"halt": True})`（L2924）。

### 7.3 换世界时清理

- **现状：只有消费者，没有生产者**。autonomy.py L356–365 只消费 `vision.snapshot()` 的 `events`（type="world_changed" / "world_switch" / "instance_changed"）。events 由慢脑按自由字符串发（vision.py 的 event schema 不限 type），capture 不活跃时清空。world_model.py docstring 第 2 条明说 OSC 拿不到世界标识。全仓没有任何代码发这三种 event，**清理路径现在永不触发**。
- **M0 要接 `output_log.txt` 的世界切换行，或由用户手动切 key，否则清理路径永不触发**：
  - **日志**：`%USERPROFILE%/AppData/LocalLow/VRChat/VRChat/output_log_*.txt`，本机日志里实际有 `[Behaviour] Joining wrld_<id>:<instance>`（拿 world id，和 `manual_id` 同格式）、`[Behaviour] Joining or Creating Room: <name>`、`[Behaviour] Finished entering world.`（进入完成）。M0 新写日志尾随：读到 `Joining wrld_` 记下 id，读到 `Finished entering world.` 时调 `set_world` 并发 `world_changed` event。
  - **手动**：插件 UI 调 POST `/worldmodel/world`（process.py → service.py `world_model_set_world` → world_model.py `set_world`），这是现在唯一能改 world_key 的入口；M0 让它在 key 变化时同时发 `world_changed` event。
- **清理**：autonomy 收到 event 后调 `on_world_changed`（service.py L432 构造时传入）。**现在它只做一件事**：导航器 `reset_direction_memory("world_changed")`。以下三项是 M0 要加的：动作块队列清空；技能卡和慢脑未决调用标作废；nav_memory 换到新 world_key 目录。
- **nav_memory 换目录要重开会话**：`NavMemoryStore.begin(identity)` 只在 navmesh 启动时调一次（nav_online.py L1092 → `_begin_memory` L1339），中途改 world_key 不会换目录，新世界的关键帧会继续写进旧世界的目录。M0 在 `on_world_changed` 里先 `end` 旧会话，再按新 identity `begin`；在这之前，切世界后必须手动 stop/start navmesh。
- **在线地图要清零**：旧世界的栅格和位姿图在新世界里没有意义，留着会让导航往不存在的可走区走。M0 随 nav_memory 重开一起重置在线地图（等价于 stop/start navmesh）。

**门 G5（故障路径，M0）**：
1. 手工 `taskkill /F /PID <stereo_worker>`，父进程检测到心跳断，5 s 内重启子进程，期间建图丢帧，`status()` 里 `timeouts` 计数递增。
2. 主进程自己 raise Exception，整个后端挂掉，前端 UI 显示连接断开，30 s 内不重连自动停止（这是防御性的，不是设计目标）。
3. 手工发 `/avatar/parameters/VelocityX=0.5`，再调 `stop_all_axes()`，OSC 轴归零，avatar 停住。

## 8. 接管检测与评估协议（P0-9、P2-9）

### 8.1 接管检测（两路 + 考场日志）

| 路径 | 触发 | 实现 | 适用场景 |
|---|---|---|---|
| A. 显式接管按钮 | 插件命令 API：POST `/autonomy/disarm`（process.py L414 → autonomy.py L230 `disarm`），或宿主侧 `vrc_autonomy_disarm`（`__init__.py` L3280） | **入口已有**：standalone UI「解除自主授权」（app.js L470）、navmesh 页「解除武装」（navmesh.js L117）。M0 只加控制环一侧：每拍读 `autonomy.snapshot()["armed"]`，为 False 即清空块队列、停发。disarm 后**不自动恢复**，要用户重新 arm | 用户主动接管。手柄按键映射（SteamVR Input 绑定一个 action 调同一 API）M2 可选，未测 |
| B. 期望/回读速度失配 | 期望速度（已发 `/input` 轴 × 移动速度）与回读 `/avatar/parameters/VelocityX/Z` 夹角 > 60° 或长度比 > 2.0 | M1 实现：回读按 OSC 滞后对齐后再比（见 memory `online-loc-vs-rtabmap-accuracy`），10 Hz 连续 5 拍判接管；触发时控制环本地暂停 5 s（不调 disarm，免得每次误判都要用户重新 arm），暂停中失配仍在则顺延 | 玩家推摇杆、被撞、卡墙 |
| C. 考场日志（Udon） | `output_log.txt` 写 `[TAKEOVER] <timestamp>` | 离线解析，标注数据集用 | 评估专用 |

- **路径 A（显式按钮）在 M0 接入**，B 在 M1 接入。C 只在 M3 离线标注考场数据时用，运行时不读日志。
- **为什么没有「OSC source 标签」路径**：策略发的 `/input/Horizontal` 和玩家按手柄，VRChat 回读 `/avatar/parameters/VelocityX/Z` 完全同形，在现有信号里根本不可分（osc.py L894 的 `"source": name` 在 `_vertical_history` 分支，`_motion_history` L904 没有 source）。`set_axes()` 发的是 `/input/*`，不写回 `_motion_history`。唯一能用的是路径 B（期望速度 vs 实际速度）或显式接管按钮。
- **停车逻辑**：A 是 disarm，策略停到用户重新 arm 为止。B 是控制环本地暂停 5 s，块队列清空，期间不发新块（给玩家接管窗口），之后自动恢复。**授权默认永不过期**：`autonomy.session_ttl_minutes` 默认是 0（config.py L206，plugin.toml L176），service.py L430 原样 × 60 传进去，autonomy.py L201 把 `≤ 0` 解释成 `inf`。而且加载器把这个键钳到 [0, 0]（config.py L733–739 `_bounded_int(..., minimum=0, maximum=0)`），plugin.toml 写什么都只会是 0。所以 arm 之后**唯一的退出是显式 disarm**（路径 A 按钮、`/autonomy/disarm`、`vrc_autonomy_disarm`）。autonomy.py 里「非零时夹到 60–3600 s」的分支，在当前加载器下走不到。M0 不依赖任何会话超时；需要会话上限时，要先改加载器的钳位，不是改 plugin.toml。

### 8.2 评估协议（M6）

**固定起点回归测试（自动）**：
- 每个 milestone 冻结一组关键场景（M2 起有考场）：固定出生点 + 固定目标 + 固定随机种子（如果策略有采样）。
- 每个场景跑 N=5 次，记录是否到达、路程、时长、碰撞次数、接管次数。
- **回归门**：M3 起，上一个 milestone 的及格场景在新 milestone 下成功率 ≥ 80%。

**人类基线（M6 前做一次）**：
- 由用户本人（操控水平已知）在同一批场景下手动完成，记录同样指标。
- 用途：给「策略成功率 ×%」一个参照系，判断差距是策略弱还是任务本身难。
- **不作为通过门**，只作为报告材料。

**影子模式一致性（M6）**：
- 在用户手动控制时，策略后台推理但不发送 OSC，只记日志。
- 事后按 10 s 窗口统计「策略和玩家的移动方向余弦 ≥ 0.8」的窗口占比 ≥ 70%。
- **这条才是 M6 的核心门**：策略跟人的意图一致，才敢放到真实公开实例里跑影子。

## 9. 快策略训练数据与动作块（P2-8、§5 补充）

### 9.1 动作块格式与执行

```json
{
  "type": "locomotion",
  "steps": [
    {"t": 0.0, "vx": 0.0, "vz": 1.1},
    {"t": 0.1, "vx": 0.0, "vz": 1.1},
    ...
  ],
  "metadata": {"source": "policy", "obs_t": 123.45}
}
```

- **块长 12 步 × 0.1 s = 1.2 s**（v2 是 10 步 × 1 s，按 ≥3 Hz 设计；本文 2 Hz 下理由见 §1）。控制环 10 Hz，按 `(now − obs_t)/0.1 s` 取步索引；**新块到达即替换旧块**；索引越过块尾时沿用最后一步；`now − obs_t > 0.8 s`（2 Hz）或 `> 1.3 s`（1 Hz 降级）时丢弃并停车。只有这一条换块规则。
- **metadata.source**：只在**我们自己的块队列**里有意义——策略块标 `"policy"`，回放/示教块标 `"manual"`，用于离线分析。它不是接管检测的依据：VRChat 回读的 `/avatar/parameters/VelocityX/Z` 不带来源，策略发的和玩家按的同形（§8.1）。
- **策略输出 → 动作块转换**（控制环）：策略输出是 `{vx, vz}` 的 **12 步序列**（v2 是 10 步 (v_fwd, ω)、横移固定为 0；本文改为双轴速度、不输出 ω，转向见下）。控制环逐步插 0.1 s 的时刻，打包成 JSON，再逐拍取步，按下面规则执行。

**单位与轴映射**（评审 round 10）：
- **块里的单位**：每步 `vx`（右）/ `vz`（前）是机体系 **m/s**，和标签同单位。osc 标签（M0–M1）是 VelocityX/Z 原值（前向尺度 k ≈ 0.95–1.0，memory `flow-fwd-k`）；Udon 标签（M2 起）按 §5.2 的比例 `c` 换到同一单位。上面示例的 `vz: 1.1` 就是 1.1 m/s，不是摇杆量。
- **m/s → 摇杆量**：`/input/Vertical|Horizontal` 收的是 [−1, +1] 的归一化轴。前向用 navigator.py L120–129 的实测 5 点（y = 0.13 / 0.15 / 0.20 / 0.28 / 0.30 → 0.1333 / 0.2222 / 0.4444 / 0.8 / 0.8889 m/s），它们**正好落在 v = 4.0·(y − 0.10) / 0.9 上（残差 0）**，死区 0.10。反过来：
  - 前后：`y = sgn(vz) · (0.10 + 0.9·|vz| / 4.0)`；横移：`x = sgn(vx) · (0.10 + 0.9·|vx| / 3.0)`；|v| ≤ 0.1 m/s 的轴发 0。
  - 各轴独立钳到 [−1, +1] 再发 `scheduler.submit("input_axes", {..., "duration_ms": 400})`，10 Hz 每拍重发。**必须先钳位**：scheduler.py L762 对 |x| 或 |y| > 1 抛 `CommandRejected`，异常在 tick 循环里没被单条命令兜住，scheduler 进 `fault_latched`（实测 `y = 1.1` 复现）；service.py L135 的 `_osc_axis_value` 和 osc.py L677 同样拒绝。所以不钳位时 1.1 不会被「饱和成满速」，而是整个身体输出锁死。
  - 未测的三项，M0 第一天各录一段钉住：后退和横移的曲线与死区（上式按前向同形外推）、y > 0.30 段（外推到 y = 1 是 4.0 m/s）、斜推时 VRChat 是否对 (x, y) 做模长归一化。
  - 标定按世界存（同 v2「摇杆速度是世界的 Run Speed，按世界标定」）。
- **速度平台**（`.tmp/_speed_plateau.py`，15 场剔挂机）：前进和后退的众数都在 4.0–4.1 m/s、p99 4.00，横移众数 3.0–3.1、p99 3.00；单轴超出平台的移动步只占 0.38%，斜向步合速度 p95 3.49、最大 3.92。所以满速推杆就是 4.0 / 3.0 m/s，标签不用削顶，也不用 `/input/Run`（osc.py L47 有映射、从未发送）。navigator.py L111 注释的「角色满速 2.6667 m/s」和它自己的 5 个实测点对不上（按那条线 y = 0.70 才 2.6667），本文不用这个数。
- **横移与后退不能丢**：移动步里 |vx|/|v| > 0.3 占 26.1%、> 0.7（基本纯横移）占 18.4%，后退（vz < −0.1）占 13.8%。v2 的 (v_fwd, ω) 会把这四分之一的标签投影掉。
- **转向不在动作块里**（M0–M1）：头 A 不输出 ω。控制环沿用 nav_follow `PathFollower` 的朝向律（对准 0.8 m 前视点，`max_turn_rate` 90°/s，偏差 > 50° 原地转），经 `_navigator_send_turn(delta_deg)`（service.py L2690）走 scheduler `turn`，和现有导航同一条路；地图外没有路径时，前视点就是慢脑下发的航点。
  - **不学 ω 的原因**：录制里没有可用的转向标签。4 场 params.jsonl 有 `AngularY`，取值到 1024、和 HMD 偏航角速度的相关 |r| ≤ 0.07，不是角速度；HMD 偏航在六点追踪下含转头（§5.2）。
  - **代价小**：剔挂机后 0.8 s 内 HMD 转过 > 15° 的窗口只占 2.2%，偏航角速度 p90 6.0 °/s（移动窗口 8.4）。M2 有 Udon 胶囊体 yaw 后再评估加 ω 输出。

### 9.2 数据规模与预算（M0–M2）

| 阶段 | 数据源 | 标签来源 | 新增分钟数 | 新增帧数（5 Hz）| 累计帧数 | 用途 |
|---|---|---|---|---|---|---|
| M0 | 新录 policy_obs（现有世界，非考场）| osc.jsonl → 12 步 ZOH 窗口（§5.2 policy_labels） | 80 min | 24k 帧 | 24k | 速度匹配预训练 + 旧关键帧样本 4.3k（§5.2）辅助 |
| M1 | M0 跑起来后边跑边录（现有世界，不是考场） | 同 M0：osc.jsonl → 12 步 ZOH 窗口。没有 Udon（Udon 日志要考场世界，M2 起） | 60 min | 18k 帧 | 42k | 微调 |
| M2 | 考场录制 | Udon 完整标签 | 120 min | 36k 帧 | 78k | 任务对齐 |

- **场次口径（16 vs 15）**：`navmesh_recordings/` 现有 **16 个目录、125.1 min**（202630 消失前是 17 个、130.0 min），其中 `20261010_033237_f_mv` 是 033237 的离线重处理副本（osc.jsonl / hmd.jsonl 逐字节相同，只多了 `*_poses.npz` 等重放产物），**202630 的录制目录在本轮评审期间消失**（navmesh_memory/ 的 thumb 缩略图还在，但 recordings 目录在会话中途不见了；只读脚本从未执行删除/移动，原因不明）。**凡是统计数据量、训练帧、标签覆盖率，一律按 15 场**（剔除 033237_f_mv 副本和 202630）；只有存储占用（§9.2 下文的磁盘实测）按目录算，因为副本真占盘。**哪场在哪个世界没有记录**：meta.json 都没有 world 字段（录制早于世界身份接入）。§5.3 按世界分桶之前，要先人工给 15 场补一份 `场次 → 世界` 对照表（看 kf 缩略图即可），**Club 的 hold-out 有几场，补完表才知道**；如果一场都没有，Club hold-out 改到 M1 起补录。M0 起录制要把 `WorldModel.identity()["world_key"]` 写进 meta.json。
- **旧录制无 5 Hz 图像**：15 场里 osc 标签 5 Hz 网格约 36k 个（剔挂机和 G3 过滤后；§5.2 实测），但**现有录制没有 5 Hz 图像流**。带图的只有关键帧原图 `kf_img/`（5 场 23.5 min）和 navmesh_memory 缩略图，配上 osc 标签共 **4,291 个样本，92.2% 是移动**（关键帧按运动触发，比例和 5 Hz 网格的 47% 移动正好反过来）。所以 M0 训练集**必须新录 policy_obs**，旧关键帧样本只作辅助（按 move : trans : idle ≈ 2 : 1 : 1 重采样，§5.2）。
- **M0 新录 80 min 是取定值，下面核算它够不够**（旧版从 400k 倒减的推导自相矛盾，已删）：80 min × 5 Hz ≈ 24k 窗口。按现有 15 场剔挂机后的类别比例（move 47.4% / trans 20.5% / idle 32.0%，§5.2）估：
  - 训练侧：移动窗口约 11.4k，是旧录带图移动样本（4,291 × 92.2% ≈ 4.0k）的 2.9 倍，新录成为主体，旧样本只作辅助。
  - G7 侧：hold-out 取会话内末 20%（§5.3）≈ 4.8k 窗口，其中 trans ≈ 1.0k、idle ≈ 1.5k。trans 起步召回的基线只有 18.0%，n ≈ 1,000 时 95% 区间约 ±2.4 个百分点，够分辨「比抄当前好」。
- **total 预算 400k 样本（上限，含 4 epoch）**：v2 §4（`.tmp/_plans.txt` L4537，M0 10k、M1 15k、M2 20k，4 epoch 共 180k）× 2.2 倍裕度 ≈ 400k，在 GameScaling 可接受范围 50k–1000k 内。按本表正向核算：M0–M2 新录 260 min ≈ 78k 帧，加旧关键帧样本 4.3k，共 82.3k；按 §5.3 留 35% 作 hold-out（≈ 28.8k，只评估不训练），训练 ≈ 53.5k × 4 epoch ≈ **214k**，占上限的 54%。全部 82.3k 都进训练也只有 329k，仍在上限内。§11.2 的训练时长按 400k 上限算，是保守值。实际录完按 G3 口径过滤后再定。
- **云端标注成本**：只有关键帧调云端，关键帧间隔 0.2–0.5 s 不等距（按场次不同），260 min 按平均 0.3 s 间隔约 **52,000 次调用**（M0 80 min + M1 60 min + M2 120 min）。每次 ¥0.003 × 52k ≈ ¥156，仍可忽略。
- **录制保留策略**：实测 navmesh_recordings/20261026_184318（8.5 min，1080×606 彩色，`record_images=True`）：kf_img 759 对 1182 MB（每对原图 1.6 MB PNG，编码后 JPEG q90 约 0.23 MB/对），kf/ 点云 733 MB，jsonl 合计 5 MB。现有 16 个目录 125.1 min（15 场独立会话 119.9 min + 033237_f_mv 副本 5.2 min；202630 的 5.0 min 已消失，不计入下表外推），其中带 kf_img 的 5 场 23.5 min（4 场 720×405 灰度、184318 一场 1080×606 彩色；消失的 202630 也是 1080×606 彩色），关键帧速率中位 89–170 kf/min（约 3–6 Hz 不等距）。

**存储方案按存储分辨率分档**（实测 184318 的 40 对抽样编码；policy_obs 按 5 Hz、kf_img 按 90 kf/min 线性外推；GB 按 1000 MB）：

| 方案 | policy_obs 编码 | kf_img 编码 | policy_obs MB/min | kf_img MB/min（90 kf/min） | kf/ 点云 MB/min | jsonl MB/min | 总 MB/min | 8 h | 260 min（M0–M2） |
|---|---|---|---|---|---|---|---|---|---|
| 全分辨率 PNG | 1080×606 PNG | 1080×606 PNG | 432（7.2 MB/s） | 144 | 86 | 0.6 | 663 | 318 GB | 172 GB |
| **默认（M0）** | **224² JPEG q90** | **1080×606 JPEG q90** | **8.4（0.14 MB/s）** | **21** | **86** | **0.6** | **116** | **56 GB** | **30 GB** |
| 升档（升档后总占用 ≤ 85% 才开，§1） | 448² JPEG q90 | 1080×606 JPEG q90 | 25.8（0.43 MB/s） | 21 | 86 | 0.6 | 133 | 64 GB | 35 GB |

- policy_obs 5 Hz（0.2 s），kf_img 按场次 89 kf/min 估（实测 184318 中位），kf/ 点云按 184318 比值（733 MB / 8.5 min = 86 MB/min）外推。
- **必须降档的依据**：全分辨率 PNG 8 h 就是 318 GB，超过 H 盘剩余 121 GB（2026-10-27 实测）的 2.6 倍；policy_obs 一项占总量的 65%。默认档里 policy_obs 只占 7%，大头变成 kf/ 点云（74%）。
- **default 方案总预算**：M0–M2 260 min ≈ 30 GB。H 盘剩 121 GB，按 **10 h（约 70 GB）** 预留；超出按「最久未用」删旧录制。
- **配额管理（M0）**：`navmesh_recordings/` 的配额管理要新写（按目录 mtime + 字节配额），不要复用 `NavMemoryStore.prune()`（prune 只管 `navmesh_memory`，见 service.py L453/L472）。额外清理：`.smear_cache.npz`（实测 184318 一场 375,187,113 字节 = 358 MiB）是临时缓存，超 14 天删。

**门 G6（存储方案，M0 录制前）**：
- policy_obs 存储分辨率按 §5.2：默认 224² JPEG q90（约 28 KB/对），升档后总占用仍 ≤ 85% 时（§1）才升档 448² JPEG q90（约 85 KB/对）。
- kf_img 存 1080×606 JPEG q90（约 0.23 MB/对），保持质量优先。
- H 盘配额设为 **10 h 录制**（约 70 GB），超出后按「最久未用」删旧录制 + 清理 `.smear_cache.npz`（按文件 mtime 超 14 天删除）。
- M0 第一周录满 80 min 后，实测总存储量约 9–10 GB（按上表），不需扩容；M2 前视实际用量追加。

## 10. 慢脑与地点记忆（P0-2 补充、P2-10）

### 10.1 职责划分

| 模块 | 频率 | 输入 | 输出 | 运行位置 |
|---|---|---|---|---|
| 快策略（头 A） | 2 Hz | 224²/448² 左目两帧 + 航点文本（70 token）| 12 步 × 0.1 s 动作块 | A770 本地 |
| 关键帧语义 | 关键帧触发（0.2–0.5 s 不等距，按场次不同）| 448² 单帧 + 约 300 token 提示 | 场景描述 + 物体框（grounding）| qwen3.6-flash 云端 |
| 技能整理 | 用户触发 / 难例触发（约每 10 min）| 最近技能执行日志 + 失败案例 | 压缩后的技能卡 JSON | qwen3.6-plus 云端 |
| 慢脑决策 | 事件唤醒（新地点 / 卡住 / 用户指令）| 地点记忆图 + 当前观测 + 任务上下文 | 高层目标序列（去哪、做什么）| N.E.K.O AGENT_MODEL 云端 |

- **快慢分工**：快策略处理「已知怎么走」的闭环控制（跟路径、避障、转弯），慢脑处理「去哪、为什么、下一步做什么」的规划。
- **云端往返约 5 s**（memory `llm-latency-and-spatial-memory.md`），因此慢脑下发的是目标级指令（「去厨房」），不是逐帧命令。
- **关键帧调用去重**：关键帧语义按地点记忆图去重——新地点或距上次调用 > 5 m 才调云端，否则复用缓存（15 min TTL）。

### 10.2 关键帧语义缓存（云端调用优化）

- **触发条件**：① 新关键帧 + 地点记忆判「新地点」（最近邻 > 2 m）；② 或距上次语义调用 > 5 m。
- **调用格式**：qwen3.6-flash，448² 单帧 + 提示（约 300 token），要求返回 JSON：
  ```json
  {
    "summary": "室内走廊，左侧是镜子，右侧门框",
    "objects": [
      {"label": "door", "box": [x1, y1, x2, y2], "conf": 0.92},
      {"label": "mirror", "box": [...], "conf": 0.88}
    ]
  }
  ```
- **缓存策略**：按 (world_key, 关键帧 id) 缓存响应，TTL 15 min。下次查最近邻时，若命中同一关键帧 id（视觉 Re-ID 或 MegaLoc 返回），直接取缓存，不重复调云端。
- **物体 pose 计算**：`T_map @ T_body @ T_cam @ depth_point`（depth_point 从框中心像素 + 双目深度采样）。物体 pose 不持久化，只在 15 min 缓存窗口内有效。

### 10.3 地点记忆图（cognition.py）

**格式**（与 v2 一致）：
```json
{
  "nodes": [
    {
      "id": "place_001",
      "pose": {"T_map": [[...]]},
      "kf_id": 42,
      "summary": "厨房入口",
      "visit_count": 3,
      "last_visit_t": 1698765432.1
    }
  ],
  "edges": [
    {"from": "place_001", "to": "place_002", "type": "corridor", "length_m": 3.5}
  ]
}
```

- **坐标系**：地点的 `pose` 是**写它的那个会话**的原始 `T_map` 系（与关键帧的 `T_map` 同系）。节点带 `pose_sid`（写入会话，落盘不改）和 `pose_state`（`live` = 本会话写的或本会话已落位，`stale` = 旧会话的、未对齐；`pose_state` 只在内存里，不落盘）。只有 `live` 节点能做 `goto_place` 目标。快照是累积的，一张图里可以有多个 `pose_sid`，对齐按 `pose_sid` 分组做（§4.2）。
- **回环后不追溯**：loop closure 会改回环前关键帧的 `T_map`，但已记录的地点和物体 pose 不重算，停留在旧坐标系里。误差与回环闭合前的 DR 漂移同阶（约 0.45 m，见 memory `online-navmesh-wiring.md` run6），最大可达米级（045615 的 2.3 m 是极端情况）。**这是故意的**：人的空间记忆本来就不是实时重投的全局坐标，地点之间的连通性才是关键。
- **持久化**（§4.2）：在 cognition.py 加 `save_graph({nodes, edges})` 和 `load_graph()` 每 30 s 或每 10 节点增量存一次，写 `<world_key>/graph_<timestamp>.json`。起会话时载最新快照：**拓扑和标签直接用，`pose` 先标 `stale`**，按 `pose_sid` 分组，每组等引用该 sid 的跨会话约束攒够、`estimate_gauge` 过闸后逐组落位（流程见 §4.2「跨会话载入」）。没有约束或不过闸的组一直 `stale`。
- **跨会话不合并**：旧节点落位后，与本会话新建的同地点节点也不合并（合并要解决同名冲突、属性取舍，M3 不做）。地点记忆图允许同一物理地点有多个节点（不同会话的视角），慢脑选目标时优先 `live` 且最近访问的那个。

### 10.4 输入退化处理（P2-10）

| 退化类型 | 检测 | 策略响应 |
|---|---|---|
| 双目遮挡（手、菜单） | 近场像素（深度 z = fx·b/d < max(0.5, 1.2×z_min) **追踪米**，z_min = fx·b/d_max 是该后端能测的最近深度）占比 > 30% 且持续 > 3 帧 | 当前块剩余步按 0.5× 速度执行，到块过期（距 `obs_t` > 0.8 s）就停车；遮挡期间收到的新块直接丢弃。遮挡标记连续 3 帧消失后，接受下一个新块 |
| 云端断线 | HTTP timeout > 10 s | 见 §7.1 拓扑表「云端」那行故障处理列 |
| LAS2 子进程死 | 连续 3 帧超时 | 降级 SGBM（152 ms），建图继续，重启 LAS2 子进程 |
| VRChat 世界切换 | vision.snapshot() 的 events 里出现 "world_changed"/"world_switch"/"instance_changed"（**现无生产者**，M0 接 output_log.txt 或手动切 key，见 §7.3） | 现在只重置方向记忆；M0 加：清空动作块队列和技能卡，nav_memory 按新 world_key 重开，在线地图重置 |

- **单位：全部是追踪米**。z = fx·b/d 里 b 必须用**实测生效**的 `sensors.baseline_m`（nav_online.py L763，从两眼位姿 `eye_x[1] − eye_x[0]` 算），不能用声明值 `expected_baseline_m`（L82，0.126）。L80–81 写明自编驱动是 0.126、发行版驱动是 0.063，L248 也警告「声明值（配置/驱动）不等于生效值」。所以 z 是追踪米，与 nav_mapping.py 的 `res_m` / `range_m` / `ground_tol_m` 同一口径（FACTS L40）。换成世界米要 × `world_scale`（0.755）：0.5 追踪米 ≈ 0.38 世界米，0.405 追踪米 ≈ 0.31 世界米。两套口径差约 25%，引用时必须点名是哪一套。
- **遮挡判断**：每帧统计近场像素（z < max(0.5, 1.2×z_min) 追踪米）占比，> 30% 且持续 > 3 帧时判「遮挡」。z_min = fx·b/d_max，按实测 `baseline_m` 和当前后端算：
  - SGBM：d_max = `disparity_range_px(fx)` − 1（720 档为 63）。z_min ∝ b：b = 0.126 时 0.405 追踪米，发行版 b = 0.063 时 0.203，b = 0.252 时 0.81（FACTS L260）。阈值 `max(0.5, 1.2×z_min)` 在 b = 0.063 时落到 0.5（下限起作用），b = 0.252 时变成 0.97。如果写死 0.5，b 放大后没有任何有效像素能落进阈值，遮挡永远不会触发。
  - **阈值本身 M0 定**：上面的 0.5 和 1.2× 是占位，M0 录举手、开菜单、低头片段时，**按当场实测的 `baseline_m`** 算近场占比分布再定。驱动换了（基线变了）要重测，不能沿用。
  - LAS2：d_max 是网络固定的 192 px，1080 档约 0.2 追踪米（stereo_backend.py L231–232）。
- **遮挡时的控制环行为（写死）**：当前块剩余的步按 0.5× 速度执行，到块过期（距 `obs_t` > 0.8 s，§9.1）就停车；遮挡期间收到的新块直接丢弃，不执行。遮挡标记连续 3 帧消失后，接受下一个新块。最坏情况是减速滑行 0.8 s 后停住，不会无限期半速前进。这个标记只影响控制环，不删点，不改建图。
- **为什么不用「有效匹配占比」**：LAS2 是 soft-argmax 回归（`las2_model.py` L80–82），每个像素都输出一个视差，没有无效标记；手挡在镜头前时得到的是贴近 d_max 的大视差，不是无效值。SGBM 除以 16 后无效值是 −1，0 是有效的无穷远视差。两个后端在「无效」上口径不同，近场深度占比对两者都成立。建图侧的点筛选沿用 `stereo_points` 的 `d > 1.0`（nav_mapping.py L409），与本判据无关。
- **0.5 追踪米阈值待 M0 定**：memory `near-stop-progressive-decel.md` 记了 SGBM nd=64 时饱和点比 0.405 追踪米近。M0 录一段举手、开菜单、低头的片段，按真实占比分布定阈值。
- **VR 菜单污染**（memory `vr-hud-hand-menus.md`）：16 场录制零静止像素，屏幕 UI 检测不做。跟手菜单是真遮挡，由上述遮挡检测处理。M0 起插件自开菜单时，按内部状态标记「菜单开」，期间强制遮挡标记。

## 10.5 自主探索（P0-2 补充、M2/M6）

**方案**：复用现有 frontier 导航（nav_mapping.py L1503 `frontiers()`）+ 地点记忆图，给探索目标加评分与记忆化，M2 原型、M6 闭环。

### 10.5.1 frontier 复用

- **backend 已有实现**（nav_mapping.py L1503–1560）：从观测 FREE 紧挨 UNKNOWN 的边界，目标格取可走中心区里离边界 ≤ reach_m（默认 0.6 m）、与起点连通的格。按 1.5 m 方块切段，每段出一个目标，按路径距离排序，返回前 limit 个（默认 8）。
- **NavSession 已有 explore 模式**（nav_mapping.py L1706–1710、L1901–1928）：地图更新时调 `frontiers(ng, here)`，无 frontier 时 `mode="done"`；到达时标 `frontier_reached`，下一次更新选新的。`keep_m`（默认 1.0 m）内还有 frontier 就不换，免得来回摆。

### 10.5.2 评分（M2）

现有按路径距离排序（L1547），M2 加语义评分：
1. **未访问 > 卡住次数 <3 > 距离近 > 边界长**：frontier 挂在地点记忆图节点上（见 §10.3），记录 `visit_count`、`stuck_count`（卡住 10 s 换目标时递增）；评分 = `−10 × (visit_count > 0) − 5 × (stuck_count >= 3) − path_dist_m + 0.5 × min(size_m, 4.0)`。
   - **`size_m` 是边界长度（米）**，不是面积：`Frontier.size_m` = 该段边界格数 × 格宽（nav_mapping.py L1474、L1544）。段按 `chunk_m` 1.5 m 方块切（L1504），单段的 `size_m` 实际在 `min_size_m` 0.5 m 到约 3 m 之间（斜穿方块时略超 1.5 m）。
   - **权重从 0.01 改为 0.5**：0.01 × 3 m 只有 0.03，比 `path_dist_m` 的格宽量化（0.1 m）还小，「边界长」这条等于不存在。0.5 时一段 3 m 长的边界比 0.5 m 的多 1.25 分，相当于愿意多走 1.25 m 去看更大的未知区，与「距离近 > 面积大」的优先级一致（距离差几米仍是距离说了算）。`min(·, 4.0)` 是封顶，防止长走廊一侧的长边界压过一切。0.5 和 4.0 是 M2 的起点值，按覆盖增量（§10.5.5）调。
2. **语义新奇度**（可选）：MegaLoc 描述子方差（M3 有了再加），与现有地点的描述子 cosine distance 加权。

### 10.5.3 记忆化（M3）

- **frontier 挂在地点记忆图节点上**：地点节点新增字段 `{"frontiers": [{"goal_xy_m": [...], "visit_count": int, "stuck_count": int, "last_visit_t": float}]}`。跨会话从 graph 快照恢复（§4.2 的 `load_graph()`）。
- **visit_count 更新时机**：NavSession 到达 frontier 时（L1926 `frontier_reached`），调 cognition.py 更新该节点的 `visit_count += 1`、`last_visit_t = now`。
- **stuck_count 更新时机**：卡住超过 10 s 换目标时（L1883 `stuck_for_s`），递增当前目标节点的 `stuck_count`。

### 10.5.4 安全约束（M0–M6 通用）

- **黑名单交互**：explorer.py（13 KB，已有）已实现有界状态机，不调 LLM；慢脑下发 selector + 约束后，本地状态机执行「扫描—短前进—再扫描」；探索模式下只用「扫描」部分（不 approach）。
- **急停优先**：autonomy.py（20 KB，已有）的 `AutonomyRuntime` 管授权与时效检查；scheduler 进入 `stopped_latched` / `fault_latched` 时，阻塞 `INPUT_COMMANDS`（scheduler.py L142/L150），探索暂停。
- **卡住 10 s 换目标**：nav_mapping.py L1560–1935 已实现，卡住超过 2.5 s 原地扫描（L1883 `stuck_for_s`），扫满 6 s × 3 轮仍卡 → 标临时障碍（L1781 `_mark_blocked`，TTL 30 s）→ 重规划；10 s 仍卡 → 换 frontier（由 M2 评分选）。

### 10.5.5 指标与门

- **M2**：自主探索原型能到达未探索区域，覆盖增量 ≥ 20%（相对 M1 的人类引导基线，按 `navmesh/grid` 的 FREE 格数比）。
- **M6**：无人类引导下首访地点 ≥ 3 个/会话（地点记忆图新节点数），卡住率 <30%（卡住累计时长 / 总探索时长）。

## 11. LoRA 训练与硬件预算（P2-8 补充）

### 11.1 LoRA 参数量（已核对）

Qwen3.5 语言模型（853M 总参数）有 12 种线性层：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `qk_proj`, `conv`, `A_proj`, `B_gate_proj`, `block_gate_proj`。

| 配置 | 加在哪些层 | 参数量 | 备注 |
|---|---|---|---|
| r=32 全部线性层 | 12 种全加 | 21.65M | v2 的「25M」高估了 |
| r=32 不含 MLP | 去掉 gate/up/down_proj | 11.03M | MLP 占一半参数，但对 VLA 收益未知 |

- **默认配置（M0）**：r=32，加在全部 12 种线性层，21.65M 可训练参数。
- **如果 A100 显存不够**：改 r=16（10.8M）或去掉 MLP（11.03M）。

### 11.2 训练吞吐与时长（A100 80G 单卡）

**输入形状**（实际，非 v2 的简化估算）：
- 视觉 token：224² 是 49 token，448² 是 196 token。
- 文本 token：航点 + 本体状态约 70 token。
- 总序列长 L = 49 + 70 = 119（224²）或 196 + 70 = 266（448²）。

**吞吐估算**（按 FlashAttention-2 + bf16 + gradient checkpointing）：
- A100 80G 理论 FP16 吞吐 312 TFLOPS。
- Qwen3.5-0.8B（853M 参数）单样本前向约 **2 × params × L = 2 × 853M × 119 ≈ 203 GFLOP**（224²）；训练按前向的 **× 3 = 609 GFLOP/样本**算。LoRA 冻结主干，不算主干权重梯度，实际接近 × 2，× 3 是保守值。
- A100 实际有效吞吐按 **40–50% 利用率**算，约 125–156 TFLOPS。
- 吞吐 = 有效算力 ÷ 每样本 FLOPs = 125–156 TFLOPS ÷ 609 GFLOP ≈ **205–256 样本/s**（上界）。batch_size 只决定每步多少样本，不改变每样本 FLOPs，不进这个除法。
- **取保守 120 样本/s**（利用率约 23%）。0.8B 模型、L=119 的短序列，单步矩阵太小，喂不满 A100；线性注意力层的逐块 scan（`qwen35_compile.py` 的 `fast_chunk`）是 memory-bound。M0 第一次上云跑 100 步实测后替换此数。

**训练时长**（M0–M2 总预算 400k 样本，已含 4 epoch，§9.2）：
- 400k / 120 样本/s ≈ 3,333 s ≈ **0.9 h（56 min）**。M0 单独跑（24k × 4 / 120）≈ 800 s ≈ **13 min**。
- A100 80G 租用成本（按阿里云抢占式实例，华北2，2026-10 行情）：约 ¥15/h，0.9 h 共 **¥13.5**。

### 11.3 A770 vs 云端训练

- **A770 不适合训练**：16 GB 显存只够推理（2.5 GB 权重 + 激活），训练需要优化器状态（AdamW 是参数量的 2 倍）+ 梯度 + batch，至少 40 GB。
- **A100 租用成本**（按阿里云抢占式实例，华北2，2026-10 行情）：A100 80G 约 ¥15/h，0.9 h 共 ¥13.5（见 §11.2）。
- **结论**：训练全部上云（A100 或 H100），本地 A770 只跑推理和录制。

## 12. 里程碑与时间表（P2-8）

| 里程碑 | 核心交付 | 门槛 | 时长估算 | 依赖 |
|---|---|---|---|---|
| **M0** | 头 A 2 Hz 跑起来，速度匹配；接管按钮接进控制环；世界切换生产者（output_log 尾随 + 手动切 key 发 event，§7.3） | G0 许可证+隐私 + G1 GPU 预算 + G2 pitch 不破坏建图 + G3 数据冻结 + G4 双运行时 + G5 故障路径 + G6 存储方案 + G7 防塌缩 | 2 周 | 新录 80 min（旧录 15 场 4.3k 关键帧样本仅作辅助） |
| **M1** | 边跑边录，微调 | 新录制按 M0 同口径（osc 12 步 ZOH 标签）过 G3，不需要 Udon；接管检测 B 路（期望/回读失配）上线（A 路按钮 M0 已接） | 1 周 | M0 + 60 min 新录制 |
| **M2** | 考场任务对齐 + 头 A 换流匹配头 + 自主探索原型 | 换头后按 G1 复测（p95 ≤ 200 ms 且总占用 < 90%）；Udon 完整标签覆盖率 ≥ 90%，固定起点回归测试（N=5/场，首批 ≥3 场及格）；frontier 导航能到达未探索区域（覆盖增量 ≥ 20%） | 2 周 | M1 + 考场世界 + 120 min 考场录制 |
| **M3** | 关键帧语义 + MegaLoc + 地点记忆 | 开放词汇 mAP@0.5 ≥ 0.60（qwen3.6-flash grounding），MegaLoc top-5 recall ≥ 0.70（在本项目 15 场独立录制（排除 033237_f_mv 副本、202630 已消失，§9.2）的跨会话查询测试集上，ground truth 由人工标注重访帧对定义；用现有录制的关键帧互查，人工标注约 100 对真重访帧，测 top-5 recall 作为门槛。如果不过门，退回 ORB 词袋，nav_xsession.py 已有，recall 会低但能跑），地点记忆图存取正常 | 2 周 | M2 + 云端 API 额度 |
| **M4** | 技能整理 + 难例规划 | 技能卡压缩率 ≤ 0.4（压缩后 token 数 / 原始日志；数值越小压缩越多），难例重跑成功率 ≥ 0.5 | 1 周 | M3 |
| **M5** | 慢脑事件唤醒 + 高层目标 | 事件响应延迟 p95 ≤ 8 s，目标分解可执行率 ≥ 0.8（由人工判） | 2 周 | M4 |
| **M6** | 影子模式一致性 + 自主探索闭环 | 影子模式 10 s 窗口方向余弦 ≥ 0.8 的占比 ≥ 70%，回归测试成功率 ≥ 0.8；自主探索在无人类引导下完成首访地点 ≥ 3 个（单次会话） | 1 周 | M5 + 人类基线录制 |

**总时长**：11 周（约 2.75 个月），按工作日 5 天/周、每天 6 h 有效工时算。

**关键路径**：
1. M0 的 G1（GPU 预算实测）和 G2（pitch 不破坏建图）是最大不确定性。如果不过门，降级顺序见 §1。
2. M3 的 MegaLoc 接入是第二大风险（OpenVINO IR 转换 + 精度对齐未测）。备选是纯几何的视觉词袋（ORB 描述子），recall 会低但能跑。
3. 考场世界的 Udon 脚本（M2 依赖）需要用户或合作者提供，不在本工程范围内。

**并行窗口**：
- M0–M1 期间可以并行做 M3 的 MegaLoc 离线接入（用现有录制的关键帧测 recall）。
- M3–M4 期间可以并行录人类基线（M6 依赖）。

## 13. 未尽事项与 v2 的衔接

两个头的分工沿用 v2，**互不为备选**：
- **头 A = 移动策略**（v2 §2「头 A：移动策略」，`.tmp/_plans.txt` L4411–4427）。v2 定义的就是 GameScaling-0.8B 主干 + 流匹配动作头，不逐 token 解码。本文只改两处：
  - **M0–M1 用直接回归头热身**：同一主干，输出头先用 MSE 回归 12 步 (vx, vz)（§5 的速度匹配监督），先把数据管线、2 Hz 时钟和 G1 预算跑通。回归头会把多峰动作（岔路口左/右）平均掉，只用作热身，不作交付。
  - **M2 换流匹配头**：随考场 Udon 真值标签一起上，输出 12 步动作块（§9.1，v2 是 10 步），flow matching loss + 少步 ODE 采样。换头后重测 G1 的 p95 延迟。
  - 头 A 0-shot 泛化不够时的退路是 §1 的降级顺序（1 Hz + 几何控制器 / 小动作头 / 加 N 卡），不是头 B。
- **头 B = ARDY（表达动作生成器）**，不是策略，**不进移动策略链**（v2 §4，_plans.txt L4439–4450）。v1 是离线手势库：用 motion.py 的 12 个手势（`GESTURE_DURATIONS`）作提示词离线生成变体，重定向到上半身三点（HMD + 双手），存成 `.nya` 片段（nya.py `ClipLibrary`），走现有 `play_clip` 播放。没有官方训练代码，不微调。
- **技能卡格式**（_plans.txt L3940–3960 教学闭环文档）：contract（前置条件 + 成功判据 + 超时）+ 参数 schema + 示例。M0–M6 用原子技能的技能卡只记在 cognition 的地点记忆图里（见 §10.3），不写单独文件。

**头 B（ARDY）范围**：
- 离线手势库保留：现有的 12 个程序化手势（motion.py）和 `.nya` 片段播放（nya.py + `play_clip`）都已在用，ARDY 只是往片段库里加离线生成的变体。
- v2 §4 的 **v2 档在线 H8 自回归生成**（实测 80 帧 3.2 s，约 1.25 倍实时，再占 4.7 GiB 显存）不进 M0–M6：M0 2 Hz 只留约 13% GPU 余量，放不下。

**动作块衔接（RTC）**：v2 round-2 补充的「五、动作块衔接 — RTC inpainting」（_plans.txt L4582–4589）指 Real-Time Chunking（arXiv 2506.07339）：推理下一块时冻结已执行的步，只 inpaint 后面的步，消掉换块时的速度跳变。它依赖流匹配头，M0–M1 的回归头用不上；M0 靠「块长 12 步 = 1.2 s 覆盖推理周期 + 延迟 + 新块到即替换 + 执行器限加速度」（§1、§9.1）。M2 换流匹配头后，如果换块跳变仍超执行器限幅，再按 RTC 加 `cond_mask`。

**本文新增的约束**：
- §1：GPU 预算从 3 Hz 降到 2 Hz，显存预算从 1.3 GB 提到 2.0–2.5 GB。
- §2：云端计费按 token 不按张，关键帧调用约 ¥0.003/次。
- §3：加 pitch 和 hand_ray_aim 命令，门 G2 确保 pitch 不破坏建图。
- §4：会话内回环后地点 pose 不追溯（约 0.45 m 误差）；跨会话按 `pose_sid` 分组估 gauge 逐组落位，没对齐的组 `stale`、不做导航目标。
- §5：新增 policy_obs.jsonl 流（5 Hz），关键帧 `kf_img/` 默认不开（`record_images=True` 才有）。
- §8：接管检测两路 + 考场日志（显式接管按钮 M0 / 期望-回读速度失配 M1 / Udon 日志离线），OSC 回读分不出玩家与策略，不作信号。评估协议以影子模式一致性为核心。
- §10：关键帧语义去重，物体 pose 不持久化（15 min 缓存 TTL）。

**未来扩展方向**（不在 M0–M6 范围内）：
- 社交反射与人物追踪（需 Re-ID 改闭集 + 名牌，见 memory `research-2026-09-28`）。
- 多模态输入（语音、手势识别）。

---

**方案 v3 结束**。M0 第一天先过 G0（许可证门），然后并行跑 G1（GPU 预算）和 G2（pitch 测试）。

---

## 附录 A. 评审更正记录（round 4–10）

> 原单独的更正 memo 已并入本附录，原文移到 `Docs/archive/VRChat具身agent工程方案v3-更正memo（round4）.md` 只作存档，不再维护。正文已是各轮改完后的现行写法，本附录只记「改了什么、为什么」。被后一轮推翻的条目注明推翻者，**一律以正文为准**。

### round 10

| # | 问题 | 现行写法 | 见 |
|---|---|---|---|
| R10-1 | 训练数据口径虚高约 3 倍（最重要）：37.5k 窗口里 67.2% 全零，单场贡献 63.2%；MSE 回归头会倾向输出 0，G1/G3 测不到 | 剔挂机段（OSC 静默超 30 s 且 HMD 旋转超 10 s 不变）；剩 16,409 个窗口，移动占 47.4%；**更大问题：现有录制没有 5 Hz 图像**，带图只有 4.3k 关键帧样本且 92% 是移动。M0 必须新录 policy_obs 80 min，旧样本只作辅助，且必须和新录一起按类别重采样 move : trans : idle ≈ 2 : 1 : 1。新增防塌缩门 G7（两条朴素基线+三条召回） | §0、§5.2、G3、G7、§9.2、§11.2、§12 |
| R10-2 | 单目/双目的 token 账不一致，直接影响 G1 生死：§1 的 119 token = 49 视觉 + 70 文本 是单帧，§10.1 写「双目」。按双目算 ViT p95 55 × 2 + LM 134 × 168/119 ≈ 299 ms，2 Hz 占 60% + 基线 48.6% = 108.6%，G1 必挂（p50 也是 96%） | 头 A 只吃左目：当前帧加上一个推理帧（0.4 s 前）沿 `temporal_patch_size = 2` 的时间轴打成一个 patch（v2 写「左眼当前帧加前一帧 0.5 s」，改为 0.4 s 是为了落在 5 Hz 网格上）。token 数和单帧相同，bench 的 `grid_thw=[1, gh, gw]` 就是两帧。policy_obs 存档可保留右目 | §0、§1、§10.1 |
| R10-3 | 1 Hz 降级没写块长要跟着改：7 步 0.7 s 块在 1 Hz 下，下一块要到 ≈1.19 s 才来，第一块 0.8 s 就过期——每个周期都会停车 0.4 s | 训 12 步，1 Hz 执行全部 12 步、丢弃 1.3 s；改动传播到 §0、§1、§5.2、§9.1、§10.1、§11.2、§12、§13、所有 G 口径、所有 M 列表 | §0、§1、§5.2、§9.1、§10.1、§11.2、§12、§13 |
| R10-4 | 动作块单位没定义：§9.1 的 `vz: 1.1` 是 m/s，但控制环最终走 `osc.set_axes` 直接发 `/input/Vertical`，那是归一化轴（[−1, +1]）。中间少一步「m/s ÷ 实测移动速度」的标定，不补会报错 | 标签单位是机体系 m/s。执行器反查 navigator.py 校准曲线：实测 5 点+残差 0 拟合出 **v_fwd ≈ 4.0·(y − 0.1) / 0.9**（y 是摇杆量，外推到 y=1.0 对应 4.0 m/s；横移满速 ~3.0 m/s），所以前向轴 `y = 0.1 + 0.9·vz / 4.0`，横移 `x = 0.1·sgn(vx) + 0.9·|vx| / 3.0`，各轴独立钳位到 [−1, +1]。scheduler.py L762 拒绝 \|x\| > 1 或 \|y\| > 1 并进 `fault_latched`。转向不在动作块里，改为用 nav_follow 的朝向逻辑（对准 0.8 m 前视点）或交互射线的 `hand_ray_aim` + `turn_and_look`。v2 写的「满速 2.6667 m/s」与自身数据对不上、已废弃；osc 标签里前进/后退平台在 4.0 m/s、横移 3.0，超 4.0 的只占 0.2%，不需要 Run 键；横移与后退占比 26% / 14%，v3 的 (vx, vz) 保留了 v2 丢掉的横移自由度。录制里没有可用转向标签（4 场 AngularY 取值 0–1024、相关 \|r\| ≤ 0.07） | §9.1 |
| R10-5 | G1 余量只有 3～4 个百分点：224² 单目 p95 189 ms → 86.4%，门槛 90%。延迟是 GPU 空闲时测的，同跑 VR+LAS2 时大概率会涨，基线 48.6% 虽是同跑实测，两者叠加没测过 | 通过概率按「中等」计，M0 排期按「可能要走降级 ①」留时间；降级 ① 因 12 步块不用重训，代价只是改执行频率和丢弃阈值 | §1 G1 |

**其他与之前不同的修正**（round 10）：
- ⚠️ **202630 的录制目录在本轮评审期间消失**。场次口径从「16 场 124.8 min」改为「15 场 119.9 min」（202630 不在；033237_f_mv 副本不计），数据量从「37k」改为「36k（剔挂机后）」。navmesh_memory 里它的缩略图还在，但 recordings 目录中途不见了，只读脚本从未执行删除/移动，原因不明。§5.2、G3、§9.2 改按 15 场。
- 更正评审 (a)：045615 不用整场丢，挂机尾巴占 18,940 个窗口，剔掉后移动窗口占 47.4%，045615 只占其中 22.7%，没有一场独大。
- 更正评审 (c)：10 步 × 1 s 不够，1 Hz 下一块要到 1.19 s 才来，1.0 s 的块撑不到那时，所以取 12 步。
- 更正评审 (d)：1.1 不会被饱和，`set_axes` 遇到 |v| > 1 直接报错，整条命令失败，scheduler 的 tick 循环内异常兜底（L541）把它 catch 进 `fault_latched`。
- 推翻了之前「50% 超过满速 2.6667 m/s，要削顶或加 Run」：navigator.py 的 5 个实测点正好落在 v = 4.0·(y − 0.1) / 0.9 上，残差为 0；外推到 y=1 是 4.0 m/s。录制里前进和后退速度平台都在 4.0 m/s（众数 4.0–4.1），横移 3.0 m/s（众数 3.0–3.1），超出的只占 0.2–0.4%，不需要 Run 键。注释里的「满速 2.6667」与自身数据对不上，已废弃。
- §9.2 预算算术：旧版「80 min 的依据」从 400k 倒减得 452k（超预算 13%），下一条又得 358k，两条互相矛盾。改为：80 min 是取定值，按类别比例核算训练量与 G7 hold-out 样本数；400k 是上限，正向核算（35% hold-out 不训练）训练 ≈ 214k，占上限 54%。
- 其余传播：§5.2 policy_obs 存储估算 130 min → 80 min（≈ 0.67 GB）；M3 跨会话测试集 16 场 → 15 场。

### round 9

| # | 问题 | 现行写法 | 见 |
|---|---|---|---|
| R9-1 | 快照 sid 与节点 `pose_sid` 打架。快照是累积的，B 场保存的图里有 A 场的节点，按「快照 sid = B」整体估一个 gauge，A 场节点被搬进 B 的系，偏差仍是米级 | 按 `pose_sid` 分组，每组只用 `old_sid` 等于该 sid 的约束估 gauge，逐组过闸、逐组落位；没有约束的组保持 `stale`；`table_base` 逐组查；落盘不写回落位后的坐标；prune 保留仍被引用的会话 | §4.2、§10.3 |
| R9-2 | M1 有没有 Udon，三处冲突（§5.2 说 M2 起，§9.2 / §12 写 M1 就有） | M1 只有 osc 标签。Udon 要考场世界，M2 起 | §5.2、§9.2、§12 |
| R9-3 | G1「p95 ≤ 250 ms」比预算宽：50% + 48.6% = 98.6%，过了 G1 必然不过 G4 | p95 ≤ 200 ms **且** GPU 总占用 < 90%（200 ms 对应 88.6%） | §1 G1、§12 M2 |
| R9-4 | 448² 升档条件「G1 余量 ≥ 15%」指向升档前；448² 自身占 48%，要留 15% 基线得 ≤ 37% | 「升档后总占用仍 ≤ 85% 才开」，按现基线不开 | §1、§5.2、§9.2 表、G6 |
| R9-5 | §5.3 直接写三个世界桶，但 meta.json 没有 world 字段 | §5.3 加前置指针：先补「场次 → 世界」对照表，Club hold-out 可能零场 | §5.3、§9.2 |
| R9-6 | `GetVelocity()` 一处说不用、一处在用 | 标明「标签不用，仅用于时间对齐」 | §5.2 |
| R9-7 | `PriorConfig L58–87` 只到 `max_split_m`，tile 参数在后面 | 改「L58 起」 | §4.1 |

### round 8

| # | 问题 | 现行写法 | 见 |
|---|---|---|---|
| R8-1 | 跨会话地点图没有坐标系对齐，旧节点原样载入就是米级 gauge 差 | 先载拓扑和标签、pose 标 `stale`，重定位命中后 `estimate_gauge` 过闸再落位（分组方式被 R9-1 改） | §4.2 |
| R8-2 | M2 Udon 标签口径没定 | 记位姿、离线差分；12 步要第 13 个位姿；机体系用身体朝向不用 HMD；日志自带亚秒时间戳，偏移残差 ≤ 0.05 s | §5.2、G3 |
| R8-3 | 写了「5 s 后靠 session_ttl 恢复」 | 默认 0 且加载器钳到 [0, 0] ⇒ 永不过期，arm 后唯一退出是显式 disarm | §8.1 |
| R8-4 | 「`fault_latched` 下也允许 pitch」 | 撤回。pitch 与 `turn` 同一道闩锁门 | §3.1 |
| R8-5 | 448² p95 占用 54%，对应 270 ms 无来源 | 62 + 180 = 242 ms ⇒ 48%，加基线 97% | §1 |
| R8-6 | 行号与口径一批 | 行号重核；17 个目录 vs 16 场独立会话（`033237_f_mv` 是副本）；smear 缓存写字节数；`size_m` 是边界长；`z_min` 用实测生效基线 | 各处 |

### round 7

| # | 问题 | 现行写法 | 见 |
|---|---|---|---|
| R7-1 | round 6 的丢弃阈值 0.6 s 小于块的自然寿命（0.5 s 周期 + 0.19 s p95 = 0.69 s），G1 丢弃率会是 100% | 0.8 s（= 0.5 + 0.19 + 余量 0.11） | §0、§1、§9.1、G1 |
| R7-2 | round 6 的 5 步块只预测 0.5 s，尾部 0.19 s 开环滑行 | 12 步 × 0.1 s = 1.2 s：2 Hz 前 8 步覆盖周期 + 延迟 + 迟到余量，1 Hz 用整块不重训 | §1、§9.1 |
| R7-3 | 7 步输出没有标签定义，后 6 步学不到东西 | `policy_labels`：第 k 步 (k=0…11) = osc 在 `t_obs + k×0.1 + 0.18` 处的 ZOH 值；会话末尾 mask 不补零；跨可疑空窗整条 mask；实测 15 场 mask 全 1 占 95.4%（202630 目录在 round 10 消失，按 15 场） | §5.2、G3 |
| R7-4 | G2 数 LAS2「视差 > 0」：LAS2 是 soft-argmax 回归，没有无效标记，恒近 100% | SGBM 有效像素占比 + LAS2 对 SGBM 深度一致性 + 近场占比（只记不设门） | G2 |
| R7-5 | §10.4 近场阈值 0.5 m 没标单位，且基线放大后 z_min > 0.5，永不触发 | `z < max(0.5, 1.2×z_min)` 追踪米；b 用实测生效基线（R8-6） | §10.4 |
| R7-6 | §4.1 nav_prior 引用错：两个口径不同的门合成一句；`overlay_prior` / `_nav_prior_loaded` 不存在 | 残差门与前后半交叉门分开写，指到 `_gate` / `_cross_check` / `write_prior` / `PriorConsumer` | §4.1 |

### round 6

| # | 问题 | 现行写法 | 见 |
|---|---|---|---|
| R6-1 | 接管路径 A「给 `_motion_history` 加 source」三处都错：L894 的 `source` 在 `_vertical_history` 分支且是参数名；L904 没有 source；`set_axes()` 只发不写回。策略和玩家的回读同形 | A = 显式接管按钮（POST `/autonomy/disarm` / `vrc_autonomy_disarm`，UI 入口已有），B = 期望/回读速度失配（M1） | §8.1 |
| R6-2 | 存储表单位混了（7.2 和 0.14 是 MB/s） | 统一 MB/min：全分辨率 663（8 h 318 GB）、默认 116（56 GB）、448² 133（64 GB）；policy_obs 35 h ≈ 17.6 GB | §5.2、§9.2 |
| R6-3 | 换块规则自相矛盾（「1 s 块、超块长才换」与 2 Hz 冲突） | 新块到即替换，只留一条规则（块长/阈值被 R7-1/R7-2/R10-3 改为 12 步 / 0.8 s）；位移/转角丢弃 M0 不用；M0 不需要 RTC | §0、§1、§9.1 |
| R6-4 | 头 A / 头 B 混在一起（「头 B 是备选」） | 头 A = 移动策略（M0–M1 回归热身，M2 换流匹配头）；头 B = ARDY 离线手势库，不进策略链 | §13 |
| R6-5 | 世界切换写成可用，但全仓没有生产者 | M0 接 `output_log.txt` 的世界切换行或用户手动切 key；`on_world_changed` 现在只重置方向记忆；nav_memory 只在 navmesh 启动时绑定世界 | §7.3、§10.4 |

### round 4（原 memo 正文，多数已被后续轮次改写）

| # | 项 | 状态 |
|---|---|---|
| 1 | §3.1 pitch 不调运动学，只改 HMD `rotation`（motion.py `nod` L406–410） | 现行 |
| 2 | §4.2 地点图持久化在 cognition.py（`save_graph` / `load_graph`），关键帧走 `NavMemoryStore.begin/end` + `SessionWriter` | 现行（坐标系部分被 R8-1、R9-1 补） |
| 3 | §5.2 policy_obs 在位姿线程 5 Hz 门写，`t` 与 hmd 同一 tick | 现行（编码改 224² JPEG，R6-2） |
| 4 | §7.1 云端断线：语义冻结、退几何模式、技能整理不触发 | 现行 |
| 5 | §8.1 路径 A 用 OSC source 标签 | **作废**（R6-1） |
| 6 | §9.2 存储 270 MB/min、35 h 560 GB | **作废**（R6-2） |
| 7 | §4.2 回环后地点 pose 不追溯 | 现行 |
| 8 | §11.2 训练吞吐 | **作废**：现行按 2 × params × L 估 FLOPs，取保守 120 样本/s |
| 9 | §9.2 数据规模表拆成分钟数 / 帧数 / 累计帧数 | 现行（数字被 R8-6 改为 16 场、R10-1 改为 15 场 24k 新录 + 4.3k 旧关键帧） |
| 10 | §13 v2 节号与头 A/B、RTC 引用 | 节号现行；头 A/B 与块长**作废**（R6-3、R6-4） |
| 11–16 | GPU 基线来源、person_detect 替代、policy_obs 即视频流、关键帧语义频率、视差无效值、MegaLoc 门槛来源 | 现行；第 15 条「视差 > 0 = 有效」对 LAS2 不成立（R7-4） |
| A1 | world_key 取法、目录名、存储基准路径 | 现行 |
| A2 | 世界切换「接线已就位」 | **作废**（R6-5） |
| A3 | RTC/WebSocket、source、卡住恢复行号 | 删 WebSocket 与卡住恢复行号现行；source 部分**作废**（R6-1） |

**附录结束**。
