# 「地图质量」作为打分点 + 地板/楼梯/坑/水 判别：方案与调研（2026-10）

调研日期：2026-10
适用范围：本插件（OSC 客户端 + 镜像双目 + OpenVR 头姿）的**可通行性判别**与**地图质量评测**

> 📌 **本文与既有文档的关系**
> - [`research/terrain-classification-training-2026.md`](terrain-classification-training-2026.md) —— 2026-09-13 的地形分类方案。
>   本文**不重复**它的 §1～§10，只做三件事：① 用 2024–2026 文献刷新标签与训练部分；
>   ② 补上它完全没有的**打分点设计**；③ 把它接到"长期记忆 + 自己规划路线"这条主线上。
> - [`research/traversability-perception-lit-2026.md`](traversability-perception-lit-2026.md) —— 可通行性感知的完整文献综述（本文的证据底座）。
> - [`research/map-quality-as-reward-lit-2026.md`](map-quality-as-reward-lit-2026.md) —— 地图质量作为奖励/指标的完整文献综述（§1 的证据底座）。
> - [`research/vla-alternatives-and-memory-2026.md`](vla-alternatives-and-memory-2026.md) —— VLA 替代方案 / 边缘算力 / 长期记忆文献（§5 的底座）。
> - [`research/embodied-nav-lit-review-2026.md`](embodied-nav-lit-review-2026.md) —— 导航模型全景。

---

## 0. 结论先行

1. **"打分点 = 地图质量"是一次前提变更，不是新增功能。**
   `Docs/开放世界具身智能体-项目计划.md` v1.2 §1.3 把"**无打分点**"写成了任务性质，
   §19.2 因此只能用替代指标（覆盖率、轨迹簇数、LLM 裁判）凑合。
   一旦地图质量成为真打分点，§11.1 那句"**唯一值得主动学的只有 affordance 校准**"就可以扩容。
   → **先改 §1.3 和 §19.2，再动代码。**

2. **"地图质量"必须选「效用」口径，不能选「保真度」口径。**
   保真度（和真值地图的差异）需要真值地图，本项目**永远不会有**；
   效用（用它决策，实际结果有多好）只需要"结果可测"——而本项目**恰好有**：
   OSC 回传的位置与速度就是结果的度量。**这是本项目相对学术界最占便宜的一点。**

3. **绝对不能自己给自己打分。** 仓库已经踩过一次：
   `Docs/停顿后地图错位-根因诊断（2026-10-01）.md` §505 记录了"自己给自己打分 → 结论反转"。
   → 因此打分函数**必须带一个独立的参照物**（反事实臂 / OSC 真值 / 人工 golden 集），
   且**参照物不得由被评系统产生**。

4. **地板/楼梯/坑不该做成 4 类 RGB 分割。** 几何可判、材质不可判；
   尤其 **VRChat 的水是叠在几何上的透明 shader，深度读到的是池底**——
   这从数学上证明水必须走 RGB。三类几何的判据是"指令位移 vs 实际位移"，
   水和玻璃的判据是材质。→ **两支路 + 代价敏感融合 + 弃权**（原文档结论仍然成立，予以确认）。

5. **标签的最大来源不在数据集，在 VRChat 引擎自己。**
   走上去 → 看 OSC 回报的 Y 与速度。站住 = walkable；掉下去/被弹回 = 坑；
   缓慢下沉或持续减速 = 水。**免费、无限量、域内精确。**
   代价是它是**事后**的（要先走过去才知道）——但这恰恰意味着它只能标注
   "**已经踩过的格子**"，而那正是长期记忆本来就要存的东西。两件事是同一件事。
   📌 **这不是空想**：[TartanDrive Off-Road](https://github.com/apairo-robotics/apairo)
   的逐像素可通行性真值就是这么生成的（"derived from **where the robot actually drove**"）。

6. **🔴 代价必须是"软"的，这是最容易做错、也最致命的一条。**
   硬阻断（坑=障碍、水=障碍）会让**正确地图和错误地图给出同一个结果**（都走不通），
   于是**分数没有梯度，学不动**。软代价下"把深水误判成浅水"会产生
   **稠密且有信息量的惩罚**。⇒ **坑 = `C_max`，水 = `1 + c_w`，两者必须不同。**

7. **VLA 不该上，你的判断是对的，且理由比"太重"更强。**
   `Docs/开放世界具身智能体-项目计划.md` §10.2 已经给了硬理由：π0/GR00T/OpenVLA 的
   动作头输出 7-DoF 关节角或末端位姿，本项目动作空间是 **2 维摇杆 + 头部 yaw 增量**，
   **动作头对不上，预训练权重用不了**——不是重不重的问题，是**接口层就不匹配**。
   2026 年的数据把它坐实：HM3D-v2 榜上**唯一的端到端学习策略（DD-PPO）排最后**（27.9 SR），
   榜首是无监督零样本的 84.2；VILA-8B 在 67 TOPS 上只有 0.83 tok/s。
   **该防的不是"VLA 笨"，是"VLA 没有地图"**——地图、规划器、恢复逻辑你都得自己建。

8. **长期记忆 + 自己规划路线的正确组合不是"一个更大的模型"，是四件套：**

   | 层 | 干什么 | 是不是网络 |
   |---|---|---|
   | 感知 | 像素/高程 → **cost-to-traverse 通道 + 置信通道** | ✅ 唯一要训的，2–6M 参数 |
   | 地图 | 多通道 cost map + 拓扑图，**持久、跨会话** | ❌ 数据结构 |
   | 规划 | frontier 探索 + A*/Dijkstra 在 cost 上跑 | ❌ 经典算法 |
   | 决策 | 地点级符号（去哪、为什么） | ❌ LLM，不进像素回路 |

   **"自己规划路线"是第 3 层，它早就能做，只是没有好地图给它用。**
   所以真正的瓶颈在第 1 层和第 2 层的绑定，不在模型规模。

9. **落地顺序是反直觉的：先建尺子，再练。**
   S0 数据结构 → S1 冻结规则基线（A 臂）→ **S2 打分台** → S3 标签采集器
   → S4/S5 训练 → S6 反事实对照。
   **S2 必须早于 S4**：没有打分台就无法判断学习有没有用，
   而没有 A 臂就没有反事实。详见 §7。

---

## 1. 前提变更：从「无打分点」到「有打分点」

### 1.1 现状与冲突

| 文档 | 原文口径 |
|---|---|
| 计划 v1.2 §1.3 | "**无外部目标** / **无打分点**"列为任务性质 |
| 计划 v1.2 §0 判断 1 | "没有打分点 = 不能训练策略 ⇒ 打分点必须下沉到技能层" |
| 计划 v1.2 §11.1 | "**唯一值得主动学的**是 affordance 校准" |
| 计划 v1.2 §19.2 | "评测指标（**无打分点时用什么替代**）" → 覆盖率、轨迹簇数、动作熵、LLM 裁判 |
| 计划 v1.2 §12.1 | "本项目没有的三样：特权信息、加速、外部 reward" |

你要引入的"地图质量"打分点，**正好落在被删掉的那一格**：
既不是特权信息（不需要仿真真值），也不是外部 reward（不需要人给分），
而是**从系统自身行为里长出来的**。这是它成立的原因，也是它唯一的风险来源。

⚠️ **需要在 `Docs/README.md` 冲突登记表加一行**：C 组新增一条
「计划 v1.2 §1.3『无打分点』 ← 2026-10 以『地图效用』为打分点修正」。

### 1.2 地图质量的三种口径 —— 必须选一种

这是整件事最容易做错的地方。"地图质量"至少有三种互不相同的定义：

| 口径 | 定义 | 需要什么 | 本项目可行性 |
|---|---|---|---|
| **F 保真度**<br>fidelity | 地图与**真值地图**的差异（ATE / IoU / 覆盖率） | 一张真值地图 | ❌ **永远没有**。约束第 1 条禁止伪造 |
| **U 效用**<br>utility | 用这张地图决策，**实际结果**有多好 | 只要"结果"可测 | ✅ **有**。OSC 回传位置/速度 |
| **C 一致性**<br>consistency | 地图内部自洽吗（回环闭合、跨会话重定位残差） | 地图自身 | ✅ 有，但**可被退化解骗** |

**推荐：U 为主分，C 为在线无真值代理，F 只在有局部真值时做诊断。**

理由：
- **F 不可得**，且硬凑会违反 `ROADMAP`「不可违反的约束」第 1 条（不伪造）。
- **C 会奖励退化解**：一个把所有格子都标成 `UNK` 的地图，回环残差极小、规划永远拒绝、
  代价低、从不撞——**一致性满分，效用为零**。所以 C 只能当辅助，不能当主分。
- **U 是唯一能惩罚"好看但没用"的地图**的：地图再漂亮，
  规划出来的路走不通就是 0 分。

> 📌 **F/U 之分不是我的发明，是有出处的**：
> [MapEval](https://github.com/JokerJohn/Cloud_Map_Evaluation)（RA-L 2025，[arXiv:2411.17928](https://arxiv.org/abs/2411.17928)）
> 明确把地图质量拆成 "**Global Geometric Accuracy**"（保真度）与
> "**Local Structural Consistency**"（局部结构一致性），并指出
> **后者才是避障与局部规划真正依赖的东西**，"even when global accuracy may be compromised"。

> ⚠️ 现有系统已经有一个"看起来像分数"的东西，注意别混用：
> `Docs/2.5D建图主验收报告.md` 的 `all_route_costs_uncertain_are_explicit` 显示
> **20/20 条路线边的代价都是"不确定"**。那是**校核项**（诚实的标志），不是分数。
> 它恰好说明：**当前的地图基本没有代价信息**——这正是第 2 层要补的洞。

### 1.2.1 ⚠️ 最重要的一条：要戒掉"探索奖励奖励的是地图**大**，不是地图**好**"

这是这一整块最容易走反的一步，而且**学界主流恰恰走反了**：

[Active Neural SLAM](https://arxiv.org/abs/2004.05155)（ICLR 2019）的 RL 奖励原文是
*"reward **proportional to the increase** in coverage"*，即

```
r_t = λ · (Cov_t − Cov_{t−1})
```

**它对地图对不对毫无意见**——只要变大就有分。
把"覆盖"直接当"质量"用，会系统性地训出**自信的错误地图**。

而且这个奖励有已发表的**作弊路径**：
ANS 的 Mapper 是**把 "explored area" 通道当作输出预测出来的**，
所以"探索过没有"是 agent **自己说了算**。⇒

> 🔒 **本项目的硬规则：任何"覆盖率"必须由外部定义，绝不由 agent 自己的地图读出。**
> ANS 的原文口径是"在 agent 视锥内且距离 < 3.2 m"才算已探索——
> **用视锥定义，不用自报**。本项目对应物是**OSC 走过的轨迹扫过的栅格**。

**⇒ 这正是本文坚持"效用"而不是"覆盖"的根本原因。**
顺带一条已发表的佐证：
[Ye et al., ICCV 2021](https://arxiv.org/html/2104.04112)（SemExp 同组）实测发现
加了探索奖励的 agent *"prefer to wander around goals for some time before stopping"*——
**探索奖励反而劣化了它本该提升的任务指标**。

### 1.3 具体打分函数（可直接实现）

把 U 拆成四个可测项，全部**只依赖 OSC + 系统自身动作**，不依赖真值地图：

```
map_score = w1·reach_rate          # 走通率
          + w2·(1 − detour)        # 绕路惩罚
          + w3·(1 − rescue_rate)   # 人工救回率（最高权重）
          + w4·clarity             # 校准度（见下）
```

| 分项 | 定义 | 怎么测 | 为什么要有 |
|---|---|---|---|
| `reach_rate` | 规划出的路径**首尾相接实际走通**的 episode 比例 | OSC 位置进终点半径 | 效用主干 |
| `detour` | 实际路程 ÷ 规划路程，截断到 [1, 3] | OSC 路程积分 / `PathContract.length_m` | 抓"绕远路"的地图 |
| `rescue_rate` | 需要人工拉回 / 重启的次数 | 手动 arm 门禁本来就记录 | **权重最高**：不可逆失败 |
| `clarity` | **该说 unknown 的地方是否真的说了 unknown** | 注入式测试（§6.2） | 抓过度自信的地图 |

**SPL 式写法**（借 Habitat 的形式，改成"路径可走通"而非"到达目标"）：

```
S = (1/N) · Σ_i  s_i · l_i / max(p_i, l_i)
    s_i ∈ {0,1}  = 第 i 次规划是否首尾走通
    p_i           = 实际走过的路程
    l_i           = 规划路径长度
```

这个式子的的好处：**`s_i = 0` 时该项直接归零**，
所以一个"什么都标 unknown 所以从不规划"的地图书，
因为规划不出 `l_i` 而拿不到分——退化解被结构性地堵死。

若想要**连续**版本（推荐，因为 §2.2.1 说了硬阻断会让分数变成阶跃），
用 Habitat 的 [SoftSPL](https://aihabitat.org/docs/habitat-lab/habitat.config.default_structured_configs.SoftSPLMeasurementConfig.html) 形式：

```
SoftS = max(0, 1 − d_t/d_0) · d_0 / max(d_0, d_traveled)
        ↑ 越接近终点越高   ↑ 越不绕路越高
```

`d_t` = 终点剩余距离，`d_0` = 起终点欧氏距离。**它不要求"完全到达"**，
所以一个"走通 90% 然后卡住"的地图仍能拿到分——
这正是我们想要的分辨力（区分"差"和"更差"）。

**三条配套铁律**（都来自已发表的失败案例，不是设计洁癖）：

1. 🔒 **打分时规划器必须冻结、不参与学习。**
   否则地图和规划器会**串通**：两者一起错、互相认可。
   （这条对应 §1.4 的"参照物不得由被评系统产生"，是同一条原则。）
2. 🔒 **不惩罚 agent 看不见的格子。**
   做法照抄 [Chebrolu/Stachniss, IROS 2022](https://arxiv.org/html/2210.08952)：
   他们的三个 loss **全部**被 `(1 − c^occ)` 掩码，只在可导航格上计算。
   **同样的掩码要加到奖励里**——永远不要为 agent 观测不到的区域扣分。
3. 🔒 **任何信息增益/好奇心奖励，必须有持久地图才允许启用。**
   [Remember to be Curious](https://arxiv.org/html/2605.22814)（2026）实测：
   没有持续更新的世界模型时，*"predictive errors spuriously arise in these
   revisited areas, yielding **false novelty rewards for forgotten states**"*。
   ⇒ **这条反过来为 `ROADMAP` P0「跨会话重定位」提供了奖励设计层面的论证**：
   持久地图不只是产品需求，**它是任何内在动机奖励能安全使用的前提**。

### 1.4 反事实臂：唯一能防止"自己给自己打分"的结构

上面所有分数**仍然是系统自己算的**。要可信，必须有一个**外部于被评对象**的参照。

**做法：同一条 episode 跑两臂，只换地图来源。**

| 臂 | 地图来源 | 说明 |
|---|---|---|
| **A（基线）** | 现有规则分类器 `backend/traversability.py` | 已经跑了几十场，是**真·外部参照** |
| **B（被评）** | 学到的 cost-to-traverse 网络 | 待测 |

同一起点、同一目标、同一控制律，比 `S`。差值就是学习部分的净收益。

> ✅ **A 臂必须先固化并冻结**（golden 回放 + 固定阈值），否则每轮都在动基线，
> 差值不可比——这正是 `Docs/停顿后地图错位-根因诊断（2026-10-01）.md` §505
> 记的那个错误的同构版本（那次是"配置挑中自己拽到一起的帧"）。
>
> ⚠️ 附加一条铁律：**A 臂的结果不能被用来调 B 臂的阈值**。
> 那样又变成自己给自己打分。阈值只能在 golden 人工标注集上定。

### 1.5 这条打分点**不能**拿来干什么

- ❌ 不能当**在线自我改进**的信号（它要人参与定义 episode、需要跑完整程）。
- ❌ 不能进 `body_*` 工具的 `preconditions`（约束第 2 条：视觉/预测结论不能冒充观测）。
- ❌ 不能替代技能层的 `postcondition`。它是**元层**指标，比技能级更粗。
  最细的信号仍然是 `Docs/计划 §11.3` 说的 **affordance 表**：
  "预测成功率 vs 实际成功率的相关性"。地图质量分是它的**上游**，不是替代。

### 1.6 ⚠️ 诚实性声明：「地图更好 ⇒ 导航更好」是**待验证的假设**，不是公理

这一条必须写进文档，因为它决定了这个方案会不会被悄悄架空。

**本次调研没有找到任何一篇论文干净地消融了"更精确的地图 → 更好的导航"。**
最接近的两条间接支持是：
- MapEval 主张**局部结构一致性**才是避障/局部规划依赖的东西；
- [SemExp](https://arxiv.org/html/2007.00643) 论证**显式地图表示在性能和样本效率上都优于隐式表示**。

但这两条都不是"精度提升 → 导航提升"的定量因果。

⇒ **因此 S6 那一步不是走过场，是整个方案里唯一能证伪它的实验。**
如果 `ΔS` 不显著，正确结论是"**这个方向的感知投入不划算**"，
而不是"网络还不够大"。这个失败结论**同样有价值**，应当写进验收。

---

## 2. 地形判别：为什么不做单一 4 类 RGB 分割

原 `terrain-classification-training-2026.md` §1 的结论**仍然成立**，此处只做两点加强。

### 2.1 几何三类 vs 材质一类

| 类 | 几何可判？ | 判据 |
|---|---|---|
| 地板 | ✅ | 法向近水平 + 平面拟合残差小 |
| 楼梯 | ✅ | 1D 高程剖面**近等间距阶跃**（自相关峰 + 符号变化统计）；踏面 0.12–0.35 m / 踢面 0.05–0.25 m 正好落在双目光程精度内 |
| 坑 | ✅ | 相对邻域拟合平面的**下凹**超阈值 + 坑口够宽 |
| **水** | ❌ | **几何上就是水平面，与地板完全不可分** |

**加强点一（坑）：学术界的叫法是"负障碍（negative obstacle）"，且它是自监督里信号最强的一类。**
[ViNL](https://arxiv.org/abs/2210.14791)（ICRA 2023）的方法是：
"指令位移 vs 实际位移"发散的地方标成高代价。
走进坑里时**前向滑移最大、高度增益为零**——这个失配对**负障碍的判别力
高于任何其他地形类型**。所以坑是四类里**最容易自举**的，
甚至可能不需要标签就能学出来。
另见 [PrePARE](https://arxiv.org/abs/2208.00322)（IROS 2022）：
用一个 LSTM **在掉下去之前**预测 failure event（ledge/pit/slip），
即坑是"可预判"而不只是"可事后标注"。

**加强点二（水）：文献里这是一个空地，而且是有价值的空地。**
2024–2026 的可通行性综述里，**没有任何工作让机器人从经验里学出"这是水"**。
原因说得通：水的失败是**渐变**的（变慢、打滑，但不一定掉下去），
二值成败信号对它几乎失声；而且折射让静态几何不可靠
（[MARVIS](https://arxiv.org/abs/2403.09850) 与穿水 SLAM
[Suresh et al.](https://cs.cmu.edu/~kaess/pub/Suresh19ral.pdf) 都证实）。
**这既是坏消息也是好消息**：坏消息是别指望有现成配方；
好消息是这个类在本项目里**代价明确**（掉水里 = 救援），不需要精细建模，
只要**判对**就够——可以按 §3.1 的引擎真值硬标，不需要网络学得很精细。

### 2.2 代价不是分类，是区间

学界主流**不是二值 free/not-free**，而是**四档代价**：
[Shaban et al.](https://proceedings.mlr.press/v164/shaban22a.html)（CoRL 2022）
= free / low-cost / medium-cost / obstacle；
[RELLIS-OCC / 3DTTNet](https://arxiv.org/abs/2412.08195)（2024）
= lethal / medium / low / free（逐体素）。

**映射建议**（`NavGrid` 现在是三态 `FREE/OCC/UNK`，`backend/nav_grid.py:28`）：

| 网络输出类 | 地图代价 | 规划行为 |
|---|---|---|
| 地板 | `free` (1) | 自由通行 |
| 楼梯 | `low` (1 + ε) | 通行，可上可下 |
| 水（浅） | `medium` (1 + c_w) | **通行但惩罚** —— 不要直接设成障碍 |
| 坑 | `lethal` (C_max) | 禁行 |
| 障碍 | `lethal` (C_max) | 禁行 |
| 证据不足 | `unknown` → C_max | **禁行**（但计为"未探索"，可作 frontier） |

### 2.2.1 🔴 软代价，不是硬阻断 —— 这是本文最关键的一条设计决定

代价必须建成**有界但非零的软场**：

```
C(i) = 1                                        # 基础通行
     + Σ_k w_k · 1[class(i)=k] · (1 − conf(i))   # 类别代价，随置信度衰减
     + w_slope · |∇h(i)|                        # 坡度惩罚
C(i) ← min(C(i), C_max)，C_max ≈ 1000
```

**为什么必须软、不能一刀切禁行**（这是决定性理由，不是风格偏好）：

> 用**硬阻断**时，一张**正确**的地图常常是**无信息的**——
> 因为"最短路 = ∞"这件事，对好地图和坏地图**是同一个结果**，
> 于是效用分在正确与错误之间**没有梯度**，学不动。
> 用**软代价**时，"地图声称可以廉价穿过浅水，而实际是深水"会产生一个
> **稠密且有信息量的惩罚**——分数变成地图质量的**连续函数**，而不是阶跃函数。

**对本项目的直接后果**：坑和水**必须**用不同代价，不能都当"障碍"。
坑是 `C_max`（真的过不去），水是 `1 + c_w`（能过，但很贵）。
若把水也设成硬阻断，**"这片水到底多深"这个判断就不再产生任何分数差异**，
而那恰恰是材质支路唯一要学的东西。

> 📌 这与 MapEval 那句"局部结构一致性是 essential **even when global accuracy
> may be compromised**"是同一条原理的两种说法。

⚠️ **水设成 `medium` 还是 `lethal` 是个需要拍板的决策点。**
`medium` 的理由：VRChat 里水通常有池底，站在上面其实能站住；
一刀切禁行会让"从水里穿过去"变成不可能。
但本项目部署在**私人房**、兜底救援成本高。
**建议：先按 `lethal` 上线（安全），把"浅水可涉"留作带 `rescue_rate` 惩罚的实验臂。**

> 📌 附一条**学界还没人认领的机制**，对本项目可能很有用：
> 用**牵引力/功耗**而不是"失败"作为水的信号。指令速度 vs 实际速度的**比值**，
> 配合单位路程的等效功率，能把"湿滑/涉水"和"地毯/软垫"分开——
> 二者的 Δv 相似但**功率特征不同**。零标签成本。
> （VRChat 里拿不到功率，但**可用 OSC 实测速度 / 指令速度的持续时间**近似。）

---

## 3. 标签从哪来（按性价比排序）

### 3.1 ★ 第一优先：VRChat 引擎自带的真值（本项目独有）

**这是本方案里最值钱的一条，其他项目都没有。**

判据：**下发一个前进指令，观测 OSC 回传的实际位姿与速度。**

| 观测 | 标签 |
|---|---|
| 位移 ≈ 指令位移，高度不变，站立 | `walkable` |
| 高度**阶梯上升**且位移未明显衰减 | `stairs`（并顺带量到真实踢面高度） |
| 高度持续下降 / 被弹回 / 速度骤零 | `pit` 或 `hazard` |
| 高度缓降 + 速度持续低于指令 | `water`（疑似） |
| 速度反向 / 卡墙 | `obstacle` |

**它必须有同期物，否则会被当成空想。** 有，而且非常贴：

> 📌 **[TartanDrive Off-Road](https://github.com/apairo-robotics/apairo)（LeRobot，CC-BY-4.0）**
> 的逐像素可通行性真值，生成方式原文就是
> *"derived, self-supervised, from **where the robot actually drove**"*。
> **这就是本文 §3.1 提案的已发表实现**——把"车实际走过哪"当作可通行真值。
> 55 episode / ~124k 帧。
> 另有 [STONE](https://github.com/konyul/STONE)（体素级 free/traversable/potentially/non-traversable，
> 自动标注流水线，7,000 关键帧、4 个场地 279 场景）是同一思路的 3D 版本。

⇒ 本项目的差别只是**载体不同**（OSC 遥测代替轮速计），
但**方法论完全一致，且已有可引用的先例**。这一点很重要：
它把 §3.1 从"我们想到的"变成"有先例的"。

**为什么它特别：**

1. **免费、无限量、并行度受限但够用**（1× 真实客户端，但地形确认是低频事件）。
2. **域内精确**——不经过任何 sim2real 迁移，标注的就是 VRChat 引擎自己的碰撞行为。
3. **它同时就是 §1.3 里的 `reach_rate` / `rescue_rate`**。
   **标签采集和打分采集是同一套录制，不需要两套管线。**
4. 它天然满足 `Docs/受控录制操作手册.md` 那套 arm 门禁。

**它的三个必须承认的缺点：**

1. **事后性**：要先走过去才知道。⇒ 只能标注**已踩过的格子**。
   对"我去过的地方"没问题，对"我要去的地方"无能为力——
   **后者必须靠合成 + 先验补**（§3.2）。这界定了学习的作用域。
2. **危险**：为了拿坑/水的标签要真的往坑/水里走。
   ⇒ **只能在你自己的、私人的、可重置的世界里做**（部署目标本来就是私人房，符合约束第 4 条）。
   **不要在别人的世界里做这个实验。**
3. **样本极度长尾**：地板会占 95%+。⇒ 采样必须**按类配额**，不能按帧。

> ✅ **正反馈**：这个机制同时补上了计划 v1.2 §11.3 说的 affordance 表的标签来源
> ——"**技能成败记录**"在计划 §12.1 里被列为"唯一能自动获得、且可放心试错的监督信号"。
> 地图级标签是它的**空间版本**。

### 3.2 第二优先：合成数据（补"没去过的地方"）

2026-10 实况（已核实源码，见 `traversability-perception-lit-2026.md`）：

- **Isaac Lab 原生就有坑和楼梯**。trimesh 地形含 `pit_terrain`（`pit_depth_range` /
  `double_pit`）、`gap_terrain`（`gap_width_range`）、`pyramid_stairs_terrain`、
  `rails_terrain`、`stepping_stones_terrain`（`holes_depth`）等；
  height_field 侧有 `pyramid_sloped`、`wave`、`discrete_obstacles` 等。
  ⚠️ 地形代码在 **v2.2.0 重构过**：老的 `terrain_generator_cfg.py` 函数动物园已拆成
  `isaaclab/terrains/height_field/*` 与 `isaaclab/terrains/trimesh/*` 两个子包，
  按旧教程写会 import 失败。
- **只有水要自己加**。通用仿真器都不建模气水界面。
  两条路：① 自己写 `SubTerrainBaseCfg` 子类（基类接受一个
  `function: Callable[[float, cfg], tuple[list[Trimesh], np.ndarray]]`）；
  ② 用 `TerrainImporter.import_usd` 导入现成 `.usd`/`.obj`。
  现成可参考的只有 [AquaSim](https://arxiv.org/abs/2403.09850)（MARVIS 附带的水面仿真器）。
- 备选栈：[mjlab](https://github.com/mujocolab/mjlab)（Isaac Lab API + MuJoCo Warp，
  有 11 种 box 地形 + 5 种 heightfield，**但没有 pit**）、
  [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)。

**域随机化重点**（这条比模型选型重要）：VRChat 是风格化 Unity 场景，
公开数据集（RELLIS-3D / ADE20K / SUES）全是真实照片，域差比一般 sim2real 更棘手。
随机化维度：材质、光照（含只有自发光的暗世界）、雾与后处理、色彩映射、曝光、FOV、
**相机高度与俯仰角**（`config.py:87` 的 `height_m` 范围 0.8–2.0 m 必须覆盖）。

**硬负样本是准确率的真正来源**：镜面地板、抛光大理石、玻璃栈道、蓝绿色地毯、
播放水视频的显示屏、**地板上的阴影 vs 深色坑**、俯视时的天花板。

#### 3.2.1 一个好消息：真值可以"解析式"生成，不必用 Replicator

关键技巧：**直接从高程场数组算出逐像素标签**，而不是靠渲染管线的语义传感器：

```
h < -depth_thresh                          → PIT
|∇h| > slope_thresh 且是局部极大            → DROP / CLIFF
沿射线 ∇h 的符号翻转次数 > 1                → STAIRS
在水面多边形内 且 h < water_level           → WATER
局部高差 < 0.05 且上方无邻居                → OBSTACLE
否则                                        → FLAT
```

这套规则**免费、精确、可单元测试**，而且它是 §4 训练阶段伪标签的来源。
建议流程：先在**每一种** Isaac Lab 地形上**目视校验一遍**（约 2 小时，救两周），
再开跑。

#### 3.2.2 ⚠️ 现实检查：这个方向上有三件事比预期差

| 预期 | 实情（2026-10 核实） |
|---|---|
| 有现成的"可通行性/affordance"预训练模型 | ❌ **HuggingFace 上没有一个可用的**。只有零下载量的研究占位仓库。用 DINOv3-S + 自己的头 |
| 城市分割数据集里有水和楼梯 | ❌ Cityscapes / BDD100K / SYNTHIA / Dark Zurich **两类都没有**。ADE20K 两类都有。**Mapillary v1.2 有 Water 但无 Stair**（v2.0 有，未核实） |
| 真实越野数据能提供水的深度标签 | ✅ **RELLIS-3D 是唯一把"浅水坑 vs 深水"分开标的数据集**，另有碎石/泥/障/原木，19 类 + void，6,235 图 / 13,556 次 LiDAR 扫描。**它值得单独下载评估** |
| 合成很便宜 | ⚠️ Isaac Lab 装起来 ~15 GB + RTX + conda。**门槛是高的**。想低成本试水可用 `mujocolab/mjlab`（`uvx --from mjlab demo`），但它**没有 pit 地形** |

> 📌 一条与本项目直接相关的**优先级结论**：
> 既然 §3.1 的引擎真值能免费产出**域内精确**标签，
> 合成数据的价值就**收缩为"补没去过的地方"**——
> 它只需要覆盖**训练时没见过就没法学**的那些构型（罕见楼梯比例、超宽坑、
> 水陆过渡带），**不需要追求规模**。
> 这把合成数据从"主要工作量"降级为"补充手段"，是本次调研对投入结构影响最大的一条。

### 3.3 第三优先：蒸馏（只在校准阶段用）

大模型**离线**造伪标签 → 蒸馏到 2–6M 的小网。可借的范式：
[Velociraptor](https://proceedings.mlr.press/v270/triest25a.html)（CoRL 2024，
用 DINOv2/CLIP 特征替人标）、[RECAST](https://arxiv.org/abs/2609.32595)（VLM 语义重述 +
视觉基础模型空间接地 → 可执行 cost map）、[PRECOG](https://arxiv.org/abs/2501.03134) 类先验。
**大模型永远不上车**——只做教师。

> ⚠️ **不要**用单目 metric depth 模型替代已有双目光程。
> `ROADMAP`「明确不做」第 8 条已定，且 2026 的实测数据支持这条：
> [MoGe-2-L 在 NYUv2 上 δ1=0.967，在 KITTI 上只有 0.415](https://arxiv.org/abs/2507.02546)
> ——"metric"模型是**域脆弱**的。最新可用的 DA3 家族里
> `DA3-SMALL/BASE/LARGE/METRIC/MONO` 是 Apache-2.0（可用），
> 但 `GIANT`/`NESTED` 是 **CC-BY-NC-4.0（商用不可用）**。
> **本项目的结论仍然是：几何用双目光程，不用学习深度。**
>
> 📌 若将来**确实**不得不走单目（双目失效的远场兜底），DA3 家族比 DA-V2 更合适，
> 原因很具体：DA3**直接预测深度而非视差**，
> README 原文说因此在薄结构上"**superior geometric accuracy**"——
> **楼梯踢面正是典型的薄结构**。`DA3METRIC-LARGE` 直接输出米
> （`metric_depth = focal × net_output / 300`）。
> 但**只作为兜底信号，不进入"发布的米值"来源**（ROADMAP #8 不动）。

---

## 4. 网络：两支路 + 代价融合

### 4.1 结构

```
        ┌─ 几何支路 G ─────────────────┐
RGB ────┤   输入：度量高程剖面 L=32     ├──▶ 5 类 + abstain
        │   模型：1D 膨胀卷积 ×3        │    （~50k 参数，CPU 可训）
        ├──────────────────────────────┤
        └─ 材质支路 S ─────────────────┘
             输入：RGB 256×256 + 3~4 帧时序
             模型：MobileNetV3-Large + LR-ASPP（~2M）
             输出：{solid, water, glass, unknown}
                       ↓
              代价敏感 + 可弃权 融合 → cost 通道 + confidence 通道
```

**关键设计决定（与原文档一致，此处强化）：**

- **G 支路才是准确率的主要来源**，S 支路只负责把"水"和"玻璃/镜面"分开。
- **S 支路必须有短时序感受野**（帧差或 2–4 帧 GRU）。
  理由：水在几何上不可分，但**在运动/反射抖动上是最好分的**
  （MARVIS 的 "Motion & Geometry Aware" 就是这个道理）。
  逐帧独立分类器**看不到运动**，而运动是水唯一可靠的线索。
- **融合层必须能弃权**，且 `unknown ≠ walkable`——
  `traversability.py:93-107` 的 `_unknown()` 已经把这套哲学写进代码，延续它。

### 4.2 训练要点

1. **按类配额采样**，不是按帧（地板占 95%+，按帧 = 水和坑学不到）。
2. **代价敏感 loss**：漏判水 ≫ 误判水（掉水里 vs 停下来看一眼）。
3. **硬负样本迭代闭环比任何调参都有效**：
   跑模型 → 收集错误 → 把这些确切配置加进合成生成器 → 重训。第一周就这么干。
4. **时序一致性损失**：VRChat 地板在视野里是静态的，**闪烁纯粹是 bug**。
5. **消融高程通道**：每次都做"有/无高程输入"对照。若纯 RGB 打平，
   说明高程链路没起作用——**先修几何，别调超参**。
6. **保留规则分类器作为 A 臂**，永远不要删（§1.4）。

### 4.3 延迟与量化

`config.py` 现为采集 33 ms（~30 fps）、检测 500 ms（2 Hz）。
**不要追 90 Hz**：危害距离的变化远慢于 2 Hz。
2–6M 参数的分割网在 ORT-CUDA 上 10–20 ms，500 ms 预算里完全不是问题。
参照实测数字：[LiteViLNet](https://arxiv.org/abs/2605.21007)（2026）
在 Jetson Orin NX 上 FP16 达 68.73 FPS / ORFD 上 F=96.74%——
说明这个量级在边缘设备上有大量余量。

⚠️ **量化有一条实测反直觉结论**（NVIDIA 开发者社区 + 多处复现）：
**INT8 对 ViT-hybrid 深度解码器常常比 FP16 还慢**——
per-channel 激活离群值会把量化区间打坏。DA-V2 有同样回退。
**对策是 QAT 或 SmoothQuant，不是裸 TensorRT PTQ。**
⇒ **建议：深度+分割头在 Orin 上出货用 FP16/TensorRT，
INT8 只留给天生友好的骨干（MobileNetV4 / EfficientViT / 蒸馏版 ResNet18）。**
若要量化，**标定集必须刻意包含楼梯、水面和阴影**（各 64 张）。

### 4.4 最相关的前作：ViPlanner

[ViPlanner](https://github.com/leggedrobotics/viplanner)（CVPR 2024, ETH）
是**与本项目输出契约最接近的已发表工作**：
"基于语义图与深度图的鲁棒学习型局部规划器，**完全在仿真中训练**，
可同时用于动态室内与室外"。它有一个 `TRAINING.md` 和完整的代价图构建流水线。

**最值得抄的一条设计**：ViPlanner 的**语义代价图只在训练期用**，
推理时只跑一个稀疏关键点路径头 + 一个**碰撞概率头**（执行前检查 `δ_μ = 0.5`）。
⇒ 即"语义理解用来教，网络内敛成快路径"。

这与本项目的分工完全一致：**语义支路是训练期的老师，不是运行时的依赖。**
这也从另一个角度支持 `计划 §2.1` 的"不把 LLM 放进像素回路"。

---

## 5. 长期记忆与路线规划：为什么不需要 VLA

### 5.1 你的判断成立，且理由比"太重"更强

`计划 §10.2` 已经把"为什么主干不用 VLA"定死了。2026-10 的实测数据把它从"我们的判断"升级成"业界共识"：

| 证据 | 数字 | 出处 |
|---|---|---|
| **端到端学习策略在 ObjectNav 榜上垫底** | DD-PPO（HM3D-v2 上**唯一**的端到端学习策略）**SR 27.9 / SPL 14.2，排名最后**；榜首 ConsistNav **84.2/41.2** 是**无监督零样本**；第 4 名是**免训练的层次化 3D 场景图**（80.1/39.5） | [HM3D v2 榜](https://www.sota2.com/research/sota/object-navigation-on-hm3d-v2) |
| **算力根本不够** | VILA-1.5-**8B 在 67 TOPS 的 Orin Nano Super 上只有 0.83 tok/s**；而 **NaVILA 就是 VILA-8B 的微调** | [Jetson AI Lab](https://www.jetson-ai-lab.com/archive/benchmarks.html) |
| **NaVILA 本身也不是边缘方案** | 8B、**RTX 4090 上约 1 FPS**、预训练用了 **16 个 A100 节点（128 卡）** | [arXiv 2412.04453](https://arxiv.org/html/2412.04453v2) |
| **纯几何可以打赢 LLM 管线** | 2025 受控复现：$0 API 的几何 frontier 探索器 **SR 61.4 / SPL 36.0**，反超带 GPT-4V 的 InstructNav-GT（60.6/33.8），运行时还快 ~2.5× | [arXiv 2507.20021](https://arxiv.org/html/2507.20021v3) |
| **LLM-free 可以在笔记本上跑** | R2F：**不用 LLM、不用训练**，HM3D ObjectNav **SR 78.3 / SPL 29.6**，25 Hz @ 笔记本，$0 API | [arXiv 2603.08475](https://arxiv.org/html/2603.08475v1) |

**最本质的一条**：VLA **没有持久状态**。
OpenVLA / π0 / GR00T / Helix 用的是固定短上下文，NaVILA 自己写明"8 帧就够"——
**一个 3000 步的任务装不进去**。这不是模型不够大，是**结构上做不到**。
而"持久地图"对一个正在建图的 agent 来说**不是功能，就是任务本身**。

> 📌 一句话版本：**该防的不是"VLA 笨"，是"VLA 没有地图"。**
> 地图、规划器、恢复逻辑你无论如何都要自己建，
> 所以问题只剩"3B 模型该放在哪"——答案是放在**代价图之上**、0.2–1 Hz 的**审议层**，
> 而不是放在像素→动作的路上。

> ✅ 顺带一个对本项目有利的观察：
> 你的动作空间是 **2 维摇杆 + 头部 yaw 增量 + 扳机**。
> π0/GR00T/OpenVLA 的动作头输出 7-DoF 关节角或末端位姿。
> **动作头对不上，预训练权重用不了**——这是接口层不匹配，比"太重"更根本。
> 维持 `计划 §2.1` 原判：**主干不上 VLA**。

### 5.2 学术界对"太重"的标准解法就是解耦

[NaVILA](https://arxiv.org/abs/2412.04453)（RSS 2025，
[代码](https://github.com/AnjieCheng/NaVILA)）的结构值得直接抄：

```
VLM（低频，~2 Hz）  →  规划到"语言指令 + 路点"
                       ↓
小策略（高频）      →  在 cost map 上跟随
```

**关键点：VLM 不碰像素。** 它看的是 cost map 的符号化结果。
这与本项目 §0 结论 7 的第 4 层是**同一件事**。

本项目是 NaVILA 的**更极端版本**——连低频 VLM 都可以省掉：
LLM 只需要"去哪、为什么"（地点级），因为"怎么走"已经被
**A*/Dijkstra 在 cost map 上**解决了。这是**本项目相对 NaVILA 的简化优势**。

### 5.3 长期记忆该存什么（这是 P0 的真正内容）

`ROADMAP` P0 定的"跨会话重定位"和第 10 行"地点记忆 ❌"，
在有了 cost 通道之后，**记忆的 schema 才第一次是完整的**。每个拓扑节点要存：

| 字段 | 为什么必须有 |
|---|---|
| 位置（米制） | 基础 |
| **该处的 cost 分布** | 走这里要付多少代价（当前 20/20 条边"代价不确定"） |
| **踩过的证据** | `grounded: false` = 只看过没走过；`true` = 引擎确认过（§3.1） |
| **失败史** | 上次在这里掉了/卡了 → 代价永久上调 |
| 首见 / 末见时间 | TTL 与遗忘 |
| 描述符（BoW/AnyLoc） | 跨会话检索（`nav_bow.py` 已有） |

⚠️ **`grounded` 这个字段是本文对 P0 的实质贡献**：
它把"**看见过**"和"**确认能站**"分开存储。
现在系统只有前者，所以 `all_route_costs_uncertain` 才是 20/20。
加上 `grounded` + §3.1 的采集器，**这条指标会第一次开始下降**，
而它下降的幅度就是地图质量的直接证据——**这是一个不需要任何新定义真值的进步度量**。

### 5.4 学界给的两条设计准则（可直接抄）

**准则一：语言注入"局部、量化的偏好"，而不是"生成自由形式的计划"。**
[arXiv 2507.20021](https://arxiv.org/html/2507.20021v3) 的原话：

> *"language should inject local, metric-aware preferences that a geometry-aware planner can
> arbitrate—rather than attempting free-form plan generation."*
> （语言应当注入**局部的、带度量信息**的偏好，让**几何规划器去仲裁**，
> 而不是试图生成自由形式的计划。）

这**正是** `计划 §2.1` 的分层架构。用 LLM 的正确姿势是：
`vrc_whereami` / `vrc_place_list` 返回**地点 id 与代价**，
让 A*/Dijkstra 仲裁——**而不是让 LLM 吐裸坐标**。
`ROADMAP` P1 那句"**LLM 该给的是地点 id / 语义名，不是裸坐标**"已经有文献背书了。✅

**准则二：几何承担的负载比语言更重。**
[L3MVN](https://arxiv.org/html/2304.05501v2) 的消融：拿掉 **cost-utility 探索模块**
（退化成"找最近 frontier"）造成的性能下降，**比拿掉 LLM 还大**。

⇒ 推论：**先把 cost map 做对，再考虑要不要 LLM。** 本文的 S0–S5 顺序正是这条。

### 5.5 最大的误差源会是分割，不是几何

[PONI](https://arxiv.org/html/2201.10029)（CVPR'22 Oral）在最好的模块化管线里实测：
**光分割失败一项，就让成功率掉 14.9%（Gibson）/ 45.4%（MP3D）**。
PONI 自己写的原因是没有内置的分割失败处理机制。

**这直接说明本文的投入方向是对的**：本项目的瓶颈不是位姿、不是规划器，
是"**这片地方到底是什么**"。而 §2 已经论证了，
地板/楼梯/坑能靠几何廉价解决，**只有水和玻璃必须靠材质**——
所以那 2–6M 参数应该花在材质支路上，不是花在更大的模型上。

⚠️ **两条来自边缘部署的实测地雷**（来自 `vla-alternatives-and-memory-2026.md`）：
1. **量化会让模型直接崩到 0，而不是优雅退化**：
   EfficientViT-SAM 在 Jetson 上 FP16 编码器下 **mIoU ≈ 0**（灾难性），
   而蒸馏过的 NanoSAM 在任何精度下都没崩过。
   → **蒸馏买的是量化鲁棒性**。这条支持本文"用蒸馏而不是直接上大模型"的选择。
2. **TensorRT 化 YOLO-World 会毁掉它的 zero-shot 能力**。
   → **任何组件都要在目标精度下先测，再往上搭。**

> 仓库 `ROADMAP` P5「延迟基准必须在真实负载下测」已经写了同一件事
> （`gpu_bench_result.json` 自称"未开 VRChat"）。这里补一条**更严的版本**：
> **不只是延迟要在真实负载下测，正确性也要。**

### 5.6 长期记忆该用哪种形态：显式 vs 压缩图像

2025–26 的记忆文献收敛到了**两种形态并存**，不是二选一：

| 形态 | 代表 | 强在哪 | 弱在哪 |
|---|---|---|---|
| **显式/符号** | 3D 物体图 + **拓扑图**（ViNT/NoMaD 风格） + 场景图 | 可查询、可规划、**能算代价**、回环检测 | 依赖分割（见 §5.5） |
| **压缩图像** | [AstraNav-Memory](https://arxiv.org/html/2512.21627v1)（CVPR 2026，Qwen2.5-VL-**3B**）| DINOv3+PixelUnshuffle 把 598 token/帧压到 ~30（**16×**），300 帧历史，GOAT-Bench 终身导航 **62.7 SR / 56.9 SPL** | 注意力 **O((token×历史)²)**，边缘设备上必须砍历史；DINOv3 会糊掉地毯等纹理边界 |

**对 5–20 TOPS 的现实建议：显式为主，压缩图像为辅且历史封顶 30–50 帧。**
AstraNav 自己的消融说"历史越长越好"是**假的**（200 帧 @16× 反而不如 100 帧）。

> 📌 **"跨会话"这件事现在有了精确的学术定义，值得抄。**
> [VTM-Nav](https://arxiv.org/html/2607.14514)（2026）对 **cross-episode ObjectNav** 的定义是：
> 每次请求都是独立初始化的单目标 episode，**只有 agent 自己获得的、场景范围的知识跨 episode 留存**；
> 并问"**参数固定、导航组件不重训、无神谕**的 agent 能否复用这些经验"。
> 结果：相对"记忆重置"对照，在 HM3D v0.1/v0.2/MP3D 上 **+4.6 / +2.0 / +0.8 SR**，
> 且 SPL 相当或更高。
>
> **它的三条设计约束（固定参数、无神谕、显式重定位）与本项目完全兼容**，
> 而且 `ROADMAP` P0 正在做的跨会话采纳正是第三条。
> ⇒ 这是本项目 P0 **最直接的一篇对标文献**，建议加进 `Docs/README.md` 的参考资料。
>
> 配套的评测基准是 [GOAT-Bench](https://arxiv.org/abs/2404.06609)（NeurIPS 2024 spotlight）：
> 它专门研究"**memory 在 lifelong 场景下的影响**"。
> **单目标 ObjectNav（每次重置记忆）在结构上就无法隔离持久地图的价值**——
> 所以本项目的任何"地图有用"结论，都必须在**跨会话**口径下报，不能只报单会话。

> ✅ **对本项目的取舍**：本项目**已经有显式拓扑图**（`nav_bow.py` / `nav_memory.py` / `nav_xsession.py`），
> 压缩图像那一路**先不做**。
> 理由：`计划 §10.1` 的模型选型表已经把决策层定为 LLM，
> 而"把 300 帧历史塞进上下文"是**另一种记忆形态**，
> 与显式拓扑图**功能重叠**。等显式地图的 §1.3 打分真的跑起来、且暴露出
> "符号记忆丢掉了某些只有画面才有的信息"时，再评估这一路。
> —— **不要在没有度量的情况下加记忆形态**，这正是 §1.4 反直觉的地方。

---

## 6. 评测与门禁

### 6.1 地形网络

- 逐类 IoU + 混淆矩阵（必备但不够）
- **弃权率**——单独一等指标。弃权太多 = 没用，弃权太少 = 危险。
- **动作级指标**：每扇区"首个危害距离"的估计误差。**这才是导航真正消费的量。**
- 真实误报率：golden set 上按世界/会话分层。
- ❌ **不要用 mIoU 作主指标**——它对"水漏判"和"地板误判"的惩罚一样重，
  而现实里这两个代价差一个数量级。

### 6.2 地图打分台（新增）

| 指标 | 定义 | 频率 |
|---|---|---|
| `S`（§1.3） | SPL 式效用分 | 每 episode |
| `reach_rate` | 规划首尾走通率 | 每 episode |
| `rescue_rate` | 人工救回率 | 每 episode |
| 代价确定率 | `(1 - 成本不确定边占比)` ← **替换现在恒为 0 的那条** | 每会话 |
| 回环闭合残差 | 一致性分（辅助） | 每会话 |
| **校准度** | 注入测试：在已知是水/坑的地方，声明为 `unknown` 的比例 | 回归门 |

**校准度必须用注入测试**，因为它是唯一能抓"过度自信"的指标，
而过度自信正是 `约束第 1 条`（不伪造）在学习系统里的对应风险。

### 6.3 回归门

固定 golden 帧集 + 固定回放，阈值判定接进 `tests/`。
A 臂（规则分类器）**作为基线冻结**，永远不参与再训练。

---

## 7. 落地顺序（挂到 ROADMAP P0–P6）

| 阶段 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| **S0** | 把 `NavGrid` 从三态扩到**多通道 cost + confidence**（`nav_grid.py:28`）。**纯数据结构，不含网络。** | 无 | 地图能装下代价了 |
| **S1** | 规则分类器（现有 `traversability.py`）**输出映射到新 cost 通道**，并**冻结为 A 臂** | S0 | 基线 + golden 回放 |
| **S2** | **地图打分台**：实现 §1.3 的 `S` / `reach_rate` / `rescue_rate`，用 A 臂跑出第一组基线数字 | S1 | **从此有分可打** |
| **S3** | **引擎真值采集器**（§3.1）：走 + 观测 + 标 `grounded` | S2 | affordance 标签流 |
| **S4** | 合成数据生成（Isaac Lab 坑/楼梯 + 自制水）→ 训 G 支路 | S3 | 第一个学习组件 |
| **S5** | 训 S 支路（材质/水） | S4 | 完整 cost 通道 |
| **S6** | A/B 反事实对照，报告 `ΔS` | S5 | **学习是否有用，唯一的判据** |

**S2 优先于 S4**，这是与直觉相反但关键的一点：
**没有打分台就无法判断学习有没有用**，而没有 A 臂就没有反事实。
先建尺子，再练。

---

## 8. 与 ROADMAP「明确不做」的边界

| 条目 | 本方案是否触碰 | 说明 |
|---|---|---|
| #7 生成式/预测式世界模型 | ❌ 不碰 | 本方案全部是判别式 + 符号地图 |
| #8 学习型米制深度作为发布米值来源 | ❌ 不碰 | 几何一律走**双目光程**（`fx·B/disp`） |
| #9 完整 SLAM/VGGT/MASt3R 进在线闭环 | ❌ 不碰 | 感知支路是**局部的、随帧走的**，不参与位姿图 |
| #4 手动 arm 门禁不绕过 | ✅ 遵守 | §3.1 的采集**全部**在用户显式授权的会话内 |
| 约束 1 不伪造 | ⚠️ **新增风险** | 打分点本身引入"自评"风险 → §1.4 的反事实臂是对冲 |
| 约束 2 帧不写进 `world_state` | ✅ 遵守 | 地形判定是**代价先验**，不是观测；`grounded=true` 才来自 OSC 实测 |

**建议新增一条「明确不做」**：
> **不把"地图更密/更完整/更连续"当成成功。**
> 覆盖率是手段不是目标；一个全标 `UNK` 的地图覆盖率满分而效用为零。
> 地图的成功判据只有一条：**用它规划的路，实机上走通了。**

---

## 参考

**本项目内部**
- [`research/terrain-classification-training-2026.md`](terrain-classification-training-2026.md) —— 地形分类原始方案
- [`research/traversability-perception-lit-2026.md`](traversability-perception-lit-2026.md) —— 可通行性感知文献综述（完整证据底座）
- [`Docs/开放世界具身智能体-项目计划.md`](../Docs/开放世界具身智能体-项目计划.md) §1.3 / §10.2 / §11.1 / §11.3 / §19.2
- [`Docs/2.5D建图主验收报告.md`](../Docs/2.5D建图主验收报告.md) —— `all_route_costs_uncertain` 20/20
- [`Docs/停顿后地图错位-根因诊断（2026-10-01）.md`](../Docs/停顿后地图错位-根因诊断（2026-10-01）.md) —— "自己给自己打分"的反例

**外部**
- ViNL（ICRA 2023，零标签自监督可通行性 costmap）— https://arxiv.org/abs/2210.14791 ｜ https://github.com/SimarKareer/ViNL
- PrePARE（IROS 2022，事前预测 failure event）— https://arxiv.org/abs/2208.00322
- Shaban et al.（CoRL 2022，四档代价分类的出处）— https://proceedings.mlr.press/v164/shaban22a.html
- RELLIS-OCC / 3DTTNet（2024，逐体素四档代价 + 物理规则标签生成）— https://arxiv.org/abs/2412.08195
- RoadRunner M&M（RA-L 2025，多尺度可通行性+高程图）— https://arxiv.org/abs/2409.10940
- 风险感知速度分布图 + CVaR（MIT 2022，"可过但很贵"的正确形式化）— https://arxiv.org/abs/2203.13429
- MARVIS / AquaSim（2024，水面分割 + 气水界面仿真）— https://arxiv.org/abs/2403.09850
- 穿水双目 SLAM（RA-L 2019，折射修正）— https://cs.cmu.edu/~kaess/pub/Suresh19ral.pdf
- **必读**：[VI. When Engineering Outruns Intelligence](https://arxiv.org/html/2507.20021v3)（$0 几何基线反超 LLM 管线）、[Remember to be Curious](https://arxiv.org/html/2605.22814)、[VI.b. SemExp](https://arxiv.org/html/2007.00643)、[VI.c. Active Neural SLAM](https://arxiv.org/abs/2004.05155)、[VI.d. 学习代价图](https://arxiv.org/html/2210.08952)
- **TartanDrive Off-Road**（LeRobot，CC-BY-4.0；**"车走过哪 = 可通行"的已发表实现**）— https://github.com/apairo-robotics/apairo
- **STONE**（体素级四档可通行性，自动标注流水线）— https://github.com/konyul/STONE ｜ https://huggingface.co/datasets/Voxel51/STONE
- **RELLIS-3D**（**唯一分开标注 puddle / deep water** 的数据集，19 类 + void，6,235 图 / 13,556 次 LiDAR 扫描）— https://github.com/unmannedlab/RELLIS-3D ｜ https://arxiv.org/abs/2011.12954
- ViPlanner（CVPR 2024，**语义代价图只在训练期用**）— https://github.com/leggedrobotics/viplanner
- DINOv3（2025-08 已发布，6 个 ViT 尺寸 + 4 个 ConvNeXt；`nav_bow` 若换学习型描述子时的候选）— https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- SegFormer-B0 ADE（`nvidia/segformer-b0-finetuned-ade-512-512`，教师/兜底标签器）— https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512
- LiteViLNet（2026，边缘设备可通行性分割的实测延迟基准）— https://arxiv.org/abs/2605.21007
- awesome-traversability-analysis（该领域最佳综述仓库）— https://github.com/Ikhyeon-Cho/awesome-traversability-analysis
- Isaac Lab 地形生成器（pit/gap/stairs 原生，水需自制；v2.2.0 已重构 terrain 子包）— https://github.com/isaac-sim/IsaacLab

**地图质量作为奖励/指标（§1）**
- **When Engineering Outruns Intelligence**（2025，$0 几何基线反超 LLM 管线）— https://arxiv.org/html/2507.20021v3 ｜ https://github.com/matinaghaei/instructnav-scrutinized
- Active Neural SLAM（ICLR 2019，**覆盖增量奖励 + "视锥内且 <3.2 m" 的覆盖定义**）— https://arxiv.org/abs/2004.05155
- SemExp（ICCV 2020，**地图是可微的、损失是 CE+BCE 而非 reward**）— https://arxiv.org/html/2007.00643
- **Remember to be Curious**（2026，**无持久世界模型 → 对遗忘状态发假新颖奖励**）— https://arxiv.org/html/2605.22814
- Ye et al.（ICCV 2021，**探索奖励反而劣化任务指标**）— https://arxiv.org/html/2104.04112
- MapEval（RA-L 2025，**保真度 vs 局部结构一致性**的权威拆分；AC/COM/CD/MME/AWD/SCS）— https://arxiv.org/abs/2411.17928 ｜ https://github.com/JokerJohn/Cloud_Map_Evaluation
- 学习代价图（IROS 2022，**三个 loss 全部被 (1−c^occ) 掩码**；MPC）— https://arxiv.org/html/2210.08952
- 集合分歧当信息增益（L2M/M2IDQN，ICRA 2020）— https://arxiv.org/abs/2106.15648
- Semantic Curiosity（**无标签的视点一致性内在奖励**）— https://arxiv.org/abs/2006.09367
- Neural Topological SLAM（2020，**图记忆**的可引用定义）— https://arxiv.org/abs/2005.12256

**跨会话 / 终身记忆（§5.6）**
- VTM-Nav（2026，**cross-episode 的精确定义 + +4.6/+2.0/+0.8 SR**）— https://arxiv.org/html/2607.14514
- GOAT-Bench（NeurIPS 2024 spotlight，**lifelong 导航的评测基准**）— https://arxiv.org/abs/2404.06609
- OVAL（2026，lifelong ObjectNav + 记忆描述子）— https://arxiv.org/abs/2604.12872

**VLA 与边缘算力（§5）**
- R2F（2026，LLM-free 免训练，25 Hz @笔记本）— https://arxiv.org/html/2603.08475v1
- L3MVN（IROS 2023，LLM 给 cost-utility 打分 + FMM/A*）— https://arxiv.org/html/2304.05501v2 ｜ https://github.com/ybgdgh/L3MVN
- PONI（CVPR 2022 Oral；**分割失败单独导致 −14.9%/−45.4% 成功率**）— https://arxiv.org/html/2201.10029
- NaVILA（RSS 2025，VLM 低频规划 + 小策略高频跟随；**mapless，8 帧记忆**）— https://arxiv.org/abs/2412.04453 ｜ https://github.com/AnjieCheng/NaVILA
- HM3D ObjectNav v1 / v2 排行榜 — https://www.sota2.com/research/sota/object-navigation-on-hm3d-v2
- Jetson AI Lab 边缘 VLM 吞吐实测 — https://www.jetson-ai-lab.com/archive/benchmarks.html
- AstraNav-Memory（CVPR 2026，Qwen2.5-VL-3B + 16× 视觉 token 压缩 + 终身记忆）— https://arxiv.org/html/2512.21627v1
- NoMaD（ICRA'24 最佳论文，30M，可跑在 Jetson Orin）— https://arxiv.org/html/2310.07896
- ViNT（CoRL 2023，31M @4 Hz）— https://github.com/robodhruv/visualnav-transformer
- V-JEPA 2（1B，**16 s/action vs Cosmos 4 min/action**，同一张 4090）— https://arxiv.org/html/2506.09985v1

> ⚠️ **三处需要纠正的常见误引**（本次调研已全文核实）：
> 1. **"Neural Map (Chaplot et al.)"** —— 该文作者是 **Parisotto & Salakhutdinov**
>    （[arXiv:1702.08360](https://arxiv.org/abs/1702.08360)），不是 Chaplot。
> 2. **"SemExp 的 predictive information gain 奖励"** —— **SemExp 没有 IG 奖励**；
>    全文中 `predictive` 出现 **0 次**。它的地图用**监督 CE+BCE** 训练，
>    RL 奖励是"到最近目标物的距离减少"。IG 思想在
>    [SEER](https://arxiv.org/abs/2209.11034) 与 [MapEx](https://arxiv.org/abs/2409.15590)。
> 3. **"PCE / MPCE"** —— 查无此标准指标，**引用前必须先找到一手来源**。
>    该领域公认的探索指标是 ANS 的 **%Cov / Cov**。
>
> ⚠️ **许可证提示**：V-JEPA 2 权重为**非商业研究许可**；
> Depth-Anything-3 的 `GIANT`/`NESTED` 为 **CC-BY-NC-4.0**（商用不可用），
> `SMALL`/`BASE`/`LARGE`/`METRIC`/`MONO` 为 Apache-2.0。
> 本项目为**公开发行包**，选型时必须逐个核对（`ROADMAP` §3 已有许可证×硬件联合筛选的先例）。
