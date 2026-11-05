# -*- coding: utf-8 -*-
"""记忆卡格式化 —— 离线标签生成与在线推理**共用的唯一函数**（方案 §10.6，门 G8）。

为什么必须共用一个函数：训练标签和在线输入各写一套模板，就会静默漂移。
方案把这条和 §5.2 的图像 resize 列为同一类纪律。

格式（定长、ASCII 标签、固定顺序、两位小数）：

    [wp] 0.60 0.00 [topo] 1.20 0.15 2.30 [self] 0.00 1.10

⚠️ **2026-11-06 起卡格式升到 v2：删除 `[mode]`。** 理由是本轮的实测归因：
body+img+causal 的 −0.38 增益里 `mode` 独占 −0.40，属于**特权状态**（导航状态机
直接给出的确定性结果），不是感知能力；把它留在卡里会把「知道导航器想干什么」
误读成「模型看懂了画面」。`mode` 现在**只允许**进入运行时停止决策
（``policy_modes.decide_stop`` 的 ``nav_state``），绝不进入模型输入。
旧 ckpt（含 B3）是用 v1 卡训的，**不能用 v2 卡评测**，已不在基线表内。

三个字段：
  [wp]    前视点，机体系 dx(右)/dz(前)，**世界米**
  [topo]  沿边已走 prog、横向偏差 lateral(右正)、到边终点 remain，**世界米**
  [self]  本体速度 vx/vz，**世界米/秒**

三条硬规则：
  1. **顺序固定**，任何插入都算格式变更，要重过 G8；
  2. **缺失值一律写 0.0**，不写 NaN、不写 missing（与 §1.1 空几何槽、§11.4 geometry dropout 同一契约）；
  3. **全部世界米**。存储层是追踪米，换算只在读侧用 to_world_m() 一处完成。

训练一致性（方案 §10.6 最重要的一条）：M0 录制里没有地点图，[topo] 没有真值，
所以用 DR 轨迹离线合成同样三项（拿录制轨迹本身当「边」），**格式在 M0 冻死**，
M3 换成真实地点图时模型已经认得这些 token 的位置。
"""
from __future__ import annotations

# 导航器状态机的四种模式（不是语言模型给的）。
# ⚠️ 2026-11-06 起**定义搬到了 policy_modes.py**（唯一真源），这里只做 re-export：
# 记忆卡格式没变（0/1/2/3 与名称都一致），但 NavSession.mode 字符串的映射也只
# 在 policy_modes.NAV_STATE_TO_MODE 里有一份，禁止再在 nav_online 里写三元表达式。
from policy_modes import (MODE_EXPLORE, MODE_FOLLOW, MODE_NAMES, MODE_REPLAN,  # noqa: F401
                          MODE_STOP)

# 离线合成 [topo] 时把轨迹切成多长的「边」（世界米）
DEFAULT_EDGE_M = 6.0


def to_world_m(v: float, world_scale: float) -> float:
    """追踪米 -> 世界米。**读侧唯一的换算函数**（方案 §10.6）。

    ``world_scale`` 缺失时必须报错，不许默认 0.755 —— 否则新世界静默错 25%。
    """
    try:
        s = float(world_scale)
    except (TypeError, ValueError):
        s = float("nan")
    if not (0.1 < s < 10.0):
        raise ValueError("world_scale 缺失或离谱（%r）；方案 §10.6 要求报错而不是猜" % (world_scale,))
    return float(v) * s


def _f(v) -> str:
    """两位小数；None/NaN 一律 0.00（缺失值契约）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = 0.0
    if x != x or x in (float("inf"), float("-inf")):   # NaN / inf
        x = 0.0
    return "%.2f" % x


# 记忆卡格式版本。v1 含 [mode]，v2 删掉 [mode]（2026-11-06）。
# 训练侧写入 input_schema["card_version"]，推理侧据此拒绝用 v2 卡评旧 ckpt。
CARD_VERSION = 2
CARD_TAGS = ("[wp]", "[topo]", "[self]")
# 卡里**不允许**出现的字段（泄漏硬门）：mode 是特权导航状态，A*/任务卡尚未就绪。
FORBIDDEN_CARD_TAGS = ("[mode]", "[guide]", "Path ahead", "Task instruction")


def assert_card_clean(text: str) -> str:
    """泄漏硬门：卡里不得出现 ``[mode]`` / A* 路径 / 任务卡。

    2026-11-06 拍板：S2 是**纯感知读出实验**，mode 只用于运行时停止决策；
    A* 与任务段数据当前为 0 个，提前让它们进卡等于伪造条件能力。
    """
    for tag in FORBIDDEN_CARD_TAGS:
        if tag in text:
            raise ValueError("记忆卡含被禁字段 %r（%s）；S2 是纯感知读出实验，"
                             "mode/A*/任务卡不得进入模型输入" % (tag, text))
    return text


def build_policy_text(wp, topo, body) -> str:
    """记忆卡（v2，无 ``[mode]``）。``wp``/``topo``/``body`` 都是**世界米**口径。

    字段顺序与标签冻结；缺失传 None 即可，会自动写 0.00。
    """
    def trio(v):
        if v is None:
            return (0.0, 0.0, 0.0)
        v = list(v) + [0.0] * (3 - len(v))
        return (v[0], v[1], v[2])

    wx, wz = (_f(wp[0]), _f(wp[1])) if wp is not None else ("0.00", "0.00")
    tp, tl, tr = trio(topo)
    bx, bz = (_f(body[0]), _f(body[1])) if body is not None else ("0.00", "0.00")

    text = ("[wp] %s %s [topo] %s %s %s [self] %s %s"
            % (wx, wz, _f(tp), _f(tl), _f(tr), bx, bz))
    return assert_card_clean(text)


def build_guidance_text(guide) -> str:
    """把 A* 引导请求与质量写成定长、可缺省的 ASCII 片段。

    ``requested`` 先区分“本来就不该用 A*”（自由探索/局部提示词）和“请求了但
    路径不可用”；``valid`` 再表示规划先验是否可用。在线、离线训练和损坏增强
    都调用这个函数，避免再造一套模板。数值单位固定为秒/米，缺失值写 0.00。
    """
    if not isinstance(guide, dict):
        guide = {}
    requested = 1 if bool(guide.get("requested", False)) else 0
    valid = 1 if bool(guide.get("valid", False)) else 0
    conf = guide.get("confidence", 0.0)
    age = guide.get("pose_age_s", guide.get("age_s", 0.0))
    sigma = guide.get("sigma_m", 0.0)
    return " [guide] %d %d %s %s %s" % (requested, valid, _f(conf), _f(age), _f(sigma))


def synthesize_topo(xy, s, edge_m: float = DEFAULT_EDGE_M):
    """从一条 DR 轨迹离线合成 [topo] 三项（世界米）。

    方案 §10.6：「M0 用 DR 轨迹离线合成同样三项（拿录制轨迹本身当『边』）」。
    ``xy`` 是 (N,2) 世界米坐标，``s`` 是 (N,) 累计弧长。逐点返回 (prog, lateral, remain)。

    边的定义：沿弧长每 ``edge_m`` 切一段，当前点所在的段就是「当前边」。
      * prog    = 当前点距该边起点的弧长
      * remain  = 该边终点 - 当前点弧长
      * lateral = 当前点到「边起→边终」弦的有符号横向距离（右正）
        我们是沿轨迹走的，所以横向偏差量出的是**这段路弯了多少**，不是「走偏了」。
        这正是离线合成与真实地点图的差别，格式一致、语义可迁移。
    """
    import numpy as np

    p = np.asarray(xy, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n = len(p)
    out = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return out
    total = float(s[-1]) if n else 0.0
    # 每条边的起止弧长
    k = max(1, int(total // edge_m) + 1)
    bounds = [min(total, i * edge_m) for i in range(k + 1)]

    def point_at_s(sq: float):
        if total <= 0:
            return p[0]
        j = int(np.searchsorted(s, sq, side="right") - 1)
        j = max(0, min(j, n - 2))
        ds = s[j + 1] - s[j]
        f = 0.0 if ds <= 1e-9 else (sq - s[j]) / ds
        return p[j] + f * (p[j + 1] - p[j])

    for i in range(k):
        a, b = bounds[i], bounds[i + 1]
        sel = (s >= a) & (s <= b)
        if not sel.any():
            continue
        pa, pb = point_at_s(a), point_at_s(b)
        chord = pb - pa
        clen = float(np.hypot(*chord))
        if clen < 1e-6:
            dirv = np.array([1.0, 0.0])
        else:
            dirv = chord / clen
        right = np.array([dirv[1], -dirv[0]])   # 右向，与 waypoint_body 同约定
        idx = np.nonzero(sel)[0]
        for j in idx:
            prog = s[j] - a
            remain = max(0.0, b - s[j])
            lateral = float(np.dot(p[j] - pa, right))
            out[j] = (prog, lateral, remain)
    return out
