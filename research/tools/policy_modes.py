# -*- coding: utf-8 -*-
"""策略侧「导航模式 / 停止」语义的**唯一真源**（2026-11-06 M0 门控修复）。

为什么必须单独成文件
--------------------
改动前同一个 `mode` 字段在两条链路上有三套互不相同的实现：

1. ``policy_text.MODE_*``：记忆卡 [mode]，``0=follow 1=replan 2=explore 3=stop``；
2. ``nav_online.NavSession.mode``：字符串 ``idle / goto / explore / done``；
3. ``nav_online._policy_mode``：按**跟随器轴** ``3 if (forward<=0 and turn==0) else 0``。

第 3 套复用了第 1 套的数字，但语义完全不同（「跟随器这一拍没输出」≠「导航状态是 stop」）。
同一个数字 ``3`` 在 A 链路表示「停机」、在 B 链路表示「跟随器没动」，任何下游代码
只要两种来源都可能喂进来，就会得到自相矛盾的行为。这里把 1 + 2 收敛成唯一映射，
把第 3 套降级成**运动学提示**（``motion_hint``，字符串，不再叫 mode，也不再是 0/3）。

停止语义（本轮硬口径，见方案修订 A）
------------------------------------
``|wp| ≈ 0`` 只表示「没有路径先验」，**不是**停车条件；``wp_stop`` 参数保留但已
deprecated、不参与任何决策。停止只能来自五处（顺序即归因优先级）：

    estop       人工急停（../../runs/policy/STOP 文件 / 硬件急停）
    stale       过期保护（now - obs_t > stale_s，手里那张图已经太旧）
    safety_gate 安全门（近距障碍 veto / 双目断开，来自感知线程）
    nav_state   显式导航状态（NavSession.mode == done / stop）
    stop_head   模型停止头概率 >= 阈值（当前 checkpoint 没有这个头，接口先留）

五者都是「停」，顺序只决定 ``source`` 记成哪一个，便于事后归因。

⚠️ ``idle`` **不算**停止源（这是刻意的）：独立 VLA 接管时 NavSession 长期停在
``idle``（没有 goto 目标），把 idle 当停车会把策略永久锁死。显式停止只认
``done``（到点/任务结束）与 ``stop``（导航器自己要求停）。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 记忆卡 [mode]
MODE_FOLLOW = 0
MODE_REPLAN = 1
MODE_EXPLORE = 2
MODE_STOP = 3
MODE_NAMES = {MODE_FOLLOW: "follow", MODE_REPLAN: "replan",
              MODE_EXPLORE: "explore", MODE_STOP: "stop"}

# NavSession.mode（backend/nav_mapping.py:1622）→ 记忆卡 [mode]。
# 这是**单向**映射：只有这里一处定义，nav_online.policy_mode() 与
# nav_online.policy_guidance() 都调用它，禁止再各写一套三元表达式。
NAV_STATE_TO_MODE = {
    "goto": MODE_FOLLOW,      # 有固定目标、跟随器在走
    "explore": MODE_EXPLORE,  # 前沿探索
    "replan": MODE_REPLAN,    # 保留：当前 NavSession 还不会产出，占位
    "stop": MODE_STOP,        # 导航器显式要求停
    "done": MODE_STOP,        # 到点 / 任务结束
    "idle": MODE_STOP,        # 无目标（记忆卡如实写 3；但它**不是**硬停止源）
}

# 显式导航状态里**真的会停车**的那些（见文件头对 idle 的说明）。
NAV_STATES_STOP = ("stop", "done")

# ---------------------------------------------------------------- 停止
STOP_SOURCES = ("estop", "stale", "safety_gate", "safety_gate_unlocalized",
                "nav_state", "stop_head")
DEFAULT_STOP_HEAD_THRESH = 0.5

# ---------------------------------------------------------------- 定位安全门
# 2026-11-06：把「未定位 / 定位过期」从 wp 与 nav mode 里彻底拆出来，独立成
# 安全门。旧代码用 ``wp=(0,0)`` 隐式表达「不知道在哪」，于是「自由探索」与
# 「未定位」被判成同一件事；现在两者各自有明确来源：
#   * 未定位   → safety_gate_unlocalized
#   * 没有路径 → 只写日志注记（path_prior=none），**不**停车
#
# 定位状态字符串来自 navmesh 的 ``_est()``（当前只有 ``localized`` / ``unknown``，
# ``unknown`` 的 reason 可能是 ``pose_stale`` 或 ``not_relocalized``）。
# ``unlocalized`` 是保留给 nav_follow 真实重定位状态的名字（现在 nav_follow 用
# ``unknown + not_relocalized`` 表达）。三种一律按「不知道自己在哪」处理。
LOC_STOP_STATES = ("unlocalized", "unknown")

# 位姿过期阈值：与 NavConfig.pose_stale_s 同值（0.5 s）。调用方应显式传自己的
# 配置值；这里的默认只用于测试与没有配置的旧接线。
DEFAULT_POSE_STALE_S = 0.5

# 定位质量指标：**当前不存在**逐帧质量量。navmesh 的 ``sigma_m`` 是
# ``NavConfig.dr_sigma_m`` 的常量（DR 漂移标定值），不是逐帧测量，不能当质量用。
# 因此 ``quality_max`` 保持 None（占位、不参与判定），等将来接入真实指标再开：
#   候选来源 = ``nav_online.NavSession``/``nav_follow`` 的 relocalize 残差
#   （重定位后 ``sigma_m`` 的实测方差），或 mapper 的匹配内点率。
QUALITY_PLACEHOLDER_NOTE = ("当前无逐帧定位质量指标：navmesh.sigma_m 是 NavConfig."
                            "dr_sigma_m 常量（DR 标定），非实测质量；先用 state + "
                            "pose_age 覆盖，quality 阈值留字段占位")


def decide_localization(state, pose_age_s=None, *, state_reason=None,
                        pose_stale_s: float = DEFAULT_POSE_STALE_S,
                        quality=None, quality_max=None) -> dict:
    """定位安全门的**唯一**判定（与 wp、nav mode 完全无关）。

    ``state``：定位状态字符串。``None`` 表示**没有定位子系统**（导航未启动 /
    引导读取失败）—— 这种情况**不**触发停止：独立 VLA 不依赖 navmesh 也能跑，
    拿不到定位信息不等于「定位坏了」。

    ``state_reason``：定位侧给出的 reason（``pose_stale`` / ``not_relocalized``）。
    ``quality``：越高越差的质量量；``quality_max=None`` 时完全不参与判定。

    返回 ``{"ok": bool, "reason": str|None, "detail": {...}}``。
    reason 取值：``pose_stale`` / ``unlocalized`` / ``unknown`` /
    ``pose_age_missing`` / ``quality_over_threshold``；``None`` = 放行。
    """
    detail = {
        "state": None if state is None else str(state),
        "state_reason": None if state_reason is None else str(state_reason),
        "pose_age_s": None if pose_age_s is None else float(pose_age_s),
        "pose_stale_s": float(pose_stale_s),
        "quality": quality,
        "quality_max": quality_max,
        "quality_note": QUALITY_PLACEHOLDER_NOTE,
    }
    if state is None:
        detail["applicable"] = False
        return {"ok": True, "reason": None, "detail": detail}
    detail["applicable"] = True

    try:
        s = str(state).strip().lower()
    except Exception:  # noqa: BLE001
        s = "unknown"
    try:
        age = None if pose_age_s is None else float(pose_age_s)
    except (TypeError, ValueError):
        age = None
    try:
        stale_s = float(pose_stale_s)
    except (TypeError, ValueError):
        stale_s = DEFAULT_POSE_STALE_S

    issues = []
    # 1) 位姿过期：最具体、最可操作，优先报它（即使同时 state=unknown）。
    if age is not None and age > stale_s:
        issues.append("pose_stale")
    elif str(state_reason or "").strip().lower() == "pose_stale":
        issues.append("pose_stale")
    # 2) 未定位 / 未知：fail-closed。
    if s == "unlocalized":
        issues.append("unlocalized")
    elif s not in ("localized",) and "pose_stale" not in issues:
        issues.append("unknown")
    # 3) 声称已定位却没有位姿时间戳：口径矛盾，按过期处理（用更明确的 reason）。
    if s == "localized" and age is None:
        issues.append("pose_age_missing")
    # 4) 质量阈值：当前 quality_max 恒为 None，不参与；接真实指标后再开。
    if quality_max is not None and quality is not None:
        try:
            if float(quality) > float(quality_max):
                issues.append("quality_over_threshold")
        except (TypeError, ValueError):
            pass

    reason = issues[0] if issues else None
    detail["issues"] = issues
    return {"ok": not issues, "reason": reason, "detail": detail}



def from_session_mode(nav_state) -> int:
    """NavSession.mode（字符串）→ 记忆卡 [mode]。未知状态按 follow。

    未知按 follow 而不是 stop：导航状态字符串漂移（新状态名）时，
    宁可让模型按「正常跟随」跑，也不要凭空造出一个停车条件。
    """
    try:
        s = str(nav_state or "").strip().lower()
    except Exception:  # noqa: BLE001
        return MODE_FOLLOW
    return NAV_STATE_TO_MODE.get(s, MODE_FOLLOW)


def from_follower_axes(forward, turn_rate) -> str:
    """跟随器轴 → **运动学提示**（不再是 mode）。

    返回值只用于日志/诊断（``hold`` / ``turn`` / ``drive``），任何地方都不得
    把它当 ``mode`` 传给模型或当停止条件。它描述的是「跟随器这一拍有没有输出」，
    不描述导航状态。
    """
    try:
        f = float(forward or 0.0)
        t = float(turn_rate or 0.0)
    except (TypeError, ValueError):
        return "hold"
    if f <= 0.0 and t == 0.0:
        return "hold"
    if t != 0.0 and f <= 0.0:
        return "turn"
    return "drive"


def nav_state_is_stop(nav_state) -> bool:
    return bool(str(nav_state or "").strip().lower() in NAV_STATES_STOP)


def decide_stop(*, estop: bool = False, stale: bool = False, safety_veto: bool = False,
                localization_ok=None, localization_reason=None,
                nav_state=None, stop_head_p=None,
                stop_head_thresh: float = DEFAULT_STOP_HEAD_THRESH,
                prior_stop: bool = False, prior_source=None, prior_reason=None,
                wp_near_zero=None) -> dict:
    """唯一的停止判定。**wp 不参与**（只作为注记原样带出，便于日志对照）。

    ``localization_ok``：定位安全门的结果（``policy_modes.decide_localization``
    的 ``ok``）。``None`` = 没有定位信息（导航未启动），不触发停止；``False`` =
    未定位/位姿过期 → ``source="safety_gate_unlocalized"``，具体触发项写进
    ``stop_reason``（``unlocalized`` / ``unknown`` / ``pose_stale`` / ...）。

    ``prior_stop`` / ``prior_source``：上游（runtime）已经判过的停止（导航状态 /
    停止头）。执行器把自己能看到的 estop / stale / safety_gate 合并进来时用，
    这样优先级表只有一份，不会两头各排一次。

    返回 ``{"stop": bool, "source": str|None, "reason": str|None, "detail": {...}}``。
    ``source`` 为 None 表示不停；``detail["wp_near_zero"]`` 是**注记**，
    取 True 只说明「没有路径先验」，绝不改变 ``stop``。
    """
    detail = {
        "estopped": bool(estop),
        "stale": bool(stale),
        "safety_veto": bool(safety_veto),
        "localization_ok": localization_ok,
        "localization_reason": localization_reason,
        "nav_state": None if nav_state is None else str(nav_state),
        "stop_head_p": None if stop_head_p is None else float(stop_head_p),
        "stop_head_thresh": float(stop_head_thresh),
        "prior_stop": bool(prior_stop),
        "prior_source": prior_source,
        "prior_reason": prior_reason,
        # 路径先验注记：None=调用方没提供；False=有先验；True=无先验。
        "wp_near_zero": wp_near_zero,
        "path_prior": ("none" if wp_near_zero else "soft")
                     if wp_near_zero is not None else "unknown",
    }

    if estop:
        return {"stop": True, "source": "estop", "reason": "estop", "detail": detail}
    if stale:
        return {"stop": True, "source": "stale", "reason": "frame_stale", "detail": detail}
    if safety_veto:
        return {"stop": True, "source": "safety_gate", "reason": "obstacle_veto",
                "detail": detail}
    # 定位安全门：显式 False 才停（None = 没有定位子系统，不适用）。
    if localization_ok is False:
        return {"stop": True, "source": "safety_gate_unlocalized",
                "reason": localization_reason or "unknown", "detail": detail}
    if prior_stop:
        return {"stop": True, "source": prior_source or "nav_state",
                "reason": prior_reason or (prior_source or "nav_state"),
                "detail": detail}
    if nav_state_is_stop(nav_state):
        return {"stop": True, "source": "nav_state", "reason": str(nav_state).lower(),
                "detail": detail}
    if stop_head_p is not None:
        try:
            p = float(stop_head_p)
        except (TypeError, ValueError):
            p = 0.0
        if p >= float(stop_head_thresh):
            return {"stop": True, "source": "stop_head", "reason": "stop_head_prob",
                    "detail": detail}
    return {"stop": False, "source": None, "reason": None, "detail": detail}


def apply_stop_to_block(block, stop: bool):
    """停止时整块归零；**唯一**允许把块清零的地方。

    不 import numpy：本模块要能被 `.venv`（无 torch/numpy 依赖面最小的路径）导入。
    ``block`` 是嵌套 list（或任何可迭代行）；行长度按原样保留（2 维/3 维均可）。
    """
    if not stop:
        return block
    return [[0.0] * len(row) for row in block]
