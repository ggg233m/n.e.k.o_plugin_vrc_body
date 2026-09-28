"""地点身份回归门（pooldeck/hall 混淆对 + 全部已知回环）。

背景
----
`ep39`（t=60.15–62.05，真值 hall）曾被 argmax 规则错误合并到 `ep23`（t≈36.1，真值 pooldeck），
ORB 内点 **205**，比全部真阳性都高，任何内点阈值都无法分开（§19.5）。

改判据为**图结构共识**（候选内点按已确认簇求和，迭代至稳定）后：
    argmax    : ok=5 wrong=1  (R=0.500 W=0.100)
    consensus : ok=6 wrong=0  (R=0.600 W=0.000)
只有 `qep39` 一条被改判，5 个真阳性全部保留。

本脚本把这套期望**固化成断言**，任何改动（切分、门槛、索引、共识规则）导致回归即失败。
不要在这里放宽断言来"让测试通过"——失败说明流水线改坏了。

用法
----
  .venv/Scripts/python.exe research/tools/regression_place_identity.py
退出码 0 = 全部通过；1 = 有断言失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.tools.episode_loop_eval import (  # noqa: E402
    build_episode_evidence, cand_stats, candidate_margin, consensus_assign,
    evaluate_assign, fold_independent_evidence, is_positive, promotion_report,
    route_history_seed_edges, visual_seed_edges,
)

import subprocess  # noqa: E402

TT = dict(min_sep=8.0, t_prov=20, t_conf=40, t_vote=20, n_vote=2)

# 期望：查询 episode -> 该次匹配所依据的帧的 GT 地点（必须命中）
EXPECT_GT = {
    30: "hall",        # ep30 内为 hall 帧，原 argmax 已正确
    31: "hall",
    35: "pooldeck",
    36: "pooldeck",
    38: "dancepool",
    39: "hall",        # ★ 反例：argmax 给 pooldeck(205)，共识必须翻成 hall
}
# 反例：这条查询**不得**再落到 pooldeck
FORBIDDEN_GT = {39: "pooldeck"}
# argmax 与共识之间允许被改判的查询（集合必须恰好等于它）
EXPECT_CHANGED = {39}


def guard_unit_tests() -> list[str]:
    """人工构造的反例：证明三道保护各自都能**翻转一次错误决定**。

    背景：在真实素材上（n=1）这三道保护是**惰性的**——注入错误高置信边后污染
    不扩散，`seed_level` 换档也不改任何结果（见 `.tmp/guard_ablation.json`）。
    换句话说，真实数据证明不了保护有用，因为它们不改变输出。
    所以这里用**手工构造的输入**把"保护的必要性"做成可判定的断言，
    而不是靠真实数据碰运气。保护若不生效，这些断言就会失败。
    """
    fails: list[str] = []

    # 时间轴：1..5 挤在同一访问时段（间隔 1.5s << indep_gap）；7 与 8 相隔 80s
    ep_t = [(0.0, 0.0)] * 12
    for k, pi in enumerate((1, 2, 3, 4, 5)):
        ep_t[pi] = (10.0 + k * 1.5, 11.0 + k * 1.5)
    ep_t[7] = (60.0, 61.0)
    ep_t[8] = (140.0, 141.0)

    # --- T1：独立访问规则——"大而不独立"的簇必须输给"小但独立"的簇 ---
    R = [{"pi": p, "max_inl": 40} for p in (1, 2, 3, 4, 5)]
    S = [{"pi": p, "max_inl": 40} for p in (7, 8)]
    r_free = fold_independent_evidence(R, ep_t, 0, 0.0)
    s_free = fold_independent_evidence(S, ep_t, 0, 0.0)
    r_guard = fold_independent_evidence(R, ep_t, 100, 8.0)
    s_guard = fold_independent_evidence(S, ep_t, 100, 8.0)
    if not r_free[0] > s_free[0]:
        fails.append(f"T1 前提不成立：无保护时大簇未获胜 R={r_free[0]} S={s_free[0]}")
    if r_guard[1] != 1:
        fails.append(f"T1 R 的独立访问条数={r_guard[1]} 期望 1")
    if s_guard[1] != 2:
        fails.append(f"T1 S 的独立访问条数={s_guard[1]} 期望 2")
    if not s_guard[0] > r_guard[0]:
        fails.append(f"T1 独立访问规则未翻转决定：R={r_guard[0]} S={s_guard[0]}")

    # --- T2：单条贡献封顶——一条超强证据不得压过两条中等但独立的证据 ---
    one_free = fold_independent_evidence([{"pi": 1, "max_inl": 400}], ep_t, 0, 8.0)
    two_free = fold_independent_evidence([{"pi": 7, "max_inl": 90},
                                          {"pi": 8, "max_inl": 90}], ep_t, 0, 8.0)
    one_cap = fold_independent_evidence([{"pi": 1, "max_inl": 400}], ep_t, 100, 8.0)
    two_cap = fold_independent_evidence([{"pi": 7, "max_inl": 90},
                                         {"pi": 8, "max_inl": 90}], ep_t, 100, 8.0)
    if not one_free[0] > two_free[0]:
        fails.append(f"T2 前提不成立：无封顶时单条强证据未获胜 "
                     f"{one_free[0]} {two_free[0]}")
    if not two_cap[0] > one_cap[0]:
        fails.append(f"T2 封顶未翻转决定：单条={one_cap[0]} 两条={two_cap[0]}")

    # --- T3：播种边际门——歧义的 confirmed_visual_loop 不得当种子 ---
    m_amb = candidate_margin([{"max_inl": 205}, {"max_inl": 126}])
    if not 1.5 < m_amb < 2.0:
        fails.append(f"T3 candidate_margin({m_amb:.3f}) 期望约 1.63")
    if m_amb >= 2.0:
        fails.append("T3 歧义边(205/126)竟满足 seed_margin=2.0")
    # 边际是"最优/次优"：决定性边边际大，歧义边边际小
    if not m_amb < candidate_margin([{"max_inl": 120}, {"max_inl": 30}]):
        fails.append("T3 边际排序方向错误（歧义边边际应小于决定性边）")

    # --- T4：可撤销——独立证据不足只能是 hypothesis；删掉指派即回到未确认 ---
    detail = {
        5: {"cluster": 1, "cluster_score": 10, "cluster_score_runner_up": 0,
            "n_independent_visits": 1, "independent_members": [1],
            "pick": 1, "pick_inl": 50},
        6: {"cluster": 2, "cluster_score": 20, "cluster_score_runner_up": 0,
            "n_independent_visits": 2, "independent_members": [2, 3],
            "pick": 2, "pick_inl": 60},
        "_meta": {},
    }
    pr = promotion_report({}, {5: 1, 6: 2}, detail, 2)
    if pr["n_promoted"] != 1 or pr["n_alias_retained"] != 1:
        fails.append(f"T4 提升判定错误：promoted={pr['n_promoted']} "
                     f"alias={pr['n_alias_retained']}（期望 1/1）")
    if not all(r["state"] == "identity_hypothesis" for r in pr["promoted"]):
        fails.append("T4 提升后的条目状态不是 identity_hypothesis")
    pr2 = promotion_report({}, {6: 2}, detail, 2)
    if pr2["n_hypothesis"] != 1:
        fails.append(f"T4 撤销失败：删掉指派后仍有 {pr2['n_hypothesis']} 条 hypothesis")
    return fails


def degrade_label_test(G, keys) -> list[str]:
    """证明"路线历史降级为视觉代理"的溯源标签能存活进共识元数据。

    背景（任务 #39）：CLI 把 `route_history_seed_edges()` 的 `seed_edges` 喂给
    `consensus_assign`。降级路径下 `seed_edges` 非空（原样保留视觉边），旧代码
    只要 `seed_edges` 非空就无条件把 `seed_source` 标成 `external(route/semantic)`，
    于是"视觉代理降级"被误标成"可用路线/语义通道"，审计时无法分辨真实用过哪条通道。
    这里钉住：降级路径的 `seed_source` 不得是 `external(route/semantic)`，且
    `route_history_available` 必须为 False；可用路径则必须为 True。
    """
    fails: list[str] = []
    vis = visual_seed_edges(G, keys, TT["t_prov"], TT["t_conf"], TT["t_vote"],
                            TT["n_vote"], TT["min_sep"])

    # 降级：无动作日志 —— 旧代码会把这条非空 seed_edges 误标为 external。
    no_log = route_history_seed_edges(vis, None)
    _, _, cdet = consensus_assign(
        G, keys, TT["t_prov"], TT["t_conf"], TT["t_vote"], TT["n_vote"],
        TT["min_sep"], seed_edges=no_log["seed_edges"],
        seed_source_hint=no_log["source"],
        route_history_available=no_log["available"])
    meta = cdet.get("_meta", {})
    if meta.get("seed_source") == "external(route/semantic)":
        fails.append("降级路径被误标为 external(route/semantic)："
                     f"seed_source={meta.get('seed_source')!r}")
    if meta.get("route_history_available") is not False:
        fails.append("降级路径 route_history_available 应为 False，"
                     f"实际={meta.get('route_history_available')!r}")
    if meta.get("seed_source") != "visual_proxy":
        fails.append("降级路径 seed_source 应为 visual_proxy，"
                     f"实际={meta.get('seed_source')!r}")

    # 对照：动作日志两端都走过 —— 应当如实标成可用。
    walked = [{"episode": i, "n_records": 5, "osc_forward_distance": 1.0}
              for i in range(G["n_ep"])]
    ok_log = route_history_seed_edges(vis, walked)
    _, _, cdet2 = consensus_assign(
        G, keys, TT["t_prov"], TT["t_conf"], TT["t_vote"], TT["n_vote"],
        TT["min_sep"], seed_edges=ok_log["seed_edges"],
        seed_source_hint=ok_log["source"],
        route_history_available=ok_log["available"])
    meta2 = cdet2.get("_meta", {})
    if meta2.get("route_history_available") is not True:
        fails.append("可用路径 route_history_available 应为 True，"
                     f"实际={meta2.get('route_history_available')!r}")
    return fails



def main() -> int:
    ep = REPO / ".tmp/episodes.npz"
    idx = REPO / ".tmp/loop_frame_index.npz"
    if not ep.exists() or not idx.exists():
        print(f"[skip] 缺少输入：{ep} / {idx}", file=sys.stderr)
        return 1

    G = build_episode_evidence(ep, idx, TT["min_sep"])
    keys = list(range(G["n_ep"]))

    asg, hist, cdet = consensus_assign(G, keys, TT["t_prov"], TT["t_conf"],
                                       TT["t_vote"], TT["n_vote"], TT["min_sep"])
    cons = evaluate_assign(G, keys, asg, TT["min_sep"], TT["t_vote"])

    base: dict[int, int] = {}
    for qi in keys:
        q = G["per_query"][qi]
        if not is_positive(q, G, TT["min_sep"]):
            continue
        st = cand_stats(q, TT["t_vote"])
        if st and st[0]["max_inl"] >= TT["t_conf"]:
            base[qi] = st[0]["pi"]
    basel = evaluate_assign(G, keys, base, TT["min_sep"], TT["t_vote"])

    fails: list[str] = []

    got = {r["query_episode"]: (r["assigned"], r["assigned_gt"], r["max_inl"])
           for r in cons["rows"]}
    print("=" * 74)
    print("地点身份回归门")
    print("=" * 74)
    print("  %-4s %-14s %-10s %-12s %-6s %s" % (
        "qep", "window", "assigned→", "GT@best", "inl", "verdict"))
    for r in cons["rows"]:
        print("  %-4d %6.2f-%6.2f %-10s %-12s %-6d %s" % (
            r["query_episode"], r["t"][0], r["t"][1], str(r["assigned"]),
            str(r["assigned_gt"]), r["max_inl"], r["verdict"]))
    print()
    print("  argmax    : ok=%d wrong=%d unassigned=%d  R=%.3f W=%.3f" % (
        basel["ok"], basel["wrong"], basel["unassigned"],
        basel["recall"] or 0, basel["wrong_rate"] or 0))
    print("  consensus : ok=%d wrong=%d unassigned=%d  R=%.3f W=%.3f" % (
        cons["ok"], cons["wrong"], cons["unassigned"],
        cons["recall"] or 0, cons["wrong_rate"] or 0))
    print()

    # --- 断言 1：期望的帧级地点全部命中 ---
    for qi, want in EXPECT_GT.items():
        if qi not in got:
            fails.append(f"qep{qi}: 未被指派（期望 GT={want}）")
            continue
        _, g, inl = got[qi]
        if g != want:
            fails.append(f"qep{qi}: GT@best={g!r} 期望 {want!r}（inl={inl}）")

    # --- 断言 2：反例不得再落到 pooldeck ---
    for qi, bad in FORBIDDEN_GT.items():
        if qi in got and got[qi][1] == bad:
            fails.append(f"qep{qi}: 又回到 {bad!r}——共识规则失效")

    # --- 断言 3：被改判的查询集合必须恰好是 EXPECT_CHANGED ---
    changed = {r["query_episode"] for r in cons["rows"]
               if base.get(r["query_episode"]) != r["assigned"]}
    if changed != EXPECT_CHANGED:
        fails.append(f"改判集合={sorted(changed)} 期望 {sorted(EXPECT_CHANGED)}")

    # --- 断言 4：共识不得损失任何真阳性 ---
    if cons["ok"] < basel["ok"]:
        fails.append(f"共识召回下降：{basel['ok']} -> {cons['ok']}")
    if cons["wrong"] > 0:
        fails.append(f"共识后仍有误合并：{cons['wrong']}")

    # --- 断言 5：迭代必须收敛（不得靠打满迭代次数） ---
    if len(hist) > 1 and hist[-1]["changed"] != 0:
        fails.append(f"共识未收敛：{hist}")

    # --- 断言 6：argmax 基线本身必须仍然"是错的"（否则这个回归失去意义） ---
    if basel["wrong"] == 0:
        fails.append("argmax 基线已无误合并——阈值或数据变了，需重定基线")

    # --- 断言 7：保护参数必须真的被应用（不是被静默忽略的死参数） ---
    meta = cdet.get("_meta", {})
    if (meta.get("contrib_cap"), meta.get("indep_gap_s"),
            meta.get("seed_margin"), meta.get("seed_level")) != (
            100, 8.0, 2.0, "visual_decisive"):
        fails.append(f"保护参数未按默认生效：{meta}")

    # --- 断言 8：歧义的 ep39 不得当播种边（边际 205/126≈1.63 < 2.0） ---
    seeds = set(meta.get("seed_episodes", []))
    if 39 in seeds:
        fails.append(f"歧义 episode 39 竟被当作高置信种子：seeds={sorted(seeds)}")

    # --- 断言 9：可撤销——qep39 必须保留为可撤销的 identity_hypothesis ---
    promo = promotion_report(G, asg, cdet, 2)
    row39 = [r for r in promo["promoted"] + promo["alias_retained"]
             if r["query_episode"] == 39]
    if not row39:
        fails.append("qep39 未出现在 identity_hypothesis 报告里")
    elif row39[0]["assigned"] != asg.get(39) or \
            row39[0]["state"] != "identity_hypothesis":
        fails.append(f"qep39 状态异常：{row39[0]}")

    # --- 断言 10：人工构造的保护反例必须全部通过 ---
    fails.extend(guard_unit_tests())

    # --- 断言 11：路线历史通道只做"必要条件的过滤"，且无日志必须显式降级 ---
    visual = dict(asg)          # 以共识指派代表"高置信视觉边"集合
    no_log = route_history_seed_edges(visual, None)
    if no_log["available"] or no_log["source"] != "visual_proxy":
        fails.append(f"无动作日志时未显式降级：source={no_log['source']}")
    if no_log["seed_edges"] != visual:
        fails.append("无动作日志时应回退到视觉代理（视觉边原样保留）")

    blank = [{"episode": i, "n_records": 0, "osc_forward_distance": 0.0}
             for i in range(G["n_ep"])]
    filt = route_history_seed_edges(visual, blank)
    if filt["seed_edges"]:
        fails.append(f"无任何动作记录时仍播种 {len(filt['seed_edges'])} 条边")
    if not set(filt["seed_edges"]).issubset(set(visual)):
        fails.append("路线历史通道新增了视觉上没有的边（只允许过滤）")

    walked = [{"episode": i, "n_records": 5, "osc_forward_distance": 1.0}
              for i in range(G["n_ep"])]
    if route_history_seed_edges(visual, walked)["seed_edges"] != visual:
        fails.append("两端都有动作记录时不应丢边")

    # --- 断言 12：路线历史通道真的接在运行链路上（防"死代码"复发） ---
    # 之前的教训：`route_history_seed_edges()` 曾只被文档提到、CLI 从不调用，
    # 于是"有日志也没用"。这里同时钉住"闸门与共识共用同一份播种规则"和
    # "CLI 真的暴露了 --action-log"。
    vis = visual_seed_edges(G, keys, TT["t_prov"], TT["t_conf"], TT["t_vote"],
                            TT["n_vote"], TT["min_sep"])
    if set(vis) != set(meta.get("seed_episodes", [])):
        fails.append("闸门与共识的视觉播种集不一致："
                     f"gate={sorted(vis)} consensus={sorted(meta.get('seed_episodes', []))}")
    if not vis:
        fails.append("视觉播种集为空——播种规则或阈值变了，回归失去意义")
    try:
        help_out = subprocess.run(
            [sys.executable, str(REPO / "research" / "tools" / "episode_loop_eval.py"), "--help"],
            capture_output=True, text=True, timeout=120).stdout
    except Exception as exc:                                    # noqa: BLE001
        help_out = ""
        fails.append(f"无法运行 CLI --help：{exc}")
    if "--action-log" not in help_out:
        fails.append("CLI 未暴露 --action-log：路线历史通道又变成死代码")

    # --- 断言 13：真实配置必须能构造（守一个已发生过的 P0） ---
    # `PluginConfig.from_mapping` 里加了 `action_timeline_fps=...`，但 dataclass 漏了
    # 同名字段 ⇒ `DriverLogConfig.__init__() got an unexpected keyword argument`
    # ⇒ **整个 BackendService 构造不出来**（插件起不来）。当时 60 条单测 + 12 条断言
    # 全绿，因为没有一条真的去构造配置。这里补上。
    try:
        import types as _types

        if "neko_anyadance_body" not in sys.modules:
            _pkg = _types.ModuleType("neko_anyadance_body")
            _pkg.__path__ = [str(REPO)]        # type: ignore[attr-defined]
            sys.modules["neko_anyadance_body"] = _pkg
        from neko_anyadance_body.config import PluginConfig
        _cfg = PluginConfig.from_mapping({})
        if float(_cfg.driver_log.action_timeline_fps) <= 0:
            fails.append("driver_log.action_timeline_fps 默认值非法")
    except Exception as exc:                                    # noqa: BLE001
        fails.append(f"PluginConfig.from_mapping 失败（后端将无法构造）："
                     f"{type(exc).__name__}: {exc}")

    # --- 断言 14：离线时间基准 harness 必须整体通过（阶段一验收器） ---
    # 用合成数据钉住「帧源 → VideoTimebase → 帧索引/帧号 → 动作日志 →
    # episode_action_summary → 路线闸门 → 共识」这一段。产出**只是**
    # offline_timebase_validation，**不是**真实路线历史 / VRChat / AnyaDance 验证。
    # 用 --no-real-artifacts 让回归门快且不依赖 .tmp 大文件。
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO / "research" / "tools" / "offline_timebase_harness.py"),
             "--no-real-artifacts"],
            capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            fails.append("离线时间基准 harness 失败：\n"
                         + (proc.stdout or proc.stderr)[-800:])
        elif "offline_timebase_validation" not in proc.stdout:
            fails.append("harness 未显式标注 offline_timebase_validation")
    except Exception as exc:                                    # noqa: BLE001
        fails.append(f"无法运行离线时间基准 harness：{exc}")

    # --- 断言 15：路线历史降级标签必须存活进共识元数据（防 #39 回归） ---
    # 无动作日志时，CLI 仍会把非空 seed_edges 喂给共识；若共识把来源一律标成
    # external(route/semantic)，"视觉代理降级"就被误标成可用通道，审计无法分辨。
    # 这里钉住降级路径如实标 visual_proxy 且 route_history_available=False。
    fails.extend(degrade_label_test(G, keys))

    if fails:
        print("❌ 回归失败：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ 全部断言通过：ep39 由 pooldeck 改判为 hall，5 个真阳性无损失，"
          "共识后零误合并。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
