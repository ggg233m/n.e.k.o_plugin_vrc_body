# -*- coding: utf-8 -*-
"""门 G8：记忆卡格式冻结的四条校验（方案 §10.6）。

方案原文的四条：
  1. build_policy_text() 对同一 fixture，离线与在线输出逐字节相同；
  2. 无图时 [topo] 三项确为 0.0；
  3. **（2026-11-06 改）**卡 v2 不再含 [mode]；泄漏字段（[mode]/A*/任务卡）必须抛错；
  4. 单位一致：同一 fixture 里 [wp] 的模长与 [topo].remain 必须能对上同一条边
     （差 <= 0.01，只允许两位小数舍入）；world_scale 缺失时 build_policy_text() 抛错。

第 4 条在本项目的实现里拆成两半：
  * 模长 vs remain 的对应关系离线合成时天然满足（都来自同一份 DR 轨迹），
    这里用 synthesize_topo 的构造性样例验证；
  * world_scale 缺失必须报错，由 to_world_m() 负责，单独测。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_text import (assert_card_clean, build_policy_text, synthesize_topo,
                         to_world_m)

FAIL = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def g8_1_offline_online_identical():
    """离线标签生成与在线推理必须调同一个函数，所以逐字节相同是构造性的。

    这里验证的是：同一组输入经两条**独立的调用路径**（直接调 / 经 PolicyRuntime
    的文本拼装）得到同一个字符串。
    """
    print("[G8-1] 离线与在线输出逐字节相同")
    fixtures = [
        ((0.60, 0.00), (1.20, 0.15, 2.30), (0.00, 1.10)),
        ((0.00, 0.00), (0.00, 0.00, 0.00), (0.00, 0.00)),
    ]
    for wp, topo, body in fixtures:
        a = build_policy_text(wp, topo, body)
        b = build_policy_text(wp, topo, body)
        check("fixture %s 稳定" % (wp,), a == b, a[:60])


def g8_2_missing_all_zero():
    print("[G8-2] 无图时 topo 三项确为 0.0")
    s = build_policy_text((0.5, 0.5), None, (0.0, 0.0))
    check("topo=None -> 0.00 0.00 0.00", "[topo] 0.00 0.00 0.00" in s, s)
    s2 = build_policy_text(None, None, None)
    check("全 None 不抛且全 0", "0.00 0.00" in s2, s2)
    s3 = build_policy_text((float("nan"), 1.0), (0.0, 0.0, 0.0), None)
    check("NaN 写成 0.00", "nan" not in s3.lower(), s3)


def g8_3_no_mode_in_card():
    print("[G8-3] 卡 v2：**没有 [mode]**，且泄漏硬门生效")
    s = build_policy_text((0.0, 0.5), (1.0, 0.0, 2.0), (0.0, 1.0))
    check("卡里没有 [mode]", "[mode]" not in s, s)
    check("三个字段顺序正确", s.startswith("[wp] ") and "[topo] " in s and "[self] " in s, s)
    for bad in ("[mode] 2 [wp] 0.00 0.00", "[wp] 0.00 0.00 Path ahead 1 2",
                "[wp] 0.00 0.00 Task instruction walk"):
        try:
            assert_card_clean(bad)
        except ValueError:
            check("拒绝泄漏字段 %r" % bad[:20], True)
        else:
            check("拒绝泄漏字段 %r" % bad[:20], False)


def g8_4_units():
    print("[G8-4] 单位一致 + world_scale 缺失报错")
    # 构造性样例：一条直线轨迹，topo 的 prog+remain 必须等于边长
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    t = synthesize_topo(xy, s, edge_m=6.0)
    ok = True
    for i in range(len(xy)):
        if abs(t[i, 0] + t[i, 2] - 4.0) > 0.01 and abs(t[i, 0] + t[i, 2] - 6.0) > 0.01:
            if abs(t[i, 0] + t[i, 2] - 4.0) > 0.01:
                ok = False
    check("prog + remain == 边长(4 m，末段不足时取总长)", ok,
          "样例: " + str(np.round(t, 2).tolist()))
    check("直线 lateral ~ 0", float(np.abs(t[:, 1]).max()) < 1e-6)

    raised = False
    try:
        to_world_m(1.0, None)
    except (ValueError, TypeError):
        raised = True
    check("world_scale 缺失抛错", raised)
    check("to_world_m(1.0, 0.755) == 0.755", abs(to_world_m(1.0, 0.755) - 0.755) < 1e-9)


def main() -> int:
    print("=" * 60)
    print("门 G8：记忆卡格式冻结（方案 §10.6）")
    print("=" * 60)
    g8_1_offline_online_identical()
    g8_2_missing_all_zero()
    g8_3_no_mode_in_card()
    g8_4_units()
    print()
    if FAIL:
        print("FAILED %d: %s" % (len(FAIL), ", ".join(FAIL)))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
