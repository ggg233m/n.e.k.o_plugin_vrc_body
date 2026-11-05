# -*- coding: utf-8 -*-
"""转向链路端到端测试：必须证明 ω 真的走到执行出口，且单位正确。

覆盖本轮修的三个缺陷：
  D1 send_axes 只有两轴，turn 用关键字传会 TypeError -> 旧代码 except 后静默丢弃
  D2 omega_to_turn 用未定义变量 w_deg_s -> NameError
  D3 SafetyGate 的 w_max 默认 90（deg/s）而数据是 rad/s -> 差 57 倍
"""
import sys
sys.path.insert(0, "research/tools")
import math
from policy_executor import omega_to_turn, SafetyGate, PolicyExecutor

fails = []


def check(name, fn):
    try:
        fn()
        print("  PASS %s" % name)
    except AssertionError as e:
        fails.append(name)
        print("  FAIL %s: %s" % (name, e))


class Recorder:
    """记录两条通道各自收到了什么。"""
    def __init__(self):
        self.axes = []
        self.turns = []

    def send_axes(self, x, y, d):
        self.axes.append((x, y, d))
        return True

    def send_turn(self, deg):
        self.turns.append(float(deg))
        return True


def t_omega_conversion():
    # rad/s -> 归一化轴：1.57 rad/s 应接近满舵 1.0
    assert abs(omega_to_turn(1.57, 1.57) - 1.0) < 1e-6, "满舵应归一化到 1.0"
    assert abs(omega_to_turn(0.0, 1.57)) < 1e-9, "0 应映射到 0"
    assert abs(omega_to_turn(-1.57, 1.57) + 1.0) < 1e-6, "反向满舵应为 -1.0"
    assert omega_to_turn(99.0, 1.57) == 1.0, "必须钳位到 1.0"


def t_turn_reaches_channel():
    """核心：turn 必须出现在 send_turn，不能被 send_axes 的 TypeError 吞掉。"""
    rec = Recorder()
    ex = PolicyExecutor(rec.send_axes, dry_run=False, send_turn=rec.send_turn,
                        w_max=1.57, step_ms=100)
    # 一块 12 步，每步 (vx, vz, ω)，ω=1.0 rad/s 转 12 步
    block = [[0.0, 1.0, 1.0]] * 12
    # ⚠️ obs_t 必须是**当前时刻**：执行循环用 age = now - obs_t 判过期，
    # 传 0.0 会得到巨大的 age，块立刻被判 stale 而整块归零（第一次写测试时踩到）。
    import time as _t
    ex.submit(block, _t.perf_counter(), {})
    ex.start()
    import time; time.sleep(1.25)
    ex.stop()
    assert rec.axes, "摇杆通道一次都没收到"
    assert rec.turns, "转向通道一次都没收到 —— turn 又被丢弃了"
    # 每步转角 = 1.0/1.57 * 1.57 rad * 0.1 s = 0.1 rad = 5.73 deg
    expect = math.degrees(1.0 * 0.1)
    assert abs(abs(rec.turns[0]) - expect) < 1e-6, \
        "转角 %.4f deg 与预期 %.4f deg 不符（单位错？）" % (rec.turns[0], expect)


def t_no_turn_when_zero():
    """ω=0 时不应发转向指令（避免每拍都发一个 0 度转身）。"""
    rec = Recorder()
    ex = PolicyExecutor(rec.send_axes, dry_run=False, send_turn=rec.send_turn,
                        w_max=1.57, step_ms=100)
    import time as _t2
    ex.submit([[0.0, 1.0, 0.0]] * 12, _t2.perf_counter(), {})
    ex.start()
    import time; time.sleep(0.6)
    ex.stop()
    assert not rec.turns, "ω=0 时不该发转向，实际发了 %r" % rec.turns[:3]


def t_gate_units():
    """SafetyGate 的 w_max 必须是 rad/s 量级，不能再是 90(deg/s)。"""
    g = SafetyGate()
    assert g.w_max <= 5.0, "w_max=%.2f 像是 deg/s，数据是 rad/s" % g.w_max
    assert g.dw_max <= 10.0, "dw_max=%.2f 像是 deg/s^2" % g.dw_max
    b = g.apply([[0.0, 1.0, 100.0]] * 12, False, False, 0.0)
    assert abs(b[0, 2]) <= g.w_max + 1e-9, "ω 没被 w_max 钳住"


def t_turn_dropped_counter():
    """没接 send_turn 时必须计数，不能静默。"""
    rec = Recorder()
    ex = PolicyExecutor(rec.send_axes, dry_run=False, send_turn=None,
                        w_max=1.57, step_ms=100)
    import time as _t3
    ex.submit([[0.0, 1.0, 1.0]] * 12, _t3.perf_counter(), {})
    ex.start()
    import time; time.sleep(0.6)
    ex.stop()
    assert ex.n_turn_dropped > 0, "转向被丢弃却没有计数"


check("omega_to_turn 单位与钳位（D2）", t_omega_conversion)
check("turn 真的到达执行出口（D1）", t_turn_reaches_channel)
check("ω=0 不发转向", t_no_turn_when_zero)
check("SafetyGate w_max 是 rad/s 量级（D3）", t_gate_units)
check("转向被丢弃时有计数", t_turn_dropped_counter)

if fails:
    print()
    print("FAILED %d/5" % len(fails))
    raise SystemExit(1)
print()
print("OK 5/5")
