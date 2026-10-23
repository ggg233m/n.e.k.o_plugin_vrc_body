# -*- coding: utf-8 -*-
r"""``pacing.py`` 的自检台。Lead 把节拍器接进 ``nav_online.py`` 之后，跑这一条命令就能复核。

用法::

    .venv\Scripts\python.exe -u research\tools\pacing_selftest.py
    .venv\Scripts\python.exe -u research\tools\pacing_selftest.py --hz 20 --n 400
    .venv\Scripts\python.exe -u research\tools\pacing_selftest.py --quick

退出码：0 = 全部通过；1 = 有检查项不达标（会把不达标项逐条打出来）。

它量七件事，每一件都对着 ``pacing.py`` docstring 里的某个实测数字：

  1. 时钟域：``Pacer`` 实际用的 clock 的读数台阶必须远小于周期（§4）。
  2. 定拍精度：``Pacer`` 的 ``|err|p99`` 必须 ≤ 周期的 1%（§2 结论 2.3）。
  3. 频率：实测 Hz 必须等于目标 Hz（±0.5%）。**这条是给 Lead 复核"有没有接线"用的。**
  4. 无忙等：``spin_frac`` 必须 ≤ 10%（§2：预算 1.5 ms / 周期 50 ms ⇒ 实测 2.33%）。
  5. 停机：``stop.set()`` 到 ``wait()`` 返回的延迟必须 ≤ SLICE_S + 5 ms（§5）。
  6. 掉拍：body 稳定超一个周期时不得出现零间隔连续发拍（§3）。
  7. 并发：N 条线程各自的频率必须都等于目标（§8）。

对照台（``--compare``）还会把 Event.wait / time.sleep / busy-spin 一起量一遍，
给出"为什么不用它们"的当场证据，而不是引用本文档的数字。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pacing import (DEFAULT_SPIN_S, HIGH_RES_CLOCK, SLICE_S, Gate, Pacer, RateMeter,
                            check_clock, clock_quantum, sleep_until)


# ---------------------------------------------------------------- 小工具

def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(len(s) * q)))]


def sd(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else float("nan")


class Checker:
    def __init__(self):
        self.fails = []
        self.rows = []

    def check(self, ok, label, detail):
        self.rows.append((("PASS" if ok else "FAIL"), label, detail))
        print("  [%s] %-32s %s" % ("PASS" if ok else "FAIL", label, detail))
        if not ok:
            self.fails.append((label, detail))


# ---------------------------------------------------------------- 1. 时钟

def check_clock_domain(chk, period):
    print("\n== 1. 时钟域 ==")
    budget = period * 0.02
    # perf_counter 必须够细；monotonic 必须**不够细**（它是 GetTickCount64，15.625 ms）。
    # 两条都断言成"预期"，否则这项检查只是复述代码。
    q_pc = clock_quantum(time.perf_counter, samples=4000)
    chk.check(q_pc is not None and q_pc <= budget, "perf_counter 够细",
              "台阶 %.4f ms（预算 %.3f ms）" % ((q_pc or float("inf")) * 1e3, budget * 1e3))
    time.sleep(0.06)   # 让 15.625 ms 粒度的 clock 有机会前进几步
    q_mo = clock_quantum(time.monotonic, samples=40000)
    # None = 采样窗口内一次都没前进 ⇒ **更**粗，同样算"符合预期"。
    mo_too_coarse = (q_mo is None) or (q_mo > budget)
    chk.check(mo_too_coarse, "monotonic 太粗（预期如此）",
              "%s > 预算 %.3f ms ⇒ 必须用 perf_counter"
              % ("窗口内未前进(>%.1f ms)" % (0.06 * 1e3) if q_mo is None
                 else "台阶 %.4f ms" % (q_mo * 1e3), budget * 1e3))
    warn = check_clock(time.monotonic, period, samples=4000)
    chk.check(warn is not None, "check_clock 抓住 monotonic",
              "正确报错: %s" % (warn or "**没报错！**"))
    warn2 = check_clock(HIGH_RES_CLOCK, period, samples=4000)
    chk.check(warn2 is None, "check_clock 放行 perf_counter", str(warn2 or "ok"))


# ---------------------------------------------------------------- 2/3/4. 定拍

def measure_pacer(period, n, spin_s=None):
    """跑 n 拍，返回 (errs_s, telemetry)。用 ``advance`` 不必要——这里就是纯 ``wait()``。"""
    p = Pacer(period_s=period, spin_s=spin_s, name="selftest")
    p.reset()
    errs = []
    t_first = None
    for tick in p.ticks():
        if t_first is None:
            t_first = tick.actual
        errs.append(tick.late)
        if len(errs) >= n:
            break
    return errs, p.telemetry(), p


def check_pacing(chk, period, n, spin_s=None):
    print("\n== 2/3/4. 定拍（%d 拍，周期 %.2f ms，自旋 %.2f ms） =="
          % (n, period * 1e3, (DEFAULT_SPIN_S if spin_s is None else spin_s) * 1e3))
    errs, tel, p = measure_pacer(period, n, spin_s)
    a = [abs(e) for e in errs]
    budget = period * 0.01
    chk.check(pct(a, .99) <= budget, "|err|p99 <= 周期 1%",
              "%.4f ms (预算 %.4f ms)  p50 %.4f  max %.4f  sd %.4f"
              % (pct(a, .99) * 1e3, budget * 1e3, pct(a, .5) * 1e3,
                 max(a) * 1e3, sd(errs) * 1e3))
    hz_target = 1.0 / period
    hz = tel["achieved_hz"] or 0.0
    chk.check(abs(hz - hz_target) <= hz_target * 0.005, "实测 Hz == 目标 Hz (±0.5%)",
              "%.4f Hz vs 目标 %.4f Hz  gap_p50 %.4f ms"
              % (hz, hz_target, (p._rate.gap_p50_s() or 0.0) * 1e3))
    sf = tel["spin_frac"] or 0.0
    chk.check(sf <= 0.10, "spin_frac <= 10% (无忙等)",
              "%.2f%%（预算 %.2f ms = 周期的 %.1f%%）"
              % (sf * 100, tel["spin_budget_ms"], 100.0 * tel["spin_budget_ms"] / (period * 1e3)))
    chk.check(tel["clock_warning"] is None, "clock_warning 为空",
              str(tel["clock_warning"] or "ok"))
    return errs, tel


# ---------------------------------------------------------------- 5. 停机

def check_stop_latency(chk, period, trials=24):
    print("\n== 5. 停机延迟（目标线程正睡在 Pacer.wait() 里） ==")
    bound = SLICE_S + 0.005
    lats = []
    for i in range(trials):
        stop = threading.Event()
        p = Pacer(period_s=period, stop=stop, name="stop")
        p.reset()
        ready = threading.Event()
        done = threading.Event()
        seen = []

        def body():
            ready.set()
            for _ in p.ticks():
                pass
            seen.append(time.perf_counter())
            done.set()

        th = threading.Thread(target=body, daemon=True)
        th.start()
        ready.wait()
        # 让目标线程进到 sleep 里（相位分散，避免每次都在同一段）
        time.sleep(0.004 + (i % 9) * 0.005)
        t_set = time.perf_counter()
        stop.set()
        if not done.wait(2.0):
            lats.append(float("inf"))
            continue
        lats.append(seen[0] - t_set)
    lat_ms = [x * 1e3 for x in lats]
    chk.check(max(lat_ms) <= bound * 1e3, "stop 延迟 <= SLICE+5ms",
              "p50 %.2f ms  p95 %.2f ms  max %.2f ms (上界 %.1f ms)"
              % (pct(lat_ms, .5), pct(lat_ms, .95), max(lat_ms), bound * 1e3))


# ---------------------------------------------------------------- 6. 掉拍

def check_overrun(chk, period, seconds=4.0):
    print("\n== 6. 掉拍行为（每拍 body 稳定超时 2.5 倍周期） ==")
    p = Pacer(period_s=period, spin_s=0.0, name="overrun")
    p.reset()
    fires = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        tick = p.wait()
        if tick.stopped:
            break
        fires.append(tick.actual)
        if len(fires) > 4000:
            break
        time.sleep(period * 2.5 - 0.0)      # body 超载
    gaps = [fires[i + 1] - fires[i] for i in range(len(fires) - 1)]
    mn = min(gaps) if gaps else 0.0
    tel = p.telemetry()
    chk.check(mn >= period * 0.5, "超载时无零间隔连发",
              "最小拍间隔 %.2f ms（周期 %.2f ms）；%d 拍 / %.1f s = %.2f Hz；skipped_total %d"
              % (mn * 1e3, period * 1e3, len(fires), seconds, len(fires) / seconds,
                 tel["skipped_total"]))
    chk.check(tel["skipped_total"] > 0, "掉拍被计数", "skipped_total=%d" % tel["skipped_total"])


# ---------------------------------------------------------------- 6b. 双目门悬崖

def check_gate(chk, period, n=400):
    """复现 nav_online.py:1344 的悬崖：位姿 tick 实测 p50 = 46.889 ms，双目 100 ms。

    相对累加 ⇒ 0.1/0.046889 = 2.13 ⇒ 必须等到第 3 拍 ⇒ 7.11 Hz，比配置低 29%。
    """
    print("\n== 6b. 双目门：绝对累加 vs 相对累加（位姿 tick = 46.889 ms，双目 100 ms） ==")
    tick_p = 0.046889
    stereo_p = 0.1
    ticks = [i * tick_p for i in range(n)]

    gate = Gate(period_s=stereo_p, name="stereo")
    gate.reset(ticks[0])
    fires = [t for t in ticks if gate.due(t)]
    hz_abs = (len(fires) - 1) / (fires[-1] - fires[0])

    ns = ticks[0]
    fires_rel = []
    for t in ticks:
        if t >= ns:
            fires_rel.append(t)
            ns = t + stereo_p          # ← nav_online.py:1344 现状
    hz_rel = (len(fires_rel) - 1) / (fires_rel[-1] - fires_rel[0])

    chk.check(hz_abs >= 9.9, "Gate 绝对累加 ≈ 10 Hz",
              "%.3f Hz（%d 次触发 / %.2f s）" % (hz_abs, len(fires), fires[-1] - fires[0]))
    chk.check(hz_rel < hz_abs - 1.0, "相对累加确实塌（对照）",
              "%.3f Hz ⇒ 绝对累加修掉了 %.1f%% 的缺口"
              % (hz_rel, 100.0 * (hz_abs - hz_rel) / hz_rel))
    tel = gate.telemetry()
    chk.check(tel["grids_per_fire"] <= 1.05, "每格都发（无补发）",
              "grids_per_fire=%.3f skipped=%d" % (tel["grids_per_fire"], tel["skipped_total"]))
    # 幂等：同一时刻连调两次，第二次必须 False（否则一次 tick 会连发两帧双目）
    gate2 = Gate(period_s=stereo_p)
    gate2.reset(0.0)
    first = gate2.due(0.15)
    second = gate2.due(0.15)
    chk.check(first and not second, "due() 幂等安全",
              "first=%s second=%s（同刻连调只发一次）" % (first, second))


# ---------------------------------------------------------------- 7. 并发

def check_concurrency(chk, period, n, threads=4):
    print("\n== 7. 并发（%d 条线程各自一个 Pacer） ==" % threads)
    results = [None] * threads
    barrier = threading.Barrier(threads)

    def worker(i):
        barrier.wait()
        errs, tel, _ = measure_pacer(period, n)
        results[i] = (errs, tel)

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    w0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join(n * period * 3 + 5.0)
    wall = time.perf_counter() - w0
    hz_target = 1.0 / period
    bad = []
    desc = []
    for i, r in enumerate(results):
        if r is None:
            bad.append("thread %d 没结束" % i)
            continue
        errs, tel = r
        hz = tel["achieved_hz"] or 0.0
        desc.append("%.3fHz" % hz)
        if abs(hz - hz_target) > hz_target * 0.01:
            bad.append("thread %d 频率 %.3f != %.3f" % (i, hz, hz_target))
    ok = not bad
    chk.check(ok, "4 线程频率全部达标",
              "%s（墙钟 %.2f s）%s" % (" ".join(desc), wall, "" if ok else "  " + "; ".join(bad)))


# ---------------------------------------------------------------- 对照台

def compare(period, n):
    """当场把其它原语量一遍，作为"为什么不用它"的证据。"""
    print("\n== 对照台：其它原语在同一台机器上（%d 拍，周期 %.2f ms） ==" % (n, period * 1e3))
    stop = threading.Event()

    def bench(fn):
        d = time.perf_counter()
        errs = []
        for _ in range(n):
            d += period
            fn(d)
            errs.append((time.perf_counter() - d) * 1e3)
        return errs

    def a_sleep(deadline):
        while True:
            rem = deadline - time.perf_counter()
            if rem <= 0.0:
                return
            time.sleep(rem)

    def b_spin(deadline):
        pc = time.perf_counter
        while pc() < deadline:
            pass

    def d_event(deadline):
        while True:
            rem = deadline - time.perf_counter()
            if rem <= 0.0:
                return
            stop.wait(rem)

    rows = [("time.sleep 到截止点", a_sleep), ("busy-spin", b_spin),
            ("Event.wait 到截止点", d_event)]
    # Pacer 自己也放进同一张表，确保口径一致
    def c_pacer(_deadline):
        Pacer(period_s=1e-9).wait() if False else None

    print("  %-24s %10s %10s %10s" % ("原语", "|err|p50", "|err|p99", "max"))
    for label, fn in rows:
        e = bench(fn)
        a = [abs(x) for x in e]
        print("  %-24s %9.3f %10.3f %10.3f" % (label, pct(a, .5), pct(a, .99), max(a)))
    # Pacer 走自己一条，口径 = tick.late
    p = Pacer(period_s=period)
    p.reset()
    errs = []
    for tick in p.ticks():
        errs.append(tick.late * 1e3)
        if len(errs) >= n:
            break
    a = [abs(x) for x in errs]
    print("  %-24s %9.3f %10.3f %10.3f   ← 本模块"
          % ("Pacer (sleep+spin)", pct(a, .5), pct(a, .99), max(a)))


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="pacing.py 自检台")
    ap.add_argument("--hz", type=float, default=20.0, help="目标频率（默认 20 = pose_hz）")
    ap.add_argument("--n", type=int, default=400, help="每项采样的拍数（默认 400）")
    ap.add_argument("--quick", action="store_true", help="= --n 120，快速冒烟")
    ap.add_argument("--compare", action="store_true", help="额外跑对照台")
    ap.add_argument("--no-overrun", action="store_true", help="跳过掉拍测试（它要 4 s）")
    args = ap.parse_args(argv)
    n = 120 if args.quick else max(40, args.n)
    period = 1.0 / args.hz

    print("pacing_selftest: python %s" % sys.version.split()[0])
    print("目标 %.3f Hz（周期 %.3f ms），每项 %d 拍" % (args.hz, period * 1e3, n))

    chk = Checker()
    check_clock_domain(chk, period)
    check_pacing(chk, period, n)
    check_stop_latency(chk, period)
    check_gate(chk, period)
    if not args.no_overrun:
        check_overrun(chk, period)
    check_concurrency(chk, period, max(40, n // 2))
    if args.compare:
        compare(period, min(n, 200))

    print("\n== 汇总 ==")
    if chk.fails:
        print("  FAILED  %d 项：" % len(chk.fails))
        for label, detail in chk.fails:
            print("    * %s -> %s" % (label, detail))
        return 1
    print("  ALL PASS（%d 项）" % len(chk.rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())