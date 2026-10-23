# -*- coding: utf-8 -*-
"""共享节拍/计时工具：绝对截止点定拍 + 有界自旋 + 有界停止。

================================ 0. 一句话结论 ================================

定拍用 ``Pacer``（``time.sleep`` 粗睡 + ≤1.5 ms 自旋收尾 + ``perf_counter`` 绝对网格）；
**不要**再拿 ``threading.Event.wait(timeout)`` 当定时器，它在本机被向上量化到定时器 tick。
停机用 ``Event``，但只让它当**打断器**，不当**定时器**。

================================ 1. 病根（实测，非推断） ================================

本机 Windows 11 26100 / CPython 3.12.9。系统定时器 tick 可被 ``NtQueryTimerResolution`` 直接读到：
``min=0.5000 ms  max=15.6250 ms``，**当前值**受别的进程影响（实测在 1.0000 与 15.6250 之间漂）。

``threading.Event.wait(timeout)`` 的路径是 ``threading.py:637`` → ``Condition.wait:359`` →
``waiter.acquire(True, timeout)`` → ``thread_nt.h`` → ``WaitForSingleObjectEx`` /
``SleepConditionVariableSRW``，两者都**睡在 tick 边界上**，于是超时被**向上取整**到 tick：

    请求 0.50 ms → 实测 14.36 ms      请求 16.00 ms → 实测 30.48 ms
    请求 1.00 ms → 实测 14.13 ms      请求 20.00 ms → 实测 26.39 ms
    请求 2.00 ms → 实测 13.16 ms      请求 31.25 ms → 实测 46.15 ms
    请求 8.00 ms → 实测 22.58 ms      请求 49.70 ms → 实测 58.76 ms
    请求 100.0 ms → 实测 103.16 ms（p99 冲到 115.73 ms，正好一个 tick）

``time.sleep()`` **没有**这个病。CPython 3.12 的 ``pysleep`` 在 Windows 走
``CreateWaitableTimerExW(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION)``，实测：

    请求 49.70 ms → 实测 50.08 ms        请求 100.0 ms → 实测 100.31 ms
    请求 1.00 ms  → 实测  1.48 ms（地板 ~0.5 ms，见下）

**而且 tick 调不调它都这样**——这是本模块最重要的发现之一：强制 tick=15.625 ms 后重测，
``time.sleep`` 的 |err|p99 = 0.658 ms（tick=1 ms 时 0.787 ms），**完全不受 tick 影响**。
受影响的是 ``Event.wait``（|err|p99 从 10.668 → 7.390 ms，两个值都不可用）。

低端地板（同一台机器，n=300）：

    请求   0 us → 实测 0.8 us      请求 500 us → 实测 1039 us
    请求 200 us → 实测 549 us      请求 1000 us → 实测 1477 us

所以**短等待不能靠 sleep**（它比要求至少多 0.4 ms），只能靠自旋收尾
（自旋落点实测：超出截止点 p50 0.60 us / p99 2.40 us / max 5.40 us）。

================================ 2. 定拍原语对照（实测） ================================

绝对截止点定拍，目标 50 ms，n=400。误差 = 实际唤醒 − 该拍的理论截止点，
``p99`` 一律取 |err| 的第 99 百分位。CPU 是**本进程**占用，单线程台架。

tick = 1.0000 ms（本机现役）：

    原语                       |err|p50   |err|p99   max      CPU
    A time.sleep                 0.420      0.787   1.173     0.00%
    B busy-spin                  0.003      0.008   2.578    98.44%
    C sleep+0.5 ms 自旋          0.002      1.082   4.430     0.00%
    C sleep+1.0 ms 自旋          0.002      0.114   4.107     3.91%
    C sleep+1.5 ms 自旋          0.002      0.099   7.824     5.00%
    C sleep+3.0 ms 自旋          0.003      0.040   4.930     6.17%
    D Event.wait                 2.347     10.668  14.593     0.23%
    E sleep + timeBeginPeriod(1) 0.307      0.715   0.956     0.16%
    F Event.wait + tBMP(1)       1.011      2.463   6.511     0.08%
    G sleep+1.5spin + tBMP(1)    0.002      0.981   6.126     0.00%

tick = 15.6250 ms（强制，= Windows 默认态）：

    原语                       |err|p50   |err|p99   max      CPU
    A time.sleep                 0.300      0.658   0.960     0.23%
    B busy-spin                  0.003      0.017   0.601    98.91%
    C sleep+1.5 ms 自旋          0.002      0.017   0.102     0.00%
    D Event.wait                 1.463      7.390  11.448     0.23%
    E sleep + tBMP(1)            0.340      2.665   6.495     0.00%
    F Event.wait + tBMP(1)       1.011      2.989   3.738     0.00%
    G sleep+1.5spin + tBMP(1)    0.002      0.045   3.337     0.08%

**结论 2.1**：``Event.wait`` 无论 tick 调到多少都不适合当定时器：最好情况（tBMP(1)）
也是 |err|p99 = 2.46~2.99 ms，而 G（sleep+自旋，**不动全局状态**）在同一台机器上是 0.045~0.981 ms。
tBMP 帮不上忙，因为 ``Event.wait`` 的唤醒走的是另一条路径，tick 变细只是把量化从 15.6 缩到 1.0，
而 1.0 ms 对 50 ms 预算仍然是 2% 的硬抖动。

**结论 2.2**：``timeBeginPeriod(1)`` 不值得。它是**进程级全局**的：

  * 它让**所有**定时器都失去合并（timer coalescing），不只是我们这条线程。Windows 的
    省电/调度全靠把相邻定时器合并到一个唤醒点上；关掉以后空闲功耗与整机抖动都变差。
  * 实测本机 tick 本来就被别的进程顶在 1.0 ms 过，说明**别人也在这么干**——再加一个
    调用方只会把栈越叠越高（这是引用计数的，见 ``timeEndPeriod`` 必须配对）。
  * 收益只有 0.72 ms → 0.10 ms 的 p99，而 G 用自旋免费拿到了。
  * 反过来，它对 ``time.sleep`` **没有**帮助（见 §1），而 sleep 才是本模块的粗睡原语。

**唯一可以接受 tBMP 的场合**：整条链路都要求「唤醒即精确」且不能用自旋（例如必须在
低功耗区间里跑、CPU 预算卡死到 0）。本仓库没有这种场合。**本模块不调用它。**

**结论 2.3**：自旋预算不是一个"越大越好"的旋钮，有最优区间。扫描（tick=1 ms，n=400）：

    预算      |err|p50   |err|p99   max      占空比(=预算/周期)
    0.00 ms     0.348      0.821    4.808     0.0%
    0.30 ms     0.048      0.517    2.843     0.6%
    0.50 ms     0.002      2.459    7.691     1.0%
    0.80 ms     0.002      0.097    5.712     1.6%
    1.00 ms     0.002      0.016    0.102     2.0%
    1.50 ms     0.002      0.018    0.193     3.0%
    2.00 ms     0.003      0.074    5.446     4.0%
    3.00 ms     0.003      0.035    3.020     6.0%
    5.00 ms     0.003      0.027    7.170    10.0%

曲线在 1.0~1.5 ms 处**收敛到噪声底**（p99 ≤ 0.02 ms），再往上买不到东西，只是多烧 CPU。
0.5 ms 反而更差（p99 2.459 ms）——因为它盖不住 ``time.sleep`` 的唤醒误差：
实测 ``sleep`` 超出量 p50 0.324 / p90 0.536 / p99 1.702 / max 5.924 ms。
预算必须 ≥ 这个 p99，否则"睡醒时已经过了 ``deadline − 预算``"的自旋段长度为 0，等于没自旋。

**所以 ``DEFAULT_SPIN_S = 0.0015``**：≈ sleep 唤醒误差 p99（1.7 ms）的量级、
= 1 个 Windows tick 的 1/10、对 20 Hz 是 3% 占空、对 10 Hz 是 1.5%。

自旋占空是**精确可测**的（用 ``perf_counter`` 夹住自旋段本身），实测：

    预算 0.50 ms → 真正在自旋的时间占墙钟 0.36%
    预算 1.50 ms → 2.33%     ← 默认值；20 Hz 下 ≈ 一个核的 2.3%
    预算 3.00 ms → 5.34%

注意 2.33% < 名义 3.0%：因为 ``sleep`` 常常自己就睡过了 ``deadline − 预算``，自旋段被吃掉。
**这是好事**（自旋是兜底，不是主力），也是为什么"CPU 占用"永远远低于占空比。

================================ 3. 掉拍策略：跳拍对齐，不追赶 ================================

两种策略在 6 s 里跑了同样的负载（周期 50 ms，随机 15% 的拍 body 人为耗时 120 ms）：

    策略                          6 s 内实拍   实际频率    最小拍间隔
    catch-up (deadline += period)     99 次    16.50 Hz     0.0 ms  ← 忙等风暴
    skip-to-alignment                 82 次    13.67 Hz    50.0 ms

**catch-up 的 16.50 Hz 是假象**：它不是"跑得更快"，而是把积压的拍**背靠背**放出来
（最小间隔 0.0 ms 就是证据）。在 ``_perception_loop`` 里这会直接毁掉航位推算——
``DeadReckoner.update(tick, ...)`` 拿的是绝对时刻，两拍同一时刻 ⇒ 积分区间 ≈ 0，
那几帧的位移直接丢；而拍号还在涨，``RateMeter`` 却报出漂亮的 16.5 Hz。
这正是本仓库 ``RateMeter`` docstring 说的"声明值≠生效值"。

**所以选 skip-to-alignment**：``deadline`` 永远落在网格 ``t0 + n*period`` 上，
过载时把 ``n`` 直接推到"第一个严格在未来的格子"，跳过的拍数记进 ``Tick.skipped``。

实现上有个必须区分的阈值，否则会误伤正常调度抖动：

  * ``0 ≤ 迟到 < period``：**照常发这一拍**（迟到但有效）。因为下一拍的等待量是
    ``period − 迟到 > 0``，所以**永远不会出现零长度等待**，忙等风暴在结构上不可能发生。
  * ``迟到 ≥ period``：说明整整一拍丢了，跳到 ``n = floor((now−t0)/period) + 1``。

这条规则有个漂亮的副产品：当 body 稳定耗时 ``1.2×period`` 时，拍间隔会在
1.2p / 0.8p 之间交替，**平均严格等于 p**，频率不掉，且最小间隔 0.8p > 0 仍然安全。
只有 body ≥ 2p 才真正降频（实测 2.5p ⇒ 0.4× 目标频率 + ``skipped`` 计数外显）。

================================ 4. 时钟域（本模块最危险的一条） ================================

``backend/nav_online.py:821`` 的默认是 ``clock: Callable[[], float] = time.monotonic``。
在这台 CPython 3.12.9 上 ``time.monotonic`` 的实现是 **GetTickCount64()**，
``time.get_clock_info("monotonic").resolution`` 自报 0.015625 s，实测读数台阶：

    clock           读数台阶 p50    纯自旋定拍的 |err|p50   |err|p99
    monotonic            16.000 ms       6.000 ms      13.000 ms
    perf_counter          0.0003 ms      0.001 ms       0.043 ms

**6000 倍的差距**。而且 ``timeBeginPeriod(1)`` 修不了它（实测：tBMP(1) 之后
``monotonic`` 的台阶仍是 16.000 ms，因为 GetTickCount64 不受定时器精度影响）。

后果：哪怕 ``Pacer`` 完美唤醒，只要**读时间是 15.6 ms 粒度**，误差就是 ±8 ms 均方根，
一切自旋都白费。所以 ``Pacer`` 默认 ``clock=time.perf_counter``，并且在构造时
**实测一次时钟粒度**（``check_clock``），粒度超过周期的 2% 就写进 ``clock_warning``，
让 ``status()`` 能把它报出来。这与 ``baseline_mismatch`` / ``stereo_ticks`` 是同一套哲学：
凡是"声明的精度"都要有一个"实测的精度"跟它并排。

⚠️ **不要跨域混用**。``monotonic`` 与 ``perf_counter`` 的差值在 2 s 内实测漂了 10.3 ms
（因为 monotonic 自己是量子化的，差值当然跳）。deadline 只能用**同一个** clock 算。

================================ 5. 停机延迟（有界，且可以算） ================================

``Event.wait`` 只有一件事做得好：被 set 时**立刻**醒（实测 p50 0.06 ms / max 0.09 ms，
即使它正在等 1.0 s）。所以分工是：**定时归 sleep，打断归 Event**。
但要让打断能穿透一次长 sleep，就把等待切成 ≤ ``SLICE_S`` 的小片，每片前查一次 stop。

实测打断延迟（在目标线程正睡着的时候 set，n=40）：

    目标正在做            停止通知延迟 p50    p95      max
    Event.wait(1.0 s)        0.06 ms       0.08 ms   0.09 ms
    time.sleep(36 ms) 单块   20.96 ms     33.21 ms  33.27 ms
    4 × sleep(9.5 ms)        4.21 ms      8.33 ms   8.34 ms
    2 × sleep(19.5 ms)      12.30 ms     16.79 ms  16.81 ms

``SLICE_S = 0.05``（= 20 Hz 的一整拍）。于是：

    **停机延迟上界 = SLICE_S + 单次 sleep 超时上界 ≈ 50 + 1.2 = 51 ms**

典型值 ≈ 半个片长 = 25 ms。对比 ``nav_online.py:1063-1071`` 的 ``join(timeout=2.0)``，
51 ms 是它的 1/40，而且**四条线程的最坏情况是并行发生的**，不是相加。
片长比 0.5 ms 的 sleep 地板大两个数量级，所以切碎不会引入抖动（上面 9.5 ms 那行就是证据）。

⚠️ 已知代价：``Pacer`` 的停机延迟是 ~25 ms 量级，**不是** ``Event.wait`` 的 0.06 ms。
这是为"定时精度"付的价。**只在真的需要定拍的地方付**——见 §6 的分类表。

================================ 6. 谁该用什么（本文档被 backend 引用时的口径） ================================

    call site                                  类  推荐原语                期望
    nav_online._perception_loop  pose_hz      1   Pacer(hz=20)            20.000 Hz, |err|p99 0.02 ms
    nav_online._control_loop     control_hz   1   Pacer(hz=10)            10.000 Hz, |err|p99 0.02 ms
    nav_online  双目门 next_stereo            2   Pacer(hz=10) + 绝对累加  10.00 Hz（见 §7）
    navigator._run (已经是绝对 deadline)       2   保持，但把 wait 换成    tick 精度 0.02 ms
                                                  sleep_until()
    nav_online._stereo_worker_loop  min_period 3   直接 time.sleep(rest)   不再被抬到 31 ms
    nav_online._stereo_q.get(0.1)              3   不动（Event 路径）      唤醒 0.11 ms 已足够
    nav_online._map_event.wait(0.3)            3   不动                    事件驱动
    world_state._changed.wait(remaining)        3   不动                    事件驱动
    vision._stop.wait(interval_s)              3   不动                    采集上限，不是定拍
    service / obs_capture / nav_xsession /      4   不动（Event.wait 长超时） 15.6 ms 抖动对 1 s 是 1.5%
      client 的停机等待

一句话：**只有 CLASS 1/2 需要 ``Pacer``**。CLASS 3/4 的等待语义是"睡到我被叫醒"，
不是"在某个时刻醒"，换过来只会把停机延迟从 0.06 ms 恶化到 25 ms，没有任何收益。

================================ 7. 双目门的悬崖（关键，单独说） ================================

现状 ``nav_online.py:1344``：``if tick >= next_stereo: next_stereo = tick + c.stereo_period_s``。
右边是 ``tick``（**实际节拍**）而不是 ``next_stereo``（**理论节拍**），所以误差会累加，
而且是**向上量化到整数个 pose tick**：

    位姿 tick 实测 p50 = 46.889 ms（不是 50）
    0.1 / 0.046889 = 2.13 ⇒ 门被抬到第 3 个 tick = 140.7 ms ⇒ 7.11 Hz
    0.1 / 0.050    = 2.00 ⇒ 正好第 2 个 tick ⇒ 10.00 Hz

悬崖的形状：``floor(0.1 / tick)`` 在 ``tick = 50.000 ms`` 处从 2 掉到 1…不，是反过来：
``tick`` 略小于 50 时 ``0.1/tick > 2``，要等第 3 拍；``tick`` 略大于 50 时 ``0.1/tick < 2``，
第 2 拍就够。**50.005 ms → 10.00 Hz，49.995 ms → 6.67 Hz**，中间没有任何过渡。
这就是"相对累加"在周期比值不是整数时的典型崩溃：频率由**抖动方向**决定，不由配置决定。

**绝对累加**为什么对：``next_stereo += 0.1`` 让 ``next_stereo`` 永远落在
``t0 + k×0.1`` 的网格上，与 ``tick`` 的实际值无关。判定 ``tick >= next_stereo`` 时，
量化误差只影响**这一次落在哪个 tick 上**（±1 tick），不会滚进下一次。
长期频率 = 1/0.1 = 10.00 Hz，**误差不累积**（这是数论事实：网格不依赖被测量）。

代价：网格与 tick 之比不是整数时，会有**交替**（例如 tick=50.7 ms，0.1/0.0507=1.97，
则双目间隔在 2×50.7=101.4 ms 与 3×50.7=152.1 ms 之间交替）。**这是对的**：
平均严格 100 ms。要消除交替只能提高 tick 精度（§4 的 ``perf_counter``），不能靠改门。

掉满一拍时怎么办——**立刻发，不补发，并且外显**：

  * 不补发（不要连发两帧双目）：SGBM 一帧 ~21 ms，连发两帧会把位姿线程的 50 ms 预算吃穿，
    而第二帧与第一帧的基线几乎重合，信息量≈0。这跟 ``_stereo_q`` 的 ``maxsize=1`` 丢最新是同一个取舍。
  * 立刻发：迟到的一帧比没有帧好（``stereo_stale_s`` 超了控制线程就要 ``block="stereo_stale"`` 停车）。
  * 外显：把 ``skipped`` 累加进 ``RateMeter`` 的旁路统计，否则"频率掉了"又只能靠事后反推。

岔开一句：``next_stereo += period`` 的**浮点**漂移要不要担心？实测（period=0.1）：

    n=10         累加 0.99999999999999989   乘法 1        差 -1.11e-16 s
    n=1e3        累加 99.999999999998593    乘法 100      差 -1.41e-12 s
    n=1e5        累加 10000.000000018848    乘法 10000    差 +1.88e-08 s
    n=1e6        累加 100000.00000133288    乘法 100000   差 +1.33e-06 s   （= 27.8 小时连续运行）

27.8 小时后偏差 1.33 us，是 60 ms tick 的 1/45015。300 次随机抖动试验里，加法与乘法版本
的触发次数**一次都没差过**（抖动 2 ms 完全淹没 1e-16）。所以两者等价，**用加法**即可；
真要绝对干净就在 ``reset()`` 时记 ``t0``、用 ``t0 + k*period`` 现算（``Pacer`` 就是这么做的）。

================================ 8. 并发与线程安全 ================================

每个 ``Pacer`` 实例自带全部可变状态（``_n`` / ``_t0`` / ``_late`` / ``_rate``），
**没有模块级可变状态、没有全局锁、不调 timeBeginPeriod、不装信号处理器**。
一条线程一个实例即可并发；实例之间零共享。
``RateMeter`` 里那把锁只保护自己的 ``deque``，不跨实例，也不会被 ``status()`` 长时间持有。

⚠️ 一个 ``Pacer`` 实例**只能被一条线程用**（它是有状态的迭代器）。要 N 条线程就建 N 个。

    # 正确
    self._pose_pacer = Pacer(hz=c.pose_hz, stop=self._stop, name="pose")
    # 错误：两条线程共用一个 Pacer ⇒ 拍号互相踩，频率翻倍或乱跳

================================ 9. 本模块**不**解决的事 ================================

  * **GC 停顿**：CPython 的循环 GC 一次可以停几 ms，自旋也救不了。实测 max 抖动
    7.8 ms 的那几拍基本都能对上 ``gc`` 的节奏。缓解见 ``Docs``，不在本模块范围。
  * **GIL 争用**：本模块的自旋会**持有 GIL**（``perf_counter`` 不释放）。1.5 ms/50 ms = 3%，
    且 ``sys.setswitchinterval()`` 实测 5 ms，所以一次自旋通常不会触发强制切换 ⇒
    其它纯 Python 线程最多被挡 1.5 ms。对 OpenCV（计算时释放 GIL）无影响，
    对 ``torch.compile`` 的 Dynamo guard 有影响（见 ``nav_online`` 里 LAS2 那段 80× 的教训）。
  * **优先级反转 / 系统整体过载**：Pacer 不设线程优先级，``OSError`` 之外的失败一律不吞。


================================ 10. 接线样例（给 Lead 抄） ================================

这里只给**形状**，真正的改动由 Lead 在 ``backend/nav_online.py`` 里做。

**(a) 感知循环** —— 现在（``nav_online.py:1291-1293, 1417``）::

    while not self._stop.is_set():
        tick = self._clock()            # ← 默认 time.monotonic = 15.6 ms 粒度！
        ...
        self._stop.wait(max(0.0, period - (self._clock() - tick)))

改成::

    self._pose = Pacer(hz=c.pose_hz, stop=self._stop, clock=time.perf_counter,
                       name="pose")          # 在 _reset() 里建，stop()/start() 会重建
    self._pose.reset()
    for tick in self._pose.ticks():
        t = tick.actual                   # 这一拍的时刻，拿去喂 DR/KeyframePolicy
        ...
    # 循环体里删掉所有 self._stop.wait(...) —— 定拍权已经交给 Pacer

⚠️ 三件事必须同时做，否则等于没改：
  1. ``tick = self._clock()`` 换成 ``tick.actual``（**别**再读一次 ``self._clock()``，
     否则拍号与 DR 积分区间差几十 us）；
  2. ``self._clock`` 必须从 ``time.monotonic`` 换成 ``time.perf_counter``（§4）；
  3. ``_reset()`` 里 ``Pacer(...)`` 要重建（``t0`` 属于会话）。

**(b) 双目门** —— 现在（``nav_online.py:1344-1345``）::

    if tick >= next_stereo:
        next_stereo = tick + c.stereo_period_s      # ← 相对累加，7.94 Hz 的根因

改成::

    # __init__/_reset 里: self._stereo_gate = Gate(period_s=c.stereo_period_s, name="stereo")
    # _perception_loop 开头: self._stereo_gate.reset(first_tick)
    if self._stereo_gate.due(tick):
        ...                                        # 不补发：一次 tick 至多一帧双目

**(c) 控制循环** —— 同样是 ``Pacer(hz=c.control_hz, ...)``，把 ``:1638`` 那句删掉。

**(d) 双目工作线程的最小间隔**（``:1498-1500``）::

    rest = c.stereo_min_period_s - (time.perf_counter() - t0)
    if rest > 0:
        self._stop.wait(rest)          # 0.02 s 被 Event.wait 抬到 30.95 ms（实测）

改成::

    if rest > 0:
        sleep_until(time.perf_counter() + rest, stop=self._stop, spin_s=0.0005)

这里只需要"别睡过头"，不需要精确落点，所以 ``spin_s=0.5 ms``（够盖住 sleep 的 p50 0.32 ms）。

**(e) 不要动的**：``_stereo_q.get(0.1)`` / ``_map_event.wait(0.3)`` / ``_stop.wait(1.0)``
/ ``world_state._changed.wait`` / ``vision._stop.wait`` / service|obs_capture|client 的停机等待。
它们要的是"睡到被叫醒"，``Event.wait`` 已经做到 0.06 ms 唤醒，换成 ``Pacer`` 只会把
停机延迟从 0.06 ms 恶化到 ~29 ms。

"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable, Iterator, NamedTuple

__all__ = [
    "Pacer", "Gate", "RateMeter", "Tick", "clock_quantum", "check_clock",
    "sleep_until", "DEFAULT_SPIN_S", "SLICE_S", "HIGH_RES_CLOCK",
]

# 高精度时钟。``time.monotonic`` 在 CPython 3.12/Windows 上是 GetTickCount64，
# 粒度 15.625 ms —— 见 §4，那是 6000 倍的误差源。这里显式命名，供 grep。
HIGH_RES_CLOCK = time.perf_counter

# 自旋预算（秒）。1.5 ms 的推导见 §2 结论 2.3：
#   * 必须 ≥ time.sleep 唤醒误差 p99（实测 1.702 ms）的量级，否则自旋段长度为 0；
#   * 不必超过 1.5 ms（1.0~1.5 ms 处 p99 已收敛到 0.02 ms，再往上只烧 CPU）；
#   * = 1 个 Windows tick 的 1/10，20 Hz 下占空 3%（实测真自旋 2.33%）。
DEFAULT_SPIN_S = 0.0015

# 分段等待片长（秒）。= 20 Hz 的一整拍。停机延迟上界 = SLICE_S + sleep 超时上界 ≈ 51 ms。
# 比 time.sleep 地板（~0.5 ms）大两个数量级，切碎不引入额外抖动（实测 4×9.5 ms 片 ⇒ 4.21 ms）。
SLICE_S = 0.05

# 时钟粒度检测的采样数。2000 次 perf_counter 约 1 ms，一次性成本，换掉一个静默灾难。
_CLOCK_SAMPLES = 2000

_OK = 0
_STOPPED = 1

def clock_quantum(clock: Callable[[], float] = HIGH_RES_CLOCK,
                  *, samples: int = _CLOCK_SAMPLES) -> float | None:
    """实测 ``clock`` 的读数台阶（秒）。返回观察到的**最小非零步进**的中位数。

    为什么要这么个函数：``time.get_clock_info("monotonic").resolution`` 自报 0.015625 s，
    但那是**声明值**；本仓库的教训是声明值必须和实测值并排（``RateMeter`` /
    ``baseline_m`` 同一套）。这里实测的是"两次读数之间最小能分辨多少"。

    注意：这不是分辨率的严格定义（读数可能在粗台阶内部有更细的量化），但对"能不能用来定拍"
    这个用途足够了——粗台阶会直接暴露成 |err| 的下限（实测 montonic 6.0 ms vs perf_counter 0.001 ms）。

    采样成本：2000 次调用约 1 ms（perf_counter 单次 ~0.3 us）。**不要**放进热循环。
    """
    prev = clock()
    steps: list[float] = []
    for _ in range(max(16, int(samples))):
        c = clock()
        d = c - prev
        if d > 0.0:
            steps.append(d)
            prev = c
    if not steps:
        return None                    # 采样窗口内一次都没前进 ⇒ 粒度比窗口还粗
    steps.sort()
    return steps[len(steps) // 2]


def check_clock(clock: Callable[[], float] = HIGH_RES_CLOCK, period_s: float = 0.05,
                *, samples: int = _CLOCK_SAMPLES) -> str | None:
    """时钟够不够细？不够就返回一句人话（给 ``status()`` 直接外显），够就返回 None。

    阈值 = 周期的 2%。20 Hz（50 ms）⇒ 1 ms。perf_counter（0.3 us）远低于它；
    monotonic（15.625 ms）是它的 15 倍，会被立刻抓住。
    """
    q = clock_quantum(clock, samples=samples)
    if q is None:
        return ("clock_quantum_unbounded: 采样窗口内 clock 一次都没前进；"
                "这条 clock 不能用来定拍（≤ %.2f Hz）" % (1.0 / max(period_s, 1e-9)))
    budget = max(1e-6, float(period_s) * 0.02)
    if q > budget:
        return ("clock_too_coarse: 实测台阶 %.4f ms > 预算 %.4f ms（周期的 2%%）；"
                "请用 time.perf_counter，不要用 time.monotonic"
                % (q * 1e3, budget * 1e3))
    return None


class Tick(NamedTuple):
    """一拍的结果。``stopped=True`` 时 ``actual`` 是**放弃等待的时刻**，不是截止点。

    ``late`` = ``actual - deadline``（秒，可为负 = 提前醒，正常应为微秒级正值）。
    ``skipped`` = 这一拍之前被跳过的整拍数（>0 说明 body 超了一个周期，见 §3）。
    """

    index: int
    deadline: float
    actual: float
    late: float
    skipped: int
    stopped: bool


class RateMeter:
    """最近 N 次事件的**实际**频率。用来把"配置的频率"和"跑出来的频率"分开报。

    ⚠️ 这个类**故意**只做加减，不做补偿：``Pacer`` 的绝对网格已经把频率钉死了，
    meter 只负责**证伪**（如果它报出来的不是 20.00，那就是真出事了，而不是被"修正"掉了）。

    线程安全：``deque.append`` 在 CPython 下是原子的；``hz()`` 快照一次再做算术，
    最多拿到略微撕裂的窗口，不会抛异常（与 ``nav_online.RateMeter`` 同一取舍）。
    """

    __slots__ = ("_t",)

    def __init__(self, n: int = 64) -> None:
        self._t: deque[float] = deque(maxlen=max(2, int(n)))

    def mark(self, t: float) -> None:
        self._t.append(t)

    def samples(self) -> int:
        return len(self._t)

    def reset(self) -> None:
        self._t.clear()

    def hz(self) -> float | None:
        """窗口内 (n−1)/跨度。样本 < 2 返回 None（**不是** 0 —— 那会被读成卡死）。"""
        t = self._t
        if len(t) < 2:
            return None
        span = t[-1] - t[0]
        return None if span <= 0.0 else (len(t) - 1) / span

    def gap_p50_s(self) -> float | None:
        """间隔中位。**这个数才看得出量化台阶**：它应当 ≈ period，不应是整数个 tick。"""
        try:
            vals = list(self._t)
        except RuntimeError:           # 另一线程正在改写 deque
            return None
        if len(vals) < 3:
            return None
        gaps = sorted(vals[i + 1] - vals[i] for i in range(len(vals) - 1))
        return gaps[len(gaps) // 2]


class Pacer:
    """绝对截止点的定拍器。**一条线程一个实例**（有状态）。

    用法::

        pacer = Pacer(hz=cfg.pose_hz, stop=self._stop, clock=self._clock, name="pose")
        while not pacer.stopped:
            tick = pacer.wait()          # tick.actual 就是这一拍的"当下"
            if tick.stopped:
                break
            ...                          # body 用 tick.actual，不要自己再读一次时钟

    或 ``for tick in pacer.ticks(): ...``（更紧凑，但 body 里要自己判 ``stopped``）。

    参数：
      * ``hz`` / ``period_s``：二选一。周期必须有限且 > 0。
      * ``clock``：**必须是单调且高精度的**。默认 ``perf_counter``；传 ``time.monotonic``
        会被 ``check_clock`` 抓住并外显（见 §4），不会**报错**——因为可能有测试注入假时钟。
      * ``stop``：可选 ``Event``。set 后 ``wait()`` 在 ≤ ``SLICE_S + 1.2 ms`` 内返回
        ``Tick(stopped=True)``。传 ``None`` 表示不可打断（只用于确定会在有限拍内结束的场合）。
      * ``spin_s``：自旋预算。``None`` ⇒ 用 ``DEFAULT_SPIN_S``；``0`` ⇒ 纯 sleep（不准但 0 CPU）。
      * ``slice_s``：分段片长，决定停机延迟上界。
      * ``phase_s``：首拍相对 ``reset()`` 时刻的偏移，用来让多条线程错开相位
        （例如双目线程不要和位姿线程同时醒）。**只影响首拍**，之后是严格网格。
      * ``name``：只用于 ``telemetry()`` / ``status()``。
    """

    __slots__ = ("period", "name", "_clock", "_stop", "_slice", "_spin", "_phase",
                 "_t0", "_n", "_skipped", "_spin_s", "_awake_s", "_late", "_rate",
                 "_clock_warning")

    def __init__(self, hz: float | None = None, *, period_s: float | None = None,
                 clock: Callable[[], float] = HIGH_RES_CLOCK,
                 stop: "threading.Event | None" = None,
                 spin_s: float | None = DEFAULT_SPIN_S,
                 slice_s: float = SLICE_S,
                 phase_s: float = 0.0,
                 name: str = "") -> None:
        if (hz is None) == (period_s is None):
            raise TypeError("Pacer: 传 hz 或 period_s 二者之一")
        p = (1.0 / float(hz)) if hz is not None else float(period_s)
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError("Pacer: period 必须有限且 > 0，收到 %r" % (p,))
        self.period = p
        self.name = name
        self._clock = clock
        self._stop = stop
        self._slice = max(0.0, float(slice_s))
        self._spin = self._clamp_spin(spin_s, p)
        self._phase = float(phase_s)
        self._t0: float | None = None
        self._n = 0
        self._skipped = 0
        self._spin_s = 0.0
        self._awake_s = 0.0
        self._late: deque[float] = deque(maxlen=256)
        self._rate = RateMeter(64)
        # 构造时**实测一次**时钟粒度。见 §4：这是唯一能在开工前抓住 monotonic 的地方。
        self._clock_warning = check_clock(clock, p)

    @staticmethod
    def _clamp_spin(spin_s: float | None, period: float) -> float:
        """自旋预算的三重封顶。单看 p99 曲线 3 ms 更好，但这两条封顶是给别人让路的：

          * ``period × 10%``：20 Hz ⇒ 1.5 ms（= 默认值，正好卡在曲线上）。但 100 Hz 时
            只剩 1 ms —— 高频率本来就更需要精度，所以不能让它无脑占满；
            反过来 1 Hz 时 10% = 100 ms，必须被下面那条压住。
          * 4 ms 绝对上限：超过 4 ms 就**不再叫"自旋收尾"而叫"忙等"**了，此时宁可
            让 ``time.sleep`` 直接决定（|err|p99 ≈ 0.8 ms 也够用）。8 个 tick 的忙等会
            明显抢走同进程其它 Python 线程的 GIL 时间片。
        """
        if spin_s is None:
            spin_s = DEFAULT_SPIN_S   # None = "用默认值"；要关掉自旋请显式传 0
        s = max(0.0, float(spin_s))
        return min(s, period * 0.10, 0.004)

    # ---- 生命周期 ----

    def reset(self, t0: float | None = None) -> None:
        """重新对齐网格。``t0=None`` ⇒ 以当前时刻（+``phase_s``）为第 0 拍。

        ``nav_online.py`` 的 ``stop()``/``start()`` 会换会话，**必须**跟着 ``reset()``，
        否则 ``t0`` 还是上一个会话的，第一条 ``wait()`` 会判定"迟到了一万拍"。
        """
        self._t0 = (self._clock() + self._phase) if t0 is None else float(t0)
        self._n = 0
        self._skipped = 0
        self._spin_s = 0.0
        self._awake_s = 0.0
        self._late.clear()
        self._rate.reset()

    # ---- 内部：等到 deadline ----

    def _wait_until(self, deadline: float) -> int:
        """睡/自旋到 ``deadline``。返回 ``_OK`` 或 ``_STOPPED``。

        结构 = 粗睡循环 + 自旋收尾：

            while 剩余 > 预算:
                sleep(min(剩余 − 预算, SLICE_S))    # 粗睡，**不受 tick 量化影响**
                if stop 被 set: return _STOPPED      # 打断，上界 = SLICE_S + sleep 超时
            自旋到 deadline                            # 精修，落点 p99 2.4 us

        两个细节值得记：
          1. ``sleep(剩余 − 预算)`` 而不是 ``sleep(剩余)``：把唤醒误差留给自旋吸收。
          2. 片长封顶 ``SLICE_S`` 是**停机延迟**的唯一来源；没有它，一次 0.3 s 的
             ``Event.wait`` 就能让 ``stop()`` 等 0.3 s。
        """
        clock = self._clock
        stop = self._stop
        spin = self._spin
        slice_s = self._slice
        while True:
            if stop is not None and stop.is_set():
                return _STOPPED
            rem = deadline - clock()
            if rem <= spin:
                break                              # 交给自旋收尾
            nap = rem - spin
            if nap > slice_s:
                nap = slice_s
            time.sleep(nap)
        t_spin = clock()
        if t_spin >= deadline:
            return _OK                             # 睡过头了，自旋段为 0（不是错误，见 §2）
        while clock() < deadline:
            pass
        self._spin_s += clock() - t_spin
        return _OK

    # ---- 公开：取一拍 ----

    def wait(self) -> Tick:
        """阻塞到下一拍的截止点，返回这一拍。**唯一的阻塞出口**。

        网格：``deadline = t0 + n × period``。``n`` 是**绝对拍号**，不是累加器 ——
        每一拍的误差都不会带进下一拍（§7 的绝对累加，这里是时间轴上的同一件事）。
        """
        if self._t0 is None:
            self.reset()
        t0 = self._t0
        assert t0 is not None
        period = self.period
        clock = self._clock
        n = self._n
        deadline = t0 + n * period
        now = clock()
        skipped = 0
        # 掉拍处理（§3）：只有**整拍**级别的落后才跳，迟到但不足一拍照常发。
        # 阈值是 period 而不是 0，正是为了避免"迟到 1 us ⇒ 等待量 0 ⇒ 忙等风暴"。
        if now >= deadline + period:
            k = int(math.floor((now - t0) / period)) + 1
            skipped = k - n
            n = k
            deadline = t0 + n * period
            self._skipped += skipped
        t_start = clock()
        status = self._wait_until(deadline)
        actual = clock()
        late = actual - deadline
        self._n = n + 1
        if status == _STOPPED:
            return Tick(n, deadline, actual, late, skipped, True)
        self._late.append(late)
        self._awake_s += actual - t_start
        self._rate.mark(actual)
        return Tick(n, deadline, actual, late, skipped, False)

    def ticks(self) -> Iterator[Tick]:
        """``for tick in pacer.ticks():``。``stopped`` 时直接结束，不 yield 那一拍。"""
        while True:
            t = self.wait()
            if t.stopped:
                return
            yield t

    def advance(self, t: float) -> Tick:
        """把网格**锚定到外部给定的时刻**（例如感知循环里已经拿到的 ``tick``）。

        为什么需要：``nav_online._perception_loop`` 的 ``tick = self._clock()`` 同时被
        ``DeadReckoner.update`` / ``KeyframePolicy.decide`` / ``RateMeter`` 消费。
        如果 ``Pacer`` 自己另读一次时钟，body 里的时刻和 ``Tick.actual`` 会差几十 us，
        于是"拍号"和"DR 积分区间"对不上。用 ``advance()`` 让**外部时刻**当这一拍。

        与 ``wait()`` 的区别：``advance(t)`` **不睡**，只推进网格并结算统计。
        """
        if self._t0 is None:
            self.reset(t)
        t0 = self._t0
        assert t0 is not None
        period = self.period
        n = self._n
        deadline = t0 + n * period
        skipped = 0
        if t >= deadline + period:
            k = int(math.floor((t - t0) / period)) + 1
            skipped = k - n
            n = k
            deadline = t0 + n * period
            self._skipped += skipped
        self._n = n + 1
        late = t - deadline
        self._late.append(late)
        self._rate.mark(t)
        return Tick(n, deadline, t, late, skipped, False)

    @property
    def stopped(self) -> bool:
        return self._stop is not None and self._stop.is_set()

    def next_deadline(self) -> float:
        """下一个截止点（不推进）。给"剩余时间"类计算用，例如双目门。"""
        if self._t0 is None:
            self.reset()
        assert self._t0 is not None
        return self._t0 + self._n * self.period

    # ---- 诊断 ----

    def telemetry(self) -> dict:
        """给 ``status()["rates"]`` 用。字段命名与 ``nav_online.RateMeter`` 保持同族，
        但这里**多报三项实测**：``late_p99_ms``（抖动）、``skipped_total``（掉拍）、
        ``spin_frac``（真正在自旋的比例 —— 用来证明"没有 busy-spin"）。
        """
        late = sorted(self._late)
        n = len(late)

        def q(x: float) -> float | None:
            return None if n == 0 else late[min(n - 1, int(n * x))] * 1e3

        hz = self._rate.hz()
        return {
            "name": self.name,
            "target_hz": round(1.0 / self.period, 3),
            "achieved_hz": None if hz is None else round(hz, 3),
            "period_ms": round(self.period * 1e3, 3),
            "spin_budget_ms": round(self._spin * 1e3, 3),
            "slice_ms": round(self._slice * 1e3, 3),
            "ticks": self._n,
            "skipped_total": self._skipped,
            "samples": n,
            "late_p50_ms": q(0.50),
            "late_p99_ms": q(0.99),
            "late_max_ms": None if n == 0 else late[-1] * 1e3,
            # 真正自旋的时间 / 真正醒着的时间。这是"没有 busy-spin"的**证据**，不是估计。
            "spin_frac": (None if self._awake_s <= 0.0 else self._spin_s / self._awake_s),
            "clock_warning": self._clock_warning,
        }


class Gate:
    """**不是**定时器：一个"距离下一个网格点还有多久"的门（双目门 / trail 门 / 关键帧门）。

    为什么单独给一个类而不是让调用方自己写 ``if tick >= next: next = tick + period``：
    后者是本仓库实测过的 bug（``nav_online.py:1344``，7.94 Hz vs 目标 10 Hz）。
    把它封装掉，就没人能再写错"相对累加"了。

    用法（在已有的位姿循环里，**不额外等待**）::

        gate = Gate(period_s=c.stereo_period_s)
        gate.reset(tick0)
        ...
        if gate.due(tick):          # tick = 本拍的实际时刻
            do_stereo()

    ``due(t)`` 内部做三件事：
      1. ``t < next`` ⇒ 返回 False（**不动** ``next``，这是绝对累加的关键）；
      2. ``t >= next`` ⇒ ``while next <= t: next += period``（一次性推进过所有过期格子，
         但**只让调用方发一次**，不补发）；
      3. 记 ``skipped``（跳过了几个格子）和 ``fires``（真正发了几次），供 ``telemetry()``。

    与 ``Pacer`` 的区别：``Pacer`` 会**阻塞**，``Gate`` 只回答"现在该不该做"。
    位姿循环已经有 ``Pacer`` 在管节拍了，双目/trend 这些"一个节拍里做几件事"的
    只能用 ``Gate``——否则一条线程里两个阻塞定时器会互相打架。
    """

    __slots__ = ("period", "name", "_clock", "_next", "_t0", "_fires", "_skipped",
                 "_last_fire")

    def __init__(self, hz: float | None = None, *, period_s: float | None = None,
                 clock: Callable[[], float] = HIGH_RES_CLOCK,
                 name: str = "") -> None:
        if (period_s is None) and (hz is None):
            raise TypeError("Gate: 传 hz 或 period_s 二者之一")
        p = (1.0 / float(hz)) if period_s is None else float(period_s)
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError("Gate: period 必须有限且 > 0，收到 %r" % (p,))
        self.period = p
        self.name = name
        self._clock = clock
        self._next = -math.inf
        self._t0: float | None = None
        self._fires = 0
        self._skipped = 0
        self._last_fire: float | None = None

    def reset(self, t0: float | None = None) -> None:
        """``t0=None`` ⇒ 现在；第一拍**立即到期**（与位姿循环"第一帧就采双目"对齐）。"""
        base = self._clock() if t0 is None else float(t0)
        self._t0 = base
        self._next = base
        self._fires = 0
        self._skipped = 0
        self._last_fire = None

    def due(self, t: float | None = None) -> bool:
        """这一拍该不该做。**幂等安全**：同一时刻连调两次，第二次返回 False。"""
        if self._next == -math.inf:
            self.reset(t)
        now = self._clock() if t is None else float(t)
        if now < self._next:
            return False
        # 绝对累加：next 只由它自己推进，永远不看 now。这正是悬崖的解药。
        crossed = 0
        while self._next <= now:
            self._next += self.period
            crossed += 1
        if crossed > 1:
            self._skipped += crossed - 1     # 只记"本该发但没发"的格子
        self._fires += 1
        self._last_fire = now
        return True

    def remaining(self, t: float | None = None) -> float:
        """距离下一次到期还有多久（秒，可为负 = 已经迟到）。"""
        if self._next == -math.inf:
            return 0.0
        now = self._clock() if t is None else float(t)
        return self._next - now

    def telemetry(self) -> dict:
        return {
            "name": self.name,
            "target_hz": round(1.0 / self.period, 3),
            "period_ms": round(self.period * 1e3, 3),
            "fires": self._fires,
            "skipped_total": self._skipped,
            # 每次触发实际跨过的格子数。1.0 = 每格都发；>1 说明在丢格（= 上游太慢）。
            "grids_per_fire": (None if self._fires == 0 else
                               round((self._fires + self._skipped) / self._fires, 3)),
        }

def sleep_until(deadline: float, *,
                clock: Callable[[], float] = HIGH_RES_CLOCK,
                stop: "threading.Event | None" = None,
                spin_s: float = DEFAULT_SPIN_S,
                slice_s: float = SLICE_S) -> bool:
    """一次性"睡到这个绝对时刻"。返回 False 表示被 ``stop`` 打断。

    给**不是定拍、但需要一个确定时刻**的场合用。典型：``navigator.py:1773`` 的
    ``self._stop_event.wait(min(period, deadline - now))`` —— 它已经是绝对截止点了，
    只是原语选错（Event.wait 被 tick 量化），换成这个即可，网格逻辑不用动。

    注意 ``spin_s`` 这里**不按周期封顶**（没有周期），只受 4 ms 绝对上限约束：
    一次性等待的调用方既然显式传了预算，就应该得到它。
    """
    clock = clock
    budget = min(max(0.0, float(spin_s)), 0.004)
    slice_s = max(0.0, float(slice_s))
    deadline = float(deadline)
    while True:
        if stop is not None and stop.is_set():
            return False
        rem = deadline - clock()
        if rem <= budget:
            break
        nap = rem - budget
        if nap > slice_s:
            nap = slice_s
        time.sleep(nap)
    now = clock()
    if now < deadline:
        while clock() < deadline:
            pass
    return True