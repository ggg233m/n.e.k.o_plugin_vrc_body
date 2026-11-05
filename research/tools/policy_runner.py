# -*- coding: utf-8 -*-
"""2 Hz policy 运行环：抓帧 → 0.2 s 前帧 → PolicyRuntime.act → 动作块回调。

设计成后端无关，三个可调用对象由调用方注入：
  * ``frame_source()``  → 当前左目 RGB（1080×606，numpy uint8）；
  * ``wp_provider()``   → 兼容旧模型的局部引导点；生产链路优先使用
    ``guidance_provider()``，它把 A* 先验作为可丢弃的软引导输入；
  * ``body_provider()`` → 当前本体速度 ``(vx, vz)`` m/s（直接读 OSC 最新值）。
``prev_source()`` 可选：返回「距今 0.2 s」那一帧 RGB。训练侧两帧间距就是 0.2 s
（唯一真源 policy_histframe.HISTORY_FRAME_S），
但 1 Hz 的 runner 自己只能采到 1 s 间隔（超过 ``_prev`` 的 0.15 s 容差），此时
不给 ``prev_source`` 就会把当前帧复制一份当历史。帧流在 7.9–10 Hz 的调用方
（navmesh）应提供这一路。
``on_block(block, obs_t, meta)`` 由控制器消费（§9.1 的 10 Hz 执行环）；默认 dry-run 只记录。

安全：单拍耗时超过 ``stale_s``（默认 0.8 s，2 Hz 的丢弃阈值）就输出全零块并标 ``stale``。
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

from policy_histframe import HISTORY_FRAME_S


class PolicyRunner:
    def __init__(self, runtime, frame_source, wp_provider=None, body_provider=None,
                 on_block=None, hz: float = 2.0, dry_run: bool = True,
                 stale_s: float = 0.8, log_path: str | None = None,
                 prev_source=None, captured_at_source=None, path_source=None,
                 mode_provider=None, guidance_provider=None, prompt_provider=None):
        self.rt = runtime
        self.frame_source = frame_source
        # 帧的**采集时刻**（perf_counter 域）。给了就用它当 obs_t，不给则退回本拍开始时刻。
        #
        # 为什么必须有这一条（2026-11-02，剖析 §15.19/§15.20 之后的相位诊断）：
        # obs_t 原本在**推理之前**打戳（见 _loop 的 t0），而 PolicyExecutor 按
        # k = int(age / 0.1) 在块里取步（policy_executor.py:121）。于是块到手时
        # age 已经等于整段推理延迟 L，k 直接从 L/0.1 起步 —— **块的前几步永远不执行**。
        #
        # 生产配置 L≈0.35 s ⇒ 只执行 idx 4-8；而 §15.20 抓到的真实块形在那一段正好是
        # 模型的「减速-停」负速谷底（整块均值 0.274 m/s，执行窗只有 0.042）。约 85% 的
        # 前进量被整段丢掉。**推理越快窗越靠前，会落进更深的谷底** —— 这就是为什么
        # 把延迟从 413.8 压到 128.9 ms（2.94×）之后顿挫并没有消失。
        #
        # 用采集时刻当 obs_t 不会让块「变新」（图像本来就是那时拍的，这个信息是真的），
        # 它只是**让 age 如实反映「画面有多旧」**，从而让执行步索引与模型自己设想的
        # 时间轴对齐。丢块判定也随之变得诚实：stale 现在真的在测「我拿着一张多老的图」。
        self.captured_at_source = captured_at_source
        # 路径形状供给（可选）。给 k 个机体系点让模型看见弯；None = 退回单点 prompt。
        # 见 policy_runtime.PROMPT_PATH 的注释：启用会偏离训练分布，只在 A/B 时开。
        self.path_source = path_source
        # 导航状态供给：只用于停止决策，**不进记忆卡**（卡 v2 无 [mode]）。
        self.mode_provider = mode_provider
        self.prev_source = prev_source
        self.wp_provider = wp_provider
        self.body_provider = body_provider
        # 新链路的唯一导航输入口。它只提供软引导快照，不拥有动作出口。
        self.guidance_provider = guidance_provider
        # 场次级任务提示；只在每拍读取当前值，不让调用方逐帧遥控动作。
        self.prompt_provider = prompt_provider
        self.on_block = on_block
        self.period = 1.0 / float(hz)
        self.dry_run = bool(dry_run)
        self.stale_s = float(stale_s)
        self.buf: deque[tuple[float, object]] = deque(maxlen=8)
        self._stop = threading.Event()
        self._th = None
        self.log = Path(log_path).open("a", encoding="utf-8") if log_path else None

    # ------------------------------------------------------------ 帧历史
    def _push(self, t: float, frame) -> None:
        self.buf.append((t, frame))

    def _prev(self, t: float):
        """取 t−HISTORY_FRAME_S（当前 0.2 s）最近的一帧；没有就返回 None（调用方用当前帧重复）。"""
        target = t - HISTORY_FRAME_S
        best = None
        for ts, f in self.buf:
            if best is None or abs(ts - target) < abs(best[0] - target):
                best = (ts, f)
        if best is not None and abs(best[0] - target) <= 0.15:
            return best[1]
        return None

    # ------------------------------------------------------------ 主循环
    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.004, next_t - now))
                continue
            next_t = max(next_t + self.period, now)
            t0 = time.perf_counter()
            try:
                cur = self.frame_source()
                if cur is None:
                    self._emit_zero(t0, "no_frame")
                    continue
                self._push(t0, cur)
                prev = None
                if self.prev_source is not None:
                    try:
                        prev = self.prev_source()
                    except Exception:  # noqa: BLE001 - 历史帧拿不到就退回本拍缓冲
                        prev = None
                if prev is None:
                    prev = self._prev(t0)
                if prev is None:
                    prev = cur
                guidance = None
                if self.guidance_provider is not None:
                    try:
                        candidate = self.guidance_provider()
                        guidance = dict(candidate) if isinstance(candidate, dict) else None
                    except Exception:  # noqa: BLE001 - 引导失败时退化为纯视觉
                        guidance = None
                if guidance is not None:
                    wp = guidance.get("waypoint") or (0.0, 0.0)
                elif self.wp_provider is not None:
                    wp = self.wp_provider()
                else:
                    wp = (0.0, 0.0)
                body = self.body_provider() if self.body_provider is not None else (0.0, 0.0)
                # obs_t 用**图像采集时刻**，见 __init__ 的 captured_at_source 注释。
                # 拿不到（provider 缺失/抛错/给的不是数）就退回本拍开始时刻 ——
                # 退路必须与旧行为逐位一致，否则这条改动会变成「第二个 bug」。
                obs_t = t0
                if self.captured_at_source is not None:
                    try:
                        cap = self.captured_at_source()
                    except Exception:  # noqa: BLE001 - 读不到就退回旧口径
                        cap = None
                    if isinstance(cap, (int, float)) and cap > 0:
                        # 采集时刻晚于 t0 是不可能的（那意味着帧还没拍）；夹一下防跨域误用。
                        obs_t = min(float(cap), t0)
                path = None if guidance is None else guidance.get("path")
                if path is None and self.path_source is not None:
                    try:
                        path = self.path_source()
                    except Exception:  # noqa: BLE001 - 拿不到路径就退回单点
                        path = None
                # 只在真有路径时才传 path：注入式 runtime（测试替身、旧实现）的 act()
                # 没有这个关键字参数，无条件传会把每一拍都炸成 error:TypeError。
                mode = 0
                if guidance is not None and guidance.get("mode") is not None:
                    try:
                        mode = int(guidance.get("mode") or 0)
                    except (TypeError, ValueError):
                        mode = 0
                elif self.mode_provider is not None:
                    try:
                        mode = int(self.mode_provider())
                    except Exception:  # noqa: BLE001 - 读不到按 follow
                        mode = 0
                # 注入式 runtime（测试替身、旧实现）的 act() 没有这些关键字参数，
                # 无条件传会把每一拍炸成 error:TypeError —— 按能力探测后再传。
                extra = {}
                prompt_text = None
                if path is not None:
                    extra["path"] = path
                if guidance is not None:
                    extra["guide"] = guidance
                    # 显式导航状态（NavSession.mode 字符串）：停止判定的来源之一。
                    # 它不是「privileged 动作输入」——runtime 只拿它做 decide_stop，
                    # 不会把它写进喂给模型的记忆卡（那是 guidance["mode"] 的事）。
                    if guidance.get("nav_state") is not None:
                        extra["nav_state"] = guidance.get("nav_state")
                    # 定位安全门：**独立**于 wp / nav mode。None = 没有定位信息，
                    # 不适用（独立 VLA 无 navmesh 时走这条）。
                    if guidance.get("localization_ok") is not None:
                        extra["localization_ok"] = bool(guidance["localization_ok"])
                        extra["localization_reason"] = guidance.get("localization_reason")
                if self.prompt_provider is not None:
                    try:
                        prompt = self.prompt_provider()
                    except Exception:  # noqa: BLE001 - 提示读取失败退化为无提示
                        prompt = None
                    if prompt:
                        prompt_text = str(prompt)
                        extra["task_prompt"] = prompt_text
                try:
                    out = self.rt.act(cur, prev, wp, body, mode=mode, **extra)
                except TypeError:
                    # 旧注入式 runtime 可能连 path 也没有；兼容回退必须把
                    # 所有新增关键字一起去掉，而不是第二次继续传 path。
                    out = self.rt.act(cur, prev, wp, body)
                latency = out["latency_ms"]
                stale = latency > self.stale_s * 1000
                block = out["block"]
                if stale:
                    block = self._zero_block()
                # ⚠️ 不能写 `0 if not path else len(path)`：path 可能是 numpy 数组，
                # 对它求布尔会抛 ValueError（与 2026-11-01 那个 prev 帧的 bug 同款）。
                path_n = len(path) if path is not None else 0
                # 2026-11-06：停止语义从 `gated` 收敛到 `stop`（policy_modes）；
                # `gated` 仍然写进 meta，作为旧日志/旧消费端的兼容别名。
                stop = bool(out.get("stop", out.get("gated", False)))
                loc_ok = (None if guidance is None
                          else guidance.get("localization_ok"))
                loc_reason = (None if guidance is None
                              else guidance.get("localization_reason"))
                meta = dict(latency_ms=round(latency, 1), stop=stop,
                            stop_source=out.get("stop_source"),
                            stop_reason=out.get("stop_reason"),
                            stop_detail=out.get("stop_detail"),
                            stop_head_p=out.get("stop_head_p"),
                            gated=stop, stale=stale,
                            # 定位安全门也进 meta：executor 会再判一次（幂等），
                            # 这样即使注入的 runtime 替身忽略该字段也不会漏停。
                            localization_ok=loc_ok,
                            localization_reason=loc_reason,
                            wp=list(wp), body=list(body),
                            frame_age_ms=round((t0 - obs_t) * 1000, 1),
                            path_n=path_n,
                            task_prompt=prompt_text)
                if guidance is not None:
                    # 只记录引导质量，当前 checkpoint 仍按 input_schema 决定是否消费 wp。
                    # 这组字段用于训练时做引导损坏增强和在线排障。
                    for key in ("valid", "confidence", "pose_age_s", "sigma_m", "reason", "source"):
                        if key in guidance:
                            meta[f"guide_{key}"] = guidance[key]
                # 最近一拍快照：真驱动模式下不写 jsonl，/policy/state 靠它看模型视角
                self._last_block = block
                self._last_meta = meta
                if self.dry_run:
                    self._log(dict(obs_t=obs_t, block=block, **meta))
                elif self.on_block is not None:
                    self.n_on_block = getattr(self, "n_on_block", 0) + 1
                    self.on_block(block, obs_t, meta)
            except Exception as exc:  # noqa: BLE001 - 单拍失败必须停车，不能把异常带进控制环
                self._emit_zero(t0, f"error:{type(exc).__name__}")
                if self.log:
                    self.log.write(json.dumps({"t": t0, "error": repr(exc)}, ensure_ascii=False) + "\n")

    def _emit_zero(self, t0: float, reason: str) -> None:
        block = self._zero_block()
        if self.dry_run:
            self._log(dict(obs_t=t0, block=block, reason=reason))
        elif self.on_block is not None:
            self.on_block(block, t0, {"reason": reason})

    def _log(self, obj: dict) -> None:
        if self.log:
            self.log.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.log.flush()

    def _zero_block(self):
        """按 runtime 的动作维度生成零块，避免三通道模型退化成二维块。"""
        try:
            act_dim = max(2, int(getattr(self.rt, "act_dim", 2)))
        except (TypeError, ValueError):
            act_dim = 2
        return [[0.0] * act_dim for _ in range(12)]

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        self._th = threading.Thread(target=self._loop, name="policy-runner", daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        if self.log:
            self.log.close()
