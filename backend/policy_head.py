# -*- coding: utf-8 -*-
"""Service 侧 VLA 头：把 PolicyRuntime / PolicyRunner / PolicyExecutor 组装成一个可选 head。

不直接依赖 service 的内部字段；由 Service.start() 注入六个可调用对象：
  * ``frame_source()``  → 当前左目 RGB（1080×606，uint8）或 None
  * ``prev_frame_source()`` → 距今 0.4 s 的左目 RGB 或 None（训练侧两帧间距）
  * ``wp_provider()``   → 机体系局部航点 (dx, dz) 米
  * ``body_provider()`` → 当前 OSC 本体速度 (vx, vz) m/s
  * ``send_axes(x, y, duration_ms)`` → 是否接受（接 ``Service._navigator_send_axes``）
  * ``hold_provider()`` → True = 外部按住轴（转向代管原地转中，见 nav_online.policy_hold）

默认 ``dry_run=True``：只记录块，不发动作；``Service`` 用配置 ``[policy].enabled`` 决定是否构造。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "research" / "tools"))
from policy_executor import PolicyExecutor, SafetyGate  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402
from policy_runtime import PolicyRuntime  # noqa: E402


class PolicyHead:
    def __init__(self, *, model_dir: str, run_dir: str, frame_source: Callable[[], Any],
                 wp_provider: Callable[[], tuple[float, float]],
                 body_provider: Callable[[], tuple[float, float]],
                 send_axes: Callable[[float, float, int], bool] | None,
                 log_dir: str, dry_run: bool = True, hz: float = 1.0,
                 v_max: float = 1.2, dv_max: float = 0.5, wp_stop: float = 0.02,
                 stale_s: float = 1.3, max_len: int = 192,
                 prev_frame_source: Callable[[], Any] | None = None,
                 captured_at_source: Callable[[], float] | None = None,
                 path_source: Callable[[], Any] | None = None,
                 graph: bool = False, near_obstacle: Callable[[], tuple[bool, int]] | None = None,
                 hold_provider: Callable[[], bool] | None = None,
                 floor_provider: Callable[[], float] | None = None,
                 floor_frac: float = 0.6,
                 logger: Any = None):
        self.logger = logger
        self.dry_run = bool(dry_run)
        self._send = send_axes
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.rt = PolicyRuntime(model_dir, run_dir, max_len=max_len, wp_stop=wp_stop,
                                graph=graph)
        # 近距障碍否决：navmesh 感知线程已经在算 near_obstacle，这里只读它的最新值。
        # 方案 §13 要求「策略输出前过 near_obstacle veto」，否则 0.8 s 开环会盲走。
        self._near_obstacle = near_obstacle
        self.executor = PolicyExecutor(
            self._send_axes if not dry_run else (lambda x, y, d: True),
            stale_s=stale_s, stop_file=str(self._log_dir / "STOP"),
            dry_run=dry_run, gate=SafetyGate(v_max=v_max, dv_max=dv_max,
                                             floor_frac=floor_frac),
            hold_provider=hold_provider, floor_provider=floor_provider)
        self.runner = PolicyRunner(
            self.rt, frame_source, wp_provider, body_provider,
            on_block=None if dry_run else self._on_block,
            hz=hz, dry_run=dry_run, stale_s=stale_s,
            prev_source=prev_frame_source,
            captured_at_source=captured_at_source,
            path_source=path_source,
            log_path=str(self._log_dir / "policy_live.jsonl"))

    def _on_block(self, block, obs_t, meta):
        """策略块落地前过近距障碍否决。

        ``near_obstacle`` 是 navmesh 感知线程每帧算好的 ``(stop, n_pts)``；
        正面 0.7 m 内点数够多就整块置零，交回原地。**不是**替代 SafetyGate ——
        SafetyGate 管限速限加速，这里管「前面有东西就别动」。
        """
        meta = dict(meta or {})
        if self._near_obstacle is not None:
            try:
                stop, n_pts = self._near_obstacle()
            except Exception:  # noqa: BLE001 - 读不到就当没有，不阻塞策略
                stop, n_pts = False, 0
            if stop:
                meta["near_obstacle"] = n_pts
                meta["vetoed"] = True
                block = [[0.0, 0.0]] * len(block)
        self.executor.submit(block, obs_t, meta)

    def _send_axes(self, x: float, y: float, duration_ms: int) -> bool:
        if self._send is None:
            return False
        return bool(self._send(x, y, int(duration_ms)))

    def start(self):
        self.executor.start()
        self.runner.start()
        if self.logger:
            self.logger.info("policy head started (dry_run=%s, graph=%s, veto=%s)",
                             self.dry_run, self.rt.graph_enabled,
                             self._near_obstacle is not None)

    def stop(self):
        self.runner.stop()
        self.executor.stop()
        if self.logger:
            self.logger.info("policy head stopped")
