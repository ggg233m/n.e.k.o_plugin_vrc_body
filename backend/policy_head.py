# -*- coding: utf-8 -*-
"""Service 侧 VLA 头：把 PolicyRuntime / PolicyRunner / PolicyExecutor 组装成一个可选 head。

不直接依赖 service 的内部字段；由 Service.start() 注入六个可调用对象：
  * ``frame_source()``  → 当前左目 RGB（1080×606，uint8）或 None
  * ``prev_frame_source()`` → 距今 0.2 s 的左目 RGB 或 None（训练侧两帧间距，
    见 policy_histframe.HISTORY_FRAME_S；缺帧时 runner 回退成当前帧复制）
  * ``wp_provider()``   → 兼容旧接线的局部引导点；生产链路使用
    ``guidance_provider()``，把 A* 作为可丢弃的软引导输入
  * ``body_provider()`` → 当前 OSC 本体速度 (vx, vz) m/s
  * ``send_axes(x, y, duration_ms)`` → 是否接受（接 ``Service._navigator_send_axes``）
  * ``hold_provider()`` → 可选外部按住轴；独立 VLA 不接入 A* 的 policy_hold

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


def make_runtime(model_dir: str, run_dir: str, *, max_len: int, wp_stop: float,
                 graph: bool, logger=None):
    """建 runtime：**优先进程内，torch 缺失时自动回退到 `.venv_xpu` 子进程**。

    为什么需要这层（2026-11-05）：
      主后端跑在 `.venv`（Py3.12，**没有 torch/transformers**），而 PolicyRuntime
      顶部就 `import torch`。此前是模块级 `from policy_runtime import PolicyRuntime`，
      在 `.venv` 里**整个 policy_head 都 import 不进来** —— service 的 try/except
      把它兜成「没有策略头」，日志只留一句 policy_start_error，看不出是缺 torch。

      现在改成惰性：能 import 就用进程内（旧行为，逐位不变）；不能 import 且仓库里
      有 `.venv_xpu` 就用 RemotePolicyRuntime（子进程，接口完全一致）。
      两者都不行才抛，由 service 记 policy_start_error。

    ⚠️ 退回条件是 ImportError（真的没有 torch），不是所有异常 ——
    不能把「模型损坏/配置错」也悄悄降级成子进程，那会掩盖真实故障。
    """
    try:
        from policy_runtime import PolicyRuntime
        return PolicyRuntime(model_dir, run_dir, max_len=max_len,
                             wp_stop=wp_stop, graph=graph)
    except ImportError as exc:
        if logger:
            logger.info("policy runtime 进程内不可用（%r），改走 .venv_xpu worker", exc)
        from .policy_backend import RemotePolicyRuntime, find_worker_python
        py = find_worker_python()
        if not py:
            raise RuntimeError(
                "进程内缺 torch（%r），且找不到 .venv_xpu；"
                "请把 VRC_POLICY_PYTHON 指向装了 torch 的解释器" % (exc,)) from exc
        return RemotePolicyRuntime(model_dir, run_dir, max_len=max_len,
                                   wp_stop=wp_stop, graph=graph, python=py,
                                   logger=logger)


class PolicyHead:
    def __init__(self, *, model_dir: str, run_dir: str, frame_source: Callable[[], Any],
                 wp_provider: Callable[[], tuple[float, float]] | None = None,
                 body_provider: Callable[[], tuple[float, float]] | None = None,
                 send_axes: Callable[[float, float, int], bool] | None = None,
                 log_dir: str = ".", dry_run: bool = True, hz: float = 1.0,
                 send_turn: Callable[[float], bool] | None = None,
                 v_max: float = 1.2, dv_max: float = 0.5, wp_stop: float = 0.02,
                 stale_s: float = 1.3, max_len: int = 192,
                 prev_frame_source: Callable[[], Any] | None = None,
                 captured_at_source: Callable[[], float] | None = None,
                 path_source: Callable[[], Any] | None = None,
                 graph: bool = False, near_obstacle: Callable[[], tuple[bool, int]] | None = None,
                 hold_provider: Callable[[], bool] | None = None,
                 floor_provider: Callable[[], float] | None = None,
                 floor_frac: float = 0.6,
                 mode_provider: Callable[[], int] | None = None,
                 guidance_provider: Callable[[], dict[str, Any]] | None = None,
                 prompt_provider: Callable[[], str | None] | None = None,
                 logger: Any = None):
        self.logger = logger
        self.dry_run = bool(dry_run)
        self._send = send_axes
        # 转向通道（度/次）。对应 service._navigator_send_turn。
        self._send_turn_fn = send_turn
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # 2026-11-05：改走工厂。`.venv` 主后端没有 torch，会自动回退到
        # `.venv_xpu` 子进程（见 make_runtime 的注释）；有 torch 时行为不变。
        self.rt = make_runtime(model_dir, run_dir, max_len=max_len,
                               wp_stop=wp_stop, graph=graph, logger=logger)
        # 近距障碍否决只使用感知线程的安全数据，不使用 A* 路径或跟随器输出。
        self._near_obstacle = near_obstacle
        self.executor = PolicyExecutor(
            self._send_axes if not dry_run else (lambda x, y, d: True),
            stale_s=stale_s, stop_file=str(self._log_dir / "STOP"),
            dry_run=dry_run, gate=SafetyGate(v_max=v_max, dv_max=dv_max,
                                             floor_frac=floor_frac),
            hold_provider=hold_provider, floor_provider=floor_provider,
            # ⚠️ 转向走独立通道：VRChat 摇杆只有 x/y 两轴，
            # 早期把 turn 塞进 send_axes 会被 TypeError 兜底静默丢掉。
            send_turn=(self._send_turn if not dry_run else None))
        self.runner = PolicyRunner(
            self.rt, frame_source, wp_provider, body_provider,
            mode_provider=mode_provider,
            on_block=None if dry_run else self._on_block,
            hz=hz, dry_run=dry_run, stale_s=stale_s,
            prev_source=prev_frame_source,
            captured_at_source=captured_at_source,
            path_source=path_source,
            guidance_provider=guidance_provider,
            prompt_provider=prompt_provider,
            log_path=str(self._log_dir / "policy_live.jsonl"))

    def _on_block(self, block, obs_t, meta):
        """策略块落地前过近距障碍否决。

        ``near_obstacle`` 是 navmesh 感知线程每帧算好的 ``(stop, n_pts)``；
        正面 0.7 m 内点数够多就整块置零，交回原地。

        2026-11-06：这里**不再**自己清零块，只标 ``safety_veto``（唯一停止判定的
        一种来源，见 policy_modes）。清零由 ``PolicyExecutor.submit`` 统一做，
        这样 `stop_source` 能如实记成 ``safety_gate``，而不是被上游的布尔短路吃掉。
        """
        meta = dict(meta or {})
        if self._near_obstacle is not None:
            try:
                stop, n_pts = self._near_obstacle()
            except Exception:  # noqa: BLE001 - 读不到就当没有，不阻塞策略
                stop, n_pts = False, 0
            if stop:
                meta["near_obstacle"] = n_pts
                meta["safety_veto"] = True
                meta["vetoed"] = True  # 兼容旧日志字段
        self.executor.submit(block, obs_t, meta)

    def _send_axes(self, x: float, y: float, duration_ms: int) -> bool:
        if self._send is None:
            return False
        return bool(self._send(x, y, int(duration_ms)))

    def _send_turn(self, delta_deg: float) -> bool:
        """转向通道：对应 service._navigator_send_turn(delta_deg)，单位**度**。"""
        if self._send_turn_fn is None:
            return False
        return bool(self._send_turn_fn(float(delta_deg)))

    def start(self):
        self.executor.start()
        self.runner.start()
        if self.logger:
            self.logger.info("policy head started (dry_run=%s, graph=%s, veto=%s)",
                             self.dry_run,
                             getattr(self.rt, "graph_enabled", False),
                             self._near_obstacle is not None)

    def stop(self):
        self.runner.stop()
        self.executor.stop()
        # 子进程模式必须显式关：否则 worker 会带着 A770 的 XPU 上下文变成孤儿
        # （stereo_backend.py:885 记过同一个坑：terminate 只杀包装层）。
        closer = getattr(self.rt, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 - 关不掉不该阻塞停机
                if self.logger:
                    self.logger.warning("policy runtime close 失败: %r", exc)
        if self.logger:
            self.logger.info("policy head stopped")
