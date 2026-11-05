# -*- coding: utf-8 -*-
"""策略 worker 的父进程侧适配器（跑在 `.venv`，**不 import torch**）。

为什么需要它
-------------
主后端 `.venv` 没有 torch/transformers，而 PolicyRuntime 必须有。
本仓已有同型解法：`stereo_backend._las2_python()` 用 `.venv_las` 解释器拉起
`backend.stereo_worker`，父进程全程不 import torch。这里沿用同一套：

  主后端(.venv)  ──stdin/stdout 行 JSON──▶  policy_worker(.venv_xpu)
                                              └─ PolicyRuntime + torch

`RemotePolicyRuntime` 的 `act()` 签名与返回值**与 `PolicyRuntime.act()` 一致**，
所以 `PolicyRunner` 不需要任何改动就能用它。

与 stereo_backend 的差别（都是有意的）
------------------------------------
| 项 | 双目 | 策略 | 理由 |
| --- | --- | --- | --- |
| IPC | 共享内存+命名信号量 | stdin/stdout 行 JSON | 策略 2 Hz、~0.9 MB/拍，管道足够 |
| 保活 | PHASE/HEARTBEAT 控制块 | 读超时 + 进程存活探测 | 每拍一问一答，不需要心跳 |
| 解释器 | `.venv_las` | `.venv_xpu` | 各自的 torch 栈 |
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
SPAWN_TIMEOUT_S = 180.0   # 载模型 + 首次前向（未 compile）实测 ~40 s，留足余量
CALL_TIMEOUT_S = 30.0     # 单拍超时：远大于实测 0.42 s，超了就是真卡住


class PolicyWorkerError(RuntimeError):
    pass


def find_worker_python() -> str | None:
    """worker 用哪个解释器。

    优先 `VRC_POLICY_PYTHON`（测试/换机时钉死），其次仓库里的 `.venv_xpu`。
    找不到返回 None —— 调用方据此判断「本机没有可用的 torch 环境」，
    退回「没有策略头」而不是崩掉整个后端。
    """
    explicit = os.environ.get("VRC_POLICY_PYTHON")
    if explicit:
        return explicit
    for rel in (os.path.join(".venv_xpu", "Scripts", "python.exe"),
                os.path.join(".venv_xpu", "bin", "python")):
        cand = os.path.join(PROJECT_DIR, rel)
        if os.path.isfile(cand):
            return cand
    return None


def _worker_env() -> dict:
    """子进程环境。把主后端的 Py3.12 相关项剔掉，其余照传。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # 父进程若设了 VRC_POLICY_DRYRUN 等策略开关，子进程需要同一份（它只读 MODEL/RUN）。
    # 从 .venv 继承来的 PYTHONPATH 可能指向 3.12 的 site-packages，必须清掉。
    env.pop("PYTHONPATH", None)
    return env


class RemotePolicyRuntime:
    """在 `.venv_xpu` 子进程里跑 PolicyRuntime，接口与本地版一致。

    只实现 PolicyRunner 需要的那部分：`act()` 返回 `block / gated / obs_t / latency_ms`。
    """

    def __init__(self, model_dir: str, run_dir: str, size=(448, 256), max_len: int = 192,
                 wp_stop: float = 0.02, compile: bool = False, device: str | None = None,
                 graph: bool = False, merge: bool = True, python: str | None = None,
                 logger=None):
        self.wp_stop = float(wp_stop)
        self.size = (int(size[0]), int(size[1]))
        self.act_dim = 0
        self.schema = {}
        self.drop_wp = False
        self.path_in = False
        self._id = 0
        self._lock = threading.Lock()
        self._logger = logger
        self.python = python or find_worker_python()
        if not self.python:
            raise PolicyWorkerError(
                "找不到 .venv_xpu 解释器（可用 VRC_POLICY_PYTHON 指定）")
        env = _worker_env()
        env["VRC_POLICY_MODEL"] = str(model_dir)
        env["VRC_POLICY_RUN"] = str(run_dir)
        env["VRC_POLICY_MAXLEN"] = str(int(max_len))
        env["VRC_POLICY_COMPILE"] = "1" if compile else "0"
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [self.python, "-m", "backend.policy_worker"],
            cwd=PROJECT_DIR, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, creationflags=creationflags)
        self._handshake()

    # ---------------------------------------------------------------- 协议
    def _send(self, obj) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            raise PolicyWorkerError("写 worker 失败: %r" % (exc,)) from exc

    def _recv(self, timeout_s: float = CALL_TIMEOUT_S) -> dict:
        """读一行 JSON。用线程 + 超时，避免 Windows 管道阻塞把主循环挂死。"""
        box = {}

        def _read():
            try:
                box["line"] = self._proc.stdout.readline()
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=_read, daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            raise PolicyWorkerError("worker 读超时（%.0fs）" % timeout_s)
        if "err" in box:
            raise PolicyWorkerError("读 worker 失败: %r" % (box["err"],))
        line = box.get("line") or ""
        if not line.strip():
            tail = ""
            try:
                tail = (self._proc.stderr.read() or "")[-1500:]
            except Exception:  # noqa: BLE001
                pass
            raise PolicyWorkerError("worker 提前关闭（rc=%r）。stderr 尾部：%s"
                                    % (self._proc.poll(), tail))
        try:
            return json.loads(line)
        except Exception as exc:  # noqa: BLE001
            raise PolicyWorkerError("协议错位，收到非 JSON：%r" % (line[:300],)) from exc

    def _call(self, obj, timeout_s: float = CALL_TIMEOUT_S) -> dict:
        self._send(obj)
        resp = self._recv(timeout_s)
        if not resp.get("ok"):
            raise PolicyWorkerError("worker 报错: %s" % (resp.get("error"),))
        return resp

    def _handshake(self) -> None:
        try:
            r = self._call({"op": "hello"}, timeout_s=SPAWN_TIMEOUT_S)
        except Exception:
            self.close()
            raise
        self.schema = r.get("schema") or {}
        self.act_dim = int(r.get("act_dim") or 0)
        self.drop_wp = bool(r.get("drop_wp"))
        self.path_in = bool(r.get("path_in"))
        self.torch_version = r.get("torch")
        if self.act_dim <= 0:
            self.close()
            raise PolicyWorkerError("worker 握手返回 act_dim=%r" % (self.act_dim,))

    # ---------------------------------------------------------------- 对外
    def act(self, cur_rgb, prev_rgb, wp, body, preprocessed: bool = False,
            path=None, mode: int = 0, topo=None, guide=None,
            task_prompt: str | None = None, nav_state=None, stop_head=None,
            safety_veto: bool = False, localization_ok=None,
            localization_reason=None) -> dict:
        """与 PolicyRuntime.act() 同签名同返回。帧走 base64 原始 RGB（无损）。

        ⚠️ 2026-11-06：父进程侧**不再**做 `|wp| < wp_stop → 整块归零`。
        停止判定只在远端 `PolicyRuntime.act` 里做一次（单一真源），这里只转发
        ``nav_state`` / ``stop_head`` / ``safety_veto`` 并原样带回结果。
        """
        import numpy as np
        t0 = time.perf_counter()
        if preprocessed:
            cur = np.ascontiguousarray(cur_rgb, dtype=np.uint8)
            prev = np.ascontiguousarray(prev_rgb, dtype=np.uint8)
        else:
            # 尺寸变换放在**父进程**做：只需要 numpy/PIL，不需要 torch，
            # 而且能少传很多字节（WGC 原图可能远大于 448x256）。
            # 与训练侧**同一个函数**（policy_resize.to_policy_rgb，policy_runtime 也用它）。
            # 只依赖 numpy+PIL，所以 .venv 里可跑；绝不复写第二份几何。
            from policy_resize import to_policy_rgb
            cur = to_policy_rgb(np.asarray(cur_rgb, dtype=np.uint8), self.size)
            prev = to_policy_rgb(np.asarray(prev_rgb, dtype=np.uint8), self.size)
        h, w = int(cur.shape[0]), int(cur.shape[1])
        cur_b = base64.b64encode(cur.tobytes()).decode("ascii")
        prev_b = base64.b64encode(prev.tobytes()).decode("ascii")
        with self._lock:
            self._id += 1
            rid = self._id
            resp = self._call(dict(
                op="act", id=rid, cur=cur_b, prev=prev_b, shape=[h, w],
                wp=[float(wp[0]), float(wp[1])],
                body=[float(body[0]), float(body[1])],
                mode=int(mode), path=(None if path is None else [[float(a), float(b)] for a, b in path]),
                guide=(None if guide is None else dict(guide)),
                task_prompt=(None if not task_prompt else str(task_prompt)[:96]),
                nav_state=(None if nav_state is None else str(nav_state)),
                stop_head=(None if stop_head is None else float(stop_head)),
                safety_veto=bool(safety_veto),
                localization_ok=(None if localization_ok is None else bool(localization_ok)),
                localization_reason=(None if localization_reason is None
                                     else str(localization_reason)),
            ))
        block = resp.get("block")
        stop = bool(resp.get("stop", resp.get("gated", False)))
        if stop:
            block = [[0.0] * self.act_dim for _ in range(len(block) if block else 12)]
        return dict(block=block, stop=stop,
                    stop_source=resp.get("stop_source"),
                    stop_reason=resp.get("stop_reason"),
                    stop_detail=resp.get("stop_detail"),
                    gated=stop, obs_t=t0,
                    latency_ms=(time.perf_counter() - t0) * 1000.0)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    self._send({"op": "stop"})
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if proc.poll() is None:
                # 只用 terminate 会留下孤儿（正是 stereo_backend:885 记录的坑）：
                # 启动器包装层被杀，真正的 python 还在占 XPU。必须 /T 连子树一起收。
                if os.name == "nt":
                    try:
                        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                       capture_output=True, timeout=10)
                    except Exception:  # noqa: BLE001
                        proc.kill()
                else:
                    proc.kill()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
