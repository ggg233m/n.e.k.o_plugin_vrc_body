# -*- coding: utf-8 -*-
"""策略推理独立进程（跑在 `.venv_xpu`）：载 PolicyRuntime，收帧返回动作块。

为什么要独立进程
----------------
主后端跑在 `.venv`（Py3.12，**没有 torch/transformers**），而 PolicyRuntime 必须用
`.venv_xpu`（Py3.14 + torch-xpu）。本仓已有同型先例：`backend/stereo_worker.py` 由
`.venv` 主后端用 `.venv_las` 解释器拉起（见 stereo_backend._las2_python）。
这里沿用**同一个架构决策**（独立解释器 + 子进程 + 父进程不 import torch），
只把 IPC 从共享内存换成 stdin/stdout 的行 JSON —— 理由是数据量差两个数量级：

* 双目：每帧 2×H×W×3 的 uint8 + 每帧都要算（30 Hz 级）⇒ 共享内存 + 命名信号量；
* 策略：每拍 2 张 448×256×3 + **2 Hz** ⇒ 行 JSON 足够（实测 ~0.9 MB/拍）。

协议（一行一条 JSON，UTF-8）
--------------------------
父 → 子：
  {"op":"hello"}                              握手，返回 schema/act_dim
   {"op":"act","id":N,"cur":"<b64>","prev":"<b64>",
   "shape":[h,w],"wp":[x,y],"body":[vx,vz],"mode":0,"path":null,
   "task_prompt":"free explore"}
  {"op":"stop"}                                退出
子 → 父：
  {"op":"hello","ok":true,"schema":{...},"act_dim":3,"torch":"..."}
  {"op":"act","id":N,"ok":true,"block":[[vx,vz,w]x12],"gated":false,"latency_ms":405.6}
  {"op":"error","id":N,"ok":false,"error":"..."}

协议通道与日志通道必须分开
--------------------------
`PolicyRuntime.__init__` 会 print 到 **stdout**（[runtime] input_schema / adapter /
[chunk] chunk_size），若不重定向，父进程读到的第一行就不是 JSON —— 协议当场错位。
本文件在进程启动时把 fd 1 复制一份留作协议通道，再 dup2(2, 1)，
于是所有 print/tqdm 落到 **stderr**，协议通道保持干净。
"""
from __future__ import annotations

import base64
import json
import os
import sys
import traceback

# 必须在 import torch/transformers 之前做，才能盖住它们后续的 print。
_PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)  # 之后 fd 1 指向 stderr，库的 print 不再污染协议

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "research", "tools"))
sys.path.insert(0, _REPO)


def _emit(obj) -> None:
    _PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO.flush()


def _b64_to_rgb(b64: str, shape):
    import numpy as np
    h, w = int(shape[0]), int(shape[1])
    raw = base64.b64decode(b64)
    n = h * w * 3
    if len(raw) != n:
        raise ValueError("帧字节数 %d != %dx%dx3=%d" % (len(raw), h, w, n))
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)


def main() -> int:
    rt = None
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                _emit(dict(op="error", ok=False, error="bad json: %r" % (exc,)))
                continue
            op = msg.get("op")
            rid = msg.get("id")

            if op == "stop":
                _emit(dict(op="stop", ok=True))
                return 0

            if op == "hello":
                if rt is None:
                    import torch
                    from policy_runtime import PolicyRuntime
                    model_dir = os.environ.get("VRC_POLICY_MODEL") or ""
                    run_dir = os.environ.get("VRC_POLICY_RUN") or ""
                    if not model_dir or not run_dir:
                        raise RuntimeError("需要 VRC_POLICY_MODEL 与 VRC_POLICY_RUN")
                    # wp_stop=0：wp≈0 **不是**停车条件（2026-11-06），这里传 0
                    # 只是关掉「有没有路径先验」的日志注记阈值；停止判定在
                    # PolicyRuntime.act 的 decide_stop 里统一做。
                    rt = PolicyRuntime(model_dir, run_dir, size=(448, 256),
                                       max_len=int(os.environ.get("VRC_POLICY_MAXLEN") or 192),
                                       compile=os.environ.get("VRC_POLICY_COMPILE", "0") == "1",
                                       wp_stop=0.0)
                _emit(dict(op="hello", ok=True, schema=getattr(rt, "schema", {}),
                           act_dim=int(getattr(rt, "act_dim", 0)),
                           torch=getattr(__import__("torch"), "__version__", "?"),
                           drop_wp=bool(getattr(rt, "drop_wp", False)),
                           path_in=bool(getattr(rt, "path_in", False))))
                continue

            if op == "act":
                if rt is None:
                    _emit(dict(op="error", id=rid, ok=False, error="hello not done"))
                    continue
                cur = _b64_to_rgb(msg["cur"], msg["shape"])
                prev = _b64_to_rgb(msg["prev"], msg["shape"])
                out = rt.act(cur, prev, tuple(msg.get("wp") or (0.0, 0.0)),
                             tuple(msg.get("body") or (0.0, 0.0)),
                             preprocessed=False,
                             path=msg.get("path"),
                             mode=int(msg.get("mode") or 0),
                             guide=msg.get("guide"),
                             task_prompt=msg.get("task_prompt"),
                             nav_state=msg.get("nav_state"),
                             stop_head=msg.get("stop_head"),
                             safety_veto=bool(msg.get("safety_veto", False)),
                             localization_ok=msg.get("localization_ok"),
                             localization_reason=msg.get("localization_reason"))
                _emit(dict(op="act", id=rid, ok=True, block=out["block"],
                           stop=bool(out.get("stop", out.get("gated", False))),
                           stop_source=out.get("stop_source"),
                           stop_reason=out.get("stop_reason"),
                           gated=bool(out.get("gated", out.get("stop", False))),
                           latency_ms=float(out["latency_ms"])))
                continue

            _emit(dict(op="error", id=rid, ok=False, error="unknown op %r" % (op,)))
    except Exception as exc:  # noqa: BLE001 - 任何异常都要回给父进程，不能死在管道上
        _emit(dict(op="fatal", ok=False, error=repr(exc),
                   tb=traceback.format_exc()[-2000:]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
