# -*- coding: utf-8 -*-
"""因果性验证：causal 模式下的 [wp]/[topo] 必须与**未来数据**完全无关。

做法：把一条 DR 轨迹在 t_obs 之后的部分随机打乱，重新计算特征；
如果 causal 实现正确，特征必须逐位不变。旧 near/far 模式则会变。
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/tools")
import policy_causal as PC

rng = np.random.default_rng(0)
n = 200
t = np.arange(n, dtype=np.float64) * 0.1
p = np.stack([np.linspace(0, 10, n), np.linspace(0, 10, n) * 0.3], axis=1)
seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])
path = dict(t=t, p=p, s=s)
fwd = (1.0, 0.0)
t_obs = float(t[100])

base = PC.causal_topo(t_obs, path, fwd)
print("[基准] t_obs=%.1f  prog=%.4f lateral=%.4f remain=%.4f" % ((t_obs,) + base))

bad = 0
for trial in range(5):
    p2 = p.copy()
    p2[101:] += rng.normal(0, 50.0, size=(n - 101, 2))
    seg2 = np.linalg.norm(np.diff(p2, axis=0), axis=1)
    s2 = np.concatenate([[0.0], np.cumsum(seg2)])
    got = PC.causal_topo(t_obs, dict(t=t, p=p2, s=s2), fwd)
    same = all(abs(a - b) < 1e-12 for a, b in zip(base, got))
    print("  trial %d 未来扰动 ×50m -> %s  %s" % (trial, got, "不变 OK" if same else "变了 BUG"))
    if not same:
        bad += 1

print()
if bad:
    print("FAIL: %d/5 次扰动改变了特征，说明仍在使用未来" % bad)
    raise SystemExit(1)
print("PASS: causal topo 对未来扰动完全不敏感")
