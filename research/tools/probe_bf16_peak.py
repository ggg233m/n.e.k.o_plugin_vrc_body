# -*- coding: utf-8 -*-
"""A770 BF16 峰值 vs 我们实际负载 —— 说明「算力强」为什么帮不上忙。

三组对照：
  A. 大矩阵乘（XMX/DPAS 才吃得到的形态）
  B. decode 形态的 GEMV（M=1，纯带宽受限，XMX 用不上）
  C. 我们模型一次 prefill 的理论 FLOP 与实测耗时
"""
import time, numpy as np, torch

dev = "xpu"
sync = torch.xpu.synchronize

def bench(fn, n=10, warm=3):
    for _ in range(warm): fn()
    sync()
    t = time.perf_counter()
    for _ in range(n): fn()
    sync()
    return (time.perf_counter() - t) / n

print("=== A. 大矩阵乘（吃 XMX）===")
for N in (2048, 4096, 8192):
    a = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
    dt = bench(lambda: a @ b, n=(10 if N <= 4096 else 4))
    fl = 2.0 * N**3
    print("  %5d^3  %8.1f ms   %6.1f TFLOPS" % (N, dt*1000, fl/dt/1e12))
    del a, b

print("=== B. decode 形态 GEMV（M=1，用不上 XMX）===")
for N in (1024, 4096):
    w = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
    x = torch.randn(N, 1, device=dev, dtype=torch.bfloat16)
    dt = bench(lambda: w @ x, n=50)
    fl = 2.0 * N * N
    gb = N * N * 2 / 1e9
    print("  %5d^2 @ (N,1)  %8.1f us   %6.1f TFLOPS   %5.0f GB/s"
          % (N, dt*1e6, fl/dt/1e12, gb/dt))
    del w, x

print("=== C. 我们模型的实际 FLOP ===")
P = 0.896e9
print("  decode 1 token : %.2f GFLOP  -> 39.3 TFLOPS 下 %.3f ms" % (2*P/1e9, 2*P/39.3e12*1e3))
print("  prefill 182 tok: %.1f GFLOP   -> 39.3 TFLOPS 下 %.1f ms" % (2*P*182/1e9, 2*P*182/39.3e12*1e3))
print("  权重 BF16      : %.2f GB     -> 446 GB/s 下 %.2f ms（带宽下限）" % (P*2/1e9, P*2/446e9*1e3))
