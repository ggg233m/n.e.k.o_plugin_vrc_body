# -*- coding: utf-8 -*-
import torch
import torch._inductor.config as ic
ic.cpp_wrapper = True
x = torch.randn(64, 64, device='xpu', dtype=torch.bfloat16)
f = torch.compile(lambda a: torch.relu(a @ a).sum(-1))
try:
    f(x); torch.xpu.synchronize(); print('CPP_WRAPPER OK')
except Exception as e:
    print('CPP_WRAPPER FAIL')
    print(str(e)[:2500])