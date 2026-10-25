# -*- coding: utf-8 -*-
"""测试全局开关：默认禁用 LAS2 子进程。

为什么必须在 conftest 里拦：``BackendService.start()`` 会顺手调 ``navmesh.prewarm()``，
那是「后端启动就预热」的设计（生产上对，避免会话首帧等 80 s）。但测试套件里有若干
用例真的会 ``service.start()``，于是每次跑测试都会拉起一个 ``.venv_las`` 的 3.14 子进程
并在里面开始 5 min 的 torch.compile —— 测试不该有这种副作用，CI 上更是灾难。

要跑真 LAS2 的用例自己 ``mock.patch.dict(os.environ, {"NEKO_LAS_DISABLE": "0"})``，
并把 ``stereo_backend._DISABLED`` 改回 False（模块加载时读一次），再显式用
``subprocess=True`` + echo 引擎。
"""
from __future__ import annotations

import os

# 必须早于任何 backend 导入：stereo_backend 在模块加载时读一次。
os.environ.setdefault("NEKO_LAS_DISABLE", "1")
