"""独立后端与宿主项目之间的集成接缝。

只有本模块知道当前项目的调度器、传输层、配置和动作片段库所在位置。换用
其他宿主项目时，只需替换此适配器，``service.py``、进程协议以及视觉/世界
状态模块都无需改变。
"""

from __future__ import annotations

from ..behavior import resolve_expression
from ..config import PluginConfig
from ..driver_log import DriverLogListener
from ..host_vmc import HostVmcController
from ..nya import ClipLibrary
from ..osc import VrchatOscBridge
from ..scheduler import BodyCommand, BodyScheduler
from ..vmc_idle import VmcIdleRelay

__all__ = [
    "BodyCommand",
    "BodyScheduler",
    "ClipLibrary",
    "DriverLogListener",
    "HostVmcController",
    "PluginConfig",
    "VmcIdleRelay",
    "VrchatOscBridge",
    "resolve_expression",
]
