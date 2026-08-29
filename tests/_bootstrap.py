"""绕过依赖宿主 SDK 的插件入口，只装载纯 Python 运行时。"""

from pathlib import Path
import sys
import types


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# 把 yui_npc_controller 目录本身加入 sys.path，
# 这样 `from runtime.yui_protocol import ...` 可以直接解析，
# 同时 `yui_npc_controller.runtime.*` 也能通过下面的包注册解析。
# 注意：不加项目父目录，确保测试不会意外导入相邻项目或宿主入口。
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# 注册顶层包 yui_npc_controller，__path__ 指向本目录，
# 使得 yui_npc_controller.runtime.* 可以被 import 找到。
if "yui_npc_controller" not in sys.modules:
    package = types.ModuleType("yui_npc_controller")
    package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
    package.__spec__ = None  # type: ignore[assignment]
    sys.modules["yui_npc_controller"] = package
