"""让测试与 N.E.K.O 插件 SDK 解耦 —— 在**收集阶段**就拦住，不靠运气。

问题链（逐层在 pytest 9.1.1 源码里核过，不是猜的）
--------------------------------------------------
1. `pyproject.toml` 用 `addopts = "--import-mode=importlib"`，rootdir 与 confcutdir 都是仓库根。
2. `Session.collect()` 把**仓库根**也放进待收集路径
   （`_pytest/main.py`：`paths` 从 argpath 的 parents 往上补）。
3. 根目录有 `__init__.py` ⇒ `_pytest/python.py:181 pytest_collect_directory`
   返回 `Package` 而不是 `Dir`。
4. `Package.setup()`（`_pytest/python.py:678`）会执行
   `importtestmodule(self.path / "__init__.py")` —— **早于任何测试模块的导入**。
5. 该文件若已被 pytest 从 `sys.modules` 里取到就直接返回
   （`_pytest/pathlib.py` `import_path` importlib 分支：
   "If the given module name is already in sys.modules, do not import it again."），
   **否则真的执行**根 `__init__.py` ⇒ 第 21 行
   `from plugin.sdk.plugin import ...` ⇒ `ModuleNotFoundError: No module named 'plugin'`。

为什么 `tests/_bootstrap.py` 挡不住：它替换的是 `neko_anyadance_body`，
而 pytest 执行根 `__init__.py` 用的是**另一个模块名**，且时机比测试模块更早。

⚠️ 那个模块名是**算出来的，不是常量**：根目录名 `n.e.k.o_plugin_vrc_body` 含点、
不是合法标识符 ⇒ `resolve_pkg_root_and_module_name` 抛 `CouldNotResolvePathError`
⇒ 回落到 `module_name_from_path(path, rootdir)` ⇒ `"__init__"`。
本文件因此**按同一个算式取名字**，而不是硬编码它 —— 根目录改名后仍然对。

本文件做两件事
--------------
A. 预置 `sys.modules[<根 __init__.py 的模块名>]`，让第 5 步命中缓存、不执行入口。
B. 再装 `neko_anyadance_body` 占位包（与 `tests/_bootstrap.py` 同一语义），
   使 `neko_anyadance_body.backend.*` 可导入而**不触发**插件宿主入口。

⚠️ 这是**开发期**资产：`pyproject.toml` 的 `exclude_dirs` 含 `tests`、
`exclude_files` 含 `test_*.py`；根 conftest 同理不该进发行包（已核 `neko.build` 只收运行源码）。
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent


def _root_init_module_name() -> str:
    """复刻 pytest 给「根 __init__.py」算出的模块名。

    优先问 pytest 自己（同版本同结果）；问不到就退回保守值 __init__。
    ⚠️ 不要在这里 import pytest 之外的东西：本文件在收集最早期执行。
    """
    try:
        from _pytest.pathlib import module_name_from_path
        return module_name_from_path(ROOT / "__init__.py", ROOT)
    except Exception:  # pragma: no cover - 只有 pytest 内部结构变化才会走到
        return "__init__"


_NAME = _root_init_module_name()

if _NAME not in sys.modules:
    _stub = types.ModuleType(_NAME)
    _stub.__doc__ = ("测试期占位模块：由根 conftest.py 预置，"
                     "避免 pytest 收集时执行插件宿主入口（它依赖 N.E.K.O 插件 SDK）。")
    sys.modules[_NAME] = _stub

if "neko_anyadance_body" not in sys.modules:
    _pkg = types.ModuleType("neko_anyadance_body")
    _pkg.__doc__ = "测试期占位包：__path__ 指向仓库根，不执行插件宿主入口。"
    _pkg.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
    sys.modules["neko_anyadance_body"] = _pkg
