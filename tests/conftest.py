"""pytest 配置：把 yui_npc_controller 作为独立顶层包加载。"""

from pathlib import Path
import sys
import types

_TESTS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _TESTS_DIR.parent

# importlib 模式不会自动加入 tests/，这里显式加入。
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# 将 yui_npc_controller 目录本身加入 sys.path。
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# 先把 plugin.sdk.plugin 测试替身注入 sys.modules。
# NekoPluginBase 必须是类，否则插件入口无法继承。
if "plugin.sdk.plugin" not in sys.modules:
    for _mod_name in ("plugin", "plugin.sdk", "plugin.sdk.plugin"):
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = types.ModuleType(_mod_name)

    def _deco(*a, **kw):
        return a[0] if (len(a) == 1 and callable(a[0]) and not kw) else (lambda f: f)

    class _NekoPluginBase:
        def __init__(self, ctx=None): ...

    class _Ok:
        def __init__(self, value=None): self.value = value

    class _Err:
        def __init__(self, message=""): self.message = message

    class _UI:
        def __getattr__(self, _n): return _deco
        def __call__(self, *a, **kw): return _deco(*a, **kw)

    _sdk = sys.modules["plugin.sdk.plugin"]
    _sdk.NekoPluginBase = _NekoPluginBase  # type: ignore[attr-defined]
    _sdk.Ok = _Ok  # type: ignore[attr-defined]
    _sdk.Err = _Err  # type: ignore[attr-defined]
    _sdk.ui = _UI()  # type: ignore[attr-defined]
    for _name in ("lifecycle", "llm_tool", "neko_plugin", "plugin_entry"):
        setattr(_sdk, _name, _deco)

# 注册顶层包，使 yui_npc_controller.runtime.* 可以正常导入。
if "yui_npc_controller" not in sys.modules:
    _pkg = types.ModuleType("yui_npc_controller")
    _pkg.__path__ = [str(_PACKAGE_ROOT)]  # type: ignore[attr-defined]
    _pkg.__spec__ = None  # type: ignore[assignment]
    sys.modules["yui_npc_controller"] = _pkg
