"""N.E.K.O 插件包的最小静态冒烟测试。"""

from pathlib import Path


def test_plugin_manifest_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "plugin.toml").read_text(encoding="utf-8")
    assert 'id = "yui_npc_controller"' in text
    assert 'entry = "plugin.plugins.yui_npc_controller:YuiNpcControllerPlugin"' in text
    assert (root / "config.example.toml").is_file()
