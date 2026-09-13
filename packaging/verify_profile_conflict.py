"""复现旧包 profile 归属冲突，验证修订包安装且不改旧配置。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.host.resolve()), str(args.host.resolve() / "plugin")]
    from neko_plugin_cli.core import inspect_package
    from plugin.server.application.plugin_cli.service import PluginCliService
    from profile_safe import omit_default_profile

    old = root / "dist/neko_anyadance_body-0.13.21-windows-x64-py311.neko-plugin"
    new = old.with_name(old.stem + "-profile-safe.neko-plugin")
    omit_default_profile(old, new)
    inspected = inspect_package(new)
    assert inspected.payload_hash_verified and not inspected.profile_names
    with zipfile.ZipFile(old) as before, zipfile.ZipFile(new) as after:
        files = [n for n in before.namelist() if n.startswith("payload/plugins/")]
        assert all(before.read(n) == after.read(n) for n in files)
        assert after.testzip() is None
    sandbox = Path(tempfile.mkdtemp(prefix="profile-conflict-", dir=root / "build"))
    plugins = sandbox / "plugins"
    profiles = sandbox / "profiles"
    legacy = profiles / "neko_anyadance_body/default.toml"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b'name = "user-kept-profile"\n')
    before_bytes = legacy.read_bytes()
    service = PluginCliService()
    original_rename = Path.rename
    rename_retries = 0
    def rename_after_scan(path: Path, target: Path):
        nonlocal rename_retries
        # Windows 扫描刚解压的 DLL 时可能短暂占用目录；仅在测试沙箱内有限重试。
        assert path.resolve().is_relative_to(sandbox)
        assert Path(target).resolve().is_relative_to(sandbox)
        for attempt in range(21):
            try:
                return original_rename(path, target)
            except PermissionError:
                if attempt == 20:
                    raise
                rename_retries += 1
                time.sleep(0.5)
    # 模拟旧版遗留、归属账本为空的目录，使用与导入端点相同的暂存安装逻辑。
    with patch.object(Path, "rename", rename_after_scan), patch("plugin.server.application.plugin_cli.service.get_install_source_manager",
               return_value=SimpleNamespace(list_entries=lambda **kwargs: [])):
        try:
            service._install_via_staging_sync(package=old, plugins_root=plugins,
                                              profiles_root=profiles, on_conflict="fail")
        except Exception as exc:
            assert "existing package profile ownership does not match" in str(exc), str(exc)
        else:
            raise AssertionError("旧包没有复现预期冲突")
        assert not (plugins / "neko_anyadance_body").exists()
        assert legacy.read_bytes() == before_bytes
        result = service._install_via_staging_sync(package=new, plugins_root=plugins,
                                                  profiles_root=profiles, on_conflict="fail")
        assert len(result.installed_plugins) == 1
        assert result.profile_dir is None
        assert legacy.read_bytes() == before_bytes
    sha = hashlib.sha256(new.read_bytes()).hexdigest()
    new.with_suffix(new.suffix + ".sha256").write_text(f"{sha}  {new.name}\n", encoding="ascii")
    report = {"package": new.name, "bytes": new.stat().st_size, "sha256": sha,
              "old_package_conflict_reproduced": True, "new_package_install_passed": True,
              "existing_profile_unchanged": True, "plugin_payload_byte_identical": True,
              "windows_rename_retries": rename_retries,
              "payload_hash_verified": True, "installed_plugin": str(plugins / "neko_anyadance_body")}
    (root / "dist/profile-safe-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
