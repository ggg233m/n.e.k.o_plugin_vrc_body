"""去除非必需的默认 profile，保留目标宿主已有配置及其归属记录。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tomllib
import zipfile


def omit_default_profile(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("输出必须使用新文件，保留原始包")
    with zipfile.ZipFile(source) as archive:
        assert "sign.toml" not in archive.namelist(), "不能修改已签名安装包"
        profiles = [n for n in archive.namelist() if n.startswith("payload/profiles/")]
        assert profiles == ["payload/profiles/default.toml"], profiles
        profile = tomllib.loads(archive.read(profiles[0]).decode("utf-8"))
        # 只允许去除本插件无业务参数的自动生成默认配置。
        assert profile == {"name": "default", "enabled_plugins": ["neko_anyadance_body"],
                           "plugin": {"neko_anyadance_body": {"enabled": True, "auto_start": False}}}
        names = [n for n in archive.namelist() if n not in profiles]
        digest = hashlib.sha256()
        for name in sorted(n for n in names if n.startswith("payload/")):
            digest.update(name.removeprefix("payload/").encode("utf-8"))
            digest.update(b"\0")
            digest.update(archive.read(name))
            digest.update(b"\0")
        metadata = archive.read("metadata.toml").decode("utf-8")
        metadata, count = re.subn(r'(?m)^hash = "[0-9a-f]+"$', f'hash = "{digest.hexdigest()}"', metadata)
        assert count == 1
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for name in names:
                output.writestr(name, metadata.encode("utf-8") if name == "metadata.toml" else archive.read(name))
