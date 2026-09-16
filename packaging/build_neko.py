"""使用宿主官方打包器构建并在隔离目录验证安装包。"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True, help="N.E.K.O 源码根目录")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # 元数据探测子进程也必须使用 UTF-8，且不能在暂存树产生字节码。
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    (root / "build").mkdir(exist_ok=True)
    # 大型模型和依赖的暂存放在项目磁盘，避免占满系统临时目录。
    tempfile.tempdir = str(root / "build")
    sys.path[:0] = [str(args.host.resolve()), str(args.host.resolve() / "plugin")]
    from neko_plugin_cli.core import build_plugin, inspect_package, install_package

    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    versions = {d.metadata["Name"]: d.version for d in metadata.distributions(path=[str(root / "vendor")])}
    assert versions, "请先将运行依赖安装到 vendor"
    (root / "DEPENDENCIES.json").write_text(json.dumps({
        "platform": "Windows x64", "python": "CPython 3.11",
        "packages": dict(sorted(versions.items())),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out = root / "dist" / f"{project['name']}-{project['version']}-windows-x64-py311.neko-plugin"
    result = build_plugin(root, out_file=out)
    from profile_safe import omit_default_profile
    safe_out = out.with_name(out.stem + "-profile-safe.neko-plugin")
    omit_default_profile(out, safe_out)
    out = safe_out
    inspected = inspect_package(out)
    assert inspected.payload_hash_verified is True
    assert not inspected.profile_names
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        assert archive.testzip() is None
        prefix = f"payload/plugins/{project['name']}/"
        for required in ("__init__.py", "plugin.toml", "pyproject.toml", "plugin.meta.json", "backend/process.py", "ui/panel.tsx", "backend/standalone_ui/index.html"):
            assert prefix + required in names, required
        for name in names:
            parts = Path(name).parts
            assert not set(parts) & {".git", ".venv", "build", "dist", "__pycache__", ".trae"}, name
            assert Path(name).name not in {"backend.settings.json", "world_memory.json", "store.db", ".env"}, name
        model_dir = prefix + "models/person_detect_v1.3_s/"
        for record in json.loads(archive.read(model_dir + "checksums.json")):
            data = archive.read(model_dir + record["file"])
            assert len(data) == record["bytes"]
            assert hashlib.sha256(data).hexdigest().lower() == record["sha256"].lower()
        reid_dir = prefix + "models/osnet_x0_25_msmt17/"
        for record in json.loads(archive.read(reid_dir + "checksums.json")):
            data = archive.read(reid_dir + record["file"])
            assert len(data) == record["bytes"]
            assert hashlib.sha256(data).hexdigest().lower() == record["sha256"].lower()
    # 仅安装到工作区新建的测试目录，不接触当前宿主插件和配置。
    install_root = Path(tempfile.mkdtemp(prefix="安装验证 空格-", dir=root / "build"))
    installed = install_package(out, plugins_root=install_root / "plugins", profiles_root=install_root / "profiles")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{sha}  {out.name}\n", encoding="ascii")
    report = {
        "package": str(out), "bytes": out.stat().st_size, "sha256": sha,
        "inspection": inspected.model_dump(mode="json"), "installation": installed.model_dump(mode="json"),
        "installed_plugin": str(install_root / "plugins" / project["name"]),
        "model_checksums_verified": True, "archive_crc_verified": True,
    }
    (root / "build/package-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"package": str(out), "bytes": report["bytes"], "installed_plugin": report["installed_plugin"], "payload_hash_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
