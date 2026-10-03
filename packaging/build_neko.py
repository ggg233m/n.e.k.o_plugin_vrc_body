"""使用宿主官方打包器构建并在隔离目录验证安装包。"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tomllib
import zipfile


# 每次构建留下的隔离安装目录。每次都是一整份插件树 + 完整 vendor/（numpy、
# onnxruntime、opencv、PIL、dxcam、winrt 的 .pyd 全套），上白 MB；不清就会无限累积。
# 不用 try/finally 删当前目录——`package-verification.json` 的 `installed_plugin`
# 和 packaging/README 的 verify_runtime.py 流程还要用它，删了那条流程就断了。
# 改成「下次构建开始时清掉上一次的」。
_STALE_PREFIXES = ("安装验证 空格-", "profile-conflict-")


def prune_stale_install_dirs(root: Path, enabled: bool = True) -> list[str]:
    """删除上一轮构建遗留的隔离安装目录。返回被删除的目录名。"""
    if not enabled:
        return []
    build_dir = root / "build"
    if not build_dir.is_dir():
        return []
    removed: list[str] = []
    for child in sorted(build_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith(_STALE_PREFIXES):
            continue
        try:
            shutil.rmtree(child)
        except OSError as exc:            # 被占用或权限问题不该让构建失败
            print(f"⚠ 未能清理 {child}: {exc}", file=sys.stderr)
            continue
        removed.append(child.name)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True, help="N.E.K.O 源码根目录")
    parser.add_argument("--keep-install", action="store_true",
                        help="保留上一轮的隔离安装目录（默认在新构建开始时清掉）")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # 元数据探测子进程也必须使用 UTF-8，且不能在暂存树产生字节码。
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    (root / "build").mkdir(exist_ok=True)
    for name in prune_stale_install_dirs(root, enabled=not args.keep_install):
        print(f"已清理上一轮安装残留：build/{name}", file=sys.stderr)
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
        "install_dir_transient": True,
        "install_dir_note": "该目录是临时验证产物，下一次运行 build_neko.py 时会被自动清理；"
                            "verify_runtime.py 必须在本次构建后、下次构建前运行。",
        "model_checksums_verified": True, "archive_crc_verified": True,
    }
    (root / "build/package-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"package": str(out), "bytes": report["bytes"], "installed_plugin": report["installed_plugin"], "payload_hash_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
