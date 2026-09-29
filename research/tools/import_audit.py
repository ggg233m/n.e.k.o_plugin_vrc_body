# -*- coding: utf-8 -*-
"""发行面导入审计：找出 shipped 面（插件根 + ``backend/``）里所有"解析不到"的导入。

为什么需要它
------------
本仓库经历过一次把 ``.slam_probe/`` 大部分内容迁进主项目（``research/``）的搬迁。
搬迁最容易留下的不是报错，而是**跨根依赖**：代码写 ``import 某模块``，该模块
曾经在同目录、现在在另一个根里，于是运行期才失败（或者被 ``try`` 吞掉后静默降级）。

``Docs/SLAM代码地图.md`` 曾断言"``backend/`` 有 4 个模块运行时去 ``.slam_probe`` import"，
而 2026-09-29 的实扫证明该断言来自**已删除的文件**（详见
``Docs/文档勘误与过时清单（2026-09-29）.md``）。本工具就是那条实扫的可复现版本。

它做什么
--------
纯 ``ast`` 解析，**不导入任何被审计模块**，因此没有副作用、不会建连接、不加载模型。
逐个收集 ``import X`` / ``from X import`` 的顶层名，排除：

* 标准库（``sys.stdlib_module_names``）
* 仓库内本地模块（插件根与 ``backend/`` 的 ``*.py`` basename）
* 已安装的第三方（``importlib.util.find_spec``）

剩下的即"解析不到"。它们要么是可选依赖（应当被 ``try`` / ``find_spec`` 守卫，
如 ``mss`` / ``openvino`` / ``winrt``），要么是宿主提供的运行时模块
（如 ``plugin`` = N.E.K.O 宿主 SDK），要么就是**跨根依赖**。

判读
----
* ``plugin`` / ``mss`` / ``openvino`` 之类的**若干个**是正常的。
* 一旦出现 ``slam_core`` / ``relocalization`` / ``online_pose`` / ``nav_plan`` /
  ``route_executor`` / ``realtime_depth`` / ``live_mapping`` 等名字 —— 那就是
  ``.slam_probe`` 依赖回来了，发行版会在用户机器上炸。

用法
----
    .venv/Scripts/python.exe research/tools/import_audit.py
    .venv/Scripts/python.exe research/tools/import_audit.py --json .tmp/import_audit.json

退出码：0 = 只出现白名单外的可选/宿主模块；1 = 出现疑似跨根依赖（或文件不存在）。
白名单可用 ``--allow`` 覆盖。
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 这些是"解析不到但正常"的：宿主运行时 + 可选依赖（代码里都有守卫）。
DEFAULT_ALLOW = frozenset({"plugin", "mss", "openvino", "winrt", "dxcam", "ultralytics", "torch"})

# 疑似跨根依赖的信号：这些名字只应存在于 .slam_probe / research 内部。
SUSPECT_HINTS = (
    "slam_core", "relocalization", "online_pose", "nav_plan", "nav_target",
    "route_executor", "realtime_depth", "live_mapping", "metric_scale_calibrator",
    "frame_geometry", "offline_route_replay", "relocalization_observer",
    "nav_map_builder", "scale_calib", "pose_graph", "run_motion",
)


def local_module_names() -> set[str]:
    """插件根与 backend/ 的 .py basename —— 视为本地模块。"""
    names: set[str] = set()
    for directory in (REPO, REPO / "backend"):
        if directory.is_dir():
            names.update(p.stem for p in directory.glob("*.py"))
    names.discard("__init__")
    return names


def top_level_imports(path: Path) -> set[str]:
    """一个文件里的顶层导入名。语法错误时如实报告并跳过该文件。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        print(f"  !! 语法错误，跳过 {path.name}: {exc}", file=sys.stderr)
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def audit(target_dirs: list[Path]) -> dict:
    local = local_module_names()
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    files: list[Path] = []
    for directory in target_dirs:
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.py")))

    all_names: set[str] = set()
    unresolved: dict[str, list[str]] = {}
    for path in files:
        for name in top_level_imports(path):
            all_names.add(name)
            if name in stdlib or name in local:
                continue
            try:
                if importlib.util.find_spec(name) is not None:
                    continue
            except (ImportError, ValueError):
                pass
            unresolved.setdefault(name, []).append(path.relative_to(REPO).as_posix())

    return {
        "files_scanned": len(files),
        "local_modules": len(local),
        "all_top_level_imports": sorted(all_names),
        "unresolved": {k: sorted(v) for k, v in sorted(unresolved.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--json", default="", help="把结果写到该路径")
    parser.add_argument("--allow", default="", help="额外允许的模块名，逗号分隔")
    args = parser.parse_args()

    allowed = set(DEFAULT_ALLOW) | {x.strip() for x in args.allow.split(",") if x.strip()}
    result = audit([REPO, REPO / "backend"])

    print(f"扫描文件数: {result['files_scanned']}")
    print(f"本地模块名: {result['local_modules']}")
    unresolved = result["unresolved"]
    print(f"解析不到的顶层导入: {len(unresolved)}")
    for name, locations in unresolved.items():
        mark = "  *" if name not in allowed else "   "
        print(f"{mark} {name:<26} 出现于: {', '.join(locations[:6])}"
              + (f" …(共{len(locations)})" if len(locations) > 6 else ""))

    suspects = sorted(n for n in unresolved if n in SUSPECT_HINTS)
    unexpected = sorted(n for n in unresolved if n not in allowed)

    if suspects:
        print(f"\n[FAIL] 疑似跨根依赖（.slam_probe 依赖回来了）: {', '.join(suspects)}")
    if unexpected:
        print(f"\n[WARN] 白名单外的解析不到项: {', '.join(unexpected)}")
    if not suspects and not unexpected:
        print("\n[OK] 发行面没有跨根依赖；解析不到的只有宿主/可选依赖。")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**result, "allowed": sorted(allowed),
                                   "suspects": suspects, "unexpected": unexpected},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")

    return 1 if (suspects or unexpected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
