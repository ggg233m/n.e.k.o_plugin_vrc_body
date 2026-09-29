# -*- coding: utf-8 -*-
"""文档体检：把「文档漂移」从"靠人记得查"变成一条命令。

为什么需要它
------------
本仓库 2026-09 做过一次 ``.slam_probe/`` → ``research/`` 的迁移，之后文档与代码长期漂移：
30 份文档里出现过 **48 个不存在的 .py 文件名**、**81 处失效路径**、
**一句假断言抄在 6 个地方**。这些都不是靠通读发现的，是靠**可复算的判据**发现的。

本工具把当时那批一次性脚本固化成常设检查。设计上分两类，因为它们的**判据强度不同**：

**A. 默认跑（硬门禁）——markdown 结构完整性。** 判据客观、无歧义，可以 fail：
  1. ``**`` 全文件计数为奇数 —— 从失配处起整份文档加粗状态错乱（渲染崩坏）
  2. 连续句号 ``。。`` —— 拼接/替换留下的痕迹
  3. ``[text](target)`` 的 target 里出现 ASCII ``()`` —— 会在括号处提前闭合链接

**B. ``--refs`` 才跑（软报告）——引用存在性。** 这类**不能当门禁**：
本仓库的历史实验报告**合法地**引用已删除的模块与旧路径（那是当时的实况，是证据）。
所以这里只产出**候选清单**供人工裁决，默认不影响退出码。

用法
----
    .venv/Scripts/python.exe research/tools/doc_health.py
    .venv/Scripts/python.exe research/tools/doc_health.py --refs
    .venv/Scripts/python.exe research/tools/doc_health.py --refs --json .tmp/doc_health.json

退出码：0 = A 类干净（或只命中已登记项）；1 = A 类有新问题。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 扫描范围：Docs/ 全部 + 根目录 + backend 的说明文档
DOC_GLOBS = ["Docs/*.md", "*.md", "backend/*.md", "research/**/*.md"]

# vendored / 构建目录不参与"文件是否存在"的索引
SKIP_DIR_PARTS = {".venv", ".git", "__pycache__", "deps", "build-ninja", ".buildtmp",
                  ".zcinline-test", "boost-build-obj", "boost-install", "node_modules"}
SKIP_PREFIX = ("boost-", "glew-")

# A 类里**已登记、已知、待用户裁决**的项目。命中的不再计入失败，但会在报告里显式列出。
# 登记理由必须写在注释里 —— 不要用它来"让检查变绿"。
KNOWN_MD_ISSUES = {
    # 第 3 行只有 3 个 `**`（奇数），导致全文件 167 个 `**` 失衡。
    # 两种改法都说得通（只强调"永远" / 加粗整个分句），按仓库"不猜"惯例留给作者定。
    # 见 Docs/文档勘误与过时清单（2026-09-29）.md §7.8
    "Docs/丢跟踪根因-图优化撕碎地图几何（2026-09-23）.md": "bold_parity: 待作者选改法（勘误 §7.8）",
}


def doc_files() -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in DOC_GLOBS:
        for p in REPO.glob(pattern):
            if p.is_file():
                seen[p] = None
    return sorted(seen)


def existing_py() -> dict[str, list[str]]:
    """仓库内实际存在的 .py 索引：basename -> 相对路径列表。"""
    index: dict[str, list[str]] = {}
    for path in REPO.rglob("*.py"):
        if set(path.parts) & SKIP_DIR_PARTS:
            continue
        if any(any(part.startswith(pre) for pre in SKIP_PREFIX) for part in path.parts):
            continue
        if "ORB_SLAM3" in path.parts and "Thirdparty" in path.parts:
            continue
        index.setdefault(path.name, []).append(path.relative_to(REPO).as_posix())
    return index


_FENCE = re.compile(r"^\s*```")
_INLINE_CODE = re.compile(r"`[^`]*`")


def _renderable_lines(text: str) -> list[str]:
    """返回"会被 markdown 渲染成样式"的那些行，行号与原文件一一对应。

    必须**先剔除围栏代码块与行内代码 span**再数 `**` —— 否则任何一份
    "讲解 markdown 语法"的文档都会误报：写成 `` `**` `` 时那两个星号
    出现在行内代码里，**不构成加粗**。这一条是踩过的坑：
    本工具第一版就因此把新写的 `文档清理方法（2026-09-29）.md` 判为 FAIL。
    围栏内的行用空串占位，以保持行号不错位。
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else _INLINE_CODE.sub("", line))
    return out


def check_markdown(path: Path, text: str) -> list[dict]:
    """A 类：markdown 结构完整性。"""
    out: list[dict] = []
    lines = _renderable_lines(text)

    stars = sum(ln.count("**") for ln in lines)
    if stars % 2:
        first_odd = next((i for i, ln in enumerate(lines, 1) if ln.count("**") % 2), None)
        out.append({
            "kind": "bold_parity",
            "severity": "error",
            "detail": "可渲染文本里 `**` 计数 %d（奇数）；首个奇数行 L%s —— 从该行起加粗状态错乱"
                      % (stars, first_odd),
        })

    for i, ln in enumerate(lines, 1):
        if "。。" in ln:
            out.append({"kind": "double_period", "severity": "warn",
                        "detail": "L%d 出现连续句号 `。。`" % i})
        for m in re.finditer(r"\]\(([^)]*)\)", ln):
            target = m.group(1)
            if "(" in target or ")" in target:
                out.append({"kind": "risky_link", "severity": "warn",
                            "detail": "L%d 链接目标含 ASCII 括号，会提前闭合：%s"
                                      % (i, target[:60])})
    return out


def check_refs(files: list[Path], index: dict[str, list[str]]) -> dict:
    """B 类：引用存在性（只产候选，不作门禁）。"""
    py_mention: dict[str, set[str]] = {}
    path_mention: dict[str, set[str]] = {}
    tool_mention: dict[str, set[str]] = {}

    code_text = ""
    for p in (REPO / "tool_defs.py", REPO / "__init__.py"):
        if p.exists():
            code_text += p.read_text(encoding="utf-8", errors="replace")
    # 工具名只可能注册在**插件面**：仓库根 + backend/。
    # 判据是"这个名字在某处插件代码里出现过" —— 面板专属命（如 body_freeze）**不在
    # tool_defs.py 的注册表里**，所以不能只查注册表，否则会误报（这一条是踩过的坑：
    # 我第一版只扫 tool_defs.py + __init__.py 的 "name": 模式，误报了 body_freeze）。
    #
    # 范围刻意收窄，且**用 list + join 而不是 +=**：
    # 第一版写成 `all_code += p.read_text(...)` 循环累加，在 4000+ 个文件上是 O(n²)
    # 的字符串拷贝（≈120 GB memcpy），实测会卡死到超时 —— 用 faulthandler 才定位到。
    chunks: list[str] = [code_text]
    for pattern in ("*.py", "backend/**/*.py"):
        for p in REPO.glob(pattern):
            if not p.is_file() or set(p.parts) & SKIP_DIR_PARTS:
                continue
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                chunks.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    all_code = "".join(chunks)

    rx_py = re.compile(r"\b([A-Za-z0-9_]+\.py)\b")
    rx_path = re.compile(r"((?:backend|research|Docs|tests)[/\\][A-Za-z0-9_./\\-]+\.(?:py|md|toml|json))")
    rx_tool = re.compile(r"`((?:body|vrc|world)_[a-z0-9_]+)`")

    for p in files:
        rel = p.relative_to(REPO).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in rx_py.finditer(text):
            name = m.group(1)
            if name not in index:
                py_mention.setdefault(name, set()).add(rel)
        for m in rx_path.finditer(text):
            raw = m.group(1)
            if not (REPO / raw.replace("\\", "/")).exists():
                path_mention.setdefault(raw, set()).add(rel)
        for m in rx_tool.finditer(text):
            name = m.group(1)
            if name not in all_code:
                tool_mention.setdefault(name, set()).add(rel)

    return {
        "missing_py_basenames": {k: sorted(v) for k, v in sorted(py_mention.items())},
        "missing_relative_paths": {k: sorted(v) for k, v in sorted(path_mention.items())},
        "missing_tool_names": {k: sorted(v) for k, v in sorted(tool_mention.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--refs", action="store_true", help="额外跑引用存在性检查（软报告）")
    ap.add_argument("--json", default="", help="把结果写到该路径")
    args = ap.parse_args()

    files = doc_files()
    index = existing_py()

    md_results: dict[str, list[dict]] = {}
    hard_fail = 0
    acknowledged: list[str] = []
    for p in files:
        rel = p.relative_to(REPO).as_posix()
        issues = check_markdown(p, p.read_text(encoding="utf-8", errors="replace"))
        if not issues:
            continue
        is_known = rel in KNOWN_MD_ISSUES
        if is_known:
            acknowledged.append("%s  (%s)" % (rel, KNOWN_MD_ISSUES[rel]))
        else:
            hard_fail += len([i for i in issues if i["severity"] == "error"]) or len(issues)
        md_results[rel] = [dict(i, acknowledged=is_known) for i in issues]

    print("[A] markdown 结构完整性 —— 扫描 %d 份文档" % len(files))
    if not md_results:
        print("    全部干净")
    for rel, issues in md_results.items():
        tag = "已登记" if issues[0].get("acknowledged") else "**新问题**"
        print("    %-58s %s" % (rel, tag))
        for i in issues:
            print("        [%s] %s" % (i["severity"], i["detail"]))
    if acknowledged:
        print("    已登记项（不计入失败，请勿遗忘）：")
        for a in acknowledged:
            print("        - %s" % a)

    report: dict = {"files_scanned": len(files), "markdown": md_results}

    if args.refs:
        refs = check_refs(files, index)
        report["refs"] = refs
        print()
        print("[B] 引用存在性 —— 候选清单（历史报告合法引用旧路径/已删模块，需人工裁决）")
        print("    *.py 提到但仓库无同名文件: %d 个" % len(refs["missing_py_basenames"]))
        for k, v in list(refs["missing_py_basenames"].items())[:12]:
            print("        %-32s <- %s" % (k, ", ".join(v[:3])))
        if len(refs["missing_py_basenames"]) > 12:
            print("        ... 其余见 --json 输出")
        print("    相对路径提到但不解析: %d 个" % len(refs["missing_relative_paths"]))
        for k, v in list(refs["missing_relative_paths"].items())[:8]:
            print("        %-52s <- %s" % (k, ", ".join(v[:2])))
        print("    工具名提到但代码里从未出现: %d 个" % len(refs["missing_tool_names"]))
        for k, v in refs["missing_tool_names"].items():
            print("        %-32s <- %s" % (k, ", ".join(sorted(v))))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n已写入 %s" % out)

    print()
    print("[A] 结论：%s" % ("PASS" if hard_fail == 0 else "FAIL（%d 处新问题）" % hard_fail))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
