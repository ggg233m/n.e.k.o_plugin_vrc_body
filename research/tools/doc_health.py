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

# 扫描范围：Docs/ 全部（含 archive/ 子目录！）+ 根目录 + backend 的说明文档
# ⚠️ 2026-10-08：原来写的是 "Docs/*.md"，**不递归** —— 一旦把文档移进
# Docs/archive/，那些文件就**静默脱离门禁**。分层改造时必须同步改成 **。
DOC_GLOBS = ["Docs/**/*.md", "*.md", "backend/*.md", "research/**/*.md"]

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
    "Docs/archive/丢跟踪根因-图优化撕碎地图几何（2026-09-23）.md": "bold_parity: 待作者选改法（勘误 §7.8）",
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


def check_archive_paths(files: list[Path]) -> list[str]:
    """E 类：指向已归档文档、但漏写 `archive/` 前缀的**路径引用**。

    分层改造后最容易漏的一类：文件在 `Docs/archive/`，而别处仍写 `Docs/xxx.md`。
    纯文本搜索看不出来（裸文件名到处都是，包括索引表本身），
    所以判据必须收紧到**带 `Docs/` 前缀的路径**。

    两条豁免（都是误报源，实测踩过）：
      ① `Docs/README.md` —— 索引自身仍在现役层，与 `archive/README.md` 同名不同物；
      ② **归档件之间互引** —— 历史层内部引用旧路径是**合法的**
         （本仓规矩：历史报告不回改，只在其内部自洽即可）。
    """
    arch = {p.name for p in (REPO / "Docs" / "archive").glob("*.md")}
    bad: list[str] = []
    for p in files:
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith("Docs/archive/"):
            continue  # 豁免 ②
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r"Docs/([^`\s\)\]、，]*?\.md)", text):
            name = m.group(1)
            if name.startswith("archive/") or name not in arch:
                continue
            if name == "README.md":
                continue  # 豁免 ①
            bad.append(f"{rel} 引用了 Docs/{name}，但该文件已在 Docs/archive/{name}")
    return sorted(set(bad))


def _has_section(text: str, sec: str) -> bool:
    """目标文本里是否存在小节 `sec`。

    `sec` 可能是：
      * 中文章号 ``四``      -> 匹配 ``## 四、`` / ``## 四 `` / ``### 四、``
      * 阿拉伯全号 ``17.8``  -> 匹配 ``### 17.8``
      * 带条目的 ``四.0``    -> 回退到父号 ``四``（作者常写 ``§四.0`` 指"§四 的第 0 条"）

    判据刻意放宽到"**前缀命中**"并**逐级回退父号**：写 ``§3`` 时目标里有 ``### 3.1``
    也算指得中（作者常以父号概称整组）。
    这是有意的**假阴性优先**取向 —— 见本函数调用处的说明。

    ⚠️ 回退**不是**可有可无的宽松：第一版没有它，``§四.0``（目标真身是
    ``## 四、已知陷阱`` 下的第 0 条）被报成失效，而它其实指得中。
    """
    def match(s: str) -> bool:
        if re.fullmatch(r"[\d.]+", s):
            # 阿拉伯号：`17.8` 精确命中；`(?![\d])` 保证 `3` 不误吞 `31`
            return re.search(r"^#{2,4}\s*" + re.escape(s) + r"(?![\d])", text, re.M) is not None
        return re.search(
            r"^#{2,4}\s*" + re.escape(s) + r"\s*[、.．:：\s]", text, re.M) is not None

    if match(sec):
        return True
    # 逐级回退：`四.0` -> `四`；`17.8.1` -> `17.8` -> `17`
    s = sec
    while "." in s:
        s = s.rsplit(".", 1)[0]
        if match(s):
            return True
    return False


def check_section_pointers() -> list[str]:
    """F 类：**跨文档小节号指针**是否还指得中。

    2026-10-08 拆分踩到的坑，必须机器化才不再复发：
    `Docs/README.md` 的 §四（文档状态总表）与 §三（冲突登记表）在分层改造中
    被搬走/重编号，全仓 **32 处**指针（`Docs/README.md` §三 **C18** 一类）
    当场变成空指针 —— 而 A/C/D/E 四道门**一条都没拦住**：
      * D 类只比"文件名集合"，指针内容不在它的视野；
      * A 类只查 markdown 结构（`**` 配平、括号）；
      * E 类只查 `Docs/` 前缀路径。
    判据：形如 `` `xxx.md` §N `` 的引用，目标文件里必须真有该小节标题。
    对 `Docs/README.md` 额外要求 **§号连续**（不许缺号，缺号说明有指针会指空）。

    ⚠️ **取向是"假阴性优先"**：只报**明确指不中**的，宁可漏报也不误报。
    第一版把 `§3.1`/`§17.8-4` 一律当小节号整串匹配，一次报了 38 条、
    其中绝大多数是子小节号导致的误报（实测逐条核过）。误报的门会被绕过。

    ⚠️ **② 有个已知盲区，必须说清**：它查"该小节**在不在**"，查不出
    "**同一个号换了含义**"。本次事故正是后者 —— 拆分后 `§三` 仍在（改成了
    文档地图），旧指针 `Docs/README.md` §三 **C18** 因此**照样通过**。
    真正抓到本次回归的是 ①（`§四` 被抽走 ⇒ 一/二/三/**五** 断档）
    与 ③（冲突条目**必须**落在 `CONFLICTS.md`）。②只防"号被删"。

    豁免 `Docs/archive/`：历史层合法地指向**当时**的文档结构
    （如 `verification_report.md` 记 `README §四 4.2` 是当时的实况），
    与本仓"历史报告不回改"的规矩一致 —— 同 E 类豁免 ②。
    """
    cn = "一二三四五六七八九十"
    readme = REPO / "Docs" / "README.md"
    bad: list[str] = []

    # ① README 自身的小节号必须连续（缺 §四 就是本次事故的直接特征）
    if readme.exists():
        heads = readme.read_text(encoding="utf-8", errors="replace").splitlines()
        seen = [cn.index(m.group(1)) + 1 for l in heads if l.startswith("## ")
                if (m := re.match(r"## ([" + cn + r"])[、.]", l))]
        if seen and seen != list(range(1, len(seen) + 1)):
            bad.append(
                "Docs/README.md 小节号不连续：实际「%s」应为「%s」—— "
                "缺号说明有跨文件指针会指空（2026-10-08 §四 事故）"
                % ("".join(cn[i - 1] for i in seen), "".join(cn[i - 1] for i in range(1, len(seen) + 1)))
            )

    # ③ 冲突条目指针必须落在 CONFLICTS.md（本次事故的**直接**判据）
    #    形如 `Docs/README.md` §三 **C18** / `Docs/README.md` **C9**
    #
    #    ⚠️ 中间**只允许** 空白 + 可选 `§N`：第一版写成 `[^\n`]{0,12}?`（任意 12 字符），
    #    把 ROADMAP.md 里 `Docs/多帧融合…md` / **C29** 这种"**并列两个引用**"误当成
    #    "把 C29 指向多帧融合"——实测踩到。并列与指针在文本上必须分开。
    #
    #    ⚠️ 必须容忍 **markdown 链接形态** ``[`x.md`](x.md) §三 **C18**`` ——
    #    本仓库把反引号文件名再包一层链接是常见写法（ROADMAP.md 三处如此），
    #    只认裸反引号的版本会**漏检真实的指针回归**（变异测试实测漏掉）。
    #    链接尾巴是 ``](url)`` 而**不是** ``(url)`` —— 那个 ``]`` 必须一起吃掉，
    #    第一版漏了它，于是整条正则对链接形态静默失配（变异测试第二次实测漏掉）。
    c_pat = re.compile(
        r"`([^`\n]+?\.md)`(?:\]\([^)\n]*\))?\s*(?:§[" + cn + r"\d][\d.]*)?\s*"
        r"\*\*(C\d+(?:\s*[～~/、]\s*C?\d+)*)\*\*"
    )
    for p in doc_files():
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith("Docs/archive/"):
            continue  # 历史层豁免（它记的是当时的登记位置）
        for m in c_pat.finditer(text := p.read_text(encoding="utf-8", errors="replace")):
            target, cid = m.group(1), m.group(2)
            if target.split("/")[-1] == "CONFLICTS.md":
                continue
            bad.append(
                f"{rel} 把冲突条目 `{cid}` 指向了 {target} —— "
                f"冲突登记表已独立为 Docs/CONFLICTS.md"
            )

    # ② 全仓跨文件小节指针必须指得中
    #    形如 `x.md` §三 / `x.md` §三、 / `x.md` §17.8 / [`x.md`](x.md) §三
    pat = re.compile(r"`([^`\n]+?\.md)`(?:\]\([^)\n]*\))?\s*§([" + cn + r"\d][\d.]*)")
    for p in doc_files():
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith("Docs/archive/"):
            continue  # 历史层豁免（同 E 类豁免 ②）
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            target, sec = m.group(1), m.group(2).rstrip(".")
            # 自身引用（`本文 §二` / 同文件内）不查：改号时会一起改
            if target.split("/")[-1] == p.name:
                continue
            cands = [p.parent / target, REPO / "Docs" / target,
                     REPO / "Docs" / "archive" / target, REPO / target]
            tgt = next((c for c in cands if c.is_file()), None)
            if tgt is None:
                # ⚠️ 只报"**带目录前缀**的路径"（`Docs/x.md`）或索引自身。
                # 裸文件名（`audit/REPORT.md`、`REPORT.md`、`vrchat-cache-…md`）**不报**：
                # 仓库里同名报告有 7 份 `REPORT.md`，且外部调研稿可能只存在于
                # 会话工作区而异机器不可得 —— 对它们做"不存在"的断定必然是误报
                # （第一版就此误报过 `audit/REPORT.md` §2.4，实测踩到）。
                if target.startswith(("Docs/", "backend/", "research/")):
                    bad.append(f"{rel} 引用了不存在的文档: {target} §{sec}")
                continue
            t = tgt.read_text(encoding="utf-8", errors="replace")
            if not _has_section(t, sec):
                bad.append(f"{rel} 的指针失效: `{target}` §{sec} —— 目标文件没有该小节")
    return sorted(set(bad))


def check_facts() -> list[str]:
    """C 类：FACTS.md 里引用的 `文档名:行号` 必须真实存在。

    这是**文档维护规矩第 2 条**（"改结论必须改源头"）的机器化：
    真值单源一旦指向不存在的行，就等于把读者送进空指针。
    """
    facts = REPO / "Docs" / "FACTS.md"
    if not facts.exists():
        return ["Docs/FACTS.md 不存在 —— 真值单源缺失"]
    text = facts.read_text(encoding="utf-8", errors="replace")
    bad: list[str] = []
    # 形如 `xxx.md:123` 或 `xxx.md:12-34`
    for m in re.finditer(r"`([^`]+\.md):(\d+)(?:-(\d+))?`", text):
        name, a, b = m.group(1), int(m.group(2)), m.group(3)
        cand = [REPO / name, REPO / "Docs" / name, REPO / "Docs" / "archive" / name]
        hit = next((c for c in cand if c.exists()), None)
        if hit is None:
            bad.append(f"FACTS.md 引用的文档不存在: {name}")
            continue
        n = len(hit.read_text(encoding="utf-8", errors="replace").splitlines())
        hi = int(b) if b else a
        if a > n or hi > n:
            bad.append(f"FACTS.md 行号越界: {name}:{a}{'-'+b if b else ''} (该文件共 {n} 行)")
    return bad


def check_index_coverage() -> list[str]:
    """D 类：Docs/ 实际文档集合 vs README 索引登记项，做差集。

    ⚠️ **这就是 README §五 规矩 8 自己写了、但一直没实现的那道门。**
    2026-09-30 → 10-04 新增 11 份文档、一份都没进索引，而门禁一条告警都没触发 ——
    因为旧门禁只查 markdown 结构，**语义陈旧完全不在检查范围**。
    """
    readme = REPO / "Docs" / "README.md"
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8", errors="replace")
    registered = set(re.findall(r"^\| `(?:archive/)?([^`]+\.md)` \|", text, re.M))

    actual = {p.name for p in (REPO / "Docs").glob("*.md")}
    actual |= {p.name for p in (REPO / "Docs" / "archive").glob("*.md")}
    actual.discard("README.md")  # 索引自身不必登记

    missing = sorted(actual - registered)
    return [f"文档未登记进 README §三 索引: Docs/{n}" for n in missing]


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

    # C 类：FACTS.md 证据可追溯性（硬门禁 —— 真值单源不许指向空处）
    facts_bad = check_facts()
    report["facts"] = facts_bad
    print()
    print("[C] FACTS.md 证据可追溯性 —— 硬门禁")
    if not facts_bad:
        print("    全部可追溯")
    for b in facts_bad:
        print("    ** %s" % b)
    hard_fail += len(facts_bad)

    # D 类：文档索引覆盖差集（硬门禁 —— 规矩 8 的机器化）
    cov_bad = check_index_coverage()
    report["index_coverage"] = cov_bad
    print()
    print("[D] 文档索引覆盖 —— 硬门禁（README §五 规矩 8）")
    if not cov_bad:
        print("    实有集合与索引登记项一致")
    for b in cov_bad:
        print("    ** %s" % b)
    hard_fail += len(cov_bad)

    # E 类：归档路径前缀（硬门禁 —— 分层改造后最容易漏的一类）
    arch_bad = check_archive_paths(files)
    report["archive_paths"] = arch_bad
    print()
    print("[E] 归档路径前缀 —— 硬门禁（Docs/xxx.md 应写成 Docs/archive/xxx.md）")
    if not arch_bad:
        print("    全部正确")
    for b in arch_bad:
        print("    ** %s" % b)
    hard_fail += len(arch_bad)

    # F 类：跨文档小节指针（硬门禁 —— 2026-10-08 拆分事故的机器化）
    sec_bad = check_section_pointers()
    report["section_pointers"] = sec_bad
    print()
    print("[F] 跨文档小节指针 —— 硬门禁（`x.md` §N 必须在目标里指得中）")
    if not sec_bad:
        print("    全部指得中")
    for b in sec_bad:
        print("    ** %s" % b)
    hard_fail += len(sec_bad)

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
