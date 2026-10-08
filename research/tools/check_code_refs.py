# -*- coding: utf-8 -*-
"""校验文档里的 `文件.py:行号` 引用是否仍指向它声称的符号。

为什么需要这个：C20 登记过「权威矩阵自己的代码行号大面积失效」，修完一次后又复发
（2026-10-08 实测抓到 3 处）。行号会随任何代码改动漂 —— 所以要么改成符号名，
要么让门禁每次跑一遍。

判据：把 `文件.py:行号` 前后 120 字符窗口里的反引号标识符当作"期望符号"，
      与目标行 ±2 行窗口做匹配。对上 = OK，对不上 = 报 MISS 供人工裁决。
"""
from __future__ import annotations

import pathlib
import re
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = re.compile(r"([A-Za-z0-9_/]+\.py):(\d+)")
SYM = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})(?:\(\))?`")
SKIP_PARTS = {".venv", ".git", "build", "deps", "__pycache__"}


def locate(name: str) -> pathlib.Path | None:
    """按 basename 在已知源码根下找文件。

    ⚠️ 不要用 `ROOT.rglob(basename)` —— 这个仓库有 `.slam_probe`（69,614 文件）、
    `.venv`、`vendor`、`build`，全树 rglob 会跑到分钟级（实测踩到）。
    只在源码根里搜，且带深度上限。
    """
    p = ROOT / name
    if p.exists():
        return p
    base = pathlib.Path(name).name
    for root in ("backend", "research", "tests", "tools", ".tmp", "packaging", "ui", "."):
        d = ROOT / root
        if not d.is_dir():
            continue
        try:
            for c in d.glob(f"**/{base}"):
                if not (SKIP_PARTS & set(c.parts)):
                    return c
        except OSError:
            continue
    return None


def check(doc: pathlib.Path) -> tuple[int, int, list[str]]:
    text = doc.read_text(encoding="utf-8", errors="replace")
    ok = miss = 0
    detail: list[str] = []
    for m in REF.finditer(text):
        fname, lineno = m.group(1), int(m.group(2))
        p = locate(fname)
        if p is None:
            detail.append(f"[文件缺失] {fname}:{lineno}")
            miss += 1
            continue
        src = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if lineno > len(src):
            detail.append(f"[越界] {fname}:{lineno}（共 {len(src)} 行）")
            miss += 1
            continue
        lo, hi = max(0, m.start() - 120), min(len(text), m.end() + 120)
        syms = [s for s in SYM.findall(text[lo:hi]) if not s.endswith(".py")]
        if not syms:
            continue
        window = "\n".join(src[max(0, lineno - 3): lineno + 2]).lower()
        hit = [s for s in syms if s.lower() in window]
        if hit:
            ok += 1
        else:
            miss += 1
            detail.append(
                f"[MISS] {fname}:{lineno} 期望 {syms[:3]} "
                f"实际: {src[lineno - 1].strip()[:56]}"
            )
    return ok, miss, detail


def main() -> int:
    targets = [ROOT / "ROADMAP.md", *sorted((ROOT / "Docs").glob("*.md"))]
    total_miss = 0
    for t in targets:
        if not t.exists():
            continue
        ok, miss, detail = check(t)
        if miss == 0 and ok == 0:
            continue
        rel = t.relative_to(ROOT).as_posix()
        flag = "OK" if miss == 0 else f"{miss} 处待核"
        print(f"{rel:<44} 对上 {ok:>3} / {flag}")
        for d in detail:
            print(f"      {d}")
        total_miss += miss
    print()
    print(f"合计待核: {total_miss}")
    return 0  # 行号会漂，先作软报告；确认稳定后再升为硬门禁


if __name__ == "__main__":
    raise SystemExit(main())
