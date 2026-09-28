"""运行时代码不得依赖 research/（离线工具、录制器；不进安装包）。"""

from __future__ import annotations

import ast
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RESEARCH_TOP = {"research", "tools", "slam"}
_RECORDER_MODULES = {p.stem for p in (ROOT / "research" / "recorder").glob("*.py")}


def _runtime_files():
    yield from ROOT.glob("*.py")
    yield from (ROOT / "backend").rglob("*.py")


def _bad_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in _RESEARCH_TOP or top in _RECORDER_MODULES:
                yield node.lineno, name


class ResearchIsolationTest(unittest.TestCase):
    def test_runtime_does_not_import_research(self):
        hits = [f"{p.relative_to(ROOT)}:{ln} import {n}"
                for p in _runtime_files() for ln, n in _bad_imports(p)]
        self.assertEqual(hits, [])

    def test_research_excluded_from_package(self):
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("research", cfg["tool"]["neko"]["build"]["exclude_dirs"])

    def test_no_research_code_left_at_old_locations(self):
        for old in ("tools", "slam", "audit"):
            self.assertFalse((ROOT / old).exists(), f"{old}/ 应已并入 research/")


if __name__ == "__main__":
    unittest.main()
