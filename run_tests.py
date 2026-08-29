"""测试启动脚本：先把 vendor/ 加到 sys.path，再启动 pytest。"""
import sys
from pathlib import Path

VENDOR = str(Path(__file__).parent / "vendor")
ROOT = str(Path(__file__).parent)

for path in (VENDOR, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# vendor/sitecustomize.py を手動で実行して plugin.sdk スタブを登録する。
import importlib.util
spec = importlib.util.spec_from_file_location("sitecustomize", Path(VENDOR) / "sitecustomize.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import pytest
sys.exit(pytest.main(sys.argv[1:]))
