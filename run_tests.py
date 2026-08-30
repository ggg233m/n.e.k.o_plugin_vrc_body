"""测试启动脚本：先把 vendor/ 加到 sys.path，再启动 pytest。"""
import sys
from pathlib import Path

VENDOR = str(Path(__file__).parent / "vendor")
ROOT = str(Path(__file__).parent)

for path in (VENDOR, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# 旧 vendor 可能携带测试桩；正式 sync 只保留声明的第三方依赖，因此按存在性加载。
import importlib.util
sitecustomize = Path(VENDOR) / "sitecustomize.py"
if sitecustomize.is_file():
    spec = importlib.util.spec_from_file_location("sitecustomize", sitecustomize)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

import pytest
sys.exit(pytest.main(sys.argv[1:]))
