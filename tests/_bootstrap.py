"""Load core modules without importing the N.E.K.O SDK-dependent plugin entry point."""

from pathlib import Path
import sys
import types

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if "neko_anyadance_body" not in sys.modules:
    package = types.ModuleType("neko_anyadance_body")
    package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["neko_anyadance_body"] = package

