# -*- coding: utf-8 -*-
"""A* 软引导与 VLA 输入契约的回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "tools"))

from build_policy_dataset import guidance_at  # noqa: E402
from policy_text import build_guidance_text  # noqa: E402


def test_invalid_or_stale_guidance_is_kept_as_a_negative_input() -> None:
    rows = [{
        "t": 10.0,
        "valid": True,
        "waypoint": [1.0, 2.0],
        "path": [[1.0, 2.0]],
        "confidence": 0.9,
    }]
    stale = guidance_at(rows, 10.8, max_age_s=0.5)
    assert stale is not None
    assert stale["valid"] is False
    assert stale["waypoint"] == [0.0, 0.0]
    assert stale["path"] == []
    assert stale["reason"] == "guidance_stale"


def test_guidance_text_has_fixed_missing_value_contract() -> None:
    assert build_guidance_text({"requested": False, "valid": False}) \
        == " [guide] 0 0 0.00 0.00 0.00"
    assert build_guidance_text({"requested": True, "valid": True, "confidence": 0.75,
                                "pose_age_s": 0.2, "sigma_m": 0.1}) \
        == " [guide] 1 1 0.75 0.20 0.10"

