import pytest

from backend.time_alignment import CLOCK_DOMAIN, TimeAlignmentBuffer


def test_motion_interpolates_to_wgc_timestamp():
    alignment = TimeAlignmentBuffer(max_age_s=0.25, max_gap_s=0.25)
    alignment.add_motion(1.0, 0.0, 0.0)
    alignment.add_motion(1.2, 1.0, 2.0)
    result = alignment.sample_motion(1.1)
    assert result["available"] is True
    assert result["method"] == "interpolated"
    # 1.0 / 1.1 / 1.2 在 float64 中都不精确：插值比例 (1.1-1.0)/(1.2-1.0) 正确舍入后的实数结果
    # 就是 0.5000000000000006（用 Fraction 验证过：这是实数比值的正确舍入，不是实现的精度损失），
    # 无论写成 ratio 还是重心插值都得到同一个值。所以精确相等在 float 下不可达，这里按容差比较：
    # rel=1e-9 比 float64 舍入误差（~1e-16）宽 7 个数量级，仍能捕捉真正的插值错误。
    assert result["velocity_x"] == pytest.approx(0.5, rel=1e-9)
    assert result["velocity_z"] == pytest.approx(1.0, rel=1e-9)
    assert result["clock_domain"] == CLOCK_DOMAIN


def test_stale_and_gap_are_not_fabricated_as_zero():
    alignment = TimeAlignmentBuffer(max_age_s=0.1, max_gap_s=0.1)
    alignment.add_motion(1.0, 0.0, 0.0)
    assert alignment.sample_motion(1.2)["reason"] == "sample_stale"
    alignment.add_motion(2.0, 1.0, 1.0)
    alignment.add_motion(2.5, 2.0, 2.0)
    assert alignment.sample_motion(2.25)["reason"] == "timestamp_gap"

