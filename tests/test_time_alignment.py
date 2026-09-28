from backend.time_alignment import TimeAlignmentBuffer


def test_motion_interpolates_to_wgc_timestamp():
    alignment = TimeAlignmentBuffer(max_age_s=0.25, max_gap_s=0.25)
    alignment.add_motion(1.0, 0.0, 0.0)
    alignment.add_motion(1.2, 1.0, 2.0)
    result = alignment.sample_motion(1.1)
    assert result["available"] is True
    assert result["method"] == "interpolated"
    assert result["velocity_x"] == 0.5
    assert result["velocity_z"] == 1.0
    assert result["clock_domain"] == "monotonic"


def test_stale_and_gap_are_not_fabricated_as_zero():
    alignment = TimeAlignmentBuffer(max_age_s=0.1, max_gap_s=0.1)
    alignment.add_motion(1.0, 0.0, 0.0)
    assert alignment.sample_motion(1.2)["reason"] == "sample_stale"
    alignment.add_motion(2.0, 1.0, 1.0)
    alignment.add_motion(2.5, 2.0, 2.0)
    assert alignment.sample_motion(2.25)["reason"] == "timestamp_gap"

