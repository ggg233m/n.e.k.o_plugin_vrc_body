"""世界目标在缩放、转向和 X 镜像后必须回到同一个位置。"""
import math
import pytest
import _bootstrap
from yui_npc_controller.runtime.motion_space import MotionSpace


@pytest.mark.parametrize("yaw",[0,90,-90,37,180])
@pytest.mark.parametrize("scale",[.5,1,1.5])
def test_world_path_roundtrip_preserves_target_and_speed(yaw,scale):
    space=MotionSpace(origin=[10,2,-7],yaw=yaw,scale=scale)
    target=[11,2,-5]
    binding=space.verified_paths({"desk":[target]},max_speed=.5)
    assert space.world_point(binding["paths"]["desk"][0])==pytest.approx(target)
    assert binding["max_speed"]*scale==pytest.approx(.5)


def test_source_forward_matches_rotated_unity_forward():
    space=MotionSpace(origin=[0,0,0],yaw=90,scale=.5)
    assert space.source_point([1,0,0])==pytest.approx([0,2])


def test_cannot_flatten_stairs_or_unbounded_path():
    space=MotionSpace(origin=[0,0,0],yaw=0,scale=1)
    for point in ([0,.2,1],[0,0,6],[float("nan"),0,0]):
        with pytest.raises(ValueError): space.source_point(point)


def test_surface_capability_allows_model_projection_without_losing_world_plan():
    space=MotionSpace(origin=[0,0,0],yaw=0,scale=1,surface_path=True)
    assert space.source_point([0,2,1])==[0,1]
    from integrations.motion_service.motion_modes import validate_plan
    plan=validate_plan(dict(mode='path',origin_xz=[0,0],path=[[0,0],[0,1]],
        surface_path_xyz=[[0,0,0],[0,2,1]]))
    assert plan['surface_path_xyz'][1]==[0,2,1]
    with pytest.raises(ValueError):validate_plan(dict(plan,surface_path_xyz=[[0,0,0]]))
