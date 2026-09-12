"""路径条件的速度/转向边界，避免起步第一帧硬转弯和长步。"""
import math
import pytest
import _bootstrap
from integrations.motion_service.motion_modes import path_profile,sample_profile
from integrations.motion_service.timeline import MotionTimeline
from tests.test_motion_timeline import Generator


def test_turn_then_accelerate_and_brake_with_bounded_velocity():
    plan=dict(path=[[0,0],[1,0],[1,1]],max_speed=.5,start_heading=0.)
    profile=path_profile(plan);dt=.005
    values=[sample_profile(plan,profile,i*dt) for i in range(math.ceil(profile[1]/dt)+3)]
    assert values[0]==([0,0],0.)
    assert values[-1][0]==pytest.approx([1,1])
    speeds=[]
    for (a,h),(b,j) in zip(values,values[1:]):
        speeds.append(math.dist(a,b)/dt)
        assert speeds[-1]<=.50001
        assert abs(j-h)/dt<=math.radians(120)+1e-6
        assert (abs(b[1])<1e-6 and 0<=b[0]<=1) or (abs(b[0]-1)<1e-6 and 0<=b[1]<=1)
    for a,b in zip(speeds,speeds[1:]):assert abs(a-b)/dt<=.75001
    assert sample_profile(plan,profile,.3)[0]==[0,0]


def test_world_arrival_duration_includes_turn_and_acceleration():
    t=MotionTimeline(Generator(),lambda *_:{})
    plan=dict(mode='path',origin_xz=[0,0],path=[[0,0],[1,0]],max_speed=.5)
    op=t.update('walk',plan,duration_s=2)
    assert op['end_frame']/20>3


def test_invalid_new_path_does_not_cancel_previous_intent():
    t=MotionTimeline(Generator(),lambda *_:{})
    op=t.update('idle',{});t.generate()
    with pytest.raises(ValueError):t.update('walk',dict(mode='path',path=[[0,0],[1,0]],max_speed=0))
    assert t.records[op['op_id']]['status']=='accepted' and t.version==1


def test_path_uses_sparse_constraints_but_idle_remains_fixed():
    from types import SimpleNamespace
    from integrations.motion_service.motion_modes import compile_plan
    intent=SimpleNamespace(start_frame=0,constraints=dict(mode='path',origin_xz=[0,0],path=[[0,0],[0,4]],max_speed=.8))
    data=compile_plan(intent,0,40,20)
    assert data['root_indices']==list(range(9,40,10))
    assert len(data['root_xz'])==40
    intent.constraints=dict(mode='idle',origin_xz=[0,0])
    assert 'root_indices' not in compile_plan(intent,0,40,20)


def test_approved_corner_keeps_moving_with_continuous_heading_and_bounded_speed():
    plan=dict(path=[[0,0],[0,2],[2,2]],corner_trims=[0,.4,0],max_speed=.8,start_heading=0.)
    profile=path_profile(plan);dt=.002
    values=[sample_profile(plan,profile,i*dt) for i in range(math.ceil(profile[1]/dt)+2)]
    corner_speeds=[];speeds=[]
    for (a,h),(b,j) in zip(values,values[1:]):
        speed=math.dist(a,b)/dt;speeds.append(speed)
        assert speed<=.80001
        assert abs(j-h)/dt<=math.radians(120)+.01
        # 圆滑曲线不离开世界批准的原折线走廊。
        assert min(abs(b[0]),abs(b[1]-2))<=.101
        if .05<b[0]<.35 and 1.65<b[1]<1.95:corner_speeds.append(speed)
    assert corner_speeds and min(corner_speeds)>.2
    assert max(abs(a-b)/dt for a,b in zip(speeds,speeds[1:]))<.76
    assert values[-1][0]==pytest.approx([2,2])


def test_unapproved_corner_still_stops_and_tiny_sections_are_finite():
    plan=dict(path=[[0,0],[0,2],[2,2],[2,2.01]],corner_trims=[0,.4,0,0],max_speed=.8,start_heading=0.)
    profile=path_profile(plan)
    assert len(profile[0])==2 and math.isfinite(profile[1])
    second=profile[0][1]
    assert sample_profile(plan,profile,second['start']+.1)[0]==[2,2]


def test_unverified_or_oversized_corner_metadata_is_rejected():
    from integrations.motion_service.motion_modes import validate_plan
    plan=dict(mode='path',origin_xz=[0,0],path=[[0,0],[0,1],[1,1]],max_speed=.8)
    for trims in ([0,1,0],[.1,0,0],[0,float('nan'),0],[0,.2]):
        with pytest.raises(ValueError,match='invalid_corner_trims'):
            validate_plan(dict(plan,corner_trims=trims))
