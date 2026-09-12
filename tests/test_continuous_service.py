"""持续服务只从世界确认推进，兼容接口不能抢占现有会话。"""
import _bootstrap
import pytest
from integrations.motion_service.continuous import ContinuousService
from integrations.motion_service.motion_modes import compile_plan,validate_plan
from tests.test_motion_timeline import Generator

def fixture():
    g=Generator();g.ready=True
    s=ContinuousService(g)
    identity=dict(stream_id='a'*32,session=9,pose_epoch=1,world_id='test')
    s.open(dict(identity,world_receipt=dict(identity,type='npc.stream_prepared',protocol='neko-pose/2'),plan=dict(mode='idle',origin_xz=[0,0])))
    return s,identity

def test_preparation_does_not_claim_armed_and_requires_real_apply_identity():
    s,i=fixture();s.step();b=s.pull(i)['block']
    assert not s.status(i)['armed']
    with pytest.raises(ValueError):s.ack(dict(i,world_receipt={}))
    s.arm(dict(i,world_receipt=dict(i,type='npc.stream_armed')))
    s.received(dict(i,wire_epoch=1,world_receipt=dict(i,type='npc.pose_ack',state='received',op_id=i['stream_id'],pose_session=i['session'],pose_sequence=2,stream_epoch=1)))
    e=dict(i,type='npc.stream_progress',executed_frame=b['end_frame'],version=b['version'],op_id=b['op_id'])
    assert s.ack(dict(i,world_receipt=e))['status']=='running'
    assert s.ack(dict(i,world_receipt=e))['status']=='running'

def test_expired_or_closed_stream_cannot_be_reopened_by_same_request():
    s,i=fixture();s.seen-=3
    with pytest.raises(ValueError):s.renew(i)
    assert s.status(i)['closed']
    answer=s.open(dict(i,world_receipt=dict(i,type='npc.stream_prepared',protocol='neko-pose/2'),plan=dict(mode='idle',origin_xz=[0,0])))
    assert answer['closed']

def test_path_and_heading_follow_verified_path_and_reject_bad_modes():
    from integrations.motion_service.timeline import Intent
    plan=dict(mode='path',origin_xz=[0,0],path=[[0,0],[0,1],[1,1]],max_speed=1)
    out=compile_plan(Intent('a',1,'walk','explicit',0,80,plan),20,4,20)
    assert out['root_xz'][0]==pytest.approx([0,.5*.75*1.05**2])
    assert out['heading'][0]==pytest.approx(0.)
    with pytest.raises(ValueError):validate_plan(dict(mode='climb',origin_xz=[0,0]))
    with pytest.raises(ValueError):validate_plan(dict(plan,path=[[10,10],[0,1]]))
