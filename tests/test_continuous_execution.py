"""跨服务和适配器验证预收流水线，避免重新引入每块等待播放的串行依赖。"""
from types import SimpleNamespace
import _bootstrap
from yui_npc_controller.runtime.continuous_execution import ContinuousExecution
from integrations.motion_service.continuous import ContinuousService
from tests.test_motion_timeline import Generator


def test_path_speed_and_duration_use_world_units_for_short_legs():
    from yui_npc_controller.runtime.motion_space import MotionSpace
    import pytest
    session=SimpleNamespace(last_log_sequence=0,add_event_listener=lambda callback:None,
                            _target_anchor_id=lambda key:1,capabilities=[])
    driver=ContinuousExecution(None,None,session)
    driver.ready=lambda s:True
    driver.space=MotionSpace(origin=[0,0,0],yaw=0,scale=.7841774)
    driver._control=lambda *a,**k:dict(points=[[0,0,0],[0,0,2]],request_seq=1)
    driver._request=lambda path,**k:k
    task=driver.perform(session,prompt='Walk forward.',target_key='target',mode='path')
    assert task['plan']['max_speed']*driver.space.scale==pytest.approx(.8)
    assert task['duration_s']==pytest.approx(2.9)


def test_rejected_world_path_does_not_submit_an_intent():
    from yui_npc_controller.runtime.motion_space import MotionSpace
    session=SimpleNamespace(last_log_sequence=0,add_event_listener=lambda cb:None,_target_anchor_id=lambda key:1)
    driver=ContinuousExecution(None,None,session);driver.ready=lambda s:True
    driver.space=MotionSpace(origin=[0,0,0],yaw=0,scale=1)
    driver._control=lambda *a,**k:dict(status='failed',reason='path_ground_missing')
    driver._request=lambda *a,**k:(_ for _ in ()).throw(AssertionError('拒绝后不能创建动作'))
    assert driver.perform(session,prompt='walk',target_key='rug')['error']=='path_ground_missing'


def test_stair_route_speed_preserves_world_units_and_arrival_duration():
    from yui_npc_controller.runtime.motion_space import MotionSpace
    import pytest
    session=SimpleNamespace(last_log_sequence=0,add_event_listener=lambda cb:None,_target_anchor_id=lambda key:1,capabilities=[])
    driver=ContinuousExecution(None,None,session);driver.ready=lambda s:True
    driver.space=MotionSpace(origin=[0,2,0],yaw=0,scale=.8,surface_path=True)
    driver._control=lambda *a,**k:dict(points=[[0,2,0],[0,0,4]],request_seq=1)
    driver._request=lambda path,**k:k
    result=driver.perform(session,prompt='walk',target_key='stairs',mode='path',end_condition='arrived')
    assert result['plan']['max_speed']*.8==pytest.approx(.45)
    assert result['duration_s']==pytest.approx(4/.45+.4)

class ValidGenerator(Generator):
    ready=True
    def generate_candidate(self,*args):
        pose,sample=super().generate_candidate(*args)
        pose['local_rot_mats']=[[[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]] for _ in range(27)] for _ in range(40)]
        pose['root_positions']=[[0.,.9544,0.] for _ in range(40)]
        return pose,sample


def test_accepted_path_before_first_frame_survives_matching_stream_loss_only():
    session=SimpleNamespace(last_log_sequence=0,add_event_listener=lambda cb:None,estop=False)
    driver=ContinuousExecution(None,None,session)
    driver.path_goal=dict(op_id='pending',accepted=True,arguments=dict(prompt='walk',target_key='stairs'))
    driver.world_release=dict(reason='pose_lease_lost')
    assert driver.resumable_path_goal()['op_id']=='pending'
    driver.path_goal['accepted']=False
    assert driver.resumable_path_goal() is None

    driver.path_goal['accepted']=True
    for reason in ['stream_closed','authority_lost','explicit_stop']:
        driver.world_release=dict(reason=reason)
        assert driver.resumable_path_goal() is None
    driver.world_release=dict(reason='pose_lease_lost')
    for terminal in [dict(type='npc.operation_completed'),dict(type='npc.operation_cancelled',reason='replaced')]:
        driver.path_terminal=terminal
        assert driver.resumable_path_goal() is None
    driver.path_terminal=None
    session.estop=True
    assert driver.resumable_path_goal() is None


def test_surface_failure_requires_world_avoidance_capability_and_preserves_stop():
    session=SimpleNamespace(last_log_sequence=0,add_event_listener=lambda cb:None,estop=False,capabilities=[])
    driver=ContinuousExecution(None,None,session)
    driver.path_goal=dict(op_id='walk',accepted=True,arguments=dict(target_key='stairs'))
    driver.path_terminal=dict(type='npc.operation_cancelled',reason='foot_no_ground')
    assert driver.resumable_path_goal() is None
    session.capabilities=['pose_surface_recovery_v1']
    for reason in ['foot_no_ground','unsupported_ground','obstacle','foot_correction_limit','foot_unreachable','pelvis_correction_limit']:
        driver.path_terminal['reason']=reason
        assert driver.resumable_path_goal()['op_id']=='walk'
    driver.stop()
    assert driver.resumable_path_goal() is None


def test_three_blocks_can_be_received_before_first_application_receipt():
    binding=dict(stream_id='a'*32,session=9,pose_epoch=1,world_id='test')
    service=ContinuousService(ValidGenerator())
    service.open(dict(binding,world_receipt=dict(binding,type='npc.stream_prepared',protocol='neko-pose/2'),plan=dict(mode='idle',origin_xz=[0,0])))
    service.arm(dict(binding,world_receipt=dict(binding,type='npc.stream_armed')))
    service.step()
    session=SimpleNamespace(session=9,world_id='test',last_log_sequence=0,estop=False,
        control_state='external',capabilities=['pose_stream_v2'],add_event_listener=lambda callback:None)
    backend=SimpleNamespace(_request=lambda path,data:{'received':service.received,'ack':service.ack}[path.split('/')[-1]](data))
    driver=ContinuousExecution(backend,None,session)
    driver.binding=binding;driver.status='running'
    actor=SimpleNamespace(count=0,acks=[],stop=lambda:None)
    def send(events,lane):
        payload=[];bits=pending=0
        for event in events[1:-1]:
            pending|=((event.number<<7)|event.value)<<bits;bits+=14
            while bits>=8:
                payload.append(pending&255);pending>>=8;bits-=8
        packet=bytes(payload)
        seq=int.from_bytes(packet[11:15],'little')
        epoch=int.from_bytes(packet[7:11],'little')
        actor.count+=1
        driver.ingest(dict(binding,npc='yui',type='npc.pose_ack',pose_epoch=epoch,stream_epoch=1,
            pose_session=9,pose_sequence=seq,state='received',op_id='a'*32,log_seq=actor.count))
        return ('pose',actor.count)
    actor.send=send;actor.ack=lambda ticket,session:actor.acks.append(ticket)
    driver.sender=actor
    driver._control=lambda version,expected,**v:dict(op_id=v['op_id'],version=v['intent_version'],wire_epoch=2)
    blocks=[]
    for _ in range(3):
        b=service.pull(binding)['block'];blocks.append(b);driver._execute(b)
    assert actor.count==6 and len(actor.acks)==6
    assert service.timeline.delivered_frame==12 and service.timeline.executed_frame==0
    assert service.timeline.committed_frame==0
    first=blocks[0]
    driver.ingest(dict(binding,npc='yui',type='npc.stream_progress',log_seq=7,op_id=first['op_id'],
        version=first['version'],executed_frame=4,committed_frame=8))
    driver._process_applications()
    assert service.timeline.executed_frame==4 and service.timeline.committed_frame==8
    assert driver.applied_frame==4


def test_piggyback_progress_requires_real_receipt_identity_and_cannot_claim_future():
    import pytest
    binding=dict(stream_id='b'*32,session=9,pose_epoch=1,world_id='test')
    service=ContinuousService(ValidGenerator())
    service.open(dict(binding,world_receipt=dict(binding,type='npc.stream_prepared',protocol='neko-pose/2'),plan=dict(mode='idle',origin_xz=[0,0])))
    service.arm(dict(binding,world_receipt=dict(binding,type='npc.stream_armed')));service.step()
    block=service.pull(binding)['block']
    ack=dict(binding,type='npc.pose_ack',pose_session=9,stream_epoch=1,pose_epoch=2,
        op_id='b'*32,state='received',pose_sequence=2)
    service.received(dict(binding,world_receipt=ack,wire_epoch=2))
    item=dict(op_id=block['op_id'],version=block['version'],executed_frame=4,committed_frame=4)
    # 只有接收ACK没有实际播放字段时，绝不能当成执行完成。
    with pytest.raises(ValueError):service.ack(dict(binding,world_receipt=ack,progress_index=0))
    for extra in [dict(stream_epoch=2),dict(world_id='other'),dict(pose_sequence=1)]:
        with pytest.raises(ValueError):service.ack(dict(binding,world_receipt=dict(ack,stream_progress=[item],**extra),progress_index=0))
    assert service.timeline.executed_frame==0
    result=service.ack(dict(binding,world_receipt=dict(ack,stream_progress=[item]),progress_index=0))
    assert service.timeline.executed_frame==4


def test_exchange_reports_real_progress_and_returns_next_block_without_advancing_playback():
    import pytest
    binding=dict(stream_id='c'*32,session=9,pose_epoch=1,world_id='test')
    service=ContinuousService(ValidGenerator())
    service.open(dict(binding,world_receipt=dict(binding,type='npc.stream_prepared',protocol='neko-pose/2'),plan=dict(mode='idle',origin_xz=[0,0])))
    service.arm(dict(binding,world_receipt=dict(binding,type='npc.stream_armed')));service.step()
    first=service.pull(binding)['block']
    ack=dict(binding,type='npc.pose_ack',pose_session=9,stream_epoch=1,pose_epoch=2,
             op_id='c'*32,state='received',pose_sequence=2)
    progress=dict(binding,type='npc.stream_progress',op_id=first['op_id'],version=first['version'],executed_frame=4,committed_frame=4)
    for invalid in [dict(applications=[{}]*9),dict(pull='yes'),dict(received=[]),dict(session=10)]:
        with pytest.raises(ValueError):service.exchange({**binding,**invalid})
    assert service.timeline.delivered_frame==service.timeline.executed_frame==0
    result=service.exchange(dict(binding,received=dict(world_receipt=ack,wire_epoch=2),applications=[dict(world_receipt=progress)]))
    assert result['block']['first_frame']==4 and result['block']['end_frame']==8
    assert service.timeline.delivered_frame==service.timeline.executed_frame==4
    assert service.timeline.reserved_frame==8
    # 已执行证据可以重报；不会消费第二块，也不能伪造后续执行。
    repeated=service.exchange(dict(binding,applications=[dict(world_receipt=progress)]))
    assert repeated['block']==result['block'] and service.timeline.executed_frame==4
    with pytest.raises(ValueError):service.exchange(dict(binding,applications=[dict(world_receipt=dict(progress,executed_frame=8))]))


def test_exchange_failure_does_not_advance_host_projection():
    import pytest
    session=SimpleNamespace(last_log_sequence=0,add_event_listener=lambda _:None)
    def reject(path,data):raise ValueError('world_proof_rejected')
    driver=ContinuousExecution(SimpleNamespace(_request=reject),None,session)
    driver.binding={};driver.applied_frame=4
    driver._pending_applications=lambda:[(8,{},dict(world_receipt={}))]
    with pytest.raises(ValueError):driver._exchange(pull=False)
    assert driver.applied_frame==4
