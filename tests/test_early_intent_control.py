"""候选尚未生成时可声明切换，但不能越过未确认旧帧或重复续租。"""
from types import SimpleNamespace
import pytest
import _bootstrap
from integrations.motion_service.timeline import MotionTimeline
from tests.test_motion_timeline import Generator
from yui_npc_controller.runtime.continuous_execution import ContinuousExecution


def test_control_waits_for_old_receipt_and_does_not_advance_timeline():
    timeline=MotionTimeline(Generator(),lambda *args:{},stream_id='a'*32)
    timeline.update('idle',{'mode':'idle'},duration_s=None)
    descriptor=timeline.pending_control()
    assert descriptor['first_frame']==0 and descriptor['version']==1
    timeline.outstanding={'first_frame':0}
    assert timeline.pending_control() is None
    timeline.outstanding=None;timeline.reserved_frame=4
    assert timeline.pending_control() is None
    timeline.reserved_frame=0;timeline.future.append({'first_frame':0})
    assert timeline.pending_control() is None
    timeline.future.clear()
    assert timeline.pending_control()==descriptor
    assert timeline.executed_frame==timeline.committed_frame==0
    timeline.close()
    assert timeline.pending_control() is None


def driver():
    session=SimpleNamespace(last_log_sequence=0,add_event_listener=lambda _:None)
    item=ContinuousExecution(None,None,session)
    item.binding=dict(stream_id='a'*32,world_id='test',session=9,pose_epoch=1)
    item.sender=SimpleNamespace();item.sent_frame=40;item.version=1
    return item


def description():
    return dict(stream_id='a'*32,op_id='b'*32,version=2,first_frame=40,
                intent_end_frame=120,mode='free_action',plan_request=0)


def test_announcement_sent_once_before_candidate_without_claiming_playback():
    item=driver();calls=[]
    def control(*args,**kw):
        calls.append(kw)
        return dict(op_id=kw['op_id'],version=kw['intent_version'],wire_epoch=3)
    item._control=control
    item._announce_intent(description());item._announce_intent(description())
    assert len(calls)==1 and calls[0]['first_sequence']==21
    assert item.version==2 and item.wire_epoch==3
    assert item.sent_frame==40 and item.applied_frame==0


@pytest.mark.parametrize('change',[{'first_frame':44},{'first_frame':36},{'version':0},{'stream_id':'c'*32}])
def test_invalid_boundary_never_sends(change):
    item=driver();item._control=lambda *a,**k:pytest.fail('无效边界不得发包')
    with pytest.raises(ValueError):item._announce_intent(dict(description(),**change))
    assert item.version==1


def test_failed_control_ack_does_not_install_new_epoch():
    item=driver();item._control=lambda *a,**k:dict(op_id='c'*32,version=2,wire_epoch=3)
    with pytest.raises(RuntimeError,match='mismatch'):item._announce_intent(description())
    assert item.version==1 and item.wire_epoch is None


def test_rebind_preserves_pose_and_rejects_corrupt_packet():
    from yui_npc_controller.runtime.pose_codec import encode,decode,rebind_pose_epoch
    packet=encode(9,2,21,2000,[.1,.9,.2],[[0,0,0,1]]*27,[1,0,1,0])
    result=decode(rebind_pose_epoch(packet,3))
    assert result==dict(decode(packet),epoch=3)
    broken=bytearray(packet);broken[23]^=1
    with pytest.raises(ValueError,match='crc'):rebind_pose_epoch(bytes(broken),3)
    for epoch in (0,-1,True,0x80000000):
        with pytest.raises(ValueError):rebind_pose_epoch(packet,epoch)
