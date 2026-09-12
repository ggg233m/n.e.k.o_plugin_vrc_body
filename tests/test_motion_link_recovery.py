"""故障暂停与人工急停分离，恢复必须取得新回执且不能复用任务。"""
from types import SimpleNamespace
from unittest.mock import Mock,patch
import _bootstrap
from yui_npc_controller.runtime.pose_shared_port import SharedPort
from yui_npc_controller.runtime.pose_sender import PoseSender
from yui_npc_controller.runtime.yui_protocol import MidiEvent
from yui_npc_controller.runtime.motion_backend import MotionBackend,MotionBackendConfig
from yui_npc_controller.runtime.yui_transport import YuiReliableTransport


def test_timeout_only_pauses_on_both_legacy_and_new_worlds():
    for recovery in (False,True):
        sent=[]
        port=SharedPort(sent.append,7,1,('estop','pose_stop'),fault_events=('fault',) if recovery else None)
        port.submit('pose',1,['data'],0);port.step(0);port.step(.51)
        assert sent==(['data','fault'] if recovery else ['data','pose_stop'])
        port.stop('explicit_stop');port.stop('explicit_stop')
        assert sent==(['data','fault','estop'] if recovery else ['data','pose_stop','estop'])


def test_sender_manual_stop_after_fault_always_escalates_once():
    sent=[];sender=PoseSender(sent.append,7,1,pose_pipeline=True,fault_recovery=True)
    sender.fault_stop();assert sender.closed.wait(1)
    sender.stop();sender.stop();sender.close(close_sink=False)
    assert len(sent)==2 and sent[0]==MidiEvent('cc',15,113,1)
    assert sent[1].type=='note_on' and sent[1].number==127


def test_receipt_send_exception_is_fault_not_manual_estop():
    import pytest
    from yui_npc_controller.runtime.pose_receipts import PoseReceipts
    sender=Mock();sender.send.side_effect=RuntimeError('receipt_timeout')
    receipts=PoseReceipts(sender,world_id='world',session=7,epoch=1)
    with pytest.raises(RuntimeError):receipts.send(1,[])
    sender.fault_stop.assert_called_once();sender.stop.assert_not_called()


def test_same_backend_world_recovery_recreates_session_and_keeps_manual_latch():
    now=[1.]
    backend=MotionBackend(MotionBackendConfig(enabled=True),clock=lambda:now[0]);backend._seen=1
    backend._health=dict(ready=True,protocol='yui-motion/1',continuous_protocol='yui-motion-stream/2',instance='same')
    session=SimpleNamespace(world_id='world',session=7,estop=False,control_state='external',capabilities=['pose_stream_v2','pose_link_recovery_v1'])
    transport=SimpleNamespace(motion_recovery_blocked=False,recover_motion_link=Mock(return_value=False))
    backend._continuous_environment=(transport,session,lambda:False)
    old=SimpleNamespace(status='failed',error='pose_lease_lost',closed=SimpleNamespace(is_set=lambda:True),sender=SimpleNamespace(core=SimpleNamespace(reason='receipt_timeout')),close=Mock())
    backend._continuous=old;backend._continuous_instance=('same','world',7)
    with patch('yui_npc_controller.runtime.continuous_execution.ContinuousExecution') as fresh:
        backend.refresh_continuous();backend.refresh_continuous()
        assert transport.recover_motion_link.call_count==1;fresh.assert_not_called()
        now[0]=4;backend._seen=4;transport.recover_motion_link.return_value=True
        transport.motion_recovery_blocked=True;backend.refresh_continuous();fresh.assert_not_called()
        transport.motion_recovery_blocked=False;session.estop=True;backend.refresh_continuous();fresh.assert_not_called()
        session.estop=False;backend.refresh_continuous();fresh.assert_called_once()
        assert backend._continuous is not old;old.close.assert_called_once()
        assert backend._base_task is None


def test_link_probe_never_clears_estop_and_replay_is_not_recovery():
    session=SimpleNamespace(estop=False,world_id='world',session=7,control_state='external')
    transport=YuiReliableTransport(Mock(),session);transport.start_heartbeat=Mock()
    transport.send_command=Mock(return_value=SimpleNamespace(status='succeeded',ack_replayed=True))
    assert not transport.recover_motion_link()
    assert transport.send_command.call_count==1
    transport.send_command.reset_mock();transport.send_command.return_value.ack_replayed=False
    assert transport.recover_motion_link()
    assert [c.args[0] for c in transport.send_command.call_args_list]==['SNAPSHOT_REQUEST','STOP','SET_CONTROL_MODE']
    transport.motion_recovery_blocked=True;transport.send_command.reset_mock()
    assert not transport.recover_motion_link();transport.send_command.assert_not_called()
