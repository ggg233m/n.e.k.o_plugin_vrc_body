"""状态快照晚到和 ACK 重放不得倒退控制权。"""
import _bootstrap
from yui_npc_controller.runtime.yui_session import YuiSessionState

def state_event(revision, state, seq=1, session=99):
    return dict(type='npc.state', session=session, world_id='test', log_seq=seq,
                state=state, estop=state=='estop', state_revision=revision, pos=[1,0,2], active_ops=[])

def ack_event(revision, state, seq=2, replayed=False):
    return dict(type='npc.ack', session=99, world_id='test', log_seq=seq,
                seq=seq,cmd_id=1,cmd='SET_MODE',request_hash='0000',ok=True,
                state=state,state_revision=revision,replayed=replayed)

def test_queued_state_cannot_overwrite_new_ack():
    s=YuiSessionState();s._reset_for_new_session(99)
    s.ingest(state_event(1,'action'))
    s.ingest(ack_event(3,'external'))
    s.ingest(state_event(2,'action',3))
    assert s.control_state=='external'
    s.ingest(state_event(4,'external',4))
    assert s.npc_state['state']=='external'

def test_replayed_ack_cannot_clear_estop():
    s=YuiSessionState();s._reset_for_new_session(99)
    s.ingest(state_event(10,'estop'))
    s.ingest(ack_event(11,'external',replayed=True))
    assert s.estop and s.control_state=='estop'

def test_versionless_or_other_session_state_cannot_replace_versioned_state():
    s=YuiSessionState();s._reset_for_new_session(99)
    s.ingest(state_event(5,'external'))
    old=state_event(2,'action',2);old.pop('state_revision');s.ingest(old)
    s.ingest(state_event(100,'action',3,session=98))
    assert s.control_state=='external'
    s._reset_for_new_session(100)
    s.ingest(state_event(1,'action',4,session=100))
    assert s.control_state=='action'


def test_late_snapshot_cannot_clear_newer_estop_or_restore_old_action():
    s=YuiSessionState();s._reset_for_new_session(99)
    s.ingest(state_event(10,'estop'))
    s._apply_snapshot_section('session',dict(control_state='external',estop=False,caps=['pose_stream_v2']))
    s._apply_snapshot_section('npc',dict(state='action',pos=[5,0,5]))
    assert s.estop and s.control_state=='estop' and s.npc_state['state']=='estop'
    assert 'pose_stream_v2' in s.capabilities


def test_world_terminal_prevents_queued_action_state_and_late_start_resurrection():
    s=YuiSessionState();s._reset_for_new_session(99)
    s.ingest(state_event(1,'action'))
    terminal=dict(type='npc.operation_completed',session=99,world_id='test',log_seq=2,
        op_id='motion-a',kind='motion',state_revision=3,control_state='external',result='motion_completed',elapsed_ms=4000)
    s.ingest(terminal)
    s.ingest(state_event(2,'action',3))
    s.ingest(dict(terminal,type='npc.operation_started',state_revision=2,control_state='action',log_seq=4))
    assert s.control_state=='external' and s.operations['motion-a']['status']=='succeeded'
    s.ingest(dict(terminal,type='npc.operation_cancelled',log_seq=5,reason='late_cleanup'))
    assert s.operations['motion-a']['status']=='succeeded'


def test_old_operation_completion_cannot_clear_newer_estop():
    s=YuiSessionState();s._reset_for_new_session(99)
    s.ingest(state_event(10,'estop'))
    s.ingest(dict(type='npc.operation_completed',session=99,world_id='test',log_seq=2,
        op_id='older-motion',kind='motion',state_revision=9,control_state='external',result='motion_completed'))
    assert s.estop and s.control_state=='estop'
    assert s.operations['older-motion']['status']=='succeeded'
