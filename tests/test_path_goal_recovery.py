"""只续接世界确认因链路故障取消的路径目标，绝不重播旧操作。"""
from types import SimpleNamespace
import pytest
import _bootstrap
from yui_npc_controller.runtime.continuous_execution import ContinuousExecution
from yui_npc_controller.runtime.motion_space import MotionSpace


def make_driver():
    session=SimpleNamespace(session=7,world_id='test',estop=False,last_log_sequence=0,
        control_state='external',capabilities=['pose_stream_v2'],
        add_event_listener=lambda callback:None,_target_anchor_id=lambda key:1)
    driver=ContinuousExecution(None,None,session)
    driver.binding=dict(session=7,world_id='test',stream_id='s',pose_epoch=1)
    driver.status='running'
    driver.space=MotionSpace(origin=[0,0,0],yaw=0,scale=1)
    driver._control=lambda *a,**k:dict(points=[[0,0,0],[0,0,2]],request_seq=1)
    driver._request=lambda path,**k:dict(status='accepted',**k)
    task=driver.perform(session,prompt='walk',target_key='bench')
    return driver,session,task


@pytest.mark.parametrize('kind,reason,expected',[
    ('npc.operation_cancelled','stream_underrun',True),
    ('npc.operation_cancelled','pose_lease_lost',True),
    ('npc.operation_cancelled','replaced',False),
    ('npc.operation_cancelled','stopped',False),
    ('npc.operation_completed','arrived',False),
    ('npc.operation_failed','path_blocked',False),
])
def test_late_world_terminal_controls_resume(kind,reason,expected):
    driver,s,task=make_driver()
    driver.halt.set()
    assert driver.resumable_path_goal() is None
    driver.ingest(dict(session=7,world_id='test',npc='yui',log_seq=1,
        type=kind,reason=reason,op_id=task['op_id']))
    assert bool(driver.resumable_path_goal()) is expected
    if expected:
        driver.halt.clear()
        goal=driver.resumable_path_goal()
        next_task=driver.perform(s,**goal['arguments'])
        assert next_task['op_id']!=task['op_id']
    driver.stop()
    assert driver.resumable_path_goal() is None


def test_new_intent_and_foreign_receipt_cannot_revive_previous_goal():
    driver,s,task=make_driver()
    driver.ingest(dict(session=8,world_id='test',npc='yui',log_seq=99,
        type='npc.operation_cancelled',reason='stream_underrun',op_id=task['op_id']))
    assert driver.resumable_path_goal() is None
    driver.perform(s,prompt='wave')
    driver.ingest(dict(session=7,world_id='test',npc='yui',log_seq=100,
        type='npc.operation_cancelled',reason='stream_underrun',op_id=task['op_id']))
    assert driver.resumable_path_goal() is None


def test_late_older_cancellation_cannot_override_completion():
    driver,s,task=make_driver()
    driver.halt.set()
    for seq,kind in [(3,'npc.operation_completed'),(2,'npc.operation_cancelled')]:
        driver.ingest(dict(session=7,world_id='test',npc='yui',log_seq=seq,
            type=kind,reason='stream_underrun',op_id=task['op_id']))
    assert driver.resumable_path_goal() is None


def test_backend_submits_recovery_once_and_new_intent_discards_pending():
    from unittest.mock import Mock
    from yui_npc_controller.runtime.motion_backend import MotionBackend,MotionBackendConfig
    backend=MotionBackend(MotionBackendConfig(enabled=True),clock=lambda:1)
    backend._health=dict(ready=True,protocol='yui-motion/1',continuous_protocol='yui-motion-stream/2',instance='model')
    backend._seen=1
    driver,s,task=make_driver()
    driver.perform=Mock(return_value=dict(status='unknown',error='timeout'))
    backend._continuous=driver
    backend._continuous_instance=('model','test',7)
    backend.configure_continuous(SimpleNamespace(motion_recovery_blocked=False),s)
    goal=dict(instance=('model','test',7),op_id=task['op_id'],arguments=dict(prompt='walk',target_key='bench'))
    backend._resume_goal=goal
    backend.refresh_continuous();backend.refresh_continuous()
    assert driver.perform.call_count==1
    assert backend.execution_status()['path_recovery']['status']=='unknown'
    backend._resume_goal=goal
    backend.perform(s,prompt='wave')
    backend.refresh_continuous()
    assert driver.perform.call_count==2


@pytest.mark.parametrize('attempts,can_resume',[(2,True),(3,False)])
def test_surface_retry_budget_clears_stale_goal(monkeypatch,attempts,can_resume):
    from unittest.mock import Mock
    from yui_npc_controller.runtime.motion_backend import MotionBackend,MotionBackendConfig
    import yui_npc_controller.runtime.continuous_execution as execution_module
    backend=MotionBackend(MotionBackendConfig(enabled=True),clock=lambda:100)
    backend._health=dict(ready=True,protocol='yui-motion/1',continuous_protocol='yui-motion-stream/2',instance='model')
    backend._seen=100
    driver,session,task=make_driver()
    session.capabilities+=['pose_link_recovery_v1','pose_surface_recovery_v1']
    driver.status='failed';driver.error='obstacle';driver.closed.set()
    goal=dict(op_id=task['op_id'],arguments=dict(prompt='walk',target_key='bench'))
    driver.resumable_path_goal=Mock(return_value=goal)
    driver.close=Mock()
    transport=SimpleNamespace(motion_recovery_blocked=False,recover_motion_link=lambda:True)
    backend.configure_continuous(transport,session)
    backend._continuous=driver;backend._continuous_instance=('model','test',7)
    backend._surface_retry_count=attempts
    # 残留待恢复意图也必须在耗尽预算时清理，不能被下一会话复活。
    backend._resume_goal=dict(goal,instance=('model','test',7))
    replacement=Mock()
    monkeypatch.setattr(execution_module,'ContinuousExecution',Mock(return_value=replacement))
    backend.refresh_continuous()
    assert bool(backend._resume_goal) is can_resume
    if not can_resume:
        assert backend._resume_result['error']=='surface_recovery_exhausted'
    replacement.start.assert_called_once()
