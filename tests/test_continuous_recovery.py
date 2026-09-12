"""恢复必须经历健康边沿，新握手不能解除急停或复用旧任务。"""
from types import SimpleNamespace
from unittest.mock import Mock,patch
import _bootstrap
from yui_npc_controller.runtime.motion_backend import MotionBackend,MotionBackendConfig


def test_same_backend_recovery_requires_observed_disconnect_and_respects_estop():
    backend=MotionBackend(MotionBackendConfig(enabled=True),clock=lambda:1)
    backend._seen=1
    backend._health=dict(ready=True,protocol='yui-motion/1',continuous_protocol='yui-motion-stream/2',instance='same')
    session=SimpleNamespace(world_id='world',session=7,estop=False,control_state='external',capabilities=['pose_stream_v2'])
    backend._continuous_environment=(Mock(),session,lambda:False)
    old=Mock(status='failed')
    backend._continuous=old;backend._continuous_instance=('same','world',7)
    with patch('yui_npc_controller.runtime.continuous_execution.ContinuousExecution') as execution:
        backend.refresh_continuous();execution.assert_not_called()
        backend._health['ready']=False
        backend.refresh_continuous();old.close.assert_called_once();execution.assert_not_called()
        backend._health['ready']=True;session.estop=True
        backend.refresh_continuous();execution.assert_not_called()
        session.estop=False
        backend.refresh_continuous();execution.assert_called_once()
        execution.return_value.start.assert_called_once()
        backend.refresh_continuous();execution.assert_called_once()
        assert backend._base_task is None and backend._base_signature is None
