"""执行创建与停止竞争时，不得在停止之后开始播放。"""
import threading
from unittest.mock import Mock

import _bootstrap
from yui_npc_controller.runtime.motion_dispatch import MotionDispatch
from test_motion_execution import TASK, Backend, World


def test_dispatch_runs_actual_execution_and_closes_world():
    world=World();world.close=Mock(side_effect=world.stop)
    factory=Mock(return_value=world);factory.ready.return_value=True
    dispatch=MotionDispatch(Backend(),factory)
    dispatch.start(object(),TASK)
    dispatch.execution.thread.join(2)
    assert dispatch.snapshot()["status"]=="succeeded"
    dispatch.close()
    world.close.assert_called_once()
    assert not dispatch.ready(object())


def test_stop_while_preparing_world_prevents_late_start():
    entered=threading.Event();release=threading.Event()
    world=World();world.close=Mock(side_effect=world.stop)
    def create(*args):
        entered.set();release.wait(2);return world
    factory=Mock(side_effect=create);factory.ready.return_value=True
    dispatch=MotionDispatch(Backend(),factory)
    errors=[]
    def run():
        try: dispatch.start(object(),TASK)
        except RuntimeError as exc: errors.append(str(exc))
    worker=threading.Thread(target=run);worker.start()
    assert entered.wait(1)
    dispatch.stop();release.set();worker.join(2)
    assert errors==["execution_stopped_during_prepare"]
    assert not world.sent and world.stopped
    assert dispatch.execution is None
