from types import SimpleNamespace
import threading
import _bootstrap
from yui_npc_controller.runtime.pose_lifecycle import PoseLifecycle


def setup():
    listeners=[]
    session=SimpleNamespace(world_id="world",session=7,capabilities={"pose_stream_v1"},
        control_state="external",add_event_listener=listeners.append,remove_event_listener=listeners.remove)
    calls=[]
    def attach(epoch):
        calls.append(("attach",epoch))
        return SimpleNamespace(closed=threading.Event(),stop=lambda: calls.append(("stop",)))
    transport=SimpleNamespace(attach_pose_sender=attach,detach_pose_sender=lambda: calls.append(("detach",)))
    return PoseLifecycle(transport,session),session,calls,listeners


def test_offline_does_not_take_port_and_loss_cannot_auto_resume():
    life,session,calls,listeners=setup()
    life.update(dict(ready=False))
    assert not calls
    ready=dict(ready=True,instance="service",epoch=3)
    life.update(ready)
    assert calls==[("attach",3)]
    life.update(dict(ready=False))
    life.update(ready)
    assert calls==[("attach",3),("detach",)]
    session.session=8
    life.update(ready)
    assert calls[-1]==("attach",3)
    life.close()
    assert not listeners
    before=len(calls)
    life.update(ready)
    assert len(calls)==before


def test_missing_world_capability_preserves_legacy():
    life,session,calls,listeners=setup()
    session.capabilities.clear()
    life.update(dict(ready=True,instance="service",epoch=3))
    assert not calls
    life.close()
