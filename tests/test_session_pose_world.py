"""正式适配器不得把单个完成标志、旧世界或错误任务的事件当作完成。"""
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time
import pytest
import _bootstrap
from yui_npc_controller.runtime.session_pose_world import SessionPoseWorld
from yui_npc_controller.runtime.yui_protocol import MidiEvent


TASK=dict(status="accepted",op_id="a"*32,session=7,epoch=3,prompt_version=3)


def setup(terminal, *, emit_applied=True):
    listeners=[]
    session=SimpleNamespace(world_id="world",session=7,estop=False,control_state="external",
        last_log_sequence=10,add_event_listener=listeners.append,remove_event_listener=listeners.remove)
    counter=[10]
    def emit(event_type,**fields):
        counter[0]+=1
        event=dict(world_id="world",session=7,npc="yui",log_seq=counter[0],type=event_type,
                   op_id=TASK["op_id"],pose_session=7,pose_epoch=19)
        event.update(fields)
        for listener in tuple(listeners): listener(event)
    sent=[0]
    def send(events,lane):
        sent[0]+=1
        seq=sent[0]
        emit("npc.pose_ack",state="received",pose_sequence=seq)
        if seq==1 and emit_applied:
            emit("npc.pose_applied",state="applied",pose_sequence=seq)
        elif seq>1:
            if "pose" in terminal: emit("npc.pose_completed",state="succeeded",pose_sequence=seq)
            if "operation" in terminal: emit("npc.operation_completed",kind="motion",result="motion_completed")
        return ("pose",seq)
    sender=SimpleNamespace(send=send,ack=Mock(),stop=Mock())
    def arm(task,epoch):
        emit("npc.pose_armed",state="armed",pose_sequence=0)
        return ("control",1)
    world=SessionPoseWorld(session,TASK,sender,19,arm)
    return world,session,sender,emit,listeners


@pytest.mark.parametrize("terminal",[set(),{"pose"},{"operation"},{"pose","operation"}])
def test_received_frame_can_request_finish_but_cannot_replace_dual_terminal(terminal):
    world,_,sender,_,_=setup(terminal,emit_applied=False)
    world.begin(TASK)
    assert world.finish(TASK,1,.01) is None
    world.receipts.send(1,[MidiEvent("cc",15,110,1)])
    assert world.applied==0
    assert bool(world.finish(TASK,1,.01))==(terminal=={"pose","operation"})
    world.close()


def test_delayed_operation_log_after_pose_completion_still_requires_actual_event():
    world,_,_,emit,_=setup({"pose"},emit_applied=False)
    world.begin(TASK)
    world.receipts.send(1,[MidiEvent("cc",15,110,1)])
    def delayed():
        time.sleep(.08)
        emit("npc.operation_completed",kind="motion",result="motion_completed")
    thread=threading.Thread(target=delayed)
    thread.start()
    assert world.finish(TASK,1,.01)["production_receipt"]
    thread.join()
    world.close()


@pytest.mark.parametrize("terminal",[set(),{"pose"},{"operation"},{"pose","operation"}])
def test_requires_both_world_terminals_and_releases_listener(terminal):
    world,session,sender,emit,listeners=setup(terminal)
    assert world.begin(TASK)==19
    world.receipts.send(1,[MidiEvent("cc",15,110,1)])
    assert world.wait_applied(1,.01)
    receipt=world.finish(TASK,1,.01)
    complete=terminal=={"pose","operation"}
    assert bool(receipt)==complete
    if complete:
        assert receipt["operation_receipt"]["op_id"]==TASK["op_id"]
        assert receipt["production_receipt"]
    world.close()
    assert not listeners
    assert sender.stop.called!=complete


def test_stale_identity_cannot_arm_and_failure_stops_sender():
    world,session,sender,emit,_=setup(set())
    emit("npc.pose_armed",state="armed",op_id="old")
    emit("npc.pose_armed",state="armed",pose_epoch=18)
    emit("npc.pose_armed",state="armed",world_id="other")
    assert not world.armed
    emit("npc.operation_failed",err="obstacle")
    assert world.failure=="obstacle" and sender.stop.called
    assert not world.is_current()
    world.close()


def test_session_change_cuts_off_old_receipts():
    world,session,sender,emit,_=setup(set())
    session.session=8
    emit("npc.pose_armed",state="armed")
    assert not world.is_current() and sender.stop.called
    world.close()


def test_combined_receipt_needs_explicit_applied_sequence_and_identity():
    world,session,sender,emit,_=setup(set())
    assert world.begin(TASK)==19
    sender.send=lambda *args, **kwargs: ("pose",1)
    world.receipts.send(1,[MidiEvent("cc",15,110,1)])
    emit("npc.pose_ack",state="received",pose_sequence=1)
    assert world.applied==0
    emit("npc.pose_ack",state="received",pose_sequence=1,applied_sequence=1,pose_epoch=18)
    assert world.applied==0
    emit("npc.pose_ack",state="received",pose_sequence=1,applied_sequence=1)
    assert world.applied==1
    world.close()
