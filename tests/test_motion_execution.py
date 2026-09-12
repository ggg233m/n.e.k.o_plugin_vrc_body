import copy
import threading
import time

import _bootstrap
import pytest
from yui_npc_controller.runtime.motion_execution import MotionExecution
from yui_npc_controller.runtime.pose_frames import compile_chunk


TASK=dict(status="accepted",op_id="task",session=7,epoch=3,prompt_version=3)


def chunk():
    return dict(TASK,sequence=0,fps=20,final=True,pose={
        "local_rot_mats":[[[[1,0,0],[0,1,0],[0,0,1]] for _ in range(27)] for _ in range(2)],
        "root_positions":[[0,.95,0],[0,.95,0]],"foot_contacts":[[1]*4,[1]*4]})


class Backend:
    def __init__(self): self.calls=[]
    def _request(self,path,data=None):
        self.calls.append((path,data))
        return {"chunk":chunk(),"status":"awaiting_world"} if path=="/chunks" else dict(TASK,status="isolated_completed")


class World:
    def __init__(self,applied=True,completed=True):
        self.receipts=self;self.applied=applied;self.completed=completed;self.sent=[];self.stopped=False
    def is_current(self): return not self.stopped
    def begin(self,task): return 12
    def send(self,seq,events): self.sent.append((seq,events))
    def wait_received(self,*args): return True
    def wait_applied(self,*args): return self.applied
    def finish(self,task,*args): return dict(state="succeeded",op_id=task["op_id"],pose_session=7,pose_epoch=12,pose_sequence=2) if self.completed else None
    def stop(self): self.stopped=True


def execute(world):
    backend=Backend(); execution=MotionExecution(backend,world)
    execution.start(TASK); execution.thread.join(2)
    assert not execution.thread.is_alive()
    return execution,backend


def test_received_pose_is_not_application_or_completion():
    for world in (World(applied=False),World(completed=False)):
        execution,backend=execute(world)
        assert execution.snapshot()["status"]=="failed"
        assert world.stopped
        assert not any(path=="/ack" for path,_ in backend.calls)


def test_completion_after_application_releases_service_chunk():
    execution,backend=execute(World())
    assert execution.snapshot()["status"]=="succeeded"
    ack=[data for path,data in backend.calls if path=="/ack"][0]
    assert ack["terminal"] and ack["completion"]["op_id"]=="task"


@pytest.mark.parametrize("completed",[False,True])
def test_world_terminal_gate_allows_overlap_without_accepting_received_as_success(completed):
    world=World(applied=False,completed=completed)
    world.completion_confirms_application=True
    world.wait_applied=lambda *args: (_ for _ in ()).throw(AssertionError("不得逐段串行等待插值"))
    execution,backend=execute(world)
    assert execution.snapshot()["status"]==("succeeded" if completed else "failed")
    assert any(path=="/ack" for path,_ in backend.calls)==completed


def test_bad_late_frame_rejected_before_any_frame_compiled():
    bad=chunk(); bad["pose"]["local_rot_mats"][-1][26][0][0]=-1
    with pytest.raises(ValueError,match="reflection"):
        compile_chunk(bad,TASK,expected_chunk=0,first_frame=1,wire_epoch=12)
    wrong=chunk();wrong["epoch"]=2
    with pytest.raises(ValueError,match="stale_chunk"):
        compile_chunk(wrong,TASK,expected_chunk=0,first_frame=1,wire_epoch=12)


def test_contact_release_between_wire_frames_is_not_lost():
    from yui_npc_controller.runtime.pose_codec import decode
    data=chunk()
    for key in data['pose']:
        data['pose'][key]*=2
    # 第一对中间抬脚，下一对重新落脚；旧降采样会把两包都发成持续接触。
    data['pose']['foot_contacts']=[[1,1,1,1],[0,1,0,1],[1,1,1,1],[1,1,1,1]]
    packets=compile_chunk(data,TASK,expected_chunk=0,first_frame=1,wire_epoch=12,packet_bytes=True)
    assert len(packets)==2 and all(len(p)==133 for p in packets)
    assert decode(packets[0])['contacts']==[0,1,0,1]
    assert decode(packets[1])['contacts']==[1,1,1,1]


@pytest.mark.parametrize('value',[float('nan'),-1,2,None,'1'])
def test_discarded_contact_frame_still_requires_validation(value):
    data=chunk();data['pose']['foot_contacts'][1][0]=value
    with pytest.raises(ValueError,match='invalid_contacts'):
        compile_chunk(data,TASK,expected_chunk=0,first_frame=1,wire_epoch=12)


def test_cancel_during_network_read_never_arms_world():
    entered=threading.Event(); release=threading.Event()
    class Slow(Backend):
        def _request(self,path,data=None):
            if path=="/chunks": entered.set();release.wait(1)
            return super()._request(path,data)
    world=World(); execution=MotionExecution(Slow(),world)
    execution.start(TASK);assert entered.wait(1)
    execution.stop(); release.set();execution.thread.join(2)
    assert world.stopped and not world.sent
    assert execution.snapshot()["status"]=="cancelled"


def test_invalid_first_chunk_does_not_arm_world():
    class Bad(Backend):
        def _request(self,path,data=None):
            reply=super()._request(path,data)
            if path=="/chunks": reply["chunk"]["pose"]["root_positions"][-1]=[float("nan"),0,0]
            return reply
    world=World()
    world.begin=lambda task: (_ for _ in ()).throw(AssertionError("不得先武装"))
    execution=MotionExecution(Bad(),world)
    execution.start(TASK);execution.thread.join(2)
    assert execution.snapshot()["status"]=="failed"
    assert execution.snapshot()["error"]!="AssertionError"
    assert not world.sent


def test_production_receipt_cannot_finish_with_isolated_service_ack():
    world=World()
    finish=world.finish
    world.finish=lambda *args: dict(finish(*args),production_receipt=True)
    execution,_=execute(world)
    assert execution.snapshot()["status"]=="failed"
    assert execution.snapshot()["error"]=="service_completion_unconfirmed"



@pytest.mark.parametrize("send_s,ack_s,spacing",[(.01,.01,.1),(.035,.085,.12)])
def test_frame_schedule_has_no_ack_padding_or_catchup(send_s,ack_s,spacing):
    class Clock:
        t=0.
        def __call__(self): return self.t
    clock=Clock()
    class Cancellation:
        def is_set(self): return False
        def wait(self,seconds): clock.t+=max(0,seconds)
        def set(self): pass
    class FramesBackend(Backend):
        def _request(self,path,data=None):
            reply=super()._request(path,data)
            if path=="/chunks":
                for key in reply["chunk"]["pose"]:
                    reply["chunk"]["pose"][key]*=4
            return reply
    world=World()
    starts=[]
    def send(seq,events):
        starts.append(clock.t)
        clock.t+=send_s
    def ack(*args):
        clock.t+=ack_s
        return True
    world.send=send
    world.wait_received=ack
    world.finish=lambda task,*args: dict(state="succeeded",op_id="task",pose_session=7,pose_epoch=12,pose_sequence=5)
    execution=MotionExecution(FramesBackend(),world,clock=clock)
    execution.cancelled=Cancellation()
    execution._run(TASK)
    assert execution.snapshot()["status"]=="succeeded"
    assert starts==pytest.approx([i*spacing for i in range(4)])
    assert len(execution.snapshot()["timing"]["frames"])==4
