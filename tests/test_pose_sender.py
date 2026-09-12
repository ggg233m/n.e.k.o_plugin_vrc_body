"""用真实候选传输入口验证 worker 所有权与并发停止。"""
import threading
import time
from types import SimpleNamespace
import _bootstrap
from yui_npc_controller.runtime.pose_sender import PoseSender
from yui_npc_controller.runtime.yui_transport import YuiReliableTransport
from yui_npc_controller.runtime.yui_protocol import MidiEvent


def test_existing_heartbeat_and_stop_use_one_worker():
    calls=[]
    def sink(event):
        calls.append((threading.get_ident(),event))
    actor=PoseSender(sink,7,3)
    session=SimpleNamespace(set_host_arm_authorized=lambda value: None, session=7, ack_generation=0,
        wait_for_ack=lambda *args, **kwargs: SimpleNamespace(ok=True,session=7))
    transport=YuiReliableTransport(sink,session,shared_sender=actor)
    try:
        transport.send_heartbeat()
        transport.send_estop()
        assert actor.closed.wait(1)
        assert len({thread for thread,event in calls})==1
        assert calls[-2][1].number==127
        assert calls[-1][1]==MidiEvent("cc",15,112,1)
    finally:
        transport.close()


def test_concurrent_stop_interrupts_pose_and_unblocks_sender():
    started=threading.Event()
    calls=[]
    errors=[]
    def sink(event):
        calls.append(event)
        started.set()
    actor=PoseSender(sink,7,3)
    def send():
        try:
            actor.send([MidiEvent("note_off",15,1,1)]*78,lane="pose")
        except RuntimeError as exc:
            errors.append(str(exc))
    worker=threading.Thread(target=send)
    worker.start()
    assert started.wait(1)
    actor.stop()
    worker.join(1)
    actor.close()
    assert not worker.is_alive() and errors
    assert len(calls)<80
    assert [e.number for e in calls[-2:]]==[127,112]


def test_sink_failure_stops_without_retry():
    calls=[]
    def sink(event):
        calls.append(event)
        if event.channel==14:
            raise OSError("已断开")
    actor=PoseSender(sink,7,3)
    try:
        try:
            actor.send([MidiEvent("cc",14,1,1)])
            assert False,"不能把发送失败报告为成功"
        except RuntimeError:
            pass
    finally:
        actor.close()
    assert sum(e.channel==14 for e in calls)==1


def test_existing_command_timeout_does_not_retransmit():
    calls=[]
    sink=calls.append
    actor=PoseSender(sink,7,3)
    session=SimpleNamespace(session=7,ack_generation=0,wait_for_ack=lambda *args,**kwargs: None)
    transport=YuiReliableTransport(sink,session,shared_sender=actor)
    try:
        try:
            transport.send_command("STOP")
            assert False,"缺少 ACK 必须报告失败"
        except TimeoutError:
            pass
    finally:
        transport.close()
    assert sum(e.type=="note_on" and e.number==7 for e in calls)==1


def test_pose_ack_releases_worker_credit():
    calls=[]
    actor=PoseSender(calls.append,7,3)
    try:
        ticket=actor.send([MidiEvent("note_off",15,1,1)]*2,lane="pose")
        actor.ack(ticket,7)
        # ACK 与下一个提交按同一 mailbox 顺序处理，无跨线程直接改核心状态。
        second=actor.send([MidiEvent("note_off",15,2,1)]*2,lane="pose")
        assert second[1]>ticket[1]
    finally:
        actor.close()


def test_confirmed_graceful_close_does_not_latch_estop():
    calls=[]
    actor=PoseSender(calls.append,7,3)
    ticket=actor.send([MidiEvent("note_off",15,1,1)]*2,lane="pose")
    actor.ack(ticket,7)
    actor.close(graceful=True)
    assert actor.gracefully_closed
    assert len(calls)==2


def test_unconfirmed_graceful_close_still_stops():
    calls=[]
    actor=PoseSender(calls.append,7,3)
    actor.send([MidiEvent("note_off",15,1,1)],lane="pose")
    actor.close(graceful=True)
    assert not actor.gracefully_closed
    assert [event.number for event in calls[-2:]]==[127,112]

def test_world_release_can_settle_unconfirmed_pose_without_estop():
    from yui_npc_controller.runtime.pose_receipts import PoseReceipts
    calls=[]
    actor=PoseSender(calls.append,7,3)
    receipts=PoseReceipts(actor,world_id='test',session=7,epoch=3)
    receipts.send(1,[MidiEvent('note_off',15,1,1)]*2)
    receipts.quiesce_after_world_release()
    assert not receipts.wait_received(1,.01) and not actor.stop_requested.is_set()
    # 已确认世界释放后，再由真实控制ACK提供端口读取水位。
    ticket=actor.send([MidiEvent('note_on',14,7,0)],lane='control')
    actor.ack(ticket,7)
    actor.close(graceful=True)
    assert actor.gracefully_closed and all(e.number!=127 for e in calls)
