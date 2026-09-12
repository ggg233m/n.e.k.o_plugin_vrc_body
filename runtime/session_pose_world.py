"""正式 NEKO 日志适配器；需要真实武装、末帧应用和 operation_completed。"""
import threading

from .pose_codec import encode_finish, midi_events
from .pose_receipts import PoseReceipts
from .yui_protocol import MidiEvent


class SessionPoseWorld:
    # 世界完成事件由最后一帧插值完成触发；接收回执只用于释放端口信用。
    completion_confirms_application = True

    def __init__(self, session, task, sender, wire_epoch, arm):
        if type(wire_epoch) is not int or wire_epoch<=0:
            raise ValueError("invalid_wire_epoch")
        self.session=session
        self.task=dict(task)
        self.world_id=session.world_id
        self.session_id=session.session
        self.wire_epoch=wire_epoch
        self.sender=sender
        self.arm=arm
        self.condition=threading.Condition(threading.RLock())
        self.stopped=False
        self.closed=False
        self.failure=None
        self.armed=False
        self.applied=0
        self.finish_sequence=None
        self.pose_completed=None
        self.operation_completed=None
        self.last_log_sequence=session.last_log_sequence or 0
        self.receipts=PoseReceipts(sender,world_id=self.world_id,session=self.session_id,epoch=wire_epoch)
        if task.get("session")!=self.session_id:
            raise ValueError("task_session_mismatch")
        session.add_event_listener(self.ingest)

    def is_current(self):
        return (not self.closed and not self.stopped and self.failure is None
                and self.session.world_id==self.world_id and self.session.session==self.session_id
                and not self.session.estop and self.session.control_state in {"external","action","moving"})

    def ingest(self,event):
        # 正式遥测头使用 npc；适配到姿态回执的 npc_id，拒绝冲突别名。
        if "npc" in event:
            if "npc_id" in event and event["npc_id"] != event["npc"]:
                return
            event = dict(event, npc_id=event["npc"])
        with self.condition:
            if event.get("type")=="sys.boot" or not self.is_current():
                self.failure="world_binding_lost"
                self.receipts.stop()
                self.condition.notify_all()
                return
            if (event.get("world_id"),event.get("session"),event.get("npc_id"))!=(self.world_id,self.session_id,"yui"):
                return
            log_seq=event.get("log_seq")
            if type(log_seq) is not int or log_seq<=self.last_log_sequence:
                return
            self.last_log_sequence=log_seq
            kind=event.get("type")
            if kind=="npc.pose_ack":
                self.receipts.ingest(event)
                applied = event.get("applied_sequence")
                if (self.armed and event.get("op_id") == self.task["op_id"]
                        and type(applied) is int and applied == event.get("pose_sequence")
                        and all(type(event.get(k)) is int for k in ("pose_session", "pose_epoch"))
                        and (event["pose_session"], event["pose_epoch"]) == (self.session_id, self.wire_epoch)
                        and event.get("state") == "received" and 0 < applied <= self.receipts.last_sequence):
                    self.applied = max(self.applied, applied)
            elif event.get("op_id")==self.task["op_id"]:
                pose_identity=(all(type(event.get(k)) is int for k in ("pose_session","pose_epoch"))
                    and (event["pose_session"],event["pose_epoch"])==(self.session_id,self.wire_epoch))
                seq=event.get("pose_sequence")
                if kind=="npc.pose_armed" and pose_identity and event.get("state")=="armed" and type(seq) is int and seq==0:
                    self.armed=True
                elif kind=="npc.pose_applied" and self.armed and pose_identity and event.get("state")=="applied":
                    if type(seq) is int and 0<seq<=self.receipts.last_sequence:
                        self.applied=max(self.applied,seq)
                elif (kind=="npc.pose_completed" and pose_identity and event.get("state")=="succeeded"
                      and self.finish_sequence is not None and seq==self.finish_sequence):
                    self.pose_completed=dict(event)
                elif kind=="npc.operation_completed" and event.get("kind")=="motion" and self.finish_sequence is not None:
                    if event.get("result")=="motion_completed":
                        self.operation_completed=dict(event)
                elif kind in {"npc.operation_cancelled","npc.operation_failed"}:
                    self.failure=event.get("reason") or event.get("err") or "world_operation_failed"
                    self.receipts.stop()
            self.condition.notify_all()

    def begin(self,task):
        if task!=self.task or not self.is_current():
            raise RuntimeError("world_binding_lost")
        # arm 必须走同一个端口调度器并返回发送 ticket；世界回执才释放信用。
        ticket=self.arm(task,self.wire_epoch)
        with self.condition:
            self.condition.wait_for(lambda:self.armed or not self.is_current(),.5)
            if not self.is_current() or not self.armed:
                self.receipts.stop()
                raise RuntimeError("world_arm_unconfirmed")
            self.sender.ack(ticket,self.session_id)
            return self.wire_epoch

    def wait_applied(self,sequence,timeout):
        with self.condition:
            self.condition.wait_for(lambda:self.applied>=sequence or not self.is_current(),timeout)
            return self.is_current() and self.applied==sequence

    def finish(self,task,sequence,timeout):
        if task!=self.task or not self.is_current() or not self.armed or self.receipts.last_received!=sequence:
            return None
        self.finish_sequence=sequence+1
        packet=encode_finish(self.session_id,self.wire_epoch,self.finish_sequence)
        self.receipts.send(self.finish_sequence,tuple(MidiEvent(*e) for e in midi_events(packet)))
        if not self.receipts.wait_received(self.finish_sequence,timeout):
            return None
        with self.condition:
            self.condition.wait_for(lambda:self.pose_completed or not self.is_current(),timeout)
            if not self.is_current() or not self.pose_completed:
                return None
            # 姿态已实际完成，普通操作终态可能排在 20 行/秒的遥测队列中。
            # 只延长这一条日志的等待；不续发姿态、不放宽端口或世界接收租约。
            self.condition.wait_for(lambda:self.operation_completed or not self.is_current(),max(timeout,1.5))
            if not self.is_current() or not self.pose_completed or not self.operation_completed:
                return None
            return dict(self.pose_completed,operation_receipt=self.operation_completed,production_receipt=True)

    def stop(self):
        self.stopped=True
        self.receipts.stop()
        with self.condition:
            self.condition.notify_all()

    def close(self):
        # 完整成功后只撤销监听；共享端口继续服务后续任务、文字及心跳。
        if not self.pose_completed or not self.operation_completed:
            self.stop()
        self.closed=True
        self.session.remove_event_listener(self.ingest)
