"""姿态 wire 身份与内部端口 ticket 的有界绑定，不信任任意日志 ACK。"""
import threading


class PoseReceipts:
    def __init__(self, sender, *, world_id, session, epoch, max_in_flight=1):
        if not world_id or type(session) is not int or session <= 0 or type(epoch) is not int or epoch <= 0:
            raise ValueError("姿态执行身份无效")
        if type(max_in_flight) is not int or max_in_flight not in (1,2):
            raise ValueError("姿态在途窗口必须为1或2")
        self.sender, self.world_id = sender, world_id
        self.session, self.epoch = session, epoch
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self._pending = {}
        self.max_in_flight = max_in_flight
        self.last_sequence = 0
        self.last_log_sequence = 0
        self.stopped = False
        self.last_received = 0

    @property
    def pending(self):
        return next(iter(self._pending.values()),None)

    def send(self, sequence, events):
        with self.lock:
            if self.stopped or len(self._pending) >= self.max_in_flight:
                raise RuntimeError("姿态回执尚未完成或已停止")
            if type(sequence) is not int or sequence <= self.last_sequence:
                raise ValueError("姿态序号必须递增")
            self.last_sequence = sequence
            pending={"sequence":sequence,"ticket":None,"early_ack":False}
            self._pending[sequence]=pending
        try:
            ticket=self.sender.send(events,lane="pose")
        except Exception:
            self.stop(fault=True)
            raise
        with self.lock:
            if self.stopped or self._pending.get(sequence) is not pending:
                raise RuntimeError("发送期间姿态会话已停止")
            pending["ticket"]=ticket
            if pending["early_ack"]:
                self.sender.ack(ticket,self.session)
                self._pending.pop(sequence,None)
                self.last_received=max(self.last_received,sequence)
                self.condition.notify_all()
        return ticket

    def ingest(self, event):
        with self.lock:
            if self.stopped or not self._pending:
                return False
            if event.get("type")!="npc.pose_ack" or event.get("state")!="received" or event.get("npc_id")!="yui":
                return False
            for key in ("session","pose_session","pose_epoch","pose_sequence","log_seq"):
                if type(event.get(key)) is not int or event[key]<=0:
                    return False
            pending=self._pending.get(event["pose_sequence"])
            if pending is None:return False
            if (event.get("world_id"),event["session"],event["pose_session"],event["pose_epoch"],event["pose_sequence"]) != (
                self.world_id,self.session,self.session,self.epoch,pending["sequence"]):
                return False
            if event["log_seq"]<=self.last_log_sequence:
                return False
            self.last_log_sequence=event["log_seq"]
            if pending["ticket"] is None:
                # 允许日志先于发送 Future 唤醒返回，但最多保留一个已验证提前 ACK。
                pending["early_ack"]=True
            else:
                self.sender.ack(pending["ticket"],self.session)
                self._pending.pop(event["pose_sequence"],None)
                self.last_received=max(self.last_received,event["pose_sequence"])
                self.condition.notify_all()
            return True

    def wait_received(self, sequence, timeout=.5):
        with self.condition:
            self.condition.wait_for(lambda:self.stopped or self.last_received>=sequence, timeout)
            return not self.stopped and self.last_received>=sequence and not any(s<=sequence for s in self._pending)

    def quiesce_after_world_release(self):
        """世界已明确撤销身体授权；停止新姿态，端口信用仍由后续真实ACK结清。"""
        with self.condition:
            self.stopped=True
            self._pending.clear()
            self.condition.notify_all()

    def stop(self, *, fault=False):
        with self.lock:
            self.stopped=True
            self._pending.clear()
            self.condition.notify_all()
        if fault:getattr(self.sender,'fault_stop',self.sender.stop)()
        else:self.sender.stop()
