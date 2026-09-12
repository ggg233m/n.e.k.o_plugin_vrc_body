"""姿态实验的单端口调度核心；只允许一个调度线程调用，尚未接管正式发送器。"""
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    lane: str
    sequence: int
    events: tuple
    deadline: float
    offset: int = 0


class SharedPort:
    PRIORITY = ("control", "heartbeat", "pose", "text")

    def __init__(self, sink, session, epoch, stop_events, *, timeout=.5, pose_pipeline=False, fault_events=None):
        if len(stop_events) != 2:
            raise ValueError("必须提供原控制与姿态的两个停止事件")
        self.sink, self.session, self.epoch = sink, session, epoch
        self.pose_pipeline = pose_pipeline
        self.stop_events, self.timeout = tuple(stop_events), timeout
        if fault_events is not None and len(fault_events)!=1:raise ValueError('故障暂停必须仅占一个预留事件')
        self.fault_events=None if fault_events is None else tuple(fault_events)
        self.estop_sent=False
        self.queues = {lane: deque() for lane in self.PRIORITY}
        self.last_sequence = {lane: 0 for lane in self.PRIORITY}
        self.active = None
        self.serial = self.retired = 0
        self.unconfirmed = deque()
        self.pending = {}
        self.next_send = 0.
        self.window = deque()
        self.stopped = False
        self.reason = None
        self.stop_errors = []
        self.last_completed = None
        self.audit = deque(maxlen=32)
        self.audit_pending = {}

    @property
    def outstanding(self):
        return self.serial - self.retired

    def submit(self, lane, sequence, events, now):
        if self.stopped:
            return False
        if lane not in self.queues or type(sequence) is not int or sequence <= self.last_sequence[lane]:
            raise ValueError("通道或顺序号无效")
        events = tuple(events)
        if not events or len(events) > (78 if lane == "pose" else 10):
            raise ValueError("事务必须有界；文字按小块提交")
        if sum(map(len,self.queues.values())) >= 16:
            return False
        if lane == "pose" and (self.queues[lane] or (self.active and self.active.lane == lane)
                               or (not self.pose_pipeline and any(key[0] == lane for key in self.pending))):
            return False
        self.last_sequence[lane] = sequence
        self.queues[lane].append(Transaction(lane,sequence,events,now+self.timeout))
        entry=dict(lane=lane,sequence=sequence,queued=now,event_count=len(events),first_sent=None,last_sent=None,acked=None)
        self.audit.append(entry);self.audit_pending[(lane,sequence)]=entry
        return True

    def _expired(self, now):
        if any(now >= item[2] for item in self.unconfirmed):
            self.stop("receipt_timeout")
        return self.stopped

    def step(self, now):
        if self._expired(now) or now < self.next_send:
            return False
        while self.window and now-self.window[0] >= 1:
            self.window.popleft()
        if len(self.window) >= 998:
            return False
        if self.active is None:
            for lane in self.PRIORITY:
                queue = self.queues[lane]
                while queue and queue[0].deadline <= now:
                    expired=queue.popleft()
                    entry=self.audit_pending.pop((expired.lane,expired.sequence),None)
                    if entry is not None:entry['expired']=now
                if queue and (self.outstanding < 112 if self.pose_pipeline and lane == "pose" else self.outstanding+len(queue[0].events) <= 112):
                    self.active = queue.popleft()
                    break
        tx = self.active
        if tx is None or self.outstanding >= 112:
            return False
        if tx.deadline <= now:
            self.stop("transaction_expired")
            return False
        entry=self.audit_pending.get((tx.lane,tx.sequence))
        if entry is not None and tx.offset==0:entry["first_sent"]=now
        # 先占用信用；底层报错也按可能已经发出处理，不重试。
        self.serial += 1
        # 控制回执容忍日志回传抖动；姿态仍保留原来的短租约。
        receipt_timeout = max(self.timeout, 1.5) if tx.lane in ('heartbeat', 'control') else self.timeout
        self.unconfirmed.append((self.serial,now,now+receipt_timeout))
        self.window.append(now)
        # 单帧允许较短发送突发；998条/滚动秒与112条未确认上限保持不变。
        self.next_send = now+(1/4000 if self.pose_pipeline else 1/2000)
        try:
            self.sink(tx.events[tx.offset])
        except Exception:
            self.stop("send_uncertain")
            return False
        tx.offset += 1
        if tx.offset == len(tx.events):
            self.last_completed = (tx.lane,tx.sequence)
            if entry is not None:entry["last_sent"]=now
            # 只有完整命令或姿态 ACK 可形成端口读取进度证据。
            if tx.lane in ("control", "heartbeat", "pose"):
                self.pending[(tx.lane,tx.sequence)] = self.serial
            if tx.lane=="text":self.audit_pending.pop((tx.lane,tx.sequence),None)
            self.active = None
        return True

    def ack(self, lane, session, epoch, sequence, now):
        if self._expired(now) or (session,epoch) != (self.session,self.epoch):
            return False
        watermark = self.pending.get((lane,sequence))
        if watermark is None:
            return False
        self.retired = max(self.retired,watermark)
        for key,end in self.pending.items():
            if end<=self.retired:
                entry=self.audit_pending.pop(key,None)
                if entry is not None:entry["acked"]=now
        while self.unconfirmed and self.unconfirmed[0][0] <= self.retired:
            self.unconfirmed.popleft()
        self.pending = {key:end for key,end in self.pending.items() if end > self.retired}
        return True

    def stop(self, reason="explicit_stop"):
        if self.stopped:
            # 故障暂停之后仍可升级人工急停，两个预留事件总量不变。
            if reason=='explicit_stop' and not self.estop_sent:
                self.estop_sent=True;self.reason=reason
                try:self.sink(self.stop_events[0])
                except Exception as exc:self.stop_errors.append(type(exc).__name__)
            return
        self.stopped, self.reason = True, reason
        logger.warning('YUI_PORT_STOP reason=%s session=%s epoch=%s outstanding=%s active_lane=%s',
                       reason, self.session, self.epoch, self.outstanding,
                       self.active.lane if self.active else None)
        self.active = None
        for queue in self.queues.values():
            queue.clear()
        self.pending.clear()
        self.audit_pending.clear()
        # 最多 112 条普通未确认事件，再加两个停止，仍为有界发送。
        explicit=reason=='explicit_stop'
        self.estop_sent=explicit
        # 旧世界只停止姿态桥，不把普通故障升级成全局急停。
        fault_events = self.fault_events if self.fault_events is not None else self.stop_events[1:]
        for event in (self.stop_events if explicit else fault_events):
            try:
                self.sink(event)
            except Exception as exc:
                self.stop_errors.append(type(exc).__name__)
