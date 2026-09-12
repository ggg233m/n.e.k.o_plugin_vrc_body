"""版本化持续会话接口；世界回执负责授权、应用边界及任务终态。"""
from copy import deepcopy
import threading
import time
from .timeline import MotionTimeline
from .motion_modes import compile_plan, validate_plan

class ContinuousService:
    def __init__(self,generator,*,clock=time.monotonic,lease_s=2.):
        self.generator,self.clock,self.lease_s=generator,clock,lease_s
        self.lock=threading.RLock()
        self.timeline=None
        self.binding=None
        self.seen=float('-inf')
        self.reason=None
        self.armed=False

    def _current(self,data):
        if self.timeline and not self.timeline.closed and self.clock()-self.seen>=self.lease_s:
            self.timeline.close('world_lease_expired');self.reason='world_lease_expired'
        if not self.timeline or self.timeline.closed or any(data.get(k)!=self.binding[k] for k in ('stream_id','session','pose_epoch','world_id')):
            raise ValueError('stream_binding_lost')
        return self.timeline

    def open(self,data):
        receipt=data.get('world_receipt',{})
        identity={k:data.get(k) for k in ('stream_id','session','pose_epoch','world_id')}
        if (not self.generator.ready or type(identity['session']) is not int or identity['session']<=0
            or type(identity['pose_epoch']) is not int or identity['pose_epoch']<=0
            or not isinstance(identity['world_id'],str) or not identity['world_id']
            or not isinstance(identity['stream_id'],str) or len(identity['stream_id'])!=32
            or any(c not in '0123456789abcdef' for c in identity['stream_id'])):
            raise ValueError('invalid_stream_binding')
        if receipt.get('type')!='npc.stream_prepared' or receipt.get('protocol')!='neko-pose/2' or any(receipt.get(k)!=v for k,v in identity.items()):
            raise ValueError('stream_arm_unconfirmed')
        plan=validate_plan(data.get('plan',{}))
        with self.lock:
            if self.timeline:
                if identity==self.binding:
                    return self.status(identity)
                if not self.timeline.closed:
                    raise ValueError('stream_already_bound')
            if data.get('seed') is not None:
                self.generator.reseed(data['seed'])
            self.binding=identity
            self.timeline=MotionTimeline(self.generator,compile_plan,stream_id=identity['stream_id'])
            self.timeline.update('A person stands calmly with relaxed arms.',plan,source='base',duration_s=None)
            self.seen=self.clock();self.reason=None;self.armed=False
            return self.status(identity)

    def arm(self,data):
        with self.lock:
            self._current(data)
            receipt=data.get('world_receipt',{})
            if receipt.get('type')!='npc.stream_armed' or any(receipt.get(k)!=v for k,v in self.binding.items()):
                raise ValueError('stream_arm_unconfirmed')
            self.armed=True
            return self.status(data)

    def renew(self,data):
        with self.lock:
            self._current(data);self.seen=self.clock()
            return self.status(data)

    def update(self,data):
        with self.lock:
            timeline=self._current(data)
            plan=validate_plan(data['plan'])
            return timeline.update(data['prompt'],plan,source=data.get('source','explicit'),
                                   duration_s=data.get('duration_s',4),operation_id=data.get('op_id'))

    def pull(self,data):
        with self.lock:
            timeline=self._current(data)
            block=timeline.pull()
            return {'block':block,'control':timeline.pending_control() if block is None else None,
                    'stream':timeline.snapshot()}

    def received(self,data):
        with self.lock:
            timeline=self._current(data)
            event=data.get('world_receipt',{})
            block=timeline.outstanding
            if (not self.armed or not block or event.get('type')!='npc.pose_ack'
                or event.get('state')!='received' or event.get('op_id')!=self.binding['stream_id']
                or event.get('pose_sequence')!=block['end_frame']//2
                or event.get('pose_session')!=self.binding['session']
                or event.get('stream_epoch')!=self.binding['pose_epoch']
                or type(data.get('wire_epoch')) is not int or data['wire_epoch']<=0
                or event.get('pose_epoch')!=data['wire_epoch']
                or any(event.get(k)!=self.binding[k] for k in ('session','world_id'))):
                raise ValueError('stream_receive_unconfirmed')
            return timeline.received(dict(stream_id=self.binding['stream_id'],version=block['version'],
                op_id=block['op_id'],received_frame=block['end_frame']))

    def exchange(self,data):
        """一次往返交换接收证据、实际播放进度与下一块，仍分别校验其含义。"""
        applications=data.get('applications',[])
        received=data.get('received')
        if (not isinstance(applications,list) or len(applications)>8
            or any(not isinstance(item,dict) for item in applications)
            or (received is not None and not isinstance(received,dict))
            or type(data.get('pull',True)) is not bool):
            raise ValueError('invalid_stream_exchange')
        with self.lock:
            self._current(data)
            identity=dict(self.binding)
            if received is not None:
                self.received(dict(received,**identity))
            for item in applications:
                self.ack(dict(item,**identity))
            return self.pull(identity) if data.get('pull',True) else {'block':None}

    def ack(self,data):
        with self.lock:
            timeline=self._current(data)
            if not self.armed:raise ValueError('stream_not_armed')
            event=data.get('world_receipt',{})
            if event.get('type')=='npc.pose_ack':
                index=data.get('progress_index');items=event.get('stream_progress')
                if (event.get('state')!='received' or event.get('op_id')!=self.binding['stream_id']
                    or event.get('pose_session')!=self.binding['session'] or event.get('stream_epoch')!=self.binding['pose_epoch']
                    or any(event.get(k)!=self.binding[k] for k in ('session','world_id'))
                    or type(event.get('pose_sequence')) is not int or event['pose_sequence']<=0
                    or not isinstance(items,list) or len(items)>2 or type(index) is not int or not 0<=index<len(items)):
                    raise ValueError('stream_piggyback_unconfirmed')
                item=items[index]
                if (not isinstance(item,dict) or set(item)!={'op_id','version','executed_frame','committed_frame'}
                    or type(item['executed_frame']) is not int or item['executed_frame']>event['pose_sequence']*2):
                    raise ValueError('stream_piggyback_invalid_progress')
                event=dict(self.binding,**item,type='npc.stream_progress')
            block=timeline.delivered.get(event.get('executed_frame'))
            if not block or event.get('type')!='npc.stream_progress' or any(event.get(k)!=v for k,v in self.binding.items()):
                raise ValueError('stream_application_unconfirmed')
            if type(event.get('executed_frame')) is not int or event['executed_frame']!=block['end_frame'] or event.get('op_id')!=block['op_id'] or event.get('version')!=block['version']:
                raise ValueError('stream_application_mismatch')
            terminal=data.get('operation_receipt',{})
            completed=(terminal.get('type')=='npc.operation_completed' and terminal.get('op_id')==block['op_id']
                and terminal.get('result')=='motion_completed' and terminal.get('kind')=='motion'
                and terminal.get('session')==self.binding['session'] and terminal.get('world_id')==self.binding['world_id'])
            return timeline.acknowledge(dict(stream_id=self.binding['stream_id'],version=block['version'],
                op_id=block['op_id'],executed_frame=event['executed_frame'],committed_frame=event.get('committed_frame',event['executed_frame']),
                world_applied=True,operation_completed=completed))

    def close(self,data=None):
        with self.lock:
            if self.timeline:
                if data is not None and any(data.get(k)!=self.binding[k] for k in self.binding):
                    raise ValueError('stream_binding_mismatch')
                self.timeline.close('service_closed')
            self.reason='service_closed'
            return {'status':'closed'}

    def status(self,data):
        with self.lock:
            if not self.timeline or any(data.get(k)!=self.binding[k] for k in self.binding):
                raise ValueError('stream_binding_mismatch')
            result=self.timeline.snapshot()
            if data.get('op_id'):
                result['operation']=deepcopy(self.timeline.records.get(data['op_id'],{'status':'unknown'}))
            return dict(result,protocol='yui-motion-stream/2',reason=self.reason,armed=self.armed)

    def step(self):
        with self.lock:
            if not self.timeline or self.timeline.closed:return False
            try:self._current(self.binding)
            except ValueError:return False
            timeline=self.timeline
        try:return timeline.generate()
        except Exception as exc:
            with self.lock:self.reason=type(exc).__name__
            return False
