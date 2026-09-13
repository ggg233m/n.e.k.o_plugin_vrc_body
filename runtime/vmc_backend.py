"""独立 VMC 动作后端；仅复用世界骨骼通道，不连接 ARDY 或加载模型。"""
import threading
import time
import uuid
from .motion_backend import MotionBackend, MotionBackendConfig
from .vmc_receiver import VmcReceiver
from .vmc_sharing import pose_from_sample

class VmcStream:
    def __init__(self, receiver):
        self.receiver=receiver; self.lock=threading.RLock(); self.binding=None
        self.pending=None; self.delivered={}; self.next_frame=0; self.armed=False
        self.op_id=None; self.next_at=0.; self.applied_frame=0

    def _identity(self,data):
        if self.binding is None or any(data.get(k)!=v for k,v in self.binding.items()):
            raise ValueError('vmc_stream_identity_mismatch')

    def request(self,path,data):
        with self.lock:
            if path=='open':
                if self.binding is not None: raise ValueError('vmc_stream_already_open')
                binding={k:data[k] for k in ('stream_id','session','world_id','pose_epoch')}
                receipt=data.get('world_receipt',{})
                if receipt.get('type')!='npc.stream_prepared' or any(receipt.get(k)!=v for k,v in binding.items()):
                    raise ValueError('vmc_world_prepare_unconfirmed')
                if self.receiver.latest() is None: raise ValueError('vmc_source_unavailable')
                self.binding=binding; self.pending=None; self.delivered={}; self.next_frame=0
                self.op_id=uuid.uuid4().hex; self.armed=False; self.applied_frame=0; self.next_at=0.
                return {'status':'opened'}
            self._identity(data)
            if path=='close':
                self.binding=None; self.pending=None; self.delivered={}; self.armed=False
                return {'status':'closed'}
            if path=='arm':
                receipt=data.get('world_receipt',{})
                if receipt.get('type')!='npc.stream_armed' or any(receipt.get(k)!=v for k,v in self.binding.items()):
                    raise ValueError('vmc_world_arm_unconfirmed')
                self.armed=True
                return {'status':'armed'}
            if path=='world':
                if self.receiver.latest() is None: raise ValueError('vmc_source_stale')
                return {'status':'ready'}
            if path=='pull': return self.pull()
            if path=='received': return self.received(data)
            if path=='ack': return self.ack(data)
            if path=='exchange':
                received=data.get('received'); applications=data.get('applications',[])
                if not isinstance(applications,list) or len(applications)>8: raise ValueError('invalid_vmc_exchange')
                if received is not None: self.received(received)
                for item in applications: self.ack(item)
                return self.pull() if data.get('pull',True) else {'block':None}
            raise ValueError('VMC 仅共享宿主动作，不接受 ARDY 动作生成请求')

    def pull(self):
        if self.pending is not None: return {'block':self.pending}
        sample=self.receiver.latest()
        if sample is None: raise ValueError('vmc_source_stale')
        if time.monotonic()<self.next_at: return {'block':None}
        self.pending=dict(self.binding,op_id=self.op_id,version=1,mode='idle',final=False,
                          first_frame=self.next_frame,end_frame=self.next_frame+4,pose=pose_from_sample(sample))
        # 按播放节奏采样，只允许世界通道原有的有界预收缓冲。
        self.next_at=time.monotonic()+.18
        return {'block':self.pending}

    def received(self,data):
        event=data.get('world_receipt',{}); block=self.pending
        if (not self.armed or block is None or event.get('type')!='npc.pose_ack' or event.get('state')!='received'
            or event.get('op_id')!=self.binding['stream_id'] or event.get('pose_sequence')!=block['end_frame']//2
            or event.get('pose_session')!=self.binding['session'] or event.get('stream_epoch')!=self.binding['pose_epoch']
            or type(data.get('wire_epoch')) is not int or data['wire_epoch']<=0 or event.get('pose_epoch')!=data['wire_epoch']
            or any(event.get(k)!=self.binding[k] for k in ('session','world_id'))):
            raise ValueError('vmc_receive_unconfirmed')
        self.delivered[block['end_frame']]=block; self.next_frame=block['end_frame']; self.pending=None
        return {'status':'received'}

    def ack(self,data):
        event=data.get('world_receipt',{})
        if not self.armed: raise ValueError('vmc_not_armed')
        if event.get('type')=='npc.pose_ack':
            index=data.get('progress_index'); items=event.get('stream_progress')
            if (event.get('state')!='received' or event.get('op_id')!=self.binding['stream_id']
                or event.get('pose_session')!=self.binding['session'] or event.get('stream_epoch')!=self.binding['pose_epoch']
                or any(event.get(k)!=self.binding[k] for k in ('session','world_id'))
                or not isinstance(items,list) or len(items)>2 or type(index) is not int or not 0<=index<len(items)
                or type(event.get('pose_sequence')) is not int): raise ValueError('vmc_progress_unconfirmed')
            item=items[index]
            if (not isinstance(item,dict) or set(item)!={'op_id','version','executed_frame','committed_frame'}
                or type(item['executed_frame']) is not int or item['executed_frame']>event['pose_sequence']*2):
                raise ValueError('vmc_invalid_progress')
            event=dict(self.binding,**item,type='npc.stream_progress')
        block=self.delivered.get(event.get('executed_frame'))
        if (event.get('type')!='npc.stream_progress' or block is None
            or any(event.get(k)!=v for k,v in self.binding.items())
            or event.get('op_id')!=block['op_id'] or event.get('version')!=block['version']):
            raise ValueError('vmc_application_unconfirmed')
        self.applied_frame=max(self.applied_frame,block['end_frame'])
        self.delivered={k:v for k,v in self.delivered.items() if k>=self.applied_frame-20}
        return {'status':'applied'}

class VmcBackend(MotionBackend):
    def __init__(self,config,*,changed=None,receiver=None):
        super().__init__(MotionBackendConfig(enabled=config.enabled,poll_s=.2),changed=changed)
        self.receiver=receiver or VmcReceiver(config)
        self.stream=VmcStream(self.receiver); self.instance='vmc-'+uuid.uuid4().hex

    def start(self):
        self.receiver.start()
        super().start()

    def close(self):
        # 先释放世界姿态通道，再停止接收并恢复宿主设置。
        super().close()
        self.receiver.close()

    def snapshot(self):
        state=self.receiver.snapshot()
        return dict(state,backend='vmc',ready=state['ready'] and not self._stop.is_set(),generation=self._generation)

    def _request(self,path,data=None):
        if path=='/health':
            return dict(protocol='yui-motion/1',ready=self.receiver.latest() is not None,
                        continuous_protocol='yui-motion-stream/2',instance=self.instance)
        if path=='/cancel': return {'status':'cancelled','epoch':1}
        if path.startswith('/streams/'): return self.stream.request(path.removeprefix('/streams/'),data)
        raise ValueError('VMC 不连接 ARDY 服务')

    def bind_execution(self,world_factory):
        # 不启用旧版单次 ARDY 通道，也不保留其租约线程。
        world_factory.close()

    def world_ready(self,session):
        # 此方法决定 ARDY 生成工具是否曝光；VMC 不能冒充生成后端。
        return False

    def update_base_intent(self,*args,**kwargs): return None

    def perform(self,*args,**kwargs):
        return dict(status='failed',error='vmc_sharing_active',midi_sent=False)
