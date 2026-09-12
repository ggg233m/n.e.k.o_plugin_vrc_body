"""持续运动执行适配器；世界应用进度推进时间线，接收ACK只归还MIDI信用。"""
from collections import deque
import threading
import math
import time
import uuid

from .motion_space import MotionSpace
from .pose_codec import encode_stream_control, midi_events, rebind_pose_epoch
from .pose_frames import compile_chunk
from .pose_receipts import PoseReceipts
from .yui_protocol import MidiEvent


class ContinuousExecution:
    protocol = 'yui-motion-stream/2'
    modes = {'idle': 0, 'free_action': 1, 'path': 2, 'velocity': 3, 'keyframes': 4}

    def __init__(self, backend, transport, session, *, expired=lambda: False, seed=None):
        self.backend, self.transport, self.session = backend, transport, session
        self.expired = expired
        self.seed = seed
        self.wrap_root='pose_wrapped_root_v1' in getattr(session,'capabilities',[])
        self.condition = threading.Condition(threading.RLock())
        self.wire_lock = threading.RLock()
        self.halt = threading.Event()
        self.close_requested = threading.Event()
        self.closed = threading.Event()
        self.thread = None
        self.sender = self.receipts = self.space = None
        self.binding = None
        self.events = deque(maxlen=128)
        self.last_log = session.last_log_sequence or 0
        self.request_seq = 0
        self.version = 0
        self.wire_epoch = None
        self.status = 'unavailable'
        self.error = None
        self.stage='created'
        self.last_request=None
        self.max_request_seconds=0.
        self.request_audit=deque(maxlen=32)
        self.origin_xz = [0., 0.]
        self.last_block = None
        self.sent_frame = self.applied_frame = 0
        self.blocks = {}
        self.application_events = {}
        self.received_ack = None
        self.world_release = None
        self.path_goal = None
        self.path_terminal = None
        self.session.add_event_listener(self.ingest)

    def eligible(self):
        return (not self.halt.is_set() and not self.expired() and not self.session.estop
                and self.session.session > 0 and bool(self.session.world_id)
                and self.session.control_state in {'external', 'action'}
                and 'pose_stream_v2' in self.session.capabilities
                and (self.binding is None or
                     (self.binding['session'], self.binding['world_id']) ==
                     (self.session.session, self.session.world_id)))

    def ready(self, session):
        return session is self.session and self.eligible() and self.status == 'running'

    def snapshot(self):
        with self.condition:
            return dict(protocol=self.protocol,status=self.status,error=self.error,
                        stream_id=self.binding['stream_id'] if self.binding else None,
                        executed_frame=self.last_block['end_frame'] if self.last_block else 0,
                        sent_frame=self.sent_frame,applied_frame=self.applied_frame,stage=self.stage,
                        last_request=self.last_request,max_request_seconds=self.max_request_seconds,
                        request_audit=[dict(x) for x in list(self.request_audit)[-12:]],
                        port_reason=getattr(getattr(self.sender,'core',None),'reason',None),
                        world_release_reason=(self.world_release or {}).get('reason'),
                        port_audit=[dict(x) for x in list(getattr(getattr(self.sender,'core',None),'audit',[]))[-8:]])

    def ingest(self, event):
        if 'npc' in event:
            if 'npc_id' in event and event['npc_id'] != event['npc']:
                return
            event = dict(event, npc_id=event['npc'])
        with self.condition:
            # 故障后仍接收同一身份的终态，防止丢失完成回执后错误续走。
            identity = (event.get('session'), event.get('world_id'), event.get('npc_id'))
            if (self.path_goal and identity == (self.session.session, self.session.world_id, 'yui')
                and event.get('op_id') == self.path_goal['op_id']
                and type(event.get('log_seq')) is int
                and event['log_seq'] > max(self.last_log,(self.path_terminal or {}).get('log_seq',0))
                and (self.path_terminal or {}).get('type')!='npc.operation_completed'
                and event.get('type') in {'npc.operation_completed','npc.operation_cancelled','npc.operation_failed'}):
                self.path_terminal = dict(event)
            if event.get('type') == 'sys.boot':
                self.path_goal = None
            if self.binding and (event.get('type') == 'sys.boot' or not self.eligible()):
                self.halt.set()
                if self.sender:
                    if self.session.estop:self.sender.stop()
                    else:self.sender.fault_stop()
                self.condition.notify_all()
                return
            if (event.get('session'), event.get('world_id'), event.get('npc_id')) != (
                    self.session.session, self.session.world_id, 'yui'):
                return
            seq = event.get('log_seq')
            if type(seq) is not int or seq <= self.last_log:
                return
            self.last_log = seq
            if self.receipts and event.get('type') == 'npc.pose_ack' and event.get('op_id') == self.binding['stream_id']:
                if self.receipts.ingest(event):
                    self.received_ack=dict(event)
                    items=event.get('stream_progress',[])
                    if not isinstance(items,list) or len(items)>2:items=[]
                    for index,item in enumerate(items):
                        if not isinstance(item,dict):continue
                        frame=item.get('executed_frame')
                        if type(frame) is int and frame>self.applied_frame:
                            self.application_events[frame]=dict(self.binding,**{k:item.get(k) for k in ('op_id','version','executed_frame','committed_frame')},
                                pose_ack_receipt=dict(event),progress_index=index)
            if event.get('type', '').startswith('npc.stream_') or event.get('type') in {
                    'npc.operation_completed', 'npc.operation_cancelled', 'npc.operation_failed'}:
                self.events.append(dict(event))
            if (self.binding and event.get('type')=='npc.stream_released'
                and all(event.get(k)==self.binding[k] for k in self.binding)):
                self.world_release=dict(event)
                if event.get('reason') not in {'stream_closed','normal_close'}:
                    self.error=self.error or event.get('reason') or 'world_stream_released'
                self.close_requested.set()
                if self.receipts:
                    self.receipts.quiesce_after_world_release()
            if event.get('type')=='npc.stream_progress':
                frame=event.get('executed_frame')
                if type(frame) is int and frame>self.applied_frame:
                    self.application_events[frame]=dict(event)
            self.condition.notify_all()

    def _wait(self, kind, predicate, timeout=.5):
        def find():
            return next((e for e in self.events if e.get('type') == kind and predicate(e)), None)
        with self.condition:
            self.condition.wait_for(lambda: find() is not None or not self.eligible(), timeout)
            event = find()
            if event is None or not self.eligible():
                raise RuntimeError(kind + '_unconfirmed')
            return dict(event)

    def _control(self, version, expected, *, op_id=None, **values):
        # 与200ms块互斥，保证控制标记不能插到尚未执行的上一意图中。
        with self.wire_lock:
            self.request_seq += 1
            request = self.request_seq
            binding = self.binding
            packet = encode_stream_control(version,binding['session'],binding['pose_epoch'],request,
                                           op_id or binding['stream_id'],**values)
            ticket = self.sender.send(tuple(MidiEvent(*e) for e in midi_events(packet)), 'pose')
            event = self._wait(expected,lambda e: e.get('request_seq') == request
                               and e.get('stream_id') == binding['stream_id']
                               and type(e.get('pose_epoch')) is int and e['pose_epoch'] > 0
                               and (version == 6 or e['pose_epoch'] == binding['pose_epoch']))
            self.sender.ack(ticket,binding['session'])
            return event

    def _request(self, path, **extra):
        started=time.perf_counter()
        self.last_request=path
        outcome='failed'
        try:
            result=self.backend._request('/streams/' + path,dict(self.binding,**extra))
            outcome='ok'
            return result
        finally:
            duration=time.perf_counter()-started
            self.max_request_seconds=max(self.max_request_seconds,duration)
            self.request_audit.append(dict(path=path,started=started,duration_s=duration,outcome=outcome))

    def start(self):
        if self.thread or not self.eligible() or self.session.control_state != 'external':
            return False
        self.thread = threading.Thread(target=self._run,name='yui-continuous-motion',daemon=True)
        self.thread.start()
        return True

    def _run(self):
        graceful = False
        lease_stop = threading.Event()
        lease_thread = None
        try:
            self.status = 'preparing'
            self.binding = dict(stream_id=uuid.uuid4().hex,session=self.session.session,
                                world_id=self.session.world_id,pose_epoch=1)
            self.sender = self.transport.attach_pose_sender(1,pose_pipeline=True)
            prepared = self._control(6,'npc.stream_prepared')
            self.binding['pose_epoch'] = prepared['pose_epoch']
            self.space = MotionSpace(origin=prepared.get('origin'),yaw=prepared.get('yaw'),scale=prepared.get('scale'),
                                     max_distance=10000 if self.wrap_root else 5,
                                     surface_path='pose_surface_path_v1' in self.session.capabilities)
            self._request('open',world_receipt=prepared,seed=self.seed,plan=dict(mode='idle',origin_xz=[0.,0.]))
            def lease():
                while not lease_stop.wait(.4):
                    try:
                        if not self.eligible():
                            raise RuntimeError('world_binding_lost')
                        self._request('world')
                    except Exception:
                        self.error = 'stream_lease_lost'
                        self.close_requested.set()
                        return
            lease_thread = threading.Thread(target=lease,name='yui-continuous-lease',daemon=True)
            lease_thread.start()
            deadline = time.monotonic() + 3
            block = None
            while self.eligible() and time.monotonic() < deadline:
                block = self._request('pull').get('block')
                if block:
                    break
                if self.halt.wait(.025):
                    break
            if not block:
                raise RuntimeError('stream_prewarm_timeout')
            self._compile(block)  # 在武装前校验首块，避免坏数据抢走原系统控制权。
            armed = self._control(7,'npc.stream_armed')
            self._request('arm',world_receipt=armed)
            self.receipts = PoseReceipts(self.sender,world_id=self.binding['world_id'],
                                         session=self.binding['session'],epoch=self.binding['pose_epoch'],max_in_flight=2)
            self.status = 'running'
            changed=getattr(self.backend,'_changed',None)
            if changed:changed()
            while self.eligible() and not self.close_requested.is_set():
                self.stage='apply_receipts'
                # 最多600ms可替换预收缓冲；200ms锁定边界由世界播放时钟单独回报。
                if self.sent_frame-self.applied_frame>=12:
                    self._exchange(pull=False)
                    if self.sent_frame-self.applied_frame>=12:
                        self.halt.wait(.01)
                        continue
                if block:
                    with self.wire_lock:
                        self.stage='send_block'
                        received=self._execute(block,defer_received=True)
                else:
                    received=None
                self.stage='pull_block'
                block = self._exchange(received=received,pull=True).get('block')
                if not block:
                    self.halt.wait(.01)
            if self.close_requested.is_set() and self.eligible():
                self.status = 'releasing'
                if self.world_release is None:
                    self._control(9,'npc.stream_released')
                self.transport.send_command("STOP")
                graceful = True
        except Exception as exc:
            self.error = self.error or str(exc)
            # 后端异常且世界仍有效：请求正常交回。身份丢失或急停不能走这条路径。
            if self.status == 'running' and self.eligible() and self.sender and not self.sender.closed.is_set():
                try:
                    if self.world_release is None:
                        self._control(9,'npc.stream_released')
                    self.transport.send_command("STOP")
                    graceful = True
                except Exception:
                    pass
        finally:
            lease_stop.set()
            if lease_thread:
                lease_thread.join(1)
            if self.binding:
                try:
                    self._request('close')
                except Exception:
                    pass
            if self.sender:
                try:
                    self.transport.detach_pose_sender(graceful=graceful)
                except Exception as exc:
                    self.error = self.error or str(exc)
            self.status = 'closed' if graceful and not self.error else 'failed'
            self.closed.set()
            changed=getattr(self.backend,'_changed',None)
            if changed:changed()

    def _compile(self, block, *, packet_bytes=False):
        if (block.get('stream_id') != self.binding['stream_id']
            or type(block.get('first_frame')) is not int or block['first_frame'] < 0
            or block['first_frame'] % 4 or block.get('end_frame') != block['first_frame']+4
            or block.get('mode') not in self.modes):
            raise ValueError('invalid_stream_block')
        task = dict(session=self.binding['session'],epoch=self.binding['pose_epoch'],
                    op_id=block['op_id'],prompt_version=block['version'])
        chunk = dict(task,sequence=1,fps=20,pose=block['pose'])
        packets = compile_chunk(chunk,task,expected_chunk=1,first_frame=block['first_frame']//2+1,
                                wire_epoch=self.wire_epoch or self.binding['pose_epoch'],max_frames=4,packet_bytes=packet_bytes,wrap_root=self.wrap_root)
        if len(packets) != 2:
            raise ValueError('invalid_stream_block_size')
        return packets

    def _execute(self, block, *, defer_received=False):
        packets = self._compile(block,packet_bytes=True)
        first = block['first_frame']//2+1
        if block['version'] != self.version:
            self._announce_intent(block)
            packets=[rebind_pose_epoch(packet,self.wire_epoch) for packet in packets]
        # 两个线帧允许在112条信用内交叠传输，不把每次ACK往返串进播放时长。
        for index,packet in enumerate(packets):
            events=tuple(MidiEvent(*event) for event in midi_events(packet))
            self.receipts.send(first+index,events)
        if not self.receipts.wait_received(first+len(packets)-1,.5):
            raise RuntimeError('stream_receive_timeout')
        with self.condition:
            received=dict(self.received_ack or {})
        if received.get('pose_sequence')!=block['end_frame']//2:
            raise RuntimeError('stream_received_identity_lost')
        payload=dict(world_receipt=received,wire_epoch=self.wire_epoch)
        if not defer_received:
            self._request('received',**payload)
        self.blocks[block['end_frame']]=block
        self.sent_frame=block['end_frame']
        return payload

    def _announce_intent(self, descriptor):
        # 有限、已验证的边界描述；同一版本只发送一次，不添加续租包或逐帧往返。
        with self.wire_lock:
            if descriptor.get('version')==self.version:return
            first=descriptor.get('first_frame')
            version=descriptor.get('version')
            if (descriptor.get('stream_id')!=self.binding['stream_id'] or type(first) is not int
                or first!=self.sent_frame or first%4 or type(version) is not int or version<=self.version
                or descriptor.get('mode') not in self.modes):
                raise ValueError('invalid_intent_announcement')
            marker=self._control(8,'npc.stream_control',op_id=descriptor['op_id'],first_sequence=first//2+1,
                intent_version=version,end_sequence=(descriptor.get('intent_end_frame') or 0)//2,
                mode=self.modes[descriptor['mode']],plan_request=descriptor.get('plan_request',0))
            if marker.get('op_id')!=descriptor['op_id'] or marker.get('version')!=version:
                raise RuntimeError('intent_marker_mismatch')
            epoch=marker.get('wire_epoch')
            if type(epoch) is not int or epoch<=0:raise RuntimeError('stream_wire_epoch_missing')
            self.wire_epoch=epoch
            self.receipts=PoseReceipts(self.sender,world_id=self.binding['world_id'],
                                      session=self.binding['session'],epoch=epoch,max_in_flight=2)
            self.version=version

    def _pending_applications(self):
        with self.condition:
            pending=list(sorted(self.application_events.items()))
            events=list(self.events)
        applications=[]
        for frame,progress in pending[:8]:
            block=self.blocks.get(frame)
            if not block:
                continue
            if (any(progress.get(k)!=self.binding[k] for k in self.binding)
                or progress.get('op_id')!=block['op_id'] or progress.get('version')!=block['version']):
                raise RuntimeError('stream_application_identity_mismatch')
            terminal={}
            if block['final']:
                terminal=next((e for e in events if e.get('type')=='npc.operation_completed'
                    and e.get('op_id')==block['op_id'] and e.get('kind')=='motion'
                    and e.get('result')=='motion_completed'),None)
                if not terminal:
                    break
            applications.append((frame,block,dict(world_receipt=progress.get('pose_ack_receipt',progress),
                          progress_index=progress.get('progress_index'),operation_receipt=terminal)))
        return applications

    def _accept_applications(self, applications):
        for frame,block,payload in applications:
            self.applied_frame=frame
            self.origin_xz=[block['pose']['root_positions'][-1][axis] for axis in (0,2)]
            self.last_block=block
            with self.condition:
                self.application_events.pop(frame,None)
            for older in list(self.blocks):
                if older<frame-20:
                    self.blocks.pop(older,None)

    def _process_applications(self):
        for item in self._pending_applications():
            self._request('ack',**item[2])
            self._accept_applications([item])

    def _exchange(self, *, received=None, pull=True):
        applications=self._pending_applications()
        if received is None and not applications and not pull:
            return {'block':None}
        result=self._request('exchange',received=received,pull=pull,
                             applications=[item[2] for item in applications])
        # 网络失败不提前推进本机应用水位；世界回执不等于后端已确认处理。
        self._accept_applications(applications)
        if pull and result.get('block') is None and result.get('control'):
            self._announce_intent(result['control'])
        return result

    def perform(self, session, *, prompt, target_key=None, mode=None, duration_s=None, end_condition=None, source='explicit', **extra):
        if source == 'explicit':
            self.discard_path_goal()
        if not self.ready(session):
            return dict(status='failed',error='continuous_execution_unavailable',midi_sent=False)
        if extra:
            return dict(status='failed',error='unsupported_motion_arguments',midi_sent=False)
        mode = mode or ('path' if target_key else 'free_action')
        end_condition=end_condition or ('arrived' if target_key else 'until_replaced' if source=='base' else 'duration')
        if end_condition not in {'duration','arrived','until_replaced'}:
            return dict(status='failed',error='unsupported_end_condition',midi_sent=False)
        if end_condition=='arrived' and not target_key:
            return dict(status='failed',error='arrival_target_required',midi_sent=False)
        if target_key and (mode!='path' or end_condition=='duration'):
            return dict(status='failed',error='path_requires_arrival_or_replacement',midi_sent=False)
        duration_s=None if end_condition=='until_replaced' else duration_s or 4
        if mode not in {'idle','free_action','path'}:
            return dict(status='failed',error='mode_not_world_validated',midi_sent=False)
        try:
            # 世界速度单位为米/秒；模型轨迹已除以运动尺度，速度也必须同步转换。
            plan = dict(mode=mode,origin_xz=list(self.origin_xz),max_speed=.8/self.space.scale)
            if target_key:
                anchor = session._target_anchor_id(target_key)
                if type(anchor) is not int or not 0 <= anchor <= 126:
                    raise ValueError('verified_anchor_unavailable')
                with self.wire_lock:
                    receipt = self._control(10,'npc.stream_path',anchor_id=anchor)
                    if receipt.get('status')=='failed':
                        return dict(status='failed',error=receipt.get('reason','world_path_rejected'),midi_sent=True)
                    path = [self.space.source_point(p) for p in receipt['points']]
                    plan.update(mode='path',origin_xz=path[0],path=path,plan_request=receipt['request_seq'])
                    if self.space.surface_path:
                        plan['surface_path_xyz']=[list(p) for p in receipt['points']]
                        if max(p[1] for p in receipt['points'])-min(p[1] for p in receipt['points'])>.25:
                            # 跨楼层路线使用独立步速，给NPC较短的腿保留换阶和支撑转移时间。
                            plan['max_speed']=min(plan['max_speed'],.45/self.space.scale)
                    if 'pose_curved_path_v1' in self.session.capabilities and 'corner_trims' in receipt:
                        plan['corner_trims']=[v/self.space.scale for v in receipt['corner_trims']]
                    if end_condition=='arrived':
                        duration_s=max(.2,sum(math.dist(a,b) for a,b in zip(path,path[1:]))/plan['max_speed']+.4)
                    op_id=uuid.uuid4().hex
                    # 先登记再发送，极短路径的完成回执也不会越过登记窗口。
                    if source=='explicit' and end_condition=='arrived':
                        with self.condition:
                            self.path_goal=dict(op_id=op_id,accepted=False,arguments=dict(prompt=prompt,target_key=target_key,
                                mode='path',end_condition='arrived',source=source))
                            self.path_terminal=None
                    result=self._request('intent',prompt=prompt,plan=plan,duration_s=duration_s,
                                         source=source,op_id=op_id)
                    if source=='explicit':
                        with self.condition:
                            if self.path_goal and self.path_goal['op_id']==op_id:
                                if result.get('status')=='accepted':self.path_goal['accepted']=True
                                else:self.discard_path_goal()
                    return result
            return self._request('intent',prompt=prompt,plan=plan,duration_s=duration_s,
                                 source=source,op_id=uuid.uuid4().hex)
        except Exception as exc:
            if source=='explicit':self.discard_path_goal()
            return dict(status='failed',error=str(exc),midi_sent=False)

    def discard_path_goal(self):
        with self.condition:
            self.path_goal=None
            self.path_terminal=None

    def resumable_path_goal(self):
        with self.condition:
            terminal=self.path_terminal or {}
            recoverable={'stream_underrun','pose_lease_lost','transport_fault','stream_receive_timeout'}
            if 'pose_surface_recovery_v1' in getattr(self.session,'capabilities',[]):
                recoverable.update({'foot_no_ground','unsupported_ground','obstacle','foot_correction_limit','foot_unreachable','pelvis_correction_limit'})
            # 租约可能在新任务首帧之前失效，此时世界只会取消仍在播放的旧待机任务。
            # 仅恢复已明确接受且无任务终态的语义目标；未知提交、替换和急停均不恢复。
            pending_lost=bool(self.path_goal and self.path_goal.get('accepted') and not terminal
                              and (self.world_release or {}).get('reason') in recoverable)
            if (not self.session.estop and self.path_goal
                and (pending_lost or (terminal.get('type')=='npc.operation_cancelled'
                     and terminal.get('reason') in recoverable))):
                return dict(op_id=self.path_goal['op_id'],arguments=dict(self.path_goal['arguments']))
            return None

    def stop(self):
        self.discard_path_goal()
        self.halt.set()
        if self.sender:
            self.sender.stop()
        with self.condition:
            self.condition.notify_all()

    def close(self):
        self.close_requested.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(3)
            if self.thread.is_alive():
                self.stop()
        self.session.remove_event_listener(self.ingest)
