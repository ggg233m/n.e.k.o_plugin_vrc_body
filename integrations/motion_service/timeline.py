"""连续运动时间线：未来可撤销，已提交的200ms块不可倒写。"""
from collections import deque, OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import uuid


@dataclass(frozen=True)
class Intent:
    operation_id: str
    version: int
    prompt: str
    source: str
    start_frame: int
    end_frame: int | None
    constraints: dict


class MotionTimeline:
    """不持有GPU锁；候选历史仅在版本仍有效时提交到预测队列。"""
    block_frames = 4
    fps = 20
    planning_lead_frames = 24

    def __init__(self, generator, compile_constraints, *, stream_id=None, future_limit=80):
        if generator.fps != self.fps or generator.frames % self.block_frames:
            raise ValueError('unsupported_continuous_horizon')
        self.generator = generator
        self.compile_constraints = compile_constraints
        self.stream_id = stream_id or uuid.uuid4().hex
        self.future_limit = future_limit
        self.lock = threading.RLock()
        self.intent = None
        self.version = 0
        self.committed_frame = self.executed_frame = self.generated_frame = 0
        self.committed_history = self.future_history = None
        self.future = deque()
        self.outstanding = None
        self.reserved_frame = self.delivered_frame = 0
        self.delivered = OrderedDict()
        self.checkpoints = {0: None}
        self.checkpoint_pose = {}
        self.records = OrderedDict()
        self.closed = False
        self.generating = False
        self.discarded_candidates = 0

    def update(self, prompt, constraints, *, source='explicit', duration_s=4, operation_id=None):
        if not isinstance(prompt,str) or not 1 <= len(prompt.strip()) <= 320:
            raise ValueError('invalid_prompt')
        if source not in {'explicit','base'}:
            raise ValueError('invalid_source')
        if duration_s is not None and (type(duration_s) not in (int,float) or not math.isfinite(duration_s) or not .2 <= duration_s <= 300):
            raise ValueError('invalid_duration')
        with self.lock:
            if self.closed:
                raise RuntimeError('stream_closed')
            old = self.intent
            if source == 'base' and any(r.get('source')=='explicit' and r['status'] not in {'succeeded','cancelled','failed'} for r in self.records.values()):
                return {'status':'deferred','reason':'explicit_intent_active'}
            if old and old.source == source and old.prompt == prompt and old.constraints == constraints and duration_s is None:
                return dict(self.records[old.operation_id])
            operation_id = operation_id or uuid.uuid4().hex
            if operation_id in self.records:
                raise ValueError('operation_id_reused')
            # 推理在仍可修改的未来检查点开始，预留1.2秒完成候选和传输（实测路径首次候选加控制传输超过0.8秒）。
            # 已接收的帧仍可由世界仲裁，但宿主不以放大不可变窗口换取吞吐。
            boundary = self.reserved_frame
            if self.delivered_frame:
                boundary = max(boundary, self.executed_frame + self.planning_lead_frames)
                boundary = math.ceil(boundary/self.block_frames)*self.block_frames
            if boundary not in self.checkpoints:
                raise ValueError('planning_checkpoint_not_ready')
            selected_constraints=deepcopy(constraints)
            if selected_constraints.get('mode') in {'path','velocity'}:
                from .motion_modes import path_profile
                selected_constraints['start_heading']=self.checkpoint_pose.get(boundary,{}).get('heading',0.)
                _,travel_seconds,_=path_profile(selected_constraints)
                if duration_s is not None:
                    duration_s=max(duration_s,travel_seconds+.4)
                    if duration_s>300:raise ValueError('path_duration_exceeds_limit')
            for record in self.records.values():
                end = record.get('end_frame')
                if record['status'] in {'accepted','running'} and (end is None or end > boundary):
                    record.update(status='cancelling',reason='replaced',boundary=boundary)
            self.version += 1
            end = None if duration_s is None else boundary + math.ceil(duration_s*self.fps/self.block_frames)*self.block_frames
            if selected_constraints.get('mode')=='idle' and boundary in self.checkpoint_pose:
                point=self.checkpoint_pose[boundary]
                selected_constraints.update(origin_xz=point['root_xz'],heading=point['heading'])
            self.intent = Intent(operation_id,self.version,prompt,source,boundary,end,selected_constraints)
            # 保留已发出但未执行的块，仅撤销还未交给世界的未来。
            self.future = deque(block for block in self.future if block['end_frame']<=boundary)
            self.generated_frame = boundary
            self.future_history = self.checkpoints[boundary]
            self.checkpoints = {frame:history for frame,history in self.checkpoints.items() if frame<=boundary}
            self.records[operation_id] = dict(status='accepted',op_id=operation_id,stream_id=self.stream_id,version=self.version,start_frame=boundary,end_frame=end,source=source)
            if len(self.records) > 256:
                for key in list(self.records):
                    if len(self.records) <= 256:
                        break
                    if self.records[key]['status'] in {'succeeded','cancelled','failed'}:
                        self.records.pop(key)
            return dict(self.records[operation_id])

    def generate(self):
        with self.lock:
            if self.closed or not self.intent or self.generating or len(self.future)*self.block_frames+self.generator.frames > self.future_limit:
                return False
            intent = self.intent
            if intent.end_frame is not None and self.generated_frame >= intent.end_frame:
                return False
            self.generating = True
            start, history = self.generated_frame, self.future_history
        try:
            constraints = self.compile_constraints(intent, start, self.generator.frames, self.fps)
            pose, sample = self.generator.generate_candidate(intent.prompt,constraints,history)
            for key in ('local_rot_mats','root_positions','foot_contacts'):
                if len(pose.get(key,())) != self.generator.frames:
                    raise ValueError('invalid_candidate_frames')
            count = self.generator.frames if intent.end_frame is None else min(self.generator.frames,intent.end_frame-start)
            blocks = []
            for offset in range(0,count,self.block_frames):
                stop = offset+self.block_frames
                blocks.append(dict(stream_id=self.stream_id,version=intent.version,op_id=intent.operation_id,
                    first_frame=start+offset,end_frame=start+stop,final=start+stop==intent.end_frame,
                    intent_end_frame=intent.end_frame,mode=intent.constraints.get("mode","idle"),
                    plan_request=intent.constraints.get("plan_request",0),
                    pose={key:deepcopy(value[offset:stop]) for key,value in pose.items()},
                    history=self.generator.history_at(sample,stop)))
            with self.lock:
                if self.closed or self.intent is not intent:
                    self.discarded_candidates += 1
                    return False
                self.future.extend(blocks)
                self.checkpoints.update({block['end_frame']:block['history'] for block in blocks})
                for block in blocks:
                    position=block['pose']['root_positions'][-1]
                    heading=block['pose'].get('root_heading',[intent.constraints.get('heading',0.)])[-1]
                    self.checkpoint_pose[block['end_frame']]=dict(root_xz=[position[0],position[2]],heading=heading)
                self.generated_frame = start+count
                self.future_history = blocks[-1]['history']
                if intent.end_frame is not None and self.generated_frame == intent.end_frame:
                    # 提前生成任务后的待机，不等世界终态再启动GPU导致断粮。
                    self.version += 1
                    base_id = uuid.uuid4().hex
                    position = pose['root_positions'][count-1]
                    base_constraints = dict(intent.constraints,mode='idle',origin_xz=[position[0],position[2]],
                        heading=pose.get('root_heading',[intent.constraints.get('heading',0.)]*count)[count-1],path=None,keyframes=[])
                    self.intent = Intent(base_id,self.version,'A person stands calmly with relaxed arms.',
                                         'base',self.generated_frame,None,base_constraints)
                    self.records[base_id] = dict(status='accepted',op_id=base_id,stream_id=self.stream_id,
                                                version=self.version,start_frame=self.generated_frame,end_frame=None,source='base')
                return True
        except Exception as exc:
            with self.lock:
                if self.intent is intent and not self.closed:
                    self.records[intent.operation_id].update(status='failed',error=str(exc))
                    self.closed = True
                    raise
                self.discarded_candidates += 1
                return False
        finally:
            with self.lock:
                self.generating = False

    def pull(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('stream_closed')
            # 在一次200ms提交内重试读取返回同一块，不能消费两次历史。
            if self.outstanding is not None:
                return deepcopy({k:v for k,v in self.outstanding.items() if k!='history'})
            if not self.future:
                return None
            block = self.future.popleft()
            assert block['first_frame'] == self.reserved_frame
            self.reserved_frame = block['end_frame']
            self.outstanding = block
            record = self.records[block['op_id']]
            if record['status'] == 'accepted':
                record['status'] = 'running'
            return deepcopy({k:v for k,v in block.items() if k!='history'})

    def pending_control(self):
        """旧帧已确认后即可声明新边界，不等待GPU候选；不推进任何播放水位。"""
        with self.lock:
            intent=self.intent
            if (self.closed or intent is None or self.outstanding is not None or self.future
                or self.delivered_frame!=intent.start_frame or self.reserved_frame!=intent.start_frame):
                return None
            return dict(stream_id=self.stream_id,op_id=intent.operation_id,version=intent.version,
                        first_frame=intent.start_frame,intent_end_frame=intent.end_frame,
                        mode=intent.constraints.get('mode','idle'),plan_request=intent.constraints.get('plan_request',0))

    def received(self, receipt):
        """世界接收只推进可替换缓冲，不声明应用或任务完成。"""
        with self.lock:
            block=self.outstanding
            if (self.closed or block is None or
                (receipt.get('stream_id'),receipt.get('version'),receipt.get('op_id'),receipt.get('received_frame')) !=
                (self.stream_id,block['version'],block['op_id'],block['end_frame'])):
                raise ValueError('receive_identity_mismatch')
            self.delivered[block['end_frame']]=block
            self.delivered_frame=block['end_frame']
            self.outstanding=None
            return self.snapshot()

    def acknowledge(self, receipt):
        """接收与应用分别验证；延迟终态可以补充，但不能伪造未来应用。"""
        with self.lock:
            end=receipt.get('executed_frame')
            block=self.delivered.get(end)
            if self.closed or block is None or receipt.get('world_applied') is not True:
                raise ValueError('unexpected_application_receipt')
            if (receipt.get('stream_id'),receipt.get('version'),receipt.get('op_id')) != (self.stream_id,block['version'],block['op_id']):
                raise ValueError('application_identity_mismatch')
            committed=receipt.get('committed_frame',end)
            if type(committed) is not int or committed%4 or not end<=committed<=self.delivered_frame:
                raise ValueError('invalid_world_commit_boundary')
            self.executed_frame=max(self.executed_frame,end)
            self.committed_frame=max(self.committed_frame,committed)
            self.committed_history=self.checkpoints[self.committed_frame]
            for record in self.records.values():
                if record['status']=='cancelling' and self.executed_frame>=record['boundary']:
                    record['status']='cancelled'
            record=self.records[block['op_id']]
            if block['final'] and record['status'] not in {'cancelling','cancelled'}:
                if receipt.get('operation_completed') is not True:
                    raise ValueError('operation_terminal_missing')
                record['status']='succeeded'
            # 保留有界历史以处理遥测延迟，早于最近一秒的记录可以释放。
            for frame in list(self.delivered):
                if frame<self.executed_frame-20:
                    del self.delivered[frame]
            for frame in list(self.checkpoints):
                if frame<self.executed_frame-20:
                    del self.checkpoints[frame]
                    self.checkpoint_pose.pop(frame,None)
            return dict(record)

    def close(self, reason='closed'):
        with self.lock:
            self.closed = True
            self.future.clear()
            for record in self.records.values():
                if record['status'] not in {'succeeded','failed','cancelled'}:
                    record.update(status='cancelled',reason=reason)
            self.committed_history = self.future_history = None
            self.outstanding = None
            self.delivered.clear()
            self.checkpoints.clear()
            self.checkpoint_pose.clear()

    def snapshot(self):
        with self.lock:
            return dict(stream_id=self.stream_id,version=self.version,closed=self.closed,
                        generated_frame=self.generated_frame,committed_frame=self.committed_frame,
                        reserved_frame=self.reserved_frame,delivered_frame=self.delivered_frame,
                        executed_frame=self.executed_frame,buffered_frames=len(self.future)*self.block_frames,
                        discarded_candidates=self.discarded_candidates,
                        operation=dict(self.records[self.intent.operation_id]) if self.intent else None)
