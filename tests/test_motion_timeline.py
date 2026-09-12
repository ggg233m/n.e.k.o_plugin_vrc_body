"""区分可替换预收缓冲、世界200ms锁定边界和实际应用。"""
import pytest
import _bootstrap
from integrations.motion_service.timeline import MotionTimeline

class Generator:
    fps=20
    frames=40
    hook=None
    def generate_candidate(self,prompt,constraints,history=None):
        if self.hook:
            hook,self.hook=self.hook,None
            hook()
        sample=list(history or [])+[(prompt,i) for i in range(40)]
        return {k:[v]*40 for k,v in [('root_positions',[0,0,0]),('local_rot_mats',[]),('foot_contacts',[1,1,1,1])]},sample
    def history_at(self,sample,frames):
        end=len(sample)-40+frames
        return sample[max(0,end-20):end]

def timeline():return MotionTimeline(Generator(),lambda *_:{})
def receipt(block,terminal=True):
    return dict(stream_id=block['stream_id'],version=block['version'],op_id=block['op_id'],executed_frame=block['end_frame'],world_applied=True,operation_completed=terminal)
def receive(t,b):
    t.received(dict(stream_id=b['stream_id'],version=b['version'],op_id=b['op_id'],received_frame=b['end_frame']))
def apply(t,b):
    receive(t,b);return t.acknowledge(receipt(b))

def test_replacement_uses_selected_checkpoint_and_preserves_locked_prefix():
    t=timeline();t.update('walk',{},duration_s=4);assert t.generate()
    old=t.pull();apply(t,old)
    locked=t.committed_history.copy()
    candidate=t.update('wave',{},duration_s=4)
    assert candidate['start_frame']==28 and t.committed_frame==4 and t.committed_history==locked
    assert t.future_history==t.checkpoints[28]
    assert all(b['end_frame']<=28 for b in t.future)
    assert t.generate()
    while True:
        b=t.pull();apply(t,b)
        if b['op_id']==candidate['op_id']:break
        assert b['end_frame']<=28
    assert b['first_frame']==28 and b['version']==candidate['version']

def test_stale_inflight_generation_never_commits_history():
    t=timeline();t.update('walk',{})
    t.generator.hook=lambda:t.update('wave',{})
    assert not t.generate();assert not t.future and t.future_history is None
    assert t.discarded_candidates==1 and t.generate() and t.pull()['version']==2

def test_terminal_does_not_close_stream_and_idle_is_prefetched():
    t=timeline();op=t.update('wave',{},duration_s=.2);t.generate();t.generate()
    b=t.pull();assert b['final'];apply(t,b)
    assert t.records[op['op_id']]['status']=='succeeded' and not t.closed
    b=t.pull();assert not b['final'] and t.intent.source=='base'

def test_receiving_future_is_not_application_or_immutable_commit():
    t=timeline();t.update('wave',{},duration_s=.2);t.generate();t.generate()
    first=t.pull();receive(t,first)
    second=t.pull();receive(t,second)
    assert t.delivered_frame==8 and t.executed_frame==t.committed_frame==0
    with pytest.raises(ValueError):t.acknowledge(dict(receipt(first),world_applied=False))
    with pytest.raises(ValueError):t.acknowledge(receipt(first,False))
    assert t.records[first['op_id']]['status']=='running'
    t.acknowledge(receipt(first));assert t.records[first['op_id']]['status']=='succeeded'

def test_one_hundred_scheduled_switches_never_revive_discarded_future():
    t=timeline()
    for i in range(100):
        op=t.update(str(i),{});assert t.generate()
        while True:
            b=t.pull();assert b is not None
            apply(t,b)
            if b['op_id']==op['op_id']:break
            assert b['end_frame']<=op['start_frame']
    assert t.executed_frame==2776 and len(t.committed_history)==20
    assert len(t.checkpoints)<40

def test_close_cannot_resurrect_inflight_or_accept_new_intent():
    t=timeline();t.update('walk',{});t.generator.hook=t.close
    assert not t.generate()
    with pytest.raises(RuntimeError):t.update('wave',{})
