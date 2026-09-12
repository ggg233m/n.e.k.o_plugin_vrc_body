"""缩短单帧发送不能放宽滚动秒预算、未确认信用或停止上限。"""
from collections import deque
import pytest
import _bootstrap
from yui_npc_controller.runtime.pose_shared_port import SharedPort
from yui_npc_controller.runtime.yui_protocol import MidiEvent


@pytest.mark.parametrize('pipeline,tick_seconds',[(False,.0006),(True,.0003)])
def test_short_bursts_keep_rolling_budget_and_credit_bound(pipeline,tick_seconds):
    event=MidiEvent('cc', 15, 1, 1)
    sent=[]
    now=0.
    port=SharedPort(lambda e:sent.append(now),7,1,[event,event],pose_pipeline=pipeline)
    sequence=0
    window=deque()
    seen=0
    for tick in range(round(2.04/tick_seconds)):
        now=tick*tick_seconds
        if now>=sequence*.1 and not port.pending and not port.active and not port.queues['pose']:
            sequence+=1
            assert port.submit('pose',sequence,[event]*78,now)
        port.step(now)
        assert port.outstanding<=112
        for key in list(port.pending):
            assert port.ack(key[0],7,1,key[1],now)
        for timestamp in sent[seen:]:
            while window and timestamp-window[0]>=1:
                window.popleft()
            window.append(timestamp)
            assert len(window)<=998
        seen=len(sent)
        assert not port.stopped
    assert len(sent)>1500
    assert sent[77]-sent[0]<.06
    if pipeline:assert sent[77]-sent[0]<.025


def test_full_throttle_overload_keeps_998_normal_events_and_reserved_stop():
    event=MidiEvent('cc',15,1,1);sent=[]
    port=SharedPort(sent.append,7,1,[event,event],pose_pipeline=True)
    sequence=0
    for tick in range(4000):
        now=tick*.0003
        if not port.pending and not port.active and not port.queues['pose']:
            sequence+=1;assert port.submit('pose',sequence,[event]*78,now)
        port.step(now)
        if port.stopped:break
        for lane,seq in list(port.pending):assert port.ack(lane,7,1,seq,now)
    # 不受控生产者耗尽滚动预算时不能继续普通发送，事务失效后只补两个急停。
    assert port.stopped and port.reason=='receipt_timeout'
    assert len(sent)==1000 and port.outstanding<=112


def test_missing_ack_still_stops_without_sending_another_pose():
    event=MidiEvent('cc',15,1,1)
    sent=[]
    port=SharedPort(sent.append,7,1,[event,event])
    assert port.submit('pose',1,[event]*78,0)
    for tick in range(100):
        port.step(tick*.0006)
    assert len(sent)==78
    assert not port.submit('pose',2,[event]*78,.07)
    port.step(.51)
    assert port.stopped and port.reason=='receipt_timeout'
    assert len(sent)==80


def test_pipeline_stops_at_112_until_verified_receipt_then_continues_same_packet():
    event=MidiEvent('cc',15,1,1);sent=[]
    port=SharedPort(sent.append,7,1,[event,event],pose_pipeline=True)
    assert port.submit('pose',1,[event]*78,0)
    for tick in range(100):port.step(tick*.0006)
    assert len(sent)==78
    assert port.submit('pose',2,[event]*78,.06)
    for tick in range(100,200):
        port.step(tick*.0006);assert port.outstanding<=112
    assert len(sent)==112 and port.active.offset==34
    assert not port.ack('pose',8,1,1,.12)
    port.step(.121);assert len(sent)==112
    assert port.ack('pose',7,1,1,.122)
    for tick in range(210,310):port.step(tick*.0006)
    assert len(sent)==156 and port.outstanding==78
    assert port.ack('pose',7,1,2,.19)
    assert port.outstanding==0 and not port.stopped


def test_pipeline_missing_ack_keeps_reserved_stop_within_114_events():
    event=MidiEvent('cc',15,1,1);sent=[]
    port=SharedPort(sent.append,7,1,[event,event],pose_pipeline=True)
    port.submit('pose',1,[event]*78,0)
    for tick in range(100):port.step(tick*.0006)
    port.submit('pose',2,[event]*78,.06)
    for tick in range(100,200):port.step(tick*.0006)
    port.step(.51)
    assert len(sent)==114 and port.stopped and port.reason=='receipt_timeout'


def test_expired_unsent_transactions_do_not_accumulate_audit_entries():
    event=MidiEvent('cc',15,1,1)
    port=SharedPort(lambda _:None,7,1,[event,event])
    for sequence in range(1,50):
        now=sequence*2.
        assert port.submit('pose',sequence,[event]*78,now)
        port.step(now+.6)
        assert not port.audit_pending and not port.stopped
    assert len(port.audit)==32 and all('expired' in item for item in port.audit)
