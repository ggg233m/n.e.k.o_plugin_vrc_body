"""校准、独立收发、互斥配置及真实 UDP 的验证。"""
import math
import socket
import struct
import time
from types import SimpleNamespace
import pytest
from yui_npc_controller.runtime import vmc_sharing as vmc
from yui_npc_controller.runtime.vmc_receiver import VmcConfig, VmcReceiver
from yui_npc_controller.runtime.vmc_backend import VmcBackend, VmcStream
from yui_npc_controller.runtime.config import YuiPluginConfig
from yui_npc_controller.runtime.control_panel import validated_patch


def frame(capture,rotations=None,missing=None):
    capture.feed('/VMC/Ext/T',[1.])
    for name in vmc.BONES.values():
        if name!=missing: capture.feed('/VMC/Ext/Bone/Pos',[name,0.,1.,0.,*(rotations or {}).get(name,vmc.IDENTITY)])
    return capture.feed('/VMC/Ext/T',[2.])


def calibrated():
    c=vmc.Capture(); c.calibrate=True
    return frame(c)


def test_calibrated_fk_matches_host_global_rotation():
    c=vmc.Capture(); c.calibrate=True
    rest={'LeftUpperArm':(0,0,math.sin(.4),math.cos(.4))}
    sample=frame(c,rest)
    assert all(abs(q[3]-1)<1e-6 for q in sample['rotations'].values())
    sample=frame(c,dict(rest,Spine=(0,math.sin(.2),0,math.cos(.2))))
    result=vmc.pose_from_sample(sample)
    assert result['root_positions']==[[0,.9544,0]]*4
    from yui_npc_controller.runtime.pose_frames import matrix_quaternion
    globals_=[]
    for i,m in enumerate(result['local_rot_mats'][0]):
        q=matrix_quaternion(m); p=vmc.PARENTS[i]
        globals_.append(vmc.multiply(globals_[p],q) if p>=0 else q)
    for i,name in vmc.BONES.items():
        x,y,z,w=globals_[i]
        assert abs(sum(a*b for a,b in zip((x,-y,-z,w),sample['rotations'][name])))>1-1e-6


def test_incomplete_and_stale_frames_are_never_replayed():
    c=vmc.Capture(); c.calibrate=True
    assert frame(c) is not None
    assert frame(c,missing='LeftHand') is None
    c.feed('/VMC/Ext/OK',[0]); assert frame(c) is None
    sample=calibrated(); sample['captured_at']-=1
    with pytest.raises(ValueError,match='stale'): vmc.pose_from_sample(sample)


def test_config_and_panel_enforce_mutual_exclusion():
    with pytest.raises(ValueError,match='不能同时'):
        YuiPluginConfig.from_mapping({'ardy':{'enabled':True},'vmc':{'enabled':True}})
    assert validated_patch({'yui':{'ardy':{'enabled':True}}},{'vmc.enabled':True})=={'yui':{'vmc':{'enabled':True},'ardy':{'enabled':False}}}
    assert validated_patch({'yui':{'vmc':{'enabled':True}}},{'ardy.enabled':True})=={'yui':{'ardy':{'enabled':True},'vmc':{'enabled':False}}}
    with pytest.raises(ValueError): validated_patch({}, {'ardy.enabled':True,'vmc.enabled':True})
    with pytest.raises(ValueError): VmcConfig.from_mapping({'host_api_url':'http://example.com:48911'})


def receiver_stub():
    return SimpleNamespace(latest=calibrated,snapshot=lambda:dict(ready=True,state='receiving',frames=1),start=lambda:None,close=lambda:None)


def opened_stream():
    stream=VmcStream(receiver_stub())
    identity=dict(stream_id='a'*32,session=7,world_id='world',pose_epoch=3)
    stream.request('open',dict(identity,world_receipt=dict(identity,type='npc.stream_prepared')))
    stream.request('arm',dict(identity,world_receipt=dict(identity,type='npc.stream_armed')))
    return stream,identity


def test_vmc_has_no_ardy_dependency_and_requires_world_receipts():
    backend=VmcBackend(VmcConfig(enabled=True),receiver=receiver_stub())
    backend._opener.open=lambda *a,**k: pytest.fail('不能连接 ARDY')
    assert backend._request('/health')['ready']
    assert backend.snapshot()['backend']=='vmc'
    assert not backend.world_ready(None)
    stream,identity=opened_stream(); block=stream.request('pull',identity)['block']
    with pytest.raises(ValueError): stream.request('received',dict(identity,world_receipt={}))
    assert stream.next_frame==0 and stream.applied_frame==0
    receipt=dict(type='npc.pose_ack',state='received',op_id=identity['stream_id'],pose_sequence=2,
                 pose_session=7,stream_epoch=3,pose_epoch=4,session=7,world_id='world')
    stream.request('received',dict(identity,world_receipt=receipt,wire_epoch=4))
    assert stream.next_frame==4 and stream.applied_frame==0
    progress=dict(identity,type='npc.stream_progress',op_id=block['op_id'],version=1,executed_frame=4,committed_frame=4)
    with pytest.raises(ValueError): stream.request('ack',dict(identity,world_receipt=dict(progress,executed_frame=8)))
    stream.request('ack',dict(identity,world_receipt=progress)); assert stream.applied_frame==4
    stream.request('close',identity)
    with pytest.raises(ValueError): stream.request('pull',identity)


def osc(address,args):
    def string(s):
        b=s.encode()+b'\0'; return b+b'\0'*((-len(b))%4)
    tags=''; payload=b''
    for arg in args:
        if isinstance(arg,str): tags+='s'; payload+=string(arg)
        else: tags+='f'; payload+=struct.pack('>f',arg)
    return string(address)+string(','+tags)+payload


def test_receiver_real_udp_calibrates_and_restores_prior_host_settings():
    # 只替换 HTTP 宿主；UDP 使用真实本机端口，覆盖线程退出与原配置恢复。
    probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); probe.bind(('127.0.0.1',0)); port=probe.getsockname()[1]; probe.close()
    prior=dict(enabled=False,host='127.0.0.1',port=39539,send_rate_hz=60)
    state=dict(prior,t_pose_generation=0,t_pose_requested=False); calls=[]
    def api(path,payload):
        calls.append((path,payload))
        if path=='/api/config/page_config': return {'autostart_csrf_token':'test'}
        if path=='/api/vmc/enable': state.update(payload,enabled=True)
        if path=='/api/vmc/disable': state['enabled']=False
        if path=='/api/vmc/t_pose': state.update(t_pose_generation=1,t_pose_requested=False)
        return dict(state)
    receiver=VmcReceiver(VmcConfig(enabled=True,listen_port=port),requester=api)
    sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    receiver.start()
    try:
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and receiver.latest() is None:
            sender.sendto(osc('/VMC/Ext/T',[1.]),('127.0.0.1',port))
            for name in vmc.BONES.values(): sender.sendto(osc('/VMC/Ext/Bone/Pos',[name,0.,1.,0.,0.,0.,0.,1.]),('127.0.0.1',port))
            time.sleep(.03)
        sample=receiver.latest(); assert sample is not None,receiver.snapshot()
        from yui_npc_controller.runtime.pose_frames import compile_chunk
        from yui_npc_controller.runtime.pose_codec import decode
        task=dict(op_id='test',session=7,epoch=3,prompt_version=1)
        packets=compile_chunk(dict(task,sequence=1,fps=20,pose=vmc.pose_from_sample(sample)),task,
                              expected_chunk=1,first_frame=1,wire_epoch=3,max_frames=4,packet_bytes=True)
        assert len(packets)==2 and decode(packets[1])['sequence']==2
        receiver.sample=dict(sample,captured_at=time.time()-1)
        assert receiver.latest() is None
    finally:
        receiver.close(); sender.close()
    assert not receiver.thread.is_alive()
    assert {k:state[k] for k in prior}==prior
    assert receiver.snapshot()['state']=='stopped'
