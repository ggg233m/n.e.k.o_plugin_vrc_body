"""严格校验生成分段并生成有界的 10 FPS 姿态包，不依赖 NumPy 或模型库。"""
import math

from .pose_codec import encode, midi_events
from .yui_protocol import MidiEvent


def matrix_quaternion(m):
    if not isinstance(m, list) or len(m) != 3 or any(not isinstance(r,list) or len(r)!=3 for r in m):
        raise ValueError("invalid_rotation_shape")
    if any(type(v) not in (int,float) or not math.isfinite(v) for r in m for v in r):
        raise ValueError("invalid_rotation_number")
    for i in range(3):
        for j in range(3):
            if abs(sum(m[k][i]*m[k][j] for k in range(3))-(1 if i==j else 0))>.03:
                raise ValueError("rotation_not_orthonormal")
    det=sum(m[0][i]*(m[1][(i+1)%3]*m[2][(i+2)%3]-m[1][(i+2)%3]*m[2][(i+1)%3]) for i in range(3))
    if det<.95:
        raise ValueError("rotation_is_reflection")
    candidates=[1+m[0][0]-m[1][1]-m[2][2],1-m[0][0]+m[1][1]-m[2][2],
                1-m[0][0]-m[1][1]+m[2][2],1+m[0][0]+m[1][1]+m[2][2]]
    k=max(range(4),key=candidates.__getitem__)
    q=[0.]*4; q[k]=math.sqrt(max(0,candidates[k]))/2; d=4*q[k]
    if k==3:
        q[:3]=[(m[2][1]-m[1][2])/d,(m[0][2]-m[2][0])/d,(m[1][0]-m[0][1])/d]
    else:
        i,j=(k+1)%3,(k+2)%3
        q[i],q[j]=(m[i][k]+m[k][i])/d,(m[j][k]+m[k][j])/d
        q[3]=(m[j][i]-m[i][j])/d
    return q


def compile_chunk(chunk, task, *, expected_chunk, first_frame, wire_epoch, max_frames=40, packet_bytes=False, wrap_root=False):
    if not isinstance(chunk,dict) or any(chunk.get(k)!=task.get(k) for k in ("op_id","session","epoch","prompt_version")):
        raise ValueError("stale_chunk_identity")
    if type(chunk.get("sequence")) is not int or chunk["sequence"]!=expected_chunk or chunk.get("fps")!=20:
        raise ValueError("invalid_chunk_order_or_rate")
    pose=chunk.get("pose")
    if not isinstance(pose,dict):
        raise ValueError("invalid_pose")
    rotations,roots,contacts=(pose.get(k) for k in ("local_rot_mats","root_positions","foot_contacts"))
    if not isinstance(rotations,list) or not 2<=len(rotations)<=max_frames or len(rotations)%2:
        raise ValueError("invalid_frame_count")
    if not isinstance(roots,list) or not isinstance(contacts,list) or len(roots)!=len(rotations) or len(contacts)!=len(rotations):
        raise ValueError("incomplete_pose")
    for contact in contacts:
        if not isinstance(contact,list) or len(contact)!=4 or any(type(v) not in (int,float,bool) or not math.isfinite(v) or not 0<=v<=1 for v in contact):
            raise ValueError('invalid_contacts')
    output=[]
    # 先校验整个分段再发送，坏的后半段不得在已经播放前半段后才被发现。
    for index,frame in enumerate(rotations):
        if not isinstance(frame,list) or len(frame)!=27:
            raise ValueError("invalid_joint_count")
        q=[matrix_quaternion(m) for m in frame]
        sequence=first_frame+index//2
        selected=contacts[index]
        if index%2==0:
            # 姿态降采样不能吞掉奇数帧的离地信号，否则下一次落脚会沿用旧锚点。
            # 两个模型帧都接触才允许线上100ms区间锁脚；包长和四位接触字段不变。
            selected=[min(a,b) for a,b in zip(selected,contacts[index+1])]
        packet=encode(task["session"],wire_epoch,sequence,((sequence-1)*100)%65536,roots[index],q,selected,wrap_root=wrap_root)
        if index%2==0:
            output.append(packet if packet_bytes else tuple(MidiEvent(*e) for e in midi_events(packet)))
    return output
