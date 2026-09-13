"""临时 VMC 待机共享；保留世界回执、导航及故障恢复通道。"""
import math
import struct
import time

PARENTS = [-1,0,1,2,3,4,5,4,7,8,9,10,10,4,13,14,15,16,16,0,19,20,21,0,23,24,25]
BONES = dict(zip([0,1,4,5,6,7,8,9,10,13,14,15,16,19,20,21,23,24,25],
    ['Hips','Spine','Chest','Neck','Head','RightShoulder','RightUpperArm','RightLowerArm','RightHand',
     'LeftShoulder','LeftUpperArm','LeftLowerArm','LeftHand','RightUpperLeg','RightLowerLeg','RightFoot',
     'LeftUpperLeg','LeftLowerLeg','LeftFoot']))
SOURCE_PARENTS = {'Hips':None,'Spine':'Hips','Chest':'Spine','UpperChest':'Chest','Neck':'UpperChest','Head':'Neck'}
for side in ('Left','Right'):
    for bone,parent in [('Shoulder','UpperChest'),('UpperArm',side+'Shoulder'),('LowerArm',side+'UpperArm'),
                        ('Hand',side+'LowerArm'),('UpperLeg','Hips'),('LowerLeg',side+'UpperLeg'),('Foot',side+'LowerLeg')]:
        SOURCE_PARENTS[side+bone] = parent
IDENTITY = (0.,0.,0.,1.)

def normalize(q):
    if len(q)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in q):
        raise ValueError('invalid_quaternion')
    length=math.sqrt(sum(v*v for v in q))
    if length<1e-6: raise ValueError('zero_quaternion')
    return tuple(v/length for v in q)

def inverse(q): return (-q[0],-q[1],-q[2],q[3])

def multiply(a,b):
    x,y,z,w=a; X,Y,Z,W=b
    return normalize((w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z))

def matrix(q):
    x,y,z,w=normalize(q)
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]

def decode_osc(data,depth=0):
    """仅解码需要的 OSC 类型，并限制嵌套深度与数据报长度。"""
    if depth>8 or not data or len(data)>65507: raise ValueError('invalid_osc_size')
    if data.startswith(b'#bundle\0'):
        if len(data)<16: raise ValueError('short_bundle')
        offset=16; messages=[]
        while offset<len(data):
            size,=struct.unpack_from('>I',data,offset); offset+=4
            if not size or offset+size>len(data): raise ValueError('invalid_bundle_size')
            messages.extend(decode_osc(data[offset:offset+size],depth+1)); offset+=size
        return messages
    offset=0
    def string():
        nonlocal offset
        end=data.index(b'\0',offset); value=data[offset:end].decode('utf-8'); padded=(end+4)//4*4
        if padded>len(data) or any(data[end:padded]): raise ValueError('invalid_padding')
        offset=padded
        return value
    address,tags=string(),string()
    if not address.startswith('/') or not tags.startswith(','): raise ValueError('invalid_osc_header')
    args=[]
    for tag in tags[1:]:
        if tag=='s': args.append(string())
        elif tag in 'fi':
            value,=struct.unpack_from('>'+tag,data,offset); offset+=4
            if not math.isfinite(value): raise ValueError('nonfinite_osc')
            args.append(value)
        else: raise ValueError('unsupported_osc_type')
    if offset!=len(data): raise ValueError('trailing_osc')
    return [(address,args)]

class Capture:
    """宿主 T 消息为帧起点；每帧必须重新包含全部必需骨骼。"""
    def __init__(self):
        self.pending={}; self.started=False; self.rest=None; self.rest_height=None
        self.calibrate=False; self.frames=0

    def feed(self,address,args):
        if address=='/VMC/Ext/OK' and args==[0]:
            self.pending={}; self.started=False; self.rest=None
            return None
        if address=='/VMC/Ext/T':
            result=self.finish() if self.started else None
            self.pending={}; self.started=True
            return result
        if address=='/VMC/Ext/Bone/Pos' and self.started:
            if len(args)!=8 or args[0] not in SOURCE_PARENTS: return None
            if any(type(v) not in (float,int) or not math.isfinite(v) for v in args[1:]):
                raise ValueError('invalid_bone')
            self.pending[args[0]]=(args[1:4],normalize(args[4:]))
        return None

    def finish(self):
        if not set(BONES.values())<=self.pending.keys(): return None
        globals_={}
        def global_q(name):
            if name not in globals_:
                parent=SOURCE_PARENTS[name]
                if parent=='UpperChest' and parent not in self.pending: parent='Chest'
                globals_[name]=multiply(global_q(parent),self.pending[name][1]) if parent else self.pending[name][1]
            return globals_[name]
        for name in self.pending: global_q(name)
        height=self.pending['Hips'][0][1]
        if self.calibrate:
            if not .1<height<3: raise ValueError('invalid_rest_height')
            self.rest,self.rest_height=globals_,height; self.calibrate=False
        if self.rest is None: return None
        if set(globals_)!=set(self.rest):
            self.rest=None
            return None
        delta={name:multiply(globals_[name],inverse(self.rest[name])) for name in BONES.values()}
        self.frames+=1
        return dict(version=1,captured_at=time.time(),rotations=delta,
                    height_offset=max(-.3,min(.3,(height-self.rest_height)*.9544/self.rest_height)),frames=self.frames)

def pose_from_sample(sample):
    """把已校准的宿主姿态转换成四帧 Core27 数据，不读取 ARDY 或临时文件。"""
    if sample.get('version') != 1 or not 0 <= time.time()-sample['captured_at'] <= .5:
        raise ValueError('vmc_source_stale')
    globals_=[]; local=[]
    for index,parent in enumerate(PARENTS):
        if index in BONES:
            x,y,z,w=normalize(sample['rotations'][BONES[index]])
            q=(x,-y,-z,w)  # 与世界端 X 镜像互逆。
        else: q=globals_[parent]
        globals_.append(q)
        local.append(matrix(multiply(inverse(globals_[parent]),q) if parent>=0 else q))
    height=sample['height_offset']
    if type(height) not in (float,int) or not math.isfinite(height) or abs(height)>.3:
        raise ValueError('invalid_vmc_height')
    return dict(local_rot_mats=[local for _ in range(4)],root_positions=[[0,.9544+height,0] for _ in range(4)],
                foot_contacts=[[0,0,0,0] for _ in range(4)])
