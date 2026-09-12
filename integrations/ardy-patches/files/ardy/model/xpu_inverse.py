"""XPU 骨骼转换：缓存固定拓扑并编译原有矩阵计算。"""
import einops
import torch
import threading
from ardy.geometry import cont6d_to_matrix

class _InverseCore(torch.nn.Module):
    def __init__(self,rep):
        super().__init__()
        skeleton=rep.skeleton
        parents=skeleton.joint_parents.detach().cpu().tolist()
        self.root=int(skeleton.root_idx)
        self.ps=rep.ps
        self.count=len(parents)
        levels=[]
        done={self.root}
        pending=set(range(self.count))-done
        while pending:
            level=sorted(i for i in pending if parents[i] in done)
            if not level:raise ValueError('骨架拓扑存在环或断开的根节点')
            levels.append(level);done.update(level);pending.difference_update(level)
        self.level_count=len(levels)
        device=skeleton.neutral_joints.device
        neutral=skeleton.neutral_joints.detach().clone()
        neutral=neutral-neutral[self.root]
        offsets=neutral.clone()
        for i,p in enumerate(parents):
            if i!=self.root:offsets[i]-=neutral[p]
        self.register_buffer('offsets',offsets)
        self.register_buffer('parents',skeleton.joint_parents.detach().clone())
        for n,level in enumerate(levels):
            self.register_buffer(f'indices_{n}',torch.tensor(level,device=device,dtype=torch.long))
            self.register_buffer(f'parents_{n}',torch.tensor([parents[i] for i in level],device=device,dtype=torch.long))

    def forward(self,features):
        root,heading,positions,rotation,velocity,contacts=einops.unpack(features,self.ps,'batch time *')
        global_rot=cont6d_to_matrix(rotation)
        batch,frames=features.shape[:2]
        matrices=global_rot.reshape(-1,self.count,3,3)
        parent_rot=matrices[:,self.parents].clone()
        parent_rot[:,self.root]=torch.eye(3,device=features.device,dtype=features.dtype)
        local=parent_rot.transpose(-1,-2)@matrices
        offsets=self.offsets.expand(matrices.shape[0],-1,-1)
        transforms=torch.cat((torch.nn.functional.pad(local,[0,0,0,1]),torch.nn.functional.pad(offsets.unsqueeze(-1),[0,0,0,1],value=1.)),dim=-1)
        chain=torch.zeros_like(transforms)
        chain[:,self.root]=transforms[:,self.root]
        for n in range(self.level_count):
            indices=getattr(self,f'indices_{n}')
            parents=getattr(self,f'parents_{n}')
            chain[:,indices]=chain[:,parents]@transforms[:,indices]
        posed=chain[:,:,:3,3].reshape(batch,frames,self.count,3)+root[:,:,None]
        return dict(local_rot_mats=local.reshape(batch,frames,self.count,3,3),global_rot_mats=global_rot,posed_joints=posed,root_positions=root,smooth_root_pos=root,foot_contacts=contacts>0.5,global_root_heading=heading)

def make_fast_inverse(rep):
    """创建可撤销实例包装；CPU、其他精度及位置解码继续调用原实现。"""
    original=rep.inverse
    cache={}
    lock=threading.RLock()
    def signature():
        values=(rep.skeleton.neutral_joints,rep.skeleton.joint_parents)
        if any(v.is_inference() for v in values):return None
        return tuple((id(v),v._version,str(v.device),str(v.dtype)) for v in values)+(int(rep.skeleton.root_idx),repr(rep.ps))
    def convert(features,is_normalized,posed_joints_from='rotations',return_numpy=False):
        if features.device.type!='xpu' or features.dtype!=torch.float32 or features.ndim not in (2,3) or posed_joints_from!='rotations':
            return original(features,is_normalized,posed_joints_from,return_numpy)
        key=signature()
        if key is None or torch.is_grad_enabled():return original(features,is_normalized,posed_joints_from,return_numpy)
        key=key+(str(features.device),)
        if cache.get('key')!=key:
            core=_InverseCore(rep).to(device=features.device,dtype=features.dtype).eval()
            cache.clear()
            cache.update(key=key,core=core,compiled=torch.compile(core,backend='inductor',dynamic=False,fullgraph=True))
        batched=features.ndim==3
        value=features if batched else features.unsqueeze(0)
        if is_normalized:value=rep.unnormalize(value)
        result=cache['compiled'](value)
        inverse.calls+=1
        if not batched:result={k:v.squeeze(0) for k,v in result.items()}
        if return_numpy:result={k:v.detach().cpu().numpy() for k,v in result.items()}
        return result
    def inverse(features,is_normalized,posed_joints_from='rotations',return_numpy=False):
        with lock:
            return convert(features,is_normalized,posed_joints_from,return_numpy)
    inverse.original=original
    inverse.cache=cache
    inverse.calls=0
    inverse.xpu_inverse_compiled=True
    return inverse

def enable_fast_inverse(rep):
    """只在兼容的 ARDY 表示上安装；调用方应处于 no_grad 推理上下文。"""
    from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
    if not isinstance(rep,ArdyMotionRep):return False
    if getattr(rep.inverse,'xpu_inverse_compiled',False):return True
    rep.inverse=make_fast_inverse(rep)
    return True
