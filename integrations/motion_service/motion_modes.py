"""把已验证的世界规划转换为模型约束；不接受模型猜测的场景可达性。"""
import math
from copy import deepcopy

BASIC_MODES = frozenset({'idle','free_action','path','velocity','keyframes'})

def point(value,dimensions=2):
    if not isinstance(value,(list,tuple)) or len(value)!=dimensions or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):
        raise ValueError('invalid_motion_point')
    return list(value)

def validate_plan(plan, *, frames=None):
    if not isinstance(plan,dict) or plan.get('mode','idle') not in BASIC_MODES:
        raise ValueError('unsupported_motion_mode')
    out=deepcopy(plan)
    out['mode']=out.get('mode','idle')
    out['origin_xz']=point(out.get('origin_xz'))
    speed=out.get('max_speed',.5)
    if type(speed) not in (int,float) or not math.isfinite(speed) or not 0<=speed<=2:
        raise ValueError('invalid_motion_speed')
    out['max_speed']=float(speed)
    if any(key in out and (type(out[key]) not in (int,float) or not math.isfinite(out[key])) for key in ('heading','start_heading')):
        raise ValueError('invalid_motion_heading')
    if out['mode'] in {'path','velocity'}:
        path=out.get('path')
        if not isinstance(path,list) or not 1<=len(path)<=64:
            raise ValueError('verified_path_required')
        out['path']=[point(p) for p in path]
        if 'corner_trims' in out:
            trims=out['corner_trims']
            if (not isinstance(trims,list) or len(trims)!=len(path)
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in trims)
                or trims[0]!=0 or trims[-1]!=0):raise ValueError('invalid_corner_trims')
            for i in range(1,len(path)-1):
                if trims[i]>.45*min(math.dist(path[i-1],path[i]),math.dist(path[i],path[i+1]))+1e-6:
                    raise ValueError('invalid_corner_trims')
        if 'surface_path_xyz' in out:
            surface=out['surface_path_xyz']
            if not isinstance(surface,list) or len(surface)!=len(path):raise ValueError('invalid_surface_path')
            out['surface_path_xyz']=[point(p,3) for p in surface]
        if math.dist(out['path'][0],out['origin_xz'])>.05:
            raise ValueError('path_origin_mismatch')
    keyframes=out.get('keyframes',[])
    if not isinstance(keyframes,list) or len(keyframes)>64:
        raise ValueError('invalid_keyframes')
    seen=set()
    for item in keyframes:
        if item.get('kind') not in {'fullbody','end_effector'} or type(item.get('frame')) is not int or item['frame']<0:
            raise ValueError('invalid_keyframe')
        positions=item.get('positions');rotations=item.get('rotations')
        if not isinstance(positions,list) or len(positions)!=27 or not isinstance(rotations,list) or len(rotations)!=27:
            raise ValueError('core27_keyframe_required')
        for pos in positions:point(pos,3)
        for matrix in rotations:
            if len(matrix)!=3:raise ValueError('invalid_keyframe_rotation')
            for row in matrix:point(row,3)
            determinant=sum(matrix[0][i]*(matrix[1][(i+1)%3]*matrix[2][(i+2)%3]-matrix[1][(i+2)%3]*matrix[2][(i+1)%3]) for i in range(3))
            if abs(determinant-1)>.03:raise ValueError('invalid_keyframe_rotation')
            for i in range(3):
                for j in range(3):
                    if abs(sum(matrix[k][i]*matrix[k][j] for k in range(3))-(1 if i==j else 0))>.03:
                        raise ValueError('invalid_keyframe_rotation')
        joints=item.get('joints') if item['kind']=='end_effector' else ['*']
        if not isinstance(joints,list) or not joints or any(not isinstance(j,str) or not j for j in joints):
            raise ValueError('invalid_constraint_joints')
        for joint in joints:
            identity=(item['frame'],joint)
            if identity in seen or (item['frame'],'*') in seen or (joint=='*' and any(f==item['frame'] for f,j in seen)):
                raise ValueError('conflicting_keyframes')
            seen.add(identity)
    return out

def path_profile(plan):
    """沿验证路径先转向再起步；每段使用有界加减速，拐角不硬改朝向。"""
    if any(plan.get('corner_trims',[])):
        from .curved_path import curved_profile
        return curved_profile(plan)
    path=plan['path'];heading=plan.get('start_heading',0.)
    speed=plan['max_speed'];acceleration=.75;turn_rate=math.radians(120)
    segments=[];clock=0.
    for a,b in zip(path,path[1:]):
        length=math.dist(a,b)
        if length<1e-6:continue
        if speed<=0:raise ValueError('path_requires_positive_speed')
        wanted=math.atan2(b[0]-a[0],b[1]-a[1])
        delta=(wanted-heading+math.pi)%(2*math.pi)-math.pi
        # 三次平滑转向的峰值速度是平均值的1.5倍，延长时长以保持原角速度上限。
        turn=1.5*abs(delta)/turn_rate
        peak=min(speed,math.sqrt(length*acceleration))
        ramp=peak/acceleration;cruise=max(0.,length/peak-ramp)
        segments.append(dict(a=a,b=b,length=length,start=clock,heading=heading,delta=delta,
            turn=turn,peak=peak,ramp=ramp,cruise=cruise,travel=2*ramp+cruise))
        clock+=turn+2*ramp+cruise;heading+=delta
    return segments,clock,heading


def sample_profile(plan,profile,seconds):
    segments,total,final_heading=profile
    for seg in segments:
        t=seconds-seg['start']
        if t<seg['turn']:
            u=max(0,t)/max(seg['turn'],1e-6)
            return list(seg['a']),seg['heading']+seg['delta']*u*u*(3-2*u)
        t-=seg['turn']
        if t<=seg['travel']:
            if 'nodes' in seg:
                from .curved_path import sample_curve
                return sample_curve(seg,t)
            peak,ramp,cruise=seg['peak'],seg['ramp'],seg['cruise']
            if t<ramp:distance=.5*peak/ramp*t*t
            elif t<ramp+cruise:distance=.5*peak*ramp+peak*(t-ramp)
            else:distance=seg['length']-.5*peak/ramp*(seg['travel']-t)**2
            fraction=max(0.,min(1.,distance/seg['length']))
            return [a+(b-a)*fraction for a,b in zip(seg['a'],seg['b'])],seg['heading']+seg['delta']
    return list(plan['path'][-1]),final_heading


def compile_plan(intent,first_frame,frames,fps):
    plan=validate_plan(intent.constraints)
    mode=plan['mode'];origin=plan['origin_xz']
    result={'origin_xz':origin,'movement':mode}
    if mode=='idle':
        result['root_xz']=[origin.copy() for _ in range(frames)]
        result['heading']=[plan.get('heading',0.)]*frames
    elif mode in {'path','velocity'}:
        profile=path_profile(plan)
        values=[sample_profile(plan,profile,max(0,first_frame-intent.start_frame+i+1)/fps) for i in range(frames)]
        result['root_xz']=[p for p,h in values]
        result['heading']=[h for p,h in values]
        # 每500ms提供路径目标，保留中间帧的跨步空间；转弯连续性由验证曲线和限速负责。
        stride=max(1,round(fps*.5))
        result['root_indices']=sorted(set(list(range(stride-1,frames,stride))+[frames-1]))
    # 自由动作不强行固定每一帧根位置，世界执行器按协商能力验证导航区域和碰撞，旧世界仍可采用半径限制。
    result['keyframes']=[]
    for item in plan.get('keyframes',[]):
        absolute=intent.start_frame+item['frame']
        if first_frame<=absolute<first_frame+frames:
            selected=deepcopy(item);selected['frame']=absolute-first_frame
            result['keyframes'].append(selected)
    return result
