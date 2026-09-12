"""将世界验证过的平地路径转换成 Core27 坐标，绝不从语义目录推测路径。"""
import math


def _number(value,low,high):
    if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError("invalid_motion_space_number")
    return float(value)


class MotionSpace:
    def __init__(self, *, origin, yaw, scale, max_distance=5, surface_path=False):
        if not isinstance(origin,(tuple,list)) or len(origin)!=3:
            raise ValueError("invalid_motion_origin")
        self.origin=tuple(_number(v,-10000,10000) for v in origin)
        angle=math.radians(_number(yaw,-360,360))
        self.cos,self.sin=math.cos(angle),math.sin(angle)
        self.scale=_number(scale,.2,3)
        self.max_distance=_number(max_distance,.1,10000)
        self.surface_path=surface_path

    def source_point(self,point):
        if not isinstance(point,(tuple,list)) or len(point)!=3:
            raise ValueError("invalid_path_point")
        x,y,z=(_number(v,-10000,10000)-o for v,o in zip(point,self.origin))
        # 当前执行器只验证平地；台阶不能投影到平面后冒充可执行路径。
        if not self.surface_path and abs(y)>.08:
            raise ValueError("unsupported_path_height")
        if math.hypot(x,z)>self.max_distance:
            raise ValueError("path_out_of_range")
        return [-(self.cos*x-self.sin*z)/self.scale,
                (self.sin*x+self.cos*z)/self.scale]

    def world_point(self,point):
        if not isinstance(point,(tuple,list)) or len(point)!=2:
            raise ValueError("invalid_source_point")
        limit=max(32.,self.max_distance/self.scale)
        x=-_number(point[0],-limit,limit)*self.scale
        z=_number(point[1],-limit,limit)*self.scale
        return [self.origin[0]+self.cos*x+self.sin*z,self.origin[1],
                self.origin[2]-self.sin*x+self.cos*z]

    def verified_paths(self,paths, *, max_speed):
        if not isinstance(paths,dict) or len(paths)>32:
            raise ValueError("invalid_verified_paths")
        result={}
        for key,path in paths.items():
            if not isinstance(key,str) or not 1<=len(key)<=64 or not isinstance(path,list) or not 1<=len(path)<=64:
                raise ValueError("invalid_verified_path")
            result[key]=[self.source_point(point) for point in path]
        speed=_number(max_speed,0,2)/self.scale
        if speed>3:
            raise ValueError("source_speed_out_of_range")
        return dict(root=[0.,0.],paths=result,max_speed=speed)
