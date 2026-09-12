"""只对世界批准的拐角圆滑；按曲率限速，沿弧长连续加减速。"""
import bisect
import math


def curved_profile(plan):
    path=plan['path'];trims=plan['corner_trims'];limit=plan['max_speed']
    if limit<=0:raise ValueError('path_requires_positive_speed')
    acceleration=.75;turn_rate=math.radians(120)
    blocks=[];nodes=[]

    def append(p,heading,curvature=0.):
        if nodes and math.dist(nodes[-1][0],p)<1e-9:return
        if nodes:heading=nodes[-1][1]+(heading-nodes[-1][1]+math.pi)%(2*math.pi)-math.pi
        cap=min(limit,math.sqrt(.5/curvature),turn_rate/curvature) if curvature>1e-8 else limit
        nodes.append((list(p),heading,cap))

    def line(a,b):
        length=math.dist(a,b)
        if length<1e-9:return
        heading=math.atan2(b[0]-a[0],b[1]-a[1])
        steps=max(2,math.ceil(length/.025))
        for i in range(steps+1):append([x+(y-x)*i/steps for x,y in zip(a,b)],heading)

    current=path[0]
    for i in range(1,len(path)-1):
        a,b,c=path[i-1:i+2];trim=trims[i]
        if trim<=1e-9:
            line(current,b)
            if nodes:blocks.append(nodes)
            nodes=[];current=b
            continue
        before=math.dist(a,b);after=math.dist(b,c)
        if min(before,after)<1e-9:raise ValueError('invalid_corner_trims')
        entry=[y+(x-y)*trim/before for x,y in zip(a,b)]
        leave=[x+(y-x)*trim/after for x,y in zip(b,c)]
        line(current,entry)
        # 二次曲线在两端与原路线相切；64个弧长节点只在意图编译时使用。
        for j in range(65):
            u=j/64;v=1-u
            p=[v*v*x+2*v*u*y+u*u*z for x,y,z in zip(entry,b,leave)]
            d=[2*(v*(y-x)+u*(z-y)) for x,y,z in zip(entry,b,leave)]
            dd=[2*(z-2*y+x) for x,y,z in zip(entry,b,leave)]
            norm=math.hypot(*d)
            if norm<1e-8:raise ValueError('invalid_corner_trims')
            curvature=abs(d[0]*dd[1]-d[1]*dd[0])/norm**3
            append(p,math.atan2(d[0],d[1]),curvature)
        current=leave
    line(current,path[-1])
    if nodes:blocks.append(nodes)
    segments=[];clock=0.;heading=plan.get('start_heading',0.)
    for nodes in blocks:
        if len(nodes)<2:continue
        distances=[math.dist(a[0],b[0]) for a,b in zip(nodes,nodes[1:])]
        speeds=[n[2] for n in nodes];speeds[0]=speeds[-1]=0.
        for i,d in enumerate(distances):speeds[i+1]=min(speeds[i+1],math.sqrt(speeds[i]**2+2*acceleration*d))
        for i in range(len(distances)-1,-1,-1):speeds[i]=min(speeds[i],math.sqrt(speeds[i+1]**2+2*acceleration*distances[i]))
        times=[0.]
        for i,d in enumerate(distances):times.append(times[-1]+2*d/(speeds[i]+speeds[i+1]))
        delta=(nodes[0][1]-heading+math.pi)%(2*math.pi)-math.pi
        shift=heading+delta-nodes[0][1]
        nodes=[(p,h+shift,v) for p,h,v in nodes]
        turn=1.5*abs(delta)/turn_rate
        segments.append(dict(a=nodes[0][0],start=clock,heading=heading,delta=delta,turn=turn,
                             travel=times[-1],nodes=nodes,times=times,speeds=speeds,distances=distances))
        clock+=turn+times[-1];heading=nodes[-1][1]
    return segments,clock,heading


def sample_curve(segment,t):
    times=segment['times'];nodes=segment['nodes'];speeds=segment['speeds']
    i=min(len(times)-2,max(0,bisect.bisect_right(times,t)-1))
    dt=times[i+1]-times[i];elapsed=max(0,min(dt,t-times[i]))
    distance=speeds[i]*elapsed+.5*(speeds[i+1]-speeds[i])/dt*elapsed**2
    u=max(0,min(1,distance/segment['distances'][i]))
    a,h,_=nodes[i];b,j,_=nodes[i+1]
    return [x+(y-x)*u for x,y in zip(a,b)],h+(j-h)*u
