using UdonSharp;
using UnityEngine;
using UnityEngine.AI;

// 持续运动会话的控制帧与语义边界；与有限任务共享唯一 MIDI 接收和骨骼执行器。
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoArdyStreamBridge : UdonSharpBehaviour
{
    // CRC16-CCITT每字节两次固定查表，结果与原逐位算法完全一致。
    private int[] _crcNibbles={0x0000,0x1021,0x2042,0x3063,0x4084,0x50A5,0x60C6,0x70E7,0x8108,0x9129,0xA14A,0xB16B,0xC18C,0xD1AD,0xE1CE,0xF1EF};
    public NekoArdyWorldBridge world;
    public bool enableContinuous;
    [HideInInspector] public bool active;
    [HideInInspector] public bool surfaceStepAllowed;
    [HideInInspector] public Vector3 validatedRoot;
    private byte[] _packet=new byte[133];
    private int _count,_bits,_pending;
    private bool _receiving;
    private int _session,_epoch,_request,_blockedSession;
    private string _streamId,_opId;
    private float _preparedUntil;
    private Vector3 _origin,_intentOrigin;
    private int _intentVersion,_endSequence,_mode,_pathRequest;
    private Vector3[] _preparedPath;
    private Vector3[] _activePath;
    private Vector3[] _blockedPoints=new Vector3[4];
    private float[] _blockedUntil=new float[4];
    private int _blockedNext;

    public void RememberBlockedPoint(Vector3 point)
    {
        // 世界侧短期记忆跨运动会话保留；只在实际支撑/碰撞失败时写入。
        // 胶囊撞墙的接触点可能在腰部，避让属于当前脚底所在楼层，不能拿腰高去采样NavMesh。
        point.y=world.rig.motionRoot.position.y;
        _blockedPoints[_blockedNext]=point;
        _blockedUntil[_blockedNext]=Time.realtimeSinceStartup+30f;
        _blockedNext=(_blockedNext+1)%4;
    }

    public bool NearBlockedPoint(Vector3 point)
    {
        for(int i=0;i<4;i++)if(_blockedUntil[i]>Time.realtimeSinceStartup && (point-_blockedPoints[i]).sqrMagnitude<1f)return true;
        return false;
    }

    private bool AvoidsBlocked(Vector3[] points)
    {
        for(int h=0;h<4;h++) {
            if(_blockedUntil[h]<=Time.realtimeSinceStartup)continue;
            for(int i=0;i<points.Length-1;i++) {
                Vector3 a=points[i],b=points[i+1],d=b-a,offset=_blockedPoints[h]-a;
                d.y=0;offset.y=0;
                float u=d.sqrMagnitude>.0001f?Vector3.Dot(offset,d)/d.sqrMagnitude:0;
                // 起点是最后一次验证成功的位置，允许沿远离失败点的方向退出邻域。
                if(i==0 && u<=0)continue;
                // 允许从外侧接近危险圈旁的安全终点；实际支撑和胶囊净空仍必须通过。
                if(i==points.Length-2 && u>=1 && offset.sqrMagnitude>.0001f) {
                    Vector3 remaining=b-_blockedPoints[h];remaining.y=0;
                    if(remaining.sqrMagnitude>=.1225f)continue;
                }
                Vector3 near=Vector3.Lerp(a,b,Mathf.Clamp01(u));
                if(Mathf.Abs(near.y-_blockedPoints[h].y)>1f)continue;
                near.y=_blockedPoints[h].y;
                if(Vector3.Distance(near,_blockedPoints[h])<.7f)return false;
            }
        }
        return true;
    }

    private void RefreshBlockedPoints()
    {
        // 仅重规划时复核最多四个故障点；地面恢复或障碍消失后，不继续绕行已不存在的危险。
        for(int h=0;h<4;h++) {
            if(_blockedUntil[h]<=Time.realtimeSinceStartup)continue;
            Vector3 point=_blockedPoints[h];bool supported=true;float ground=point.y;
            for(int i=0;i<5;i++) {
                Vector3 offset=i==0?Vector3.zero:i==1?Vector3.right*.12f:i==2?Vector3.left*.12f:i==3?Vector3.forward*.12f:Vector3.back*.12f;
                RaycastHit floor;
                if(!Physics.Raycast(point+offset+Vector3.up*.3f,Vector3.down,out floor,.45f,world.rig.environmentMask.value,QueryTriggerInteraction.Ignore)) { supported=false;break; }
                float slope=floor.collider==world.rig.stairRampCollider?.819152f:.9f;
                if(floor.normal.y<slope || Mathf.Abs(floor.point.y-point.y)>.08f) { supported=false;break; }
                ground=Mathf.Max(ground,floor.point.y);
            }
            if(!supported)continue;
            float radius=.18f*world.rig.BindHeight(0)/.9544f;
            RaycastHit support;Vector3 supportStart=point+Vector3.up*(radius+.3f);
            if(!Physics.SphereCast(supportStart,radius,Vector3.down,out support,.6f,world.rig.environmentMask.value,QueryTriggerInteraction.Ignore))continue;
            ground=Mathf.Max(ground,supportStart.y-support.distance-radius);
            Vector3 bottom=new Vector3(point.x,ground+radius+.025f,point.z);
            if(!Physics.CheckCapsule(bottom,bottom+Vector3.up*Mathf.Max(.1f,1.55f*world.rig.BindHeight(0)/.9544f-2*radius),radius,world.rig.environmentMask.value,QueryTriggerInteraction.Ignore))
                _blockedUntil[h]=0;
        }
    }

    private bool DetourHasSupport(Vector3[] points)
    {
        for(int i=0;i<points.Length-1;i++) {
            Vector3 a=points[i],b=points[i+1],direction=b-a;direction.y=0;
            Vector3 side=Vector3.Cross(Vector3.up,direction.normalized)*.25f;
            int count=Mathf.Min(128,Mathf.CeilToInt(direction.magnitude/.25f));
            for(int j=1;j<=count;j++) {
                Vector3 p=Vector3.Lerp(a,b,(float)j/count);bool nearby=false;
                for(int h=0;h<4;h++) {
                    Vector3 delta=p-_blockedPoints[h];delta.y=0;
                    if(_blockedUntil[h]>Time.realtimeSinceStartup && delta.sqrMagnitude<4f)nearby=true;
                }
                if(!nearby)continue;
                NavMeshHit nav;
                if(!NavMesh.SamplePosition(p,out nav,.65f,NavMesh.AllAreas))return false;
                p=nav.position;
                for(int foot=-1;foot<=1;foot++) {
                    RaycastHit floor;
                    if(!Physics.Raycast(p+side*foot+Vector3.up*.35f,Vector3.down,out floor,.9f,world.rig.environmentMask.value,QueryTriggerInteraction.Ignore))return false;
                    float limit=floor.collider==world.rig.stairRampCollider?.819152f:.9f;
                    if(floor.normal.y<limit)return false;
                    if(foot==0) {
                        float radius=.18f*world.rig.BindHeight(0)/.9544f;
                        // 与最终执行器一致：地毯边缘仍承重时，身体不能按中心射线落到低层地面。
                        RaycastHit support;Vector3 supportStart=p+Vector3.up*(radius+.3f);
                        if(!Physics.SphereCast(supportStart,radius,Vector3.down,out support,.6f,world.rig.environmentMask.value,QueryTriggerInteraction.Ignore))return false;
                        float ground=Mathf.Max(floor.point.y,supportStart.y-support.distance-radius);
                        Vector3 bottom=new Vector3(p.x,ground+radius+.025f,p.z);
                        if(Physics.CheckCapsule(bottom,bottom+Vector3.up*Mathf.Max(.1f,1.55f*world.rig.BindHeight(0)/.9544f-2*radius),radius,world.rig.environmentMask.value,QueryTriggerInteraction.Ignore))return false;
                    }
                }
            }
        }
        return true;
    }

    private Vector3[] FindDetour(Vector3 from,Vector3 to)
    {
        Vector3 hazard=_blockedPoints[(_blockedNext+3)%4];
        // 已停在禁入圈内时，单个拐点常会让第二段再次穿过障碍；先退出，再沿侧边绕过。
        Vector3 forward=to-from;forward.y=0;forward.Normalize();
        Vector3 lateral=Vector3.Cross(Vector3.up,forward);
        for(int ring=0;ring<4;ring++)for(int side=-1;side<=1;side+=2) {
            float clearance=ring==0?.9f:ring==1?1.4f:ring==2?2.2f:3.4f;
            Vector3 entry=from-forward*.35f+lateral*(side*clearance);
            Vector3 exit=hazard+forward*clearance+lateral*(side*clearance);
            NavMeshHit entryNav,exitNav;
            if(!NavMesh.SamplePosition(entry,out entryNav,.35f,NavMesh.AllAreas)
                || !NavMesh.SamplePosition(exit,out exitNav,.35f,NavMesh.AllAreas)
                || !world.router.IsPointInsideActivityBounds(entryNav.position)
                || !world.router.IsPointInsideActivityBounds(exitNav.position))continue;
            var begin=new NavMeshPath();var around=new NavMeshPath();var end=new NavMeshPath();
            if(!NavMesh.CalculatePath(from,entryNav.position,NavMesh.AllAreas,begin) || begin.status!=NavMeshPathStatus.PathComplete
                || !NavMesh.CalculatePath(entryNav.position,exitNav.position,NavMesh.AllAreas,around) || around.status!=NavMeshPathStatus.PathComplete
                || !NavMesh.CalculatePath(exitNav.position,to,NavMesh.AllAreas,end) || end.status!=NavMeshPathStatus.PathComplete)continue;
            Vector3[] a=begin.corners,b=around.corners,c=end.corners;
            int total=a.Length+b.Length+c.Length-2;
            if(a.Length<2 || b.Length<2 || c.Length<2 || total>12)continue;
            Vector3[] bypass=new Vector3[total];int cursor=0;
            for(int k=0;k<a.Length;k++)bypass[cursor++]=a[k];
            for(int k=1;k<b.Length;k++)bypass[cursor++]=b[k];
            for(int k=1;k<c.Length;k++)bypass[cursor++]=c[k];
            if(AvoidsBlocked(bypass) && DetourHasSupport(bypass))return bypass;
        }
        // 有界候选只在失败后寻路时计算，不进入每帧执行或MIDI数据路径。
        for(int ring=0;ring<4;ring++)for(int i=0;i<8;i++) {
            float angle=i*Mathf.PI*.25f,radius=ring==0?.85f:ring==1?1.4f:ring==2?2.2f:3.4f;
            Vector3 candidate=hazard+new Vector3(Mathf.Cos(angle)*radius,0,Mathf.Sin(angle)*radius);
            NavMeshHit nav;
            if(!NavMesh.SamplePosition(candidate,out nav,.5f,NavMesh.AllAreas) || !world.router.IsPointInsideActivityBounds(nav.position))continue;
            var first=new NavMeshPath();var last=new NavMeshPath();
            if(!NavMesh.CalculatePath(from,nav.position,NavMesh.AllAreas,first) || first.status!=NavMeshPathStatus.PathComplete
                || !NavMesh.CalculatePath(nav.position,to,NavMesh.AllAreas,last) || last.status!=NavMeshPathStatus.PathComplete)continue;
            Vector3[] left=first.corners,right=last.corners;
            if(left.Length<1 || right.Length<1 || left.Length+right.Length-1>12)continue;
            Vector3[] joined=new Vector3[left.Length+right.Length-1];
            for(int k=0;k<left.Length;k++)joined[k]=left[k];
            for(int k=1;k<right.Length;k++)joined[left.Length+k-1]=right[k];
            if(AvoidsBlocked(joined) && DetourHasSupport(joined))return joined;
        }
        return null;
    }
    private Transform _preparedTarget,_activeTarget;
    private Vector3 _preparedTargetPosition,_activeTargetPosition;
    private Transform[] _targets=new Transform[4];
    private Vector3[] _targetPositions=new Vector3[4];
    private bool _intentEnded;
    private string[] _progress=new string[8];
    private int _progressHead,_progressCount;
    private float _lastPoseAck;
    private int _acceptedVersion,_incomingEnd,_intentHead,_intentCount;
    private string[] _intentIds=new string[4];
    private int[] _versions=new int[4],_starts=new int[4],_ends=new int[4],_modes=new int[4],_pathCounts=new int[4];
    private Vector3[] _paths=new Vector3[48];

    public bool Ready()
    {
        return enableContinuous && enabled && gameObject.activeInHierarchy && world!=null && world.Ready()
            && world.streamBridge==this && world.receiver.streamBridge==this && (_blockedSession<=0 || _blockedSession!=world.router.GetSession());
    }

    public void ReceiveControl(int channel,int number,int value)
    {
        if(channel!=15 || !Ready()) return;
        // 通信故障只撤销本次运动授权，不设置人工急停，也不允许复用旧帧。
        if(number==113 && value==1) {
            _receiving=false;_preparedUntil=0;
            if(active)world.router.CancelExternalPose("transport_fault");
            return;
        }
        if(number==112) { if(active) world.router.CancelExternalPose("explicit_stop"); _blockedSession=world.router.GetSession(); return; }
        if(number==110 && value==1) { _count=_bits=_pending=0; _receiving=true; }
        if(number==111 && value==1 && _receiving) { _receiving=false; Commit(); }
    }

    public void ReceiveData(int channel,int number,int velocity)
    {
        if(channel!=15 || !_receiving || number<0 || number>127 || velocity<0 || velocity>127) return;
        _pending|=((number<<7)|velocity)<<_bits;_bits+=14;
        while(_bits>=8) {
            if(_count>=133) { _receiving=false;return; }
            _packet[_count++]=(byte)(_pending&255);_pending>>=8;_bits-=8;
        }
    }

    private int IntAt(int offset) { return _packet[offset]|(_packet[offset+1]<<8)|(_packet[offset+2]<<16)|(_packet[offset+3]<<24); }
    private string Identity()
    {
        return "\"stream_id\":"+world.router.telemetry.J(_streamId)+",\"pose_epoch\":"+_epoch;
    }
    private void Event(string kind,string body)
    {
        world.router.telemetry.EmitForced(kind,Identity()+",\"request_seq\":"+_request+(body==""?"":","+body));
    }

    private void Commit()
    {
        if(_count!=133 || _bits!=0 || _packet[0]!=65 || _packet[1]!=80 || _packet[2]<6 || _packet[2]>10) return;
        int crc=65535;
        for(int i=0;i<131;i++) {
            crc^=_packet[i]<<8;
            crc=((crc<<4)^_crcNibbles[(crc>>12)&15])&65535;
            crc=((crc<<4)^_crcNibbles[(crc>>12)&15])&65535;
        }
        if(crc!=(_packet[131]|(_packet[132]<<8)) || IntAt(3)!=world.router.GetSession() || !world.router.HasLocalDriverAuthority()) return;
        for(int i=67;i<131;i++) if(_packet[i]!=0)return;
        if(_packet[21]!=0 || _packet[22]!=0 || IntAt(11)<=0) return;
        string op="";
        for(int i=23;i<55;i++) { int c=_packet[i];if(!((c>=48&&c<=57)||(c>=97&&c<=102)))return;op+=((char)c).ToString(); }
        int command=_packet[2];
        if(command==6) {
            if(active || world.router.GetControlState()!=NekoMidiRouter.STATE_EXTERNAL || world.router.ActiveOpsJson()!="[]" || op==_streamId) return;
            _streamId=op;_session=world.router.GetSession();_epoch=world.AllocatePoseEpoch();_request=IntAt(11);
            _origin=world.rig.motionRoot.position;world.router.telemetry.motionStreamActive=true;_preparedUntil=Time.realtimeSinceStartup+5f;
            Event("npc.stream_prepared","\"protocol\":\"neko-pose/2\",\"origin\":"+world.router.telemetry.Vec3(_origin)
                +",\"yaw\":"+world.router.telemetry.F2(world.rig.motionRoot.eulerAngles.y)+",\"scale\":"+world.rig.MotionScale().ToString("F6"));
            return;
        }
        if(_session!=world.router.GetSession() || IntAt(7)!=_epoch || IntAt(11)<=_request)return;
        _request=IntAt(11);
        if(command==7) {
            if(active || op!=_streamId || Time.realtimeSinceStartup>=_preparedUntil || Vector3.Distance(world.rig.motionRoot.position,_origin)>.02f)return;
            if(!world.router.BeginMotionStream(_streamId,world))return;
            active=true;_intentVersion=_acceptedVersion=_intentHead=_intentCount=0;_opId=null;
            _progressHead=_progressCount=0;_lastPoseAck=Time.realtimeSinceStartup;
            world.receiver.operationId=_streamId;world.receiver.expectedSession=_session;world.receiver.expectedEpoch=_epoch;
            world.receiver.ArmLab();world.receiver.ResetStreamClock();
            Event("npc.stream_armed","\"protocol\":\"neko-pose/2\"");return;
        }
        if(!active || !world.router.TouchExternalPose(_streamId))return;
        if(command==8) {
            int first=IntAt(17),version=IntAt(55),end=IntAt(59),pathRequest=IntAt(63);
            int mode=_packet[15];
            if((first<=world.receiver.StreamCommittedFrame()/2 || first>world.receiver.streamReceivedSequence+1) || version<=_acceptedVersion || _intentCount>=4 || (end!=0&&end<first) || mode>4)return;
            if((mode==2 || mode==3) && (pathRequest!=_pathRequest || _preparedPath==null))return;
            // 仅丢弃200ms锁定窗口之外的预收帧；更换线代次防止旧包复活。
            world.receiver.TrimStreamFuture(first);
            while(_intentCount>0 && _starts[(_intentHead+_intentCount-1)%4]>=first) _intentCount--;
            int wireEpoch=world.AllocatePoseEpoch();
            world.receiver.expectedEpoch=wireEpoch;
            int slot=(_intentHead+_intentCount)%4;
            _intentIds[slot]=op;_versions[slot]=version;_starts[slot]=first;_ends[slot]=end;_modes[slot]=mode;
            _pathCounts[slot]=(mode==2||mode==3)?_preparedPath.Length:0;
            _targets[slot]=_pathCounts[slot]>0?_preparedTarget:null;
            _targetPositions[slot]=_preparedTargetPosition;
            for(int i=0;i<_pathCounts[slot];i++) _paths[slot*12+i]=_preparedPath[i];
            _intentCount++;_acceptedVersion=version;_incomingEnd=end;
            Event("npc.stream_control","\"op_id\":"+world.router.telemetry.J(op)+",\"version\":"+version+",\"wire_epoch\":"+wireEpoch);return;
        }
        if(command==9 && op==_streamId) {
            world.router.EndMotionStream();
            return;
        }
        if(command==10 && op==_streamId) {
            int anchor=_packet[16];
            if(world.router.anchorTransforms==null || anchor>=world.router.anchorTransforms.Length || world.router.anchorTransforms[anchor]==null){PathFailed("target_missing");return;}
            Vector3 from=world.rig.motionRoot.position,to=world.router.anchorTransforms[anchor].position;
            var path=new NavMeshPath();
            if(!NavMesh.CalculatePath(from,to,NavMesh.AllAreas,path) || path.status!=NavMeshPathStatus.PathComplete){PathFailed("path_incomplete");return;}
            Vector3[] corners=path.corners;
            RefreshBlockedPoints();
            bool detour=!AvoidsBlocked(corners);
            if(detour) {
                corners=FindDetour(from,to);
                if(corners==null){PathFailed("safe_detour_unavailable");return;}
            }
            if(corners.Length==0 || corners.Length>12){PathFailed("path_corner_limit");return;}
            // 最短路径常贴着导航网格边缘，直接向内圆滑会切入障碍；先为转弯留出外侧余量。
            for(int i=1;i<corners.Length-1;i++) {
                if(detour)continue;
                Vector3 incoming=corners[i]-corners[i-1],outgoing=corners[i+1]-corners[i];
                incoming.y=0;outgoing.y=0;
                if(incoming.magnitude<.1f || outgoing.magnitude<.1f)continue;
                Vector3 outward=incoming.normalized-outgoing.normalized;
                if(outward.sqrMagnitude<.01f)continue;
                Vector3 candidate=corners[i]+outward.normalized*Mathf.Min(.16f,.2f*Mathf.Min(incoming.magnitude,outgoing.magnitude));
                NavMeshHit hit;
                if(world.router.IsPointInsideActivityBounds(candidate)
                    && NavMesh.SamplePosition(candidate,out hit,.04f,NavMesh.AllAreas)
                    && !NavMesh.Raycast(corners[i-1],candidate,out hit,NavMesh.AllAreas)
                    && !NavMesh.Raycast(candidate,corners[i+1],out hit,NavMesh.AllAreas))corners[i]=candidate;
            }
            float[] trims=new float[corners.Length];
            // 每次路径请求只做一次有界拐角检查；不增加逐帧报文或寻路。
            for(int i=1;i<corners.Length-1;i++) {
                if(detour)continue;
                Vector3 incoming=corners[i]-corners[i-1],outgoing=corners[i+1]-corners[i];
                incoming.y=0;outgoing.y=0;
                float before=incoming.magnitude,after=outgoing.magnitude;
                if(before<.05f || after<.05f || Vector3.Dot(incoming.normalized,outgoing.normalized)<-.5f)continue;
                float trim=Mathf.Min(.4f,.44f*Mathf.Min(before,after));
                for(int attempt=0;attempt<3;attempt++) {
                    Vector3 entry=corners[i]-incoming.normalized*trim,leave=corners[i]+outgoing.normalized*trim;
                    Vector3 previous=entry;bool valid=true;
                    for(int sample=1;sample<=8;sample++) {
                        float u=sample/8f,v=1f-u;
                        Vector3 next=v*v*entry+2f*v*u*corners[i]+u*u*leave;
                        NavMeshHit hit;
                        if(!world.router.IsPointInsideActivityBounds(next)
                            || !NavMesh.SamplePosition(next,out hit,.04f,NavMesh.AllAreas)
                            || NavMesh.Raycast(previous,next,out hit,NavMesh.AllAreas)) {valid=false;break;}
                        previous=next;
                    }
                    if(valid) {trims[i]=trim;break;}
                    trim*=.5f;
                }
            }
            // NavMesh带烘焙高度偏差，以碰撞地面核对平面路径，不能直接拿烘焙Y当脚底高度。
            for(int i=0;i<corners.Length;i++) {
                RaycastHit floor;
                if(!Physics.Raycast(corners[i]+Vector3.up*.3f,Vector3.down,out floor,.6f,world.rig.environmentMask.value,QueryTriggerInteraction.Ignore))
                {PathFailed("path_ground_missing");return;}
                if(floor.normal.y<(floor.collider==world.rig.stairRampCollider?.819152f:.9f)){PathFailed("path_slope");return;}
                corners[i].y=floor.point.y;
                // 保留三维路径；局部高度过渡与碰撞由最终执行器逐帧检查。
                if(!world.router.IsPointInsideActivityBounds(corners[i])){PathFailed("path_outside_bounds");return;}
            }
            _preparedTarget=world.router.anchorTransforms[anchor];_preparedTargetPosition=_preparedTarget.position;
            _preparedPath=corners;_pathRequest=_request;
            string points="[";
            for(int i=0;i<corners.Length;i++)points+=(i==0?"":",")+world.router.telemetry.Vec3(corners[i]);
            string rounding="[";
            for(int i=0;i<trims.Length;i++)rounding+=(i==0?"":",")+trims[i].ToString("F4",System.Globalization.CultureInfo.InvariantCulture);
            Event("npc.stream_path","\"status\":\"accepted\",\"surface_path\":true,\"points\":"+points+"],\"corner_trims\":"+rounding+"]");
        }
    }

    private void PathFailed(string reason)
    {
        // 复用本次路径回执确认控制包；拒绝不能变成MIDI信用超时。
        Event("npc.stream_path","\"status\":\"failed\",\"reason\":"+world.router.telemetry.J(reason));
    }

    public int StreamEpoch() { return _epoch; }

    public bool AcceptSequence(int sequence)
    {
        return active && _acceptedVersion>0 && sequence==world.receiver.streamReceivedSequence+1
            && (_incomingEnd==0 || sequence<=_incomingEnd);
    }

    public bool BeforeApply(int sequence)
    {
        if(!active)return false;
        if(_intentCount>0 && sequence>=_starts[_intentHead]) {
            int slot=_intentHead;
            if(sequence!=_starts[slot] || !world.router.BeginStreamTask(_intentIds[slot])) {
                world.router.CancelExternalPose("stream_intent_boundary_lost");return false;
            }
            _opId=_intentIds[slot];_intentVersion=_versions[slot];_endSequence=_ends[slot];_mode=_modes[slot];
            _intentEnded=false;_intentOrigin=world.rig.motionRoot.position;
            int count=_pathCounts[slot];_activePath=new Vector3[count];
            _activeTarget=_targets[slot];_activeTargetPosition=_targetPositions[slot];
            for(int i=0;i<count;i++)_activePath[i]=_paths[slot*12+i];
            _intentIds[slot]=null;_intentHead=(_intentHead+1)%4;_intentCount--;
        }
        return _opId!=null && !_intentEnded;
    }

    private NavMeshPath _localRootPath;

    public bool ValidateRoot(Vector3 desired)
    {
        surfaceStepAllowed=false;
        if(!active || world.rig.motionRoot==null)return false;
        if((_mode==2 || _mode==3) && (_activeTarget==null || !_activeTarget.gameObject.activeInHierarchy
            || Vector3.Distance(_activeTarget.position,_activeTargetPosition)>.05f))return false;
        // 生成姿态不按一米圆或二十厘米路径走廊判死刑；根位置只校验当前可达导航区域。
        Vector3 current=world.rig.motionRoot.position;NavMeshHit start,end,edge;
        // 无渲染斜面替换了台阶碰撞，导航烘焙高度仍可保留台阶；采样中心沿导航表面前进。
        // 实际根高度继续由碰撞支撑决定，导航高度不能直接抬升身体或穿过楼层。
        if(!NavMesh.SamplePosition(current,out start,1.25f,NavMesh.AllAreas))return false;
        Vector3 navDesired=new Vector3(desired.x,start.position.y,desired.z);
        if(!NavMesh.SamplePosition(navDesired,out end,.5f,NavMesh.AllAreas))return false;
        Vector2 delta=new Vector2(desired.x-end.position.x,desired.z-end.position.z);
        // 五厘米以内的导航边缘误差直接投影回网格，根仍留在导航区域内，不因微小误差中断姿态。
        if(delta.sqrMagnitude>.0025f)return false;
        Vector3 reachable=end.position;
        if(NavMesh.Raycast(start.position,end.position,out edge,NavMesh.AllAreas)) {
            // 直线受阻不等于导航区域不可达；只在此分支查询短路径，沿下一拐点执行。
            if(_localRootPath==null)_localRootPath=new NavMeshPath();
            bool routed=false;
            if(NavMesh.CalculatePath(start.position,end.position,NavMesh.AllAreas,_localRootPath)
                && _localRootPath.status==NavMeshPathStatus.PathComplete) {
                Vector3[] corners=_localRootPath.corners;
                if(corners.Length>=2 && corners.Length<=6) {
                    float length=0;
                    for(int i=1;i<corners.Length;i++)length+=Vector3.Distance(corners[i-1],corners[i]);
                    // 这里只修正局部绕角；需要大幅绕行仍交给世界路径重规划。
                    if(length<=Vector3.Distance(start.position,end.position)+.25f) {
                        reachable=corners[1];routed=true;
                    }
                }
            }
            if(!routed) {
                // 最近点恰在边界上也可能报告命中；微小跨界裁回当前可达侧，不越过空洞。
                Vector2 overshoot=new Vector2(desired.x-edge.position.x,desired.z-edge.position.z);
                if(overshoot.sqrMagnitude>.0025f)return false;
                Vector3 inward=start.position-edge.position;
                reachable=edge.position+Vector3.ClampMagnitude(inward,.001f);
            }
        }
        validatedRoot=new Vector3(reachable.x,desired.y,reachable.z);
        surfaceStepAllowed=Mathf.Abs(start.position.y-end.position.y)>.01f;
        return true;
    }

    public void Applied(int sequence)
    {
        if(!active || _opId==null || (sequence%2)!=0)return;
        if(_endSequence==sequence && !_intentEnded) {
            // 时间线走完不等于走到目标；必须检查世界实际位置后才能完成。
            if((_mode==2 || _mode==3) && (_activePath==null || _activePath.Length==0 || _activeTarget==null
                || !_activeTarget.gameObject.activeInHierarchy || Vector3.Distance(_activeTarget.position,_activeTargetPosition)>.05f
                || Vector3.Distance(world.rig.motionRoot.position,_activePath[_activePath.Length-1])>Mathf.Clamp(world.router.locomotion.stopDistance,.01f,.3f))) {
                world.router.CancelExternalPose("stream_target_not_reached");return;
            }
            if(!world.router.CompleteStreamTask(_opId)) { world.router.CancelExternalPose("stream_completion_failed");return; }
            _intentEnded=true;
        }
        if(_progressCount>=8) { world.router.CancelExternalPose("stream_progress_overflow");return; }
        _progress[(_progressHead+_progressCount)%8]="\"op_id\":"+world.router.telemetry.J(_opId)+",\"version\":"+_intentVersion+",\"executed_frame\":"+(sequence*2)+",\"committed_frame\":"+world.receiver.StreamCommittedFrame();
        _progressCount++;
    }

    // 播放回执来自实际Applied，借下一个ACK传输；每条最多两个，保持950字节上限。
    public string TakeProgressJson()
    {
        _lastPoseAck=Time.realtimeSinceStartup;
        string body="[";int count=Mathf.Min(2,_progressCount);
        for(int i=0;i<count;i++) {
            body+=(i==0?"":",")+"{"+_progress[_progressHead]+"}";
            _progress[_progressHead]=null;_progressHead=(_progressHead+1)%8;_progressCount--;
        }
        return body+"]";
    }

    private void Update()
    {
        // 初始缓冲已满或输入中断时仍独立反馈，避免宿主等待进度形成死锁。
        if(active && _progressCount>0 && Time.realtimeSinceStartup-_lastPoseAck>.15f) {
            Event("npc.stream_progress",_progress[_progressHead]);
            _progress[_progressHead]=null;_progressHead=(_progressHead+1)%8;_progressCount--;
            _lastPoseAck=Time.realtimeSinceStartup;
        }
        if(!active && _preparedUntil>0 && Time.realtimeSinceStartup>=_preparedUntil) {
            _preparedUntil=0;world.router.telemetry.motionStreamActive=false;
        }
    }

    public void ReleasePose()
    {
        active=false;_progressHead=_progressCount=0;_preparedUntil=0;world.router.telemetry.motionStreamActive=false;_intentHead=_intentCount=_acceptedVersion=0;
        world.receiver.EmergencyStop();world.rig.RestoreAnimator();
        Event("npc.stream_released","\"reason\":"+world.router.telemetry.J(world.router.poseReleaseReason));
    }
}
