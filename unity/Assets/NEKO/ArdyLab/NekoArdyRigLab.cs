using UdonSharp;
using UnityEngine;

// 受世界控制权约束的姿态执行器；复杂接触能力须独立验收后启用。
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoArdyRigLab : UdonSharpBehaviour
{
    public bool labApproved;
    public Animator targetAnimator;
    public Transform motionRoot;
    public NekoArdyPoseLab receiver;
    public float interpolationSeconds=.1f;
    public float transitionSeconds=.3f;
    public float transitionAngularSpeed=240f;
    public string releaseIdleState="YUI_Procedural_Locomotion";
    private float _entryStarted;
    private float _exitStarted;
    private bool _exiting;
    private Quaternion[] _entryRotations=new Quaternion[19];
    private Quaternion[] _exitRotations=new Quaternion[19];
    private Quaternion[] _exitTargets=new Quaternion[19];
    private Quaternion[] _exitApplied=new Quaternion[19];
    private Quaternion[] _previousRotations=new Quaternion[19];
    private bool _entryLimited,_applyingFrame;
    private float _entryAngleBudget;
    private int _angleBudgetFrame=-1;
    private Vector3 _previousRootPosition,_previousHipsLocal;
    private Quaternion _previousRootRotation;
    public string stopDiagnostics;
    private Quaternion _rootYawFrom,_rootYawTo;
    private Vector3 _exitHipsLocal;
    public int appliedSequence;
    public Transform[] targetBones=new Transform[19];
    // 由编辑器从蒙皮绑定矩阵提取；不得从接管瞬间的动画姿势重新标定。
    public Quaternion[] bindRotations=new Quaternion[19];
    public Vector3[] bindPositions=new Vector3[19];
    public bool bindPoseConfigured;
    public bool rootEnabled;
    public bool feetEnabled;
    public bool logApplied;
    public LayerMask environmentMask=1;
    public UnityEngine.AI.NavMeshAgent navigationWriter;
    public float maxSpeed=2f;
    public float maxDistance=5f;
    public float maxFootCorrection=.25f;
    public float maxFootError;
    public string stopReason;
    private Vector3 _origin;
    private Vector3 _sourceOrigin;
    private Vector3 _rootFrom;
    private Vector3 _rootTo;
    private Vector3 _hipsLocal;
    private Vector3 _hipsRestoreLocal;
    private float _scale;
    private float _motionScale;
    private float _heightFrom;
    private float _heightTo;
    public Collider stairRampCollider;
    private bool _onRamp;
    private Vector3 _blockedPoint;
    private float _rampPoseOffset;
    private int _rampPoseFrame=-1;
    private bool _rootStarted;
    private int _frameTime;
    private bool[] _locked=new bool[2];
    private float[] _contactRetryAt=new float[2];
    public int softContactReleases;
    public string lastSoftContactReason;
    private Vector3[] _anchors=new Vector3[2];
    private float[] _ankleHeight=new float[2];
    private int _reportedSequence;
    private Transform[] _bones;
    // 只用于退出时恢复接管前姿势，不参与重定向。
    private Quaternion[] _bind=new Quaternion[19];
    private Quaternion[] _from=new Quaternion[19];
    private Quaternion[] _to=new Quaternion[19];
    private Quaternion[] _global=new Quaternion[27];
    private int[] _parents=new int[]{-1,0,1,2,3,4,5,4,7,8,9,10,10,4,13,14,15,16,16,0,19,20,21,0,23,24,25};
    private int[] _source=new int[]{0,1,4,5,6,7,8,9,10,13,14,15,16,19,20,21,23,24,25};
    private Quaternion _referenceRotation;
    private float _blendStarted;
    private bool _calibrated;
    private bool _animatorWasEnabled;

    private Transform RootTransform()
    {
        return motionRoot!=null ? motionRoot : targetAnimator.transform;
    }

    // 绑定位置存于运动根的局部空间，运行时只应用当前根比例，不采样动画姿势。
    public float BindHeight(int index)
    {
        if(bindPositions==null || index<0 || index>=bindPositions.Length) return 0f;
        return RootTransform().TransformVector(bindPositions[index]).y;
    }

    public float MotionScale()
    {
        if(bindPositions==null || bindPositions.Length!=19) return 0f;
        Transform root=RootTransform();
        float right=root.TransformVector(bindPositions[14]-bindPositions[13]).magnitude
            +root.TransformVector(bindPositions[15]-bindPositions[14]).magnitude;
        float left=root.TransformVector(bindPositions[17]-bindPositions[16]).magnitude
            +root.TransformVector(bindPositions[18]-bindPositions[17]).magnitude;
        // Core27绑定数据：髋关节到踝关节0.86821米。步幅依腿长，不能依头身比例。
        return (left+right)*.5f/.86821f;
    }

    private bool AuthorityValid()
    {
        if(receiver==null || !receiver.requireAuthority) return true;
        NekoMidiRouter router=receiver.authorityRouter;
        return router!=null && router.locomotion!=null && router.locomotion.externalPoseActive
            && router.GetSession()==receiver.expectedSession && router.IsExternalPoseAuthorized(receiver.operationId);
    }

    public void Calibrate()
    {
        if(!labApproved || targetAnimator==null || !targetAnimator.isHuman || receiver==null) return;
        if(!AuthorityValid()) { Fail("authority_lost"); return; }
        if(motionRoot!=null && !targetAnimator.transform.IsChildOf(motionRoot)) { Fail("invalid_motion_root"); return; }
        // 有导航写入者时拒绝接管，不擅自停用或覆盖原导航。
        if(rootEnabled && navigationWriter!=null && navigationWriter.enabled) { Fail("root_writer_conflict"); return; }
        if(!bindPoseConfigured || bindRotations==null || bindRotations.Length!=19 || bindPositions==null || bindPositions.Length!=19) { Fail("bind_pose_missing"); return; }
        for(int i=0;i<19;i++) {
            Vector3 position=bindPositions[i];
            if(float.IsNaN(position.sqrMagnitude) || float.IsInfinity(position.sqrMagnitude)) { Fail("bind_position_invalid"); return; }
            Quaternion b=bindRotations[i];
            float norm=b.x*b.x+b.y*b.y+b.z*b.z+b.w*b.w;
            if(float.IsNaN(norm) || float.IsInfinity(norm) || Mathf.Abs(norm-1f)>.01f) { Fail("bind_pose_invalid"); return; }
        }
        _referenceRotation=RootTransform().rotation;
        if(targetBones==null || targetBones.Length!=19) return;
        _bones=targetBones;
        for(int i=0;i<19;i++)
        {
            if(_bones[i]==null) return;
            _bind[i]=Quaternion.Inverse(_referenceRotation)*_bones[i].rotation;
            _entryRotations[i]=_bones[i].rotation;
            _from[i]=_bones[i].rotation;
            _to[i]=_from[i];
        }
        _animatorWasEnabled=targetAnimator.enabled;
        _entryStarted=Time.realtimeSinceStartup; _exiting=false; _entryLimited=true;
        _origin=RootTransform().position;
        _rootFrom=_origin; _rootTo=_origin; _rootYawFrom=_rootYawTo=RootTransform().rotation;
        _hipsRestoreLocal=_bones[0].localPosition;
        _hipsLocal=_bones[0].parent.InverseTransformPoint(RootTransform().TransformPoint(bindPositions[0]));
        _scale=BindHeight(0)/.9544f;
        _motionScale=MotionScale();
        if(rootEnabled && (_scale<.2f || _scale>3f || _motionScale<.2f || _motionScale>3f)) { Fail("invalid_scale"); return; }
        _ankleHeight[0]=BindHeight(18);
        _ankleHeight[1]=BindHeight(15);
        if(feetEnabled && (_ankleHeight[0]<0f || _ankleHeight[1]<0f || _ankleHeight[0]>.3f*_scale || _ankleHeight[1]>.3f*_scale))
        { Fail("invalid_bind_ankle_height"); return; }
        _rootStarted=false; appliedSequence=0; maxFootError=0; stopReason=""; stopDiagnostics="";
        _heightFrom=0; _heightTo=0; _rampPoseOffset=0; _rampPoseFrame=-1;
        _locked[0]=false; _locked[1]=false;
        _contactRetryAt[0]=0;_contactRetryAt[1]=0;softContactReleases=0;lastSoftContactReason="";
        _reportedSequence=0;
        targetAnimator.enabled=false;
        _calibrated=true;
    }

    public void ApplyLatest()
    {
#if !COMPILER_UDONSHARP && UNITY_EDITOR
        if(!Application.isPlaying) _entryAngleBudget=8f;
        else
#endif
        if(_angleBudgetFrame!=Time.frameCount) {
            _angleBudgetFrame=Time.frameCount;
            _entryAngleBudget=Mathf.Min(8f,transitionAngularSpeed*Time.unscaledDeltaTime);
        }
        _applyingFrame=false;
        if(_exiting) { ApplyExit(); return; }
        if(receiver!=null && !receiver.stopped && !AuthorityValid()) { Fail("authority_lost"); return; }
        if(!_calibrated && labApproved && receiver!=null && !receiver.stopped) Calibrate();
        if(!_calibrated || !labApproved || receiver==null) return;
        if(receiver.stopped) { RestoreAnimator(); return; }
        receiver.SelectStreamFrame();
        if(receiver.stopped || !_calibrated) return;
        for(int i=0;i<19;i++) _previousRotations[i]=_bones[i].rotation;
        _previousRootPosition=RootTransform().position;_previousRootRotation=RootTransform().rotation;
        _previousHipsLocal=_bones[0].localPosition;_applyingFrame=true;
        if(receiver.approvedSequence>0 && receiver.lastSequence==receiver.approvedSequence && receiver.approvedSequence!=appliedSequence)
        {
            if(receiver.streamBridge!=null && receiver.streamBridge.active && !receiver.streamBridge.BeforeApply(receiver.lastSequence)) return;
            if(rootEnabled && !PrepareRoot()) return;
            for(int j=0;j<27;j++)
                _global[j]=_parents[j]<0 ? receiver.decodedRotations[j] : _global[_parents[j]]*receiver.decodedRotations[j];
            Quaternion rootQ=_global[0];
            Vector3 forward=(_referenceRotation*new Quaternion(rootQ.x,-rootQ.y,-rootQ.z,rootQ.w))*Vector3.forward;
            forward.y=0;
            _rootYawFrom=RootTransform().rotation;
            _rootYawTo=forward.sqrMagnitude>.001f?Quaternion.LookRotation(forward):_rootYawFrom;
            for(int i=0;i<19;i++)
            {
                Quaternion q=_global[_source[i]];
                // Core27 左侧为 +X，当前 Avatar 左侧为 -X；按 X 镜像转换旋转。
                Quaternion converted=new Quaternion(q.x,-q.y,-q.z,q.w);
                // 连续帧只插值模型姿态；IK后的旋转不得反馈到下一帧生成姿态。
                _from[i]=appliedSequence>0 && !_entryLimited ? _to[i] : _bones[i].rotation;
                _to[i]=_referenceRotation*converted*bindRotations[i];
            }
            if(appliedSequence==0) _entryStarted=Time.realtimeSinceStartup;
            appliedSequence=receiver.lastSequence;
            _blendStarted=receiver.streamBridge!=null && receiver.streamBridge.active ? receiver.presentationStart : Time.realtimeSinceStartup;
        }
        if(appliedSequence==0) return;
        float t=Mathf.Clamp01((Time.realtimeSinceStartup-_blendStarted)/Mathf.Max(.001f,interpolationSeconds));
        if(rootEnabled && !MoveRoot(t)) return;
        float entry=Mathf.SmoothStep(0f,1f,Mathf.Clamp01((Time.realtimeSinceStartup-_entryStarted)/Mathf.Max(.001f,transitionSeconds)));
        float remaining=0;
        float entryStep=_entryAngleBudget,usedAngle=0f;
        for(int i=0;i<19;i++) {
            Quaternion wanted=Quaternion.Slerp(_entryRotations[i],Quaternion.Slerp(_from[i],_to[i],t),entry);
            Quaternion applied=_entryLimited?Quaternion.RotateTowards(_previousRotations[i],wanted,entryStep):wanted;
            remaining=Mathf.Max(remaining,Quaternion.Angle(applied,wanted));
            usedAngle=Mathf.Max(usedAngle,Quaternion.Angle(_previousRotations[i],applied));
            _bones[i].rotation=applied;
        }
        _entryAngleBudget=Mathf.Max(0f,_entryAngleBudget-usedAngle);
        if(entry>=1f && remaining<.25f) _entryLimited=false;
        if(rootEnabled) _bones[0].localPosition=Vector3.Lerp(_hipsRestoreLocal,_hipsLocal,entry);
        if(rootEnabled) _bones[0].position+=Vector3.up*Mathf.Lerp(_heightFrom,_heightTo,t);
        if(feetEnabled && rootEnabled && receiver.contactMask>=0)
        {
            bool left=(receiver.contactMask&1)!=0, right=(receiver.contactMask&4)!=0;
            PrepareFoot(0,18,left); PrepareFoot(1,15,right);
            if(receiver.stopped) return;
            // 双腿到已锁定接触点的可达范围决定骨盆高度，不把体型差异误判成地面故障。
            bool soft=SoftPose();
            float leftDrop=PelvisDrop(0,16,17,18),rightDrop=PelvisDrop(1,13,14,15);
            // 支撑修正按绑定腿长限幅，适用于平地和斜面；不拉长骨骼或扩大足部修正范围。
            float supportLimit=Mathf.Min(.25f,.3f*.86821f*_motionScale);
            // 接触是可放弃的姿态辅助，不是生成动作的硬合格线；单腿不可达时只释放该脚。
            if(soft && leftDrop>supportLimit) { ReleaseSoftContact(0,"pelvis_reach");leftDrop=0; }
            if(soft && rightDrop>supportLimit) { ReleaseSoftContact(1,"pelvis_reach");rightDrop=0; }
            float rampTarget=-Mathf.Max(leftDrop,rightDrop);
            if(-rampTarget>supportLimit+.001f) {
                stopDiagnostics="support_drop="+(-rampTarget)+",limit="+supportLimit+",root="+RootTransform().position;
                Fail("pelvis_correction_limit");return;
            }
            // 新支撑必须先满足几何约束；只对解除补偿的方向做时间平滑，不能让锁定脚悬空等待。
            rampTarget=Mathf.Clamp(rampTarget,-supportLimit,0);
            if(!soft && rampTarget<_rampPoseOffset) _rampPoseOffset=rampTarget;
            if(_rampPoseFrame!=Time.frameCount)
            {
                _rampPoseFrame=Time.frameCount;
                _rampPoseOffset=Mathf.MoveTowards(_rampPoseOffset,rampTarget,.8f*Time.deltaTime);
            }
            _bones[0].position+=Vector3.up*_rampPoseOffset;
            // 先同时求两腿所需的骨盆下降量，再求解双腿，避免第二条腿破坏第一条腿的锁定。
            float drop=Mathf.Max(PelvisDrop(0,16,17,18),PelvisDrop(1,13,14,15));
            if(!soft && drop>.12f) { stopDiagnostics="drop="+drop+",ramp="+_onRamp+",offset="+_rampPoseOffset+",target="+rampTarget+",mask="+receiver.contactMask+",left="+_anchors[0]+",right="+_anchors[1]+",root="+RootTransform().position; Fail("pelvis_correction_limit"); return; }
            if(!soft)_bones[0].position-=Vector3.up*drop;
            ConstrainFoot(0,16,17,18,left);
            if(receiver.stopped) return;
            ConstrainFoot(1,13,14,15,right);
        }
        if(receiver.stopped) return;
        if(t>=.999f && appliedSequence>0)
        {
            if(_reportedSequence!=appliedSequence)
            {
                receiver.MarkApplied(appliedSequence);
                _reportedSequence=appliedSequence;
            }
            if(receiver.finishSequence==appliedSequence+1)
            {
                receiver.CompletePose();
                RestoreAnimator();
            }
        }
    }

    public void RestoreAnimator()
    {
        if(!_calibrated) return;
        if(logApplied) Debug.Log("[NEKO_ARDY_LAB]{\"type\":\"rig.restored\",\"sequence\":"+appliedSequence+",\"distance_mm\":"+Mathf.RoundToInt(Vector3.Distance(RootTransform().position,_origin)*1000)+",\"foot_error_mm\":"+Mathf.RoundToInt(maxFootError*1000)+"}");
        for(int i=0;i<19;i++) if(_bones[i]!=null) { _exitRotations[i]=_bones[i].rotation; _exitApplied[i]=_exitRotations[i]; }
        _exitHipsLocal=_bones[0].localPosition;
        _calibrated=false;
        // 真正释放才进入过渡，起点是最后实际姿态，不回写旧世界位置。
        _exitStarted=Time.realtimeSinceStartup; _exiting=true;
        var router=receiver!=null ? receiver.authorityRouter : null;
        if(router!=null && router.locomotion!=null) router.locomotion.poseTransitionActive=true;
        targetAnimator.enabled=_animatorWasEnabled || (router!=null && receiver.requireAuthority);
        if(targetAnimator.enabled && targetAnimator.HasState(0,Animator.StringToHash(releaseIdleState))) {
            targetAnimator.SetInteger("FidgetId",0);
            targetAnimator.SetFloat("MoveForward",0f);targetAnimator.SetFloat("MoveSide",0f);
            targetAnimator.SetFloat("Turn",0f);targetAnimator.SetFloat("MoveCycleRate",1f);
            targetAnimator.CrossFadeInFixedTime(releaseIdleState,transitionSeconds,0);
        }
    }

    private void Fail(string reason)
    {
        stopReason=reason;
        if(receiver!=null && receiver.streamBridge!=null && (reason=="foot_no_ground" || reason=="unsupported_ground" || reason=="obstacle"))
            receiver.streamBridge.RememberBlockedPoint(_blockedPoint);
        // 约束失败的中间姿态从未通过验证，不能拿它作为退出起点显示一帧。
        if(_applyingFrame && _bones!=null) {
            RootTransform().position=_previousRootPosition;RootTransform().rotation=_previousRootRotation;
            for(int i=0;i<19;i++) _bones[i].rotation=_previousRotations[i];
            _bones[0].localPosition=_previousHipsLocal;_applyingFrame=false;
        }
        if(receiver!=null && receiver.requireAuthority && receiver.authorityRouter!=null
            && receiver.authorityRouter.locomotion.externalPoseActive)
            receiver.authorityRouter.CancelExternalPose(reason);
        else if(receiver!=null && !receiver.stopped) receiver.EmergencyStop();
        RestoreAnimator();
        if(logApplied) Debug.Log("[NEKO_ARDY_LAB]{\"type\":\"rig.stopped\",\"reason\":\""+reason+"\"}");
    }

    private bool PrepareRoot()
    {
        if(feetEnabled && receiver.contactMask<0 && !SoftPose()) { Fail("missing_contacts"); return false; }
        if(navigationWriter!=null && navigationWriter.enabled) { Fail("root_writer_conflict"); return false; }
        Vector3 source=receiver.decodedRoot;
        if(!_rootStarted) { _sourceOrigin=source; _frameTime=receiver.frameMilliseconds; }
        int elapsed=(receiver.frameMilliseconds-_frameTime+65536)%65536;
        Vector3 delta=(source-_sourceOrigin)*_motionScale;
        Vector3 desired=_origin+_referenceRotation*new Vector3(-delta.x,0,delta.z);
        if(receiver.streamBridge!=null && receiver.streamBridge.active) desired.y=RootTransform().position.y;
        if(_rootStarted && (elapsed<=0 || elapsed>500 || (!SoftPose() && Vector3.Distance(desired,_rootTo)>maxSpeed*elapsed*.001f+.01f)))
        { Fail("trajectory_speed_or_time"); return false; }
        if(receiver.streamBridge!=null && receiver.streamBridge.active && !receiver.streamBridge.ValidateRoot(desired))
        { Fail("stream_path_constraint"); return false; }
        if(receiver.streamBridge!=null && receiver.streamBridge.active)desired=receiver.streamBridge.validatedRoot;
        // 持续模式已按当前意图验证路径走廊/自由活动范围；旧有限任务半径不能截断路径终点的待机。
        if((receiver.streamBridge==null || !receiver.streamBridge.active) && Vector3.Distance(desired,_origin)>maxDistance)
        { Fail("trajectory_distance"); return false; }
        if(!SoftPose() && Mathf.Abs((source.y-.9544f)*_motionScale)>.3f) { Fail("trajectory_height"); return false; }
        _rootStarted=true; _frameTime=receiver.frameMilliseconds;
        _rootFrom=RootTransform().position; _rootTo=desired;
        // 骨盆同样保留独立模型高度；地形和接触补偿仅在最终执行阶段叠加。
        _heightFrom=appliedSequence>0 ? _heightTo : _bones[0].position.y-_bones[0].parent.TransformPoint(_hipsLocal).y;
        _heightTo=(source.y-.9544f)*_motionScale;
        return true;
    }

    private bool MoveRoot(float blend)
    {
        if(navigationWriter!=null && navigationWriter.enabled) { Fail("root_writer_conflict"); return false; }
        Vector3 current=RootTransform().position;
        Vector3 desired=Vector3.Lerp(_rootFrom,_rootTo,blend);
        _blockedPoint=desired;
        bool surface=receiver.streamBridge!=null && receiver.streamBridge.active;
        float groundReference=surface?current.y:_origin.y;
        if(surface) desired.y=current.y;
        float radius=.18f*_scale;
        RaycastHit floor,support;
        // 中心射线验证真实地面坡度；球扫在接缝产生的圆弧法线不能用作坡度。
        bool centerGround=Physics.Raycast(desired+Vector3.up*.3f,Vector3.down,out floor,.6f,environmentMask.value,QueryTriggerInteraction.Ignore);
        // 相同脚底范围给出身体可下降到的高度，防止误把仍承重的地面边缘当障碍。
        Vector3 supportStart=desired+Vector3.up*(radius+.3f);
        if(!Physics.SphereCast(supportStart,radius,Vector3.down,out support,.6f,environmentMask.value,QueryTriggerInteraction.Ignore))
        { Fail("unsupported_ground"); return false; }
        // 中心射线落入接缝时，仅允许脚底范围内法线合格的真实支撑接替；不能跨越无支撑空洞。
        if(!centerGround || (floor.normal.y<.9f && floor.collider!=stairRampCollider)) floor=support;
        _onRamp=stairRampCollider!=null && floor.collider==stairRampCollider;
        // 已标注影院斜面约33度；其他地面的坡度门槛不变。
        if(floor.normal.y<(_onRamp?.819152f:.9f)) { Fail("unsupported_ground");return false; }
        desired.y=Mathf.Max(floor.point.y,supportStart.y-support.distance-radius);
        // 地毯边缘仍承重时中心射线可能落到低层地面，按最终支撑高度检查局部高差。
        if(Mathf.Abs(desired.y-groundReference)>.08f) { Fail("unsupported_ground");return false; }
        Vector3 step=desired-current;
        Vector3 bottom=current+Vector3.up*(radius+.025f);
        Vector3 top=current+Vector3.up*Mathf.Max(radius+.025f,1.55f*_scale-radius);
        if(Physics.CheckCapsule(bottom,top,radius,environmentMask.value,QueryTriggerInteraction.Ignore))
        {
            // 仅失败时取得碰撞对象，正常逐帧路径不分配数组。
            Collider[] overlap=Physics.OverlapCapsule(bottom,top,radius,environmentMask.value,QueryTriggerInteraction.Ignore);
            stopDiagnostics="overlap="+(overlap.Length>0?overlap[0].name:"unknown")+",root="+current;
            Fail("obstacle"); return false;
        }
        RaycastHit obstacle=new RaycastHit();
        // 行走时提前留出重新接管的站立余量；已记录障碍旁的退出仍按真实碰撞边界执行。
        float margin=Vector3.Distance(_rootTo,_rootFrom)>.02f && (receiver.streamBridge==null || !receiver.streamBridge.NearBlockedPoint(current))?.15f:.01f;
        bool blocked=step.magnitude>.0001f && Physics.CapsuleCast(bottom,top,radius,step.normalized,out obstacle,step.magnitude+margin,environmentMask.value,QueryTriggerInteraction.Ignore);
        if(blocked && obstacle.distance>step.magnitude+.01f) {
            float slope=obstacle.collider==stairRampCollider?.819152f:.9f;
            if(obstacle.normal.y>=slope)blocked=false;
            // 圆角接触低矮地面边缘不是提前刹停依据；到达该步之前仍重新验证真实支撑和实际碰撞。
            else if(obstacle.normal.y>.5f && obstacle.point.y<=Mathf.Max(current.y,desired.y)+.08f)blocked=false;
        }
        // 当前实际步长内的碰撞仍无条件阻止，不增加额外逐帧查询。
        if(blocked)
        {
            _blockedPoint=obstacle.point;
            stopDiagnostics="cast="+obstacle.collider.name+",distance="+obstacle.distance+",step="+step+",floor="+floor.collider.name+",normal="+obstacle.normal;
            Fail("obstacle"); return false;
        }
        RootTransform().position=desired;
        RootTransform().rotation=Quaternion.Slerp(_rootYawFrom,_rootYawTo,blend);
        // 每帧从绑定局部位置开始，防止骨盆偏移累加。
        _bones[0].localPosition=_hipsLocal;
        return true;
    }

    private bool SoftPose()
    {
        return receiver!=null && receiver.streamBridge!=null && receiver.streamBridge.active;
    }

    private void ReleaseSoftContact(int side,string reason)
    {
        _locked[side]=false;_contactRetryAt[side]=Time.realtimeSinceStartup+.2f;
        softContactReleases++;lastSoftContactReason=reason;
    }

    private void PrepareFoot(int side,int end,bool contact)
    {
        Transform c=_bones[end];
        if(!contact) { _locked[side]=false;_contactRetryAt[side]=0; return; }
        RaycastHit floor;
        if(!_locked[side])
        {
            if(SoftPose() && Time.realtimeSinceStartup<_contactRetryAt[side])return;
            if(!Physics.Raycast(c.position+Vector3.up*.25f,Vector3.down,out floor,.6f,environmentMask.value,QueryTriggerInteraction.Ignore))
            { if(SoftPose()) { ReleaseSoftContact(side,"foot_no_ground");return; } _blockedPoint=c.position;stopDiagnostics="foot="+end+",missing="+c.position;Fail("foot_no_ground"); return; }
            _anchors[side]=new Vector3(c.position.x,floor.point.y+_ankleHeight[side],c.position.z);
            _locked[side]=true;
        }
    }

    private float PelvisDrop(int side,int upper,int lower,int end)
    {
        if(!_locked[side]) return 0;
        Vector3 d=_bones[upper].position-_anchors[side];
        float reach=(Vector3.Distance(_bones[upper].position,_bones[lower].position)+Vector3.Distance(_bones[lower].position,_bones[end].position))*.995f;
        float horizontal=d.x*d.x+d.z*d.z;
        if(horizontal>=reach*reach) return 1;
        return Mathf.Max(0,d.y-Mathf.Sqrt(reach*reach-horizontal));
    }

    private void ConstrainFoot(int side,int upper,int lower,int end,bool contact)
    {
        if(!contact || !_locked[side]) return;
        Transform a=_bones[upper]; Transform b=_bones[lower]; Transform c=_bones[end];
        Vector3 target=_anchors[side];
        if(Vector3.Distance(c.position,target)>maxFootCorrection) { if(SoftPose())ReleaseSoftContact(side,"foot_correction_limit");else Fail("foot_correction_limit"); return; }
        Quaternion footRotation=c.rotation;
        Vector3 ac=target-a.position;
        float lengthA=Vector3.Distance(a.position,b.position), lengthB=Vector3.Distance(b.position,c.position);
        float distance=ac.magnitude;
        if(distance<.001f || distance>lengthA+lengthB+.015f) { if(SoftPose())ReleaseSoftContact(side,"foot_unreachable");else Fail("foot_unreachable"); return; }
        float reach=Mathf.Clamp(distance,Mathf.Abs(lengthA-lengthB)+.0001f,lengthA+lengthB-.0001f);
        Vector3 direction=ac.normalized;
        Vector3 bend=Vector3.ProjectOnPlane(b.position-a.position,direction).normalized;
        if(bend.sqrMagnitude<.01f) bend=Vector3.ProjectOnPlane(_referenceRotation*Vector3.forward,direction).normalized;
        float along=(lengthA*lengthA-lengthB*lengthB+reach*reach)/(2*reach);
        Vector3 knee=a.position+direction*along+bend*Mathf.Sqrt(Mathf.Max(0,lengthA*lengthA-along*along));
        a.rotation=Quaternion.FromToRotation(b.position-a.position,knee-a.position)*a.rotation;
        b.rotation=Quaternion.FromToRotation(c.position-b.position,target-b.position)*b.rotation;
        c.rotation=footRotation;
        maxFootError=Mathf.Max(maxFootError,Vector3.Distance(c.position,target));
    }

    private void ApplyExit()
    {
        if(targetAnimator==null || _bones==null) { FinishExit(); return; }
        float elapsed=Mathf.Clamp01((Time.realtimeSinceStartup-_exitStarted)/Mathf.Max(.001f,transitionSeconds));
        float blend=Mathf.SmoothStep(0f,1f,elapsed);
        // 必须先缓存全部目标。写父骨骼后再读子骨骼会把混合结果当成目标反复叠加。
        for(int i=0;i<19;i++) if(_bones[i]!=null) _exitTargets[i]=_bones[i].rotation;
        Vector3 hipsTarget=_bones[0].localPosition;
        float remaining=0,step=Mathf.Min(8f,transitionAngularSpeed*Time.unscaledDeltaTime);
        for(int i=0;i<19;i++) if(_bones[i]!=null) {
            Quaternion wanted=Quaternion.Slerp(_exitRotations[i],_exitTargets[i],blend);
            _exitApplied[i]=Quaternion.RotateTowards(_exitApplied[i],wanted,step);
            _bones[i].rotation=_exitApplied[i];
            remaining=Mathf.Max(remaining,Quaternion.Angle(_exitApplied[i],_exitTargets[i]));
        }
        _bones[0].localPosition=Vector3.Lerp(_exitHipsLocal,hipsTarget,blend);
        // 默认300ms；大角度起点需要继续保持独占过渡，不能在未混合完时硬交回。
        if(elapsed>=1f && remaining<.25f) {
            FinishExit();
        }
    }

    private void LateUpdate()
    {
        // 渲染偶发停顿后最多执行四个已缓冲帧；每一帧仍完整经过控制权、碰撞和接触验证。
        // 不改时间戳，不丢帧冒充执行，不将渲染停顿误判为网络欠载。
        for(int i=0;i<4;i++) {
            int previous=receiver!=null?receiver.executedSequence:0;
            ApplyLatest();
            if(receiver==null || receiver.stopped || _exiting || receiver.streamBridge==null || !receiver.streamBridge.active
                || receiver.executedSequence<=previous)break;
        }
    }
    private void FinishExit()
    {
        _exiting=false;
        var router=receiver!=null ? receiver.authorityRouter : null;
        if(router!=null && router.locomotion!=null) router.locomotion.poseTransitionActive=false;
    }
    private void OnDisable() { RestoreAnimator(); FinishExit(); }
}
