using UdonSharp;
using UnityEngine;
using UnityEngine.AI;

// 正式会话的路径准备与动作接管。只有显式配置完整后才能启用。
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoArdyWorldBridge : UdonSharpBehaviour
{
    // CRC16-CCITT每字节两次固定查表，结果与原逐位算法完全一致。
    private int[] _crcNibbles={0x0000,0x1021,0x2042,0x3063,0x4084,0x50A5,0x60C6,0x70E7,0x8108,0x9129,0xA14A,0xB16B,0xC18C,0xD1AD,0xE1CE,0xF1EF};
    public NekoMidiRouter router;
    public NekoArdyPoseLab receiver;
    public NekoArdyRigLab rig;
    public NekoArdyStreamBridge streamBridge;
    [HideInInspector] public bool executionReady;

    public void RefreshCapability()
    {
        executionReady=Ready() && gameObject.activeInHierarchy && enabled
            && receiver.gameObject.activeInHierarchy && receiver.enabled
            && rig.gameObject.activeInHierarchy && rig.enabled
            && (router.GetSession()<=0 || _blockedSession!=router.GetSession());
        if(router!=null && router.poseExecutor==this) { router.poseCapabilityReady=executionReady; router.poseContinuousReady=executionReady && streamBridge!=null && streamBridge.Ready(); }
    }
    private byte[] _packet=new byte[133];
    private int _count;
    private int _bits;
    private int _pending;
    private bool _receiving;
    private string _operation;
    private int _session;
    private int _epoch;
    private int _request;
    private Vector3 _origin;
    private Vector3 _target;
    private float _yaw;
    private float _preparedUntil;
    private bool _active;
    private bool _moving;
    private int _blockedSession;
    private bool _completing;

    public int AllocatePoseEpoch() { _epoch++; return _epoch; }

    public bool Ready()
    {
        if (!(router!=null && receiver!=null && rig!=null && router.locomotion!=null
            && router.enablePoseStream && router.poseExecutor==this && router.enableOperationLifecycle
            && receiver.worldBridge==this && receiver.labEnabled && receiver.requireAuthority
            && receiver.authorityRouter==router && !receiver.allowIsolatedArm
            && rig.targetAnimator==router.locomotion.animator && rig.motionRoot==router.locomotion.npcRoot
            && rig.navigationWriter==router.locomotion.navAgent && rig.rootEnabled && rig.feetEnabled
            && router.locomotion.npcRoot!=null && router.locomotion.animator!=null && router.locomotion.navAgent!=null
            && rig.bindPoseConfigured && rig.bindRotations!=null && rig.bindRotations.Length==19
            && rig.bindPositions!=null && rig.bindPositions.Length==19 && rig.BindHeight(0)>.19f
            && rig.labApproved && rig.targetBones!=null && rig.targetBones.Length==19)) return false;
        for(int i=0;i<19;i++) if(rig.targetBones[i]==null) return false;
        return true;
    }

    public void ReceiveControl(int channel,int number,int value)
    {
        if(channel!=15 || !Ready()) return;
        if(streamBridge!=null) streamBridge.ReceiveControl(channel,number,value);
        if(number==113 && value==1) { _receiving=false; return; }
        if(number==112) {
            if(_active) router.CancelExternalPose("explicit_stop");
            _blockedSession=router.GetSession(); _preparedUntil=0; _receiving=false; return;
        }
        if(number==110 && value==1) { _count=0; _bits=0; _pending=0; _receiving=true; }
        if(number==111 && value==1 && _receiving) { _receiving=false; Commit(); }
    }

    public void ReceiveData(int channel,int number,int velocity)
    {
        if(streamBridge!=null) streamBridge.ReceiveData(channel,number,velocity);
        if(channel!=15 || !_receiving || number<0 || number>127 || velocity<0 || velocity>127) return;
        _pending|=((number<<7)|velocity)<<_bits; _bits+=14;
        while(_bits>=8) {
            if(_count>=133) { _receiving=false; return; }
            _packet[_count++]=(byte)(_pending&255); _pending>>=8; _bits-=8;
        }
    }

    private int ReadInt(int offset)
    {
        return _packet[offset]|(_packet[offset+1]<<8)|(_packet[offset+2]<<16)|(_packet[offset+3]<<24);
    }

    private void Commit()
    {
        if(_count!=133 || _bits!=0 || _packet[0]!=65 || _packet[1]!=80 || (_packet[2]!=4 && _packet[2]!=5)) return;
        int crc=65535;
        for(int i=0;i<131;i++) {
            crc^=_packet[i]<<8;
            crc=((crc<<4)^_crcNibbles[(crc>>12)&15])&65535;
            crc=((crc<<4)^_crcNibbles[(crc>>12)&15])&65535;
        }
        if(crc!=(_packet[131]|(_packet[132]<<8)) || ReadInt(3)!=router.GetSession() || !router.HasLocalDriverAuthority()) return;
        for(int i=17;i<23;i++) if(_packet[i]!=0) return;
        for(int i=55;i<131;i++) if(_packet[i]!=0) return;
        if(_packet[2]==5 && (_packet[15]!=0 || _packet[16]!=0)) return;
        if(_blockedSession==router.GetSession()) return;
        string op="";
        for(int i=23;i<55;i++) {
            int c=_packet[i];
            if(!((c>=48 && c<=57) || (c>=97 && c<=102))) return;
            op+=((char)c).ToString();
        }
        if(_packet[2]==4) Prepare(op);
        else Begin(op);
    }

    private string Identity()
    {
        return "\"op_id\":"+router.telemetry.J(_operation)+",\"pose_session\":"+_session
            +",\"pose_epoch\":"+_epoch+",\"request_seq\":"+_request;
    }

    private void Prepare(string op)
    {
        if(_active || router.GetControlState()!=NekoMidiRouter.STATE_EXTERNAL || router.ActiveOpsJson()!="[]" || ReadInt(11)<=0) return;
        // 相同准备包不可刷新寿命，也不可在完成后重新使用。
        if(op==_operation) return;
        _preparedUntil=0; _operation=op; _session=router.GetSession(); _request=ReadInt(11); _epoch++;
        _origin=router.locomotion.npcRoot.position; _yaw=router.locomotion.npcRoot.eulerAngles.y;
        _moving=_packet[15]==1;
        if(_packet[15]>1 || rig.targetBones[0]==null) return;
        float scale=rig.MotionScale();
        if(scale<.2f || scale>3f) return;
        _target=_origin;
        if(_moving) {
            int id=_packet[16];
            if(router.anchorTransforms==null || id>=router.anchorTransforms.Length || router.anchorTransforms[id]==null) return;
            _target=router.anchorTransforms[id].position;
        }
        var path=new NavMeshPath();
        Vector3[] corners=new Vector3[]{_origin};
        if(_moving) {
            if(!NavMesh.CalculatePath(_origin,_target,NavMesh.AllAreas,path) || path.status!=NavMeshPathStatus.PathComplete) return;
            corners=path.corners;
        }
        if(corners.Length<1 || corners.Length>16) return;
        if(_moving) {
            Vector3 end=corners[corners.Length-1];
            if(new Vector2(end.x-_target.x,end.z-_target.z).magnitude>.2f) return;
            // anchor 可位于地面上方；行走终点使用已验证的 NavMesh 落点。
            _target=end;
        }
        for(int i=0;i<corners.Length;i++) {
            if(!router.IsPointInsideActivityBounds(corners[i]) || Mathf.Abs(corners[i].y-_origin.y)>.08f
                || Vector3.Distance(corners[i],_origin)>rig.maxDistance) return;
        }
        _preparedUntil=Time.realtimeSinceStartup+5f;
        var t=router.telemetry;
        t.EmitForced("npc.motion_prepared",Identity()+",\"origin\":"+t.Vec3(_origin)+",\"yaw\":"+t.F2(_yaw)
            +",\"scale\":"+scale.ToString("F6")+",\"max_speed\":0.5,\"path_count\":"+corners.Length);
        for(int i=0;i<corners.Length;i++)
            t.EmitForced("npc.motion_path",Identity()+",\"index\":"+i+",\"point\":"+t.Vec3(corners[i]));
    }

    private void Begin(string op)
    {
        if(_active || op!=_operation || ReadInt(7)!=_epoch || ReadInt(11)!=1 || _session!=router.GetSession()
            || Time.realtimeSinceStartup>=_preparedUntil) return;
        var root=router.locomotion.npcRoot;
        if(Vector3.Distance(root.position,_origin)>.02f || Mathf.Abs(Mathf.DeltaAngle(root.eulerAngles.y,_yaw))>1f) return;
        if(router.BeginExternalPoseOperation(op,this)!=null) return;
        _active=true; _preparedUntil=0;
        receiver.operationId=op; receiver.expectedSession=_session; receiver.expectedEpoch=_epoch;
        receiver.ArmLab();
        if(receiver.stopped) { router.CancelExternalPose("pose_arm_failed"); return; }
        receiver.EmitPoseEvent("npc.pose_armed",0,"armed");
    }

    public void CompleteMotion()
    {
        if(!_active || receiver.stopped || receiver.finishSequence<=0 || receiver.executedSequence!=receiver.lastSequence) return;
        if(_moving && Vector3.Distance(router.locomotion.npcRoot.position,_target)>.2f) {
            router.CancelExternalPose("target_not_reached"); return;
        }
        _completing=true;
        if(!router.CompleteExternalPose(_operation)) {
            _completing=false;
            router.CancelExternalPose("pose_completion_rejected");
        }
        _completing=false;
    }

    public void ReleasePose()
    {
        if(streamBridge!=null && streamBridge.active) { streamBridge.ReleasePose(); return; }
        if(!_completing) _blockedSession=_session;
        _active=false;
        receiver.EmergencyStop();
        rig.RestoreAnimator();
    }

    private void Update()
    {
        if(_active && (!Ready() || receiver.stopped || _session!=router.GetSession())) {
            _blockedSession=_session;
            router.CancelExternalPose("pose_stream_stopped");
        }
    }
}
