using UdonSharp;
using UnityEngine;

// 仅用于隔离协议验证，不发布 capability、不修改角色、也不抢占正式控制权。
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoArdyPoseLab : UdonSharpBehaviour
{
    // CRC16-CCITT每字节两次固定查表，结果与原逐位算法完全一致。
    private int[] _crcNibbles={0x0000,0x1021,0x2042,0x3063,0x4084,0x50A5,0x60C6,0x70E7,0x8108,0x9129,0xA14A,0xB16B,0xC18C,0xD1AD,0xE1CE,0xF1EF};
    public bool labEnabled;
    public NekoNpcTelemetry telemetry;
    public bool editorLogProbe;
    public bool allowIsolatedArm;
    public bool requireAuthority;
    public NekoMidiRouter authorityRouter;
    public NekoArdyWorldBridge worldBridge;
    public NekoArdyStreamBridge streamBridge;
    public int streamReceivedSequence;
    public float presentationStart;
    private float _streamClock;
    private int _queueHead,_queueCount;
    private Vector3[] _queueRoots=new Vector3[8];
    private Quaternion[] _queueRotations=new Quaternion[216];
    private int[] _queueSequences=new int[8];
    private int[] _queueContacts=new int[8];
    private int[] _queueTimes=new int[8];
    private float[] _queueArrivals=new float[8];
    private float[] _queueDispatchAllowance=new float[8];
    public int dispatchDelayedFrames;
    public float maxDispatchAllowanceSeconds;
    public int catchupFrames;
    public float maxPresentationDelaySeconds;
    public string operationId;
    public float receiveLeaseSeconds = .5f;
    public int submittedFrames;
    public int approvedSequence;
    private bool _readyToSubmit;
    public int expectedSession = 1;
    public int expectedEpoch = 1;
    public int acceptedFrames;
    public int rejectedFrames;
    public int lastSequence;
    public int frameMilliseconds;
    public Vector3 decodedRoot;
    [HideInInspector] public int rootWindowWraps;
    private bool _rootWindowStarted;
    private int _unwrappedRootX,_unwrappedRootZ,_rawRootX,_rawRootZ;
    public int contactMask = -1;
    public int finishSequence;
    public int executedSequence;
    public Quaternion[] decodedRotations = new Quaternion[27];
    public bool stopped = true;
    public float lastAcceptedAt;
    private byte[] _packet = new byte[133];
    private int _count;
    private int _pending;
    private int _bits;
    private bool _receiving;

    public void ArmLab()
    {
        if (!labEnabled || expectedSession <= 0 || expectedEpoch <= 0) return;
        if (requireAuthority && (authorityRouter == null || !authorityRouter.TouchExternalPose(operationId)
            || authorityRouter.GetSession() != expectedSession)) return;
        stopped = false;
        lastSequence = 0;
        approvedSequence = 0;
        contactMask = -1;
        finishSequence = 0;
        executedSequence = 0;
        _receiving = false;
        _readyToSubmit = false;
        lastAcceptedAt = Time.realtimeSinceStartup;
    }

    public void EmergencyStop()
    {
        if (stopped) return;
        stopped = true;
        _queueHead=_queueCount=0;
        _receiving = false;
        _readyToSubmit = false;
        // 旧代次不能在停止后再次被使用，必须由实验协调器明确设置新代次。
        expectedEpoch++;
    }

    public override void MidiControlChange(int channel, int number, int value)
    {
        if (worldBridge != null) worldBridge.ReceiveControl(channel,number,value);
        if (channel != 15 || !labEnabled) return;
        if (number==114 && value==1 && allowIsolatedArm && !requireAuthority)
        {
            // 仅隔离场景启用；正式世界必须通过宿主会话与控制权握手。
            ArmLab();
            if(editorLogProbe) Debug.Log("[NEKO_ARDY_LAB]{\"type\":\"lab.armed\",\"session\":"+expectedSession+",\"epoch\":"+expectedEpoch+"}");
            return;
        }
        if (number == 112) { EmergencyStop(); return; }
        if (number == 113 && value == 1) { EmergencyStop(); return; }
        if (stopped) return;
        if (LeaseExpired()) { EmergencyStop(); return; }
        if (number == 110 && value == 1)
        {
            _count = 0; _pending = 0; _bits = 0; _receiving = true;
        }
        else if (number == 111 && value == 1)
        {
            if (_receiving) CommitPacket();
            _receiving = false;
        }
    }

    public override void MidiNoteOff(int channel, int number, int velocity)
    {
        if (worldBridge != null) worldBridge.ReceiveData(channel,number,velocity);
        if (channel != 15 || !labEnabled || stopped || !_receiving) return;
        if (number < 0 || number > 127 || velocity < 0 || velocity > 127) { Reject(); return; }
        _pending |= ((number << 7) | velocity) << _bits;
        _bits += 14;
        while (_bits >= 8)
        {
            if (_count >= 133) { Reject(); return; }
            _packet[_count++] = (byte)(_pending & 255);
            _pending >>= 8;
            _bits -= 8;
        }
    }

    private void Reject() { rejectedFrames++; _receiving = false; }

    private int ReadInt(int offset)
    {
        return _packet[offset] | (_packet[offset+1] << 8) | (_packet[offset+2] << 16) | (_packet[offset+3] << 24);
    }

    private float ReadMillimeters(int offset)
    {
        int value = _packet[offset] | (_packet[offset+1] << 8);
        return (value >= 32768 ? value - 65536 : value) * .001f;
    }

    private void CommitPacket()
    {
        // 控制帧由桥接器校验，不混入姿态时间线或消耗姿态序号。
        if(_count==133 && _packet[2]>=4 && _packet[2]<=10) return;
        bool continuous=streamBridge!=null && streamBridge.active;
        if (LeaseExpired()) { EmergencyStop(); Reject(); return; }
        if (_count != 133 || _bits != 0 || _packet[0] != 65 || _packet[1] != 80 || !(_packet[2] == 1 || (_packet[2] & 15) == 2 || _packet[2] == 3 || (continuous && (_packet[2]&15)==12)))
        { Reject(); return; }
        int crc = 65535;
        for (int i=0;i<131;i++)
        {
            crc ^= _packet[i] << 8;
            crc=((crc<<4)^_crcNibbles[(crc>>12)&15])&65535;
            crc=((crc<<4)^_crcNibbles[(crc>>12)&15])&65535;
        }
        int sequence = ReadInt(11);
        if (crc != (_packet[131] | (_packet[132] << 8)) || ReadInt(3) != expectedSession
            || ReadInt(7) != expectedEpoch || sequence <= (continuous?streamReceivedSequence:lastSequence))
        { Reject(); return; }
        if (_packet[2] == 3)
        {
            if(continuous) { Reject(); return; }
            // 结束控制帧也有完整身份和 CRC，必须紧接已接收的最后姿态。
            if (_packet[15]!=1 || _packet[16]!=0 || lastSequence<=0 || sequence!=lastSequence+1 || finishSequence!=0)
            { Reject(); return; }
            finishSequence=sequence;
            lastAcceptedAt=Time.realtimeSinceStartup;
            EmitPoseEvent("npc.pose_ack",sequence,"received");
            return;
        }
        if(finishSequence!=0) { Reject(); return; }
        if(continuous && (_queueCount>=8 || !streamBridge.AcceptSequence(sequence)
            || authorityRouter==null || !authorityRouter.TouchExternalPose(operationId))) { Reject(); return; }
        int slot=(_queueHead+_queueCount)%8;
        Vector3 root=new Vector3(ReadMillimeters(17), ReadMillimeters(19), ReadMillimeters(21));
        int contacts=_packet[2] == 1 ? -1 : (_packet[2] >> 4);
        if(continuous) { _queueRoots[slot]=root; _queueContacts[slot]=contacts; }
        else { decodedRoot=root; contactMask=contacts; }
        for(int joint=0;joint<27;joint++)
        {
            // Udon 的 int 转 uint 使用检查转换；逐字节组装避免负数转换溢出。
            int offset = 23+joint*4;
            uint value = (uint)_packet[offset] | ((uint)_packet[offset+1] << 8)
                | ((uint)_packet[offset+2] << 16) | ((uint)_packet[offset+3] << 24);
            int largest = (int)(value & 3u);
            int shift = 2;
            Quaternion q = new Quaternion(0,0,0,0);
            float sum = 0;
            for(int axis=0;axis<4;axis++)
            {
                if(axis==largest) continue;
                float component = ((value >> shift) & 1023u) / 1023f * 1.41421356237f - .70710678118f;
                q[axis] = component;
                sum += component*component;
                shift += 10;
            }
            q[largest] = Mathf.Sqrt(Mathf.Max(0,1-sum));
            if(continuous) _queueRotations[slot*27+joint]=q;
            else decodedRotations[joint] = q;
        }
        if(continuous) {
            int timestamp=_packet[15] | (_packet[16]<<8);
            // 序号展开时间戳，验证回绕后的100ms固定采样间隔。
            if(timestamp!=(((sequence-1)%65536)*100)%65536) { Reject(); return; }
            if((_packet[2]&15)==12) {
                // 身份、顺序、CRC和时间验证后才展开水平坐标，不让坏包污染窗口。
                int x=Mathf.RoundToInt(root.x*1000f),z=Mathf.RoundToInt(root.z*1000f);
                if(_rootWindowStarted) {
                    _unwrappedRootX+=(x-(_unwrappedRootX%65536)+98304)%65536-32768;
                    _unwrappedRootZ+=(z-(_unwrappedRootZ%65536)+98304)%65536-32768;
                    if(Mathf.Abs(x-_rawRootX)>32768 || Mathf.Abs(z-_rawRootZ)>32768)rootWindowWraps++;
                } else {
                    _unwrappedRootX=x;_unwrappedRootZ=z;_rootWindowStarted=true;
                }
                _rawRootX=x;_rawRootZ=z;
                _queueRoots[slot]=new Vector3(_unwrappedRootX*.001f,root.y,_unwrappedRootZ*.001f);
            }
            _queueSequences[slot]=sequence; _queueTimes[slot]=timestamp; _queueArrivals[slot]=Time.realtimeSinceStartup;
            // MIDI回调也运行在Unity主线程；停顿期间无法观测真实端口到达时刻。
            // 只记录本次实际帧间隔中超出50ms的部分，最多匹配四帧追赶能力，不重置播放时钟。
            _queueDispatchAllowance[slot]=Mathf.Min(.4f,Mathf.Max(0f,Time.unscaledDeltaTime-.05f));
            _queueCount++;
            // 实测约220ms的主线程分发抖动会耗尽200ms起播缓冲；只延后起播，不拉长帧时长。
            if(_streamClock<=0) _streamClock=Time.realtimeSinceStartup+.4f;
            streamReceivedSequence=sequence; lastAcceptedAt=Time.realtimeSinceStartup;
            acceptedFrames++; submittedFrames++;
            EmitPoseEvent("npc.pose_ack",sequence,"received");
            return;
        }
        frameMilliseconds = _packet[15] | (_packet[16] << 8);
        lastSequence = sequence;
        lastAcceptedAt = Time.realtimeSinceStartup;
        acceptedFrames++;
        _readyToSubmit = true;
    }

    private bool LeaseExpired()
    {
        if(streamBridge!=null && streamBridge.active)
            return authorityRouter==null || !authorityRouter.IsExternalPoseAuthorized(operationId);
        return Time.realtimeSinceStartup-lastAcceptedAt >= Mathf.Clamp(receiveLeaseSeconds,.1f,1f);
    }

    public void EmitPoseEvent(string eventType,int sequence,string state)
    {
        string body="\"pose_session\":"+expectedSession+",\"pose_epoch\":"+expectedEpoch
            +",\"pose_sequence\":"+sequence+",\"state\":\""+state+"\"";
        if(streamBridge!=null && streamBridge.active) body+=",\"stream_epoch\":"+streamBridge.StreamEpoch();
        if(requireAuthority && telemetry!=null) body+=",\"op_id\":"+telemetry.J(operationId);
        if(streamBridge!=null && streamBridge.active && eventType=="npc.pose_ack")
            body+=",\"stream_progress\":"+streamBridge.TakeProgressJson();
        if(requireAuthority && eventType=="npc.pose_ack" && executedSequence==sequence)
            body+=",\"applied_sequence\":"+executedSequence;
        if(telemetry!=null && telemetry.GetSession()==expectedSession) telemetry.EmitForced(eventType,body);
        else if(editorLogProbe) Debug.Log("[NEKO_ARDY_LAB]{\"type\":\""+eventType+"\","+body+"}");
    }

    public void CompletePose()
    {
        if(stopped || finishSequence==0 || LeaseExpired() || executedSequence!=lastSequence || _readyToSubmit) return;
        if(requireAuthority && (authorityRouter==null || !authorityRouter.TouchExternalPose(operationId))) { EmergencyStop(); return; }
        EmitPoseEvent("npc.pose_completed",finishSequence,"succeeded");
        if(requireAuthority && worldBridge!=null) worldBridge.CompleteMotion();
        EmergencyStop();
    }

    public void MarkApplied(int sequence)
    {
        if(stopped || sequence<=0 || sequence!=approvedSequence || sequence!=lastSequence || LeaseExpired()) return;
        if(executedSequence==sequence) return;
        executedSequence=sequence;
        if(streamBridge!=null && streamBridge.active) streamBridge.Applied(sequence);
        // 正式模式只在完成控制帧后报告终态，避免逐帧应用日志占满预算。
        if(!requireAuthority) EmitPoseEvent("npc.pose_applied",sequence,"applied");
    }

    // 同一批输入中的停止先清除待提交帧；这里只批准姿态，仍不写入角色骨骼。
    public void FlushPose()
    {
        if (!_readyToSubmit) return;
        if (stopped || LeaseExpired()) { EmergencyStop(); return; }
        if(requireAuthority && (authorityRouter==null || !authorityRouter.TouchExternalPose(operationId))) { EmergencyStop(); return; }
        _readyToSubmit = false;
        submittedFrames++;
        approvedSequence=lastSequence;
        // 接收信用与插值完成分开；下一帧传输可与当前帧插值重叠。
        if(requireAuthority) { EmitPoseEvent("npc.pose_ack",lastSequence,"received"); return; }
        string body = "\"pose_session\":" + expectedSession + ",\"pose_epoch\":" + expectedEpoch
            + ",\"pose_sequence\":" + lastSequence + ",\"state\":\"received\"";
        if (telemetry != null && telemetry.GetSession() == expectedSession)
            telemetry.EmitForced("npc.pose_ack",body);
        else if (editorLogProbe)
            Debug.Log("[NEKO_ARDY_LAB]{\"type\":\"npc.pose_ack\"," + body + "}");
    }

    public void ResetStreamClock()
    {
        streamReceivedSequence=0; _queueHead=_queueCount=0; _streamClock=0; catchupFrames=0; maxPresentationDelaySeconds=0;
        dispatchDelayedFrames=0;maxDispatchAllowanceSeconds=0;
        _rootWindowStarted=false;rootWindowWraps=0;
        presentationStart=0;
    }

    public void TrimStreamFuture(int first)
    {
        while(_queueCount>0 && _queueSequences[(_queueHead+_queueCount-1)%8]>=first) _queueCount--;
        streamReceivedSequence=first-1;
    }

    public int StreamCommittedFrame()
    {
        if(_streamClock<=0) return 0;
        int end=Mathf.Max(0,Mathf.FloorToInt((Time.realtimeSinceStartup+.2f-_streamClock)/.2f)*4);
        return Mathf.Max(executedSequence*2,Mathf.Min(end,(streamReceivedSequence/2)*4));
    }

    public void SelectStreamFrame()
    {
        if(stopped || streamBridge==null || !streamBridge.active || _queueCount==0
            || executedSequence!=lastSequence) return;
        int slot=_queueHead,sequence=_queueSequences[slot];
        float scheduled=_streamClock+(sequence-1)*.1f;
        if(Time.realtimeSinceStartup<scheduled) return;
        // 按到达时间识别真正的晚包；已及时缓冲的帧允许由执行器在渲染停顿后有界追赶。
        if(_queueArrivals[slot]>scheduled+.15f+_queueDispatchAllowance[slot]) {
            authorityRouter.CancelExternalPose("stream_underrun"); return;
        }
        if(_queueArrivals[slot]>scheduled+.15f) {
            dispatchDelayedFrames++;
            maxDispatchAllowanceSeconds=Mathf.Max(maxDispatchAllowanceSeconds,_queueDispatchAllowance[slot]);
        }
        float delay=Time.realtimeSinceStartup-scheduled;
        maxPresentationDelaySeconds=Mathf.Max(maxPresentationDelaySeconds,delay);
        if(delay>.1f)catchupFrames++;
        decodedRoot=_queueRoots[slot]; contactMask=_queueContacts[slot];
        for(int joint=0;joint<27;joint++) decodedRotations[joint]=_queueRotations[slot*27+joint];
        frameMilliseconds=_queueTimes[slot]; lastSequence=approvedSequence=sequence;
        presentationStart=scheduled; _queueHead=(_queueHead+1)%8; _queueCount--;
    }

    private void LateUpdate() { FlushPose(); }

    private void Update()
    {
        if (!stopped && LeaseExpired()) EmergencyStop();
    }
}
