/*
 * NekoNpcTelemetry —— YUI v1.1/v1.2/v1.3 上行日志的唯一出口（UdonSharp）
 *
 * 所有业务脚本继续只提交 type/body；本类统一补齐公共头、分配 log_seq、执行
 * 20 行/滑动秒预算并按 UTF-8 字节数限制单行。超预算事件进入有界队列，不能
 * 丢失 DISCOVER 目录、ACK 或 operation 生命周期。session 由成功 DISCOVER 安装。
 */
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoNpcTelemetry : UdonSharpBehaviour
{
    [Header("YUI 身份")]
    [Tooltip("旧世界保持 1.1；语义地图为 1.2；Region 定位/局部导航为 1.3")]
    public string specVersion = "1.1";
    [Tooltip("稳定世界标识；发布前必须改成实际 world id 或项目内约定的稳定 id")]
    public string worldId = "wrld_neko_n4lab";

    [Tooltip("协议固定为 yui")]
    public string npcId = "yui";

    [Tooltip("可选显示名门；为空只适合本地自动化测试")]
    public string driverDisplayName = "";

    [Tooltip("调试开关；发布时必须关闭")]
    public bool logOnAllClients = false;

    [Header("预算（v1.1 冻结值）")]
    [Tooltip("运行时会强制钳到 1..20")]
    public int logBudgetPerSec = 20;
    public float eventMergeWindow = 0.5f;
    public float errCooldown = 5.0f;

    [System.NonSerialized] public int emittedLines;
    [System.NonSerialized] public int droppedBudget;
    [System.NonSerialized] public int droppedMerged;
    [System.NonSerialized] public int droppedTooLong;
    [System.NonSerialized] public int droppedTotal;
    [System.NonSerialized] public int logWrapCount;

    private const int MaxJsonUtf8Bytes = 950;
    private const int KeySlots = 48;
    private const int MaxBudget = 20;
    // 日志时间只保留到毫秒；若正好每 1.000 秒释放一批，接收端按闭区间
    // 统计 [t-1,t] 时会把前后两批同时计入。增加 50ms 保护带，确保序列化后
    // 的任意滑动秒也不会跨边界叠加两批，同时只给 DISCOVER 增加极小延迟。
    private const float BudgetWindowSeconds = 1.05f;
    // 普通目录/状态最多占 16 行；为 ACK、错误和安全事件保留 4 行滑动窗余量。
    private const int ForcedBudgetReserve = 4;
    private const int NormalQueueSize = 128;
    private const int ForcedQueueSize = 32;
    private const int LogSeqNormalMax = 2147483646;

    private string[] _keys = new string[KeySlots];
    private float[] _keyLastTime = new float[KeySlots];
    private int _keyCursor;
    public bool motionStreamActive;
    private float[] _normalLineTimes = new float[MaxBudget];
    private int _normalLineCursor;
    private float[] _nonReceiptLineTimes = new float[MaxBudget];
    private int _nonReceiptLineCursor;
    private float[] _lineTimes = new float[MaxBudget];
    private int _lineTimeCursor;
    private int _nextLogSeq = 1;
    private int _session;
    private string[] _normalTypes = new string[NormalQueueSize];
    private string[] _normalBodies = new string[NormalQueueSize];
    private int _normalHead;
    private int _normalTail;
    private int _normalCount;
    private string[] _forcedTypes = new string[ForcedQueueSize];
    private string[] _forcedBodies = new string[ForcedQueueSize];
    private int _forcedHead;
    private int _forcedTail;
    private int _forcedCount;

    private int _droppedSinceReport;
    private int _firstDroppedLogSeq;
    private int _lastDroppedLogSeq;
    private bool _isDriverByName;
    private bool _driverNameResolved;
    private NekoMidiRouter _router;

    void Start()
    {
        for (int i = 0; i < KeySlots; i++)
        {
            _keys[i] = null;
            _keyLastTime[i] = -999f;
        }
        for (int i = 0; i < MaxBudget; i++) { _lineTimes[i] = -999f; _normalLineTimes[i] = -999f; _nonReceiptLineTimes[i] = -999f; }
        _router = GetComponent<NekoMidiRouter>();
        Emit("sys.boot", "\"ready\":true");
    }

    void Update()
    {
        FlushQueues();
        // 丢行报告只在队列排空且有预算时创建，避免报告本身挤占协议事件。
        if (_droppedSinceReport <= 0 || !CanLogHere() || _forcedCount > 0 || _normalCount > 0 || !HasNormalBudget()) return;
        int seq = AllocateLogSeq();
        string body = "\"dropped_since_last_report\":" + _droppedSinceReport
            + ",\"dropped_total\":" + droppedTotal
            + ",\"first_dropped_log_seq\":" + _firstDroppedLogSeq
            + ",\"last_dropped_log_seq\":" + _lastDroppedLogSeq
            + ",\"wrap_count\":" + logWrapCount;
        if (WriteAllocated(seq, "sys.telemetry", body, true))
        {
            _droppedSinceReport = 0;
            _firstDroppedLogSeq = 0;
            _lastDroppedLogSeq = 0;
        }
    }

    private int _stateRevision;
    public void SetSession(int session)
    {
        if (_session != session) {
            // 旧会话排队事件不能套上新会话日志头。
            _normalHead = _normalTail = _normalCount = 0;
            _forcedHead = _forcedTail = _forcedCount = 0;
            _stateRevision = 0;
        }
        _session = session;
    }
    private string CaptureStateRevision(string type, string body)
    {
        bool operation=type=="npc.operation_started" || type=="npc.operation_completed"
            || type=="npc.operation_cancelled" || type=="npc.operation_failed";
        if (type != "npc.state" && type != "npc.ack" && !operation) return body;
        // 版本在事件生成时分配，不在出队时重新变新。
        _stateRevision++;
        body += ",\"state_revision\":" + _stateRevision;
        if(operation && _router!=null)body+=",\"control_state\":"+J(_router.StateName(_router.GetControlState()));
        return body;
    }
    public int GetSession() { return _session; }
    public bool IsTouchEnabled() { return _router != null && _router.IsTouchEnabled(); }

    public bool IsDisplayNameAllowed()
    {
        VRCPlayerApi local = Networking.LocalPlayer;
        if (local == null) return false;
        return driverDisplayName == null || driverDisplayName.Length == 0 || local.displayName == driverDisplayName;
    }

    // 握手后以同步 driverPid + 三对象 ownership 为权威；握手前仅使用显示名便利门。
    public bool IsDriver()
    {
        if (_router != null && _router.GetSession() > 0) return _router.HasLocalDriverAuthority();
        if (_driverNameResolved) return _isDriverByName;
        VRCPlayerApi local = Networking.LocalPlayer;
        if (local == null) return false;
        _isDriverByName = IsDisplayNameAllowed();
        _driverNameResolved = true;
        return _isDriverByName;
    }

    public void Emit(string type, string body)
    {
        body = CaptureStateRevision(type, body);
        if (!CanLogHere()) return;
        if (!FitsEvent(type, body)) { droppedTooLong++; RecordDrop(AllocateLogSeq(), 2); return; }
        if (_forcedCount == 0 && _normalCount == 0 && HasNormalBudget()) WriteAllocated(AllocateLogSeq(), type, body, true);
        else EnqueueNormal(type, body);
    }

    // MIDI ACK/ESTOP 需要在授权失败或 ownership 丢失时仍可回传给本机后端。
    public void EmitForced(string type, string body)
    {
        body = CaptureStateRevision(type, body);
        if (!FitsEvent(type, body)) { droppedTooLong++; RecordDrop(AllocateLogSeq(), 2); return; }
        if (HasBudget() && (motionStreamActive && IsReceipt(type) || _forcedCount == 0 && HasNonReceiptBudget())) WriteAllocated(AllocateLogSeq(), type, body, false);
        else EnqueueForced(type, body);
    }

    public void EmitEvent(string type, string key, string body)
    {
        if (!MergeAllow(key, eventMergeWindow)) { droppedMerged++; RecordDrop(AllocateLogSeq(), 1); return; }
        if (!CanLogHere()) return;
        Emit(type, body);
    }

    // 兼容旧调用面；未知错误统一落为冻结的 internal_error。
    public void EmitErr(string err, string detail)
    {
        string normalized = IsFrozenError(err) ? err : "internal_error";
        EmitProtocolError(normalized, "safety", false, -1, detail);
    }

    public void EmitProtocolError(string err, string source, bool fatal, int relatedSeq, string detail)
    {
        if (!MergeAllow("err:" + err, errCooldown)) { droppedMerged++; RecordDrop(AllocateLogSeq(), 1); return; }
        if (!CanLogHere()) return;
        string body = "\"err\":" + J(err)
            + ",\"code\":" + ErrorCode(err)
            + ",\"source\":" + J(source)
            + ",\"fatal\":" + B(fatal)
            + ",\"related_seq\":" + (relatedSeq < 0 ? "null" : relatedSeq.ToString())
            + ",\"detail\":" + (detail == null ? "null" : J(detail));
        EmitForced("sys.err", body);
    }

    private bool CanLogHere()
    {
        return logOnAllClients || IsDriver();
    }

    private bool HasBudget()
    {
        int limit = Mathf.Clamp(logBudgetPerSec, 1, MaxBudget);
        float now = Time.timeSinceLevelLoad;
        int recent = 0;
        for (int i = 0; i < MaxBudget; i++)
        {
            if (now - _lineTimes[i] < BudgetWindowSeconds) recent++;
        }
        return recent < limit;
    }

    private bool HasNormalBudget()
    {
        if(motionStreamActive) {
            if(!HasBudget() || !HasNonReceiptBudget()) return false;
            int count=0;
            for(int i=0;i<MaxBudget;i++) if(Time.timeSinceLevelLoad-_normalLineTimes[i]<BudgetWindowSeconds)count++;
            // 仍保持20行总预算；持续动作时普通快照最多占两行，其余留给真实回执。
            return count<2;
        }
        int limit = Mathf.Clamp(logBudgetPerSec, 1, MaxBudget);
        int reserve = Mathf.Min(ForcedBudgetReserve, Mathf.Max(0, limit - 1));
        int normalLimit = limit - reserve;
        float now = Time.timeSinceLevelLoad;
        int recent = 0;
        for (int i = 0; i < MaxBudget; i++)
        {
            if (now - _lineTimes[i] < BudgetWindowSeconds) recent++;
        }
        return recent < normalLimit;
    }

    private void ConsumeBudget()
    {
        _lineTimes[_lineTimeCursor] = Time.timeSinceLevelLoad;
        _lineTimeCursor = (_lineTimeCursor + 1) % MaxBudget;
    }

    private bool IsReceipt(string type)
    {
        return type=="npc.pose_ack" || type=="npc.ack" || type=="npc.stream_control" || type=="npc.stream_progress"
            || type=="npc.stream_prepared" || type=="npc.stream_armed" || type=="npc.stream_path"
            || type=="npc.stream_released";
    }

    private bool HasNonReceiptBudget()
    {
        if(!motionStreamActive)return true;
        int count=0;
        for(int i=0;i<MaxBudget;i++)if(Time.timeSinceLevelLoad-_nonReceiptLineTimes[i]<BudgetWindowSeconds)count++;
        // 回执决定端口信用，其他遥测最多占六行；20行总预算保持不变。
        return count<6;
    }

    private void FlushQueues()
    {
        if (!CanLogHere()) return;
        while (_forcedCount > 0 || _normalCount > 0)
        {
            string type;
            string body;
            bool normal=_forcedCount==0;
            if (_forcedCount > 0)
            {
                if (!HasBudget()) return;
                int chosen=0;
                if(motionStreamActive)for(int i=0;i<_forcedCount;i++) {
                    if(IsReceipt(_forcedTypes[(_forcedHead+i)%ForcedQueueSize])) {chosen=i;break;}
                }
                int slot=(_forcedHead+chosen)%ForcedQueueSize;
                type = _forcedTypes[slot]; body = _forcedBodies[slot];
                if(motionStreamActive && !IsReceipt(type) && !HasNonReceiptBudget())return;
                // 在分配日志序号前优先取回执，其余事件保持原有相对顺序。
                for(int i=chosen;i<_forcedCount-1;i++) {
                    int a=(_forcedHead+i)%ForcedQueueSize,b=(_forcedHead+i+1)%ForcedQueueSize;
                    _forcedTypes[a]=_forcedTypes[b];_forcedBodies[a]=_forcedBodies[b];
                }
                _forcedTail=(_forcedTail+ForcedQueueSize-1)%ForcedQueueSize;
                _forcedTypes[_forcedTail]=null;_forcedBodies[_forcedTail]=null;_forcedCount--;
            }
            else
            {
                if (!HasNormalBudget()) return;
                type = _normalTypes[_normalHead]; body = _normalBodies[_normalHead];
                _normalTypes[_normalHead] = null; _normalBodies[_normalHead] = null;
                _normalHead = (_normalHead + 1) % NormalQueueSize; _normalCount--;
            }
            WriteAllocated(AllocateLogSeq(), type, body, normal);
        }
    }

    private void EnqueueNormal(string type, string body)
    {
        // 合并积压状态快照，操作生命周期不合并。
        if (type == "npc.state" || type == "player.pose") {
            for (int i=0; i<_normalCount; i++) {
                int slot=(_normalHead+i)%NormalQueueSize;
                if (_normalTypes[slot] == type && (type=="npc.state" || PosePage(_normalBodies[slot])==PosePage(body))) {
                    _normalBodies[slot]=body; return;
                }
            }
        }
        if (_normalCount >= NormalQueueSize) { droppedBudget++; RecordDrop(AllocateLogSeq(), 0); return; }
        _normalTypes[_normalTail] = type; _normalBodies[_normalTail] = body;
        _normalTail = (_normalTail + 1) % NormalQueueSize; _normalCount++;
    }

    private string PosePage(string body)
    {
        // 玩家姿态按分页合并，保留同页最新批次；目录与生命周期不合并。
        int start=body.IndexOf("\"page\":");
        if(start<0)return body;
        int end=body.IndexOf(",",start);
        return end<0?body.Substring(start):body.Substring(start,end-start);
    }

    private void EnqueueForced(string type, string body)
    {
        if (_forcedCount >= ForcedQueueSize) { droppedBudget++; RecordDrop(AllocateLogSeq(), 0); return; }
        _forcedTypes[_forcedTail] = type; _forcedBodies[_forcedTail] = body;
        _forcedTail = (_forcedTail + 1) % ForcedQueueSize; _forcedCount++;
    }

    private int AllocateLogSeq()
    {
        if (_nextLogSeq > LogSeqNormalMax)
        {
            // 实际运行几乎不可能到达；仍保留冻结的回绕语义。
            int wrap = 2147483647;
            if (CanLogHere() && HasBudget())
            {
                ConsumeBudget();
                string marker = Header(wrap, "sys.log_wrap")
                    + ",\"previous_log_seq\":2147483646,\"next_log_seq\":1,\"wrap_count\":" + (logWrapCount + 1) + "}";
                Debug.Log("[NEKO]" + marker);
                emittedLines++;
                logWrapCount++;
            }
            _nextLogSeq = 1;
        }
        int value = _nextLogSeq;
        _nextLogSeq++;
        return value;
    }

    private bool WriteAllocated(int seq, string type, string body, bool normal)
    {
        string json = Header(seq, type);
        if (body != null && body.Length > 0) json = json + "," + body;
        json = json + "}";
        if (Utf8ByteCount(json) > MaxJsonUtf8Bytes)
        {
            droppedTooLong++;
            RecordDrop(seq, 2);
            return false;
        }
        ConsumeBudget();
        if(!IsReceipt(type)) {
            _nonReceiptLineTimes[_nonReceiptLineCursor]=Time.timeSinceLevelLoad;
            _nonReceiptLineCursor=(_nonReceiptLineCursor+1)%MaxBudget;
        }
        if(normal) {
            _normalLineTimes[_normalLineCursor]=Time.timeSinceLevelLoad;
            _normalLineCursor=(_normalLineCursor+1)%MaxBudget;
        }
        Debug.Log("[NEKO]" + json);
        emittedLines++;
        return true;
    }

    // Router 在写入动态文本前用同一序列化路径预检 950B，避免先尝试超长日志再丢失事件。
    public bool FitsEvent(string type, string body)
    {
        int seq = _nextLogSeq > LogSeqNormalMax ? 1 : _nextLogSeq;
        string json = Header(seq, type) + (body == null || body.Length == 0 ? "}" : "," + body + "}");
        return Utf8ByteCount(json) <= MaxJsonUtf8Bytes;
    }

    private string Header(int seq, string type)
    {
        string spec = specVersion == "1.3" ? "1.3" : (specVersion == "1.2" ? "1.2" : "1.1");
        return "{\"v\":1,\"spec\":" + J(spec) + ",\"session\":" + _session
            + ",\"world_id\":" + J(worldId)
            + ",\"npc\":" + J(npcId)
            + ",\"log_seq\":" + seq
            + ",\"t\":" + Time.timeSinceLevelLoad.ToString("F3")
            + ",\"type\":" + J(type);
    }

    private void RecordDrop(int seq, int reason)
    {
        droppedTotal++;
        if (reason == 0) droppedBudget++;
        if (_droppedSinceReport == 0) _firstDroppedLogSeq = seq;
        _lastDroppedLogSeq = seq;
        _droppedSinceReport++;
    }

    private bool MergeAllow(string key, float window)
    {
        float now = Time.timeSinceLevelLoad;
        for (int i = 0; i < KeySlots; i++)
        {
            if (_keys[i] != null && _keys[i] == key)
            {
                if (now - _keyLastTime[i] < window) return false;
                _keyLastTime[i] = now;
                return true;
            }
        }
        _keys[_keyCursor] = key;
        _keyLastTime[_keyCursor] = now;
        _keyCursor = (_keyCursor + 1) % KeySlots;
        return true;
    }

    private int Utf8ByteCount(string value)
    {
        if (value == null) return 0;
        int bytes = 0;
        for (int i = 0; i < value.Length; i++)
        {
            int c = value[i];
            if (c <= 0x7F) bytes += 1;
            else if (c <= 0x7FF) bytes += 2;
            else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < value.Length)
            {
                int d = value[i + 1];
                if (d >= 0xDC00 && d <= 0xDFFF) { bytes += 4; i++; }
                else bytes += 3;
            }
            else bytes += 3;
        }
        return bytes;
    }

    private bool IsFrozenError(string err)
    {
        return ErrorCode(err) != 34 || err == "internal_error";
    }

    public int ErrorCode(string err)
    {
        if (err == "unknown_cmd") return 1;
        if (err == "not_handshaken") return 3;
        if (err == "not_driver") return 4;
        if (err == "not_owner") return 5;
        if (err == "invalid_state") return 6;
        if (err == "estop_latched") return 7;
        if (err == "invalid_param") return 8;
        if (err == "reserved_bits") return 9;
        if (err == "target_out_of_bounds") return 10;
        if (err == "target_not_on_navmesh") return 11;
        if (err == "no_path") return 12;
        if (err == "target_missing") return 13;
        if (err == "slot_unknown") return 14;
        if (err == "unsupported_capability") return 15;
        if (err == "action_not_found") return 16;
        if (err == "action_busy") return 17;
        if (err == "expression_not_found") return 18;
        if (err == "text_preset_not_found") return 19;
        if (err == "transfer_busy") return 20;
        if (err == "transfer_missing") return 21;
        if (err == "transfer_seq_mismatch") return 22;
        if (err == "text_too_long") return 23;
        if (err == "length_mismatch") return 24;
        if (err == "crc_mismatch") return 25;
        if (err == "invalid_utf8") return 26;
        if (err == "text_timeout") return 27;
        if (err == "stream_incomplete") return 28;
        if (err == "seq_conflict") return 29;
        if (err == "voice_unavailable") return 30;
        if (err == "ownership_failed") return 31;
        if (err == "session_conflict") return 32;
        if (err == "rate_limited") return 33;
        if (err == "driver_auth_failed") return 35;
        if (err == "action_seq_conflict") return 36;
        if (err == "speech_seq_conflict") return 37;
        if (err == "catalog_invalid") return 38;
        return 34;
    }

    public string J(string value)
    {
        if (value == null) return "\"\"";
        string escaped = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
        return "\"" + escaped + "\"";
    }

    public string F2(float value) { return value.ToString("F2"); }
    public string F1(float value) { return value.ToString("F1"); }
    public string B(bool value) { return value ? "true" : "false"; }
    public string Vec3(Vector3 value)
    {
        return "[" + F2(value.x) + "," + F2(value.y) + "," + F2(value.z) + "]";
    }
}
