/*
 * NekoMidiRouter —— YUI NPC 协议 v1.1 的唯一 MIDI 路由、安全状态机与生命周期协调器。
 * AnyDance/YOLO 不得导入、回退或共享本脚本的任何状态。
 */
using UdonSharp;
using UnityEngine;
using UnityEngine.AI;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoMidiRouter : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;
    public NekoNpcLocomotion locomotion;
    public NekoNpcPerception perception;
    public NekoNpcNameplate nameplate;

    [Header("世界与握手")]
    public string worldName = "YUI Matchbox";
    public int driverClaimCode;
    public int catalogRevision = 2;

    [Header("双包围盒")]
    public Vector3 wireBoundsMin = new Vector3(-9f, -1f, -9f);
    public Vector3 wireBoundsMax = new Vector3(9f, 6f, 9f);
    public Vector3 boundsMin = new Vector3(-8f, 0f, -8f);
    public Vector3 boundsMax = new Vector3(8f, 5f, 8f);

    [Header("MIDI 冻结参数")]
    public int midiChannel;
    public float watchdogSec = 3f;
    public float dupWindowSec = 5f;

    [Header("显式发布的 capability")]
    public bool enableGoto = true;
    public bool enableFollow;
    public bool enableWander;
    public bool enableActions;
    public bool enableExpressions;
    public bool enableTextPreset = true;
    public bool enableTextUtf8;
    public bool enableRayScan = true;
    public bool enableTouch = true;
    public bool enablePlayerPose = true;
    public bool enableSnapshot = true;
    public bool enableSocialSignals;
    public bool enableAnchors;
    public bool enableOperationLifecycle = true;

    [Header("动作目录（id 为数组下标）")]
    public string[] actionNames = new string[] { "greet", "nod", "shake_head", "wave", "bow", "explain", "think", "celebrate", "listen", "confused", "point_left", "point_right", "point_forward", "shrug", "laugh", "comfort" };
    public string[] actionSemanticKeys = new string[] { "greet", "agree_nod", "disagree_shake_head", "greet_wave", "apologize_bow", "explain", "think", "celebrate", "listen", "confused", "point_left", "point_right", "point_forward", "shrug", "laugh", "comfort" };
    public string[] actionDescriptionsZh = new string[] { "普通问候姿态", "点头表示同意", "摇头表示否定", "向目标友好挥手", "道歉或致意鞠躬", "做解释性手势", "进入思考姿态", "开心庆祝", "安静倾听", "表现困惑", "指向左侧", "指向右侧", "指向前方", "耸肩表示不确定", "开心大笑", "做安慰性姿态" };
    [Tooltip("每项必须是 1..8 个语义标签组成的 JSON 数组")]
    public string[] actionIntentTagsJson = new string[] { "[\"greeting\"]", "[\"agreement\"]", "[\"disagreement\"]", "[\"greeting\",\"friendly\"]", "[\"apology\"]", "[\"explanation\"]", "[\"thinking\"]", "[\"celebration\"]", "[\"listening\"]", "[\"confusion\"]", "[\"pointing\"]", "[\"pointing\"]", "[\"pointing\"]", "[\"uncertainty\"]", "[\"happiness\"]", "[\"comfort\",\"friendly\"]" };
    public string[] actionTargetRequired = new string[] { "none", "none", "none", "player", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "player" };
    public bool[] actionSpeechCompatible = new bool[] { true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true };
    public string[] actionLayers = new string[] { "upper_body", "upper_body", "upper_body", "upper_body", "full_body", "upper_body", "upper_body", "full_body", "upper_body", "upper_body", "upper_body", "upper_body", "upper_body", "upper_body", "upper_body", "upper_body" };
    public int[] actionDurationMs = new int[] { 1800, 1200, 1400, 1800, 2200, 2400, 2600, 2200, 2000, 1800, 1600, 1600, 1600, 1500, 2000, 2200 };
    public bool[] actionLoopable = new bool[16];
    public string[] actionMovement = new string[] { "allow", "allow", "allow", "allow", "block", "allow", "allow", "block", "allow", "allow", "allow", "allow", "allow", "allow", "allow", "allow" };
    public int[] actionPriority = new int[] { 30, 30, 30, 40, 60, 30, 30, 70, 20, 30, 30, 30, 30, 30, 40, 50 };
    public bool[] actionInterruptible = new bool[] { true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true };
    public int[] actionFadeInMs = new int[] { 120, 120, 120, 120, 150, 120, 120, 150, 120, 120, 120, 120, 120, 120, 120, 150 };
    public int[] actionFadeOutMs = new int[] { 180, 180, 180, 180, 220, 180, 180, 220, 180, 180, 180, 180, 180, 180, 180, 220 };

    [Header("表情目录（核心 0..3 固定，可追加扩展）")]
    public string[] expressionNames = new string[] { "neutral", "happy", "sad", "surprised", "huff", "cry", "wink", "idle" };
    public string[] expressionSemanticKeys = new string[] { "neutral", "happy", "sad", "surprised", "huff", "cry", "wink", "idle" };
    public string[] expressionDescriptionsZh = new string[] { "中立表情", "开心表情", "悲伤表情", "惊讶表情", "气鼓鼓的表情", "哭泣表情", "眨眼表情", "回到待机表情" };
    public int[] expressionDefaultDurationMs = new int[] { 0, 3000, 3000, 1800, 2500, 3000, 1200, 0 };
    public int[] expressionFadeMs = new int[] { 150, 150, 180, 120, 150, 180, 100, 150 };

    [Header("静态文案目录")]
    public string[] textPresetNames = new string[] { "welcome_home", "hello", "please_wait", "over_here", "goodbye" };
    public string[] textPresets = new string[] { "欢迎回家～", "你好呀！", "稍等一下…", "我在这里！", "再见，路上小心" };
    public int[] textPresetSeconds = new int[] { 5, 5, 5, 5, 5 };

    [Header("Inspector 明确发布的 Anchor")]
    public Transform[] anchorTransforms = new Transform[0];
    public string[] anchorSemanticKeys = new string[0];
    public string[] anchorDescriptionsZh = new string[0];
    public bool[] anchorHasYaw = new bool[0];
    public float[] anchorArrivalRadius = new float[0];
    [Tooltip("每项必须是 1..8 个语义标签组成的 JSON 数组")]
    public string[] anchorTagsJson = new string[0];

    // 旧场景/旧 Editor 脚本兼容字段；Animator Builder 仍会写入，但协议目录以 actionNames 为准。
    [HideInInspector] public string[] animNames = new string[0];

    private const int CMD_SET_MODE = 0x01;
    private const int CMD_GOTO_XZ = 0x02;
    private const int CMD_SET_SPEED = 0x03;
    private const int CMD_TURN_TO = 0x04;
    private const int CMD_LOOK_AT = 0x05;
    private const int CMD_PLAY_ANIM = 0x06;
    private const int CMD_STOP = 0x07;
    private const int CMD_TEXT_PRESET = 0x08;
    private const int CMD_RAY_SCAN = 0x09;
    private const int CMD_SET_RATE = 0x0A;
    private const int CMD_HEARTBEAT = 0x0B;
    private const int CMD_DISCOVER = 0x0C;
    private const int CMD_CLEAR_ESTOP = 0x0D;
    private const int CMD_STOP_ACTION = 0x0E;
    private const int CMD_SNAPSHOT_REQUEST = 0x0F;
    private const int CMD_SET_TARGET = 0x10;
    private const int CMD_LOOK_AT_XYZ = 0x11;
    private const int CMD_SET_EXPRESSION = 0x12;
    private const int CMD_TEXT_BEGIN = 0x13;
    private const int CMD_TEXT_COMMIT = 0x14;
    private const int CMD_SPEECH_CUE = 0x15;
    private const int CMD_SET_CONTROL_MODE = 0x16;
    private const int CMD_ESTOP = 0x7F;

    public const int STATE_UNHANDSHAKEN = 0;
    public const int STATE_SAFE_IDLE = 1;
    public const int STATE_EXTERNAL = 2;
    public const int STATE_MOVING = 3;
    public const int STATE_ACTION = 4;
    public const int STATE_ESTOP = 5;

    private const int Q14 = 16383;
    private const int DupSlots = 64;
    private const int LaneMovement = 0;
    private const int LaneLook = 1;
    private const int LaneAction = 2;
    private const int LaneExpression = 3;

    private int _p0hi, _p0lo, _p1hi, _p1lo, _p2hi, _p2lo, _p3, _p4, _p5;
    private int _session;
    private int _state = STATE_UNHANDSHAKEN;
    private int _targetSlot = -1;
    private bool _configValid;
    private NekoNpcSync _sync;

    private int[] _dupSession = new int[DupSlots];
    private int[] _dupSeq = new int[DupSlots];
    private int[] _dupCmd = new int[DupSlots];
    private int[] _dupHash = new int[DupSlots];
    private float[] _dupTime = new float[DupSlots];
    private bool[] _dupOk = new bool[DupSlots];
    private string[] _dupErr = new string[DupSlots];
    private int[] _dupState = new int[DupSlots];
    private int _dupCursor;
    private int _dupHit = -1;

    private float _lastHeartbeat = -1f;
    private bool _watchdogArmed;

    private string[] _opId = new string[4];
    private string[] _opKind = new string[4];
    private int[] _opSeq = new int[4];
    private int[] _opHash = new int[4];
    private float[] _opStartedAt = new float[4];

    private int _currentActionId = -1;
    private int _currentActionSeq;
    private bool _currentActionLoop;
    private int _preparedActionId = -1;
    private int _preparedActionSeq;
    private bool _preparedActionLoop;
    private int _preparedExpressionId = -1;
    private int _preparedExpressionWeight;
    private int _preparedExpressionDurationSec;
    private int _preparedPost;
    private string _decodedText;
    private int[] _seenActionIdPlusOne = new int[16384];
    private bool[] _seenActionLoop = new bool[16384];
    private int[] _seenActionGeneration = new int[16384];
    private int _historyGeneration = 1;
    private bool _cancelMovementForPreparedAction;

    private int _textTransferSeq;
    private int _textRawLength;
    private int _textCrc16;
    private int _textDisplaySeconds;
    private int _textDisplayRaw;
    private float _textDeadline;
    private int[] _textPacked = new int[439];
    private int _textPackedCount;
    private int _textPackedExpected;
    private bool[] _seenTextTransfer = new bool[16384];
    private int[] _seenTextGeneration = new int[16384];
    private string _pendingText;
    private int _pendingTextTransferSeq;
    private int _pendingTextRawLength;
    private int _pendingTextCrc16;
    private int _pendingTextDisplaySeconds;
    [System.NonSerialized] public int textInvalidEvents;

    [System.NonSerialized] public int commandsHandled;
    [System.NonSerialized] public int commandsRejected;

    void Start()
    {
        watchdogSec = 3f;
        dupWindowSec = 5f;
        midiChannel = Mathf.Clamp(midiChannel, 0, 15);
        for (int i = 0; i < DupSlots; i++)
        {
            _dupSession[i] = -1; _dupSeq[i] = -1; _dupCmd[i] = -1; _dupHash[i] = -1;
            _dupTime[i] = -999f; _dupErr[i] = null; _dupState[i] = STATE_UNHANDSHAKEN;
        }
        if (locomotion != null && locomotion.npcRoot != null) _sync = locomotion.npcRoot.GetComponent<NekoNpcSync>();
        if (nameplate != null && nameplate.telemetry == null) nameplate.telemetry = telemetry;
        if (nameplate == null || nameplate.bubbleText == null) { enableTextPreset = false; enableTextUtf8 = false; }
        _configValid = ValidateConfiguration();
    }

    private bool ValidateConfiguration()
    {
        string detail = null;
        if (telemetry == null || locomotion == null || perception == null || nameplate == null) detail = "YUI v1.1 依赖引用不完整";
        else if (telemetry.npcId != "yui") detail = "npcId 必须固定为 yui";
        else if (telemetry.worldId == null || telemetry.worldId.Length < 1 || telemetry.worldId.Length > 64) detail = "worldId 必须为 1..64 字符";
        else if (driverClaimCode < 0 || driverClaimCode > Q14) detail = "driverClaimCode 必须为 0..16383";
        else if (catalogRevision < 1) detail = "catalogRevision 必须大于 0";
        else if (_sync == null) detail = "NPC 根缺少 NekoNpcSync";
        else if ((enableGoto || enableFollow || enableWander) && !locomotion.HasNavMeshAgent()) detail = "导航能力要求 NavMeshAgent";
        else if (enableWander && !locomotion.HasWanderWaypoints()) detail = "wander 至少需要两个 Inspector 航点";
        else if (wireBoundsMin.x > boundsMin.x - 1f || wireBoundsMin.y > boundsMin.y - 1f || wireBoundsMin.z > boundsMin.z - 1f
              || wireBoundsMax.x < boundsMax.x + 1f || wireBoundsMax.y < boundsMax.y + 1f || wireBoundsMax.z < boundsMax.z + 1f) detail = "wireBounds 必须在 activityBounds 每侧外扩至少 1m";
        else if (boundsMin.x >= boundsMax.x || boundsMin.y >= boundsMax.y || boundsMin.z >= boundsMax.z) detail = "activityBounds min 必须小于 max";
        else if (enableActions) detail = ValidateActionCatalog();
        if (detail == null && enableExpressions) detail = ValidateExpressionCatalog();
        if (detail == null && enableTextPreset) detail = ValidateTextPresetCatalog();
        if (detail == null && enableAnchors) detail = ValidateAnchorCatalog();
        if (detail != null)
        {
            if (telemetry != null) telemetry.EmitProtocolError("catalog_invalid", "safety", true, -1, detail);
            return false;
        }
        return true;
    }

    private string ValidateActionCatalog()
    {
        if (actionNames == null || actionNames.Length < 16 || actionNames.Length > 127) return "action 目录必须包含核心 0..15";
        int n = actionNames.Length;
        if (actionSemanticKeys == null || actionSemanticKeys.Length != n || actionDescriptionsZh == null || actionDescriptionsZh.Length != n
            || actionIntentTagsJson == null || actionIntentTagsJson.Length != n || actionTargetRequired == null || actionTargetRequired.Length != n
            || actionSpeechCompatible == null || actionSpeechCompatible.Length != n || actionLayers == null || actionLayers.Length != n
            || actionDurationMs == null || actionDurationMs.Length != n || actionLoopable == null || actionLoopable.Length != n
            || actionMovement == null || actionMovement.Length != n || actionPriority == null || actionPriority.Length != n
            || actionInterruptible == null || actionInterruptible.Length != n || actionFadeInMs == null || actionFadeInMs.Length != n
            || actionFadeOutMs == null || actionFadeOutMs.Length != n) return "action 目录字段长度不一致";
        string[] core = new string[] { "greet", "agree_nod", "disagree_shake_head", "greet_wave", "apologize_bow", "explain", "think", "celebrate", "listen", "confused", "point_left", "point_right", "point_forward", "shrug", "laugh", "comfort" };
        for (int i = 0; i < n; i++)
        {
            if (!ValidKey(actionNames[i]) || !ValidKey(actionSemanticKeys[i])) return "action name/semantic_key 非法";
            if (i < 16 && actionSemanticKeys[i] != core[i]) return "核心 action semantic_key 不可更改";
            int bytes = Utf8ByteCount(actionDescriptionsZh[i]);
            if (bytes < 1 || bytes > 80 || actionDurationMs[i] < 1 || actionDurationMs[i] > 600000) return "action 描述或时长非法";
            if (actionTargetRequired[i] != "none" && actionTargetRequired[i] != "player" && actionTargetRequired[i] != "point") return "action target_required 非法";
            if (actionLayers[i] != "upper_body" && actionLayers[i] != "full_body") return "action layer 非法";
            if (actionMovement[i] != "allow" && actionMovement[i] != "block") return "action movement 非法";
            if (actionLayers[i] == "full_body" && actionMovement[i] != "block") return "full_body 动作必须阻断移动";
            if (actionPriority[i] < 0 || actionPriority[i] > 100 || actionFadeInMs[i] < 0 || actionFadeInMs[i] > 5000 || actionFadeOutMs[i] < 0 || actionFadeOutMs[i] > 5000) return "action priority/fade 非法";
        }
        return null;
    }

    private string ValidateExpressionCatalog()
    {
        if (expressionNames == null || expressionNames.Length < 4 || expressionNames.Length > 127) return "expression 目录必须包含核心 0..3";
        int n = expressionNames.Length;
        if (expressionSemanticKeys == null || expressionSemanticKeys.Length != n || expressionDescriptionsZh == null || expressionDescriptionsZh.Length != n
            || expressionDefaultDurationMs == null || expressionDefaultDurationMs.Length != n || expressionFadeMs == null || expressionFadeMs.Length != n) return "expression 目录字段长度不一致";
        string[] core = new string[] { "neutral", "happy", "sad", "surprised" };
        for (int i = 0; i < n; i++)
        {
            if (!ValidKey(expressionNames[i]) || !ValidKey(expressionSemanticKeys[i]) || (i < 4 && expressionSemanticKeys[i] != core[i])) return "expression 核心语义非法";
            int bytes = Utf8ByteCount(expressionDescriptionsZh[i]);
            if (bytes < 1 || bytes > 80 || expressionDefaultDurationMs[i] < 0 || expressionDefaultDurationMs[i] > 600000 || expressionFadeMs[i] < 0 || expressionFadeMs[i] > 5000) return "expression 元数据非法";
        }
        return null;
    }

    private string ValidateTextPresetCatalog()
    {
        int n = textPresets == null ? 0 : textPresets.Length;
        if (n > 127 || textPresetNames == null || textPresetNames.Length != n || textPresetSeconds == null || textPresetSeconds.Length != n) return "text_preset 目录字段长度不一致";
        for (int i = 0; i < n; i++) if (!ValidKey(textPresetNames[i]) || Utf8ByteCount(textPresets[i]) < 1 || Utf8ByteCount(textPresets[i]) > 384 || textPresetSeconds[i] < 1 || textPresetSeconds[i] > 127) return "text_preset 元数据非法";
        return null;
    }

    private string ValidateAnchorCatalog()
    {
        int n = anchorTransforms == null ? 0 : anchorTransforms.Length;
        if (n < 3 || n > 127 || anchorSemanticKeys == null || anchorSemanticKeys.Length != n || anchorDescriptionsZh == null || anchorDescriptionsZh.Length != n
            || anchorHasYaw == null || anchorHasYaw.Length != n || anchorArrivalRadius == null || anchorArrivalRadius.Length != n || anchorTagsJson == null || anchorTagsJson.Length != n) return "anchor 目录至少三项且字段长度一致";
        for (int i = 0; i < n; i++)
        {
            if (anchorTransforms[i] == null || !ValidKey(anchorSemanticKeys[i]) || Utf8ByteCount(anchorDescriptionsZh[i]) < 1 || Utf8ByteCount(anchorDescriptionsZh[i]) > 80) return "anchor 字段非法";
            if (!InsideActivityBounds(anchorTransforms[i].position) || anchorArrivalRadius[i] < 0.1f || anchorArrivalRadius[i] > 2f) return "anchor 越界或到达半径非法";
            NavMeshHit hit;
            if (!NavMesh.SamplePosition(anchorTransforms[i].position, out hit, 0.5f, NavMesh.AllAreas)) return "anchor 未落在 NavMesh";
        }
        return null;
    }

    private bool ValidKey(string value)
    {
        if (value == null || value.Length < 1 || value.Length > 32) return false;
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
        }
        return true;
    }

    public override void MidiControlChange(int channel, int number, int value)
    {
        if (channel == 2)
        {
            if (!enableTextUtf8) return;
            if (number != 29 || _textTransferSeq == 0 || _textPackedCount >= _textPackedExpected) { textInvalidEvents++; return; }
            _textPacked[_textPackedCount++] = value & 127;
            return;
        }
        if (channel != midiChannel) return;
        if (number == 20) _p0hi = value; else if (number == 21) _p0lo = value;
        else if (number == 22) _p1hi = value; else if (number == 23) _p1lo = value;
        else if (number == 24) _p2hi = value; else if (number == 25) _p2lo = value;
        else if (number == 26) _p3 = value; else if (number == 27) _p4 = value; else if (number == 28) _p5 = value;
    }

    public override void MidiNoteOff(int channel, int number, int velocity) { }

    public override void MidiNoteOn(int channel, int number, int velocity)
    {
        int cmd = number; int seq = velocity;
        // 冻结规范要求 ESTOP 绕过普通控制通道过滤，任何 MIDI 通道都能紧急制动。
        if (cmd == CMD_ESTOP) { HandleEstop(seq); return; }
        if (channel != midiChannel) return;
        if (seq == 0) return;
        int p0 = _p0hi * 128 + _p0lo;
        int p1 = _p1hi * 128 + _p1lo;
        int p2 = _p2hi * 128 + _p2lo;
        int hash = RequestHash(cmd, seq, _p0hi, _p0lo, _p1hi, _p1lo, _p2hi, _p2lo, _p3, _p4, _p5);
        int requestSession = cmd == CMD_DISCOVER ? p0 + (p1 << 14) : _session;

        int duplicate = FindDuplicate(requestSession, seq, cmd, hash);
        if (duplicate == 1)
        {
            Ack(seq, cmd, hash, _dupOk[_dupHit], _dupErr[_dupHit], true, _dupState[_dupHit]);
            if (cmd == CMD_DISCOVER && _dupOk[_dupHit]) SendHelloAndCatalog(false);
            return;
        }
        if (duplicate == 2) { Reject(requestSession, seq, cmd, hash, "seq_conflict"); return; }
        if (!IsKnownCommand(cmd)) { Reject(requestSession, seq, cmd, hash, "unknown_cmd"); return; }
        if (cmd == CMD_DISCOVER) { HandleDiscover(seq, p0, p1, p2, hash, requestSession); return; }
        if (_session == 0) { Reject(requestSession, seq, cmd, hash, "not_handshaken"); return; }
        if (!IsLocalClaimedDriver()) { Reject(requestSession, seq, cmd, hash, "not_driver"); return; }
        if (!HasAllOwnership()) { Reject(requestSession, seq, cmd, hash, "not_owner"); return; }
        string err = StateError(cmd);
        if (err != null) { Reject(requestSession, seq, cmd, hash, err); return; }
        if (MissingCapability(cmd, _p3)) { Reject(requestSession, seq, cmd, hash, "unsupported_capability"); return; }
        err = ValidateRegisters(cmd, p0, p1, p2, _p3, _p4, _p5);
        if (err != null) { Reject(requestSession, seq, cmd, hash, err); return; }

        int post = 0;
        bool cancelActionForMovement = false;
        if (cmd == CMD_SET_MODE)
        {
            if (_p3 == 0) { locomotion.SetMode(NekoNpcLocomotion.MODE_IDLE); SetState(StateAfterMovement()); post = 10; }
            else if (_p3 == 1)
            {
                err = PrepareMovementCommand(); if (err == null) { cancelActionForMovement = _currentActionId >= 0 && CurrentActionBlocksMovement(); err = locomotion.StartFollow(_targetSlot); }
                if (err == null) { SetState(STATE_MOVING); post = 11; }
            }
            else if (_p3 == 2) { if (locomotion.GetMode() != NekoNpcLocomotion.MODE_GOTO) err = "invalid_state"; }
            else
            {
                err = PrepareMovementCommand(); if (err == null) { cancelActionForMovement = _currentActionId >= 0 && CurrentActionBlocksMovement(); err = locomotion.StartWander(); }
                if (err == null) { SetState(STATE_MOVING); post = 12; }
            }
        }
        else if (cmd == CMD_GOTO_XZ)
        {
            err = PrepareMovementCommand();
            if (err == null)
            {
                cancelActionForMovement = _currentActionId >= 0 && CurrentActionBlocksMovement();
                float x = Dequant(p0, wireBoundsMin.x, wireBoundsMax.x); float z = Dequant(p1, wireBoundsMin.z, wireBoundsMax.z);
                Vector3 point = new Vector3(x, locomotion.npcRoot.position.y, z);
                if (!InsideActivityBounds(point)) err = "target_out_of_bounds";
                else err = locomotion.Goto(x, z, (_p4 & 1) != 0 ? (p2 / (float)Q14) * 360f : -1f, (_p3 / 127f) * locomotion.maxSpeed, seq);
                if (err == null) { SetState(STATE_MOVING); post = 4; }
            }
        }
        else if (cmd == CMD_SET_SPEED) locomotion.SetSpeed((_p3 / 127f) * locomotion.maxSpeed);
        else if (cmd == CMD_TURN_TO)
        {
            err = PrepareMovementCommand();
            if (err == null) { cancelActionForMovement = _currentActionId >= 0 && CurrentActionBlocksMovement(); locomotion.TurnTo((p0 / (float)Q14) * 360f); SetState(STATE_MOVING); post = 5; }
        }
        else if (cmd == CMD_LOOK_AT)
        {
            if (_p3 != 127 && perception.PlayerOfSlot(_p3) == null) err = "slot_unknown";
            else { locomotion.LookAt(_p3 == 127 ? -1 : _p3); post = _p3 == 127 ? 7 : 6; }
        }
        else if (cmd == CMD_PLAY_ANIM) { err = PrepareAction(p0, _p3, (_p4 & 1) != 0); post = _preparedPost; }
        else if (cmd == CMD_STOP)
        {
            locomotion.Stop(); _targetSlot = -1; SetState(_state == STATE_SAFE_IDLE ? STATE_SAFE_IDLE : STATE_EXTERNAL); post = 8;
        }
        else if (cmd == CMD_TEXT_PRESET)
        {
            if (_p3 == 127) nameplate.ClearBubbleWithReason("explicit");
            else if (textPresets == null || _p3 >= textPresets.Length) err = "text_preset_not_found";
            else { int sec = _p4 == 0 ? textPresetSeconds[_p3] : _p4; nameplate.ShowPreset(textPresets[_p3], sec); }
        }
        else if (cmd == CMD_RAY_SCAN) post = 1;
        else if (cmd == CMD_SET_RATE) { locomotion.SetStateRate(_p3); perception.SetPoseRate(_p4); }
        else if (cmd == CMD_HEARTBEAT) { _lastHeartbeat = Time.timeSinceLevelLoad; _watchdogArmed = true; post = 2; }
        else if (cmd == CMD_CLEAR_ESTOP) { locomotion.ClearEstop(); SetState(STATE_SAFE_IDLE); }
        else if (cmd == CMD_STOP_ACTION)
        {
            if (_currentActionId >= 0 && p0 != 0 && p0 != _currentActionSeq) err = "action_not_found";
            else { SetState(StateAfterAction()); post = 13; }
        }
        else if (cmd == CMD_SNAPSHOT_REQUEST) post = 3;
        else if (cmd == CMD_SET_TARGET)
        {
            if (_p3 != 127 && perception.PlayerOfSlot(_p3) == null) err = "slot_unknown";
            else _targetSlot = _p3 == 127 ? -1 : _p3;
        }
        else if (cmd == CMD_LOOK_AT_XYZ)
        {
            Vector3 point = new Vector3(Dequant(p0, wireBoundsMin.x, wireBoundsMax.x), Dequant(p1, wireBoundsMin.y, wireBoundsMax.y), Dequant(p2, wireBoundsMin.z, wireBoundsMax.z));
            if (!InsideActivityBounds(point)) err = "target_out_of_bounds";
            else { locomotion.LookAtPoint(point, _p3 / 127f, (float)_p4, (_p5 & 1) != 0); post = _p3 == 0 ? 7 : 6; }
        }
        else if (cmd == CMD_SET_EXPRESSION) { err = PrepareExpression(_p3, _p4, _p5); post = _preparedPost; }
        else if (cmd == CMD_TEXT_BEGIN) err = BeginTextTransaction(p0, p1, p2 | (_p3 << 14), _p4);
        else if (cmd == CMD_TEXT_COMMIT) { err = ValidateTextCommit(p0, p1, p2 | (_p3 << 14), _p4); if (err == null) post = 20; }
        else if (cmd == CMD_SET_CONTROL_MODE)
        {
            if (_p3 == 0) { locomotion.WatchdogIdle(); nameplate.ClearBubbleWithReason("control_safe_idle"); SetState(STATE_SAFE_IDLE); post = 9; }
            else if (_state == STATE_SAFE_IDLE) SetState(STATE_EXTERNAL);
        }
        else err = "unsupported_capability";

        if (err != null) { Reject(requestSession, seq, cmd, hash, err); return; }
        Remember(requestSession, seq, cmd, hash, true, null, _state);
        Ack(seq, cmd, hash, true, null, false, _state);
        commandsHandled++;

        if (cancelActionForMovement) CancelAction("movement");
        if (post == 1) perception.RayScan(_p3, seq);
        else if (post == 2) telemetry.Emit("sys.pong", "\"seq\":" + seq + ",\"watchdog_age_ms\":0");
        else if (post == 3) SendSnapshot(seq);
        else if (post == 4) BeginOperation(LaneMovement, "goto", seq, hash, -1);
        else if (post == 5) BeginOperation(LaneMovement, "turn", seq, hash, -1);
        else if (post == 6) BeginOperation(LaneLook, "look", seq, hash, cmd == CMD_LOOK_AT_XYZ && _p4 > 0 ? _p4 * 1000 : -1);
        else if (post == 7) { BeginOperation(LaneLook, "look", seq, hash, 0); CompleteOperation(LaneLook, "cleared"); }
        else if (post == 8) CancelAllOperations("explicit_stop");
        else if (post == 9) CancelAllOperations("control_safe_idle");
        else if (post == 10) CancelOperation(LaneMovement, "explicit_stop");
        else if (post == 11) BeginOperation(LaneMovement, "follow", seq, hash, -1);
        else if (post == 12) BeginOperation(LaneMovement, "wander", seq, hash, -1);
        else if (post == 13) CancelAction("explicit_stop");
        else if (post == 14) StartPreparedAction(seq, hash);
        else if (post == 15) { CancelAction("replaced"); StartPreparedAction(seq, hash); }
        else if (post == 16) StartPreparedExpression(seq, hash);
        else if (post == 17) { CancelOperation(LaneExpression, "replaced"); StartPreparedExpression(seq, hash); }
        else if (post == 18) { BeginOperation(LaneExpression, "expression", seq, hash, 0); locomotion.ClearExpression(); CompleteOperation(LaneExpression, "cleared"); }
        else if (post == 20) CommitPreparedText();
    }

    private string PrepareMovementCommand()
    {
        if (_currentActionId < 0 || !CurrentActionBlocksMovement()) return null;
        return actionInterruptible[_currentActionId] ? null : "action_busy";
    }

    private bool CurrentActionBlocksMovement()
    {
        return _currentActionId >= 0 && _currentActionId < actionMovement.Length && actionMovement[_currentActionId] == "block";
    }

    private string PrepareAction(int actionSeq, int actionId, bool loop)
    {
        _preparedPost = 0;
        if (actionId < 0 || actionNames == null || actionId >= actionNames.Length) return "action_not_found";
        if (loop && !actionLoopable[actionId]) return "invalid_param";
        if (actionTargetRequired[actionId] == "player" && (_targetSlot < 0 || perception.PlayerOfSlot(_targetSlot) == null)) return "slot_unknown";
        if (actionTargetRequired[actionId] == "point") return "invalid_param";

        int seen = _seenActionGeneration[actionSeq] == _historyGeneration ? _seenActionIdPlusOne[actionSeq] : 0;
        if (seen > 0)
        {
            if (seen - 1 != actionId || _seenActionLoop[actionSeq] != loop) return "action_seq_conflict";
            return null; // 相同 actionSeq + 元数据是协议级 no-op。
        }

        if (_currentActionId >= 0)
        {
            bool mayReplace = actionInterruptible[_currentActionId]
                && (actionPriority[actionId] > actionPriority[_currentActionId]
                    || actionPriority[actionId] == actionPriority[_currentActionId]);
            if (!mayReplace) return "action_busy";
            _preparedPost = 15;
        }
        else _preparedPost = 14;

        _seenActionIdPlusOne[actionSeq] = actionId + 1;
        _seenActionLoop[actionSeq] = loop;
        _seenActionGeneration[actionSeq] = _historyGeneration;
        _preparedActionId = actionId;
        _preparedActionSeq = actionSeq;
        _preparedActionLoop = loop;

        _cancelMovementForPreparedAction = actionMovement[actionId] == "block" && _opId[LaneMovement] != null;
        if (_state != STATE_MOVING || actionMovement[actionId] == "block") SetState(STATE_ACTION);
        return null;
    }

    private void StartPreparedAction(int requestSeq, int hash)
    {
        int id = _preparedActionId;
        if (id < 0) return;
        if (_cancelMovementForPreparedAction)
        {
            locomotion.SetMode(NekoNpcLocomotion.MODE_IDLE);
            CancelOperation(LaneMovement, "movement");
        }
        _currentActionId = id;
        _currentActionSeq = _preparedActionSeq;
        _currentActionLoop = _preparedActionLoop;
        BeginOperation(LaneAction, "action", requestSeq, hash, _currentActionLoop ? -1 : actionDurationMs[id]);
        locomotion.PlayAction(id, _currentActionSeq, _currentActionLoop, actionDurationMs[id], actionLayers[id]);
        telemetry.Emit("npc.action_started", ActionCommonBody(id, _currentActionSeq, LaneAction)
            + ",\"loop\":" + telemetry.B(_currentActionLoop)
            + ",\"started_at_server_ms\":" + locomotion.GetActionStartedServerMs());
        _preparedActionId = -1; _preparedActionSeq = 0; _preparedActionLoop = false; _cancelMovementForPreparedAction = false;
    }

    private string PrepareExpression(int expressionId, int weight, int durationSec)
    {
        _preparedPost = 0;
        if (expressionId == 127) { _preparedPost = 18; return null; }
        if (expressionId < 0 || expressionNames == null || expressionId >= expressionNames.Length) return "expression_not_found";
        _preparedExpressionId = expressionId;
        _preparedExpressionWeight = weight;
        _preparedExpressionDurationSec = durationSec;
        _preparedPost = _opId[LaneExpression] == null ? 16 : 17;
        return null;
    }

    private void StartPreparedExpression(int requestSeq, int hash)
    {
        int expected = _preparedExpressionDurationSec > 0 ? _preparedExpressionDurationSec * 1000 : -1;
        BeginOperation(LaneExpression, "expression", requestSeq, hash, expected);
        locomotion.SetExpression(_preparedExpressionId, _preparedExpressionWeight / 127f, expected < 0 ? 0 : expected);
        _preparedExpressionId = -1; _preparedExpressionWeight = 0; _preparedExpressionDurationSec = 0;
    }

    private void HandleDiscover(int seq, int p0, int p1, int claim, int hash, int requestSession)
    {
        if (!_configValid) { Reject(requestSession, seq, CMD_DISCOVER, hash, "catalog_invalid"); return; }
        if (requestSession <= 0) { Reject(requestSession, seq, CMD_DISCOVER, hash, "invalid_param"); return; }
        if (claim != driverClaimCode || telemetry == null || !telemetry.IsDisplayNameAllowed()) { Reject(requestSession, seq, CMD_DISCOVER, hash, "driver_auth_failed"); return; }
        if (_p3 != 0 || _p4 != 0 || _p5 != 0) { Reject(requestSession, seq, CMD_DISCOVER, hash, "reserved_bits"); return; }
        VRCPlayerApi local = Networking.LocalPlayer;
        if (local == null) { Reject(requestSession, seq, CMD_DISCOVER, hash, "ownership_failed"); return; }
        int oldDriver = _sync.GetDriverPid();
        VRCPlayerApi oldPlayer = oldDriver > 0 ? VRCPlayerApi.GetPlayerById(oldDriver) : null;
        if (oldDriver > 0 && oldDriver != local.playerId && oldPlayer != null && oldPlayer.IsValid()) { Reject(requestSession, seq, CMD_DISCOVER, hash, "session_conflict"); return; }
        if (!ClaimAllOwnership(local)) { Reject(requestSession, seq, CMD_DISCOVER, hash, "ownership_failed"); return; }

        int previous = _session;
        bool first = previous == 0;
        bool reset = !first && previous != requestSession;
        bool preserveEstop = _state == STATE_ESTOP;
        if (first || reset)
        {
            _watchdogArmed = false;
            _lastHeartbeat = Time.timeSinceLevelLoad;
            if (!preserveEstop) locomotion.Stop();
            nameplate.ClearBubbleWithReason("session_reset");
            _targetSlot = -1;
            ClearDedupe();
            ClearSessionHistories();
            perception.RebuildSlots();
            SetState(preserveEstop ? STATE_ESTOP : STATE_SAFE_IDLE);
        }
        _session = requestSession;
        _sync.SetAuthority(local.playerId, _session, _state);
        telemetry.SetSession(_session);
        Remember(_session, seq, CMD_DISCOVER, hash, true, null, _state);
        Ack(seq, CMD_DISCOVER, hash, true, null, false, _state);
        if (reset) CancelAllOperations("session_reset");
        telemetry.Emit("sys.session", "\"previous_session\":" + previous + ",\"new_session\":" + _session
            + ",\"driver_pid\":" + local.playerId + ",\"reset\":" + telemetry.B(reset) + ",\"estop_preserved\":" + telemetry.B(preserveEstop));
        SendHelloAndCatalog(first || reset);
        commandsHandled++;
    }

    private void HandleEstop(int seq)
    {
        if (locomotion != null) locomotion.Estop();
        if (nameplate != null) nameplate.ClearBubbleWithReason("estop");
        AbortTextTransaction();
        SetState(STATE_ESTOP);
        int hash = RequestHash(CMD_ESTOP, seq, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        if (seq > 0)
        {
            int duplicate = FindDuplicate(_session, seq, CMD_ESTOP, hash);
            if (duplicate == 1) Ack(seq, CMD_ESTOP, hash, true, null, true, STATE_ESTOP);
            else { Remember(_session, seq, CMD_ESTOP, hash, true, null, STATE_ESTOP); Ack(seq, CMD_ESTOP, hash, true, null, false, STATE_ESTOP); }
        }
        CancelAllOperations("estop");
        commandsHandled++;
    }

    private void ClearSessionHistories()
    {
        _historyGeneration++;
        if (_historyGeneration <= 0)
        {
            _historyGeneration = 1;
            for (int i = 1; i <= Q14; i++) { _seenActionGeneration[i] = 0; _seenTextGeneration[i] = 0; }
        }
        AbortTextTransaction();
        _currentActionId = -1; _currentActionSeq = 0; _currentActionLoop = false;
    }

    private void SendHelloAndCatalog(bool includePlayers)
    {
        int actionCount = enableActions && actionNames != null ? actionNames.Length : 0;
        int expressionCount = enableExpressions && expressionNames != null ? expressionNames.Length : 0;
        int presetCount = enableTextPreset && textPresets != null ? textPresets.Length : 0;
        int anchorCount = enableAnchors && anchorTransforms != null ? anchorTransforms.Length : 0;
        string body = "\"world_name\":" + telemetry.J(worldName)
            + ",\"wire_bounds\":[" + telemetry.F2(wireBoundsMin.x) + "," + telemetry.F2(wireBoundsMin.y) + "," + telemetry.F2(wireBoundsMin.z) + "," + telemetry.F2(wireBoundsMax.x) + "," + telemetry.F2(wireBoundsMax.y) + "," + telemetry.F2(wireBoundsMax.z) + "]"
            + ",\"activity_bounds\":[" + telemetry.F2(boundsMin.x) + "," + telemetry.F2(boundsMin.y) + "," + telemetry.F2(boundsMin.z) + "," + telemetry.F2(boundsMax.x) + "," + telemetry.F2(boundsMax.y) + "," + telemetry.F2(boundsMax.z) + "]"
            + ",\"max_speed\":" + telemetry.F2(locomotion.maxSpeed) + ",\"watchdog_ms\":3000"
            + ",\"caps\":" + CapsJson() + ",\"cap_bits\":" + CapabilityBits() + ",\"catalog_rev\":" + catalogRevision
            + ",\"catalog_counts\":{\"action\":" + actionCount + ",\"expression\":" + expressionCount + ",\"text_preset\":" + presetCount + ",\"anchor\":" + anchorCount + "}";
        telemetry.Emit("sys.hello", body);
        SendActionCatalog(); SendExpressionCatalog(); SendTextCatalog(); SendAnchorCatalog();
        if (includePlayers) perception.DumpSlots();
    }

    private void SendEmptyCatalog(string kind)
    {
        telemetry.Emit("sys.catalog", "\"catalog_rev\":" + catalogRevision + ",\"kind\":" + telemetry.J(kind) + ",\"page\":1,\"pages\":1,\"items\":[]");
    }

    private void SendActionCatalog()
    {
        int count = enableActions && actionNames != null ? actionNames.Length : 0;
        if (count == 0) { SendEmptyCatalog("action"); return; }
        for (int i = 0; i < count; i++)
        {
            string item = "{\"id\":" + i + ",\"name\":" + telemetry.J(actionNames[i]) + ",\"semantic_key\":" + telemetry.J(actionSemanticKeys[i])
                + ",\"description_zh\":" + telemetry.J(actionDescriptionsZh[i]) + ",\"intent_tags\":" + actionIntentTagsJson[i]
                + ",\"target_required\":" + telemetry.J(actionTargetRequired[i]) + ",\"speech_compatible\":" + telemetry.B(actionSpeechCompatible[i])
                + ",\"layer\":" + telemetry.J(actionLayers[i]) + ",\"duration_ms\":" + actionDurationMs[i] + ",\"loopable\":" + telemetry.B(actionLoopable[i])
                + ",\"movement\":" + telemetry.J(actionMovement[i]) + ",\"priority\":" + actionPriority[i] + ",\"interruptible\":" + telemetry.B(actionInterruptible[i])
                + ",\"fade_in_ms\":" + actionFadeInMs[i] + ",\"fade_out_ms\":" + actionFadeOutMs[i] + "}";
            telemetry.Emit("sys.catalog", "\"catalog_rev\":" + catalogRevision + ",\"kind\":\"action\",\"page\":" + (i + 1) + ",\"pages\":" + count + ",\"items\":[" + item + "]");
        }
    }

    private void SendExpressionCatalog()
    {
        int count = enableExpressions && expressionNames != null ? expressionNames.Length : 0;
        if (count == 0) { SendEmptyCatalog("expression"); return; }
        for (int i = 0; i < count; i++)
        {
            string item = "{\"id\":" + i + ",\"name\":" + telemetry.J(expressionNames[i]) + ",\"semantic_key\":" + telemetry.J(expressionSemanticKeys[i])
                + ",\"description_zh\":" + telemetry.J(expressionDescriptionsZh[i]) + ",\"default_duration_ms\":" + expressionDefaultDurationMs[i] + ",\"fade_ms\":" + expressionFadeMs[i] + "}";
            telemetry.Emit("sys.catalog", "\"catalog_rev\":" + catalogRevision + ",\"kind\":\"expression\",\"page\":" + (i + 1) + ",\"pages\":" + count + ",\"items\":[" + item + "]");
        }
    }

    private void SendTextCatalog()
    {
        int count = enableTextPreset && textPresets != null ? textPresets.Length : 0;
        if (count == 0) { SendEmptyCatalog("text_preset"); return; }
        for (int i = 0; i < count; i++)
        {
            string item = "{\"id\":" + i + ",\"name\":" + telemetry.J(textPresetNames[i]) + ",\"text\":" + telemetry.J(textPresets[i]) + ",\"default_display_seconds\":" + textPresetSeconds[i] + "}";
            telemetry.Emit("sys.catalog", "\"catalog_rev\":" + catalogRevision + ",\"kind\":\"text_preset\",\"page\":" + (i + 1) + ",\"pages\":" + count + ",\"items\":[" + item + "]");
        }
    }

    private void SendAnchorCatalog()
    {
        int count = enableAnchors && anchorTransforms != null ? anchorTransforms.Length : 0;
        if (count == 0) { SendEmptyCatalog("anchor"); return; }
        for (int i = 0; i < count; i++)
        {
            float yaw = anchorHasYaw[i] ? Mathf.Repeat(anchorTransforms[i].eulerAngles.y, 360f) : 0f;
            string item = "{\"id\":" + i + ",\"semantic_key\":" + telemetry.J(anchorSemanticKeys[i]) + ",\"description_zh\":" + telemetry.J(anchorDescriptionsZh[i])
                + ",\"pos\":" + telemetry.Vec3(anchorTransforms[i].position) + ",\"yaw\":" + telemetry.F1(yaw) + ",\"has_yaw\":" + telemetry.B(anchorHasYaw[i])
                + ",\"arrival_radius\":" + telemetry.F2(anchorArrivalRadius[i]) + ",\"tags\":" + anchorTagsJson[i] + "}";
            telemetry.Emit("sys.catalog", "\"catalog_rev\":" + catalogRevision + ",\"kind\":\"anchor\",\"page\":" + (i + 1) + ",\"pages\":" + count + ",\"items\":[" + item + "]");
        }
    }

    private void SendSnapshot(int requestSeq)
    {
        int playerPages = perception.SnapshotPageCount();
        int parts = playerPages + 4; int part = 1;
        string watchdogAge = _lastHeartbeat < 0f ? "null" : Mathf.Max(0, Mathf.RoundToInt((Time.timeSinceLevelLoad - _lastHeartbeat) * 1000f)).ToString();
        int actionCount = enableActions && actionNames != null ? actionNames.Length : 0;
        int expressionCount = enableExpressions && expressionNames != null ? expressionNames.Length : 0;
        int presetCount = enableTextPreset && textPresets != null ? textPresets.Length : 0;
        int anchorCount = enableAnchors && anchorTransforms != null ? anchorTransforms.Length : 0;
        string sessionData = "{\"driver_pid\":" + CurrentDriverPidJson() + ",\"control_state\":" + telemetry.J(StateName(_state))
            + ",\"watchdog_age_ms\":" + watchdogAge + ",\"estop\":" + telemetry.B(_state == STATE_ESTOP) + ",\"catalog_rev\":" + catalogRevision
            + ",\"catalog_counts\":{\"action\":" + actionCount + ",\"expression\":" + expressionCount + ",\"text_preset\":" + presetCount + ",\"anchor\":" + anchorCount + "}"
            + ",\"caps\":" + CapsJson() + ",\"telemetry_dropped_total\":" + telemetry.droppedTotal + ",\"log_wrap_count\":" + telemetry.logWrapCount + "}";
        EmitSnapshotPart(requestSeq, part++, parts, "session", sessionData);
        EmitSnapshotPart(requestSeq, part++, parts, "npc", locomotion.BuildStateBody(StateName(_state), _state == STATE_ESTOP, true));
        for (int page = 1; page <= playerPages; page++) EmitSnapshotPart(requestSeq, part++, parts, "players", perception.BuildSnapshotPlayersPage(page));
        EmitSnapshotPart(requestSeq, part++, parts, "voice", "{\"state\":\"disabled\",\"error_code\":null,\"url_loaded\":false,\"last_speech_seq\":null}");
        EmitSnapshotPart(requestSeq, part, parts, "text", nameplate.BuildSnapshotText());
    }

    private void EmitSnapshotPart(int requestSeq, int part, int parts, string section, string data)
    {
        telemetry.Emit("sys.snapshot", "\"snapshot_seq\":" + requestSeq + ",\"part\":" + part + ",\"parts\":" + parts + ",\"section\":" + telemetry.J(section) + ",\"data\":" + data);
    }

    private string BeginTextTransaction(int transferSeq, int rawLength, int crc16, int displaySeconds)
    {
        if (_textTransferSeq != 0) return "transfer_busy";
        if (_seenTextGeneration[transferSeq] == _historyGeneration && _seenTextTransfer[transferSeq]) return "transfer_seq_mismatch";
        _textTransferSeq = transferSeq;
        _textRawLength = rawLength;
        _textCrc16 = crc16;
        _textDisplaySeconds = displaySeconds == 0 ? 5 : displaySeconds;
        _textDisplayRaw = displaySeconds;
        _textPackedCount = 0;
        _textPackedExpected = rawLength + (rawLength + 6) / 7;
        _textDeadline = Time.timeSinceLevelLoad + 5f;
        return null;
    }

    private string ValidateTextCommit(int transferSeq, int rawLength, int crc16, int displaySeconds)
    {
        if (_textTransferSeq == 0) return "transfer_missing";
        if (transferSeq != _textTransferSeq) { AbortTextTransaction(); return "transfer_seq_mismatch"; }
        int seconds = displaySeconds == 0 ? 5 : displaySeconds;
        if (rawLength != _textRawLength || crc16 != _textCrc16 || displaySeconds != _textDisplayRaw) { AbortTextTransaction(); return "invalid_param"; }
        if (_textPackedCount != _textPackedExpected) { AbortTextTransaction(); return "length_mismatch"; }

        int[] raw = new int[384];
        int rawPos = 0; int packedPos = 0;
        while (rawPos < _textRawLength)
        {
            if (packedPos >= _textPackedCount) { AbortTextTransaction(); return "length_mismatch"; }
            int msb = _textPacked[packedPos++];
            int group = Mathf.Min(7, _textRawLength - rawPos);
            for (int i = 0; i < group; i++)
            {
                if (packedPos >= _textPackedCount) { AbortTextTransaction(); return "length_mismatch"; }
                raw[rawPos] = _textPacked[packedPos++] | (((msb >> i) & 1) << 7);
                rawPos++;
            }
        }
        int actualCrc = 0xFFFF;
        for (int i = 0; i < _textRawLength; i++) actualCrc = CrcByte(actualCrc, raw[i]);
        if (actualCrc != _textCrc16) { AbortTextTransaction(); return "crc_mismatch"; }
        if (!TryDecodeUtf8(raw, _textRawLength)) { AbortTextTransaction(); return "invalid_utf8"; }

        _pendingText = _decodedText;
        _pendingTextTransferSeq = _textTransferSeq;
        _pendingTextRawLength = _textRawLength;
        _pendingTextCrc16 = _textCrc16;
        _pendingTextDisplaySeconds = _textDisplaySeconds;
        _seenTextTransfer[_textTransferSeq] = true;
        _seenTextGeneration[_textTransferSeq] = _historyGeneration;
        AbortTextTransaction();
        return null;
    }

    private bool TryDecodeUtf8(int[] raw, int length)
    {
        _decodedText = ""; int i = 0;
        while (i < length)
        {
            int b0 = raw[i++];
            if (b0 <= 0x7F) { _decodedText += (char)b0; continue; }
            if (b0 >= 0xC2 && b0 <= 0xDF)
            {
                if (i >= length) return false; int b1 = raw[i++]; if ((b1 & 0xC0) != 0x80) return false;
                _decodedText += (char)(((b0 & 0x1F) << 6) | (b1 & 0x3F)); continue;
            }
            if (b0 >= 0xE0 && b0 <= 0xEF)
            {
                if (i + 1 >= length) return false; int b1 = raw[i++]; int b2 = raw[i++];
                if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b0 == 0xE0 && b1 < 0xA0) || (b0 == 0xED && b1 >= 0xA0)) return false;
                _decodedText += (char)(((b0 & 15) << 12) | ((b1 & 63) << 6) | (b2 & 63)); continue;
            }
            if (b0 >= 0xF0 && b0 <= 0xF4)
            {
                if (i + 2 >= length) return false; int b1 = raw[i++]; int b2 = raw[i++]; int b3 = raw[i++];
                if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 || (b0 == 0xF0 && b1 < 0x90) || (b0 == 0xF4 && b1 >= 0x90)) return false;
                int cp = ((b0 & 7) << 18) | ((b1 & 63) << 12) | ((b2 & 63) << 6) | (b3 & 63);
                cp -= 0x10000; _decodedText += (char)(0xD800 + (cp >> 10)); _decodedText += (char)(0xDC00 + (cp & 0x3FF)); continue;
            }
            return false;
        }
        return true;
    }

    private void CommitPreparedText()
    {
        nameplate.ShowDynamic(_pendingText, _pendingTextTransferSeq, _pendingTextRawLength, _pendingTextCrc16, _pendingTextDisplaySeconds);
        string body = "\"transfer_seq\":" + _pendingTextTransferSeq + ",\"utf8_bytes\":" + _pendingTextRawLength
            + ",\"crc16\":" + telemetry.J(Hex4(_pendingTextCrc16)) + ",\"display_seconds\":" + _pendingTextDisplaySeconds + ",\"text\":" + telemetry.J(_pendingText);
        if (!telemetry.FitsEvent("npc.text_displayed", body))
            body = "\"transfer_seq\":" + _pendingTextTransferSeq + ",\"utf8_bytes\":" + _pendingTextRawLength
                + ",\"crc16\":" + telemetry.J(Hex4(_pendingTextCrc16)) + ",\"display_seconds\":" + _pendingTextDisplaySeconds + ",\"text\":null,\"text_omitted\":true";
        telemetry.Emit("npc.text_displayed", body);
        _pendingText = null; _pendingTextTransferSeq = 0; _pendingTextRawLength = 0; _pendingTextCrc16 = 0; _pendingTextDisplaySeconds = 0;
    }

    private void AbortTextTransaction()
    {
        _textTransferSeq = 0; _textRawLength = 0; _textCrc16 = 0; _textDisplaySeconds = 0; _textDisplayRaw = 0;
        _textDeadline = 0f; _textPackedCount = 0; _textPackedExpected = 0;
    }

    public int CurrentTextTransferSeq() { return nameplate == null ? 0 : nameplate.CurrentTransferSeq(); }

    private bool MissingCapability(int cmd, int p3)
    {
        bool navmesh = HasNavmeshCapability();
        if (cmd == CMD_GOTO_XZ) return !enableGoto || !navmesh;
        if (cmd == CMD_SET_MODE && p3 == 1) return !enableFollow || !navmesh;
        if (cmd == CMD_SET_MODE && p3 == 2) return !enableGoto || !navmesh;
        if (cmd == CMD_SET_MODE && p3 == 3) return !enableWander || !navmesh || !locomotion.HasWanderWaypoints();
        if (cmd == CMD_PLAY_ANIM || cmd == CMD_STOP_ACTION) return !enableActions;
        if (cmd == CMD_SET_EXPRESSION) return !enableExpressions;
        if (cmd == CMD_TEXT_BEGIN || cmd == CMD_TEXT_COMMIT) return !enableTextUtf8;
        if (cmd == CMD_SPEECH_CUE) return true;
        if (cmd == CMD_TEXT_PRESET) return !enableTextPreset;
        if (cmd == CMD_RAY_SCAN) return !enableRayScan;
        if (cmd == CMD_SNAPSHOT_REQUEST) return !enableSnapshot;
        return false;
    }

    private string ValidateRegisters(int cmd, int p0, int p1, int p2, int p3, int p4, int p5)
    {
        if (cmd == CMD_SET_MODE && (p0 != 0 || p1 != 0 || p2 != 0 || p4 != 0 || p5 != 0)) return "reserved_bits";
        if (cmd == CMD_SET_MODE && p3 > 3) return "invalid_param";
        if (cmd == CMD_GOTO_XZ) { if (p5 != 0 || (p4 & 126) != 0) return "reserved_bits"; if (p4 == 0 && p2 != 0) return "invalid_param"; }
        if (cmd == CMD_SET_SPEED && (p0 != 0 || p1 != 0 || p2 != 0 || p4 != 0 || p5 != 0)) return "reserved_bits";
        if (cmd == CMD_TURN_TO && (p1 != 0 || p2 != 0 || p3 != 0 || p4 != 0 || p5 != 0)) return "reserved_bits";
        if (cmd == CMD_LOOK_AT && (p0 != 0 || p1 != 0 || p2 != 0 || p4 != 0 || p5 != 0)) return "reserved_bits";
        if (cmd == CMD_LOOK_AT && p3 > 63 && p3 != 127) return "invalid_param";
        if (cmd == CMD_PLAY_ANIM) { if (p1 != 0 || p2 != 0 || p5 != 0 || (p4 & 126) != 0) return "reserved_bits"; if (p0 < 1 || p3 == 127) return "invalid_param"; }
        if (cmd == CMD_STOP || cmd == CMD_HEARTBEAT || cmd == CMD_CLEAR_ESTOP || cmd == CMD_SNAPSHOT_REQUEST)
            if (p0 != 0 || p1 != 0 || p2 != 0 || p3 != 0 || p4 != 0 || p5 != 0) return "reserved_bits";
        if (cmd == CMD_TEXT_PRESET) { if (p0 != 0 || p1 != 0 || p2 != 0 || p5 != 0) return "reserved_bits"; if (p3 == 127 && p4 != 0) return "invalid_param"; }
        if (cmd == CMD_RAY_SCAN) { if (p0 != 0 || p1 != 0 || p2 != 0 || p4 != 0 || p5 != 0) return "reserved_bits"; if (p3 > 1) return "invalid_param"; }
        if (cmd == CMD_SET_RATE) { if (p0 != 0 || p1 != 0 || p2 != 0 || p5 != 0) return "reserved_bits"; if (p3 > 3 || p4 > 2) return "invalid_param"; }
        if (cmd == CMD_DISCOVER && (p3 != 0 || p4 != 0 || p5 != 0)) return "reserved_bits";
        if (cmd == CMD_STOP_ACTION && (p1 != 0 || p2 != 0 || p3 != 0 || p4 != 0 || p5 != 0)) return "reserved_bits";
        if (cmd == CMD_SET_TARGET) { if (p0 != 0 || p1 != 0 || p2 != 0 || p4 != 0 || p5 != 0) return "reserved_bits"; if (p3 > 63 && p3 != 127) return "invalid_param"; }
        if (cmd == CMD_LOOK_AT_XYZ && (p5 & 126) != 0) return "reserved_bits";
        if (cmd == CMD_SET_EXPRESSION) { if (p0 != 0 || p1 != 0 || p2 != 0) return "reserved_bits"; if (p3 == 127 && (p4 != 0 || p5 != 0)) return "invalid_param"; }
        if (cmd == CMD_TEXT_BEGIN || cmd == CMD_TEXT_COMMIT) { if (p5 != 0) return "reserved_bits"; if (p0 == 0) return "invalid_param"; if (p1 < 1 || p1 > 384) return "text_too_long"; if (p3 > 3) return "invalid_param"; }
        if (cmd == CMD_SPEECH_CUE) { if (p5 != 0) return "reserved_bits"; if (p0 == 0 || p4 == 0) return "invalid_param"; }
        if (cmd == CMD_SET_CONTROL_MODE) { if (p0 != 0 || p1 != 0 || p2 != 0 || p4 != 0 || p5 != 0) return "reserved_bits"; if (p3 > 1) return "invalid_param"; }
        return null;
    }

    private string StateError(int cmd)
    {
        if (_state == STATE_ESTOP) { if (cmd == CMD_HEARTBEAT || cmd == CMD_CLEAR_ESTOP || cmd == CMD_SNAPSHOT_REQUEST) return null; return "estop_latched"; }
        if (_state == STATE_SAFE_IDLE)
        {
            if (cmd == CMD_STOP || cmd == CMD_RAY_SCAN || cmd == CMD_SET_RATE || cmd == CMD_HEARTBEAT || cmd == CMD_SNAPSHOT_REQUEST || cmd == CMD_SET_CONTROL_MODE) return null;
            return "invalid_state";
        }
        if (_state == STATE_EXTERNAL || _state == STATE_MOVING || _state == STATE_ACTION) return cmd == CMD_CLEAR_ESTOP ? "invalid_state" : null;
        return "not_handshaken";
    }

    private void Reject(int requestSession, int seq, int cmd, int hash, string err)
    {
        Remember(requestSession, seq, cmd, hash, false, err, _state);
        Ack(seq, cmd, hash, false, err, false, _state);
        commandsRejected++;
    }

    private void Ack(int seq, int cmd, int hash, bool ok, string err, bool replayed, int state)
    {
        if (telemetry == null || seq == 0) return;
        string body = "\"seq\":" + seq + ",\"cmd_id\":" + cmd + ",\"cmd\":" + telemetry.J(CommandName(cmd))
            + ",\"request_hash\":" + telemetry.J(Hex4(hash)) + ",\"ok\":" + telemetry.B(ok)
            + ",\"replayed\":" + telemetry.B(replayed) + ",\"state\":" + telemetry.J(StateName(state));
        if (!ok) body += ",\"err\":" + telemetry.J(err);
        telemetry.EmitForced("npc.ack", body);
    }

    private int FindDuplicate(int session, int seq, int cmd, int hash)
    {
        float now = Time.timeSinceLevelLoad;
        for (int i = 0; i < DupSlots; i++)
        {
            if (_dupSession[i] == session && _dupSeq[i] == seq && now - _dupTime[i] < dupWindowSec)
            {
                _dupHit = i;
                return _dupCmd[i] == cmd && _dupHash[i] == hash ? 1 : 2;
            }
        }
        _dupHit = -1; return 0;
    }

    private void Remember(int session, int seq, int cmd, int hash, bool ok, string err, int state)
    {
        if (seq == 0) return;
        _dupSession[_dupCursor] = session; _dupSeq[_dupCursor] = seq; _dupCmd[_dupCursor] = cmd; _dupHash[_dupCursor] = hash;
        _dupTime[_dupCursor] = Time.timeSinceLevelLoad; _dupOk[_dupCursor] = ok; _dupErr[_dupCursor] = err; _dupState[_dupCursor] = state;
        _dupCursor = (_dupCursor + 1) % DupSlots;
    }

    private void ClearDedupe()
    {
        for (int i = 0; i < DupSlots; i++) { _dupSession[i] = -1; _dupSeq[i] = -1; _dupCmd[i] = -1; _dupHash[i] = -1; _dupTime[i] = -999f; }
        _dupCursor = 0;
    }

    private bool ClaimAllOwnership(VRCPlayerApi local)
    {
        if (local == null || locomotion == null || locomotion.npcRoot == null || nameplate == null) return false;
        GameObject rootObject = locomotion.npcRoot.gameObject;
        Networking.SetOwner(local, rootObject); Networking.SetOwner(local, gameObject); Networking.SetOwner(local, nameplate.gameObject);
        return Networking.IsOwner(rootObject) && Networking.IsOwner(gameObject) && Networking.IsOwner(nameplate.gameObject);
    }

    private bool HasAllOwnership()
    {
        return locomotion != null && locomotion.npcRoot != null && nameplate != null
            && Networking.IsOwner(locomotion.npcRoot.gameObject) && Networking.IsOwner(gameObject) && Networking.IsOwner(nameplate.gameObject);
    }

    public bool IsLocalClaimedDriver()
    {
        VRCPlayerApi local = Networking.LocalPlayer;
        return local != null && _sync != null && _sync.GetDriverPid() == local.playerId;
    }

    public bool HasLocalDriverAuthority() { return IsLocalClaimedDriver() && HasAllOwnership(); }
    public int GetSession() { return _session; }
    public int GetControlState() { return _state; }
    public int GetTargetSlot() { return _targetSlot; }
    public bool IsTouchEnabled() { return enableTouch && _session > 0; }
    public bool IsPoseEnabled() { return enablePlayerPose && _session > 0; }
    public bool IsSocialEnabled() { return enableSocialSignals && _session > 0; }
    public int GetActionDurationMs(int id) { return actionDurationMs != null && id >= 0 && id < actionDurationMs.Length ? actionDurationMs[id] : 0; }
    public int GetActionLayerIndex(int id) { return actionLayers != null && id >= 0 && id < actionLayers.Length && actionLayers[id] == "full_body" ? 2 : 1; }

    public string ActiveOpsJson()
    {
        if (!enableOperationLifecycle) return "[]";
        string[] ids = new string[4]; float[] starts = new float[4]; int count = 0;
        for (int i = 0; i < 4; i++) if (_opId[i] != null) { ids[count] = _opId[i]; starts[count] = _opStartedAt[i]; count++; }
        for (int a = 0; a < count; a++) for (int b = a + 1; b < count; b++) if (starts[b] < starts[a])
        {
            float ft = starts[a]; starts[a] = starts[b]; starts[b] = ft;
            string st = ids[a]; ids[a] = ids[b]; ids[b] = st;
        }
        string json = "[";
        for (int i = 0; i < count; i++) { if (i > 0) json += ","; json += telemetry.J(ids[i]); }
        return json + "]";
    }

    private void BeginOperation(int lane, string kind, int seq, int hash, int expectedEndMs)
    {
        CancelOperation(lane, "replaced");
        _opKind[lane] = kind; _opSeq[lane] = seq; _opHash[lane] = hash;
        _opStartedAt[lane] = Time.timeSinceLevelLoad; _opId[lane] = _session + ":" + seq + ":" + Hex4(hash);
        if (enableOperationLifecycle)
            telemetry.Emit("npc.operation_started", "\"op_id\":" + telemetry.J(_opId[lane]) + ",\"kind\":" + telemetry.J(kind)
                + ",\"request_seq\":" + seq + ",\"request_hash\":" + telemetry.J(Hex4(hash))
                + ",\"expected_end_ms\":" + (expectedEndMs < 0 ? "null" : expectedEndMs.ToString()));
    }

    private void CompleteOperation(int lane, string result)
    {
        if (_opId[lane] == null) return;
        string opId = _opId[lane]; string kind = _opKind[lane]; int seq = _opSeq[lane]; int hash = _opHash[lane];
        int elapsed = Mathf.Max(0, Mathf.RoundToInt((Time.timeSinceLevelLoad - _opStartedAt[lane]) * 1000f));
        ClearOperation(lane);
        if (enableOperationLifecycle)
            telemetry.Emit("npc.operation_completed", "\"op_id\":" + telemetry.J(opId) + ",\"kind\":" + telemetry.J(kind)
                + ",\"request_seq\":" + seq + ",\"request_hash\":" + telemetry.J(Hex4(hash))
                + ",\"elapsed_ms\":" + elapsed + ",\"result\":" + telemetry.J(result));
    }

    private void CancelOperation(int lane, string reason)
    {
        if (_opId[lane] == null) return;
        string opId = _opId[lane]; string kind = _opKind[lane]; int seq = _opSeq[lane]; int hash = _opHash[lane];
        int elapsed = Mathf.Max(0, Mathf.RoundToInt((Time.timeSinceLevelLoad - _opStartedAt[lane]) * 1000f));
        ClearOperation(lane);
        if (enableOperationLifecycle)
            telemetry.Emit("npc.operation_cancelled", "\"op_id\":" + telemetry.J(opId) + ",\"kind\":" + telemetry.J(kind)
                + ",\"request_seq\":" + seq + ",\"request_hash\":" + telemetry.J(Hex4(hash))
                + ",\"elapsed_ms\":" + elapsed + ",\"reason\":" + telemetry.J(reason));
    }

    private void ClearOperation(int lane)
    {
        _opId[lane] = null; _opKind[lane] = null; _opSeq[lane] = 0; _opHash[lane] = 0; _opStartedAt[lane] = 0f;
    }

    private string ActionCommonBody(int actionId, int actionSeq, int lane)
    {
        return "\"op_id\":" + telemetry.J(_opId[lane]) + ",\"request_seq\":" + _opSeq[lane]
            + ",\"request_hash\":" + telemetry.J(Hex4(_opHash[lane])) + ",\"action_id\":" + actionId + ",\"action_seq\":" + actionSeq
            + ",\"semantic_key\":" + telemetry.J(actionSemanticKeys[actionId]) + ",\"name\":" + telemetry.J(actionNames[actionId])
            + ",\"layer\":" + telemetry.J(actionLayers[actionId]);
    }

    private void CancelAction(string reason)
    {
        if (_currentActionId < 0 || _opId[LaneAction] == null) { locomotion.StopAction(); _currentActionId = -1; _currentActionSeq = 0; return; }
        int id = _currentActionId; int seq = _currentActionSeq;
        int elapsed = Mathf.Max(0, Mathf.RoundToInt((Time.timeSinceLevelLoad - _opStartedAt[LaneAction]) * 1000f));
        telemetry.Emit("npc.action_cancelled", ActionCommonBody(id, seq, LaneAction) + ",\"elapsed_ms\":" + elapsed + ",\"reason\":" + telemetry.J(reason));
        CancelOperation(LaneAction, reason);
        locomotion.StopAction(); _currentActionId = -1; _currentActionSeq = 0; _currentActionLoop = false;
        if (_state == STATE_ACTION) SetState(StateAfterAction());
    }

    private void CancelAllOperations(string reason)
    {
        CancelOperation(LaneMovement, reason); CancelOperation(LaneLook, reason); CancelAction(reason); CancelOperation(LaneExpression, reason);
    }

    public void OnGotoArrived(Vector3 target, float errorMeters)
    {
        if (_opId[LaneMovement] != null && _opKind[LaneMovement] == "goto")
        {
            telemetry.Emit("npc.arrived", "\"op_id\":" + telemetry.J(_opId[LaneMovement]) + ",\"request_seq\":" + _opSeq[LaneMovement]
                + ",\"request_hash\":" + telemetry.J(Hex4(_opHash[LaneMovement])) + ",\"pos\":" + telemetry.Vec3(locomotion.npcRoot.position)
                + ",\"yaw\":" + telemetry.F1(locomotion.CurrentYaw()) + ",\"error_m\":" + telemetry.F2(errorMeters) + ",\"final\":true,\"waypoint_index\":null");
            CompleteOperation(LaneMovement, "arrived");
        }
        if (_state == STATE_MOVING) SetState(StateAfterMovement());
    }

    public void OnWanderWaypoint(int index, float errorMeters)
    {
        if (_opId[LaneMovement] == null || _opKind[LaneMovement] != "wander") return;
        telemetry.Emit("npc.arrived", "\"op_id\":" + telemetry.J(_opId[LaneMovement]) + ",\"request_seq\":" + _opSeq[LaneMovement]
            + ",\"request_hash\":" + telemetry.J(Hex4(_opHash[LaneMovement])) + ",\"pos\":" + telemetry.Vec3(locomotion.npcRoot.position)
            + ",\"yaw\":" + telemetry.F1(locomotion.CurrentYaw()) + ",\"error_m\":" + telemetry.F2(errorMeters) + ",\"final\":false,\"waypoint_index\":" + index);
    }

    public void OnMovementBlocked(string blockedReason, string cancelReason, Vector3 target)
    {
        string opId = _opId[LaneMovement] == null ? "null" : telemetry.J(_opId[LaneMovement]);
        string seq = _opId[LaneMovement] == null ? "null" : _opSeq[LaneMovement].ToString();
        string hash = _opId[LaneMovement] == null ? "null" : telemetry.J(Hex4(_opHash[LaneMovement]));
        telemetry.Emit("npc.blocked", "\"op_id\":" + opId + ",\"request_seq\":" + seq + ",\"request_hash\":" + hash
            + ",\"pos\":" + telemetry.Vec3(locomotion.npcRoot.position) + ",\"reason\":" + telemetry.J(blockedReason) + ",\"target\":" + telemetry.Vec3(target));
        CancelOperation(LaneMovement, cancelReason);
        if (_state == STATE_MOVING) SetState(StateAfterMovement());
    }

    // 保留旧 Locomotion 调用名，避免场景内旧 Udon 序列化引用失效。
    public void OnGotoBlocked(string blockedReason, string cancelReason, Vector3 target) { OnMovementBlocked(blockedReason, cancelReason, target); }

    public void OnTurnCompleted()
    {
        if (_state == STATE_MOVING) SetState(StateAfterMovement());
        if (_opKind[LaneMovement] == "turn") CompleteOperation(LaneMovement, "turned");
    }

    public void OnLookCompleted(string result) { CompleteOperation(LaneLook, result); }
    public void OnLookCancelled(string reason) { CancelOperation(LaneLook, reason); }

    public void OnActionFinished()
    {
        if (_currentActionId < 0 || _opId[LaneAction] == null) return;
        int id = _currentActionId; int seq = _currentActionSeq;
        int elapsed = Mathf.Max(0, Mathf.RoundToInt((Time.timeSinceLevelLoad - _opStartedAt[LaneAction]) * 1000f));
        telemetry.Emit("npc.action_finished", ActionCommonBody(id, seq, LaneAction) + ",\"elapsed_ms\":" + elapsed);
        CompleteOperation(LaneAction, "natural_end");
        _currentActionId = -1; _currentActionSeq = 0; _currentActionLoop = false;
        if (_state == STATE_ACTION) SetState(StateAfterAction());
    }

    public void OnExpressionExpired() { CompleteOperation(LaneExpression, "expired"); }

    public void OnDriverLeft(int pid)
    {
        if (_sync == null || _sync.GetDriverPid() != pid) return;
        if (_state != STATE_ESTOP) { locomotion.WatchdogIdle(); SetState(STATE_SAFE_IDLE); }
        nameplate.ClearBubbleWithReason("control_safe_idle"); AbortTextTransaction(); CancelAllOperations("player_left"); _sync.ClearDriver(_state);
    }

    public void OnAuthorityLost()
    {
        if (_state == STATE_ESTOP) return;
        locomotion.WatchdogIdle(); SetState(STATE_SAFE_IDLE); nameplate.ClearBubbleWithReason("control_safe_idle"); AbortTextTransaction(); CancelAllOperations("control_safe_idle");
    }

    private int StateAfterMovement() { return _currentActionId >= 0 ? STATE_ACTION : STATE_EXTERNAL; }
    private int StateAfterAction() { return _opId[LaneMovement] != null ? STATE_MOVING : STATE_EXTERNAL; }

    private void SetState(int state)
    {
        _state = state;
        if (_sync != null && Networking.IsOwner(_sync.gameObject)) _sync.SetControlState(state);
    }

    void Update()
    {
        float now = Time.timeSinceLevelLoad;
        if (_textTransferSeq != 0 && now >= _textDeadline)
        {
            int related = _textTransferSeq; AbortTextTransaction();
            telemetry.EmitProtocolError("text_timeout", "text", false, related, "transfer timed out");
        }
        if (!_watchdogArmed || _session == 0 || _state == STATE_SAFE_IDLE || _state == STATE_UNHANDSHAKEN || _state == STATE_ESTOP) return;
        float elapsed = now - _lastHeartbeat;
        if (elapsed >= watchdogSec)
        {
            string previous = StateName(_state); _watchdogArmed = false;
            locomotion.WatchdogIdle(); nameplate.ClearBubbleWithReason("watchdog"); AbortTextTransaction(); SetState(STATE_SAFE_IDLE);
            telemetry.Emit("sys.watchdog", "\"elapsed_ms\":" + Mathf.RoundToInt(elapsed * 1000f) + ",\"previous_state\":" + telemetry.J(previous));
            CancelAllOperations("watchdog");
        }
    }

    private bool InsideActivityBounds(Vector3 point)
    {
        return point.x >= boundsMin.x && point.x <= boundsMax.x && point.y >= boundsMin.y && point.y <= boundsMax.y && point.z >= boundsMin.z && point.z <= boundsMax.z;
    }

    private float Dequant(int q, float low, float high) { return low + (q / (float)Q14) * (high - low); }

    private int Utf8ByteCount(string value)
    {
        if (value == null) return 0;
        int bytes = 0;
        for (int i = 0; i < value.Length; i++)
        {
            int c = value[i];
            if (c <= 0x7F) bytes++; else if (c <= 0x7FF) bytes += 2;
            else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < value.Length && value[i + 1] >= 0xDC00 && value[i + 1] <= 0xDFFF) { bytes += 4; i++; }
            else bytes += 3;
        }
        return bytes;
    }

    private int RequestHash(int cmd, int seq, int p0hi, int p0lo, int p1hi, int p1lo, int p2hi, int p2lo, int p3, int p4, int p5)
    {
        int crc = 0xFFFF;
        crc = CrcByte(crc, cmd); crc = CrcByte(crc, seq); crc = CrcByte(crc, p0hi); crc = CrcByte(crc, p0lo);
        crc = CrcByte(crc, p1hi); crc = CrcByte(crc, p1lo); crc = CrcByte(crc, p2hi); crc = CrcByte(crc, p2lo);
        crc = CrcByte(crc, p3); crc = CrcByte(crc, p4); crc = CrcByte(crc, p5);
        return crc & 0xFFFF;
    }

    private int CrcByte(int crc, int value)
    {
        crc ^= (value & 0xFF) << 8;
        for (int bit = 0; bit < 8; bit++) crc = (crc & 0x8000) != 0 ? ((crc << 1) ^ 0x1021) & 0xFFFF : (crc << 1) & 0xFFFF;
        return crc;
    }

    private string Hex4(int value)
    {
        string hex = "0123456789ABCDEF";
        return hex.Substring((value >> 12) & 15, 1) + hex.Substring((value >> 8) & 15, 1) + hex.Substring((value >> 4) & 15, 1) + hex.Substring(value & 15, 1);
    }

    private bool IsKnownCommand(int cmd) { return (cmd >= 1 && cmd <= 22) || cmd == CMD_ESTOP; }

    private string CommandName(int cmd)
    {
        if (cmd == 1) return "SET_MODE"; if (cmd == 2) return "GOTO_XZ"; if (cmd == 3) return "SET_SPEED"; if (cmd == 4) return "TURN_TO";
        if (cmd == 5) return "LOOK_AT"; if (cmd == 6) return "PLAY_ANIM"; if (cmd == 7) return "STOP"; if (cmd == 8) return "TEXT_PRESET";
        if (cmd == 9) return "RAY_SCAN"; if (cmd == 10) return "SET_RATE"; if (cmd == 11) return "HEARTBEAT"; if (cmd == 12) return "DISCOVER";
        if (cmd == 13) return "CLEAR_ESTOP"; if (cmd == 14) return "STOP_ACTION"; if (cmd == 15) return "SNAPSHOT_REQUEST"; if (cmd == 16) return "SET_TARGET";
        if (cmd == 17) return "LOOK_AT_XYZ"; if (cmd == 18) return "SET_EXPRESSION"; if (cmd == 19) return "TEXT_BEGIN"; if (cmd == 20) return "TEXT_COMMIT";
        if (cmd == 21) return "SPEECH_CUE"; if (cmd == 22) return "SET_CONTROL_MODE"; if (cmd == 127) return "ESTOP"; return "UNKNOWN";
    }

    public string StateName(int state)
    {
        if (state == STATE_SAFE_IDLE) return "safe_idle"; if (state == STATE_EXTERNAL) return "external"; if (state == STATE_MOVING) return "moving";
        if (state == STATE_ACTION) return "action"; if (state == STATE_ESTOP) return "estop"; return "unhandshaken";
    }

    private bool HasNavmeshCapability() { return locomotion != null && locomotion.HasNavMeshAgent() && (enableGoto || enableFollow || enableWander); }

    private int CapabilityBits()
    {
        int bits = 0;
        if (enableGoto) bits += 1 << 0; if (enableFollow) bits += 1 << 1;
        if (enableWander && locomotion.HasWanderWaypoints()) bits += 1 << 2;
        if (enableActions) bits += 1 << 3; if (enableExpressions) bits += 1 << 4; if (enableTextPreset) bits += 1 << 5; if (enableTextUtf8) bits += 1 << 6;
        if (enableRayScan) bits += 1 << 8; if (enableTouch) bits += 1 << 9; if (enablePlayerPose) bits += 1 << 10; if (enableSnapshot) bits += 1 << 12;
        if (HasNavmeshCapability()) bits += 1 << 13; if (enableSocialSignals) bits += 1 << 14; if (enableAnchors) bits += 1 << 15; if (enableOperationLifecycle) bits += 1 << 16;
        return bits;
    }

    private string CapsJson()
    {
        string result = "["; bool comma = false;
        if (enableGoto) { result += "\"goto\""; comma = true; }
        if (enableFollow) { if (comma) result += ","; result += "\"follow\""; comma = true; }
        if (enableWander && locomotion.HasWanderWaypoints()) { if (comma) result += ","; result += "\"wander\""; comma = true; }
        if (enableActions) { if (comma) result += ","; result += "\"actions\""; comma = true; }
        if (enableExpressions) { if (comma) result += ","; result += "\"expressions\""; comma = true; }
        if (enableTextPreset) { if (comma) result += ","; result += "\"text_preset\""; comma = true; }
        if (enableTextUtf8) { if (comma) result += ","; result += "\"text_utf8\""; comma = true; }
        if (enableRayScan) { if (comma) result += ","; result += "\"ray_scan\""; comma = true; }
        if (enableTouch) { if (comma) result += ","; result += "\"touch\""; comma = true; }
        if (enablePlayerPose) { if (comma) result += ","; result += "\"player_pose\""; comma = true; }
        if (enableSnapshot) { if (comma) result += ","; result += "\"snapshot\""; comma = true; }
        if (HasNavmeshCapability()) { if (comma) result += ","; result += "\"navmesh\""; comma = true; }
        if (enableSocialSignals) { if (comma) result += ","; result += "\"social_signals\""; comma = true; }
        if (enableAnchors) { if (comma) result += ","; result += "\"anchors\""; comma = true; }
        if (enableOperationLifecycle) { if (comma) result += ","; result += "\"operation_lifecycle\""; }
        return result + "]";
    }

    private string CurrentDriverPidJson()
    {
        int pid = _sync == null ? -1 : _sync.GetDriverPid();
        return pid > 0 ? pid.ToString() : "null";
    }
}
