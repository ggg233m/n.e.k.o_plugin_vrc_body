/*
 * NekoNpcPerception —— 以 NPC 为原点的玩家雷达 / 槽位表 / 社交几何 / 射线扫描（UdonSharp）
 *
 * 契约：YUI v1.1 player/touch/ray schema。social capability 在当前迁移版不发布。
 * 阶段：N3 —— join/leave/pose/ray 按规范实现；wave/gaze/approach 为启发式实现，阈值可在 Inspector 调。
 *
 * 方位约定（与八向环 / NekoPlayerRadar 一致）：brg 相对 NPC 正前方，右正左负，(-180,180]。
 * 玩家标识：pid=playerId、name=displayName、slot=0..63（join 分配最小空闲槽，leave 释放）。
 *
 * 与旧雷达的关系：NekoPlayerRadar.cs（以"本地玩家头部"为原点的 near 行）可以并存；本脚本以 NPC 为原点。
 */
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoNpcPerception : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;
    [Tooltip("NPC 根（方位/距离原点）")]
    public Transform npcRoot;
    [Tooltip("眼位（gaze 判定与 ray 起点高度参考；留空用 npcRoot + 1.5m）")]
    public Transform eyeAnchor;

    [Header("周期（规范 §5.2）")]
    public float scanInterval = 0.5f;
    [Tooltip("社交几何子周期，仅对 8m 内玩家")]
    public float socialInterval = 0.1f;
    public float socialRange = 8f;
    [Tooltip("player.pose 档位：0 关 / 1=1Hz / 2=2Hz（默认 2）；仅实例内 ≥2 人时发")]
    public int poseRateLevel = 2;

    [Header("社交阈值")]
    public float gazeAngleDeg = 15f;
    public float gazeRange = 3f;
    public float gazeHoldSec = 2f;
    public float approachRange = 5f;
    public float approachMinSpeed = 0.5f;
    [Tooltip("挥手：手高于头，水平往复 ≥ waveCycles 周期（每周期 2 次反向）")]
    public int waveCycles = 2;
    public float waveWindowSec = 2.5f;
    public float waveCooldownSec = 3f;

    [Header("射线（规范 §5.4）")]
    public LayerMask environmentMask = (1 << 0) | (1 << 11);
    public float rayMaxDistance = 8f;
    public float rayHeight = 0.9f;

    // ---- 槽位表 ----
    private const int MaxSlots = 64;
    private int[] _slotPid = new int[MaxSlots];
    private string[] _slotName = new string[MaxSlots];
    private VRCPlayerApi[] _slotPlayer = new VRCPlayerApi[MaxSlots];
    private bool[] _slotUsed = new bool[MaxSlots];

    // ---- 每槽社交状态 ----
    private float[] _gazeSince = new float[MaxSlots];
    private bool[] _gazeOn = new bool[MaxSlots];
    private bool[] _approachOn = new bool[MaxSlots];
    private Vector3[] _lastPos = new Vector3[MaxSlots];
    private float[] _lastPosTime = new float[MaxSlots];
    private float[] _waveLastX = new float[MaxSlots];
    private int[] _waveDir = new int[MaxSlots];      // -1/0/+1 上次水平方向
    private int[] _waveReversals = new int[MaxSlots];
    private float[] _waveWindowStart = new float[MaxSlots];
    private float[] _waveLastEmit = new float[MaxSlots];

    private float _nextScan;
    private float _nextSocial;
    private float _nextPose;
    private int _poseBatchSeq = 1;
    private NekoMidiRouter _router;

    void Start()
    {
        if (npcRoot == null) npcRoot = transform.parent != null ? transform.parent : transform;
        if (environmentMask.value == 0) environmentMask = (1 << 0) | (1 << 11);
        _router = GetComponent<NekoMidiRouter>();
        // 注意：不要在这里重置槽位数组——OnPlayerJoined 可能先于 Start 触发；Alloc() 已初始化每槽状态
    }

    // ---------- 槽位 ----------

    public override void OnPlayerJoined(VRCPlayerApi player)
    {
        if (player == null) return;
        int slot = SlotOf(player);
        if (slot < 0) slot = Alloc(player);
        if (slot < 0) { if (telemetry != null) telemetry.EmitErr("slot_full", "超过 64 槽"); return; }
        ResolveRouter();
        if (telemetry == null || _router == null || _router.GetSession() == 0) return;
        telemetry.EmitEvent("player.join", "player.join:" + slot,
            "\"pid\":" + player.playerId + ",\"name\":" + telemetry.J(player.displayName)
            + ",\"slot\":" + slot + ",\"count\":" + VRCPlayerApi.GetPlayerCount());
    }

    public override void OnPlayerLeft(VRCPlayerApi player)
    {
        if (player == null) return;
        int slot = -1;
        for (int i = 0; i < MaxSlots; i++)
        {
            if (_slotUsed[i] && _slotPid[i] == player.playerId) { slot = i; break; }
        }
        if (slot < 0) return;
        string name = _slotName[slot];
        int pid = _slotPid[slot];
        ResolveRouter();
        bool shouldLog = telemetry != null && _router != null && _router.GetSession() > 0;
        // OnPlayerLeft 时 GetPlayerCount 可能仍含该玩家，按规范报"离开后"人数
        int count = VRCPlayerApi.GetPlayerCount() - 1; if (count < 0) count = 0;
        if (shouldLog)
            telemetry.EmitEvent("player.leave", "player.leave:" + pid,
                "\"pid\":" + pid + ",\"name\":" + telemetry.J(name) + ",\"slot\":" + slot + ",\"count\":" + count);
        Release(slot);
        if (_router != null) _router.OnDriverLeft(pid);
    }

    private int Alloc(VRCPlayerApi p)
    {
        for (int i = 0; i < MaxSlots; i++)
        {
            if (!_slotUsed[i])
            {
                _slotUsed[i] = true; _slotPid[i] = p.playerId; _slotName[i] = p.displayName; _slotPlayer[i] = p;
                _gazeSince[i] = -1f; _gazeOn[i] = false; _approachOn[i] = false; _lastPosTime[i] = -1f;
                _waveDir[i] = 0; _waveReversals[i] = 0; _waveWindowStart[i] = 0f; _waveLastEmit[i] = -999f;
                return i;
            }
        }
        return -1;
    }

    private void Release(int slot)
    {
        _slotUsed[slot] = false; _slotPid[slot] = -1; _slotName[slot] = null; _slotPlayer[slot] = null;
    }

    // 链路建立/恢复时重发全部在场玩家的 player.join（供插件重建槽位表）
    public void DumpSlots()
    {
        if (telemetry == null) return;
        int count = VRCPlayerApi.GetPlayerCount();
        for (int i = 0; i < MaxSlots; i++)
        {
            VRCPlayerApi p = PlayerOfSlot(i);
            if (p == null) continue;
            telemetry.Emit("player.join",
                "\"pid\":" + _slotPid[i] + ",\"name\":" + telemetry.J(_slotName[i])
                + ",\"slot\":" + i + ",\"count\":" + count);
        }
    }

    public void RebuildSlots()
    {
        for (int i = 0; i < MaxSlots; i++) Release(i);
        int count = VRCPlayerApi.GetPlayerCount();
        VRCPlayerApi[] players = new VRCPlayerApi[count];
        VRCPlayerApi.GetPlayers(players);
        for (int i = 0; i < players.Length; i++)
        {
            VRCPlayerApi player = players[i];
            if (player != null && player.IsValid()) Alloc(player);
        }
    }

    public int SlotOf(VRCPlayerApi p)
    {
        if (p == null) return -1;
        for (int i = 0; i < MaxSlots; i++) if (_slotUsed[i] && _slotPid[i] == p.playerId) return i;
        return -1;
    }

    public VRCPlayerApi PlayerOfSlot(int slot)
    {
        if (slot < 0 || slot >= MaxSlots || !_slotUsed[slot]) return null;
        VRCPlayerApi p = _slotPlayer[slot];
        if (p == null || !p.IsValid()) return null;
        return p;
    }

    public string NameOfSlot(int slot)
    {
        if (slot < 0 || slot >= MaxSlots || !_slotUsed[slot]) return null;
        return _slotName[slot];
    }

    public VRCPlayerApi DriverPlayer()
    {
        if (telemetry == null) return Networking.LocalPlayer;
        string dn = telemetry.driverDisplayName;
        if (dn == null || dn.Length == 0) return Networking.LocalPlayer;
        for (int i = 0; i < MaxSlots; i++)
        {
            if (_slotUsed[i] && _slotName[i] == dn) return PlayerOfSlot(i);
        }
        return null;
    }

    public void SetPoseRate(int level)
    {
        if (level < 0) level = 0; if (level > 2) level = 2;
        poseRateLevel = level;
    }

    // ---------- 几何 ----------

    private Vector3 Origin()
    {
        return eyeAnchor != null ? eyeAnchor.position : npcRoot.position + Vector3.up * 1.5f;
    }

    // brg：相对 NPC 正前方，右正左负
    public float BearingTo(Vector3 worldPos)
    {
        Vector3 fwd = npcRoot.forward; fwd.y = 0f; fwd.Normalize();
        Vector3 right = npcRoot.right; right.y = 0f; right.Normalize();
        Vector3 d = worldPos - npcRoot.position; d.y = 0f;
        return Mathf.Atan2(Vector3.Dot(d, right), Vector3.Dot(d, fwd)) * Mathf.Rad2Deg;
    }

    private float FlatDistance(Vector3 worldPos)
    {
        Vector3 d = worldPos - npcRoot.position; d.y = 0f;
        return d.magnitude;
    }

    private float YawOfDir(Vector3 dir)
    {
        return Mathf.Repeat(Mathf.Atan2(dir.x, dir.z) * Mathf.Rad2Deg, 360f);
    }

    // ---------- 周期扫描 ----------

    void Update()
    {
        if (telemetry == null) return;
        ResolveRouter();
        if (_router == null || _router.GetSession() == 0) return;
        if (!telemetry.logOnAllClients && !telemetry.IsDriver()) return; // 非 driver 客户端不做扫描（省 CPU，也不会打日志）
        float now = Time.timeSinceLevelLoad;

        if (_router.IsSocialEnabled() && now >= _nextSocial)
        {
            _nextSocial = now + socialInterval;
            SocialTick(now);
        }

        if (now >= _nextScan)
        {
            _nextScan = now + scanInterval;
            float poseHz = poseRateLevel == 1 ? 1f : (poseRateLevel == 2 ? 2f : 0f);
            if (_router.IsPoseEnabled() && poseHz > 0f && now >= _nextPose)
            {
                _nextPose = now + 1f / poseHz;
                EmitPose();
            }
        }
    }

    // player.pose：每行 ≤ PosePerLine 人，超出分页（规范 §4.1 行长 1000 上限）
    private const int PosePerLine = 4;

    private void EmitPose()
    {
        int total = 0;
        for (int i = 0; i < MaxSlots; i++) if (PlayerOfSlot(i) != null) total++;
        if (total == 0) return;
        int pages = (total + PosePerLine - 1) / PosePerLine;
        int batch = _poseBatchSeq++;
        if (_poseBatchSeq <= 0) _poseBatchSeq = 1;
        for (int page = 1; page <= pages; page++)
            telemetry.Emit("player.pose", "\"batch_seq\":" + batch + ",\"page\":" + page + ",\"pages\":" + pages + ",\"players\":[" + BuildPlayerItems(page, false) + "]");
    }

    private void SocialTick(float now)
    {
        Vector3 origin = Origin();
        for (int i = 0; i < MaxSlots; i++)
        {
            VRCPlayerApi p = PlayerOfSlot(i);
            if (p == null) continue;
            Vector3 pos = p.GetPosition();
            float d = FlatDistance(pos);
            if (d > socialRange) { _gazeSince[i] = -1f; if (_gazeOn[i]) { _gazeOn[i] = false; EmitGaze(i, false); } continue; }

            // --- gaze：头部 forward 与"指向 NPC 眼位"夹角 < 15° 且 d < 3m 持续 2s ---
            VRCPlayerApi.TrackingData head = p.GetTrackingData(VRCPlayerApi.TrackingDataType.Head);
            Vector3 toNpc = origin - head.position;
            float ang = Vector3.Angle(head.rotation * Vector3.forward, toNpc);
            bool gazing = d < gazeRange && ang < gazeAngleDeg;
            if (gazing)
            {
                if (_gazeSince[i] < 0f) _gazeSince[i] = now;
                else if (!_gazeOn[i] && now - _gazeSince[i] >= gazeHoldSec) { _gazeOn[i] = true; EmitGaze(i, true); }
            }
            else
            {
                _gazeSince[i] = -1f;
                if (_gazeOn[i]) { _gazeOn[i] = false; EmitGaze(i, false); }
            }

            // --- approach：速度矢量指向 NPC 且 d < 5m，进入时一次 ---
            if (_lastPosTime[i] >= 0f)
            {
                float dtp = now - _lastPosTime[i];
                if (dtp > 0.05f)
                {
                    Vector3 v = (pos - _lastPos[i]) / dtp; v.y = 0f;
                    Vector3 dirToNpc = npcRoot.position - pos; dirToNpc.y = 0f;
                    float closing = dirToNpc.sqrMagnitude > 0.0001f ? Vector3.Dot(v, dirToNpc.normalized) : 0f;
                    bool approaching = d < approachRange && closing > approachMinSpeed;
                    if (approaching && !_approachOn[i])
                    {
                        _approachOn[i] = true;
                        telemetry.EmitEvent("social.approach", "social.approach:" + i,
                            "\"slot\":" + i + ",\"d\":" + telemetry.F2(d) + ",\"brg\":" + telemetry.F1(BearingTo(pos))
                            + ",\"radial_speed\":" + telemetry.F2(closing));
                    }
                    else if (!approaching && (d > approachRange + 1f || closing < 0f))
                    {
                        _approachOn[i] = false; // 走远或远离后复位，下次靠近再报
                    }
                }
            }
            _lastPos[i] = pos; _lastPosTime[i] = now;

            // --- wave：任一手高于头，水平（NPC 右向分量）往复 ---
            Vector3 headPos = p.GetBonePosition(HumanBodyBones.Head);
            Vector3 hR = p.GetBonePosition(HumanBodyBones.RightHand);
            Vector3 hL = p.GetBonePosition(HumanBodyBones.LeftHand);
            Vector3 hand = hR.y >= hL.y ? hR : hL;
            bool raised = headPos.sqrMagnitude > 0.0001f && hand.y > headPos.y + 0.05f;
            if (raised)
            {
                Vector3 right = npcRoot.right; right.y = 0f; right.Normalize();
                float x = Vector3.Dot(hand - headPos, right);
                if (_waveDir[i] == 0) { _waveDir[i] = 1; _waveLastX[i] = x; _waveWindowStart[i] = now; _waveReversals[i] = 0; }
                else
                {
                    float dx = x - _waveLastX[i];
                    if (Mathf.Abs(dx) > 0.04f)
                    {
                        int dir = dx > 0f ? 1 : -1;
                        if (dir != _waveDir[i]) { _waveDir[i] = dir; _waveReversals[i]++; }
                        _waveLastX[i] = x;
                    }
                    if (now - _waveWindowStart[i] > waveWindowSec) { _waveWindowStart[i] = now; _waveReversals[i] = 0; }
                    if (_waveReversals[i] >= waveCycles * 2 && now - _waveLastEmit[i] > waveCooldownSec)
                    {
                        _waveLastEmit[i] = now; _waveReversals[i] = 0;
                        telemetry.EmitEvent("social.wave", "social.wave:" + i,
                            "\"slot\":" + i + ",\"brg\":" + telemetry.F1(BearingTo(pos))
                            + ",\"hand\":" + telemetry.J(hR.y >= hL.y ? "right" : "left"));
                    }
                }
            }
            else { _waveDir[i] = 0; _waveReversals[i] = 0; }
        }
    }

    private void EmitGaze(int slot, bool on)
    {
        VRCPlayerApi player = PlayerOfSlot(slot);
        Vector3 pos = player == null ? npcRoot.position : player.GetPosition();
        telemetry.EmitEvent("social.gaze", "social.gaze:" + slot + ":" + (on ? "1" : "0"),
            "\"slot\":" + slot + ",\"on\":" + telemetry.B(on)
            + ",\"d\":" + telemetry.F2(FlatDistance(pos)) + ",\"brg\":" + telemetry.F1(BearingTo(pos)));
    }

    // ---------- 射线扫描（RAY_SCAN） ----------

    // mode 0：腰高 8 向，顺序 0°,+45°,-45°,+90°,-90°,+135°,-135°,180°（bearing 右正）
    // mode 1：前向 ±30° 内 7 条等角射线（-30..+30，步 10°）
    public void RayScan(int mode, int requestSeq)
    {
        if (telemetry == null) return;
        Vector3 origin = npcRoot.position + Vector3.up * rayHeight;
        string bearings = "";
        string arr = "";
        int n = mode == 1 ? 7 : 8;
        for (int k = 0; k < n; k++)
        {
            float brg;
            if (mode == 1) brg = -30f + 10f * k;
            else
            {
                if (k == 0) brg = 0f; else if (k == 1) brg = 45f; else if (k == 2) brg = -45f;
                else if (k == 3) brg = 90f; else if (k == 4) brg = -90f; else if (k == 5) brg = 135f;
                else if (k == 6) brg = -135f; else brg = 180f;
            }
            Vector3 dir = Quaternion.Euler(0f, brg, 0f) * npcRoot.forward; dir.y = 0f; dir.Normalize();
            RaycastHit hit;
            float d = -1f;
            if (Physics.Raycast(origin, dir, out hit, rayMaxDistance, environmentMask.value, QueryTriggerInteraction.Ignore)) d = hit.distance;
            if (k > 0) { bearings = bearings + ","; arr = arr + ","; }
            bearings = bearings + telemetry.F1(brg);
            arr = arr + telemetry.F2(d);
        }
        telemetry.Emit("env.ray", "\"request_seq\":" + requestSeq + ",\"mode\":" + (mode == 1 ? 1 : 0) + ",\"bearings\":[" + bearings + "],\"d\":[" + arr + "]");
    }

    public int SnapshotPageCount()
    {
        int total = 0;
        for (int i = 0; i < MaxSlots; i++) if (PlayerOfSlot(i) != null) total++;
        int pages = (total + PosePerLine - 1) / PosePerLine;
        return pages < 1 ? 1 : pages;
    }

    public string BuildSnapshotPlayersPage(int page)
    {
        int pages = SnapshotPageCount();
        return "{\"page\":" + page + ",\"pages\":" + pages + ",\"players\":[" + BuildPlayerItems(page, true) + "]}";
    }

    private string BuildPlayerItems(int page, bool includeName)
    {
        int skip = (page - 1) * PosePerLine;
        int seen = 0;
        int added = 0;
        string items = "";
        for (int slot = 0; slot < MaxSlots; slot++)
        {
            VRCPlayerApi player = PlayerOfSlot(slot);
            if (player == null) continue;
            if (seen++ < skip) continue;
            if (added >= PosePerLine) break;
            Vector3 pos = player.GetPosition();
            VRCPlayerApi.TrackingData head = player.GetTrackingData(VRCPlayerApi.TrackingDataType.Head);
            Vector3 forward = head.rotation * Vector3.forward;
            forward.y = 0f;
            string yaw = forward.sqrMagnitude > 0.0001f ? telemetry.F1(YawOfDir(forward.normalized)) : "null";
            if (added > 0) items += ",";
            items += "{\"slot\":" + slot + ",\"pid\":" + player.playerId;
            if (includeName) items += ",\"name\":" + telemetry.J(_slotName[slot]);
            items += ",\"d\":" + telemetry.F2(FlatDistance(pos))
                + ",\"brg\":" + telemetry.F1(BearingTo(pos))
                + ",\"yaw\":" + yaw
                + ",\"vr\":" + telemetry.B(player.IsUserInVR()) + "}";
            added++;
        }
        return items;
    }

    private void ResolveRouter()
    {
        if (_router == null) _router = GetComponent<NekoMidiRouter>();
    }

    // 视线遮挡：NPC 眼位 → 玩家头部是否被环境挡住（供上层/调试用）
    public bool IsOccluded(VRCPlayerApi p)
    {
        if (p == null || !p.IsValid()) return false;
        Vector3 from = Origin();
        Vector3 to = p.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position;
        return Physics.Linecast(from, to, environmentMask.value, QueryTriggerInteraction.Ignore);
    }
}
