/*
 * NekoNpcSync —— NPC 网络同步与 YUI v1.1 driver/session 权威（UdonSharp, Continuous）
 *
 * 契约：《NPC Udon 接口规范 v1.0》§7.1。
 *   · driver 客户端（= 收 MIDI 的客户端 = 用户本人）持有 NPC 根 ownership；
 *   · 同步变量：pos(Vector3) / yaw(short, 0.1°) / animId(byte) / estop(bool)；
 *   · 远端做位置/朝向插值（0.15s 缓冲），禁止瞬移；晚加入者由 OnDeserialization 对齐；
 *   · driver 不在场：NPC 静止于最后同步位（整条控制链本就离线）。
 *
 * 挂载：本脚本必须挂在 NPC 根物体上（ownership 以根为准）。
 * 阶段：N5 打磨对象；本版为可用草稿。
 */
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.Continuous)]
public class NekoNpcSync : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;
    public NekoNpcLocomotion locomotion;
    public NekoMidiRouter router;
    public Animator animator;

    [Header("远端插值")]
    [Tooltip("远端追赶速度上限（m/s），≈ maxSpeed×1.5")]
    public float remoteCatchupSpeed = 3.0f;
    public float remoteTurnRateDeg = 360f;
    [Tooltip("超过此距离视为传送（首次对齐/晚加入），直接贴上")]
    public float teleportThreshold = 4.0f;

    [UdonSynced] public Vector3 syncPos;
    [UdonSynced] public short syncYaw10;   // yaw×10
    [UdonSynced] public byte syncAction;   // actionId+1（0 = null）
    [UdonSynced] public int syncActionSeq;
    [UdonSynced] public bool syncActionLoop;
    [UdonSynced] public int syncActionStartedAtServerMs;
    [UdonSynced] public byte syncExpression; // expressionId+1（0 = null）
    [UdonSynced] public float syncExpressionWeight;
    [UdonSynced] public bool syncEstop;
    [UdonSynced] public int syncDriverPid = -1;
    [UdonSynced] public int syncSession;
    [UdonSynced] public int syncControlState;

    private bool _initialized;
    private Vector3 _remoteLastPos;
    private float _remoteSpeed;

    void Start()
    {
        syncPos = transform.position;
        syncYaw10 = (short)Mathf.RoundToInt(Mathf.Repeat(transform.eulerAngles.y, 360f) * 10f);
        _remoteLastPos = transform.position;
        if (router == null && locomotion != null) router = locomotion.router;
    }

    public override void OnOwnershipTransferred(VRCPlayerApi player)
    {
        // v1.1 禁止自动抢回 ownership；必须重新 DISCOVER，避免双 driver。
        if (router != null && syncSession > 0 && player != null && !player.isLocal) router.OnAuthorityLost();
    }

    public void SetAuthority(int driverPid, int session, int controlState)
    {
        if (!Networking.IsOwner(gameObject)) return;
        syncDriverPid = driverPid;
        syncSession = session;
        syncControlState = controlState;
    }

    public void SetControlState(int controlState)
    {
        if (!Networking.IsOwner(gameObject)) return;
        syncControlState = controlState;
    }

    public void ClearDriver(int controlState)
    {
        if (!Networking.IsOwner(gameObject)) return;
        syncDriverPid = -1;
        syncControlState = controlState;
    }

    public int GetDriverPid() { return syncDriverPid; }
    public int GetSession() { return syncSession; }
    public int GetControlState() { return syncControlState; }

    void Update()
    {
        if (Networking.IsOwner(gameObject))
        {
            // driver：把 Locomotion 的结果写进同步变量（Continuous 自动发送）
            syncPos = transform.position;
            syncYaw10 = (short)Mathf.RoundToInt(Mathf.Repeat(transform.eulerAngles.y, 360f) * 10f);
            int a = locomotion != null ? locomotion.GetActionId() : -1;
            syncAction = (byte)(a + 1);
            syncActionSeq = locomotion != null ? locomotion.GetActionSeq() : 0;
            syncActionLoop = locomotion != null && locomotion.GetActionLoop();
            syncActionStartedAtServerMs = locomotion != null ? locomotion.GetActionStartedServerMs() : 0;
            int expression = locomotion != null ? locomotion.GetExpressionId() : -1;
            syncExpression = (byte)(expression + 1);
            syncExpressionWeight = locomotion != null ? locomotion.GetExpressionWeight() : 0f;
            syncEstop = locomotion != null && locomotion.GetMode() == NekoNpcLocomotion.MODE_ESTOP;
            return;
        }

        // 远端：插值到同步位
        float dt = Time.deltaTime;
        Vector3 target = syncPos;
        Vector3 cur = transform.position;
        if (!_initialized || (target - cur).magnitude > teleportThreshold)
        {
            transform.position = target;
            _remoteLastPos = target; // 传送不算位移，避免 Speed 尖峰
            _initialized = true;
        }
        else
        {
            transform.position = Vector3.MoveTowards(cur, target, remoteCatchupSpeed * dt);
        }
        Quaternion want = Quaternion.Euler(0f, syncYaw10 / 10f, 0f);
        transform.rotation = Quaternion.RotateTowards(transform.rotation, want, remoteTurnRateDeg * dt);

        if (animator != null && dt > 0f)
        {
            float raw = (transform.position - _remoteLastPos).magnitude / dt;
            _remoteSpeed = Mathf.MoveTowards(_remoteSpeed, raw, 8f * dt); // 平滑，避免包间抖动让走/站混合闪烁
            animator.SetFloat("Speed", _remoteSpeed);
            animator.SetInteger("ActionId", (int)syncAction - 1);
            animator.SetInteger("ActionSeq", syncActionSeq);
            animator.SetBool("ActionLoop", syncActionLoop);
            animator.SetInteger("ExpressionId", (int)syncExpression - 1);
            animator.SetFloat("ExpressionWeight", syncExpressionWeight);
            animator.SetBool("Estop", syncEstop);
        }
        _remoteLastPos = transform.position;
    }

    public override void OnDeserialization()
    {
        // 首包/晚加入：直接对齐（Update 的 teleportThreshold 也会兜底）
        if (!_initialized)
        {
            transform.position = syncPos;
            transform.rotation = Quaternion.Euler(0f, syncYaw10 / 10f, 0f);
            _remoteLastPos = syncPos;
            _initialized = true;
        }
        RestoreAnimatorProgress();
    }

    private void RestoreAnimatorProgress()
    {
        if (animator == null || router == null || syncAction == 0 || syncActionLoop || syncActionStartedAtServerMs <= 0) return;
        int actionId = (int)syncAction - 1;
        int duration = router.GetActionDurationMs(actionId);
        if (duration <= 0) return;
        int elapsed = Networking.GetServerTimeInMilliseconds() - syncActionStartedAtServerMs;
        float normalized = Mathf.Clamp01(Mathf.Max(0, elapsed) / (float)duration);
        animator.Play("Action_" + actionId, router.GetActionLayerIndex(actionId), normalized);
    }
}
