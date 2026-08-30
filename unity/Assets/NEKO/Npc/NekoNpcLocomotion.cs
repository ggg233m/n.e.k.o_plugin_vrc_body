/*
 * NekoNpcLocomotion —— YUI NPC v1.1/v1.2 NavMesh、注视、动作与表情执行器（UdonSharp）
 *
 * 只有当前 driver 且持有 NPC 根 ownership 的客户端执行 NavMesh/Animator；其他客户端
 * 只由 NekoNpcSync 投影同步状态。本脚本不接收 MIDI，所有协议校验和生命周期均由
 * NekoMidiRouter 统一负责，因此不会形成第二套控制栈。
 */
using UdonSharp;
using UnityEngine;
using UnityEngine.AI;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoNpcLocomotion : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;
    public NekoNpcPerception perception;
    public NekoMidiRouter router;
    public Transform npcRoot;
    public NavMeshAgent navAgent;
    public Animator animator;

    [Header("NavMeshAgent 固定参数（协议 v1.1 §11.1）")]
    public float maxSpeed = 2.0f;
    public float accel = 4.0f;
    public float turnRateDeg = 180f;
    public float stopDistance = 0.3f;
    public float followDistance = 1.2f;
    public float followRefreshSec = 0.2f;
    [Tooltip("连续巡逻在到达该距离时预切下一航点，必须大于 stoppingDistance 才能避免逐点停车")]
    public float wanderSwitchDistance = 0.9f;
    [Tooltip("v1.2 绕行每圈路点数；冻结验收配置使用 24")]
    public int orbitPointsPerLap = 24;

    [Header("Inspector 明确发布的巡逻航点")]
    [Tooltip("发布 wander 时至少两个；严格按数组 0..n-1 循环")]
    public Transform[] wanderWaypoints = new Transform[0];

    [Header("本地障碍与卡住检测")]
    public LayerMask environmentMask = (1 << 0) | (1 << 11);
    public float whiskerLength = 0.6f;
    public float whiskerAngleDeg = 25f;
    [Tooltip("射线位于胸口高度，避免把可行走楼梯的低矮踏步误判为墙体")]
    public float whiskerHeight = 1.4f;
    public float stuckSeconds = 1.5f;
    public float stuckMinMove = 0.05f;

    [Header("遥测")]
    public int stateRateLevel = 2;

    [Header("由 Animator Builder 写入")]
    public float[] animDurations = new float[0];

    public const int MODE_IDLE = 0;
    public const int MODE_FOLLOW = 1;
    public const int MODE_GOTO = 2;
    public const int MODE_WANDER = 3;
    public const int MODE_ESTOP = 4;
    public const int MODE_ORBIT = 5;

    private int _mode = MODE_IDLE;
    private float _cruise;
    private float _speed;
    private Vector3 _vel;
    private Vector3 _target;
    private float _targetYaw = -1f;
    private bool _hasTarget;
    private bool _finishingGotoYaw;
    private float _turnYaw = -1f;
    private int _followSlot = -1;
    private float _nextFollowRefresh;
    private int _wanderIndex;
    private Vector3 _validatedTarget;
    private float _arrivalDistance = 0.3f;

    // v1.2 ORBIT_ENTITY 最多三圈；额外一个点用于闭合最后一圈。
    private Vector3[] _orbitPoints = new Vector3[73];
    private int _orbitPointCount;
    private int _orbitIndex;
    private Vector3 _orbitCenter;
    private bool _orbitFaceTarget;
    private float _orbitSwitchDistance;
    private float _orbitMinAdvanceDistance;
    private float _orbitStoppingDistance;
    private Vector3 _orbitWaypointStart;
    private Vector3 _orbitLastPosition;
    private float _orbitTravelMeters;
    private float _orbitRequiredTravelMeters;
    private bool _orbitRecoveryUsed;
    private float _orbitStuckGraceUntil = -1f;

    private int _lookSlot = -1;
    private Vector3 _lookPoint;
    private bool _hasLookPoint;
    private bool _lookBodyAssist;
    private float _lookPointUntil = -1f;

    private int _actionId = -1;
    private int _actionSeq;
    private bool _actionLoop;
    private float _actionStartedAt;
    private int _actionStartedServerMs;
    private float _actionExpectedEnd = -1f;
    private int _actionLayer;
    private int _expressionId = -1;
    private float _expressionWeight;
    private float _expressionUntil = -1f;

    private bool _grounded;
    private float _stuckTimer;
    private Vector3 _stuckAnchor;
    private float _stateTimer;
    private Vector3 _lastPos;
    private bool _warnedNotOwner;
    private float _notOwnerSince = -1f;
    private int _localObstacleFrames;

    void Start()
    {
        if (npcRoot == null) npcRoot = transform.parent != null ? transform.parent : transform;
        if (navAgent == null && npcRoot != null) navAgent = npcRoot.GetComponent<NavMeshAgent>();
        if (environmentMask.value == 0) environmentMask = (1 << 0) | (1 << 11);
        maxSpeed = 2f;
        accel = 4f;
        turnRateDeg = 180f;
        stopDistance = 0.3f;
        followDistance = 1.2f;
        _cruise = maxSpeed;
        _lastPos = npcRoot.position;
        _stuckAnchor = _lastPos;
        ConfigureAgent();
        DisableAgent();
        ApplyAnimator();
    }

    private void ConfigureAgent()
    {
        if (navAgent == null) return;
        navAgent.radius = 0.25f;
        navAgent.height = 1.6f;
        navAgent.baseOffset = 0f;
        navAgent.speed = maxSpeed;
        navAgent.acceleration = accel;
        navAgent.angularSpeed = turnRateDeg;
        navAgent.stoppingDistance = stopDistance;
        navAgent.autoBraking = true;
        navAgent.updatePosition = true;
        navAgent.updateRotation = true;
    }

    public bool HasNavMeshAgent() { return navAgent != null; }

    public bool HasWanderWaypoints()
    {
        if (wanderWaypoints == null || wanderWaypoints.Length < 2) return false;
        for (int i = 0; i < wanderWaypoints.Length; i++) if (wanderWaypoints[i] == null) return false;
        return true;
    }

    public bool SetMode(int mode)
    {
        if (_mode == MODE_ESTOP) return false;
        if (mode == MODE_IDLE) { StopMovement(); return true; }
        if (mode == MODE_GOTO && _hasTarget) return StartAgentForCurrentTarget(MODE_GOTO);
        return false;
    }

    // 返回 null 表示目标与路径已完成预校验，Router 可安全 ACK。
    public string Goto(float x, float z, float yawOrNeg, float speed, int seq)
    {
        if (_mode == MODE_ESTOP) return "estop_latched";
        _arrivalDistance = stopDistance;
        string err = ValidatePathTarget(new Vector3(x, npcRoot.position.y, z));
        if (err != null) return err;
        // 连续路线会在前一 GOTO 抵达前下发下一点。这里直接更新 destination，
        // 保留 NavMeshAgent 当前速度；Router 仍会把前一操作按 v1.1 记为 replaced。
        if (CanRetargetActiveGoto())
            return RetargetActiveAgent(_validatedTarget, yawOrNeg, speed, MODE_GOTO) ? null : "no_path";
        PrepareMovementTarget(_validatedTarget, yawOrNeg, speed, MODE_GOTO);
        return StartAgentForCurrentTarget(MODE_GOTO) ? null : "no_path";
    }

    // v1.2 Anchor 使用完整 XYZ 和目录发布的到达半径，不能沿用当前 NPC 的 Y。
    public string GotoPosition(Vector3 position, float yawOrNeg, float speed, float arrivalRadius)
    {
        if (_mode == MODE_ESTOP) return "estop_latched";
        string err = ValidatePathTarget(position);
        if (err != null) return err;
        _arrivalDistance = Mathf.Clamp(arrivalRadius, 0.05f, 2f);
        if (CanRetargetActiveGoto())
            return RetargetActiveAgent(_validatedTarget, yawOrNeg, speed, MODE_GOTO) ? null : "no_path";
        PrepareMovementTarget(_validatedTarget, yawOrNeg, speed, MODE_GOTO);
        return StartAgentForCurrentTarget(MODE_GOTO) ? null : "no_path";
    }

    // 整段绕行由 Unity 在同一个 operation 内执行，Python 不展开坐标路点。
    public string StartOrbit(Vector3 center, float radius, int laps, bool counterClockwise, bool faceTarget, float speed)
    {
        if (_mode == MODE_ESTOP) return "estop_latched";
        if (navAgent == null) return "unsupported_capability";
        int pointsPerLap = Mathf.Clamp(orbitPointsPerLap, 16, 24);
        int safeLaps = Mathf.Clamp(laps, 1, 3);
        int count = safeLaps * pointsPerLap + 1;
        if (_orbitPoints == null || _orbitPoints.Length < 73) _orbitPoints = new Vector3[73];

        Vector3 fromCenter = npcRoot.position - center;
        fromCenter.y = 0f;
        float startAngle = fromCenter.sqrMagnitude > 0.0001f ? Mathf.Atan2(fromCenter.z, fromCenter.x) : 0f;
        float direction = counterClockwise ? 1f : -1f;
        Vector3 previous = npcRoot.position;
        for (int i = 0; i < count; i++)
        {
            float angle = startAngle + direction * (Mathf.PI * 2f * i / pointsPerLap);
            Vector3 requested = new Vector3(center.x + Mathf.Cos(angle) * radius, center.y, center.z + Mathf.Sin(angle) * radius);
            NavMeshHit hit;
            if (!NavMesh.SamplePosition(requested, out hit, 0.75f, NavMesh.AllAreas)) return "target_not_on_navmesh";
            NavMeshPath segment = new NavMeshPath();
            NavMeshHit previousHit;
            if (!NavMesh.SamplePosition(previous, out previousHit, 0.75f, NavMesh.AllAreas)
                || !NavMesh.CalculatePath(previousHit.position, hit.position, NavMesh.AllAreas, segment)
                || segment.status != NavMeshPathStatus.PathComplete) return "no_path";
            if (i > 0 && FlatDistance(_orbitPoints[i - 1], hit.position) < Mathf.Max(0.04f, radius * 0.03f))
                return "no_path";
            _orbitPoints[i] = hit.position;
            previous = hit.position;
        }

        _orbitCenter = center;
        _orbitFaceTarget = faceTarget;
        _orbitPointCount = count;
        _orbitSwitchDistance = Mathf.Clamp(radius * 0.12f, 0.08f, 0.45f);
        _orbitMinAdvanceDistance = Mathf.Clamp(radius * 0.10f, 0.08f, 0.35f);
        // 圆周点之间只有约 15°。若沿用普通导航的 0.3m stoppingDistance，
        // Agent 可能停在切点阈值外；绕行中间点必须使用更小的停止距离。
        _orbitStoppingDistance = Mathf.Min(0.05f, stopDistance);
        // 方向反转时 NPC 通常正位于上一圈的闭合点。不能只跳过 point[0]：
        // NavMesh 采样可能把相邻点也压到当前小片区域，随后会在尚未形成速度前
        // 被旧的“实际路程不足”保护误判为 stuck。直接选取第一个有意义的远点，
        // 最终闭合点仍保留，因此不会少走一圈。
        _orbitIndex = FirstUsefulOrbitIndex(npcRoot.position);
        _orbitWaypointStart = npcRoot.position;
        _orbitLastPosition = npcRoot.position;
        _orbitTravelMeters = 0f;
        _orbitRequiredTravelMeters = Mathf.PI * 2f * radius * safeLaps * 0.65f;
        _orbitRecoveryUsed = false;
        _orbitStuckGraceUntil = Time.timeSinceLevelLoad + 0.5f;
        _arrivalDistance = stopDistance;
        PrepareMovementTarget(_orbitPoints[_orbitIndex], -1f, speed, MODE_ORBIT);
        if (!StartAgentForCurrentTarget(MODE_ORBIT)) { ClearOrbit(); return "no_path"; }
        navAgent.autoBraking = false;
        navAgent.updateRotation = !faceTarget;
        return null;
    }

    private bool CanRetargetActiveGoto()
    {
        return _mode == MODE_GOTO && _hasTarget && !_finishingGotoYaw
            && navAgent != null && navAgent.enabled && navAgent.isOnNavMesh;
    }

    public string StartFollow(int slot)
    {
        if (_mode == MODE_ESTOP) return "estop_latched";
        if (navAgent == null) return "unsupported_capability";
        VRCPlayerApi player = perception == null ? null : perception.PlayerOfSlot(slot);
        if (player == null || !player.IsValid()) return "slot_unknown";
        string err = ValidatePathTarget(player.GetPosition());
        if (err != null) return err;
        _followSlot = slot;
        _wanderIndex = 0;
        PrepareMovementTarget(_validatedTarget, -1f, _cruise, MODE_FOLLOW);
        navAgent.stoppingDistance = followDistance;
        _nextFollowRefresh = 0f;
        return StartAgentForCurrentTarget(MODE_FOLLOW) ? null : "no_path";
    }

    public string StartWander()
    {
        if (_mode == MODE_ESTOP) return "estop_latched";
        if (!HasWanderWaypoints()) return "unsupported_capability";
        _wanderIndex = 0;
        _followSlot = -1;
        return StartWanderTarget(_wanderIndex);
    }

    private string StartWanderTarget(int index)
    {
        string err = ValidatePathTarget(wanderWaypoints[index].position);
        if (err != null) return err;
        if (_mode == MODE_WANDER && _hasTarget && navAgent != null && navAgent.enabled && navAgent.isOnNavMesh)
            return RetargetActiveAgent(_validatedTarget, -1f, _cruise, MODE_WANDER) ? null : "no_path";
        PrepareMovementTarget(_validatedTarget, -1f, _cruise, MODE_WANDER);
        navAgent.stoppingDistance = stopDistance;
        return StartAgentForCurrentTarget(MODE_WANDER) ? null : "no_path";
    }

    private string ValidatePathTarget(Vector3 requested)
    {
        _validatedTarget = requested;
        if (navAgent == null) return "unsupported_capability";
        NavMeshHit targetHit;
        if (!NavMesh.SamplePosition(requested, out targetHit, 0.5f, NavMesh.AllAreas)) return "target_not_on_navmesh";
        NavMeshHit startHit;
        if (!NavMesh.SamplePosition(npcRoot.position, out startHit, 0.5f, NavMesh.AllAreas)) return "no_path";
        NavMeshPath path = new NavMeshPath();
        if (!NavMesh.CalculatePath(startHit.position, targetHit.position, NavMesh.AllAreas, path)
            || path.status != NavMeshPathStatus.PathComplete) return "no_path";
        _validatedTarget = targetHit.position;
        return null;
    }

    private void PrepareMovementTarget(Vector3 sampled, float yawOrNeg, float speed, int mode)
    {
        DisableAgent();
        if (mode != MODE_ORBIT) ClearOrbit();
        _target = sampled;
        _targetYaw = yawOrNeg < 0f ? -1f : Mathf.Repeat(yawOrNeg, 360f);
        _hasTarget = true;
        _finishingGotoYaw = false;
        _turnYaw = -1f;
        _cruise = Mathf.Clamp(speed <= 0.01f ? maxSpeed : speed, 0.1f, maxSpeed);
        _mode = mode;
        if (mode != MODE_GOTO) _arrivalDistance = stopDistance;
        ResetStuck();
        _localObstacleFrames = 0;
    }

    private bool RetargetActiveAgent(Vector3 sampled, float yawOrNeg, float speed, int mode)
    {
        if (navAgent == null || !navAgent.enabled || !navAgent.isOnNavMesh) return false;
        if (mode != MODE_ORBIT) ClearOrbit();
        _target = sampled;
        _targetYaw = yawOrNeg < 0f ? -1f : Mathf.Repeat(yawOrNeg, 360f);
        _hasTarget = true;
        _finishingGotoYaw = false;
        _turnYaw = -1f;
        _cruise = Mathf.Clamp(speed <= 0.01f ? maxSpeed : speed, 0.1f, maxSpeed);
        _mode = mode;
        navAgent.speed = _cruise;
        navAgent.stoppingDistance = mode == MODE_FOLLOW
            ? followDistance
            : (mode == MODE_ORBIT ? _orbitStoppingDistance : _arrivalDistance);
        // GOTO 先保持巡航；只有进入最终制动距离且没有收到下一点时才启用制动。
        navAgent.autoBraking = mode == MODE_FOLLOW;
        navAgent.updateRotation = mode != MODE_ORBIT || !_orbitFaceTarget;
        navAgent.isStopped = false;
        navAgent.destination = _target;
        ResetStuck();
        _localObstacleFrames = 0;
        return true;
    }

    private bool StartAgentForCurrentTarget(int mode)
    {
        if (navAgent == null || !_hasTarget) return false;
        ConfigureAgent();
        navAgent.enabled = true;
        if (!navAgent.isOnNavMesh)
        {
            NavMeshHit startHit;
            if (!NavMesh.SamplePosition(npcRoot.position, out startHit, 0.5f, NavMesh.AllAreas)
                || !navAgent.Warp(startHit.position)) { DisableAgent(); return false; }
        }
        navAgent.speed = _cruise;
        navAgent.stoppingDistance = mode == MODE_FOLLOW
            ? followDistance
            : (mode == MODE_ORBIT ? _orbitStoppingDistance : _arrivalDistance);
        navAgent.autoBraking = mode == MODE_FOLLOW;
        navAgent.updateRotation = mode != MODE_ORBIT || !_orbitFaceTarget;
        navAgent.isStopped = false;
        navAgent.destination = _target;
        _mode = mode;
        return true;
    }

    public void SetSpeed(float speed)
    {
        _cruise = Mathf.Clamp(speed, 0.1f, maxSpeed);
        if (navAgent != null && navAgent.enabled) navAgent.speed = _cruise;
    }

    public void TurnTo(float yaw)
    {
        if (_mode == MODE_ESTOP) return;
        StopMovement();
        _turnYaw = Mathf.Repeat(yaw, 360f);
    }

    public void LookAt(int slot)
    {
        _lookSlot = slot;
        _hasLookPoint = false;
        _lookPointUntil = -1f;
    }

    public void LookAtPoint(Vector3 point, float weight, float durationSeconds, bool allowBodyAssist)
    {
        if (_mode == MODE_ESTOP) return;
        _lookPoint = point;
        _hasLookPoint = weight > 0f;
        _lookBodyAssist = allowBodyAssist;
        _lookPointUntil = durationSeconds > 0f ? Time.timeSinceLevelLoad + durationSeconds : -1f;
        _lookSlot = -1;
    }

    public void PlayAction(int actionId, int actionSeq, bool loop, int durationMs, string layer)
    {
        if (_mode == MODE_ESTOP) return;
        _actionId = actionId;
        _actionSeq = actionSeq;
        _actionLoop = loop;
        _actionStartedAt = Time.timeSinceLevelLoad;
        _actionStartedServerMs = Networking.GetServerTimeInMilliseconds();
        _actionExpectedEnd = loop ? -1f : _actionStartedAt + Mathf.Max(1, durationMs) / 1000f;
        _actionLayer = layer == "full_body" ? 2 : 1;
        ApplyAnimator();
    }

    public void StopAction()
    {
        _actionId = -1;
        _actionSeq = 0;
        _actionLoop = false;
        _actionStartedAt = 0f;
        _actionStartedServerMs = 0;
        _actionExpectedEnd = -1f;
        ApplyAnimator();
    }

    public void SetExpression(int expressionId, float weight, int durationMs)
    {
        _expressionId = expressionId;
        _expressionWeight = Mathf.Clamp01(weight);
        _expressionUntil = durationMs > 0 ? Time.timeSinceLevelLoad + durationMs / 1000f : -1f;
        ApplyAnimator();
    }

    public void ClearExpression()
    {
        _expressionId = 0;
        _expressionWeight = 0f;
        _expressionUntil = -1f;
        ApplyAnimator();
    }

    public void Stop()
    {
        if (_mode == MODE_ESTOP) return;
        StopMovement();
        ClearLookInternal();
        StopAction();
        ClearExpression();
    }

    public void Estop()
    {
        DisableAgent();
        ClearOrbit();
        _mode = MODE_ESTOP;
        _hasTarget = false;
        _finishingGotoYaw = false;
        _turnYaw = -1f;
        _followSlot = -1;
        ClearLookInternal();
        StopAction();
        ClearExpression();
        _speed = 0f;
        _vel = Vector3.zero;
        ApplyAnimator();
    }

    public void ClearEstop()
    {
        if (_mode != MODE_ESTOP) return;
        _mode = MODE_IDLE;
        _speed = 0f;
        _vel = Vector3.zero;
        ApplyAnimator();
    }

    public void WatchdogIdle()
    {
        if (_mode == MODE_ESTOP) return;
        StopMovement();
        ClearLookInternal();
        StopAction();
        ClearExpression();
        _speed = 0f;
        ApplyAnimator();
    }

    public void SetStateRate(int level) { stateRateLevel = Mathf.Clamp(level, 0, 3); }
    public int GetMode() { return _mode; }
    public int GetAnimId() { return _actionId; }
    public int GetActionId() { return _actionId; }
    public int GetActionSeq() { return _actionSeq; }
    public bool GetActionLoop() { return _actionLoop; }
    public int GetActionStartedServerMs() { return _actionStartedServerMs; }
    public int GetExpressionId() { return _expressionId; }
    public float GetExpressionWeight() { return _expressionWeight; }
    public float GetSpeed() { return _speed; }

    void Update()
    {
        bool driver = telemetry != null && telemetry.IsDriver();
        bool sessionReady = router != null && router.GetSession() != 0;
        bool owner = npcRoot != null && Networking.IsOwner(npcRoot.gameObject);
        if (!driver || !sessionReady || !owner)
        {
            DisableAgent();
            if (driver && sessionReady && !owner)
            {
                if (_notOwnerSince < 0f) _notOwnerSince = Time.timeSinceLevelLoad;
                else if (!_warnedNotOwner && Time.timeSinceLevelLoad - _notOwnerSince > 3f)
                {
                    _warnedNotOwner = true;
                    telemetry.EmitErr("not_owner", "driver 未持有 NPC 根 ownership");
                }
            }
            return;
        }
        _notOwnerSince = -1f;
        _warnedNotOwner = false;

        float dt = Time.deltaTime;
        if (dt <= 0f) return;
        float now = Time.timeSinceLevelLoad;

        if (_lookPointUntil > 0f && now >= _lookPointUntil)
        {
            _lookPointUntil = -1f;
            _hasLookPoint = false;
            if (router != null) router.OnLookCompleted("expired");
        }
        if (_lookSlot >= 0 && perception != null)
        {
            VRCPlayerApi watched = perception.PlayerOfSlot(_lookSlot);
            if (watched == null || !watched.IsValid())
            {
                _lookSlot = -1;
                if (router != null) router.OnLookCancelled("target_left");
            }
        }

        if (_actionId >= 0 && !_actionLoop && ActionNaturallyFinished(now))
        {
            StopAction();
            if (router != null) router.OnActionFinished();
        }
        if (_expressionUntil > 0f && now >= _expressionUntil)
        {
            ClearExpression();
            if (router != null) router.OnExpressionExpired();
        }

        if (_mode == MODE_ESTOP)
        {
            DisableAgent();
            _speed = 0f;
            _vel = Vector3.zero;
        }
        else Step(dt, now);

        float hz = stateRateLevel == 1 ? 1f : (stateRateLevel == 2 ? 5f : (stateRateLevel == 3 ? 10f : 0f));
        if (hz > 0f)
        {
            _stateTimer += dt;
            if (_stateTimer >= 1f / hz) { _stateTimer = 0f; EmitState(); }
        }
    }

    private bool ActionNaturallyFinished(float now)
    {
        if (animator != null && _actionLayer >= 0 && _actionLayer < animator.layerCount && now - _actionStartedAt > 0.1f)
        {
            AnimatorStateInfo info = animator.GetCurrentAnimatorStateInfo(_actionLayer);
            if (info.IsName("Action_" + _actionId) && !animator.IsInTransition(_actionLayer) && info.normalizedTime >= 1f) return true;
        }
        return _actionExpectedEnd > 0f && now >= _actionExpectedEnd + 0.25f;
    }

    private void Step(float dt, float now)
    {
        if (_mode == MODE_FOLLOW) StepFollow(now);
        if ((_mode == MODE_GOTO || _mode == MODE_FOLLOW || _mode == MODE_WANDER || _mode == MODE_ORBIT) && _hasTarget)
            StepNavMovement(dt);

        StepRotation(dt);
        Vector3 after = npcRoot.position;
        _vel = (after - _lastPos) / dt;
        _lastPos = after;
        if (_mode == MODE_IDLE) _speed = new Vector2(_vel.x, _vel.z).magnitude;
        _grounded = navAgent != null && navAgent.enabled && navAgent.isOnNavMesh;
        if (!_grounded)
        {
            RaycastHit hit;
            _grounded = Physics.Raycast(after + Vector3.up * 0.2f, Vector3.down, out hit, 0.5f, environmentMask.value, QueryTriggerInteraction.Ignore);
        }
        ApplyAnimator();
    }

    private void StepFollow(float now)
    {
        if (now < _nextFollowRefresh) return;
        _nextFollowRefresh = now + followRefreshSec;
        VRCPlayerApi player = perception == null ? null : perception.PlayerOfSlot(_followSlot);
        if (player == null || !player.IsValid()) { BlockMovement("target_left", "target_left"); return; }
        NavMeshHit hit;
        if (!NavMesh.SamplePosition(player.GetPosition(), out hit, 0.75f, NavMesh.AllAreas)) { BlockMovement("no_path", "stuck"); return; }
        _target = hit.position;
        if (navAgent != null && navAgent.enabled && navAgent.isOnNavMesh) navAgent.destination = _target;
    }

    private void StepNavMovement(float dt)
    {
        if (navAgent == null || !navAgent.enabled || !navAgent.isOnNavMesh) { BlockMovement("no_path", "stuck"); return; }
        navAgent.speed = _cruise;
        if (_mode == MODE_ORBIT)
        {
            _orbitTravelMeters += FlatDistance(npcRoot.position, _orbitLastPosition);
            _orbitLastPosition = npcRoot.position;
        }
        float flatDistance = FlatDistance(npcRoot.position, _target);
        float targetDistance = _mode == MODE_GOTO ? Vector3.Distance(npcRoot.position, _target) : flatDistance;
        _speed = navAgent.velocity.magnitude;
        float arrival = _mode == MODE_FOLLOW
            ? followDistance
            : (_mode == MODE_ORBIT ? _orbitStoppingDistance : _arrivalDistance);
        if (_mode == MODE_GOTO && !navAgent.pathPending && !navAgent.autoBraking
            && flatDistance <= GotoBrakeDistance())
            navAgent.autoBraking = true;
        if (_mode == MODE_WANDER && !navAgent.pathPending
            && flatDistance <= Mathf.Max(stopDistance + 0.05f, wanderSwitchDistance))
        {
            AdvanceWanderWaypoint(flatDistance);
            return;
        }
        if (_mode == MODE_ORBIT && !navAgent.pathPending)
        {
            bool finalPoint = _orbitIndex >= _orbitPointCount - 1;
            float waypointTravel = FlatDistance(npcRoot.position, _orbitWaypointStart);
            if (!finalPoint && flatDistance <= _orbitSwitchDistance && waypointTravel >= _orbitMinAdvanceDistance)
            {
                AdvanceOrbitWaypoint();
                return;
            }
            if (finalPoint && !navAgent.autoBraking && flatDistance <= GotoBrakeDistance()) navAgent.autoBraking = true;
        }
        if (!navAgent.pathPending && targetDistance <= arrival && _speed < 0.05f)
        {
            if (_mode == MODE_FOLLOW) { navAgent.isStopped = true; return; }
            if (_mode == MODE_ORBIT && _orbitIndex < _orbitPointCount - 1)
            {
                // 被 NavMesh 压近的中间点允许直接跳过；中间点不能进入最终圈长
                // 取证分支，否则紧接的反向绕行会在启动帧被误报 stuck。
                AdvanceOrbitWaypoint();
                return;
            }
            if (_mode == MODE_ORBIT && _orbitTravelMeters < _orbitRequiredTravelMeters)
            {
                // 防止 NavMesh 初次采样把多组圆周点压到同一小片区域后产生假成功。
                // 允许从第二个圆周点恢复一次；仍无足够实际路程则按卡住失败。
                if (!_orbitRecoveryUsed && _orbitPointCount > 1)
                {
                    _orbitRecoveryUsed = true;
                    _orbitIndex = FirstUsefulOrbitIndex(npcRoot.position);
                    _target = _orbitPoints[_orbitIndex];
                    _orbitWaypointStart = npcRoot.position;
                    navAgent.autoBraking = false;
                    navAgent.isStopped = false;
                    navAgent.destination = _target;
                    ResetStuck();
                    return;
                }
                BlockMovement("stuck", "stuck");
                return;
            }
            navAgent.isStopped = true;
            DisableAgent();
            _hasTarget = false;
            if (_mode == MODE_ORBIT) CompleteOrbit();
            else if (_targetYaw >= 0f) { _finishingGotoYaw = true; _turnYaw = _targetYaw; }
            else CompleteGoto();
        }
        else
        {
            if (_mode == MODE_FOLLOW && navAgent.isStopped) navAgent.isStopped = false;
            // GOTO/ORBIT 都在启动前完成 NavMesh 完整路径校验。楼梯踏步以及
            // 沿实体切向运动会让扇形射线产生假阳性，因此这两类确定性路径
            // 交给 NavMesh 避障，并继续保留实际位移卡住检测。射线仅用于
            // follow/wander 这类运行时目标不断变化的模式。
            if (_mode == MODE_FOLLOW || _mode == MODE_WANDER) DetectLocalObstacle();
            if (_hasTarget) DetectStuck(dt);
        }
    }

    private void AdvanceOrbitWaypoint()
    {
        if (_orbitIndex >= _orbitPointCount - 1) return;
        _orbitIndex++;
        _target = _orbitPoints[_orbitIndex];
        _orbitWaypointStart = npcRoot.position;
        navAgent.destination = _target;
        navAgent.autoBraking = false;
        navAgent.isStopped = false;
        ResetStuck();
        _localObstacleFrames = 0;
    }

    private void AdvanceWanderWaypoint(float error)
    {
        int reached = _wanderIndex;
        if (router != null) router.OnWanderWaypoint(reached, error);
        _wanderIndex = (_wanderIndex + 1) % wanderWaypoints.Length;
        string err = StartWanderTarget(_wanderIndex);
        if (err != null) BlockMovement(err == "no_path" ? "no_path" : "stuck", "stuck");
    }

    private void StepRotation(float dt)
    {
        float faceYaw = -1f;
        bool completingTurn = false;
        if (_turnYaw >= 0f) { faceYaw = _turnYaw; completingTurn = true; }
        else if (_mode == MODE_ORBIT && _orbitFaceTarget)
        {
            Vector3 d = _orbitCenter - npcRoot.position; d.y = 0f;
            if (d.sqrMagnitude > 0.01f) faceYaw = YawOf(d.normalized);
        }
        else if (_mode == MODE_IDLE && _lookSlot >= 0 && perception != null)
        {
            VRCPlayerApi player = perception.PlayerOfSlot(_lookSlot);
            if (player != null && player.IsValid())
            {
                Vector3 d = player.GetPosition() - npcRoot.position; d.y = 0f;
                if (d.sqrMagnitude > 0.01f) faceYaw = YawOf(d.normalized);
            }
        }
        else if (_mode == MODE_IDLE && _hasLookPoint && _lookBodyAssist)
        {
            Vector3 d = _lookPoint - npcRoot.position; d.y = 0f;
            if (d.sqrMagnitude > 0.01f) faceYaw = YawOf(d.normalized);
        }
        if (faceYaw < 0f) return;
        Quaternion wanted = Quaternion.Euler(0f, faceYaw, 0f);
        npcRoot.rotation = Quaternion.RotateTowards(npcRoot.rotation, wanted, turnRateDeg * dt);
        if (completingTurn && Quaternion.Angle(npcRoot.rotation, wanted) <= 2f)
        {
            _turnYaw = -1f;
            if (_finishingGotoYaw) { _finishingGotoYaw = false; CompleteGoto(); }
            else if (router != null) router.OnTurnCompleted();
        }
    }

    private void DetectStuck(float dt)
    {
        if (navAgent == null || navAgent.pathPending) { ResetStuck(); return; }
        // NavMeshAgent 从上一段自然结束的 disabled 状态重新启用时需要一小段
        // 路径稳定窗口；该窗口只抑制误报，不暂停或降低 Agent 速度。
        if (_mode == MODE_ORBIT && Time.timeSinceLevelLoad < _orbitStuckGraceUntil) { ResetStuck(); return; }
        bool expectsMotion = navAgent.desiredVelocity.magnitude > 0.1f && FlatDistance(npcRoot.position, _target) > navAgent.stoppingDistance;
        if (!expectsMotion) { ResetStuck(); return; }
        _stuckTimer += dt;
        if (_stuckTimer < stuckSeconds) return;
        Vector3 moved = npcRoot.position - _stuckAnchor; moved.y = 0f;
        if (moved.magnitude < stuckMinMove) BlockMovement("stuck", "stuck");
        ResetStuck();
    }

    private void DetectLocalObstacle()
    {
        if (navAgent == null || navAgent.desiredVelocity.sqrMagnitude < 0.01f) { _localObstacleFrames = 0; return; }
        Vector3 direction = navAgent.desiredVelocity; direction.y = 0f; direction.Normalize();
        Vector3 origin = npcRoot.position + Vector3.up * whiskerHeight;
        Vector3 left = Quaternion.Euler(0f, -whiskerAngleDeg, 0f) * direction;
        Vector3 right = Quaternion.Euler(0f, whiskerAngleDeg, 0f) * direction;
        bool blocked = Whisker(origin, direction) >= 0f && Whisker(origin, left) >= 0f && Whisker(origin, right) >= 0f;
        _localObstacleFrames = blocked ? _localObstacleFrames + 1 : 0;
        if (_localObstacleFrames >= 2) BlockMovement("local_obstacle", "local_obstacle");
    }

    private float Whisker(Vector3 origin, Vector3 direction)
    {
        RaycastHit hit;
        return Physics.Raycast(origin, direction, out hit, whiskerLength, environmentMask.value, QueryTriggerInteraction.Ignore) ? hit.distance : -1f;
    }

    private void CompleteGoto()
    {
        Vector3 completedTarget = _target;
        _mode = MODE_IDLE;
        _speed = 0f;
        ResetStuck();
        if (router != null) router.OnGotoArrived(completedTarget, Vector3.Distance(npcRoot.position, completedTarget));
    }

    private void CompleteOrbit()
    {
        int completedPoints = _orbitPointCount;
        _mode = MODE_IDLE;
        _speed = 0f;
        ResetStuck();
        ClearOrbit();
        if (router != null) router.OnOrbitCompleted(completedPoints);
    }

    private void BlockMovement(string blockedReason, string cancelReason)
    {
        Vector3 blockedTarget = _target;
        DisableAgent();
        _hasTarget = false;
        _finishingGotoYaw = false;
        _turnYaw = -1f;
        _followSlot = -1;
        ClearOrbit();
        _mode = MODE_IDLE;
        _speed = 0f;
        _localObstacleFrames = 0;
        ResetStuck();
        if (router != null) router.OnMovementBlocked(blockedReason, cancelReason, blockedTarget);
    }

    private void StopMovement()
    {
        DisableAgent();
        _mode = MODE_IDLE;
        _hasTarget = false;
        _finishingGotoYaw = false;
        _turnYaw = -1f;
        _followSlot = -1;
        ClearOrbit();
        _speed = 0f;
        _localObstacleFrames = 0;
        ResetStuck();
    }

    private void DisableAgent()
    {
        if (navAgent != null && navAgent.enabled) { navAgent.isStopped = true; navAgent.enabled = false; }
    }

    private void ClearOrbit()
    {
        _orbitPointCount = 0;
        _orbitIndex = 0;
        _orbitFaceTarget = false;
        _orbitMinAdvanceDistance = 0f;
        _orbitStoppingDistance = 0f;
        _orbitTravelMeters = 0f;
        _orbitRequiredTravelMeters = 0f;
        _orbitRecoveryUsed = false;
        _orbitStuckGraceUntil = -1f;
        if (navAgent != null) navAgent.updateRotation = true;
    }

    private int FirstUsefulOrbitIndex(Vector3 position)
    {
        float minimumDistance = Mathf.Max(_orbitSwitchDistance, _orbitMinAdvanceDistance) + 0.02f;
        int last = Mathf.Max(0, _orbitPointCount - 1);
        for (int i = 0; i < last; i++)
        {
            if (FlatDistance(position, _orbitPoints[i]) > minimumDistance) return i;
        }
        return last;
    }

    private void ClearLookInternal()
    {
        _lookSlot = -1;
        _hasLookPoint = false;
        _lookPointUntil = -1f;
    }

    private void ResetStuck()
    {
        _stuckTimer = 0f;
        if (npcRoot != null) _stuckAnchor = npcRoot.position;
    }

    private void ApplyAnimator()
    {
        if (animator == null) return;
        animator.SetFloat("Speed", _speed);
        animator.SetInteger("ActionId", _actionId);
        animator.SetInteger("ActionSeq", _actionSeq);
        animator.SetBool("ActionLoop", _actionLoop);
        animator.SetInteger("ExpressionId", _expressionId);
        animator.SetFloat("ExpressionWeight", _expressionWeight);
        animator.SetBool("Estop", _mode == MODE_ESTOP);
    }

    private float FlatDistance(Vector3 a, Vector3 b)
    {
        float dx = a.x - b.x; float dz = a.z - b.z;
        return Mathf.Sqrt(dx * dx + dz * dz);
    }

    private float GotoBrakeDistance()
    {
        float safeAccel = Mathf.Max(0.1f, accel);
        return stopDistance + (_cruise * _cruise) / (2f * safeAccel) + 0.05f;
    }

    private float YawOf(Vector3 direction) { return Mathf.Repeat(Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg, 360f); }
    public float CurrentYaw() { return Mathf.Repeat(npcRoot.eulerAngles.y, 360f); }

    public string ModeName()
    {
        if (_mode == MODE_FOLLOW) return "follow";
        if (_mode == MODE_GOTO || _finishingGotoYaw) return "goto";
        if (_mode == MODE_WANDER) return "wander";
        if (_mode == MODE_ORBIT) return "orbit";
        return "idle";
    }

    private void EmitState()
    {
        if (router == null || telemetry == null) return;
        string json = BuildStateBody(router.StateName(router.GetControlState()), router.GetControlState() == NekoMidiRouter.STATE_ESTOP, false);
        telemetry.Emit("npc.state", json.Substring(1, json.Length - 2));
    }

    public string BuildStateBody(string controlState, bool estop, bool includeSnapshotExtras)
    {
        int slot = router == null ? -1 : router.GetTargetSlot();
        string activeOps = router == null ? "[]" : router.ActiveOpsJson();
        string body = "{\"pos\":" + telemetry.Vec3(npcRoot.position)
            + ",\"yaw\":" + telemetry.F1(CurrentYaw())
            + ",\"vel\":[" + telemetry.F2(_vel.x) + "," + telemetry.F2(_vel.z) + "]"
            + ",\"speed\":" + telemetry.F2(_speed)
            + ",\"state\":" + telemetry.J(controlState)
            + ",\"mode\":" + telemetry.J(ModeName())
            + ",\"grounded\":" + telemetry.B(_grounded)
            + ",\"target_slot\":" + (slot < 0 ? "null" : slot.ToString())
            + ",\"action_id\":" + (_actionId < 0 ? "null" : _actionId.ToString())
            + ",\"action_seq\":" + (_actionSeq <= 0 ? "null" : _actionSeq.ToString())
            + ",\"expression_id\":" + (_expressionId < 0 ? "null" : _expressionId.ToString())
            + ",\"estop\":" + telemetry.B(estop)
            + ",\"active_ops\":" + activeOps;
        if (includeSnapshotExtras)
            body += ",\"target_pos\":" + (_hasTarget || _finishingGotoYaw ? telemetry.Vec3(_target) : "null")
                + ",\"action_started_at_server_ms\":" + (_actionStartedServerMs > 0 ? _actionStartedServerMs.ToString() : "null")
                + ",\"text_transfer_seq\":" + (router == null || router.CurrentTextTransferSeq() <= 0 ? "null" : router.CurrentTextTransferSeq().ToString());
        return body + "}";
    }
}
