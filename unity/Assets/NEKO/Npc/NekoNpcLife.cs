/*
 * NekoNpcLife —— YUI NPC 的"活人感"层（UdonSharp, 不同步）
 *
 * 在每个客户端本地运行，不改变协议 v1.3 的任何指令/遥测：
 *   · 注视：LateUpdate 里把头/颈/眼睛叠加转向目标——优先 Locomotion 的显式注视目标（LOOK_AT / 注视点），
 *     否则看向 lookRange 内最近的玩家；没人时偶尔随机张望。走路时幅度减小，播放动作时只保留少量注视。
 *   · 空闲小动作：站着不动、没有动作在播、未急停时，随机触发 Animator 的 Fidget_<id> 状态
 *     （FidgetId 参数，由 NekoYuiFormalAnimatorUpgrader 依据 YuiFormalAnimatorMap.json 的 idle_fidgets 生成）；
 *     名牌气泡只提示对话活跃；说话手势与休息交替，不代表真实音频活动。
 *
 * 读取的都是 Animator 参数（Speed / ActionId / Estop），driver 由 NekoNpcLocomotion 写入、远端由 NekoNpcSync 投影，
 * 因此 driver 与远端客户端逻辑一致；随机小动作允许各客户端不同（纯观感，不进遥测）。
 */
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoNpcLife : UdonSharpBehaviour
{
    [Header("依赖")]
    public Animator animator;
    public Transform npcRoot;
    public NekoNpcLocomotion locomotion;
    public NekoNpcNameplate nameplate;
    public NekoNpcSync sync;

    [Header("协调动作（由动画构建器写入）")]
    public bool coordinatedMotion;
    public float[] actionEnterSeconds = new float[0];
    public float[] actionExitSeconds = new float[0];
    public float[] actionLookYaw = new float[0];
    public float[] actionLookPitch = new float[0];
    public float targetHoldSeconds = 1.5f;
    public float targetSwitchAdvantage = 0.8f;
    public float talkingRestMin = 0.8f;
    public float talkingRestMax = 1.8f;

    [Header("注视")]
    public bool enableLook = true;
    [Tooltip("自动注视最近玩家的距离（m）")]
    public float lookRange = 4.5f;
    public float lookYawLimit = 60f;
    public float lookPitchLimit = 25f;
    [Tooltip("目标在身后超过此角度就放弃注视（不扭头）")]
    public float lookAbandonYaw = 110f;
    public float lookBlendIn = 3f;
    public float lookBlendOut = 1.5f;
    [Tooltip("注视方向平滑（越大越快）")]
    public float lookTurnSharpness = 7f;
    public float headShare = 0.65f;
    public float neckShare = 0.35f;
    public float eyeAngleLimit = 18f;
    public float eyeWeight = 0.8f;
    [Tooltip("走动时注视权重与角度缩放")]
    public float walkingLookScale = 0.6f;
    [Tooltip("播放动作（ActionId≥0）时注视权重缩放，避免盖掉点头/摇头")]
    public float actionLookScale = 0.3f;

    [Header("空闲张望（附近没人时）")]
    public float idleGazeMinInterval = 5f;
    public float idleGazeMaxInterval = 12f;
    public float idleGazeHoldMin = 1.5f;
    public float idleGazeHoldMax = 3.5f;
    public float idleGazeYaw = 45f;
    public float idleGazeWeight = 0.55f;

    [Header("空闲小动作（Animator FidgetId）")]
    public bool enableFidgets = true;
    public float fidgetMinInterval = 8f;
    public float fidgetMaxInterval = 18f;
    public int talkingFidgetId = 1;
    public int interactFidgetId = 2;
    [Tooltip("随机小动作里选『说话手势』的概率，其余为 interact")]
    public float talkingChance = 0.45f;
    public float talkingHoldMin = 3f;
    public float talkingHoldMax = 6f;
    public float interactDuration = 2.0f;
    public string locomotionStateName = "YUI_Procedural_Locomotion";

    private Transform _head;
    private Transform _neck;
    private Transform _eyeL;
    private Transform _eyeR;
    private VRCPlayerApi[] _players = new VRCPlayerApi[16];

    private float _lookWeight;
    private float _currentActionLookScale = 1f;
    private Vector3 _lookDir = Vector3.zero;
    private float _gazeUntil = -1f;
    private float _nextGazeAt;
    private Vector3 _gazeLocalDir = Vector3.forward;

    private int _fidget;
    private float _fidgetUntil = -1f;
    private float _nextFidgetAt;
    private int _lastFidget;
    private float _nextTalkingAt;
    private VRCPlayerApi _watchedPlayer;
    private float _watchUntil;
    private float _nextPlayerScan;
    private Vector3 _eyeDir;
    private Vector3 _motionLastPos;
    private float _motionLastYaw;
    private bool _bodyWasOwned;
    private int _presentedAction = -1;
    private int _presentedSeq;
    private float _nextProgressCheck;
    private float _gazeYawScale = 1f;
    private float _gazePitchScale = 1f;

    void Start()
    {
        if (npcRoot == null) npcRoot = transform.parent != null ? transform.parent : transform;
        if (animator == null) animator = npcRoot.GetComponentInChildren<Animator>();
        if (sync == null && npcRoot != null) sync = npcRoot.GetComponent<NekoNpcSync>();
        _motionLastPos = npcRoot.position;
        _motionLastYaw = npcRoot.eulerAngles.y;
        if (animator != null)
        {
            _head = animator.GetBoneTransform(HumanBodyBones.Head);
            _neck = animator.GetBoneTransform(HumanBodyBones.Neck);
            _eyeL = animator.GetBoneTransform(HumanBodyBones.LeftEye);
            _eyeR = animator.GetBoneTransform(HumanBodyBones.RightEye);
        }
        _nextFidgetAt = Time.timeSinceLevelLoad + Random.Range(fidgetMinInterval, fidgetMaxInterval) * 0.5f;
        _nextGazeAt = Time.timeSinceLevelLoad + Random.Range(idleGazeMinInterval, idleGazeMaxInterval);
    }

    void Update()
    {
        // ARDY 持有身体骨骼期间，旧动作表现停止写入。
        if (animator == null) return;
        if (locomotion != null && (locomotion.externalPoseActive || locomotion.poseTransitionActive)) {
            // 保持只读测速基准，否则交回时会把整段ARDY位移误算成单帧速度。
            _bodyWasOwned=true;
            if (npcRoot != null) { _motionLastPos=npcRoot.position; _motionLastYaw=npcRoot.eulerAngles.y; }
            return;
        }
        if (_bodyWasOwned) {
            _bodyWasOwned=false;
            if (npcRoot != null) { _motionLastPos=npcRoot.position; _motionLastYaw=npcRoot.eulerAngles.y; }
            // 丢弃接管前已过期的小动作计时，不在交接首帧补播。
            ClearFidget(Time.timeSinceLevelLoad);
        }
        if (coordinatedMotion) PresentMotion();
        float now = Time.timeSinceLevelLoad;
        float speed = animator.GetFloat("Speed");
        int actionId = animator.GetInteger("ActionId");
        bool estop = animator.GetBool("Estop");
        bool speaking = nameplate != null && nameplate.IsBubbleVisible();
        StepFidget(now, speed, actionId, estop, speaking);
    }

    private void StepFidget(float now, float speed, int actionId, bool estop, bool speaking)
    {
        if (!enableFidgets)
        {
            if (_fidget != 0) ClearFidget(now);
            return;
        }
        bool canIdle = speed < 0.05f && actionId < 0 && !estop
            && (!coordinatedMotion || Mathf.Abs(animator.GetFloat("Turn")) < 0.12f);
        if (!canIdle)
        {
            if (_fidget != 0) ClearFidget(now);
            return;
        }
        if (coordinatedMotion && speaking && _fidget >= 3)
        {
            ClearFidget(now);
            animator.CrossFadeInFixedTime(locomotionStateName, 0.2f, 0);
        }
        if (_fidget != 0)
        {
            if (_fidget == talkingFidgetId)
            {
                if (now >= _fidgetUntil) ClearFidget(now);
                return;
            }
            // 一次性小动作：状态机一进入就把参数清零（Animator 只按 exit time 退出，不会重入）。
            if (animator.GetInteger("FidgetId") != 0 && animator.GetCurrentAnimatorStateInfo(0).IsName("Fidget_" + _fidget))
                animator.SetInteger("FidgetId", 0);
            if (now >= _fidgetUntil) ClearFidget(now);
            return;
        }
        if (speaking && now >= _nextTalkingAt)
        {
            if (!animator.GetCurrentAnimatorStateInfo(0).IsName(locomotionStateName) || animator.IsInTransition(0)) return;
            StartFidget(talkingFidgetId, now, now + Random.Range(1f, 2.5f));
            return;
        }
        if (speaking) return;
        if (now < _nextFidgetAt) return;
        AnimatorStateInfo info = animator.GetCurrentAnimatorStateInfo(0);
        if (!info.IsName(locomotionStateName) || animator.IsInTransition(0)) return;
        if (coordinatedMotion)
        {
            int id = Random.Range(3, 6);
            if (id == _lastFidget) id = 3 + (id - 2) % 3;
            StartFidget(id, now, now + 2.8f);
        }
        else if (Random.value < talkingChance) StartFidget(talkingFidgetId, now, now + Random.Range(talkingHoldMin, talkingHoldMax));
        else StartFidget(interactFidgetId, now, now + interactDuration + 0.2f);
    }

    private void StartFidget(int id, float now, float until)
    {
        _fidget = id;
        _lastFidget = id;
        _fidgetUntil = until;
        animator.SetInteger("FidgetId", id);
    }

    private void ClearFidget(float now)
    {
        _fidget = 0;
        _fidgetUntil = -1f;
        animator.SetInteger("FidgetId", 0);
        _nextFidgetAt = now + Random.Range(fidgetMinInterval, fidgetMaxInterval);
        _nextTalkingAt = now + Random.Range(talkingRestMin, talkingRestMax);
    }

    public int GetFidgetId() { return _fidget; }
    public float GetLookWeight() { return _lookWeight; }

    // 所有客户端只在操作身份变化时进入动作，普通同步包不重启动画。
    private void PresentMotion()
    {
        float dt = Time.deltaTime;
        if (dt <= 0f || npcRoot == null) return;
        Vector3 pos = npcRoot.position;
        float yaw = npcRoot.eulerAngles.y;
        Vector3 velocity = (pos - _motionLastPos) / dt;
        if ((pos - _motionLastPos).sqrMagnitude > 1f) velocity = Vector3.zero;
        Vector3 local = npcRoot.InverseTransformDirection(velocity);
        bool estop = animator.GetBool("Estop");
        float turn = Mathf.Clamp(Mathf.DeltaAngle(_motionLastYaw, yaw) / dt / 180f, -1f, 1f);
        animator.SetFloat("MoveForward", estop ? 0f : Mathf.Clamp(local.z, -2f, 2f), 0.12f, dt);
        animator.SetFloat("MoveSide", estop ? 0f : Mathf.Clamp(local.x, -2f, 2f), 0.12f, dt);
        animator.SetFloat("Turn", estop ? 0f : turn, 0.1f, dt);
        // 短侧步靠步频匹配速度，避免扩大幅度后交叉腿。
        float cycleRate = 1f;
        if (Mathf.Abs(local.x) > Mathf.Abs(local.z) * 1.2f) cycleRate = Mathf.Max(1f, local.magnitude / 0.35f);
        else if (local.z < -0.05f) cycleRate = Mathf.Max(1f, local.magnitude / 0.8f);
        animator.SetFloat("MoveCycleRate", Mathf.Clamp(cycleRate, 1f, 3f), 0.12f, dt);
        _motionLastPos = pos;
        _motionLastYaw = yaw;

        int id = estop ? -1 : animator.GetInteger("ActionId");
        int seq = animator.GetInteger("ActionSeq");
        int layer = id == 4 || id == 7 ? 2 : 1;
        bool owner = Networking.IsOwner(npcRoot.gameObject);
        int started = owner ? (locomotion != null ? locomotion.GetActionStartedServerMs() : 0)
            : (sync != null ? sync.syncActionStartedAtServerMs : 0);
        float elapsed = started == 0 ? 0f : Mathf.Max(0f, (Networking.GetServerTimeInMilliseconds() - started) * 0.001f);
        float duration = id >= 0 && locomotion != null && id < locomotion.animDurations.Length ? locomotion.animDurations[id] : 0f;
        if (id != _presentedAction || (id >= 0 && seq != _presentedSeq))
        {
            if (_presentedAction >= 0)
            {
                int previousLayer = _presentedAction == 4 || _presentedAction == 7 ? 2 : 1;
                if (id < 0 || previousLayer != layer)
                    animator.CrossFadeInFixedTime("Empty", estop ? 0f : ValueAt(actionExitSeconds, _presentedAction, 0.25f), previousLayer);
            }
            if (id >= 0)
                animator.CrossFadeInFixedTime("Action_" + id, ValueAt(actionEnterSeconds, id, 0.2f), layer,
                    duration > 0f ? Mathf.Min(elapsed, duration) : 0f);
            _presentedAction = id;
            _presentedSeq = seq;
            _nextProgressCheck = Time.timeSinceLevelLoad + 0.5f;
        }
        else if (!owner && id >= 0 && duration > 0f && Time.timeSinceLevelLoad >= _nextProgressCheck)
        {
            _nextProgressCheck = Time.timeSinceLevelLoad + 0.5f;
            AnimatorStateInfo state = animator.GetCurrentAnimatorStateInfo(layer);
            float target = animator.GetBool("ActionLoop") ? elapsed / duration : Mathf.Clamp01(elapsed / duration);
            if (!animator.IsInTransition(layer) && state.IsName("Action_" + id) && Mathf.Abs(state.normalizedTime - target) * duration > 0.4f)
                animator.CrossFadeInFixedTime("Action_" + id, 0.12f, layer, target * duration);
        }
    }

    private float ValueAt(float[] values, int id, float fallback)
    {
        return values != null && id >= 0 && id < values.Length ? values[id] : fallback;
    }

    void LateUpdate()
    {
        // 注视不能在 ARDY 最终骨架上重复叠加。
        if (locomotion != null && (locomotion.externalPoseActive || locomotion.poseTransitionActive)) { _lookWeight = 0f; return; }
        if (!enableLook || animator == null || _head == null || npcRoot == null) return;
        float dt = Time.deltaTime;
        float now = Time.timeSinceLevelLoad;
        if (animator.GetBool("Estop")) { _lookWeight = 0f; _currentActionLookScale = 1f; return; }
        float speed = animator.GetFloat("Speed");
        int actionId = animator.GetInteger("ActionId");
        Vector3 headPos = _head.position;
        _gazeYawScale = Mathf.MoveTowards(_gazeYawScale, ValueAt(actionLookYaw, actionId, 1f), 4f * dt);
        _gazePitchScale = Mathf.MoveTowards(_gazePitchScale, ValueAt(actionLookPitch, actionId, 1f), 4f * dt);

        bool hasTarget = false;
        Vector3 target = Vector3.zero;
        float weightGoal = 1f;
        if (locomotion != null && locomotion.HasLookTarget())
        {
            hasTarget = true;
            target = locomotion.GetLookTargetPosition();
        }
        else
        {
            if (now >= _nextPlayerScan)
            {
                _nextPlayerScan = now + 0.2f;
                int count = VRCPlayerApi.GetPlayerCount();
                if (_players == null || _players.Length < count) _players = new VRCPlayerApi[count + 8];
                VRCPlayerApi.GetPlayers(_players);
                float currentDistance = lookRange + 1f;
                if (Utilities.IsValid(_watchedPlayer))
                {
                    currentDistance = Vector3.Distance(_watchedPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position, headPos);
                    Vector3 watchedDirection = npcRoot.InverseTransformDirection(_watchedPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position - headPos);
                    if (Mathf.Abs(Mathf.Atan2(watchedDirection.x, watchedDirection.z) * Mathf.Rad2Deg) > lookAbandonYaw) _watchedPlayer = null;
                }
                if (currentDistance > lookRange + 0.5f) _watchedPlayer = null;
                VRCPlayerApi candidate = _watchedPlayer;
                float best = Utilities.IsValid(candidate) ? currentDistance - targetSwitchAdvantage : lookRange;
                if (!Utilities.IsValid(_watchedPlayer) || now >= _watchUntil)
                {
                    for (int i = 0; i < _players.Length; i++)
                    {
                        VRCPlayerApi player = _players[i];
                        if (!Utilities.IsValid(player)) continue;
                        Vector3 h = player.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position;
                        Vector3 direction = npcRoot.InverseTransformDirection(h - headPos);
                        if (Mathf.Abs(Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg) > lookAbandonYaw) continue;
                        float d = Vector3.Distance(h, headPos);
                        if (d < best) { best = d; candidate = player; }
                    }
                    if (candidate != _watchedPlayer) { _watchedPlayer = candidate; _watchUntil = now + targetHoldSeconds; }
                }
            }
            if (Utilities.IsValid(_watchedPlayer))
            { target = _watchedPlayer.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position; hasTarget = true; }
            weightGoal = 0.9f;
        }

        if (!hasTarget && speed < 0.3f && actionId < 0)
        {
            if (now >= _gazeUntil && now >= _nextGazeAt)
            {
                float yaw = Random.Range(-idleGazeYaw, idleGazeYaw);
                float pitch = Random.Range(-8f, 12f);
                _gazeLocalDir = Quaternion.Euler(pitch, yaw, 0f) * Vector3.forward;
                _gazeUntil = now + Random.Range(idleGazeHoldMin, idleGazeHoldMax);
                _nextGazeAt = _gazeUntil + Random.Range(idleGazeMinInterval, idleGazeMaxInterval);
            }
            if (now < _gazeUntil)
            {
                hasTarget = true;
                target = headPos + npcRoot.rotation * _gazeLocalDir * 3f;
                weightGoal = idleGazeWeight;
            }
        }

        float yawLimit = speed > 0.3f ? lookYawLimit * 0.7f : lookYawLimit;
        if (hasTarget)
        {
            Vector3 dir = target - headPos;
            if (dir.sqrMagnitude < 0.01f) hasTarget = false;
            else
            {
                Vector3 local = Quaternion.Inverse(npcRoot.rotation) * dir.normalized;
                float yaw = Mathf.Atan2(local.x, local.z) * Mathf.Rad2Deg;
                float pitch = -Mathf.Asin(Mathf.Clamp(local.y, -1f, 1f)) * Mathf.Rad2Deg;
                if (Mathf.Abs(yaw) > lookAbandonYaw) hasTarget = false;
                else
                {
                    yaw = Mathf.Clamp(yaw, -yawLimit, yawLimit);
                    pitch = Mathf.Clamp(pitch, -lookPitchLimit, lookPitchLimit);
                    if (coordinatedMotion) { yaw *= _gazeYawScale; pitch *= _gazePitchScale; }
                    Vector3 clamped = npcRoot.rotation * (Quaternion.Euler(pitch, yaw, 0f) * Vector3.forward);
                    _eyeDir = _eyeDir.sqrMagnitude < 0.5f ? npcRoot.forward : _eyeDir;
                    _eyeDir = Vector3.Slerp(_eyeDir, clamped, 1f - Mathf.Exp(-20f * dt));
                    if (_lookWeight < 0.01f || _lookDir.sqrMagnitude < 0.5f) _lookDir = clamped;
                    else _lookDir = Vector3.Slerp(_lookDir, clamped, 1f - Mathf.Exp(-lookTurnSharpness * dt));
                }
            }
        }

        float scale = 1f;
        if (speed > 0.3f) scale *= walkingLookScale;
        // 先平滑动作缩放目标，再由原有注视权重阻尼跟随；速率单位是权重/秒。
        float targetActionLookScale = actionId >= 0 && !coordinatedMotion ? Mathf.Clamp01(actionLookScale) : 1f;
        _currentActionLookScale = Mathf.MoveTowards(_currentActionLookScale, targetActionLookScale, 4f * dt);
        scale *= _currentActionLookScale;
        float goal = hasTarget ? weightGoal * scale : 0f;
        _lookWeight = Mathf.MoveTowards(_lookWeight, goal, (goal > _lookWeight ? lookBlendIn : lookBlendOut) * dt);
        if (_lookWeight <= 0.001f || _lookDir.sqrMagnitude < 0.5f) return;

        Quaternion delta = Quaternion.LookRotation(_lookDir, Vector3.up) * Quaternion.Inverse(Quaternion.LookRotation(npcRoot.forward, Vector3.up));
        if (_neck != null)
        {
            Quaternion neckDelta = Quaternion.Slerp(Quaternion.identity, delta, _lookWeight * neckShare);
            _neck.rotation = neckDelta * _neck.rotation;
        }
        Quaternion headDelta = Quaternion.Slerp(Quaternion.identity, delta, _lookWeight * headShare);
        _head.rotation = headDelta * _head.rotation;

        if (_eyeL == null && _eyeR == null) return;
        Quaternion applied = Quaternion.Slerp(Quaternion.identity, delta, _lookWeight * (neckShare + headShare));
        Vector3 headForward = applied * npcRoot.forward;
        Quaternion eyeDelta = Quaternion.LookRotation(coordinatedMotion && _eyeDir.sqrMagnitude > 0.5f ? _eyeDir : _lookDir, Vector3.up) * Quaternion.Inverse(Quaternion.LookRotation(headForward, Vector3.up));
        float angle = Quaternion.Angle(Quaternion.identity, eyeDelta);
        if (angle > eyeAngleLimit && angle > 0.001f) eyeDelta = Quaternion.Slerp(Quaternion.identity, eyeDelta, eyeAngleLimit / angle);
        eyeDelta = Quaternion.Slerp(Quaternion.identity, eyeDelta, eyeWeight * _lookWeight);
        if (_eyeL != null) _eyeL.rotation = eyeDelta * _eyeL.rotation;
        if (_eyeR != null) _eyeR.rotation = eyeDelta * _eyeR.rotation;
    }
}
