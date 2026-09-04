/*
 * NekoNpcChatInput —— 本地跟随玩家、可随时唤起的世界内聊天输入框。
 *
 * UI 位置和开关只在各客户端本地更新；默认按 T 呼出、Enter 发送、Esc 关闭。
 * 提交内容通过带参数的网络事件仅发给
 * NPC 根对象当前 owner（driver），由 driver 唯一写入 player.chat_submit 日志。
 */
using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDK3.UdonNetworkCalling;
using VRC.SDKBase;
using VRC.Udon.Common.Interfaces;

[UdonBehaviourSyncMode(BehaviourSyncMode.NoVariableSync)]
public class NekoNpcChatInput : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;
    public NekoNpcPerception perception;
    public InputField inputField;
    public Text statusText;
    public Transform panelRoot;
    public VRC.SDK3.Components.VRCStation inputLockStation;

    [Header("本地跟随")]
    public Vector3 panelHeadOffset = new Vector3(-0.08f, -0.10f, 0.50f);

    [Header("本地按键")]
    public KeyCode openKey = KeyCode.T;

    [Header("输入限制")]
    public int maxCharacters = 144;
    public float submitCooldownSec = 2f;

    private const int MaxSlots = 64;
    private int[] _lastPidBySlot = new int[MaxSlots];
    private int[] _lastSequenceBySlot = new int[MaxSlots];
    private float[] _lastAcceptedAtBySlot = new float[MaxSlots];
    private int _nextClientSequence = 1;
    private float _nextLocalSubmitAt;
    private int _reportedReadySession;
    private VRCPlayerApi _localPlayer;
    private bool _inputLocked;
    private float _savedJumpImpulse;
    private bool _jumpImpulseSaved;
    private bool _stationLockRequested;

    void Start()
    {
        _localPlayer = Networking.LocalPlayer;
        maxCharacters = Mathf.Clamp(maxCharacters, 1, 144);
        submitCooldownSec = Mathf.Clamp(submitCooldownSec, 0.5f, 60f);
        if (panelRoot != null) panelRoot.gameObject.SetActive(false);
        SetStatus("");
        for (int i = 0; i < MaxSlots; i++) _lastAcceptedAtBySlot[i] = -999f;
    }

    void Update()
    {
        bool panelVisible = panelRoot != null && panelRoot.gameObject.activeSelf;
        if (!panelVisible && Input.GetKeyDown(openKey))
        {
            _Open();
            panelVisible = true;
        }
        if (panelVisible && Input.GetKeyDown(KeyCode.Escape))
        {
            _Close();
            return;
        }
        if (panelVisible && (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter)))
        {
            _Submit();
            return;
        }

        int session = telemetry == null ? 0 : telemetry.GetSession();
        if (session > 0 && session != _reportedReadySession && telemetry.IsDriver())
        {
            _reportedReadySession = session;
            telemetry.Emit("sys.chat_input_ready",
                "\"ready\":true,\"max_chars\":" + maxCharacters
                + ",\"cooldown_ms\":" + Mathf.RoundToInt(submitCooldownSec * 1000f));
        }
    }

    void LateUpdate()
    {
        if (_localPlayer == null || !_localPlayer.IsValid())
            _localPlayer = Networking.LocalPlayer;
        if (_localPlayer == null || !_localPlayer.IsValid()) return;

        bool panelVisible = panelRoot != null && panelRoot.gameObject.activeSelf;
        if (panelVisible && !_inputLocked) SetInputLocked(true);

        VRCPlayerApi.TrackingData head = _localPlayer.GetTrackingData(
            VRCPlayerApi.TrackingDataType.Head
        );
        float yaw = head.rotation.eulerAngles.y;
        Quaternion upright = Quaternion.Euler(0f, yaw, 0f);
        if (panelRoot != null)
        {
            panelRoot.position = head.position + upright * panelHeadOffset;
            panelRoot.rotation = upright;
        }
    }

    // 下划线开头：只能由本地 UI 的 SendCustomEvent 调用，禁止旧式网络事件直调。
    public void _Open()
    {
        if (panelRoot != null) panelRoot.gameObject.SetActive(true);
        SetInputLocked(true);
        if (inputField != null)
        {
            inputField.Select();
            inputField.ActivateInputField();
        }
        SetStatus(telemetry != null && telemetry.GetSession() > 0 ? "" : "NPC 正在连接…");
    }

    public void _Close()
    {
        if (panelRoot != null) panelRoot.gameObject.SetActive(false);
        if (inputField != null) inputField.DeactivateInputField();
        SetInputLocked(false);
        SetStatus("");
    }

    // InputField 单行模式按回车时会先触发 onEndEdit。由 Builder 把该事件固定
    // 转发到这里，避免只依赖 Udon Update 的逐帧按键时序而漏掉提交。
    public void _InputEndEdit()
    {
        if (panelRoot == null || !panelRoot.gameObject.activeSelf) return;
        if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter))
            _Submit();
    }

    public void _Submit()
    {
        if (inputField == null) return;
        string text = Normalize(inputField.text);
        if (text.Length == 0)
        {
            SetStatus("请输入内容");
            return;
        }
        if (telemetry == null || telemetry.GetSession() <= 0)
        {
            SetStatus("NPC 尚未连接");
            return;
        }
        float now = Time.realtimeSinceStartup;
        if (now < _nextLocalSubmitAt)
        {
            SetStatus("发送太快，请稍候");
            return;
        }
        _nextLocalSubmitAt = now + submitCooldownSec;
        int sequence = _nextClientSequence;
        _nextClientSequence = _nextClientSequence >= 2147483646
            ? 1
            : _nextClientSequence + 1;
        SendCustomNetworkEvent(
            NetworkEventTarget.Owner,
            nameof(ReceivePlayerChat),
            text,
            sequence
        );
        inputField.text = "";
        SetStatus("已发送，正在等待回复…");
        _Close();
    }

    [NetworkCallable(maxEventsPerSecond: 1)]
    public void ReceivePlayerChat(string text, int clientSequence)
    {
        if (!NetworkCalling.InNetworkCall || telemetry == null || perception == null) return;
        if (!Networking.IsOwner(gameObject) || !telemetry.IsDriver()) return;
        if (telemetry.GetSession() <= 0 || clientSequence <= 0) return;

        VRCPlayerApi sender = NetworkCalling.CallingPlayer;
        if (sender == null || !sender.IsValid()) return;
        int slot = perception.SlotOf(sender);
        if (slot < 0)
        {
            perception.RebuildSlots();
            slot = perception.SlotOf(sender);
        }
        if (slot < 0 || slot >= MaxSlots) return;

        int pid = sender.playerId;
        if (_lastPidBySlot[slot] != pid)
        {
            _lastPidBySlot[slot] = pid;
            _lastSequenceBySlot[slot] = 0;
            _lastAcceptedAtBySlot[slot] = -999f;
        }
        if (_lastSequenceBySlot[slot] == clientSequence) return;
        float now = Time.realtimeSinceStartup;
        if (now - _lastAcceptedAtBySlot[slot] < submitCooldownSec) return;

        string normalized = Normalize(text);
        if (normalized.Length == 0) return;
        _lastSequenceBySlot[slot] = clientSequence;
        _lastAcceptedAtBySlot[slot] = now;
        telemetry.Emit("player.chat_submit",
            "\"slot\":" + slot
            + ",\"pid\":" + pid
            + ",\"submit_seq\":" + clientSequence
            + ",\"text\":" + telemetry.J(normalized));
    }

    private string Normalize(string value)
    {
        if (value == null) return "";
        string normalized = value.Replace("\r", " ").Replace("\n", " ").Replace("\t", " ").Trim();
        int limit = Mathf.Clamp(maxCharacters, 1, 144);
        if (normalized.Length > limit) normalized = normalized.Substring(0, limit);
        return normalized;
    }

    private void SetStatus(string value)
    {
        if (statusText != null) statusText.text = value == null ? "" : value;
    }

    private void SetInputLocked(bool locked)
    {
        if (_localPlayer == null || !_localPlayer.IsValid())
            _localPlayer = Networking.LocalPlayer;
        if (_localPlayer == null || !_localPlayer.IsValid())
        {
            if (!locked) _inputLocked = false;
            return;
        }
        if (_inputLocked == locked) return;
        if (locked)
        {
            // ClientSim 与 VRChat 的跳跃输入不完全受 Immobilize 约束；先把跳跃冲量
            // 暂时归零，再进入一个不可自行退出的非坐姿 Station，连同 Z/C 姿态键一并锁住。
            _savedJumpImpulse = _localPlayer.GetJumpImpulse();
            _jumpImpulseSaved = true;
            _localPlayer.SetJumpImpulse(0f);
            _localPlayer.Immobilize(true);
            if (inputLockStation != null)
            {
                inputLockStation.transform.position = _localPlayer.GetPosition();
                inputLockStation.transform.rotation = _localPlayer.GetRotation();
                inputLockStation.UseStation(_localPlayer);
                _stationLockRequested = true;
            }
            _inputLocked = true;
            return;
        }

        if (_stationLockRequested && inputLockStation != null)
            inputLockStation.ExitStation(_localPlayer);
        _stationLockRequested = false;
        _localPlayer.Immobilize(false);
        if (_jumpImpulseSaved)
        {
            _localPlayer.SetJumpImpulse(_savedJumpImpulse);
            _jumpImpulseSaved = false;
        }
        _inputLocked = false;
    }

    void OnDisable()
    {
        // 场景切换、ClientSim 停止或对象被禁用时也必须释放本插件取得的移动锁。
        if (_inputLocked) SetInputLocked(false);
    }
}
