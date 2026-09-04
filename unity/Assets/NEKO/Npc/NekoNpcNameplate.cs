/*
 * NekoNpcNameplate —— YUI NPC 名牌、文本气泡与 v1.1 文本状态同步（UdonSharp, Manual）
 *
 * 动态文本只在 TEXT_COMMIT 全部校验成功后一次性写入；远端和晚加入者只读取同步后的
 * 原子状态。文本到期由 owner 清除并再次序列化，其他客户端只做本地显示投影。
 */
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using TMPro;

[UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
public class NekoNpcNameplate : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;

    [Header("显示")]
    public TextMeshPro nameText;
    public TextMeshPro bubbleText;
    public Transform nameBillboard;
    public Transform bubbleBillboard;
    public Transform headAnchor;
    public string displayName = "YUI";
    public float visibleRange = 12f;
    public float nameHeadOffset = 0.20f;
    public float bubbleHeadOffset = 0.36f;

    [UdonSynced] private string _syncText = "";
    [UdonSynced] private int _syncTransferSeq;
    [UdonSynced] private int _syncUtf8Bytes;
    [UdonSynced] private int _syncCrc16 = -1;
    [UdonSynced] private int _syncDisplayUntilServerMs = -1;

    private string _currentBubble;
    private int _currentTransferSeq;
    private int _currentUtf8Bytes;
    private int _currentCrc16 = -1;
    private int _displayUntilServerMs = -1;

    void Start()
    {
        if (nameText != null) nameText.text = displayName;
        ApplySyncedText();
    }

    // 静态 preset 也需要跨客户端显示，但 snapshot 的 transfer_seq/crc16 仍为 null。
    public void ShowPreset(string text, float seconds)
    {
        SetText(text, 0, Utf8ByteCount(text), -1, seconds);
    }

    // 只允许 Router 在完整事务提交后调用。
    public void ShowDynamic(string text, int transferSeq, int utf8Bytes, int crc16, int displaySeconds)
    {
        SetText(text, transferSeq, utf8Bytes, crc16, displaySeconds <= 0 ? 5f : displaySeconds);
    }

    private void SetText(string text, int transferSeq, int utf8Bytes, int crc16, float seconds)
    {
        if (bubbleText == null || !Networking.IsOwner(gameObject)) return;
        if (_currentTransferSeq > 0 && _currentBubble != null && _currentBubble.Length > 0)
            EmitCleared("replaced");

        _currentBubble = text == null ? "" : text;
        _currentTransferSeq = transferSeq;
        _currentUtf8Bytes = Mathf.Max(0, utf8Bytes);
        _currentCrc16 = crc16;
        _displayUntilServerMs = Networking.GetServerTimeInMilliseconds() + Mathf.Max(1, Mathf.RoundToInt(seconds * 1000f));
        ApplyLocalText();
        CopyToSync();
    }

    // 旧调用点的兼容入口；只用于静态 preset。
    public void ShowBubble(string text, float seconds) { ShowPreset(text, seconds); }

    public void ClearBubble() { ClearBubbleWithReason("explicit"); }

    public void ClearBubbleWithReason(string reason)
    {
        if (!Networking.IsOwner(gameObject)) return;
        if (_currentTransferSeq > 0 && _currentBubble != null && _currentBubble.Length > 0)
            EmitCleared(reason);
        _currentBubble = null;
        _currentTransferSeq = 0;
        _currentUtf8Bytes = 0;
        _currentCrc16 = -1;
        _displayUntilServerMs = -1;
        ApplyLocalText();
        CopyToSync();
    }

    private void CopyToSync()
    {
        _syncText = _currentBubble == null ? "" : _currentBubble;
        _syncTransferSeq = _currentTransferSeq;
        _syncUtf8Bytes = _currentUtf8Bytes;
        _syncCrc16 = _currentCrc16;
        _syncDisplayUntilServerMs = _displayUntilServerMs;
        RequestSerialization();
    }

    public override void OnDeserialization() { ApplySyncedText(); }

    private void ApplySyncedText()
    {
        _currentBubble = _syncText == null || _syncText.Length == 0 ? null : _syncText;
        _currentTransferSeq = _syncTransferSeq;
        _currentUtf8Bytes = _syncUtf8Bytes;
        _currentCrc16 = _syncCrc16;
        _displayUntilServerMs = _syncDisplayUntilServerMs;
        ApplyLocalText();
    }

    private void ApplyLocalText()
    {
        if (bubbleText != null) bubbleText.text = _currentBubble == null ? "" : _currentBubble;
    }

    private void EmitCleared(string reason)
    {
        if (telemetry == null || _currentTransferSeq <= 0) return;
        telemetry.Emit("npc.text_cleared", "\"transfer_seq\":" + _currentTransferSeq + ",\"reason\":" + telemetry.J(reason));
    }

    public int CurrentTransferSeq() { return _currentTransferSeq; }

    // 供 NekoNpcLife 在所有客户端判断“她正在说话”（气泡可见），驱动说话小动作。
    public bool IsBubbleVisible() { return _currentBubble != null && _currentBubble.Length > 0; }

    public string BuildSnapshotText()
    {
        return "{\"transfer_seq\":" + (_currentTransferSeq > 0 ? _currentTransferSeq.ToString() : "null")
            + ",\"utf8_bytes\":" + _currentUtf8Bytes
            + ",\"crc16\":" + (_currentCrc16 >= 0 && telemetry != null ? telemetry.J(Hex4(_currentCrc16)) : "null")
            + ",\"display_until_server_ms\":" + (_displayUntilServerMs > 0 ? _displayUntilServerMs.ToString() : "null")
            + ",\"text\":" + (_currentBubble == null || telemetry == null ? "null" : telemetry.J(_currentBubble)) + "}";
    }

    private int Utf8ByteCount(string value)
    {
        if (value == null) return 0;
        int bytes = 0;
        for (int i = 0; i < value.Length; i++)
        {
            int c = value[i];
            if (c <= 0x7F) bytes++;
            else if (c <= 0x7FF) bytes += 2;
            else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < value.Length && value[i + 1] >= 0xDC00 && value[i + 1] <= 0xDFFF) { bytes += 4; i++; }
            else bytes += 3;
        }
        return bytes;
    }

    private string Hex4(int value)
    {
        string hex = "0123456789ABCDEF";
        return hex.Substring((value >> 12) & 15, 1) + hex.Substring((value >> 8) & 15, 1)
            + hex.Substring((value >> 4) & 15, 1) + hex.Substring(value & 15, 1);
    }

    void Update()
    {
        if (Networking.IsOwner(gameObject) && _displayUntilServerMs > 0
            && Networking.GetServerTimeInMilliseconds() >= _displayUntilServerMs)
            ClearBubbleWithReason("expired");

        VRCPlayerApi local = Networking.LocalPlayer;
        if (local == null) return;
        Vector3 head = local.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position;
        Vector3 anchor = headAnchor == null ? transform.position : headAnchor.position;
        bool near = Vector3.Distance(anchor, head) <= visibleRange;

        // 锚点只跟随 NPC 头部；朝向分别绕各自文字枢轴更新，不能再旋转整个
        // Nameplate 根节点，否则带高度偏移的子物体会绕根节点画弧并随玩家视角漂移。
        if (headAnchor != null)
        {
            if (nameBillboard != null)
                nameBillboard.position = headAnchor.position + Vector3.up * nameHeadOffset;
            if (bubbleBillboard != null)
                bubbleBillboard.position = headAnchor.position + Vector3.up * bubbleHeadOffset;
        }
        FaceTextToViewer(nameBillboard, head);
        FaceTextToViewer(bubbleBillboard, head);
        if (nameText != null && nameText.gameObject.activeSelf != near) nameText.gameObject.SetActive(near);
        if (bubbleText != null && bubbleText.gameObject.activeSelf != near) bubbleText.gameObject.SetActive(near);
    }

    private void FaceTextToViewer(Transform target, Vector3 viewerHead)
    {
        if (target == null) return;
        Vector3 direction = target.position - viewerHead;
        if (direction.sqrMagnitude > 0.0001f)
            target.rotation = Quaternion.LookRotation(direction, Vector3.up);
    }
}
