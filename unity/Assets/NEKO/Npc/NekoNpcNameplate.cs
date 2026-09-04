/*
 * NekoNpcNameplate —— YUI NPC 角色环绕对白与 v1.1 文本状态同步（UdonSharp, Manual）
 *
 * 动态文本只在 TEXT_COMMIT 全部校验成功后一次性写入；远端和晚加入者只读取同步后的
 * 原子状态。逐字弹出只是一层基于服务器时间的本地显示投影，不会逐字发送网络事件。
 * 文本到期由 owner 清除并再次序列化，其他客户端只做本地显示投影。
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
    // 仅为兼容旧场景序列化保留；运行时会停用，Builder 会删除旧 NameText。
    public TextMeshPro nameText;
    public TextMeshPro bubbleText;
    public Transform nameBillboard;
    public Transform bubbleBillboard;
    public Transform headAnchor;
    public Transform bodyAnchor;
    public string displayName = "YUI";
    public float visibleRange = 12f;
    public float nameHeadOffset = 0.20f;
    // 对白以稳定的角色根节点为中心，避免跟随会受待机动画影响的 Head 骨骼抖动。
    public float dialogueOrbitRadius = 0.55f;
    public float dialogueAnchorHeight = 0.98f;
    public float dialogueOrbitFollowSpeed = 12f;
    public float dialogueRotationFollowSpeed = 16f;
    public float dialoguePositionDeadZone = 0.003f;
    public float dialogueRotationDeadZoneDegrees = 0.2f;
    public int revealLeadInMs = 100;
    public int revealCharacterIntervalMs = 70;
    public int revealShortPunctuationPauseMs = 120;
    public int revealLongPunctuationPauseMs = 240;
    public float minimumFullTextHoldSeconds = 4f;

    [UdonSynced] private string _syncText = "";
    [UdonSynced] private int _syncTransferSeq;
    [UdonSynced] private int _syncUtf8Bytes;
    [UdonSynced] private int _syncCrc16 = -1;
    [UdonSynced] private int _syncDisplayUntilServerMs = -1;
    [UdonSynced] private int _syncRevealStartServerMs = -1;

    private string _currentBubble;
    private int _currentTransferSeq;
    private int _currentUtf8Bytes;
    private int _currentCrc16 = -1;
    private int _displayUntilServerMs = -1;
    private int _revealStartServerMs = -1;

    private string _renderBubble = "";
    private int[] _unitStarts;
    private int[] _unitLengths;
    private int[] _unitRevealAtMs;
    private int _unitCount;
    private int _scaledCharacterIntervalMs = 70;
    private int _lastRenderedUnitCount = -1;
    private int _lastRenderedPopPhase = -1;
    private bool _dialoguePositionInitialized;

    void Start()
    {
        // 0.5.11 起不再显示头顶名称；即使旧场景尚未重跑 Builder，也不会残留 YUI。
        if (nameText != null)
        {
            nameText.text = "";
            nameText.gameObject.SetActive(false);
        }
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
        int now = Networking.GetServerTimeInMilliseconds();
        int leadInMs = Mathf.Max(0, revealLeadInMs);
        int minimumHoldMs = Mathf.Max(0, Mathf.RoundToInt(minimumFullTextHoldSeconds * 1000f));
        int requestedLifetimeMs = Mathf.Max(1, Mathf.RoundToInt(seconds * 1000f));
        // 动态回复通常已由宿主提供充足阅读时间；这里还保证极短 preset 不会在全文刚出现前消失。
        int minimumLifetimeMs = leadInMs + minimumHoldMs + 250;
        _revealStartServerMs = now + leadInMs;
        _displayUntilServerMs = now + Mathf.Max(requestedLifetimeMs, minimumLifetimeMs);
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
        _revealStartServerMs = -1;
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
        _syncRevealStartServerMs = _revealStartServerMs;
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
        _revealStartServerMs = _syncRevealStartServerMs;
        ApplyLocalText();
    }

    private void ApplyLocalText()
    {
        PrepareRevealProjection();
        RenderDialogueProjection(Networking.GetServerTimeInMilliseconds(), true);
    }

    private void PrepareRevealProjection()
    {
        _lastRenderedUnitCount = -1;
        _lastRenderedPopPhase = -1;
        _unitCount = 0;
        _renderBubble = SanitizeForTmp(_currentBubble);

        if (_renderBubble.Length == 0)
        {
            _unitStarts = null;
            _unitLengths = null;
            _unitRevealAtMs = null;
            if (bubbleText != null) bubbleText.text = "";
            return;
        }

        int capacity = Mathf.Max(1, _renderBubble.Length);
        _unitStarts = new int[capacity];
        _unitLengths = new int[capacity];
        _unitRevealAtMs = new int[capacity];

        int cursor = 0;
        int naturalTimeMs = 0;
        int intervalMs = Mathf.Max(1, revealCharacterIntervalMs);
        while (cursor < _renderBubble.Length && _unitCount < capacity)
        {
            int unitLength = NextDisplayUnitLength(_renderBubble, cursor);
            if (unitLength <= 0) unitLength = 1;
            _unitStarts[_unitCount] = cursor;
            _unitLengths[_unitCount] = unitLength;
            _unitRevealAtMs[_unitCount] = naturalTimeMs;
            naturalTimeMs += intervalMs + PunctuationPauseMs(_renderBubble, cursor, unitLength);
            cursor += unitLength;
            _unitCount++;
        }

        // displaySeconds 仍是整页总生命周期；过长内容只压缩逐字时间，不延长宿主翻页时序。
        int minimumHoldMs = Mathf.Max(0, Mathf.RoundToInt(minimumFullTextHoldSeconds * 1000f));
        int availableRevealMs = _displayUntilServerMs - _revealStartServerMs - minimumHoldMs;
        if (availableRevealMs < 1) availableRevealMs = 1;
        int scalePermille = 1000;
        if (naturalTimeMs > availableRevealMs && naturalTimeMs > 0)
            scalePermille = Mathf.Max(1, availableRevealMs * 1000 / naturalTimeMs);

        for (int i = 0; i < _unitCount; i++)
            _unitRevealAtMs[i] = _unitRevealAtMs[i] * scalePermille / 1000;
        _scaledCharacterIntervalMs = Mathf.Max(20, intervalMs * scalePermille / 1000);
    }

    private void RenderDialogueProjection(int nowServerMs, bool force)
    {
        if (bubbleText == null) return;
        if (_renderBubble.Length == 0 || _unitCount <= 0)
        {
            if (force || bubbleText.text.Length > 0) bubbleText.text = "";
            return;
        }

        int elapsedMs = nowServerMs - _revealStartServerMs;
        int visibleUnits = 0;
        if (elapsedMs >= 0)
        {
            while (visibleUnits < _unitCount && elapsedMs >= _unitRevealAtMs[visibleUnits])
                visibleUnits++;
        }

        int popPhase = 0;
        if (visibleUnits > 0)
        {
            int currentUnit = visibleUnits - 1;
            int ageMs = elapsedMs - _unitRevealAtMs[currentUnit];
            if (IsWhitespaceUnit(_renderBubble, _unitStarts[currentUnit], _unitLengths[currentUnit]))
                popPhase = 2;
            else if (ageMs < _scaledCharacterIntervalMs / 2)
                popPhase = 0;
            else if (ageMs < _scaledCharacterIntervalMs)
                popPhase = 1;
            else
                popPhase = 2;
        }

        if (!force && visibleUnits == _lastRenderedUnitCount && popPhase == _lastRenderedPopPhase) return;
        _lastRenderedUnitCount = visibleUnits;
        _lastRenderedPopPhase = popPhase;

        if (visibleUnits >= _unitCount && popPhase == 2)
        {
            bubbleText.text = ProjectRangeForTmp(_renderBubble, 0, _renderBubble.Length);
            return;
        }

        if (visibleUnits <= 0)
        {
            bubbleText.text = "<color=#FFFFFF00>" + _renderBubble + "</color>";
            return;
        }

        int unitIndex = visibleUnits - 1;
        int unitStart = _unitStarts[unitIndex];
        int unitLength = _unitLengths[unitIndex];
        int suffixStart = unitStart + unitLength;
        string prefix = ProjectRangeForTmp(_renderBubble, 0, unitStart);
        string current = ProjectRangeForTmp(_renderBubble, unitStart, unitLength);
        string suffix = ProjectRangeForTmp(_renderBubble, suffixStart, _renderBubble.Length - suffixStart);

        string animated = current;
        // 不改变字号和字符宽度，避免居中排版在每个字符弹出时左右抖动。
        // 只让当前字符做很轻的纵向位移，仍保留逐字“蹦出”的感觉。
        if (popPhase == 0)
            animated = "<voffset=-0.04em>" + current + "</voffset>";
        else if (popPhase == 1)
            animated = "<voffset=0.06em>" + current + "</voffset>";

        bubbleText.text = prefix + animated
            + (suffix.Length > 0 ? "<color=#FFFFFF00>" + suffix + "</color>" : "");
    }

    private string SanitizeForTmp(string value)
    {
        if (value == null || value.Length == 0) return "";
        // TMP 富文本由本脚本生成；外部文本中的尖括号改为全角，避免插入控制标签。
        return value.Replace("<", "＜").Replace(">", "＞");
    }

    private string ProjectRangeForTmp(string value, int start, int length)
    {
        if (value == null || length <= 0 || start < 0 || start >= value.Length) return "";
        int end = Mathf.Min(value.Length, start + length);
        int cursor = start;
        string projected = "";
        while (cursor < end)
        {
            int unitLength = NextDisplayUnitLength(value, cursor);
            if (unitLength <= 0 || cursor + unitLength > end) unitLength = end - cursor;
            string spriteTag = EmojiSpriteTag(value, cursor, unitLength);
            projected += spriteTag.Length > 0
                ? spriteTag
                : value.Substring(cursor, unitLength);
            cursor += unitLength;
        }
        return projected;
    }

    private string EmojiSpriteTag(string value, int start, int unitLength)
    {
        if (value == null || unitLength <= 0) return "";
        int codePointLength = CodePointLength(value, start);
        int significantLength = unitLength;
        if (unitLength == codePointLength + 1 && start + codePointLength < value.Length)
        {
            int variationSelector = value[start + codePointLength];
            if (variationSelector == 0xFE0E || variationSelector == 0xFE0F)
                significantLength = codePointLength;
        }
        // ZWJ/组合序列整体仍是一个逐字单元；只有 EmojiOne 确实包含的单码点
        // 才替换为 sprite，避免把未知序列拆成多个方框或伪造不存在的图形。
        if (significantLength != codePointLength) return "";

        int codePoint = CodePointAt(value, start);
        int spriteIndex = EmojiOneSpriteIndex(codePoint);
        // tint=1 让 sprite 继承外层文字颜色/透明度，隐藏后缀中的 Emoji 不会提前露出。
        return spriteIndex < 0 ? "" : "<sprite=" + spriteIndex + " tint=1>";
    }

    private int EmojiOneSpriteIndex(int codePoint)
    {
        // TextMesh Pro 示例 EmojiOne.asset 的稳定 glyph 顺序；Builder 会显式绑定该资产。
        if (codePoint == 0x1F60A) return 0;
        if (codePoint == 0x1F60B) return 1;
        if (codePoint == 0x1F60D) return 2;
        if (codePoint == 0x1F60E) return 3;
        if (codePoint == 0x1F600) return 4;
        if (codePoint == 0x1F601) return 5;
        if (codePoint == 0x1F602) return 6;
        if (codePoint == 0x1F603) return 7;
        if (codePoint == 0x1F604) return 8;
        if (codePoint == 0x1F605) return 9;
        if (codePoint == 0x1F606) return 10;
        if (codePoint == 0x1F609) return 11;
        if (codePoint == 0x1F923) return 13;
        if (codePoint == 0x263A) return 14;
        if (codePoint == 0x2639) return 15;
        return -1;
    }

    private int NextDisplayUnitLength(string value, int start)
    {
        if (value == null || start < 0 || start >= value.Length) return 0;
        if (value[start] == '\r' && start + 1 < value.Length && value[start + 1] == '\n') return 2;

        int cursor = start + CodePointLength(value, start);
        while (cursor < value.Length)
        {
            int codePoint = CodePointAt(value, cursor);
            if (IsExtendingCodePoint(codePoint))
            {
                cursor += CodePointLength(value, cursor);
                continue;
            }
            if (codePoint != 0x200D) break;

            cursor += CodePointLength(value, cursor);
            if (cursor >= value.Length) break;
            cursor += CodePointLength(value, cursor);
        }
        return cursor - start;
    }

    private int CodePointLength(string value, int index)
    {
        if (value == null || index < 0 || index >= value.Length) return 0;
        int high = value[index];
        if (high >= 0xD800 && high <= 0xDBFF && index + 1 < value.Length)
        {
            int low = value[index + 1];
            if (low >= 0xDC00 && low <= 0xDFFF) return 2;
        }
        return 1;
    }

    private int CodePointAt(string value, int index)
    {
        int high = value[index];
        if (high >= 0xD800 && high <= 0xDBFF && index + 1 < value.Length)
        {
            int low = value[index + 1];
            if (low >= 0xDC00 && low <= 0xDFFF)
                return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
        }
        return high;
    }

    private bool IsExtendingCodePoint(int codePoint)
    {
        return (codePoint >= 0x0300 && codePoint <= 0x036F)
            || (codePoint >= 0x1AB0 && codePoint <= 0x1AFF)
            || (codePoint >= 0x1DC0 && codePoint <= 0x1DFF)
            || (codePoint >= 0x20D0 && codePoint <= 0x20FF)
            || (codePoint >= 0xFE00 && codePoint <= 0xFE0F)
            || (codePoint >= 0xFE20 && codePoint <= 0xFE2F)
            || (codePoint >= 0x1F3FB && codePoint <= 0x1F3FF)
            || (codePoint >= 0xE0100 && codePoint <= 0xE01EF);
    }

    private bool IsWhitespaceUnit(string value, int start, int length)
    {
        int end = start + length;
        for (int i = start; i < end && i < value.Length; i++)
        {
            int c = value[i];
            if (c != 0x20 && c != 0x09 && c != 0x0A && c != 0x0D && c != 0x3000) return false;
        }
        return true;
    }

    private int PunctuationPauseMs(string value, int start, int length)
    {
        int end = Mathf.Min(value.Length, start + length);
        int punctuation = 0;
        int cursor = start;
        while (cursor < end)
        {
            int codePoint = CodePointAt(value, cursor);
            if (!IsExtendingCodePoint(codePoint) && codePoint != 0x200D) punctuation = codePoint;
            cursor += CodePointLength(value, cursor);
        }

        if (punctuation == 0x2C || punctuation == 0x3B || punctuation == 0x3A
            || punctuation == 0xFF0C || punctuation == 0x3001 || punctuation == 0xFF1B || punctuation == 0xFF1A)
            return Mathf.Max(0, revealShortPunctuationPauseMs);
        if (punctuation == 0x2E || punctuation == 0x21 || punctuation == 0x3F
            || punctuation == 0x0A || punctuation == 0x0D || punctuation == 0x3002
            || punctuation == 0xFF01 || punctuation == 0xFF1F || punctuation == 0x2026)
            return Mathf.Max(0, revealLongPunctuationPauseMs);
        return 0;
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

        RenderDialogueProjection(Networking.GetServerTimeInMilliseconds(), false);

    }

    void LateUpdate()
    {
        VRCPlayerApi local = Networking.LocalPlayer;
        if (local == null) return;
        Vector3 viewerHead = local.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position;
        Vector3 rangeAnchor = headAnchor == null ? transform.position : headAnchor.position;
        bool near = Vector3.Distance(rangeAnchor, viewerHead) <= visibleRange;

        if (bubbleBillboard != null)
        {
            Transform stableAnchor = bodyAnchor == null ? transform.parent : bodyAnchor;
            Vector3 stableOrigin = stableAnchor == null ? transform.position : stableAnchor.position;
            Vector3 dialogueCenter = stableOrigin + Vector3.up * dialogueAnchorHeight;
            Vector3 toViewer = viewerHead - dialogueCenter;
            toViewer.y = 0f;
            if (toViewer.sqrMagnitude < 0.0001f)
            {
                toViewer = stableAnchor == null ? Vector3.back : -stableAnchor.forward;
                toViewer.y = 0f;
            }
            if (toViewer.sqrMagnitude < 0.0001f) toViewer = Vector3.back;
            // UdonSharp 对 Vector3.Normalize() 的结构体原位写回并不可靠，必须显式接回 normalized。
            else toViewer = toViewer.normalized;

            // 以 NPC 为圆心、朝本地玩家的方向作为半径；玩家绕行时对白同步绕到角色正前方。
            Vector3 targetPosition = dialogueCenter + toViewer * dialogueOrbitRadius;
            Quaternion targetRotation = Quaternion.LookRotation(-toViewer, Vector3.up);
            float positionError = Vector3.Distance(bubbleBillboard.position, targetPosition);
            if (!_dialoguePositionInitialized || positionError > 2f)
            {
                bubbleBillboard.position = targetPosition;
                bubbleBillboard.rotation = targetRotation;
                _dialoguePositionInitialized = true;
            }
            else
            {
                if (positionError > Mathf.Max(0f, dialoguePositionDeadZone))
                {
                    float follow = Mathf.Clamp01(Time.deltaTime * Mathf.Max(0.1f, dialogueOrbitFollowSpeed));
                    bubbleBillboard.position = Vector3.Lerp(bubbleBillboard.position, targetPosition, follow);
                }
                float rotationError = Quaternion.Angle(bubbleBillboard.rotation, targetRotation);
                if (rotationError > Mathf.Max(0f, dialogueRotationDeadZoneDegrees))
                {
                    float rotationFollow = Mathf.Clamp01(Time.deltaTime * Mathf.Max(0.1f, dialogueRotationFollowSpeed));
                    bubbleBillboard.rotation = Quaternion.Slerp(bubbleBillboard.rotation, targetRotation, rotationFollow);
                }
            }
        }
        if (bubbleText != null && bubbleText.gameObject.activeSelf != near) bubbleText.gameObject.SetActive(near);
    }
}
