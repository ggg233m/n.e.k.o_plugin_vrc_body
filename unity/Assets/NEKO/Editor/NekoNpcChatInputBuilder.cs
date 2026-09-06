#if UNITY_EDITOR
using System;
using System.Reflection;
using UnityEditor;
using UnityEditor.Events;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using VRC.SDK3.Components;
using VRC.Udon;

public static class NekoNpcChatInputBuilder
{
    const string NpcRootName = "NekoNpc_YUI";
    const string UiRootName = "YuiPlayerChatUiLocal";
    const string ChineseFontPath = "Assets/NEKO/Fonts/NotoSansSC-Regular.otf";

    [MenuItem("NEKO/YUI Formal/Chat Input/Install Or Update")]
    public static void InstallOrUpdate()
    {
        GameObject npc = GameObject.Find(NpcRootName);
        if (npc == null) throw new InvalidOperationException("当前场景缺少 " + NpcRootName);
        NekoNpcTelemetry telemetry = npc.GetComponentInChildren<NekoNpcTelemetry>(true);
        NekoNpcPerception perception = npc.GetComponentInChildren<NekoNpcPerception>(true);
        if (telemetry == null || perception == null)
            throw new InvalidOperationException("YUI NPC 缺少 Telemetry 或 Perception");

        NekoNpcChatInput chat = npc.GetComponent<NekoNpcChatInput>();
        if (chat == null) chat = AddUdon<NekoNpcChatInput>(npc);
        if (chat == null) throw new InvalidOperationException("无法添加 NekoNpcChatInput UdonSharp 组件");

        GameObject uiRoot = GameObject.Find(UiRootName);
        if (uiRoot == null)
        {
            uiRoot = new GameObject(UiRootName);
            Undo.RegisterCreatedObjectUndo(uiRoot, "创建 YUI 玩家聊天界面");
        }
        Transform obsoleteLauncher = uiRoot.transform.Find("LauncherCanvas");
        if (obsoleteLauncher != null)
            Undo.DestroyObjectImmediate(obsoleteLauncher.gameObject);
        Transform panel = EnsurePanel(uiRoot.transform, out InputField input, out Text status);
        VRCStation inputLockStation = EnsureInputLockStation(uiRoot.transform);
        EnsureEventSystem();

        chat.telemetry = telemetry;
        chat.perception = perception;
        chat.inputField = input;
        chat.statusText = status;
        chat.connectionText = panel.Find("Panel/Connection").GetComponent<Text>();
        chat.historyScroll = panel.Find("Panel/History").GetComponent<ScrollRect>();
        chat.historyText = panel.Find("Panel/History/Viewport/Content").GetComponent<Text>();
        // 正式 NPC 的字幕组件在 Nameplate 子对象，不能只查根节点。
        NekoNpcNameplate nameplate = npc.GetComponentInChildren<NekoNpcNameplate>(true);
        if (nameplate != null) { nameplate.chatInput = chat; MarkDirtyAndSync(nameplate); }
        chat.panelRoot = panel;
        chat.inputLockStation = inputLockStation;
        chat.panelHeadOffset = new Vector3(0f, 0f, 0.55f);
        chat.openKey = KeyCode.T;
        chat.maxCharacters = 144;
        chat.submitCooldownSec = 2f;
        input.characterLimit = chat.maxCharacters;

        WireButton(panel.Find("Panel/SendButton").GetComponent<Button>(), chat, "_Submit");
        WireButton(panel.Find("Panel/CloseButton").GetComponent<Button>(), chat, "_Close");
        WireInputSubmit(input, chat, "_Submit");
        WireInputEndEdit(input, chat, "_InputEndEdit");

        panel.gameObject.SetActive(false);
        MarkDirtyAndSync(chat);
        EditorUtility.SetDirty(uiRoot);
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Selection.activeGameObject = uiRoot;
        Debug.Log("[NEKO] 已安装本地跟随式聊天 UI：T 呼出、Enter 发送、Esc 关闭，平时完全隐藏。");
    }

    [MenuItem("NEKO/YUI Formal/Chat Input/Validate")]
    public static void Validate()
    {
        GameObject npc = GameObject.Find(NpcRootName);
        NekoNpcChatInput chat = npc == null ? null : npc.GetComponent<NekoNpcChatInput>();
        GameObject ui = GameObject.Find(UiRootName);
        NekoNpcNameplate nameplate = npc == null ? null : npc.GetComponentInChildren<NekoNpcNameplate>(true);
        bool ok = chat != null && ui != null && chat.telemetry != null && chat.perception != null
            && chat.inputField != null && chat.statusText != null
            && chat.historyText != null && chat.historyScroll != null && chat.connectionText != null
            && nameplate != null && nameplate.chatInput == chat
            && chat.inputLockStation != null
            && chat.panelRoot != null
            && chat.panelRoot.GetComponent<Canvas>() != null
            && chat.inputField.onSubmit.GetPersistentEventCount() > 0
            && chat.inputField.onEndEdit.GetPersistentEventCount() > 0;
        if (!ok) throw new InvalidOperationException("YUI 玩家聊天 UI 未完整安装或引用缺失，请检查子对象 Nameplate 的 chatInput 回复记录绑定");
        UdonBehaviour nameplateBacking = GetBacking(nameplate);
        object savedChat;
        if (nameplateBacking == null || !nameplateBacking.publicVariables.TryGetVariableValue("chatInput", out savedChat)
            || !ReferenceEquals(savedChat, GetBacking(chat)))
            throw new InvalidOperationException("Nameplate 的回复记录绑定尚未同步到 Udon，请重新安装聊天 UI");
        Debug.Log("[NEKO] YUI 玩家聊天 UI 校验通过。T 呼出，最大 144 字，每玩家 2 秒限频。 ");
    }

    static Transform EnsurePanel(Transform root, out InputField input, out Text status)
    {
        Transform existing = root.Find("ChatCanvas");
        GameObject go = existing == null ? new GameObject("ChatCanvas", typeof(RectTransform)) : existing.gameObject;
        if (existing == null) go.transform.SetParent(root, false);
        ConfigureCanvas(go, new Vector2(620f, 480f), 0.00082f);
        RectTransform canvasRect = go.GetComponent<RectTransform>();

        // 使用 VRChat 接近的深蓝灰半透明底与青蓝高亮，不沿用插件的紫黑配色。
        RectTransform background = EnsureImage(canvasRect, "Panel", new Color(0.025f, 0.105f, 0.145f, 0.96f));
        Stretch(background, 0f, 0f, 0f, 0f);
        Text title = EnsureText(background, "Title", "和猫娘聊天", 24, TextAnchor.MiddleLeft);
        title.color = new Color(0.76f, 0.92f, 0.97f, 1f);
        Place(title.rectTransform, 22f, -8f, 460f, 36f);
        Text connection = EnsureText(background, "Connection", "○ 等待 NPC 连接", 16, TextAnchor.MiddleLeft);
        connection.color = new Color(0.48f, 0.85f, 0.82f, 1f);
        Place(connection.rectTransform, 22f, -44f, 550f, 24f);

        RectTransform history = EnsureImage(background, "History", new Color(0.018f, 0.058f, 0.08f, 0.98f));
        Place(history, 18f, -82f, 584f, 286f);
        RectTransform viewport = EnsureImage(history, "Viewport", Color.clear);
        Stretch(viewport, 14f, 14f, 12f, 12f);
        if (viewport.GetComponent<RectMask2D>() == null) viewport.gameObject.AddComponent<RectMask2D>();
        Text content = EnsureText(viewport, "Content", "还没有聊天记录。\n想聊什么，就从这里开始。", 24, TextAnchor.UpperLeft);
        content.color = new Color(0.88f, 0.94f, 0.96f, 1f);
        content.lineSpacing = 1.3f;
        content.horizontalOverflow = HorizontalWrapMode.Wrap;
        content.verticalOverflow = VerticalWrapMode.Overflow;
        RectTransform contentRect = content.rectTransform;
        contentRect.anchorMin = new Vector2(0f, 1f);
        contentRect.anchorMax = new Vector2(1f, 1f);
        contentRect.pivot = new Vector2(0.5f, 1f);
        contentRect.anchoredPosition = Vector2.zero;
        contentRect.sizeDelta = Vector2.zero;
        ContentSizeFitter fitter = content.GetComponent<ContentSizeFitter>();
        if (fitter == null) fitter = content.gameObject.AddComponent<ContentSizeFitter>();
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
        ScrollRect scroll = history.GetComponent<ScrollRect>();
        if (scroll == null) scroll = history.gameObject.AddComponent<ScrollRect>();
        scroll.viewport = viewport;
        scroll.content = contentRect;
        scroll.horizontal = false;
        scroll.vertical = true;
        scroll.movementType = ScrollRect.MovementType.Clamped;
        scroll.scrollSensitivity = 26f;
        scroll.verticalNormalizedPosition = 0f;

        RectTransform closeRect = EnsureButton(background, "CloseButton", "X", new Color(0.055f, 0.22f, 0.28f, 0.98f)).GetComponent<RectTransform>();
        Place(closeRect, 568f, -12f, 34f, 32f);
        Text closeLabel = closeRect.Find("Label").GetComponent<Text>();
        closeLabel.fontSize = 18;
        Stretch(closeLabel.rectTransform, 2f, 2f, 2f, 2f);

        input = EnsureInput(background, "InputField");
        Place(input.GetComponent<RectTransform>(), 18f, -384f, 466f, 50f);

        RectTransform sendRect = EnsureButton(background, "SendButton", "发送", new Color(0.00f, 0.58f, 0.72f, 1f)).GetComponent<RectTransform>();
        Place(sendRect, 496f, -384f, 106f, 50f);

        status = EnsureText(background, "Status", "", 16, TextAnchor.MiddleLeft);
        status.color = new Color(0.53f, 0.84f, 0.92f, 1f);
        Place(status.rectTransform, 22f, -441f, 580f, 28f);
        status.text = "Enter 发送 · Esc 收起 · 滚轮查看记录";
        Transform obsoletePointer = background.Find("DesktopPointer");
        if (obsoletePointer != null) Undo.DestroyObjectImmediate(obsoletePointer.gameObject);
        // VRChat 的 UI 层只能在菜单打开时交互；世界聊天使用 Default 层。
        SetLayerRecursively(go, 0);
        return go.transform;
    }

    static void ConfigureCanvas(GameObject go, Vector2 size, float scale)
    {
        Canvas canvas = go.GetComponent<Canvas>();
        if (canvas == null) canvas = go.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.sortingOrder = 100;
        CanvasScaler scaler = go.GetComponent<CanvasScaler>();
        if (scaler == null) scaler = go.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 10f;
        if (go.GetComponent<GraphicRaycaster>() == null) go.AddComponent<GraphicRaycaster>();
        if (go.GetComponent<VRCUiShape>() == null) go.AddComponent<VRCUiShape>();
        RectTransform rect = go.GetComponent<RectTransform>();
        rect.sizeDelta = size;
        rect.localScale = Vector3.one * scale;
    }

    static RectTransform EnsureImage(Transform parent, string name, Color color)
    {
        Transform found = parent.Find(name);
        GameObject go = found == null ? new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image)) : found.gameObject;
        if (found == null) go.transform.SetParent(parent, false);
        Image image = go.GetComponent<Image>();
        if (image == null) image = go.AddComponent<Image>();
        image.sprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/UISprite.psd");
        image.type = Image.Type.Sliced;
        image.color = color;
        return go.GetComponent<RectTransform>();
    }

    static GameObject EnsureButton(Transform parent, string name, string label, Color color)
    {
        RectTransform rect = EnsureImage(parent, name, color);
        Button button = rect.GetComponent<Button>();
        if (button == null) button = rect.gameObject.AddComponent<Button>();
        button.targetGraphic = rect.GetComponent<Image>();
        button.navigation = NoNavigation();
        Text text = EnsureText(rect, "Label", label, 22, TextAnchor.MiddleCenter);
        Stretch(text.rectTransform, 8f, 8f, 6f, 6f);
        return rect.gameObject;
    }

    static InputField EnsureInput(Transform parent, string name)
    {
        RectTransform root = EnsureImage(parent, name, new Color(0.025f, 0.060f, 0.078f, 0.99f));
        // 清理早期构建器留下的 TMP 输入框，避免同一对象上存在两个 Selectable。
        Component obsoleteInput = root.GetComponent("TMP_InputField");
        if (obsoleteInput != null) Undo.DestroyObjectImmediate(obsoleteInput);
        InputField input = root.GetComponent<InputField>();
        if (input == null) input = root.gameObject.AddComponent<InputField>();

        RectTransform viewport = EnsureImage(root, "Text Area", Color.clear);
        Stretch(viewport, 16f, 16f, 8f, 8f);
        if (viewport.GetComponent<RectMask2D>() == null) viewport.gameObject.AddComponent<RectMask2D>();
        Text placeholder = EnsureLegacyText(viewport, "Placeholder", "输入想说的话…", 21, TextAnchor.MiddleLeft);
        placeholder.fontStyle = FontStyle.Italic;
        placeholder.color = new Color(0.65f, 0.60f, 0.70f, 0.8f);
        Stretch(placeholder.rectTransform, 0f, 0f, 0f, 0f);
        Text value = EnsureLegacyText(viewport, "Text", "", 21, TextAnchor.MiddleLeft);
        Stretch(value.rectTransform, 0f, 0f, 0f, 0f);

        input.textComponent = value;
        input.placeholder = placeholder;
        input.lineType = InputField.LineType.SingleLine;
        input.contentType = InputField.ContentType.Standard;
        input.characterLimit = 144;
        input.targetGraphic = root.GetComponent<Image>();
        input.navigation = NoNavigation();
        input.customCaretColor = true;
        input.caretColor = new Color(0.18f, 0.88f, 1f, 1f);
        input.selectionColor = new Color(0.00f, 0.48f, 0.62f, 0.55f);
        return input;
    }

    static VRCStation EnsureInputLockStation(Transform root)
    {
        Transform found = root.Find("InputLockStation");
        GameObject go = found == null ? new GameObject("InputLockStation") : found.gameObject;
        if (found == null)
        {
            go.transform.SetParent(root, false);
            Undo.RegisterCreatedObjectUndo(go, "创建聊天输入锁定 Station");
        }
        VRCStation station = go.GetComponent<VRCStation>();
        if (station == null) station = Undo.AddComponent<VRCStation>(go);
        station.PlayerMobility = VRC.SDKBase.VRCStation.Mobility.Immobilize;
        station.canUseStationFromStation = false;
        station.disableStationExit = true;
        station.seated = false;
        station.stationEnterPlayerLocation = go.transform;
        station.stationExitPlayerLocation = go.transform;
        EditorUtility.SetDirty(station);
        return station;
    }

    static Navigation NoNavigation()
    {
        Navigation navigation = new Navigation();
        navigation.mode = Navigation.Mode.None;
        return navigation;
    }

    static Text EnsureLegacyText(Transform parent, string name, string value, int size, TextAnchor alignment)
    {
        Transform found = parent.Find(name);
        GameObject go = found == null ? new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Text)) : found.gameObject;
        if (found == null) go.transform.SetParent(parent, false);
        // 安装器必须可重复执行；删除旧版在同名节点上创建的 TMP 文本。
        Component obsoleteText = go.GetComponent("TextMeshProUGUI");
        if (obsoleteText != null) Undo.DestroyObjectImmediate(obsoleteText);
        Text text = go.GetComponent<Text>();
        if (text == null) text = go.AddComponent<Text>();
        text.text = value;
        text.font = RequireChineseFont();
        text.fontSize = size;
        text.alignment = alignment;
        text.color = Color.white;
        text.supportRichText = false;
        text.raycastTarget = false;
        return text;
    }

    static Text EnsureText(Transform parent, string name, string value, int size, TextAnchor alignment)
    {
        return EnsureLegacyText(parent, name, value, size, alignment);
    }

    static Font RequireChineseFont()
    {
        Font font = AssetDatabase.LoadAssetAtPath<Font>(ChineseFontPath);
        if (font == null)
            throw new InvalidOperationException("缺少世界内聊天 UI 中文字体：" + ChineseFontPath);
        return font;
    }

    static void Place(RectTransform rect, float x, float y, float width, float height)
    {
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(0f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = new Vector2(x, y);
        rect.sizeDelta = new Vector2(width, height);
    }

    static void Stretch(RectTransform rect, float left, float right, float top, float bottom)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = new Vector2(left, bottom);
        rect.offsetMax = new Vector2(-right, -top);
    }

    static void WireButton(Button button, NekoNpcChatInput chat, string eventName)
    {
        UdonBehaviour backing = GetBacking(chat);
        if (backing == null) throw new InvalidOperationException("找不到 NekoNpcChatInput 的 backing UdonBehaviour");
        button.onClick = new Button.ButtonClickedEvent();
        UnityEventTools.AddStringPersistentListener(button.onClick, backing.SendCustomEvent, eventName);
        EditorUtility.SetDirty(button);
    }

    static void WireInputEndEdit(InputField input, NekoNpcChatInput chat, string eventName)
    {
        UdonBehaviour backing = GetBacking(chat);
        if (backing == null) throw new InvalidOperationException("找不到 NekoNpcChatInput 的 backing UdonBehaviour");
        input.onEndEdit = new InputField.EndEditEvent();
        // 使用静态字符串参数调用 SendCustomEvent；不能把玩家输入正文误当成事件名。
        UnityEventTools.AddStringPersistentListener(input.onEndEdit, backing.SendCustomEvent, eventName);
        EditorUtility.SetDirty(input);
    }

    static void WireInputSubmit(InputField input, NekoNpcChatInput chat, string eventName)
    {
        UdonBehaviour backing = GetBacking(chat);
        if (backing == null) throw new InvalidOperationException("找不到 NekoNpcChatInput 的 backing UdonBehaviour");
        input.onSubmit = new InputField.SubmitEvent();
        // 回车由 InputField 的提交事件直接发送，避免依赖 Udon Update 的逐帧按键时序。
        UnityEventTools.AddStringPersistentListener(input.onSubmit, backing.SendCustomEvent, eventName);
        EditorUtility.SetDirty(input);
    }

    static void EnsureEventSystem()
    {
        if (UnityEngine.Object.FindObjectOfType<EventSystem>() != null) return;
        GameObject go = new GameObject("YuiChatEventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
        Undo.RegisterCreatedObjectUndo(go, "创建 YUI 聊天 EventSystem");
    }

    static void SetLayerRecursively(GameObject go, int layer)
    {
        go.layer = layer;
        for (int i = 0; i < go.transform.childCount; i++)
            SetLayerRecursively(go.transform.GetChild(i).gameObject, layer);
    }

    static T AddUdon<T>(GameObject go) where T : Component
    {
        try
        {
            Type undo = FindType("UdonSharpEditor.UdonSharpUndo");
            if (undo != null)
            {
                foreach (MethodInfo method in undo.GetMethods(BindingFlags.Public | BindingFlags.Static))
                {
                    if (method.Name != "AddComponent" || !method.IsGenericMethodDefinition) continue;
                    ParameterInfo[] args = method.GetParameters();
                    if (args.Length != 1 || args[0].ParameterType != typeof(GameObject)) continue;
                    T result = method.MakeGenericMethod(typeof(T)).Invoke(null, new object[] { go }) as T;
                    if (result != null) return result;
                }
            }
            return Undo.AddComponent(go, typeof(T)) as T;
        }
        catch (Exception e)
        {
            Debug.LogWarning("[NEKO] 添加 " + typeof(T).Name + " 失败：" + e.Message);
            return null;
        }
    }

    static UdonBehaviour GetBacking(Component chat)
    {
        Type utility = FindType("UdonSharpEditor.UdonSharpEditorUtility");
        MethodInfo method = utility == null ? null : utility.GetMethod("GetBackingUdonBehaviour", BindingFlags.Public | BindingFlags.Static);
        return method == null ? null : method.Invoke(null, new object[] { chat }) as UdonBehaviour;
    }

    static void MarkDirtyAndSync(UnityEngine.Object target)
    {
        EditorUtility.SetDirty(target);
        try
        {
            Type utility = FindType("UdonSharpEditor.UdonSharpEditorUtility");
            Type behaviour = FindType("UdonSharp.UdonSharpBehaviour");
            MethodInfo method = utility == null || behaviour == null ? null : utility.GetMethod(
                "CopyProxyToUdon", BindingFlags.Public | BindingFlags.Static, null, new Type[] { behaviour }, null);
            if (method != null) method.Invoke(null, new object[] { target });
        }
        catch (Exception) { }
    }

    static Type FindType(string fullName)
    {
        foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type type = assembly.GetType(fullName, false);
            if (type != null) return type;
        }
        return null;
    }
}
#endif
