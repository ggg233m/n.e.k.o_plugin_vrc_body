// NekoYuiFullNpcBuilder —— 火柴盒场景的 YUI v1.1 完整 NPC 配置入口。
#if UNITY_EDITOR
using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;

public static class NekoYuiFullNpcBuilder
{
    const string RootName = "NekoNpc_YUI";
    const string ChiffonPackage = "H:/Unity/vrc/ChiffonLite_2026-7-14/ChiffonLite/ChiffonLite_2026-7-14.unitypackage";

    static readonly string[] ExtensionKeys = {
        "cat_pose", "happy_one", "happy_two", "stand_clap", "sit_clap", "dance", "naughty", "spin",
        "jump", "pose_stand_one", "pose_stand_two", "pose_sit_one", "pose_sit_two", "pose_squat", "pose_down", "pose_float"
    };
    static readonly string[] ExtensionDescriptions = {
        "可爱的猫姿动作", "第一种开心动作", "第二种开心动作", "站立鼓掌", "坐姿鼓掌", "舞蹈动作", "调皮动作", "原地旋转",
        "跳跃动作", "第一种站立姿势", "第二种站立姿势", "第一种坐姿", "第二种坐姿", "蹲伏姿势", "趴伏姿势", "漂浮姿势"
    };

    [MenuItem("NEKO/YUI NPC/1 Import ChiffonLite")]
    public static void ImportChiffonBatch()
    {
        if (FindChiffonPrefab() != null) { Debug.Log("[NEKO] ChiffonLiteMB_Medium 已导入，跳过 package。 "); return; }
        if (!System.IO.File.Exists(ChiffonPackage)) throw new System.IO.FileNotFoundException("找不到 Chiffon package", ChiffonPackage);
        AssetDatabase.ImportPackage(ChiffonPackage, false);
        AssetDatabase.Refresh();
        Debug.Log("[NEKO] 已导入 ChiffonLite package。请等待脚本重编译后执行 ConfigureOpenSceneBatch。 ");
    }

    [MenuItem("NEKO/YUI NPC/2 Configure Open Scene Full v1.1")]
    public static void ConfigureOpenSceneBatch()
    {
        var root = GameObject.Find(RootName);
        if (root == null) throw new InvalidOperationException("当前场景缺少 " + RootName + "；先使用已有火柴盒 Rig 或运行 Build NPC Rig。 ");
        var scripts = root.transform.Find("Scripts");
        var plate = root.transform.Find("Nameplate");
        if (scripts == null || plate == null) throw new InvalidOperationException("YUI Rig 缺少 Scripts/Nameplate。 ");

        var telemetry = scripts.GetComponent<NekoNpcTelemetry>();
        var perception = scripts.GetComponent<NekoNpcPerception>();
        var locomotion = scripts.GetComponent<NekoNpcLocomotion>();
        var router = scripts.GetComponent<NekoMidiRouter>();
        var nameplate = plate.GetComponent<NekoNpcNameplate>();
        var sync = root.GetComponent<NekoNpcSync>();
        if (telemetry == null || perception == null || locomotion == null || router == null || nameplate == null || sync == null)
            throw new InvalidOperationException("YUI Rig UdonSharp 组件不完整。 ");

        Undo.RegisterCompleteObjectUndo(new UnityEngine.Object[] { telemetry, perception, locomotion, router, nameplate, sync }, "配置 YUI v1.1 完整 NPC");
        router.worldName = EditorSceneManager.GetActiveScene().name;
        router.catalogRevision = Mathf.Max(router.catalogRevision, 3);
        router.enableGoto = true; router.enableFollow = true; router.enableWander = true;
        router.enableActions = true; router.enableExpressions = true;
        router.enableTextPreset = true; router.enableTextUtf8 = true;
        router.enableRayScan = true; router.enableTouch = true; router.enablePlayerPose = true;
        router.enableSnapshot = true; router.enableSocialSignals = true; router.enableAnchors = true; router.enableOperationLifecycle = true;
        telemetry.logBudgetPerSec = 20;
        EnsureNameplateText(plate, nameplate);

        ConfigureExtendedActions(router);
        ConfigureExpressions(router);
        ConfigureAnchorsAndWander(root, router, locomotion);
        GameObject model = EnsureChiffonModel(root);

        // Animator Builder 会按实际 clip.length 回写目录时长，从而满足 ±20ms 契约。
        var controller = NekoNpcAnimatorBuilder.BuildController(false);
        var animator = model == null ? root.GetComponentInChildren<Animator>(true) : model.GetComponentInChildren<Animator>(true);
        if (animator == null) throw new InvalidOperationException("Chiffon 模型没有 Humanoid Animator。 ");
        animator.applyRootMotion = false; animator.cullingMode = AnimatorCullingMode.CullUpdateTransforms; animator.runtimeAnimatorController = controller;
        locomotion.animator = animator; locomotion.navAgent = root.GetComponent<NavMeshAgent>();
        sync.animator = animator; sync.router = router;
        PlacePerceptionAnchors(root, animator, perception);

        EditorUtility.SetDirty(telemetry); EditorUtility.SetDirty(perception); EditorUtility.SetDirty(locomotion);
        EditorUtility.SetDirty(router); EditorUtility.SetDirty(nameplate); EditorUtility.SetDirty(sync); EditorUtility.SetDirty(animator);
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        EditorSceneManager.SaveOpenScenes(); AssetDatabase.SaveAssets(); AssetDatabase.Refresh();
        Debug.Log("[NEKO] 火柴盒 YUI NPC v1.1 完整配置完成：32 action（16 核心 + 16 Chiffon 扩展）、8 expression、4 anchor、2 wander waypoint；upper_body_stream/voice_stream 保持未发布。 ");
    }

    static void EnsureNameplateText(Transform plate, NekoNpcNameplate nameplate)
    {
        // 旧火柴盒 Rig 可能只有名牌；文本组件缺失时 Router 会按规范撤销 text capability。
        var nameObject = plate.Find("NameText");
        if (nameplate.nameText == null && nameObject != null) nameplate.nameText = nameObject.GetComponent<TMPro.TextMeshPro>();
        if (nameplate.nameText == null)
        {
            nameplate.nameText = plate.GetComponentsInChildren<TMPro.TextMeshPro>(true)
                .FirstOrDefault(item => item != null && item.font != null && item.gameObject.name != "BubbleText");
        }
        var legacyName = plate.GetComponentInChildren<TextMesh>(true);
        if (legacyName != null && nameplate.nameText != null)
        {
            // 火柴盒旧 Rig 已有可用 TextMesh 名牌时，不再叠加第二个 TMP 名牌。
            var duplicate = nameplate.nameText.gameObject;
            nameplate.nameText = null;
            Undo.DestroyObjectImmediate(duplicate);
        }
        var bubbleObject = plate.Find("BubbleText");
        var bubble = bubbleObject == null ? null : bubbleObject.GetComponent<TMPro.TextMeshPro>();
        if (bubble == null || bubble.font == null)
        {
            if (bubbleObject != null) Undo.DestroyObjectImmediate(bubbleObject.gameObject);
            var source = plate.GetComponentsInChildren<TMPro.TextMeshPro>(true)
                .FirstOrDefault(item => item != null && item.font != null && item.gameObject.name != "BubbleText");
            var go = source == null
                ? new GameObject("BubbleText")
                : UnityEngine.Object.Instantiate(source.gameObject, plate, false);
            Undo.RegisterCreatedObjectUndo(go, "创建 YUI 文本气泡");
            go.name = "BubbleText";
            if (source == null) go.transform.SetParent(plate, false);
            bubbleObject = go.transform;
            bubble = go.GetComponent<TMPro.TextMeshPro>();
            if (bubble == null) bubble = go.AddComponent<TMPro.TextMeshPro>();
        }
        bubbleObject.localPosition = new Vector3(0f, 2.2f, 0f);
        bubble.text = "";
        bubble.fontSize = 2f;
        bubble.alignment = TMPro.TextAlignmentOptions.Center;
        bubble.enableWordWrapping = false;
        bubble.color = Color.white;
        nameplate.bubbleText = bubble;
    }

    static void ConfigureExtendedActions(NekoMidiRouter router)
    {
        const int core = 16; int total = 32;
        router.actionNames = Resize(router.actionNames, total);
        router.actionSemanticKeys = Resize(router.actionSemanticKeys, total);
        router.actionDescriptionsZh = Resize(router.actionDescriptionsZh, total);
        router.actionIntentTagsJson = Resize(router.actionIntentTagsJson, total);
        router.actionTargetRequired = Resize(router.actionTargetRequired, total);
        router.actionSpeechCompatible = Resize(router.actionSpeechCompatible, total);
        router.actionLayers = Resize(router.actionLayers, total);
        router.actionDurationMs = Resize(router.actionDurationMs, total);
        router.actionLoopable = Resize(router.actionLoopable, total);
        router.actionMovement = Resize(router.actionMovement, total);
        router.actionPriority = Resize(router.actionPriority, total);
        router.actionInterruptible = Resize(router.actionInterruptible, total);
        router.actionFadeInMs = Resize(router.actionFadeInMs, total);
        router.actionFadeOutMs = Resize(router.actionFadeOutMs, total);
        for (int i = 0; i < ExtensionKeys.Length; i++)
        {
            int id = core + i; string key = ExtensionKeys[i];
            router.actionNames[id] = key; router.actionSemanticKeys[id] = key; router.actionDescriptionsZh[id] = ExtensionDescriptions[i];
            router.actionIntentTagsJson[id] = (key == "dance" || key == "spin") ? "[\"performance\",\"playful\"]" : "[\"gesture\"]";
            router.actionTargetRequired[id] = "none"; router.actionSpeechCompatible[id] = key != "dance";
            router.actionLayers[id] = "full_body"; router.actionDurationMs[id] = 2000;
            router.actionLoopable[id] = key == "dance" || key == "pose_float" || key.StartsWith("pose_");
            router.actionMovement[id] = "block"; router.actionPriority[id] = 60; router.actionInterruptible[id] = true;
            router.actionFadeInMs[id] = 150; router.actionFadeOutMs[id] = 220;
        }
    }

    static void ConfigureExpressions(NekoMidiRouter router)
    {
        // 0..3 是冻结核心；idle 作为扩展语义别名保留，不能替换 neutral。
        router.expressionNames = new[] { "neutral", "happy", "sad", "surprised", "huff", "cry", "wink", "idle" };
        router.expressionSemanticKeys = new[] { "neutral", "happy", "sad", "surprised", "huff", "cry", "wink", "idle" };
        router.expressionDescriptionsZh = new[] { "中立表情", "开心表情", "悲伤表情", "惊讶表情", "气鼓鼓的表情", "哭泣表情", "眨眼表情", "回到待机表情" };
        router.expressionDefaultDurationMs = new[] { 0, 3000, 3000, 1800, 2500, 3000, 1200, 0 };
        router.expressionFadeMs = new[] { 150, 150, 180, 120, 150, 180, 100, 150 };
    }

    static void ConfigureAnchorsAndWander(GameObject root, NekoMidiRouter router, NekoNpcLocomotion locomotion)
    {
        // 根 NPC 本身位于场景顶层时不能用 parent.Find；否则每次运行都会再创建一个同名根。
        Transform holder = null;
        Transform parent = root.transform.parent;
        Transform[] siblings = parent != null
            ? parent.Cast<Transform>().Where(item => item.name == "YuiSemanticAnchors").ToArray()
            : EditorSceneManager.GetActiveScene().GetRootGameObjects().Where(item => item.name == "YuiSemanticAnchors").Select(item => item.transform).ToArray();
        if (siblings.Length > 0) holder = siblings[0];
        for (int i = 1; i < siblings.Length; i++) Undo.DestroyObjectImmediate(siblings[i].gameObject);
        if (holder == null)
        {
            var go = new GameObject("YuiSemanticAnchors"); Undo.RegisterCreatedObjectUndo(go, "创建 YUI semantic anchors");
            if (parent != null) go.transform.SetParent(parent, false); holder = go.transform;
        }
        Vector3 origin = root.transform.position;
        Vector3[] offsets = { Vector3.zero, new Vector3(0f, 0f, 2f), new Vector3(-2f, 0f, 2f), new Vector3(2f, 0f, 2f) };
        string[] objectNames = { "Anchor_Spawn", "Anchor_Plaza", "Anchor_Observation", "Anchor_PlayerMeet" };
        string[] keys = { "spawn_point", "plaza", "observation_point", "player_meeting_point" };
        string[] descriptions = { "NPC 出生点", "火柴盒广场", "观察世界的位置", "与玩家会面的地点" };
        string[] tags = { "[\"spawn\",\"safe\"]", "[\"plaza\",\"social\"]", "[\"observation\",\"explore\"]", "[\"meeting\",\"social\"]" };
        var anchors = new Transform[4];
        for (int i = 0; i < 4; i++) anchors[i] = EnsurePoint(holder, objectNames[i], origin + offsets[i]);
        router.anchorTransforms = anchors; router.anchorSemanticKeys = keys; router.anchorDescriptionsZh = descriptions;
        router.anchorHasYaw = new[] { false, true, true, true }; router.anchorArrivalRadius = new[] { 0.3f, 0.3f, 0.3f, 0.3f }; router.anchorTagsJson = tags;

        locomotion.wanderWaypoints = new[] { EnsurePoint(holder, "Wander_0", origin + new Vector3(-2f, 0f, 0f)), EnsurePoint(holder, "Wander_1", origin + new Vector3(2f, 0f, 0f)) };
        locomotion.wanderSwitchDistance = 0.9f;
    }

    static Transform EnsurePoint(Transform parent, string name, Vector3 requested)
    {
        Transform point = parent.Find(name);
        if (point == null) { var go = new GameObject(name); Undo.RegisterCreatedObjectUndo(go, "创建 " + name); go.transform.SetParent(parent, true); point = go.transform; }
        NavMeshHit hit; point.position = NavMesh.SamplePosition(requested, out hit, 3f, NavMesh.AllAreas) ? hit.position : requested;
        point.rotation = Quaternion.identity; return point;
    }

    static GameObject EnsureChiffonModel(GameObject root)
    {
        Transform container = root.transform.Find("Model");
        if (container == null) { var go = new GameObject("Model"); Undo.RegisterCreatedObjectUndo(go, "创建 Model"); go.transform.SetParent(root.transform, false); container = go.transform; }
        Transform existing = container.Find("ChiffonLiteMB_Medium_NPC");
        GameObject instance = existing == null ? null : existing.gameObject;
        if (instance == null)
        {
            GameObject prefab = FindChiffonPrefab();
            if (prefab == null) throw new InvalidOperationException("未找到 ChiffonLiteMB_Medium.prefab；先执行 Import ChiffonLite。 ");
            instance = PrefabUtility.InstantiatePrefab(prefab) as GameObject;
            if (instance == null) throw new InvalidOperationException("Chiffon prefab 实例化失败。 ");
            Undo.RegisterCreatedObjectUndo(instance, "实例化 Chiffon NPC"); instance.name = "ChiffonLiteMB_Medium_NPC";
            instance.transform.SetParent(container, false); instance.transform.localPosition = Vector3.zero; instance.transform.localRotation = Quaternion.identity; instance.transform.localScale = Vector3.one;
        }
        foreach (Transform child in container.Cast<Transform>().ToArray()) if (child != instance.transform && child.name.StartsWith("Placeholder_")) Undo.DestroyObjectImmediate(child.gameObject);
        StripAvatarOnlyComponents(instance);
        foreach (var collider in instance.GetComponentsInChildren<Collider>(true)) if (collider != null && !collider.isTrigger) Undo.DestroyObjectImmediate(collider);
        return instance;
    }

    static GameObject FindChiffonPrefab()
    {
        foreach (string guid in AssetDatabase.FindAssets("ChiffonLiteMB_Medium t:Prefab"))
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (System.IO.Path.GetFileNameWithoutExtension(path) == "ChiffonLiteMB_Medium") return AssetDatabase.LoadAssetAtPath<GameObject>(path);
        }
        return null;
    }

    static void StripAvatarOnlyComponents(GameObject avatar)
    {
        string[] banned = { "VRCAvatarDescriptor", "PipelineManager", "VRCContactReceiver", "VRCContactSender", "VRCStation", "ModularAvatar", "VRCFury", "NDMF" };
        foreach (Transform transform in avatar.GetComponentsInChildren<Transform>(true))
        {
            if (GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(transform.gameObject) > 0) GameObjectUtility.RemoveMonoBehavioursWithMissingScript(transform.gameObject);
        }
        foreach (Component component in avatar.GetComponentsInChildren<Component>(true))
        {
            if (component == null || component is Transform || component is Animator || component is Renderer) continue;
            string type = component.GetType().FullName ?? component.GetType().Name;
            if (banned.Any(item => type.IndexOf(item, StringComparison.OrdinalIgnoreCase) >= 0)) Undo.DestroyObjectImmediate(component);
        }
    }

    static void PlacePerceptionAnchors(GameObject root, Animator animator, NekoNpcPerception perception)
    {
        Transform head = animator.GetBoneTransform(HumanBodyBones.Head);
        Transform eye = root.transform.Find("EyeAnchor");
        if (head != null && eye != null)
        {
            eye.SetParent(head, true); eye.position = head.position + root.transform.forward * 0.16f + Vector3.up * 0.04f; eye.rotation = root.transform.rotation;
            perception.eyeAnchor = eye;
        }
        perception.npcRoot = root.transform;
    }

    static T[] Resize<T>(T[] source, int length)
    {
        var result = new T[length]; if (source != null) Array.Copy(source, result, Mathf.Min(source.Length, length)); return result;
    }
}
#endif
