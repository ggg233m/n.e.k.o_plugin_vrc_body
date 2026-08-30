// NekoYuiFullNpcBuilder —— 火柴盒场景的 YUI v1.1/v1.2 完整 NPC 配置入口。
#if UNITY_EDITOR
using System;
using System.Linq;
using Unity.AI.Navigation;
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

    [MenuItem("NEKO/YUI NPC/2 Configure Open Scene Full v1.2")]
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

        Undo.RegisterCompleteObjectUndo(new UnityEngine.Object[] { telemetry, perception, locomotion, router, nameplate, sync }, "配置 YUI v1.2 完整 NPC");
        telemetry.specVersion = "1.2";
        router.worldName = EditorSceneManager.GetActiveScene().name;
        router.catalogRevision = Mathf.Max(router.catalogRevision, 4);
        router.enableGoto = true; router.enableFollow = true; router.enableWander = true;
        router.enableActions = true; router.enableExpressions = true;
        router.enableTextPreset = true; router.enableTextUtf8 = true;
        router.enableRayScan = true; router.enableTouch = true; router.enablePlayerPose = true;
        router.enableSnapshot = true; router.enableSocialSignals = true; router.enableAnchors = true; router.enableOperationLifecycle = true;
        router.enableWorldMap = true; router.enableSemanticNavigation = true;
        telemetry.logBudgetPerSec = 20;
        EnsureNameplateText(plate, nameplate);

        ConfigureExtendedActions(router);
        ConfigureExpressions(router);
        ConfigureAcceptanceArena(root);
        ConfigureWorldMapAndWander(root, router, locomotion);
        GameObject model = EnsureChiffonModel(root);

        // Animator Builder 会按实际 clip.length 回写目录时长，从而满足 ±20ms 契约。
        var controller = NekoNpcAnimatorBuilder.BuildController(false);
        var animator = model == null ? root.GetComponentInChildren<Animator>(true) : model.GetComponentInChildren<Animator>(true);
        if (animator == null) throw new InvalidOperationException("Chiffon 模型没有 Humanoid Animator。 ");
        animator.applyRootMotion = false; animator.cullingMode = AnimatorCullingMode.CullUpdateTransforms; animator.runtimeAnimatorController = controller;
        locomotion.animator = animator; locomotion.navAgent = root.GetComponent<NavMeshAgent>();
        sync.animator = animator; sync.router = router;
        PlacePerceptionAnchors(root, animator, perception);
        ConfigureEyeCamPlayerVisibility(root);

        EditorUtility.SetDirty(telemetry); EditorUtility.SetDirty(perception); EditorUtility.SetDirty(locomotion);
        EditorUtility.SetDirty(router); EditorUtility.SetDirty(nameplate); EditorUtility.SetDirty(sync); EditorUtility.SetDirty(animator);
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        EditorSceneManager.SaveOpenScenes(); AssetDatabase.SaveAssets(); AssetDatabase.Refresh();
        Debug.Log("[NEKO] 火柴盒 YUI NPC v1.2 完整配置完成：32 action、8 expression、7 anchor、2 region、1 entity、6 route_edge；upper_body_stream/voice_stream 保持未发布。 ");
    }

    [MenuItem("NEKO/YUI NPC/3 Configure Matchbox v1.2 + Validate")]
    public static void ConfigureMatchboxV12Batch()
    {
        const string scenePath = "Assets/Scenes/VRCDefaultWorldScene.unity";
        if (AssetDatabase.LoadAssetAtPath<SceneAsset>(scenePath) == null)
            throw new InvalidOperationException("找不到火柴盒场景：" + scenePath);
        EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
        ConfigureOpenSceneBatch();
        ValidateOpenSceneV12Batch();
    }

    [MenuItem("NEKO/YUI NPC/4 Validate Open Scene v1.2")]
    public static void ValidateOpenSceneV12Batch()
    {
        GameObject root = GameObject.Find(RootName);
        Transform scripts = root == null ? null : root.transform.Find("Scripts");
        if (root == null || scripts == null) throw new InvalidOperationException("火柴盒缺少 YUI NPC Rig");
        NekoNpcTelemetry telemetry = scripts.GetComponent<NekoNpcTelemetry>();
        NekoNpcLocomotion locomotion = scripts.GetComponent<NekoNpcLocomotion>();
        NekoMidiRouter router = scripts.GetComponent<NekoMidiRouter>();
        if (telemetry == null || locomotion == null || router == null) throw new InvalidOperationException("YUI 核心组件不完整");
        if (telemetry.specVersion != "1.2") throw new InvalidOperationException("Telemetry 未发布 spec 1.2");
        if (!router.enableWorldMap || !router.enableSemanticNavigation || !router.enableOperationLifecycle)
            throw new InvalidOperationException("v1.2 capability 未完整启用");
        if (telemetry.logBudgetPerSec > 20) throw new InvalidOperationException("日志预算超过每滑动秒 20 行");
        if (locomotion.orbitPointsPerLap != 24) throw new InvalidOperationException("绕行验收必须使用每圈 24 路点");
        if (router.anchorTransforms == null || router.anchorTransforms.Length != 7
            || router.regionSemanticKeys == null || router.regionSemanticKeys.Length != 2
            || router.entityCenters == null || router.entityCenters.Length != 1
            || router.routeFromAnchorIds == null || router.routeFromAnchorIds.Length != 6)
            throw new InvalidOperationException("v1.2 语义目录数量不符合火柴盒验收基线");
        ValidateNoLegacyFloorOverlap();

        string[] allKeys = router.anchorSemanticKeys.Concat(router.regionSemanticKeys).Concat(router.entitySemanticKeys).ToArray();
        if (allKeys.Any(string.IsNullOrWhiteSpace) || allKeys.Distinct(StringComparer.Ordinal).Count() != allKeys.Length)
            throw new InvalidOperationException("Anchor/Region/Entity semantic_key 必须全局唯一");
        for (int i = 0; i < router.anchorTransforms.Length; i++)
        {
            if (router.anchorTransforms[i] == null) throw new InvalidOperationException("Anchor " + i + " 未绑定 Transform");
            NavMeshHit hit;
            if (!NavMesh.SamplePosition(router.anchorTransforms[i].position, out hit, 0.75f, NavMesh.AllAreas))
                throw new InvalidOperationException("Anchor " + router.anchorSemanticKeys[i] + " 不在 NavMesh 上");
        }
        if (router.anchorTransforms[4].position.y - router.anchorTransforms[3].position.y < 2.5f)
            throw new InvalidOperationException("stair_top 与 stair_bottom 高差不足，不能验收三维导航");

        Camera eyeCamera = FindEyeCamera(root);
        if (eyeCamera != null)
        {
            if (!NekoEyeCamBuilder.PlayerLayersAreVisible(eyeCamera))
                throw new InvalidOperationException("EyeCamera 必须显示 Player/PlayerLocal/MirrorReflection，并排除 UI");
        }
        for (int i = 0; i < router.routeFromAnchorIds.Length; i++)
            RequirePath(router.anchorTransforms[router.routeFromAnchorIds[i]].position, router.anchorTransforms[router.routeToAnchorIds[i]].position, "route_edge " + i);

        Vector3 center = router.entityCenters[0].position;
        float radius = Mathf.Clamp(1.5f, router.entityOrbitMinRadius[0], router.entityOrbitMaxRadius[0]);
        Vector3 previous = router.anchorTransforms[2].position;
        for (int i = 0; i <= 24; i++)
        {
            float angle = Mathf.PI * 2f * i / 24f;
            Vector3 requested = new Vector3(center.x + Mathf.Cos(angle) * radius, center.y, center.z + Mathf.Sin(angle) * radius);
            NavMeshHit hit;
            if (!NavMesh.SamplePosition(requested, out hit, 0.75f, NavMesh.AllAreas))
                throw new InvalidOperationException("central_obstacle 圆周路点 " + i + " 不在 NavMesh 上");
            RequirePath(previous, hit.position, "central_obstacle orbit segment " + i);
            previous = hit.position;
        }
        Debug.Log("[NEKO] YUI v1.2 火柴盒静态验收通过：完整 XYZ 楼梯路径与 central_obstacle 圆周路径均可达。 ");
    }

    static void RequirePath(Vector3 from, Vector3 to, string label)
    {
        NavMeshHit fromHit; NavMeshHit toHit;
        if (!NavMesh.SamplePosition(from, out fromHit, 0.75f, NavMesh.AllAreas)
            || !NavMesh.SamplePosition(to, out toHit, 0.75f, NavMesh.AllAreas))
            throw new InvalidOperationException(label + " 的端点不在 NavMesh 上");
        NavMeshPath path = new NavMeshPath();
        if (!NavMesh.CalculatePath(fromHit.position, toHit.position, NavMesh.AllAreas, path)
            || path.status != NavMeshPathStatus.PathComplete)
            throw new InvalidOperationException(label + " 不可达");
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

    static void ConfigureWorldMapAndWander(GameObject root, NekoMidiRouter router, NekoNpcLocomotion locomotion)
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
        Vector3[] positions = {
            origin,
            new Vector3(-3f, 0f, -2f),
            new Vector3(0f, 0f, 0.3f),
            new Vector3(4.5f, 0f, -3.1f),
            new Vector3(4.5f, 3.1f, 1.2f),
            new Vector3(4.5f, 3.1f, 4.2f),
            new Vector3(-2f, 0f, 3.8f)
        };
        string[] objectNames = { "Anchor_Spawn", "Anchor_GroundPlaza", "Anchor_ObstacleApproach", "Anchor_StairBottom", "Anchor_StairTop", "Anchor_UpperObservation", "Anchor_PlayerMeet" };
        string[] keys = { "spawn_point", "ground_plaza", "central_obstacle_approach", "stair_bottom", "stair_top", "upper_observation", "player_meeting_point" };
        string[] descriptions = { "NPC 出生点", "火柴盒地面广场", "中央障碍物接近点", "楼梯下端", "楼梯上端", "楼上远端观察点", "与玩家会面的地点" };
        string[] tags = { "[\"spawn\",\"safe\"]", "[\"plaza\",\"social\"]", "[\"approach\",\"obstacle\"]", "[\"stairs\",\"ground\"]", "[\"stairs\",\"upper\"]", "[\"observation\",\"explore\",\"upper\"]", "[\"meeting\",\"social\"]" };
        string[] regionKeys = { "ground_floor", "ground_floor", "ground_floor", "ground_floor", "upper_floor", "upper_floor", "ground_floor" };
        var anchors = new Transform[positions.Length];
        for (int i = 0; i < positions.Length; i++) anchors[i] = EnsurePoint(holder, objectNames[i], positions[i]);
        router.anchorTransforms = anchors; router.anchorSemanticKeys = keys; router.anchorDescriptionsZh = descriptions;
        router.anchorHasYaw = new[] { false, true, true, true, true, true, true };
        router.anchorArrivalRadius = new[] { 0.3f, 0.35f, 0.35f, 0.35f, 0.35f, 0.4f, 0.35f };
        router.anchorTagsJson = tags; router.anchorRegionKeys = regionKeys;

        router.regionSemanticKeys = new[] { "ground_floor", "upper_floor" };
        router.regionDescriptionsZh = new[] { "火柴盒地面层，包含广场、中央障碍物和楼梯入口", "约三米高的平台层，包含远端观察点" };
        router.regionTagsJson = new[] { "[\"ground\",\"social\",\"obstacle\"]", "[\"upper\",\"observation\",\"stairs\"]" };
        router.regionFloorLabels = new[] { "G", "L1" };
        router.regionEntryAnchorIds = new[] { 1, 4 };

        Transform entityHolder = holder.Find("Entities");
        if (entityHolder == null) { var go = new GameObject("Entities"); Undo.RegisterCreatedObjectUndo(go, "创建 YUI entities"); go.transform.SetParent(holder, false); entityHolder = go.transform; }
        Transform obstacleCenter = EnsureRawPoint(entityHolder, "Entity_CentralObstacle", new Vector3(0f, 0f, 2.2f));
        router.entityCenters = new[] { obstacleCenter };
        router.entitySemanticKeys = new[] { "central_obstacle" };
        router.entityDescriptionsZh = new[] { "地面层中央的固定方柱，可用于连续绕行测试" };
        router.entityTagsJson = new[] { "[\"obstacle\",\"orbitable\",\"landmark\"]" };
        router.entityRegionKeys = new[] { "ground_floor" };
        router.entityApproachAnchorIds = new[] { 2 };
        router.entityOrbitable = new[] { true };
        router.entityOrbitMinRadius = new[] { 1.0f };
        router.entityOrbitMaxRadius = new[] { 2.2f };

        router.routeFromAnchorIds = new[] { 0, 1, 1, 3, 4, 1 };
        router.routeToAnchorIds = new[] { 1, 2, 3, 4, 5, 6 };
        router.routeBidirectional = new[] { true, true, true, true, true, true };
        router.routeTraversal = new[] { "walk", "walk", "walk", "stairs", "walk", "walk" };
        router.routeRegionKeys = new[] { "ground_floor", "ground_floor", "ground_floor", "upper_floor", "upper_floor", "ground_floor" };

        locomotion.wanderWaypoints = new[] { EnsurePoint(holder, "Wander_0", new Vector3(-4f, 0f, 0f)), EnsurePoint(holder, "Wander_1", new Vector3(-2f, 0f, 4.5f)) };
        locomotion.wanderSwitchDistance = 0.9f;
        locomotion.orbitPointsPerLap = 24;
        locomotion.whiskerHeight = 1.4f;
    }

    static Transform EnsurePoint(Transform parent, string name, Vector3 requested)
    {
        Transform point = parent.Find(name);
        if (point == null) { var go = new GameObject(name); Undo.RegisterCreatedObjectUndo(go, "创建 " + name); go.transform.SetParent(parent, true); point = go.transform; }
        NavMeshHit hit; point.position = NavMesh.SamplePosition(requested, out hit, 3f, NavMesh.AllAreas) ? hit.position : requested;
        point.rotation = Quaternion.identity; return point;
    }

    static Transform EnsureRawPoint(Transform parent, string name, Vector3 position)
    {
        Transform point = parent.Find(name);
        if (point == null) { var go = new GameObject(name); Undo.RegisterCreatedObjectUndo(go, "创建 " + name); go.transform.SetParent(parent, true); point = go.transform; }
        point.position = position; point.rotation = Quaternion.identity; return point;
    }

    static void ConfigureAcceptanceArena(GameObject root)
    {
        GameObject arena = GameObject.Find("NekoYuiV12_AcceptanceArena");
        if (arena == null) { arena = new GameObject("NekoYuiV12_AcceptanceArena"); Undo.RegisterCreatedObjectUndo(arena, "创建 YUI v1.2 验收场"); }
        arena.transform.position = Vector3.zero;

        GameObject ground = EnsureBox(arena.transform, "Ground", new Vector3(0f, -0.1f, 0f), new Vector3(18f, 0.2f, 18f), 0);
        DisableOverlappingLegacyFloor(ground, arena);
        EnsureBox(arena.transform, "CentralObstacle", new Vector3(0f, 0.75f, 2.2f), new Vector3(1.2f, 1.5f, 1.2f), 11);
        EnsureBox(arena.transform, "UpperPlatform", new Vector3(4.5f, 3f, 3f), new Vector3(4f, 0.2f, 4f), 0);
        Transform stairs = arena.transform.Find("Stairs");
        if (stairs == null) { var go = new GameObject("Stairs"); Undo.RegisterCreatedObjectUndo(go, "创建验收楼梯"); go.transform.SetParent(arena.transform, false); stairs = go.transform; }
        const int steps = 12;
        for (int i = 0; i < steps; i++)
        {
            float top = (i + 1) * 0.25f;
            EnsureBox(stairs, "Step_" + i.ToString("00"), new Vector3(4.5f, top * 0.5f, -3.05f + i * 0.35f), new Vector3(1.6f, top, 0.4f), 0);
        }

        NavMeshSurface surface = arena.GetComponent<NavMeshSurface>();
        if (surface == null) surface = arena.AddComponent<NavMeshSurface>();
        // 火柴盒旧测试面若与 v1.2 Surface 重叠，SamplePosition 可能把两个 Anchor
        // 分配到互不连接的数据岛。验收场只保留这一份统一烘焙结果。
        foreach (NavMeshSurface other in UnityEngine.Object.FindObjectsOfType<NavMeshSurface>())
        {
            if (other == null || other == surface) continue;
            other.RemoveData();
            other.enabled = false;
            EditorUtility.SetDirty(other);
        }
        surface.agentTypeID = 0;
        surface.collectObjects = CollectObjects.Volume;
        surface.useGeometry = NavMeshCollectGeometry.PhysicsColliders;
        surface.layerMask = ~0;
        surface.center = new Vector3(0f, 2f, 0f);
        surface.size = new Vector3(20f, 8f, 20f);
        surface.BuildNavMesh();

        NavMeshHit spawn;
        if (NavMesh.SamplePosition(root.transform.position, out spawn, 2f, NavMesh.AllAreas)) root.transform.position = spawn.position;
        EditorUtility.SetDirty(surface); EditorUtility.SetDirty(arena);
    }

    static void DisableOverlappingLegacyFloor(GameObject acceptanceGround, GameObject arena)
    {
        Collider acceptanceCollider = acceptanceGround.GetComponent<Collider>();
        if (acceptanceCollider == null) throw new InvalidOperationException("v1.2 验收场 Ground 缺少 Collider");
        Bounds acceptanceBounds = acceptanceCollider.bounds;

        foreach (GameObject candidate in EditorSceneManager.GetActiveScene().GetRootGameObjects())
        {
            if (candidate == null || candidate == arena || candidate.name != "Floor") continue;
            if (!TryGetCombinedBounds(candidate, out Bounds legacyBounds) || !OverlapsGroundSurface(acceptanceBounds, legacyBounds)) continue;

            // 旧默认 Floor 与 v1.2 Ground 共面时会产生视觉叠层，也会被 PhysicsColliders
            // 重复采进 NavMesh。保留对象和 Transform，便于撤销；只关闭参与重叠的组件。
            Renderer[] renderers = candidate.GetComponentsInChildren<Renderer>(true);
            Collider[] colliders = candidate.GetComponentsInChildren<Collider>(true);
            Undo.RecordObjects(renderers.Cast<UnityEngine.Object>().Concat(colliders).ToArray(), "禁用重叠的旧 Floor");
            foreach (Renderer renderer in renderers)
            {
                if (renderer == null) continue;
                renderer.enabled = false;
                EditorUtility.SetDirty(renderer);
            }
            foreach (Collider collider in colliders)
            {
                if (collider == null) continue;
                collider.enabled = false;
                EditorUtility.SetDirty(collider);
            }
            EditorUtility.SetDirty(candidate);
            Debug.Log("[NEKO] 已禁用与 v1.2 Ground 共面的旧 Floor 渲染和碰撞，避免视觉/NavMesh 重叠。 ");
        }
    }

    static void ValidateNoLegacyFloorOverlap()
    {
        GameObject arena = GameObject.Find("NekoYuiV12_AcceptanceArena");
        Transform ground = arena == null ? null : arena.transform.Find("Ground");
        Collider acceptanceCollider = ground == null ? null : ground.GetComponent<Collider>();
        if (acceptanceCollider == null) throw new InvalidOperationException("v1.2 验收场 Ground 缺失");
        Bounds acceptanceBounds = acceptanceCollider.bounds;

        foreach (GameObject candidate in EditorSceneManager.GetActiveScene().GetRootGameObjects())
        {
            if (candidate == null || candidate == arena || candidate.name != "Floor") continue;
            if (!TryGetCombinedBounds(candidate, out Bounds legacyBounds) || !OverlapsGroundSurface(acceptanceBounds, legacyBounds)) continue;
            if (candidate.GetComponentsInChildren<Renderer>(true).Any(item => item != null && item.enabled)
                || candidate.GetComponentsInChildren<Collider>(true).Any(item => item != null && item.enabled))
                throw new InvalidOperationException("旧 Floor 仍与 v1.2 Ground 共面并处于启用状态");
        }
    }

    static bool TryGetCombinedBounds(GameObject root, out Bounds bounds)
    {
        Renderer[] renderers = root.GetComponentsInChildren<Renderer>(true);
        Collider[] colliders = root.GetComponentsInChildren<Collider>(true);
        bool initialized = false;
        bounds = new Bounds(root.transform.position, Vector3.zero);
        foreach (Renderer renderer in renderers)
        {
            if (renderer == null) continue;
            if (!initialized) { bounds = renderer.bounds; initialized = true; }
            else bounds.Encapsulate(renderer.bounds);
        }
        foreach (Collider collider in colliders)
        {
            if (collider == null) continue;
            if (!initialized) { bounds = collider.bounds; initialized = true; }
            else bounds.Encapsulate(collider.bounds);
        }
        return initialized;
    }

    static bool OverlapsGroundSurface(Bounds acceptance, Bounds legacy)
    {
        bool overlapsX = acceptance.min.x < legacy.max.x && acceptance.max.x > legacy.min.x;
        bool overlapsZ = acceptance.min.z < legacy.max.z && acceptance.max.z > legacy.min.z;
        bool sameSurface = Mathf.Abs(acceptance.max.y - legacy.max.y) <= 0.15f;
        return overlapsX && overlapsZ && sameSurface;
    }

    static GameObject EnsureBox(Transform parent, string name, Vector3 position, Vector3 scale, int layer)
    {
        Transform existing = parent.Find(name);
        GameObject box = existing == null ? GameObject.CreatePrimitive(PrimitiveType.Cube) : existing.gameObject;
        if (existing == null) { box.name = name; Undo.RegisterCreatedObjectUndo(box, "创建 " + name); box.transform.SetParent(parent, true); }
        box.transform.position = position; box.transform.rotation = Quaternion.identity; box.transform.localScale = scale; box.layer = layer;
        return box;
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

    static void ConfigureEyeCamPlayerVisibility(GameObject root)
    {
        Camera eyeCamera = FindEyeCamera(root);
        if (eyeCamera != null) NekoEyeCamBuilder.EnsurePlayerLayersVisible(eyeCamera);
    }

    static Camera FindEyeCamera(GameObject root)
    {
        return root.GetComponentsInChildren<Camera>(true).FirstOrDefault(item => item != null && item.gameObject.name == "EyeCamera");
    }

    static T[] Resize<T>(T[] source, int length)
    {
        var result = new T[length]; if (source != null) Array.Copy(source, result, Mathf.Min(source.Length, length)); return result;
    }
}
#endif
