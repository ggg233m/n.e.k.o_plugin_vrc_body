// NekoNpcRigBuilder —— 一键生成 YUI NPC 骨架（纯 Editor 脚本）
//
// 放置位置：Assets/NEKO/Editor/NekoNpcRigBuilder.cs（必须在 Editor 文件夹）
// 菜单：NEKO → Build NPC Rig (YUI)
//
// 生成结构（与《NPC Udon 接口规范 v1.0》§2 一致）：
//   NekoNpc_YUI            根：Rigidbody(kinematic) + NekoNpcSync
//   ├── Model              占位胶囊（无碰撞体，换成 humanoid 后删掉占位）
//   ├── EyeAnchor          眼位空物体（Stream 相机手动钉此处）
//   ├── TouchZones         Head / Cheek / HandL / HandR / Torso（isTrigger + NekoNpcTouch）
//   ├── Nameplate          名牌 + 气泡（TextMeshPro + NekoNpcNameplate）
//   └── Scripts            Telemetry / MidiRouter / Locomotion / Perception
//
// U# 组件通过反射调用 UdonSharpEditor.UdonSharpUndo.AddComponent<T> 添加并自动连线；
// 若当前 U# 版本没有该 API，会在 Console 提示改为手动 Add Component（结构与碰撞体照常生成）。
// 幂等：已存在 NekoNpc_YUI 时询问是否删除重建。

#if UNITY_EDITOR
using System;
using System.Reflection;
using TMPro;
using Unity.AI.Navigation;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;

public static class NekoNpcRigBuilder
{
    const string RootName = "NekoNpc_YUI";

    [MenuItem("NEKO/Build NPC Rig (YUI)")]
    public static void Build()
    {
        var old = GameObject.Find(RootName);
        if (old != null)
        {
            if (!EditorUtility.DisplayDialog("NEKO", "已存在 " + RootName + "，删除重建？（会丢失手动改动）", "重建", "取消"))
                return;
            Undo.DestroyObjectImmediate(old);
        }

        Vector3 spawn = Vector3.zero;
        if (Selection.activeTransform != null) spawn = Selection.activeTransform.position;

        var root = new GameObject(RootName);
        Undo.RegisterCreatedObjectUndo(root, "Build NPC Rig");
        root.transform.position = spawn;
        var rb = root.AddComponent<Rigidbody>();
        rb.isKinematic = true;
        rb.useGravity = false;
        var agent = root.AddComponent<NavMeshAgent>();
        ConfigureAgent(agent);

        // Model 占位（无碰撞体！须式射线不能打到自己）
        var model = new GameObject("Model");
        model.transform.SetParent(root.transform, false);
        var cap = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        cap.name = "Placeholder_Capsule_DeleteMe";
        cap.transform.SetParent(model.transform, false);
        cap.transform.localPosition = new Vector3(0f, 0.8f, 0f);
        cap.transform.localScale = new Vector3(0.5f, 0.8f, 0.5f);
        UnityEngine.Object.DestroyImmediate(cap.GetComponent<Collider>());
        var nose = GameObject.CreatePrimitive(PrimitiveType.Cube);
        nose.name = "Placeholder_Front_DeleteMe";
        nose.transform.SetParent(model.transform, false);
        nose.transform.localPosition = new Vector3(0f, 1.5f, 0.25f);
        nose.transform.localScale = new Vector3(0.1f, 0.1f, 0.2f);
        UnityEngine.Object.DestroyImmediate(nose.GetComponent<Collider>());

        var eye = new GameObject("EyeAnchor");
        eye.transform.SetParent(root.transform, false);
        // 眼位放在占位"鼻子"方块(z 0.15–0.35)前面，否则眼位相机整天盯着自己鼻尖（2026-08-29 实测）
        eye.transform.localPosition = new Vector3(0f, 1.5f, 0.36f);

        var zones = new GameObject("TouchZones");
        zones.transform.SetParent(root.transform, false);
        var zHead = Sphere(zones, "Head", new Vector3(0f, 1.55f, 0f), 0.25f);
        var zCheek = Sphere(zones, "Cheek", new Vector3(0f, 1.45f, 0.14f), 0.12f);
        var zHandL = Sphere(zones, "HandL", new Vector3(-0.35f, 1.0f, 0.1f), 0.12f);
        var zHandR = Sphere(zones, "HandR", new Vector3(0.35f, 1.0f, 0.1f), 0.12f);
        var zTorso = new GameObject("Torso");
        zTorso.transform.SetParent(zones.transform, false);
        zTorso.transform.localPosition = new Vector3(0f, 1.1f, 0f);
        var capCol = zTorso.AddComponent<CapsuleCollider>();
        capCol.isTrigger = true; capCol.radius = 0.25f; capCol.height = 0.7f;

        var plate = new GameObject("Nameplate");
        plate.transform.SetParent(root.transform, false);
        plate.transform.localPosition = new Vector3(0f, 0f, 0f);
        var nameText = Text(plate, "NameText", new Vector3(0f, 1.95f, 0f), "YUI", 2.5f);
        var bubbleText = Text(plate, "BubbleText", new Vector3(0f, 2.2f, 0f), "", 2f);

        var scripts = new GameObject("Scripts");
        scripts.transform.SetParent(root.transform, false);

        // ---- U# 组件（反射添加；失败则提示手动） ----
        var tel = AddUdon<NekoNpcTelemetry>(scripts);
        var per = AddUdon<NekoNpcPerception>(scripts);
        var loco = AddUdon<NekoNpcLocomotion>(scripts);
        var router = AddUdon<NekoMidiRouter>(scripts);
        var sync = AddUdon<NekoNpcSync>(root);
        var np = AddUdon<NekoNpcNameplate>(plate);
        var tHead = AddUdon<NekoNpcTouch>(zHead);
        var tCheek = AddUdon<NekoNpcTouch>(zCheek);
        var tHandL = AddUdon<NekoNpcTouch>(zHandL);
        var tHandR = AddUdon<NekoNpcTouch>(zHandR);
        var tTorso = AddUdon<NekoNpcTouch>(zTorso);

        // VRC Midi Listener：没有它 Udon 收不到任何 MIDI 事件（官方 Realtime MIDI 文档）
        bool midiOk = router != null && AddMidiListener(scripts, router);

        bool wired = tel != null && per != null && loco != null && router != null && sync != null && np != null
                     && tHead != null && tCheek != null && tHandL != null && tHandR != null && tTorso != null;
        if (wired)
        {
            per.telemetry = tel; per.npcRoot = root.transform; per.eyeAnchor = eye.transform;
            loco.telemetry = tel; loco.perception = per; loco.router = router; loco.npcRoot = root.transform; loco.navAgent = agent;
            router.telemetry = tel; router.locomotion = loco; router.perception = per; router.nameplate = np;
            sync.telemetry = tel; sync.locomotion = loco;
            np.nameText = nameText; np.bubbleText = bubbleText;
            Wire(tHead, tel, per, "head"); Wire(tCheek, tel, per, "cheek");
            Wire(tHandL, tel, per, "handL"); Wire(tHandR, tel, per, "handR"); Wire(tTorso, tel, per, "torso");
            MarkDirty(tel); MarkDirty(per); MarkDirty(loco); MarkDirty(router); MarkDirty(sync); MarkDirty(np);
            MarkDirty(tHead); MarkDirty(tCheek); MarkDirty(tHandL); MarkDirty(tHandR); MarkDirty(tTorso);
            Debug.Log("[NEKO] NPC Rig 生成完毕并已连线。接下来：1) 填 Scripts/NekoNpcTelemetry.driverDisplayName；"
                      + "2) 填 NekoMidiRouter.boundsMin/Max 与 worldName；3) 把 humanoid 模型放到 Model 下、Animator 拖给 Locomotion 与 Sync；"
                      + "4) 删除 Placeholder_* 占位物；5) 检查 Scripts 上的 VRC Midi Listener：Behaviour=NekoMidiRouter，Active Events 勾 Note On / Note Off / Control Change"
                      + (midiOk ? "（已自动配置，请目视确认）" : "（!! 未能自动添加，请手动 Add Component → VRC Midi Listener）") + "。");
        }
        else
        {
            Debug.LogWarning("[NEKO] NPC Rig 结构已生成，但 U# 组件未能自动添加（UdonSharpUndo.AddComponent 不可用）。"
                             + "请手动 Add Component：Scripts→Telemetry/Perception/Locomotion/MidiRouter，根→NekoNpcSync，"
                             + "Nameplate→NekoNpcNameplate，TouchZones 各子物体→NekoNpcTouch 并填 part，然后按 README 连线。");
        }

        Selection.activeGameObject = root;
    }

    // ---- 重连：对已存在的 NekoNpc_YUI 按名字重新建立全部引用（2026-08-29：实测 Build 后引用全为 None 的兜底）----
    [MenuItem("NEKO/Rewire NPC Rig (selected or NekoNpc_YUI)")]
    public static void Rewire()
    {
        GameObject root = Selection.activeGameObject;
        while (root != null && root.transform.parent != null && root.GetComponent<NekoNpcSync>() == null) root = root.transform.parent.gameObject;
        if (root == null || root.GetComponent<NekoNpcSync>() == null) root = GameObject.Find("NekoNpc_YUI");
        if (root == null) { Debug.LogError("[NEKO] 场景里找不到 NekoNpc_YUI（或选中的物体不在它下面）"); return; }

        Transform eye = root.transform.Find("EyeAnchor");
        Transform scripts = root.transform.Find("Scripts");
        Transform zones = root.transform.Find("TouchZones");
        Transform plate = root.transform.Find("Nameplate");
        if (scripts == null) { Debug.LogError("[NEKO] " + root.name + " 下没有 Scripts 子物体"); return; }

        var tel = scripts.GetComponent<NekoNpcTelemetry>();
        var per = scripts.GetComponent<NekoNpcPerception>();
        var loco = scripts.GetComponent<NekoNpcLocomotion>();
        var router = scripts.GetComponent<NekoMidiRouter>();
        var sync = root.GetComponent<NekoNpcSync>();
        var np = plate != null ? plate.GetComponent<NekoNpcNameplate>() : null;
        int missing = 0;
        if (tel == null) { Debug.LogWarning("[NEKO] Scripts 缺 NekoNpcTelemetry"); missing++; }
        if (per == null) { Debug.LogWarning("[NEKO] Scripts 缺 NekoNpcPerception"); missing++; }
        if (loco == null) { Debug.LogWarning("[NEKO] Scripts 缺 NekoNpcLocomotion"); missing++; }
        if (router == null) { Debug.LogWarning("[NEKO] Scripts 缺 NekoMidiRouter"); missing++; }
        if (missing > 0) { Debug.LogError("[NEKO] 缺组件，先手动 Add Component 再重连"); return; }

        Undo.RegisterCompleteObjectUndo(root, "NEKO Rewire");
        per.telemetry = tel; per.npcRoot = root.transform; per.eyeAnchor = eye;
        loco.telemetry = tel; loco.perception = per; loco.router = router; loco.npcRoot = root.transform;
        router.telemetry = tel; router.locomotion = loco; router.perception = per; router.nameplate = np;
        if (sync != null) { sync.telemetry = tel; sync.locomotion = loco; }
        if (np != null && plate != null)
        {
            var nt = plate.Find("NameText"); var bt = plate.Find("BubbleText");
            if (nt != null) np.nameText = nt.GetComponent<TextMeshPro>();
            if (bt != null) np.bubbleText = bt.GetComponent<TextMeshPro>();
        }
        int touches = 0;
        if (zones != null)
        {
            string[] names = { "Head", "Cheek", "HandL", "HandR", "Torso" };
            string[] parts = { "head", "cheek", "handL", "handR", "torso" };
            for (int i = 0; i < names.Length; i++)
            {
                var z = zones.Find(names[i]);
                var t = z != null ? z.GetComponent<NekoNpcTouch>() : null;
                if (t == null) continue;
                Wire(t, tel, per, parts[i]); MarkDirty(t); touches++;
            }
        }
        MarkDirty(tel); MarkDirty(per); MarkDirty(loco); MarkDirty(router); if (sync != null) MarkDirty(sync); if (np != null) MarkDirty(np);

        // Animator（若模型已就位）
        var anim = root.GetComponentInChildren<Animator>();
        if (anim != null) { loco.animator = anim; if (sync != null) sync.animator = anim; MarkDirty(loco); if (sync != null) MarkDirty(sync); }

        // 同场景的 NekoEyeCamDolly（N4 实验）：telemetry / eyeAnchor 一并校正（反射，Experimental 可能不在）
        int dollyFixed = 0;
        var dollyType = FindType("NekoEyeCamDolly");
        if (dollyType != null)
        {
            foreach (var d in UnityEngine.Object.FindObjectsOfType(dollyType))
            {
                var fTel = dollyType.GetField("telemetry"); var fEye = dollyType.GetField("eyeAnchor");
                if (fTel != null) fTel.SetValue(d, tel);
                if (fEye != null && eye != null) fEye.SetValue(d, eye);
                MarkDirty(d); dollyFixed++;
            }
        }

        // 兜底 B 的 NekoEyeCam（若已 Build Eye Cam）：telemetry / eyeCamera / hudQuad
        var eyeCam = scripts.GetComponent<NekoEyeCam>();
        if (eyeCam != null)
        {
            eyeCam.telemetry = tel;
            if (eye != null)
            {
                var camT = eye.Find("EyeCamera");
                if (camT != null)
                {
                    var cam = camT.GetComponent<Camera>();
                    eyeCam.eyeCamera = cam;
                    // 保留本地与远端玩家，只排除 HUD 所在 UI 层和镜面反射专用层。
                    if (cam != null) cam.cullingMask = ~((1 << 5) | (1 << 18));
                }
            }
            var hud = root.transform.Find("EyeCamHud");
            if (hud != null)
            {
                eyeCam.hudQuad = hud;
                // 用 UI 层隔离 HUD 自身，避免递归画中画，同时不再误伤 PlayerLocal。
                hud.gameObject.layer = 5;
            }
            MarkDirty(eyeCam);
        }

        // VRC Midi Listener 兜底
        bool midiOk = AddMidiListener(scripts.gameObject, router);
        EditorUtility.SetDirty(root);
        Debug.Log("[NEKO] 重连完成：" + root.name + "  触摸区 " + touches + " 个，EyeCamDolly " + dollyFixed + " 个，Midi Listener " + (midiOk ? "OK" : "需手动检查")
                  + "。请在 Inspector 确认 Scripts 上四个脚本的引用不再是 None，然后 Ctrl+S 存场景。");
    }

    [MenuItem("NEKO/Configure Matchbox NavMesh (YUI)")]
    public static void ConfigureMatchboxNavMesh()
    {
        const string scenePath = "Assets/Scenes/VRCDefaultWorldScene.unity";
        if (AssetDatabase.LoadAssetAtPath<SceneAsset>(scenePath) == null)
            throw new InvalidOperationException("找不到火柴盒场景：" + scenePath);

        Scene scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
        GameObject root = GameObject.Find(RootName);
        if (root == null) throw new InvalidOperationException("火柴盒场景缺少 " + RootName);

        Transform scripts = root.transform.Find("Scripts");
        if (scripts == null) throw new InvalidOperationException(RootName + " 缺少 Scripts 子物体");
        NekoNpcLocomotion loco = scripts.GetComponent<NekoNpcLocomotion>();
        NekoMidiRouter router = scripts.GetComponent<NekoMidiRouter>();
        if (loco == null || router == null) throw new InvalidOperationException("火柴盒场景缺少 YUI Locomotion/Router");

        NavMeshAgent agent = root.GetComponent<NavMeshAgent>();
        if (agent == null) agent = root.AddComponent<NavMeshAgent>();
        ConfigureAgent(agent);
        loco.navAgent = agent;
        loco.npcRoot = root.transform;
        loco.stopDistance = 0.3f;
        loco.wanderSwitchDistance = 0.9f;
        router.locomotion = loco;
        router.enableGoto = true;
        router.worldName = "NekoN4Lab Matchbox";

        GameObject arena = GameObject.Find("NekoNavMesh_TestArena");
        if (arena == null) arena = new GameObject("NekoNavMesh_TestArena");
        arena.transform.position = Vector3.zero;
        NavMeshSurface surface = arena.GetComponent<NavMeshSurface>();
        if (surface == null) surface = arena.AddComponent<NavMeshSurface>();
        surface.agentTypeID = 0;
        surface.collectObjects = CollectObjects.Volume;
        surface.useGeometry = NavMeshCollectGeometry.PhysicsColliders;
        surface.layerMask = ~0;
        surface.center = new Vector3(0f, 1f, 0f);
        surface.size = new Vector3(16f, 4f, 16f);

        // 固定障碍用于验证 GOTO 走 NavMesh 路径，而不是退回直线改 Transform。
        GameObject obstacle = GameObject.Find("NekoNav_TestObstacle");
        if (obstacle == null)
        {
            obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacle.name = "NekoNav_TestObstacle";
        }
        obstacle.transform.position = new Vector3(1.5f, 0.5f, 0f);
        obstacle.transform.rotation = Quaternion.identity;
        obstacle.transform.localScale = new Vector3(0.6f, 1f, 2f);
        surface.BuildNavMesh();
        EditorUtility.SetDirty(agent);
        EditorUtility.SetDirty(loco);
        EditorUtility.SetDirty(router);
        EditorUtility.SetDirty(surface);
        EditorUtility.SetDirty(arena);
        EditorUtility.SetDirty(obstacle);
        EditorSceneManager.MarkSceneDirty(scene);
        if (!EditorSceneManager.SaveScene(scene))
            throw new InvalidOperationException("保存火柴盒场景失败：" + scenePath);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        Debug.Log("[NEKO] 火柴盒 NavMesh 配置完成：YUI NavMeshAgent + 16m 测试面 + 固定绕行障碍。");
    }

    static void ConfigureAgent(NavMeshAgent agent)
    {
        agent.agentTypeID = 0;
        agent.radius = 0.25f;
        agent.height = 1.6f;
        agent.baseOffset = 0f;
        agent.speed = 2f;
        agent.acceleration = 4f;
        agent.angularSpeed = 180f;
        agent.stoppingDistance = 0.3f;
        agent.autoBraking = true;
        agent.updatePosition = true;
        agent.updateRotation = true;
        agent.enabled = false;
    }

    static GameObject Sphere(GameObject parent, string name, Vector3 localPos, float radius)
    {
        var g = new GameObject(name);
        g.transform.SetParent(parent.transform, false);
        g.transform.localPosition = localPos;
        var c = g.AddComponent<SphereCollider>();
        c.isTrigger = true;
        c.radius = radius;
        return g;
    }

    static TextMeshPro Text(GameObject parent, string name, Vector3 localPos, string text, float size)
    {
        var g = new GameObject(name);
        g.transform.SetParent(parent.transform, false);
        g.transform.localPosition = localPos;
        // 不做 180° 翻转：TMP 3D 文字可读面在 -Z 侧，NekoNpcNameplate.Update 用 LookRotation(自身-观众) 让 +Z 背向观众，正好可读
        // 用 TextMeshPro 而不是旧 TextMesh：TextMesh.text 未暴露给 Udon（U# 编译报错）
        var tm = g.AddComponent<TextMeshPro>();
        tm.text = text;
        tm.alignment = TextAlignmentOptions.Center;
        tm.enableWordWrapping = false;
        tm.fontSize = size;           // TMP 3D：fontSize ≈ 字高(米)×10
        tm.color = Color.white;
        return tm;
    }

    static void Wire(NekoNpcTouch t, NekoNpcTelemetry tel, NekoNpcPerception per, string part)
    {
        t.telemetry = tel; t.perception = per; t.part = part;
    }

    // 添加 VRC.SDK3.Midi.VRCMidiListener 并把 Behaviour 指向 router 的 UdonBehaviour、Active Events 全开。
    // 全部走反射 + SerializedObject，避免在不同 SDK 版本上因字段名差异编译失败。
    static bool AddMidiListener(GameObject go, Component router)
    {
        try
        {
            var listenerType = FindType("VRC.SDK3.Midi.VRCMidiListener");
            if (listenerType == null) { Debug.LogWarning("[NEKO] 找不到 VRC.SDK3.Midi.VRCMidiListener（SDK 未装或版本过旧）"); return false; }
            var listener = go.GetComponent(listenerType);
            if (listener == null) listener = Undo.AddComponent(go, listenerType);
            if (listener == null) return false;

            // 目标 UdonBehaviour：U# 1.x 的 proxy → backing UdonBehaviour
            UnityEngine.Object target = router;
            var util = FindType("UdonSharpEditor.UdonSharpEditorUtility");
            if (util != null)
            {
                var m = util.GetMethod("GetBackingUdonBehaviour", BindingFlags.Public | BindingFlags.Static);
                if (m != null)
                {
                    var backing = m.Invoke(null, new object[] { router }) as UnityEngine.Object;
                    if (backing != null) target = backing;
                }
            }

            var so = new SerializedObject(listener);
            var it = so.GetIterator();
            bool setBehaviour = false, setEvents = false;
            while (it.NextVisible(true))
            {
                string n = it.name.ToLowerInvariant();
                if (it.propertyType == SerializedPropertyType.ObjectReference && n.Contains("behaviour"))
                {
                    it.objectReferenceValue = target; setBehaviour = true;
                }
                else if (n.Contains("event") || n.Contains("active"))
                {
                    if (it.propertyType == SerializedPropertyType.Enum) { it.intValue = -1; setEvents = true; }      // flags 全选
                    else if (it.propertyType == SerializedPropertyType.Integer) { it.intValue = -1; setEvents = true; }
                    else if (it.propertyType == SerializedPropertyType.Boolean) { it.boolValue = true; setEvents = true; }
                }
                else if (it.propertyType == SerializedPropertyType.Boolean && (n.Contains("note") || n.Contains("control")))
                {
                    it.boolValue = true; setEvents = true;
                }
            }
            so.ApplyModifiedProperties();
            if (!setBehaviour) Debug.LogWarning("[NEKO] VRCMidiListener 的 Behaviour 字段未能自动设置，请手动指向 NekoMidiRouter");
            if (!setEvents) Debug.LogWarning("[NEKO] VRCMidiListener 的 Active Events 未能自动勾选，请手动勾 Note On / Note Off / Control Change");
            return setBehaviour;
        }
        catch (Exception e)
        {
            Debug.LogWarning("[NEKO] 添加 VRCMidiListener 失败：" + e.Message);
            return false;
        }
    }

    static void MarkDirty(UnityEngine.Object o)
    {
        if (o == null) return;
        EditorUtility.SetDirty(o);
        // U# 1.x：proxy → UdonBehaviour 的拷贝通常在保存/构建时自动发生；这里尽力主动同步一次
        try
        {
            var util = FindType("UdonSharpEditor.UdonSharpEditorUtility");
            if (util != null)
            {
                var m = util.GetMethod("CopyProxyToUdon", BindingFlags.Public | BindingFlags.Static, null,
                    new Type[] { FindType("UdonSharp.UdonSharpBehaviour") }, null);
                if (m != null) m.Invoke(null, new object[] { o });
            }
        }
        catch (Exception) { /* 非致命 */ }
    }

    // 反射：UdonSharpEditor.UdonSharpUndo.AddComponent<T>(GameObject)
    static T AddUdon<T>(GameObject go) where T : Component
    {
        try
        {
            var undo = FindType("UdonSharpEditor.UdonSharpUndo");
            if (undo != null)
            {
                foreach (var m in undo.GetMethods(BindingFlags.Public | BindingFlags.Static))
                {
                    if (m.Name != "AddComponent" || !m.IsGenericMethodDefinition) continue;
                    var ps = m.GetParameters();
                    if (ps.Length != 1 || ps[0].ParameterType != typeof(GameObject)) continue;
                    var g = m.MakeGenericMethod(typeof(T));
                    var r = g.Invoke(null, new object[] { go }) as T;
                    if (r != null) return r;
                }
            }
            // 退路：直接 AddComponent（U# 1.x 在编辑器里也能识别为 proxy）
            var direct = Undo.AddComponent(go, typeof(T)) as T;
            return direct;
        }
        catch (Exception e)
        {
            Debug.LogWarning("[NEKO] 添加 " + typeof(T).Name + " 失败：" + e.Message);
            return null;
        }
    }

    static Type FindType(string fullName)
    {
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            var t = asm.GetType(fullName, false);
            if (t != null) return t;
        }
        return null;
    }
}
#endif
