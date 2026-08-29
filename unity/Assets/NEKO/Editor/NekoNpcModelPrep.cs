// NekoNpcModelPrep —— 把 avatar 工程导出的 humanoid（mizuki）整理成世界 NPC 模型（纯 Editor 脚本）
//
// 菜单：NEKO → NPC Model: Prepare Selected Avatar
// 前置：场景里已有 NekoNpc_YUI（先跑 NEKO → Build NPC Rig），并已把 mizuki 预制体拖进场景（任意位置）。
//
// 做的事（全部可撤销 Undo）：
//   1. 递归删除 "Missing (Mono Script)" 组件——avatar 专属脚本（VRCAvatarDescriptor / PipelineManager /
//      Modular Avatar / VRCFury / lilToon 编辑器组件…）在世界工程里没有对应类型，留着会让 SDK 打包报错；
//   2. 按类型名删除世界里不该有的组件（VRCAvatarDescriptor、PipelineManager、VRCContactReceiver 等，见 stripTypeNames）；
//   3. 把模型挂到 NekoNpc_YUI/Model 下、归零、面朝 +Z，删除 Placeholder_*；
//   4. 用 humanoid 骨骼自动对位：EyeAnchor 挂到 Head 骨（眼位 = 头骨 + 前 0.16m，须在脸部网格之外）；TouchZones 的 Head/Cheek/HandL/HandR/Torso
//      分别贴到 Head/Head 前方/LeftHand/RightHand/Chest 骨骼位置（不改父级，只改位置，保持 Rig 结构不变）；
//   5. Animator：Apply Root Motion 关；若找到 Assets/NEKO/Animations/NekoNpc_YUI.controller 就赋上；
//      把 Animator 引用填给 NekoNpcLocomotion 与 NekoNpcSync；
//   6. 检查 Model 子树有没有实体碰撞体（须式射线会打到自己）——有就警告并列出。
//
// 不做的事：不改材质/shader（lilToon 需随 unitypackage 一起进世界工程）；不动 PhysBone（世界 SDK 支持，保留头发/裙摆物理）。

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public static class NekoNpcModelPrep
{
    const string RootName = "NekoNpc_YUI";
    const string ControllerPath = "Assets/NEKO/Animations/NekoNpc_YUI.controller";

    // 按"类型全名或短名"匹配删除（存在于世界 SDK 里但 NPC 不该有的；缺失脚本由 RemoveMissing 统一处理）
    static readonly string[] stripTypeNames =
    {
        "VRCAvatarDescriptor", "PipelineManager", "VRCContactReceiver", "VRCStation",
        "ModularAvatar", "VRCFury", "NDMF", "LilToon", "lilToon",
    };

    [MenuItem("NEKO/NPC Model: Prepare Selected Avatar")]
    public static void Prepare()
    {
        var avatar = Selection.activeGameObject;
        if (avatar == null) { EditorUtility.DisplayDialog("NEKO", "先在 Hierarchy 里选中拖进场景的 mizuki 预制体实例。", "好"); return; }
        var root = GameObject.Find(RootName);
        if (root == null) { EditorUtility.DisplayDialog("NEKO", "场景里没有 " + RootName + "，先跑 NEKO → Build NPC Rig (YUI)。", "好"); return; }
        if (avatar.transform.IsChildOf(root.transform) && avatar.name == "Model")
        { EditorUtility.DisplayDialog("NEKO", "请选中模型本体（mizuki），不是 Model 容器。", "好"); return; }

        var animator = avatar.GetComponent<Animator>();
        if (animator == null || animator.avatar == null || !animator.avatar.isHuman)
        {
            if (!EditorUtility.DisplayDialog("NEKO", "选中物体没有 humanoid Animator，骨骼自动对位会跳过。继续？", "继续", "取消")) return;
        }

        Undo.SetCurrentGroupName("NEKO Prepare NPC Model");
        int group = Undo.GetCurrentGroup();

        // 1) 缺失脚本
        int missing = 0;
        foreach (var t in avatar.GetComponentsInChildren<Transform>(true))
        {
            int n = GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(t.gameObject);
            if (n > 0) { Undo.RegisterCompleteObjectUndo(t.gameObject, "remove missing"); missing += GameObjectUtility.RemoveMonoBehavioursWithMissingScript(t.gameObject); }
        }

        // 2) 按类型名删除
        int stripped = 0;
        foreach (var c in avatar.GetComponentsInChildren<Component>(true))
        {
            if (c == null) continue;
            string tn = c.GetType().Name;
            string fn = c.GetType().FullName ?? tn;
            foreach (var s in stripTypeNames)
            {
                if (tn.IndexOf(s, StringComparison.OrdinalIgnoreCase) >= 0 || fn.IndexOf(s, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    Undo.DestroyObjectImmediate(c); stripped++; break;
                }
            }
        }

        // 3) 归位到 Model 下
        var model = root.transform.Find("Model");
        if (model == null)
        {
            var m = new GameObject("Model"); Undo.RegisterCreatedObjectUndo(m, "Model");
            m.transform.SetParent(root.transform, false); model = m.transform;
        }
        foreach (Transform child in model)
        {
            if (child.name.StartsWith("Placeholder_")) Undo.DestroyObjectImmediate(child.gameObject);
        }
        Undo.SetTransformParent(avatar.transform, model, "parent to Model");
        avatar.transform.localPosition = Vector3.zero;
        avatar.transform.localRotation = Quaternion.identity;
        avatar.transform.localScale = Vector3.one;

        // 4) 骨骼对位
        string boneNote = "跳过（无 humanoid）";
        if (animator != null && animator.avatar != null && animator.avatar.isHuman)
        {
            Transform head = animator.GetBoneTransform(HumanBodyBones.Head);
            Transform lh = animator.GetBoneTransform(HumanBodyBones.LeftHand);
            Transform rh = animator.GetBoneTransform(HumanBodyBones.RightHand);
            Transform chest = animator.GetBoneTransform(HumanBodyBones.Chest);
            if (chest == null) chest = animator.GetBoneTransform(HumanBodyBones.Spine);
            Vector3 fwd = root.transform.forward;

            var eye = root.transform.Find("EyeAnchor");
            if (eye != null && head != null)
            {
                Undo.SetTransformParent(eye, head, "EyeAnchor→Head");
                // 眼位要落在脸部网格之外：头骨中心到脸表面通常 0.10–0.14m（mizuki 这类头身比更大），取 0.16 再留 near clip 余量。
                // 眼位相机 nearClip=0.05，若仍看到脸/刘海，把 EyeAnchor 再往前挪或把头发层排除
                eye.position = head.position + fwd * 0.16f + Vector3.up * 0.04f;
                eye.rotation = root.transform.rotation;
            }
            var zones = root.transform.Find("TouchZones");
            if (zones != null)
            {
                Place(zones, "Head", head != null ? head.position + Vector3.up * 0.06f : (Vector3?)null);
                Place(zones, "Cheek", head != null ? head.position + fwd * 0.10f - Vector3.up * 0.02f : (Vector3?)null);
                Place(zones, "HandL", lh != null ? lh.position : (Vector3?)null);
                Place(zones, "HandR", rh != null ? rh.position : (Vector3?)null);
                Place(zones, "Torso", chest != null ? chest.position : (Vector3?)null);
            }
            boneNote = "已按 Head/LeftHand/RightHand/Chest 对位（EyeAnchor 已挂到 Head 骨）";
        }

        // 5) Animator
        if (animator != null)
        {
            Undo.RecordObject(animator, "animator");
            animator.applyRootMotion = false;
            animator.cullingMode = AnimatorCullingMode.CullUpdateTransforms;
            var ctrl = AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>(ControllerPath);
            if (ctrl != null) animator.runtimeAnimatorController = ctrl;
            var loco = root.GetComponentInChildren<NekoNpcLocomotion>(true);
            if (loco != null) { Undo.RecordObject(loco, "loco"); loco.animator = animator; EditorUtility.SetDirty(loco); }
            var sync = root.GetComponent<NekoNpcSync>();
            if (sync != null) { Undo.RecordObject(sync, "sync"); sync.animator = animator; EditorUtility.SetDirty(sync); }
            if (ctrl == null) Debug.LogWarning("[NEKO] 未找到 " + ControllerPath + "，先跑 NEKO → NPC Animator: Build Controller，再重跑本菜单或手动赋 controller。");
        }

        // 6) 自碰撞检查
        var solid = new List<string>();
        foreach (var col in avatar.GetComponentsInChildren<Collider>(true))
            if (col != null && !col.isTrigger) solid.Add(col.gameObject.name + " (" + col.GetType().Name + ")");

        Undo.CollapseUndoOperations(group);
        Selection.activeGameObject = root;

        string msg = "[NEKO] NPC 模型整理完成：删缺失脚本 " + missing + " 个，按类型删 " + stripped + " 个；骨骼对位：" + boneNote + "。";
        if (solid.Count > 0)
            msg += "\n!! Model 子树有 " + solid.Count + " 个实体碰撞体（须式射线会打到自己，请删除或改 isTrigger）：\n  - " + string.Join("\n  - ", solid.ToArray());
        Debug.Log(msg);
    }

    static void Place(Transform zones, string name, Vector3? worldPos)
    {
        if (!worldPos.HasValue) return;
        var z = zones.Find(name);
        if (z == null) return;
        Undo.RecordObject(z, "place " + name);
        z.position = worldPos.Value;
    }
}
#endif
