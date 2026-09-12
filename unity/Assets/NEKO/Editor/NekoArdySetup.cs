#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UdonSharpEditor;

// 接线只新增独立执行节点；不改模型姿势、Animator Controller 或原导航参数。
public static class NekoArdySetup
{
    private static readonly HumanBodyBones[] Bones = {
        HumanBodyBones.Hips, HumanBodyBones.Spine, HumanBodyBones.Chest,
        HumanBodyBones.Neck, HumanBodyBones.Head,
        HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
        HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
        HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot,
        HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot
    };

    public static void CaptureBindPose(NekoArdyRigLab rig)
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode || rig==null || rig.targetAnimator==null || rig.motionRoot==null
            || rig.targetBones==null || rig.targetBones.Length!=19) throw new InvalidOperationException("绑定姿势提取条件不足");
        var skins=rig.targetAnimator.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        var rotations=new Quaternion[19];
        var positions=new Vector3[19];
        for(int i=0;i<19;i++) {
            bool found=false;
            foreach(var skin in skins) {
                if(skin.sharedMesh==null) continue;
                int index=Array.IndexOf(skin.bones,rig.targetBones[i]);
                if(index<0 || index>=skin.sharedMesh.bindposes.Length) continue;
                Matrix4x4 worldBind=skin.transform.localToWorldMatrix*skin.sharedMesh.bindposes[index].inverse;
                Quaternion rotation=Quaternion.Inverse(rig.motionRoot.rotation)*worldBind.rotation;
                Vector3 position=rig.motionRoot.InverseTransformPoint(worldBind.MultiplyPoint3x4(Vector3.zero));
                if(found && Vector3.Distance(positions[i],position)>.001f) throw new InvalidOperationException("蒙皮绑定位置不一致："+rig.targetBones[i].name);
                positions[i]=position;
                if(found && Quaternion.Angle(rotations[i],rotation)>1f) throw new InvalidOperationException("蒙皮绑定姿势不一致："+rig.targetBones[i].name);
                rotations[i]=rotation; found=true;
            }
            if(!found) throw new InvalidOperationException("骨骼没有蒙皮绑定矩阵："+rig.targetBones[i].name);
        }
        Undo.RecordObject(rig,"固定 ARDY 重定向绑定姿势");
        rig.bindRotations=rotations;
        rig.bindPositions=positions;
        rig.bindPoseConfigured=true;
        EditorUtility.SetDirty(rig);
    }

    public static string Validate(NekoMidiRouter router)
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) return "请先退出运行模式";
        if (router == null || router.locomotion == null || router.telemetry == null) return "缺少 Router、Locomotion 或遥测";
        var loco = router.locomotion;
        if (loco.router != router || loco.telemetry != router.telemetry) return "现有控制器引用不一致";
        if (!router.enableOperationLifecycle) return "动作生命周期未启用";
        if (loco.npcRoot == null || loco.navAgent == null || loco.animator == null) return "缺少根节点、导航或 Animator";
        if (loco.navAgent.transform != loco.npcRoot || !loco.animator.transform.IsChildOf(loco.npcRoot)) return "导航或模型不属于同一根节点";
        if (!loco.animator.isHuman || loco.animator.avatar == null || !loco.animator.avatar.isValid) return "Humanoid Avatar 无效";
        if (Bones.Any(b => loco.animator.GetBoneTransform(b) == null)) return "缺少重定向所需的19个骨骼";
        if (router.poseExecutor != null) return "已有姿态执行器，拒绝重复接管";
        if (loco.npcRoot.Find("ARDYExecution") != null) return "同名执行节点已存在";
        NavMeshHit hit;
        if (!NavMesh.SamplePosition(loco.npcRoot.position, out hit, .3f, loco.navAgent.areaMask)) return "根节点附近没有可用 NavMesh";
        if (Mathf.Abs(hit.position.y-loco.npcRoot.position.y) > .08f) return "根节点与导航地面高度不匹配";
        float scale=(loco.animator.GetBoneTransform(HumanBodyBones.Hips).position.y-loco.npcRoot.position.y)/.9544f;
        if (scale < .2f || scale > 3f) return "角色比例超出已验证范围";
        return null;
    }

    public static string InstallContinuous(NekoArdyWorldBridge bridge,string backupDirectory)
    {
        if(EditorApplication.isPlayingOrWillChangePlaymode || bridge==null || bridge.receiver==null)
            throw new InvalidOperationException("需要编辑模式下已接线的有限动作桥");
        var scene=bridge.gameObject.scene;
        if(!scene.IsValid() || scene.isDirty || string.IsNullOrEmpty(scene.path))
            throw new InvalidOperationException("场景必须已保存且无未保存修改");
        if(bridge.streamBridge!=null || bridge.GetComponent<NekoArdyStreamBridge>()!=null)
            throw new InvalidOperationException("持续动作桥已存在");
        if(AssetDatabase.LoadAssetAtPath<UdonSharp.UdonSharpProgramAsset>("Assets/NEKO/ArdyLab/NekoArdyStreamBridge.asset")==null)
            throw new InvalidOperationException("请先导入持续动作桥的Udon程序资产并编译");
        string backup=Path.GetFullPath(backupDirectory);
        if(Directory.Exists(backup) || File.Exists(backup)) throw new InvalidOperationException("备份目录已存在");
        Directory.CreateDirectory(backup);
        File.Copy(scene.path,Path.Combine(backup,"before.unity"));
        File.WriteAllText(Path.Combine(backup,"source.txt"),Path.GetFullPath(scene.path));
        Undo.IncrementCurrentGroup();
        int group=Undo.GetCurrentGroup();
        try {
            Undo.RecordObjects(new UnityEngine.Object[]{bridge,bridge.receiver},"接线持续动作桥");
            var stream=UdonSharpComponentExtensions.AddUdonSharpComponent<NekoArdyStreamBridge>(bridge.gameObject);
            Undo.RegisterCreatedObjectUndo(stream,"新增持续动作桥");
            stream.world=bridge; stream.enableContinuous=false;
            bridge.streamBridge=stream; bridge.receiver.streamBridge=stream;
            foreach(var item in new UdonSharp.UdonSharpBehaviour[]{bridge,bridge.receiver,stream}) {
                EditorUtility.SetDirty(item); UdonSharpEditorUtility.CopyProxyToUdon(item);
            }
            EditorSceneManager.MarkSceneDirty(scene);
            if(!EditorSceneManager.SaveScene(scene)) throw new IOException("场景保存失败");
            Undo.CollapseUndoOperations(group);
            return "持续动作桥已接线，保持关闭等待验收；备份："+backup;
        } catch { Undo.RevertAllDownToGroup(group); throw; }
    }

    public static string Install(NekoMidiRouter router, string backupDirectory)
    {
        string error=Validate(router);
        if (error != null) throw new InvalidOperationException(error);
        var scene=router.gameObject.scene;
        if (!scene.IsValid() || scene.isDirty || string.IsNullOrEmpty(scene.path)) throw new InvalidOperationException("需要已保存且无未提交场景修改的场景");
        string backup=Path.GetFullPath(backupDirectory);
        if (Directory.Exists(backup) || File.Exists(backup)) throw new InvalidOperationException("备份目标已存在，拒绝覆盖");
        Directory.CreateDirectory(backup);
        File.Copy(scene.path,Path.Combine(backup,"before.unity"));
        File.WriteAllText(Path.Combine(backup,"source.txt"),Path.GetFullPath(scene.path));
        var transforms=scene.GetRootGameObjects().SelectMany(g=>g.GetComponentsInChildren<Transform>(true)).ToArray();
        var positions=transforms.Select(t=>t.localPosition).ToArray();
        var rotations=transforms.Select(t=>t.localRotation).ToArray();
        var scales=transforms.Select(t=>t.localScale).ToArray();
        Undo.IncrementCurrentGroup();
        int group=Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("接入可选 ARDY 执行器");
        try {
            Undo.RecordObject(router,"设置可选动作桥");
            var node=new GameObject("ARDYExecution");
            Undo.RegisterCreatedObjectUndo(node,"创建动作执行节点");
            node.transform.SetParent(router.locomotion.npcRoot,false);
            var receiver=UdonSharpComponentExtensions.AddUdonSharpComponent<NekoArdyPoseLab>(node);
            var rig=UdonSharpComponentExtensions.AddUdonSharpComponent<NekoArdyRigLab>(node);
            var bridge=UdonSharpComponentExtensions.AddUdonSharpComponent<NekoArdyWorldBridge>(node);
            receiver.telemetry=router.telemetry; receiver.authorityRouter=router; receiver.worldBridge=bridge;
            receiver.requireAuthority=true; receiver.allowIsolatedArm=false; receiver.labEnabled=true;
            rig.receiver=receiver; rig.targetAnimator=router.locomotion.animator;
            rig.motionRoot=router.locomotion.npcRoot; rig.navigationWriter=router.locomotion.navAgent;
            rig.targetBones=Bones.Select(b=>rig.targetAnimator.GetBoneTransform(b)).ToArray();
            rig.rootEnabled=true; rig.feetEnabled=true; rig.labApproved=true;
            rig.environmentMask=router.locomotion.environmentMask;
            CaptureBindPose(rig);
            bridge.router=router; bridge.receiver=receiver; bridge.rig=rig;
            router.poseExecutor=bridge;
            router.enablePoseStream=true;
            if (!bridge.Ready()) throw new InvalidOperationException("新建执行桥引用检查失败");
            // 默认关闭，场景接线完成不等于发布姿态能力。
            router.enablePoseStream=false;
            var listener=node.AddComponent<VRC.SDK3.Midi.VRCMidiListener>();
            listener.activeEvents=VRC.SDK3.Midi.VRCMidiListener.MidiEvents.NoteOff|VRC.SDK3.Midi.VRCMidiListener.MidiEvents.CC;
            var serialized=new SerializedObject(listener);
            serialized.FindProperty("behaviour").objectReferenceValue=UdonSharpEditorUtility.GetBackingUdonBehaviour(receiver);
            serialized.ApplyModifiedPropertiesWithoutUndo();
            foreach (var b in node.GetComponents<UdonSharp.UdonSharpBehaviour>()) UdonSharpEditorUtility.CopyProxyToUdon(b);
            UdonSharpEditorUtility.CopyProxyToUdon(router);
            for (int i=0;i<transforms.Length;i++) {
                if (transforms[i].localPosition!=positions[i] || transforms[i].localRotation!=rotations[i] || transforms[i].localScale!=scales[i])
                    throw new InvalidOperationException("原有对象姿势变化，撤销接线");
            }
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene)) throw new IOException("场景保存失败，原件保留在备份目录");
            Undo.CollapseUndoOperations(group);
            return "已接线，默认关闭；原有对象姿势保持一致；备份："+backup;
        } catch {
            Undo.RevertAllDownToGroup(group);
            throw;
        }
    }
}
#endif
