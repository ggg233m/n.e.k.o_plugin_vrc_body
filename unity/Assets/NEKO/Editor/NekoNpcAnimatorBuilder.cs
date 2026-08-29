// NekoNpcAnimatorBuilder —— 按 YUI NPC v1.1 §10 生成四层 Animator Controller。
#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

public static class NekoNpcAnimatorBuilder
{
    public const string Dir = "Assets/NEKO/Animations";
    public const string ControllerPath = Dir + "/NekoNpc_YUI.controller";
    const string UpperMaskPath = Dir + "/YuiUpperBody.mask";
    const string ChiffonStandPath = "Assets/ChiffonLite/Animation/Locomotion/Locamotion/Chiffon_Stand.anim";
    const string WalkPath = Dir + "/YuiNpc_Walk.anim";
    const string RunPath = Dir + "/YuiNpc_Run.anim";

    // 核心语义保持冻结；这里仅决定 Chiffon 动画素材如何实现这些语义。
    static readonly string[] CoreClipKeys = {
        "Emote_Neko", "Pose_Stand_1", "Naughty", "StandClap", "Pose_Stand_2", "Happy1", "Pose_Sit_1", "Jump",
        "Pose_Sit_2", "Pose_Squat_1", "Pose_Down_1", "Guruguru", "Pose_Float_1", "Pose_Squat_1", "Happy2", "SitClap"
    };

    static readonly Dictionary<string, string[]> Candidates = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
    {
        { "idle", new[] { "Chiffon_Stand", "proxy_stand_still", "stand_still", "idle" } },
        { "walk", new[] { "YuiNpc_Walk", "proxy_walk_forward", "walk_forward", "walk" } },
        { "run", new[] { "YuiNpc_Run", "proxy_run_forward", "run_forward", "run" } },
        { "Emote_Neko", new[] { "Emote_Neko", "neko" } },
        { "Happy1", new[] { "Happy1" } }, { "Happy2", new[] { "Happy2" } },
        { "StandClap", new[] { "StandClap" } }, { "SitClap", new[] { "SitClap" } },
        { "StandDance", new[] { "StandDance", "dance" } }, { "Naughty", new[] { "Naughty" } },
        { "Guruguru", new[] { "Guruguru" } }, { "Jump", new[] { "Jump" } },
        { "Pose_Stand_1", new[] { "Pose_Stand_1" } }, { "Pose_Stand_2", new[] { "Pose_Stand_2" } },
        { "Pose_Sit_1", new[] { "Pose_Sit_1" } }, { "Pose_Sit_2", new[] { "Pose_Sit_2" } },
        { "Pose_Squat_1", new[] { "Pose_Squat_1" } }, { "Pose_Down_1", new[] { "Pose_Down_1" } },
        { "Pose_Float_1", new[] { "Pose_Float_1" } },
        { "cat_pose", new[] { "Emote_Neko" } }, { "happy_one", new[] { "Happy1" } }, { "happy_two", new[] { "Happy2" } },
        { "stand_clap", new[] { "StandClap" } }, { "sit_clap", new[] { "SitClap" } }, { "dance", new[] { "StandDance" } },
        // 字典忽略大小写，naughty/Naughty 与 jump/Jump 直接复用上方核心候选，避免重复键。
        { "spin", new[] { "Guruguru" } },
        { "pose_stand_one", new[] { "Pose_Stand_1" } }, { "pose_stand_two", new[] { "Pose_Stand_2" } },
        { "pose_sit_one", new[] { "Pose_Sit_1" } }, { "pose_sit_two", new[] { "Pose_Sit_2" } },
        { "pose_squat", new[] { "Pose_Squat_1" } }, { "pose_down", new[] { "Pose_Down_1" } }, { "pose_float", new[] { "Pose_Float_1" } },
        { "neutral", new[] { "Exp_Idle", "neutral" } }, { "happy", new[] { "Exp_Happy", "happy" } },
        { "sad", new[] { "Exp_Cry", "sad" } }, { "surprised", new[] { "Exp_Wink", "surprised" } },
        { "huff", new[] { "Exp_Huff" } }, { "cry", new[] { "Exp_Cry" } }, { "wink", new[] { "Exp_Wink" } },
    };

    [MenuItem("NEKO/YUI NPC/Build v1.1 Animator")]
    public static void Build() { BuildController(true); }

    public static AnimatorController BuildController(bool promptBeforeOverwrite)
    {
        EnsureFolder("Assets/NEKO"); EnsureFolder(Dir);
        EnsureGeneratedLocomotionClip(WalkPath, "YuiNpc_Walk", 0.82f, 0.48f, 0.32f);
        EnsureGeneratedLocomotionClip(RunPath, "YuiNpc_Run", 0.52f, 0.68f, 0.48f);
        var existing = AssetDatabase.LoadAssetAtPath<AnimatorController>(ControllerPath);
        if (existing != null && promptBeforeOverwrite && !EditorUtility.DisplayDialog("NEKO", ControllerPath + " 已存在，覆盖重建？", "重建", "取消")) return existing;
        if (existing != null) AssetDatabase.DeleteAsset(ControllerPath);

        var router = UnityEngine.Object.FindObjectOfType<NekoMidiRouter>();
        string[] names = router != null && router.actionNames != null ? router.actionNames : new string[16];
        string[] semantics = router != null && router.actionSemanticKeys != null ? router.actionSemanticKeys : new string[16];
        var missing = new List<string>();
        var clips = FindAllClips();

        var ctrl = AnimatorController.CreateAnimatorControllerAtPath(ControllerPath);
        ctrl.AddParameter("Speed", AnimatorControllerParameterType.Float);
        ctrl.AddParameter("ActionId", AnimatorControllerParameterType.Int);
        ctrl.AddParameter("ActionSeq", AnimatorControllerParameterType.Int);
        ctrl.AddParameter("ActionLoop", AnimatorControllerParameterType.Bool);
        ctrl.AddParameter("ExpressionId", AnimatorControllerParameterType.Int);
        ctrl.AddParameter("ExpressionWeight", AnimatorControllerParameterType.Float);
        ctrl.AddParameter("Estop", AnimatorControllerParameterType.Bool);

        BuildLocomotion(ctrl, clips, missing);
        var upper = AddLayer(ctrl, "UpperBody", CreateUpperMask(), 1f);
        var full = AddLayer(ctrl, "FullBody", null, 1f);
        var face = AddLayer(ctrl, "FaceAndLook", null, 1f);

        var durations = new int[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            string key = i < CoreClipKeys.Length ? CoreClipKeys[i] : names[i];
            AnimationClip clip = FindClip(key, clips, missing);
            bool fullBody = router != null && router.actionLayers != null && i < router.actionLayers.Length && router.actionLayers[i] == "full_body";
            BuildActionState(fullBody ? full : upper, i, clip);
            durations[i] = clip != null ? Mathf.Max(1, Mathf.RoundToInt(clip.length * 1000f)) : (router != null && router.actionDurationMs != null && i < router.actionDurationMs.Length ? router.actionDurationMs[i] : 1800);
        }
        BuildExpressions(face, router, clips, missing);

        if (router != null)
        {
            Undo.RecordObject(router, "同步 YUI 动作时长");
            bool changed = router.actionDurationMs == null || router.actionDurationMs.Length != durations.Length;
            if (!changed) for (int i = 0; i < durations.Length; i++) if (Mathf.Abs(router.actionDurationMs[i] - durations[i]) > 20) { changed = true; break; }
            router.actionDurationMs = durations;
            router.animNames = names;
            if (changed) router.catalogRevision = Mathf.Max(1, router.catalogRevision + 1);
            EditorUtility.SetDirty(router);
        }
        var loco = UnityEngine.Object.FindObjectOfType<NekoNpcLocomotion>();
        if (loco != null)
        {
            Undo.RecordObject(loco, "同步 YUI 动作时长");
            loco.animDurations = Array.ConvertAll(durations, ms => ms / 1000f);
            EditorUtility.SetDirty(loco);
        }

        AssetDatabase.SaveAssets(); AssetDatabase.Refresh();
        Debug.Log("[NEKO] 已生成 YUI v1.1 四层 Animator：" + ControllerPath + (missing.Count == 0 ? "" : "\n缺少片段：" + string.Join(", ", missing.ToArray())));
        return ctrl;
    }

    static void EnsureGeneratedLocomotionClip(string path, string clipName, float period, float legAmplitude, float armAmplitude)
    {
        // ChiffonLite 不随包分发 walk/run；基于其 Humanoid 站姿生成最小循环步态，避免 NPC 导航时滑行。
        var previous = AssetDatabase.LoadAssetAtPath<AnimationClip>(path);
        if (previous != null) AssetDatabase.DeleteAsset(path);
        var stand = AssetDatabase.LoadAssetAtPath<AnimationClip>(ChiffonStandPath);
        var clip = stand != null ? UnityEngine.Object.Instantiate(stand) : new AnimationClip();
        clip.name = clipName; clip.frameRate = 60f;
        SetSwing(clip, "Left Upper Leg Front-Back", period, legAmplitude);
        SetSwing(clip, "Right Upper Leg Front-Back", period, -legAmplitude);
        SetSwing(clip, "Left Arm Front-Back", period, -armAmplitude);
        SetSwing(clip, "Right Arm Front-Back", period, armAmplitude);
        SetKnee(clip, "Left Lower Leg Stretch", period, false);
        SetKnee(clip, "Right Lower Leg Stretch", period, true);
        var settings = AnimationUtility.GetAnimationClipSettings(clip);
        settings.loopTime = true; settings.loopBlend = true; settings.loopBlendOrientation = true;
        settings.loopBlendPositionY = true; settings.loopBlendPositionXZ = true;
        AnimationUtility.SetAnimationClipSettings(clip, settings);
        AssetDatabase.CreateAsset(clip, path); EditorUtility.SetDirty(clip);
    }

    static void SetSwing(AnimationClip clip, string muscle, float period, float amplitude)
    {
        var curve = new AnimationCurve(new Keyframe(0f, amplitude), new Keyframe(period * 0.5f, -amplitude), new Keyframe(period, amplitude));
        AnimationUtility.SetEditorCurve(clip, EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), muscle), curve);
    }

    static void SetKnee(AnimationClip clip, string muscle, float period, bool opposite)
    {
        float low = 0.12f, high = 0.48f;
        var curve = new AnimationCurve(
            new Keyframe(0f, opposite ? high : low), new Keyframe(period * 0.25f, opposite ? low : high),
            new Keyframe(period * 0.5f, opposite ? high : low), new Keyframe(period * 0.75f, opposite ? low : high),
            new Keyframe(period, opposite ? high : low));
        AnimationUtility.SetEditorCurve(clip, EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), muscle), curve);
    }

    static void BuildLocomotion(AnimatorController ctrl, List<AnimationClip> clips, List<string> missing)
    {
        var sm = ctrl.layers[0].stateMachine; sm.name = "Base Locomotion";
        BlendTree tree; var state = ctrl.CreateBlendTreeInController("Locomotion", out tree, 0);
        tree.blendType = BlendTreeType.Simple1D; tree.blendParameter = "Speed"; tree.useAutomaticThresholds = false;
        var idle = FindClip("idle", clips, missing); var walk = FindClip("walk", clips, missing); var run = FindClip("run", clips, missing);
        if (idle != null) tree.AddChild(idle, 0f); if (walk != null) tree.AddChild(walk, 1.2f); if (run != null) tree.AddChild(run, 2f);
        sm.defaultState = state; state.writeDefaultValues = true;
        var estop = sm.AddState("Estop"); estop.motion = idle; estop.speed = 0f; estop.writeDefaultValues = true;
        var to = sm.AddAnyStateTransition(estop); to.AddCondition(AnimatorConditionMode.If, 0f, "Estop"); to.hasExitTime = false; to.duration = 0f; to.canTransitionToSelf = false;
        var from = estop.AddTransition(state); from.AddCondition(AnimatorConditionMode.IfNot, 0f, "Estop"); from.hasExitTime = false; from.duration = 0.15f;
    }

    static AnimatorControllerLayer AddLayer(AnimatorController ctrl, string name, AvatarMask mask, float weight)
    {
        var layer = new AnimatorControllerLayer { name = name, defaultWeight = weight, avatarMask = mask, blendingMode = AnimatorLayerBlendingMode.Override, stateMachine = new AnimatorStateMachine { name = name } };
        AssetDatabase.AddObjectToAsset(layer.stateMachine, ctrl); ctrl.AddLayer(layer);
        var actual = ctrl.layers[ctrl.layers.Length - 1];
        var empty = actual.stateMachine.AddState("Empty"); empty.writeDefaultValues = true; actual.stateMachine.defaultState = empty;
        return actual;
    }

    static void BuildActionState(AnimatorControllerLayer layer, int id, AnimationClip clip)
    {
        var sm = layer.stateMachine; var state = sm.AddState("Action_" + id); state.motion = clip; state.writeDefaultValues = true;
        var enter = sm.AddAnyStateTransition(state); enter.AddCondition(AnimatorConditionMode.Equals, id, "ActionId"); enter.AddCondition(AnimatorConditionMode.IfNot, 0f, "Estop"); enter.hasExitTime = false; enter.duration = 0.12f; enter.canTransitionToSelf = false;
        var exit = state.AddTransition(sm.defaultState); exit.AddCondition(AnimatorConditionMode.NotEqual, id, "ActionId"); exit.hasExitTime = false; exit.duration = 0.18f;
    }

    static void BuildExpressions(AnimatorControllerLayer layer, NekoMidiRouter router, List<AnimationClip> clips, List<string> missing)
    {
        if (router == null || router.expressionNames == null) return;
        for (int i = 0; i < router.expressionNames.Length; i++)
        {
            // idle 是宿主语义别名；面部实现复用冻结核心 neutral 的 Exp_Idle，
            // 不能走 locomotion 的同名 Chiffon_Stand。
            string semanticKey = router.expressionSemanticKeys[i];
            var clip = FindClip(semanticKey == "idle" ? "neutral" : semanticKey, clips, missing);
            var state = layer.stateMachine.AddState("Expression_" + i); state.motion = clip; state.writeDefaultValues = true;
            var enter = layer.stateMachine.AddAnyStateTransition(state); enter.AddCondition(AnimatorConditionMode.Equals, i, "ExpressionId"); enter.AddCondition(AnimatorConditionMode.Greater, 0.001f, "ExpressionWeight"); enter.hasExitTime = false; enter.duration = router.expressionFadeMs[i] / 1000f; enter.canTransitionToSelf = false;
            var exit = state.AddTransition(layer.stateMachine.defaultState); exit.AddCondition(AnimatorConditionMode.Less, 0.001f, "ExpressionWeight"); exit.hasExitTime = false; exit.duration = router.expressionFadeMs[i] / 1000f;
        }
    }

    static AvatarMask CreateUpperMask()
    {
        var old = AssetDatabase.LoadAssetAtPath<AvatarMask>(UpperMaskPath); if (old != null) AssetDatabase.DeleteAsset(UpperMaskPath);
        var mask = new AvatarMask();
        for (int i = 0; i < (int)AvatarMaskBodyPart.LastBodyPart; i++) mask.SetHumanoidBodyPartActive((AvatarMaskBodyPart)i, false);
        mask.SetHumanoidBodyPartActive(AvatarMaskBodyPart.Body, true);
        mask.SetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftArm, true); mask.SetHumanoidBodyPartActive(AvatarMaskBodyPart.RightArm, true);
        mask.SetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftFingers, true); mask.SetHumanoidBodyPartActive(AvatarMaskBodyPart.RightFingers, true);
        mask.SetHumanoidBodyPartActive(AvatarMaskBodyPart.Head, false); mask.SetHumanoidBodyPartActive(AvatarMaskBodyPart.Root, false);
        AssetDatabase.CreateAsset(mask, UpperMaskPath); return mask;
    }

    static List<AnimationClip> FindAllClips()
    {
        var result = new List<AnimationClip>();
        foreach (string guid in AssetDatabase.FindAssets("t:AnimationClip"))
        {
            var clip = AssetDatabase.LoadAssetAtPath<AnimationClip>(AssetDatabase.GUIDToAssetPath(guid)); if (clip != null) result.Add(clip);
        }
        return result;
    }

    static AnimationClip FindClip(string key, List<AnimationClip> clips, List<string> missing)
    {
        string[] candidates; if (!Candidates.TryGetValue(key, out candidates)) candidates = new[] { key };
        foreach (string candidate in candidates) foreach (var clip in clips) if (string.Equals(clip.name, candidate, StringComparison.OrdinalIgnoreCase)) return clip;
        foreach (string candidate in candidates) foreach (var clip in clips) if (clip.name.IndexOf(candidate, StringComparison.OrdinalIgnoreCase) >= 0 && clip.name.IndexOf("proxy_hands", StringComparison.OrdinalIgnoreCase) < 0) return clip;
        if (!missing.Contains(key)) missing.Add(key); return null;
    }

    static void EnsureFolder(string path)
    {
        if (AssetDatabase.IsValidFolder(path)) return;
        string parent = System.IO.Path.GetDirectoryName(path).Replace('\\', '/'); string name = System.IO.Path.GetFileName(path);
        if (!AssetDatabase.IsValidFolder(parent)) EnsureFolder(parent); AssetDatabase.CreateFolder(parent, name);
    }
}
#endif
