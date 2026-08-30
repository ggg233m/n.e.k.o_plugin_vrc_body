// NekoEyeCamBuilder —— 生成 N4 兜底 B 的"眼位相机 + HUD 小窗"（纯 Editor 脚本）
//
// 菜单：NEKO → Build Eye Cam (fallback)
// 前置：场景里已有 NekoNpc_YUI（EyeAnchor 存在；若已跑过 Prepare Selected Avatar，EyeAnchor 在 Head 骨下）。
// 生成：
//   EyeAnchor/EyeCamera         Camera（60° FOV，near 0.05，far 60，targetTexture = Assets/NEKO/Animations/EyeCamRT.renderTexture 640×360）
//   NekoNpc_YUI/EyeCamHud       Quad（Unlit/Texture 材质贴 RT；由 NekoEyeCam 每帧贴到 driver 头前右下角；仅 driver 可见）
//   NekoNpc_YUI/Scripts + NekoEyeCam（U#，引用已连好；Interact 可开关）
// 幂等：重复运行会复用已存在的物体/资产。

#if UNITY_EDITOR
using System;
using System.Reflection;
using UnityEditor;
using UnityEngine;

public static class NekoEyeCamBuilder
{
    public const int PlayerRenderLayersMask = (1 << 9) | (1 << 10) | (1 << 18);
    public const int EyeCamExcludedMask = (1 << 5);

    const string RootName = "NekoNpc_YUI";
    const string Dir = "Assets/NEKO/Animations";
    const string RtPath = Dir + "/EyeCamRT.renderTexture";
    const string MatPath = Dir + "/EyeCamHud.mat";

    [MenuItem("NEKO/Build Eye Cam (fallback)")]
    public static void Build()
    {
        var root = GameObject.Find(RootName);
        if (root == null) { EditorUtility.DisplayDialog("NEKO", "场景里没有 " + RootName + "，先跑 NEKO → Build NPC Rig (YUI)。", "好"); return; }
        var eye = FindDeep(root.transform, "EyeAnchor");
        if (eye == null) { EditorUtility.DisplayDialog("NEKO", "找不到 EyeAnchor。", "好"); return; }

        if (!AssetDatabase.IsValidFolder("Assets/NEKO")) AssetDatabase.CreateFolder("Assets", "NEKO");
        if (!AssetDatabase.IsValidFolder(Dir)) AssetDatabase.CreateFolder("Assets/NEKO", "Animations");

        var rt = AssetDatabase.LoadAssetAtPath<RenderTexture>(RtPath);
        if (rt == null)
        {
            rt = new RenderTexture(640, 360, 16, RenderTextureFormat.ARGB32);
            rt.name = "EyeCamRT";
            AssetDatabase.CreateAsset(rt, RtPath);
        }
        var mat = AssetDatabase.LoadAssetAtPath<Material>(MatPath);
        if (mat == null)
        {
            var sh = Shader.Find("Unlit/Texture");
            mat = new Material(sh != null ? sh : Shader.Find("Standard"));
            mat.name = "EyeCamHud";
            AssetDatabase.CreateAsset(mat, MatPath);
        }
        mat.mainTexture = rt;
        EditorUtility.SetDirty(mat);

        // 相机
        var camT = eye.Find("EyeCamera");
        GameObject camGo = camT != null ? camT.gameObject : new GameObject("EyeCamera");
        if (camT == null) { Undo.RegisterCreatedObjectUndo(camGo, "EyeCamera"); camGo.transform.SetParent(eye, false); }
        // 注意：Unity 的"假 null"让 ?? 失效，必须用 == null 判断
        var cam = camGo.GetComponent<Camera>();
        if (cam == null) cam = Undo.AddComponent<Camera>(camGo);
        cam.fieldOfView = 60f;
        cam.nearClipPlane = 0.05f;
        cam.farClipPlane = 60f;
        cam.targetTexture = rt;
        cam.depth = -10f;
        cam.clearFlags = CameraClearFlags.Skybox;
        cam.allowHDR = false;
        cam.allowMSAA = false;
        cam.useOcclusionCulling = true;
        // 渲染完整世界：远端 Player、本地 PlayerLocal，以及本地 Avatar 实际使用的
        // MirrorReflection。只排除 HUD 自身所在的 UI 层，避免递归画中画。
        EnsurePlayerLayersVisible(cam);
        var al = camGo.GetComponent<AudioListener>();
        if (al != null) UnityEngine.Object.DestroyImmediate(al);
        cam.enabled = false; // 运行时由 NekoEyeCam 按 driver 身份打开

        // HUD Quad
        var hudT = root.transform.Find("EyeCamHud");
        GameObject hud;
        if (hudT != null) hud = hudT.gameObject;
        else
        {
            hud = GameObject.CreatePrimitive(PrimitiveType.Quad);
            hud.name = "EyeCamHud";
            Undo.RegisterCreatedObjectUndo(hud, "EyeCamHud");
            hud.transform.SetParent(root.transform, false);
            var col = hud.GetComponent<Collider>();
            if (col != null) UnityEngine.Object.DestroyImmediate(col);
        }
        hud.GetComponent<MeshRenderer>().sharedMaterial = mat;
        hud.transform.localScale = new Vector3(0.26f, 0.26f * 9f / 16f, 1f);
        // HUD 单独放到 UI(5) 层，主视角仍能显示；EyeCamera 排除该层，避免把小窗拍进
        // RenderTexture 造成递归反馈。不能再借用 PlayerLocal(10)，否则本地 avatar 也会被一起排除。
        hud.layer = 5;
        hud.SetActive(false);

        // 脚本
        var scripts = root.transform.Find("Scripts");
        GameObject scriptsGo = scripts != null ? scripts.gameObject : root;
        var eyeCam = scriptsGo.GetComponent<NekoEyeCam>();
        if (eyeCam == null) eyeCam = AddUdon<NekoEyeCam>(scriptsGo);
        if (eyeCam != null)
        {
            eyeCam.eyeCamera = cam;
            eyeCam.hudQuad = hud.transform;
            var tel = root.GetComponentInChildren<NekoNpcTelemetry>(true);
            if (tel != null) eyeCam.telemetry = tel;
            EditorUtility.SetDirty(eyeCam);
            Debug.Log("[NEKO] Eye Cam 已生成：EyeAnchor/EyeCamera → " + RtPath + " → EyeCamHud。进世界后 driver 右下角出现小窗；"
                      + "插件侧把截图区域框到小窗即可。Interact 任一挂了 NekoEyeCam 的可交互物体可开关。");
        }
        else
        {
            Debug.LogWarning("[NEKO] 相机与 HUD 已生成，但 NekoEyeCam 组件未能自动添加，请手动 Add Component 并连 eyeCamera/hudQuad/telemetry。");
        }
        AssetDatabase.SaveAssets();
        Selection.activeGameObject = camGo;
    }

    public static void EnsurePlayerLayersVisible(Camera cam)
    {
        if (cam == null) return;
        cam.cullingMask = (cam.cullingMask | PlayerRenderLayersMask) & ~EyeCamExcludedMask;
        EditorUtility.SetDirty(cam);
    }

    public static bool PlayerLayersAreVisible(Camera cam)
    {
        return cam != null
               && (cam.cullingMask & PlayerRenderLayersMask) == PlayerRenderLayersMask
               && (cam.cullingMask & EyeCamExcludedMask) == 0;
    }

    static Transform FindDeep(Transform t, string name)
    {
        if (t.name == name) return t;
        foreach (Transform c in t)
        {
            var r = FindDeep(c, name);
            if (r != null) return r;
        }
        return null;
    }

    static T AddUdon<T>(GameObject go) where T : Component
    {
        try
        {
            Type undo = null;
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                undo = asm.GetType("UdonSharpEditor.UdonSharpUndo", false);
                if (undo != null) break;
            }
            if (undo != null)
            {
                foreach (var m in undo.GetMethods(BindingFlags.Public | BindingFlags.Static))
                {
                    if (m.Name != "AddComponent" || !m.IsGenericMethodDefinition) continue;
                    var ps = m.GetParameters();
                    if (ps.Length != 1 || ps[0].ParameterType != typeof(GameObject)) continue;
                    var r = m.MakeGenericMethod(typeof(T)).Invoke(null, new object[] { go }) as T;
                    if (r != null) return r;
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
}
#endif
