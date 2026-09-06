#if UNITY_EDITOR
using System.Reflection;
using UnityEditor;
using UnityEngine;

// 仅供 ClientSim：通过它自己的释放鼠标事件同时停止视角输入和切换 UI 光标。
// Udon/真实客户端没有同等公开接口，此编辑器适配器不会被打包进世界。
[InitializeOnLoad]
public static class NekoChatClientSimInput
{
    private static Component _input;
    private static MethodInfo _release;
    private static NekoNpcChatInput _chat;
    private static bool _owned;
    private static double _nextFind;

    static NekoChatClientSimInput()
    {
        EditorApplication.update += Update;
        AssemblyReloadEvents.beforeAssemblyReload += Release;
        EditorApplication.playModeStateChanged += state => { if (state == PlayModeStateChange.ExitingPlayMode) Release(); };
    }

    private static void Update()
    {
        if (!EditorApplication.isPlaying) { Release(); return; }
        if ((_input == null || _chat == null) && EditorApplication.timeSinceStartup >= _nextFind)
        {
            _nextFind = EditorApplication.timeSinceStartup + 1.0;
            _chat = Object.FindObjectOfType<NekoNpcChatInput>(true);
            // ClientSim 将输入组件标记为 DontSave，普通 FindObjectsOfType 会漏掉它。
            foreach (var component in Resources.FindObjectsOfTypeAll<Component>())
            {
                if (component.GetType().FullName != "VRC.SDK3.ClientSim.ClientSimBaseInput") continue;
                if (!component.gameObject.scene.IsValid() || !component.gameObject.activeInHierarchy) continue;
                _input = component;
                _release = component.GetType().GetMethod("InputMouseReleased", BindingFlags.Instance | BindingFlags.NonPublic);
                break;
            }
        }
        bool open = _chat != null && _chat.panelRoot != null && _chat.panelRoot.gameObject.activeInHierarchy;
        if (!open) { Release(); return; }
        if (_input == null || _release == null) return;
        _release.Invoke(_input, new object[] { true });
        _owned = true;
    }

    private static void Release()
    {
        if (_owned && _input != null && _release != null)
            _release.Invoke(_input, new object[] { Input.GetKey(KeyCode.Tab) });
        _owned = false;
    }
}
#endif
