// NekoYuiEditorLogBridge —— 仅供 ClientSim 闭环测试，不进入 VRChat 构建。
#if UNITY_EDITOR
using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

[InitializeOnLoad]
public static class NekoYuiEditorLogBridge
{
    static readonly object Gate = new object();
    static readonly UTF8Encoding Utf8 = new UTF8Encoding(false);

    public static string LogPath
    {
        get { return Path.GetFullPath(Path.Combine(Application.dataPath, "../Temp/YuiClientSim.log")); }
    }

    static NekoYuiEditorLogBridge()
    {
        Application.logMessageReceivedThreaded -= OnLog;
        Application.logMessageReceivedThreaded += OnLog;
    }

    static void OnLog(string condition, string stackTrace, LogType type)
    {
        if (string.IsNullOrEmpty(condition) || !condition.StartsWith("[NEKO]{", StringComparison.Ordinal)) return;
        try
        {
            lock (Gate)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(LogPath));
                File.AppendAllText(LogPath, condition + Environment.NewLine, Utf8);
            }
        }
        catch (Exception exception)
        {
            // 测试日志桥失败不能影响 Udon/ClientSim；只在 Editor 控制台给出诊断。
            Debug.LogWarning("[NEKO-EDITOR] 无法写入 ClientSim 协议日志：" + exception.Message);
        }
    }
}
#endif
