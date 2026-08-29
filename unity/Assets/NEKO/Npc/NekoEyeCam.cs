/*
 * NekoEyeCam —— N4 兜底 B："世界相机 → RenderTexture → 跟随 driver 视野的 HUD 小窗"（UdonSharp）
 *
 * 思路：不依赖 VRChat 相机/Spout。NPC 眼位挂一个普通 Unity Camera，渲染到 RenderTexture；
 *   一块只对 driver 可见的小 Quad 贴在 driver 头前固定偏移（右下角），显示这张 RT。
 *   插件侧沿用现成的屏幕截图视觉链路（win-vision），把截图区域框到这个小窗即可——
 *   分辨率 640×360 足够 YOLO/VLM 看清"眼前有谁、穿什么"。
 *
 * 代价：占 driver 画面一角；像素经二次采样。好处：今天就能跑，与 NPC 移动天然同步，无任何官方相机限制。
 * 生成：菜单 NEKO → Build Eye Cam (fallback)（NekoEyeCamBuilder.cs）。
 */
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoEyeCam : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;
    [Tooltip("NPC 眼位相机（子物体 EyeCamera 上的 Camera，targetTexture 已在编辑器指到 RT）")]
    public Camera eyeCamera;
    [Tooltip("HUD 小窗（Quad，材质贴 RT）")]
    public Transform hudQuad;

    [Header("HUD 摆位（相对 driver 头部）")]
    [Tooltip("头部局部坐标偏移：x 右 / y 上 / z 前（米）")]
    public Vector3 hudOffset = new Vector3(0.28f, -0.18f, 0.6f);
    [Tooltip("小窗宽度（米），高度按 16:9")]
    public float hudWidth = 0.26f;
    [Tooltip("是否让小窗跟随头部转动（桌面模式=跟随画面；VR 建议关，改为跟身体）")]
    public bool followHeadRotation = true;
    [Tooltip("进场即启用（仅 driver）")]
    public bool startEnabled = true;

    private bool _active;
    private bool _resolved;

    void Start()
    {
        // 诊断：不经 Telemetry 门限，直接打日志，证明本脚本被执行（2026-08-29 排查 HUD 不显示）
        Debug.Log("[NEKO-EYECAM] Start: startEnabled=" + startEnabled + " cam=" + (eyeCamera != null) + " hud=" + (hudQuad != null) + " tel=" + (telemetry != null));
        _active = startEnabled;
        Apply(false);
        if (hudQuad != null) hudQuad.localScale = new Vector3(hudWidth, hudWidth * 9f / 16f, 1f);
    }

    public override void Interact()
    {
        if (telemetry != null && !telemetry.IsDriver()) return;
        _active = !_active;
        Apply(true);
    }

    private void Apply(bool announce)
    {
        bool on = _active && (telemetry == null || telemetry.IsDriver());
        if (eyeCamera != null) eyeCamera.enabled = on;
        if (hudQuad != null) hudQuad.gameObject.SetActive(on);
        if (announce && telemetry != null) telemetry.Emit("sys.eyecam", "\"mode\":\"rt_hud\",\"on\":" + telemetry.B(on));
    }

    void Update()
    {
        if (!_resolved)
        {
            // LocalPlayer 就绪后再按 driver 身份决定一次
            if (Networking.LocalPlayer == null) return;
            _resolved = true;
            Debug.Log("[NEKO-EYECAM] resolved: driver=" + (telemetry == null || telemetry.IsDriver()) + " active=" + _active);
            Apply(true); // 进场后报一次 rt_hud on/off + 引用是否齐全，便于日志定位
            if (telemetry != null)
                telemetry.Emit("sys.eyecam", "\"mode\":\"rt_hud\",\"refs\":{\"cam\":" + telemetry.B(eyeCamera != null)
                    + ",\"hud\":" + telemetry.B(hudQuad != null) + "},\"startEnabled\":" + telemetry.B(startEnabled));
        }
        if (!_active || hudQuad == null) return;
        VRCPlayerApi local = Networking.LocalPlayer;
        if (local == null) return;
        VRCPlayerApi.TrackingData head = local.GetTrackingData(VRCPlayerApi.TrackingDataType.Head);
        Quaternion rot = head.rotation;
        if (!followHeadRotation)
        {
            Vector3 f = rot * Vector3.forward; f.y = 0f;
            if (f.sqrMagnitude > 0.0001f) rot = Quaternion.LookRotation(f.normalized, Vector3.up);
        }
        hudQuad.position = head.position + rot * hudOffset;
        hudQuad.rotation = rot;
    }
}
