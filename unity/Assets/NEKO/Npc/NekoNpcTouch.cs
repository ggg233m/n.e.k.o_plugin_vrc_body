/*
 * NekoNpcTouch —— 带身份的触摸区（UdonSharp）
 *
 * 契约：《NPC Udon 接口规范 v1.0》§5.3、§4.4（touch.enter / touch.exit）。
 *   · 一个触摸区一个本脚本 + 一个 isTrigger 碰撞体（TouchZones/Head、Cheek、HandL、HandR、Torso）
 *   · 用 OnPlayerTriggerEnter/Exit（不用 VRCContact）：直接拿到 VRCPlayerApi → 谁摸的一步到位
 *   · 语义是"某玩家（胶囊）贴近该部位区域"，不是手部骨骼；部位球尺寸按此设计（头 r≈0.25m）
 *   · 0.5s 合并窗由 Telemetry.EmitEvent 统一处理
 *
 * 阶段：N0 挂点 / N3 验收（规范 §10 第 8 条：好友贴头 → touch.enter{part:"head", name} 身份正确）
 */
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoNpcTouch : UdonSharpBehaviour
{
    [Header("依赖")]
    public NekoNpcTelemetry telemetry;
    public NekoNpcPerception perception;

    [Tooltip("部位名：head / cheek / handL / handR / torso")]
    public string part = "head";

    [System.NonSerialized] public int enterCount;

    public override void OnPlayerTriggerEnter(VRCPlayerApi player)
    {
        Report("touch.enter", player);
    }

    public override void OnPlayerTriggerExit(VRCPlayerApi player)
    {
        Report("touch.exit", player);
    }

    private void Report(string type, VRCPlayerApi player)
    {
        if (telemetry == null || !telemetry.IsTouchEnabled() || player == null || !player.IsValid()) return;
        int slot = perception != null ? perception.SlotOf(player) : -1;
        enterCount++;
        telemetry.EmitEvent(type, type + ":" + part + ":" + player.playerId,
            "\"slot\":" + slot + ",\"pid\":" + player.playerId
            + ",\"name\":" + telemetry.J(player.displayName)
            + ",\"part\":" + telemetry.J(part));
    }
}
