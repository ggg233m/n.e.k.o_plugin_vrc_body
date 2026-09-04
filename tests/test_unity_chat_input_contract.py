"""世界内本地跟随式聊天输入的静态安全契约。"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "unity" / "Assets" / "NEKO" / "Npc" / "NekoNpcChatInput.cs"
BUILDER = ROOT / "unity" / "Assets" / "NEKO" / "Editor" / "NekoNpcChatInputBuilder.cs"
NAMEPLATE = ROOT / "unity" / "Assets" / "NEKO" / "Npc" / "NekoNpcNameplate.cs"
RIG_BUILDER = ROOT / "unity" / "Assets" / "NEKO" / "Editor" / "NekoNpcRigBuilder.cs"


class UnityChatInputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat = CHAT.read_text(encoding="utf-8-sig")
        cls.builder = BUILDER.read_text(encoding="utf-8-sig")
        cls.nameplate = NAMEPLATE.read_text(encoding="utf-8-sig")
        cls.rig_builder = RIG_BUILDER.read_text(encoding="utf-8-sig")

    def test_ui_follows_each_local_player_without_syncing_its_transform(self) -> None:
        self.assertIn("BehaviourSyncMode.NoVariableSync", self.chat)
        self.assertIn("GetTrackingData", self.chat)
        self.assertIn("panelHeadOffset", self.chat)
        self.assertNotIn("[UdonSynced]", self.chat)

    def test_only_owner_receives_parameterized_submit(self) -> None:
        self.assertIn("NetworkEventTarget.Owner", self.chat)
        self.assertIn("[NetworkCallable(maxEventsPerSecond: 1)]", self.chat)
        self.assertIn("NetworkCalling.CallingPlayer", self.chat)
        self.assertIn("Networking.IsOwner(gameObject)", self.chat)
        self.assertIn("telemetry.IsDriver()", self.chat)
        self.assertIn('telemetry.Emit("player.chat_submit"', self.chat)
        self.assertNotIn("displayName", self.chat)

    def test_local_controls_are_not_legacy_network_callable(self) -> None:
        for method in ("_Open", "_Close", "_Submit", "_InputEndEdit"):
            self.assertIn(f"public void {method}()", self.chat)

    def test_builder_creates_collapsible_vrc_world_ui(self) -> None:
        self.assertIn("VRCUiShape", self.builder)
        self.assertIn("InputField", self.builder)
        self.assertIn('"和猫娘聊天"', self.builder)
        self.assertIn('WireButton(panel.Find("Panel/SendButton")', self.builder)
        self.assertIn('WireButton(panel.Find("Panel/CloseButton")', self.builder)
        self.assertIn('WireInputSubmit(input, chat, "_Submit")', self.builder)
        self.assertIn("input.onSubmit = new InputField.SubmitEvent()", self.builder)
        self.assertIn('WireInputEndEdit(input, chat, "_InputEndEdit")', self.builder)
        self.assertIn("input.onEndEdit = new InputField.EndEditEvent()", self.builder)
        self.assertIn("panel.gameObject.SetActive(false)", self.builder)

    def test_typing_locks_and_all_exit_paths_release_local_movement(self) -> None:
        self.assertIn("_localPlayer.Immobilize(true)", self.chat)
        self.assertIn("_localPlayer.Immobilize(false)", self.chat)
        self.assertIn("_localPlayer.SetJumpImpulse(0f)", self.chat)
        self.assertIn("_localPlayer.SetJumpImpulse(_savedJumpImpulse)", self.chat)
        self.assertIn("inputLockStation.UseStation(_localPlayer)", self.chat)
        self.assertIn("inputLockStation.ExitStation(_localPlayer)", self.chat)
        self.assertIn("SetInputLocked(true)", self.chat)
        self.assertIn("SetInputLocked(false)", self.chat)
        self.assertIn("void OnDisable()", self.chat)
        self.assertIn("inputField.ActivateInputField()", self.chat)
        self.assertIn("inputField.DeactivateInputField()", self.chat)
        self.assertIn("PlayerMobility = VRC.SDKBase.VRCStation.Mobility.Immobilize", self.builder)
        self.assertIn("station.disableStationExit = true", self.builder)
        self.assertIn("station.seated = false", self.builder)
        self.assertIn("navigation.mode = Navigation.Mode.None", self.builder)

    def test_overhead_text_uses_chinese_dynamic_font_and_head_anchor(self) -> None:
        self.assertIn("NotoSansSC-Dynamic SDF.asset", self.rig_builder)
        self.assertIn("AtlasPopulationMode.Dynamic", self.rig_builder)
        self.assertIn("asset.isMultiAtlasTexturesEnabled = true", self.rig_builder)
        self.assertIn("bubble.enableWordWrapping = true", self.rig_builder)
        self.assertIn("public Transform headAnchor", self.nameplate)
        self.assertIn("headAnchor.position + Vector3.up", self.nameplate)
        self.assertIn("FaceTextToViewer(nameBillboard, head)", self.nameplate)
        self.assertIn("FaceTextToViewer(bubbleBillboard, head)", self.nameplate)
        self.assertNotIn("\n        transform.rotation = Quaternion.LookRotation", self.nameplate)

    def test_chat_ui_is_hotkey_only_and_hidden_when_idle(self) -> None:
        self.assertIn("public KeyCode openKey = KeyCode.T", self.chat)
        self.assertIn("Input.GetKeyDown(openKey)", self.chat)
        self.assertIn("Input.GetKeyDown(KeyCode.Return)", self.chat)
        self.assertIn("Input.GetKeyDown(KeyCode.Escape)", self.chat)
        self.assertNotIn("launcherRoot", self.chat)
        self.assertIn('uiRoot.transform.Find("LauncherCanvas")', self.builder)
        self.assertIn("Undo.DestroyObjectImmediate", self.builder)


if __name__ == "__main__":
    unittest.main()
