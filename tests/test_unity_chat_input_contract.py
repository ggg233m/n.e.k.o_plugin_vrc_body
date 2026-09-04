"""世界内本地跟随式聊天输入的静态安全契约。"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "unity" / "Assets" / "NEKO" / "Npc" / "NekoNpcChatInput.cs"
BUILDER = ROOT / "unity" / "Assets" / "NEKO" / "Editor" / "NekoNpcChatInputBuilder.cs"
NAMEPLATE = ROOT / "unity" / "Assets" / "NEKO" / "Npc" / "NekoNpcNameplate.cs"
RIG_BUILDER = ROOT / "unity" / "Assets" / "NEKO" / "Editor" / "NekoNpcRigBuilder.cs"
FULL_BUILDER = ROOT / "unity" / "Assets" / "NEKO" / "Editor" / "NekoYuiFullNpcBuilder.cs"


class UnityChatInputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat = CHAT.read_text(encoding="utf-8-sig")
        cls.builder = BUILDER.read_text(encoding="utf-8-sig")
        cls.nameplate = NAMEPLATE.read_text(encoding="utf-8-sig")
        cls.rig_builder = RIG_BUILDER.read_text(encoding="utf-8-sig")
        cls.full_builder = FULL_BUILDER.read_text(encoding="utf-8-sig")

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

    def test_character_dialogue_uses_chinese_font_and_viewer_orbit(self) -> None:
        self.assertIn("NotoSansSC-Dynamic SDF.asset", self.rig_builder)
        self.assertIn("AtlasPopulationMode.Dynamic", self.rig_builder)
        self.assertIn("asset.isMultiAtlasTexturesEnabled = true", self.rig_builder)
        self.assertIn("bubble.enableWordWrapping = true", self.rig_builder)
        self.assertIn("bubble.richText = true", self.rig_builder)
        self.assertIn("rect.sizeDelta = new Vector2(1.6f, 0.9f)", self.rig_builder)
        self.assertIn("bubble.fontSize = 0.72f", self.rig_builder)
        self.assertIn("RequireDialogueMaterial(dialogueFont)", self.rig_builder)
        self.assertIn("bubble.spriteAsset = RequireEmojiSpriteAsset()", self.rig_builder)
        self.assertIn('material.SetFloat("_OutlineWidth", 0.16f)', self.rig_builder)
        self.assertIn('material.EnableKeyword("UNDERLAY_ON")', self.rig_builder)
        self.assertNotIn("bubble.fontMaterial", self.rig_builder)
        self.assertIn("public Transform headAnchor", self.nameplate)
        self.assertIn("public Transform bodyAnchor", self.nameplate)
        self.assertIn("toViewer = toViewer.normalized", self.nameplate)
        self.assertNotIn("toViewer.Normalize()", self.nameplate)
        self.assertIn("toViewer * dialogueOrbitRadius", self.nameplate)
        self.assertNotIn("viewerLeft * dialogueSideOffset", self.nameplate)
        self.assertIn("Vector3.Lerp(bubbleBillboard.position, targetPosition, follow)", self.nameplate)
        self.assertIn("Vector3.up * dialogueAnchorHeight", self.nameplate)
        self.assertIn("Quaternion.LookRotation(-toViewer, Vector3.up)", self.nameplate)
        self.assertIn("Quaternion.Slerp", self.nameplate)
        self.assertIn("dialoguePositionDeadZone", self.nameplate)
        self.assertIn("dialogueRotationDeadZoneDegrees", self.nameplate)
        self.assertIn("void LateUpdate()", self.nameplate)
        self.assertNotIn("TrackingDataType.Head).rotation", self.nameplate)
        self.assertNotIn("dialogueVerticalOffset", self.nameplate)
        self.assertNotIn("FaceTextToViewer(nameBillboard, head)", self.nameplate)
        self.assertNotIn("Quaternion.LookRotation(direction, Vector3.up)", self.nameplate)
        self.assertNotIn('Text(plate, "NameText"', self.rig_builder)
        self.assertIn('Undo.DestroyObjectImmediate(nt.gameObject)', self.rig_builder)
        self.assertIn('nameplate.nameBillboard = null', self.full_builder)
        self.assertNotIn("\n        transform.rotation = Quaternion.LookRotation", self.nameplate)

    def test_dialogue_reveal_is_local_projection_of_one_atomic_sync(self) -> None:
        self.assertIn("[UdonSynced] private int _syncRevealStartServerMs", self.nameplate)
        self.assertIn("_syncRevealStartServerMs = _revealStartServerMs", self.nameplate)
        self.assertIn("_revealStartServerMs = _syncRevealStartServerMs", self.nameplate)
        self.assertIn("RenderDialogueProjection(Networking.GetServerTimeInMilliseconds(), false)", self.nameplate)
        self.assertEqual(self.nameplate.count("RequestSerialization();"), 1)
        self.assertNotIn('"reveal_start_server_ms"', self.nameplate)

    def test_dialogue_reveal_handles_unicode_punctuation_and_untrusted_tags(self) -> None:
        self.assertIn("NextDisplayUnitLength", self.nameplate)
        self.assertIn("CodePointLength", self.nameplate)
        self.assertIn("0x200D", self.nameplate)
        self.assertIn("0x1F3FB", self.nameplate)
        self.assertIn("0xE0100", self.nameplate)
        self.assertIn("revealShortPunctuationPauseMs = 120", self.nameplate)
        self.assertIn("revealLongPunctuationPauseMs = 240", self.nameplate)
        self.assertIn('Replace("<", "＜").Replace(">", "＞")', self.nameplate)
        self.assertIn('"<color=#FFFFFF00>"', self.nameplate)
        self.assertIn('"<voffset=-0.04em>"', self.nameplate)
        self.assertIn('"<voffset=0.06em>"', self.nameplate)
        self.assertNotIn('"<size=82%>', self.nameplate)
        self.assertNotIn('"<size=112%>', self.nameplate)
        self.assertIn("ProjectRangeForTmp", self.nameplate)
        self.assertIn("EmojiOneSpriteIndex", self.nameplate)
        self.assertIn('"<sprite=" + spriteIndex + " tint=1>"', self.nameplate)

    def test_dialogue_reveal_preserves_total_lifetime_and_reading_hold(self) -> None:
        self.assertIn("minimumFullTextHoldSeconds = 4f", self.nameplate)
        self.assertIn("availableRevealMs", self.nameplate)
        self.assertIn("scalePermille", self.nameplate)
        self.assertIn("_displayUntilServerMs", self.nameplate)
        self.assertIn('ClearBubbleWithReason("expired")', self.nameplate)
        self.assertIn('EmitCleared("replaced")', self.nameplate)

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
