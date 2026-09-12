"""VRM 文件检查及 Core27 到 VRM 的身体姿态传输。"""
import json
import struct
import numpy as np
from scipy.spatial.transform import Rotation
from ardy.viz.viser_utils import Character

BONES = {
    "hips": "Hips", "spine": "Spine", "chest": "Spine2", "upperChest": "Spine3",
    "neck": "Neck", "head": "Head",
    "leftShoulder": "LeftShoulder", "leftUpperArm": "LeftArm", "leftLowerArm": "LeftForeArm", "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder", "rightUpperArm": "RightArm", "rightLowerArm": "RightForeArm", "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpLeg", "leftLowerLeg": "LeftLeg", "leftFoot": "LeftFoot", "leftToes": "LeftToeBase",
    "rightUpperLeg": "RightUpLeg", "rightLowerLeg": "RightLeg", "rightFoot": "RightFoot", "rightToes": "RightToeBase",
}

def validate_vrm(data):
    if len(data) > 100 * 1024 * 1024:
        raise ValueError("VRM 文件不能超过 100 MiB")
    if len(data) < 20:
        raise ValueError("文件过短，不是有效的 VRM")
    magic, version, length = struct.unpack_from("<III", data)
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError("VRM 必须是完整的 glTF 2.0 二进制文件")
    size, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A or 20 + size > len(data):
        raise ValueError("VRM 的 JSON 数据块无效")
    doc = json.loads(data[20:20+size])
    extensions = doc.get("extensions", {})
    if "VRMC_vrm" in extensions:
        bones = extensions["VRMC_vrm"].get("humanoid", {}).get("humanBones", {})
        version = "1.0"
    elif "VRM" in extensions:
        bones = {b["bone"]: b for b in extensions["VRM"].get("humanoid", {}).get("humanBones", [])}
        version = "0.x"
    else:
        raise ValueError("文件缺少 VRM humanoid 扩展")
    required = {"hips", "spine", "head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"}
    if required - bones.keys():
        raise ValueError("缺少必要人形骨骼：" + ", ".join(sorted(required-bones.keys())))
    for bone in bones.values():
        if not isinstance(bone.get("node"), int) or not 0 <= bone["node"] < len(doc.get("nodes", [])):
            raise ValueError("VRM 骨骼节点索引无效")
    # 导入仅使用文件内资源，不加载外部路径或网络贴图。
    for entry in doc.get("images", []) + doc.get("buffers", []):
        if entry.get("uri") and not entry["uri"].startswith("data:"):
            raise ValueError("请导出内嵌贴图的 VRM，不支持外部资源路径")
    return version

class VrmCharacter(Character):
    def __init__(self, name, server, skeleton, *, vrm_data, **kwargs):
        if skeleton.name != "cskel27":
            raise ValueError("当前 VRM 重定向支持 Core 27 骨架")
        kwargs["create_skinned_mesh"] = False
        super().__init__(name, server, skeleton, **kwargs)
        self.vrm_handle = server.scene.add_glb(f"/{name}/vrm", vrm_data,
            visible=kwargs.get("visible_skinned_mesh", True))
        self.vrm_handle.vrm_bones = tuple(BONES)
        neutral = skeleton.neutral_joints.detach().cpu().numpy()
        self.source_height = float(neutral[skeleton.root_idx, 1] - neutral[:, 1].min())
        self.source_indices = [skeleton.bone_index[name] for name in BONES.values()]

    def set_pose(self, joints_pos, joints_rot, foot_contacts=None, frame_idx=None, root_velocity=None):
        # 蒙皮在浏览器执行，每帧只发送主要骨骼旋转，不传整个人物顶点。
        positions = joints_pos.detach().cpu()
        rotations = joints_rot.detach().cpu()
        self._last_visual_pose = (positions, rotations, foot_contacts, frame_idx, root_velocity)
        # 隐藏骨架时，不再计算圆柱变换或发送其网格更新。
        if self.skeleton_mesh is not None and self.skeleton_mesh.joints_batched_mesh.visible:
            super().set_pose(positions, rotations, foot_contacts, frame_idx, root_velocity)
        else:
            self.cur_joints_pos, self.cur_joints_rot = positions, rotations
            self.cur_foot_contacts = foot_contacts
            if self.skeleton_mesh is not None:
                self.skeleton_mesh.cur_joints_pos = positions
        q = Rotation.from_matrix(rotations.numpy()[self.source_indices]).as_quat()
        root = positions[self.skeleton.root_idx].numpy()
        self.vrm_handle.vrm_pose = tuple(float(x) for x in np.concatenate([root, [self.source_height], q.ravel()]))

    def set_skeleton_visibility(self, visible):
        # 暂停时重新打开骨架，也立即刷新到最后一帧。
        if visible and hasattr(self, "_last_visual_pose"):
            super().set_pose(*self._last_visual_pose)
        super().set_skeleton_visibility(visible)

    def set_skinned_mesh_visibility(self, visible):
        if hasattr(self, "vrm_handle"):
            self.vrm_handle.visible = visible

    def set_skinned_mesh_opacity(self, opacity):
        # VRM 自带透明材质，保留其材质参数。
        pass

    def clear(self):
        if hasattr(self, "vrm_handle"):
            self.vrm_handle.remove()
        super().clear()
