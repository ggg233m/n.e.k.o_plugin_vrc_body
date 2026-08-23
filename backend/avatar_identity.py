"""基于本地外观特征的会话级 Avatar 身份注册表。

检测器的 ``track_id`` 只保证短时连续；目标离开画面、检测框跳变或跟踪 TTL
到期后都会变化。本模块在它上面增加一层不落盘的会话身份，把新轨迹与先前看到
的 Avatar 外观重新关联。它不声称知道 VRChat 的 ``usr_``/``avtr_`` 身份，也不
进行真人生物识别。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import secrets
from typing import Any, Iterable, Mapping, Sequence


_DEFAULT_LABELS = frozenset({"person", "player", "avatar", "人物", "玩家"})
# track_id 已经提供短期连续性，无需每帧重复计算外观。按轨迹错峰刷新可避免多人
# 场景在 10 Hz 感知循环里平白增加几十毫秒延迟。
_DESCRIPTOR_REFRESH_OBSERVATIONS = 10


def _normalized_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    left = min(1.0, max(0.0, left))
    top = min(1.0, max(0.0, top))
    right = min(1.0, max(0.0, right))
    bottom = min(1.0, max(0.0, bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def appearance_descriptor(
    frame: Any,
    bbox: Sequence[float],
) -> tuple[float, ...] | None:
    """提取轻量、亮度相对稳定的 Avatar 外观描述子。

    描述子由颜色比例直方图、低权重亮度直方图和 2x4 空间颜色统计组成。它比
    感知哈希更能容忍缩放和小幅姿态变化，又不会引入第二个神经网络或网络请求。
    这只是虚拟形象的会话内重识别特征，不是可靠的永久身份凭据。
    """

    normalized = _normalized_bbox(bbox)
    if normalized is None:
        return None
    try:
        import numpy as np  # type: ignore[import-not-found]

        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] < 3:
            return None
        height, width = array.shape[:2]
        if height < 4 or width < 4:
            return None
        left, top, right, bottom = normalized
        x0 = min(width - 1, max(0, int(math.floor(left * width))))
        y0 = min(height - 1, max(0, int(math.floor(top * height))))
        x1 = min(width, max(x0 + 1, int(math.ceil(right * width))))
        y1 = min(height, max(y0 + 1, int(math.ceil(bottom * height))))
        crop = array[y0:y1, x0:x1, :3]
        if crop.shape[0] < 3 or crop.shape[1] < 3:
            return None

        # 最多采样约 1024 个像素；重识别看的是粗外观，继续提高分辨率只会拉长
        # 控制闭环延迟，对颜色/布局描述子没有实际收益。
        stride_y = max(1, int(math.ceil(crop.shape[0] / 32)))
        stride_x = max(1, int(math.ceil(crop.shape[1] / 32)))
        sampled = crop[::stride_y, ::stride_x].astype(np.float32, copy=False)
        if sampled.size == 0:
            return None
        finite = np.nan_to_num(sampled, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.max(finite)) > 1.5:
            finite = finite * (1.0 / 255.0)
        rgb = np.clip(finite, 0.0, 1.0)
        pixels = rgb.reshape(-1, 3)
        brightness = (
            pixels[:, 0] * 0.299
            + pixels[:, 1] * 0.587
            + pixels[:, 2] * 0.114
        )
        color_sum = np.sum(pixels, axis=1, keepdims=True)
        chroma = pixels / np.maximum(color_sum, 1e-4)

        features: list[float] = []
        count = max(1, pixels.shape[0])
        for channel in range(3):
            histogram = np.histogram(chroma[:, channel], bins=8, range=(0.0, 1.0))[0]
            features.extend((histogram.astype(np.float32) / count * 2.0).tolist())
        light_histogram = np.histogram(brightness, bins=8, range=(0.0, 1.0))[0]
        features.extend((light_histogram.astype(np.float32) / count * 0.5).tolist())

        # 粗空间统计保留“上衣/下装”布局，但权重低于全局颜色分布，避免动作变化
        # 让同一个 Avatar 失配。
        sampled_height, sampled_width = rgb.shape[:2]
        for row in range(4):
            block_y0 = sampled_height * row // 4
            block_y1 = max(block_y0 + 1, sampled_height * (row + 1) // 4)
            for column in range(2):
                block_x0 = sampled_width * column // 2
                block_x1 = max(block_x0 + 1, sampled_width * (column + 1) // 2)
                block = rgb[block_y0:block_y1, block_x0:block_x1].reshape(-1, 3)
                if block.size == 0:
                    features.extend((0.0,) * 5)
                    continue
                block_sum = np.sum(block, axis=1, keepdims=True)
                block_chroma = block / np.maximum(block_sum, 1e-4)
                block_light = (
                    block[:, 0] * 0.299
                    + block[:, 1] * 0.587
                    + block[:, 2] * 0.114
                )
                features.extend((np.mean(block_chroma, axis=0) * 0.75).tolist())
                features.append(float(np.mean(block_light)) * 0.25)
                features.append(float(np.std(block_light)) * 0.25)

        vector = np.asarray(features, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-8:
            return None
        return tuple(float(item) for item in vector / norm)
    except (ImportError, TypeError, ValueError, IndexError, FloatingPointError):
        return None


def _similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    value = sum(float(left) * float(right) for left, right in zip(first, second))
    return min(1.0, max(-1.0, value)) if math.isfinite(value) else 0.0


@dataclass(frozen=True)
class IdentityAssignment:
    identity_id: str
    track_entity_id: str
    method: str
    similarity: float | None


@dataclass
class _Identity:
    identity_id: str
    label: str
    descriptor: tuple[float, ...]
    last_seen: float
    observations: int = 1
    active_track_id: int | None = None


class AvatarIdentityRegistry:
    """把易变轨迹映射到有界、仅当前进程有效的 Avatar 身份。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        similarity_threshold: float = 0.90,
        similarity_margin: float = 0.04,
        retention_s: float = 1800.0,
        max_identities: int = 128,
        session_token: str | None = None,
        eligible_labels: Iterable[str] = _DEFAULT_LABELS,
    ) -> None:
        self.enabled = bool(enabled)
        self.similarity_threshold = min(0.999, max(0.5, float(similarity_threshold)))
        self.similarity_margin = min(0.5, max(0.0, float(similarity_margin)))
        self.retention_s = min(86400.0, max(1.0, float(retention_s)))
        self.max_identities = min(1024, max(1, int(max_identities)))
        raw_token = str(session_token or secrets.token_hex(4)).strip().replace(":", "_")
        self.session_token = raw_token[:24] or secrets.token_hex(4)
        self._eligible_labels = frozenset(str(item).strip().lower() for item in eligible_labels)
        self._identities: dict[str, _Identity] = {}
        self._track_bindings: dict[int, str] = {}
        self._next_identity = 1
        self._new_count = 0
        self._reidentified_count = 0
        self._ambiguous_count = 0

    def _new_identity_id(self) -> str:
        identity_id = f"avatar:session:{self.session_token}:{self._next_identity}"
        self._next_identity += 1
        return identity_id

    def _prune(self, now: float) -> None:
        expired = {
            identity_id
            for identity_id, identity in self._identities.items()
            if now - identity.last_seen > self.retention_s
        }
        if len(self._identities) - len(expired) > self.max_identities:
            survivors = sorted(
                (
                    identity
                    for identity_id, identity in self._identities.items()
                    if identity_id not in expired
                ),
                key=lambda item: item.last_seen,
            )
            overflow = len(survivors) - self.max_identities
            expired.update(item.identity_id for item in survivors[:overflow])
        if not expired:
            return
        for identity_id in expired:
            self._identities.pop(identity_id, None)
        self._track_bindings = {
            track_id: identity_id
            for track_id, identity_id in self._track_bindings.items()
            if identity_id not in expired
        }

    def _bind(self, track_id: int, identity_id: str) -> None:
        # 一个稳定身份在同一帧只能属于一个轨迹。重新识别成功后清掉旧轨迹绑定，
        # 防止旧轨迹稍后复活并发布同一个实体 ID。
        self._track_bindings = {
            candidate_track: candidate_identity
            for candidate_track, candidate_identity in self._track_bindings.items()
            if candidate_identity != identity_id or candidate_track == track_id
        }
        self._track_bindings[track_id] = identity_id

    @staticmethod
    def _update_descriptor(
        identity: _Identity,
        descriptor: tuple[float, ...],
        *,
        weight: float,
    ) -> None:
        if len(identity.descriptor) != len(descriptor):
            return
        blended = tuple(
            (1.0 - weight) * old + weight * new
            for old, new in zip(identity.descriptor, descriptor)
        )
        norm = math.sqrt(sum(item * item for item in blended))
        if norm > 1e-8 and math.isfinite(norm):
            identity.descriptor = tuple(item / norm for item in blended)

    def assign(
        self,
        detections: Sequence[Any],
        frame: Any,
        *,
        now: float,
        source_name: str,
    ) -> dict[int, IdentityAssignment]:
        if not self.enabled:
            return {}
        self._prune(now)
        active_track_ids = {
            int(item.track_id)
            for item in detections
            if getattr(item, "track_id", None) is not None
        }
        reserved_identities = {
            self._track_bindings[track_id]
            for track_id in active_track_ids
            if track_id in self._track_bindings
        }
        assigned_this_frame: set[str] = set()
        result: dict[int, IdentityAssignment] = {}

        for detection in sorted(
            detections,
            key=lambda item: float(getattr(item, "confidence", 0.0) or 0.0),
            reverse=True,
        ):
            raw_track_id = getattr(detection, "track_id", None)
            label = str(getattr(detection, "label", "") or "").strip().lower()
            if raw_track_id is None or label not in self._eligible_labels:
                continue
            track_id = int(raw_track_id)
            track_entity_id = f"{str(source_name).replace(':', '_')}:track:{track_id}"
            bound_id = self._track_bindings.get(track_id)
            bound = self._identities.get(bound_id or "")
            if bound is not None:
                # 轨迹连续期间主身份由 tracker 保证；这里只错峰更新外观模板，避免
                # N 个玩家每帧做 N 次特征提取。track_id 参与取模可把同帧建立的身份
                # 分散到后续不同帧刷新。
                should_refresh = (
                    bound.observations + track_id
                ) % _DESCRIPTOR_REFRESH_OBSERVATIONS == 0
                descriptor = (
                    appearance_descriptor(frame, getattr(detection, "bbox", ()))
                    if should_refresh
                    else None
                )
                similarity = (
                    _similarity(bound.descriptor, descriptor)
                    if descriptor is not None
                    else None
                )
                if descriptor is not None:
                    self._update_descriptor(bound, descriptor, weight=0.12)
                bound.last_seen = now
                bound.observations += 1
                bound.active_track_id = track_id
                assigned_this_frame.add(bound.identity_id)
                result[track_id] = IdentityAssignment(
                    bound.identity_id,
                    track_entity_id,
                    "track_continuity",
                    similarity,
                )
                continue

            descriptor = appearance_descriptor(frame, getattr(detection, "bbox", ()))
            if descriptor is None:
                continue

            candidates: list[tuple[float, _Identity]] = []
            for identity in self._identities.values():
                if identity.label != label:
                    continue
                if identity.identity_id in assigned_this_frame:
                    continue
                if identity.identity_id in reserved_identities:
                    continue
                candidates.append((_similarity(identity.descriptor, descriptor), identity))
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_score = candidates[0][0] if candidates else -1.0
            second_score = candidates[1][0] if len(candidates) > 1 else -1.0
            matched = (
                candidates
                and best_score >= self.similarity_threshold
                and best_score - second_score >= self.similarity_margin
            )
            if matched:
                identity = candidates[0][1]
                self._bind(track_id, identity.identity_id)
                self._update_descriptor(identity, descriptor, weight=0.05)
                identity.last_seen = now
                identity.observations += 1
                identity.active_track_id = track_id
                self._reidentified_count += 1
                method = "appearance_reid"
                similarity: float | None = best_score
            else:
                if candidates and best_score >= self.similarity_threshold:
                    self._ambiguous_count += 1
                identity_id = self._new_identity_id()
                identity = _Identity(identity_id, label, descriptor, now, active_track_id=track_id)
                self._identities[identity_id] = identity
                self._bind(track_id, identity_id)
                self._new_count += 1
                method = "new_identity"
                similarity = None
            assigned_this_frame.add(identity.identity_id)
            result[track_id] = IdentityAssignment(
                identity.identity_id,
                track_entity_id,
                method,
                similarity,
            )
        # 通常单帧目标数远小于容量；仍在批次末尾再收口一次，保证极端输入也
        # 不会让会话身份表持续超过配置上限。
        self._prune(now)
        return result

    def status(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "session_token": self.session_token,
            "identity_count": len(self._identities),
            "track_binding_count": len(self._track_bindings),
            "new_count": self._new_count,
            "reidentified_count": self._reidentified_count,
            "ambiguous_count": self._ambiguous_count,
            "similarity_threshold": self.similarity_threshold,
            "similarity_margin": self.similarity_margin,
            "retention_s": self.retention_s,
            "max_identities": self.max_identities,
            "descriptor_refresh_observations": _DESCRIPTOR_REFRESH_OBSERVATIONS,
            "persistent": False,
        }


__all__ = ["AvatarIdentityRegistry", "IdentityAssignment", "appearance_descriptor"]
