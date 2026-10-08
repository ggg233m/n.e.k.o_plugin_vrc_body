# -*- coding: utf-8 -*-
"""跨会话地点检索（P0 接线）—— 让新会话认出**以前来过的地方**。

与会话内 BoW（``nav_loop._bow_candidates``）的分工：那里找回的是**本会话内**的长程重访；
这里把**历史会话**持久化成检索索引，新会话的每个关键帧都去历史里查"这是不是我来过的地方"，
验证通过就记一条**跨会话约束**（写到世界目录的 jsonl），供固化档/位姿合并离线消费。

设计（对着已验证的实验定死，见 ``Docs/archive/跨会话地点检索实验（2026-10-02）.md``）：

* **索引两侧统一用 des3d**（有深度 ≤8 m 的描述子子集）。原因：持久会话只存 des3d+xyz
  （``nav_memory.encode_features``），全量 des 不落盘；des3d 对称（双方都是"看得见又测得到
  深度"的点）且查询侧 transform 便宜 60%。与实验（全量 des）的差异由回放验证兜底。
* **验证**照抄 ``nav_loop.relative_pose`` 语义：旧帧从会话 npz 重建 map_kf（只用 des3d+xyz），
  当前帧用在线 KeyframeFeatures。内点/覆盖率/重投影门槛全部生效。
* **跨会话专用的门**：``min_path_m`` 与漂移半径**不可用**（两个会话各自一个 gauge，路程与
  位置都不可比）；但 **HMD yaw 一致性门可用**（旋转与平移 gauge 无关；前提 = play space
  没被重置，重置过的话 yaw_jump 检测会留痕）+ ``max_offset_m`` 可用（相对量）。
  ⚠️ **2026-10-06 修正**：wrld_home 实测下这道**绝对 6° 门吞掉大量真重合**——001523 对
  10-01 库的真匹配成簇在 signed −6…−15°（内点中位 ~210，与被采纳的同档），绝对门只留下
  10 条、共识门（``yaw_consensus=True``）下 116 条。判据改为"相对该会话对带符号 yaw 残差
  中位数"的窗（``yaw_cap_deg`` 仍是硬顶：play space 重置 / 坏会话照拦）。
  当日已落地（默认开）并完成真世界并树：holdout 中位 0.169 / 0.27 m，见 ROADMAP P0 §洞 4。
* **隔离（v1 范围）**：确认的跨会话对**只写约束 jsonl，不碰位姿图**。把当前会话"采纳"进
  历史世界系需要世界系变换层（est/mapper/DR 全链换系），那是 P0.1 的活——v1 宁可只报告
  也不伪造修正。
* 词汇树是世界相关的（跨世界未测）：manifest 记 vocab 文件 sha1，换了词汇树自动重建。

线程模型：装载/构建在后台线程（start 不阻塞）；``on_keyframe`` 只在**建图线程**调用
（与 mapper/loops 同一个写者）；``write_back`` 在 stop() 里、``_end_memory`` 之后调用
（特征已落盘、线程已 join）。
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .nav_bow import BowIndex, BowVocabulary
    from .nav_loop import KeyframeFeatures, LoopConfig, relative_pose, rotation_angle_deg
except ImportError:                                          # pragma: no cover - 离线脚本单文件加载
    from nav_bow import BowIndex, BowVocabulary              # type: ignore[no-redef]
    from nav_loop import KeyframeFeatures, LoopConfig, relative_pose, rotation_angle_deg  # type: ignore[no-redef]

__all__ = ["XSessionConfig", "XSessionIndex", "XSessionTracker", "estimate_gauge",
           "align_into", "align_into_auto", "align_world_tree", "session_pose_table"]

# 2 = 会话表标记进入索引 manifest（merged / raw）：会话被采纳后**换了一张表**，
#     旧缓存必须失效重算，否则索引会继续供旧坐标（2026-10-05，P0 收尾）。
SCHEMA = 2
_INCLUDE_STATUS = {"complete", "complete_with_errors", "interrupted"}


@dataclass
class XSessionConfig:
    enabled: bool = True
    vocab: str = "models/bow_vocab.npz"   # 与 LoopConfig.bow_vocab 同源（世界相关，换世界要重训）
    query_top: int = 8                    # 每帧跨会话检索 shortlist（先全量排序，验证预算内确认即停）
    verify_max: int = 2                   # 每帧最多验证的候选数（相对_pose ~4 ms/对，预算闸）
    min_session_kf: int = 20              # 建索引/写回的会话最少特征数（同 memory.min_session_keyframes）
    max_docs: int = 16384                 # 跨会话索引文档上限（超出保留最新会话）
    reconfirm_path_m: float = 1.5         # 同一旧 doc 两次确认之间，新会话至少要走的 OSC 路程（防原地刷约束）
    # 会话末自动采纳（P0.3b，write_back 成功后后台线程跑 align_into_auto）
    align_min_constraints: int = 8        # 对某旧会话的确认约束少于此数则不采纳（gauge 不足信）
    align_tail_m: float = 100.0           # 留出验证的"尾段"定义（路程阈值）
    align_holdout_frac: float = 0.3       # 尾段锚按 new_kf 分组留出的比例（A/B 铁律）
    align_elastic: float = 0.15           # 里程边弹性：σ = 0.05 + elastic·step（m）；0=刚性（拉不动漂移）
    # 跨会话 yaw 判据（2026-10-06，见模块 docstring 的"修正"段）。yaw_consensus=True 时：
    # θ = 该 (new, old) 对"已过几何验证候选"的带符号 yaw 残差中位数（进池即收样，不需要先确认），
    # 判 |signed − θ| ≤ loop.yaw_tol_deg；池 < yaw_consensus_min 时退回绝对门。
    yaw_consensus: bool = True            # 2026-10-06 离线闭环验证通过后默认开（ROADMAP P0 §洞 4）
    yaw_consensus_min: int = 8            # 共识起步样本数（用中位数，抗假匹配混样）
    yaw_cap_deg: float = 20.0             # 硬顶：总旋转角超它一律拒（与共识窗无关）
    loop: LoopConfig = field(default_factory=LoopConfig)  # 只消费 ratio/min_inliers/reproj_px/
    #                                                       min_coverage/yaw_tol_deg/max_offset_m


def _vocab_path(raw: str) -> Path | None:
    """与 nav_loop._init_bow 同款解析：绝对路径优先，否则相对仓库根。"""
    p = Path(str(raw or "").strip())
    if not p.is_absolute():
        cand = Path(__file__).resolve().parent.parent / raw
        p = cand if cand.exists() else Path(raw)
    return p if p.is_file() else None


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _merged_table_path(world_dir: Path, sid: str) -> Path | None:
    """该会话**最新**的已采纳表（``xsession/merged_<sid>_into_*.npz``）；没有则 None。

    多个 merged 取 mtime 最新（同一会话可能被先后采纳进不同基准）。
    """
    try:
        cands = sorted((Path(world_dir) / "xsession").glob(f"merged_{sid}_into_*.npz"),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True)
    except OSError:
        return None
    return cands[0] if cands else None


def _table_marker(world_dir: Path, sid: str) -> str:
    """索引缓存的有效性标记：这个会话**现在该用哪张表**（原表 / 哪个 merged + 其 mtime）。

    会话被采纳后表换了、坐标就变了——没有这个标记，缓存的 ``index.npz`` 会继续
    供旧坐标，消费等于没接。
    """
    mp = _merged_table_path(world_dir, sid)
    if mp is None:
        return "raw"
    try:
        return f"merged:{mp.name}:{mp.stat().st_mtime_ns}"
    except OSError:                                  # pragma: no cover - 竞态删除
        return f"merged:{mp.name}"


def session_pose_table(world_dir: Path, sid: str) -> dict[str, Any] | None:
    """会话位姿表：**优先已采纳的 merged 表**，否则回退原表 ``sessions/<sid>/poses.npz``。

    merged 是会话末采纳的产物（P0.3b，在 ``xsession/`` 下与原表并存、**不覆盖原表**——
    v1 铁律）。下游（世界索引 / 再次对齐）从这里读到它，采纳的收益才真正进入链路：

    * 索引里该会话的 xy/R 换成**基准会话系**，后续会话的约束引用的是修正后的历史；
    * 再次对齐时按 ``base_sid`` 继续往上传（链式：C→B、B→A ⇒ C 落在 A 系）。

    ids 对不上 / 文件坏 ⇒ 退回原表（宁可旧坐标，不要错误坐标）。返回 dict：
    ``ids / T_map / dist_m / has_feat / source("raw"|"merged") / base_sid``。
    """
    sdir = Path(world_dir) / "sessions" / sid
    try:
        with np.load(sdir / "poses.npz") as z:
            ids = np.asarray(z["ids"], np.int64)
            raw = {"ids": ids,
                   "T_map": np.asarray(z["T_map"], np.float64),
                   "dist_m": np.asarray(z["dist_m"], np.float64),
                   "has_feat": (np.asarray(z["has_feat"], bool) if "has_feat" in z.files
                                else np.ones(len(ids), bool)),
                   "source": "raw", "base_sid": None}
    except (OSError, KeyError, ValueError):
        return None
    mp = _merged_table_path(world_dir, sid)
    if mp is None:
        return raw
    try:
        with np.load(mp) as z:
            m_ids = np.asarray(z["ids"], np.int64)
            T_map = np.asarray(z["T_map"], np.float64)
            dist_m = np.asarray(z["dist_m"], np.float64)
            base = str(z["base_sid"]) if "base_sid" in z.files else None
    except (OSError, KeyError, ValueError):
        return raw
    if len(m_ids) != len(ids) or not np.array_equal(m_ids, ids):
        return raw
    return {"ids": ids, "T_map": T_map, "dist_m": dist_m, "has_feat": raw["has_feat"],
            "source": "merged", "base_sid": base}


class XSessionIndex:
    """世界级只读检索索引：文档 = 历史会话的关键帧（des3d 直方图 + 位姿 + 惰性特征）。"""

    def __init__(self, vocab: BowVocabulary, world_dir: Path,
                 sids: list[str], kfs: list[int], xy: np.ndarray, R: np.ndarray,
                 dist_m: np.ndarray, words: np.ndarray, indptr: np.ndarray,
                 dirs: dict[str, Path], manifest: dict[str, Any]):
        self.vocab = vocab
        self.world_dir = world_dir
        self.sids = sids
        self.kfs = kfs
        self.xy = xy
        self.R = R
        self.dist_m = dist_m
        self.words = words
        self.indptr = indptr
        self.dirs = dirs
        self.manifest = manifest
        self._bow = BowIndex(vocab, cap_docs=max(1024, len(sids) * 4))
        for i in range(len(kfs)):
            self._bow.add_words(i, words[indptr[i]:indptr[i + 1]])
        self._kf_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._kf_index: dict[tuple[str, int], int] | None = None

    @property
    def n_docs(self) -> int:
        return len(self.kfs)

    def index_of(self, sid: str, kf: int) -> int | None:
        """(sid, kf) → doc 下标（惰性建一次全表；索引 ≤ ``max_docs`` 条，内存不成问题）。"""
        if self._kf_index is None:
            self._kf_index = {(str(s), int(k)): i for i, (s, k) in enumerate(zip(self.sids, self.kfs))}
        return self._kf_index.get((str(sid), int(kf)))

    def sessions(self) -> list[str]:
        return sorted(set(self.sids))

    def query(self, words: np.ndarray, top_n: int) -> list[tuple[int, float]]:
        """words = 查询帧 des3d 的 word id（XSessionTracker 已算好，避免二次 transform）。

        norm="l1"（对称打分）：REF 侧 des3d 分布与查询侧不同（跨会话各一个感知状态），
        "min" 口径会让 des3d 少的 REF 帧次次满分霸榜、把 verify 预算全部挤掉
        （2026-10-02 wrld_home 实测 025013 排名第一的帧 89% 过不了 min_inliers 门）。
        """
        return self._bow.query_words(words, top_n=top_n, norm="l1")

    def doc(self, i: int) -> tuple[str, int, np.ndarray, np.ndarray]:
        """(sid, kf, xy, R)。"""
        return self.sids[i], int(self.kfs[i]), self.xy[i], self.R[i]

    def load_kf(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """惰性加载旧帧的 (xyz f32, des3d)——验证时才读盘，每帧一个小 npz。"""
        hit = self._kf_cache.get(i)
        if hit is not None:
            return hit
        sid, kf = self.sids[i], int(self.kfs[i])
        with np.load(self.dirs[sid] / "kf" / f"{kf:06d}.npz") as z:
            xyz = z["xyz"].astype(np.float32)
            des3d = np.asarray(z["des3d"], np.uint8)
        if len(self._kf_cache) >= 256:                 # 简单上限：验证热点集中时免反复读盘
            self._kf_cache.clear()
        self._kf_cache[i] = (xyz, des3d)
        return xyz, des3d


def _scan_sessions(world_dir: Path, min_kf: int) -> list[dict[str, Any]]:
    """列出一个世界里可入索引的会话：状态合法、特征数达标、poses/kf 都在。"""
    out = []
    sroot = world_dir / "sessions"
    if not sroot.is_dir():
        return out
    for sdir in sorted(sroot.iterdir()):
        if not sdir.is_dir():
            continue
        meta = {}
        try:
            meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("status") not in _INCLUDE_STATUS:
            continue
        n_feat = int(meta.get("features") or 0)
        if n_feat < min_kf or not (sdir / "poses.npz").is_file() or not (sdir / "kf").is_dir():
            continue
        out.append({"sid": sdir.name, "dir": sdir, "features": n_feat,
                    "table": _table_marker(world_dir, sdir.name)})
    return out


def _load_session_docs(sdir: Path, vocab: BowVocabulary) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
    """一个会话 → (kf ids, xy, R, dist, words, indptr, 表来源)。任何缺口返回 None（该会话跳过）。

    位姿走 ``session_pose_table``：**已采纳的会话读 merged 表**（采纳收益由此进入索引）；
    末位 ``"raw"|"merged"`` 是**实际用到的**表来源，供 ``build_index`` 的 info 如实报告。
    """
    tab = session_pose_table(sdir.parent.parent, sdir.name)
    if tab is None:
        return None
    ids, T_map, dist, has_feat = tab["ids"], tab["T_map"], tab["dist_m"], tab["has_feat"]
    kfs, xys, Rs, dists, wparts = [], [], [], [], []
    for row, k in enumerate(ids):
        if not has_feat[row]:
            continue
        try:
            with np.load(sdir / "kf" / f"{int(k):06d}.npz") as z:
                des3d = np.asarray(z["des3d"], np.uint8)
        except (OSError, KeyError, ValueError):
            continue
        if len(des3d) == 0:
            continue
        kfs.append(int(k))
        xys.append(T_map[row][:2, 3])
        Rs.append(T_map[row][:3, :3])
        dists.append(float(dist[row]))
        wparts.append(vocab.transform(des3d).astype(np.int16))
    if not kfs:
        return None
    words = np.concatenate(wparts) if wparts else np.zeros(0, np.int16)
    counts = np.array([len(w) for w in wparts], np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return (kfs, np.asarray(xys, np.float32), np.asarray(Rs, np.float32),
            np.asarray(dists, np.float32), words, indptr, str(tab["source"]))


def build_index(cfg: XSessionConfig, world_dir: Path, exclude_sid: str | None) -> tuple[XSessionIndex | None, dict[str, Any]]:
    """扫世界目录重建索引（也是写回后刷新同一入口）。永不抛——失败返回 (None, info)。"""
    info: dict[str, Any] = {"sessions": 0, "docs": 0, "skipped": [], "tables": {}}
    vpath = _vocab_path(cfg.vocab)
    if vpath is None:
        info["reason"] = "vocab_missing"
        return None, info
    try:
        vocab = BowVocabulary.load(str(vpath))
    except Exception:                                    # pragma: no cover - 词汇树坏了不能拖垮导航
        info["reason"] = "vocab_load_error"
        return None, info
    info["vocab_sha1"] = _sha1(vpath)
    sessions = [s for s in _scan_sessions(world_dir, cfg.min_session_kf) if s["sid"] != exclude_sid]
    # 文档上限：优先保留最新会话（sid 即时间戳）。
    kept: list[dict[str, Any]] = []
    docs = 0
    for s in reversed(sessions):
        if docs >= cfg.max_docs:
            break
        kept.append(s)
        docs += s["features"]
    kept.reverse()
    kfs_all, xy_all, R_all, dist_all, w_all, ip_all, sids_all, dirs = [], [], [], [], [], [], [], {}
    offsets = [0]
    for s in kept:
        got = _load_session_docs(s["dir"], vocab)
        if got is None:
            info["skipped"].append(s["sid"])
            continue
        kfs, xy, R, dist, words, indptr, source = got
        info["tables"][s["sid"]] = source
        sids_all += [s["sid"]] * len(kfs)
        kfs_all += kfs
        xy_all.append(xy)
        R_all.append(R)
        dist_all.append(dist)
        w_all.append(words)
        ip_all.append(indptr[:-1] + offsets[-1])
        offsets.append(offsets[-1] + len(words))
        dirs[s["sid"]] = s["dir"]
        info["sessions"] += 1
        info["docs"] += len(kfs)
    if not kfs_all:
        info["reason"] = "no_sessions"
        return None, info
    manifest = {"schema": SCHEMA, "built_wall": time.time(), "vocab_sha1": info["vocab_sha1"],
                "min_session_kf": cfg.min_session_kf,
                "sessions": [{"sid": s["sid"], "features": s["features"],
                              "table": s.get("table", "raw")} for s in kept]}
    idx = XSessionIndex(vocab, world_dir, sids_all, kfs_all,
                        np.concatenate(xy_all) if xy_all else np.zeros((0, 2), np.float32),
                        np.concatenate(R_all) if R_all else np.zeros((0, 3, 3), np.float32),
                        np.concatenate(dist_all) if dist_all else np.zeros(0, np.float32),
                        np.concatenate(w_all) if w_all else np.zeros(0, np.int16),
                        np.concatenate(ip_all + [np.array([offsets[-1]])]) if ip_all else np.zeros(1, np.int64),
                        dirs, manifest)
    # 持久化缓存：下次启动直接 load（重建 ~10 s/3400 帧，缓存读 <1 s）。
    try:
        _save_index(world_dir, idx, manifest)
    except OSError:
        info["cache_write"] = "failed"                    # 缓存写失败不影响本会话使用
    return idx, info


def _save_index(world_dir: Path, idx: XSessionIndex, manifest: dict[str, Any]) -> None:
    """原子写索引缓存 + manifest。先写 tmp 再 replace（nav_memory 同款）。"""
    xdir = world_dir / "xsession"
    xdir.mkdir(parents=True, exist_ok=True)
    tmp_npz = xdir / "index.npz.tmp"
    with _NpzWriter(tmp_npz) as w:
        w.write(idx)
    tmp_npz.replace(xdir / "index.npz")
    tmp_json = xdir / "manifest.json.tmp"
    tmp_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp_json.replace(xdir / "manifest.json")


class _NpzWriter:
    """np.savez_compressed 的临时句柄。必须走文件对象：直接给路径时 savez 会自动追加 .npz。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, idx: XSessionIndex) -> None:
        with open(self.path, "wb") as f:
            np.savez_compressed(f,
                                kfs=np.asarray(idx.kfs, np.int32),
                                sids=np.asarray(idx.sids),
                                xy=idx.xy.astype(np.float32),
                                R=idx.R.astype(np.float32),
                                dist_m=idx.dist_m.astype(np.float32),
                                words=idx.words.astype(np.int16),
                                indptr=idx.indptr.astype(np.int64))

    def __enter__(self) -> "_NpzWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        if any(exc) and self.path.exists():               # 写一半失败别留半个 tmp
            self.path.unlink(missing_ok=True)


def load_index(cfg: XSessionConfig, world_dir: Path, exclude_sid: str | None) -> tuple[XSessionIndex | None, dict[str, Any]]:
    """优先读缓存；manifest（会话集/特征数/词汇树 sha）对不上或缓存坏 → 重建。"""
    info: dict[str, Any] = {"source": "cache"}
    vpath = _vocab_path(cfg.vocab)
    if vpath is None:
        return None, {"reason": "vocab_missing"}
    try:
        vocab = BowVocabulary.load(str(vpath))
        manifest = json.loads((world_dir / "xsession" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        out = build_index(cfg, world_dir, exclude_sid)
        return out[0], {"source": "rebuild", **out[1]}
    if int(manifest.get("schema", 0)) != SCHEMA or manifest.get("vocab_sha1") != _sha1(vpath):
        out = build_index(cfg, world_dir, exclude_sid)
        return out[0], {"source": "rebuild", **out[1]}
    have = {s["sid"]: s for s in _scan_sessions(world_dir, cfg.min_session_kf) if s["sid"] != exclude_sid}
    want = {s["sid"]: (int(s.get("features") or 0), s.get("table", "raw"))
            for s in manifest.get("sessions", [])}
    # 会话表标记也算有效性：会话被采纳后换了表（raw → merged），缓存必须重算。
    if have.keys() != want.keys() or any(
            (have[k]["features"], have[k].get("table", "raw")) != v for k, v in want.items()):
        out = build_index(cfg, world_dir, exclude_sid)
        return out[0], {"source": "rebuild", **out[1]}
    try:
        with np.load(world_dir / "xsession" / "index.npz") as z:
            sids = [str(s) for s in z["sids"]]
            kfs = [int(k) for k in z["kfs"]]
            words = np.asarray(z["words"], np.int16)
            indptr = np.asarray(z["indptr"], np.int64)
            xy = np.asarray(z["xy"], np.float32)
            R = np.asarray(z["R"], np.float32)
            dist = np.asarray(z["dist_m"], np.float32)
        dirs = {sid: world_dir / "sessions" / sid for sid in set(sids)}
        if len(indptr) != len(kfs) + 1 or (len(kfs) and indptr[-1] != len(words)):
            raise ValueError("index_corrupt")
        idx = XSessionIndex(vocab, world_dir, sids, kfs, xy, R, dist, words, indptr, dirs, manifest)
    except (OSError, KeyError, ValueError):
        out = build_index(cfg, world_dir, exclude_sid)
        return out[0], {"source": "rebuild", **out[1]}
    info["sessions"] = len(set(sids))
    info["docs"] = len(kfs)
    return idx, info


def estimate_gauge(R_old: np.ndarray, p_old: np.ndarray, R_ab: np.ndarray, t_ab: np.ndarray,
                   R_new: np.ndarray, p_new: np.ndarray, weights: np.ndarray | None = None, *,
                   iters: int = 3, pos_tol_m: float = 0.5, rot_tol_deg: float = 5.0
                   ) -> dict[str, Any] | None:
    """从跨会话约束估计新会话 → 旧会话 map 系的 gauge 变换（旋转 R_G + 平移 t_G，无尺度）。

    输入每条约束 i（形状 (N,3,3) / (N,3)）：
      * ``R_old/p_old``：旧 kf 在旧会话 map 系的位姿；
      * ``R_ab/t_ab``：新帧头部在旧 kf base 系下的相对位姿（constraints.jsonl 原样）；
      * ``R_new/p_new``：新 kf 在新会话 map 系的位姿。
    模型：``p_pred = R_old @ t_ab + p_old ≈ R_G @ p_new + t_G``（旋转同式）。
    两会话 yaw 基准同源（SteamVR 锚）⇒ 期望 R_G ≈ I；``R_dev_deg`` 偏离即对齐质量自检。

    鲁棒性：加权 SVD 旋转平均 + 平移均值，IRLS 按 pos/rot 残差剔外点后重拟合。
    返回 dict（G/R_G/t_G/内外点数/残差统计），约束 <3 条或退化返回 None。
    """
    R_old = np.asarray(R_old, np.float64); p_old = np.asarray(p_old, np.float64)
    R_ab = np.asarray(R_ab, np.float64); t_ab = np.asarray(t_ab, np.float64)
    R_new = np.asarray(R_new, np.float64); p_new = np.asarray(p_new, np.float64)
    n = len(p_new)
    if n < 3 or not (len(R_old) == len(p_old) == len(R_ab) == len(t_ab) == len(R_new) == n):
        return None
    if weights is None:
        weights = np.ones(n)
    w = np.maximum(np.asarray(weights, np.float64), 1e-6)
    # 约束给出的"新帧头部在旧 map 系"观测
    p_pred = np.einsum("nij,nj->ni", R_old, t_ab) + p_old
    R_pred = np.einsum("nij,njk->nik", R_old, R_ab)
    keep = np.ones(n, bool)
    R_G = np.eye(3)
    t_G = np.zeros(3)
    for _ in range(max(1, iters)):
        # 旋转：加权 SVD 投影平均 of R_pred @ R_new^T
        Msum = np.einsum("n,nij,nkj->ik", w[keep] / w[keep].sum(), R_pred[keep], R_new[keep])
        U, _s, Vt = np.linalg.svd(Msum)
        R_G = U @ Vt
        if np.linalg.det(R_G) < 0:                    # 反射退化：镜像修正
            U[:, -1] *= -1.0
            R_G = U @ Vt
        t_G = np.average(p_pred[keep] - np.einsum("ij,nj->ni", R_G, p_new[keep]),
                         axis=0, weights=w[keep])
        e_pos = np.linalg.norm(p_pred - (p_new @ R_G.T + t_G), axis=1)
        # 逐条旋转残差：angle( R_pred_i, R_G @ R_new_i )
        d = np.einsum("ij,njk->nik", R_G, R_new)
        cross = np.einsum("nij,nij->n", R_pred, d)     # trace(R_pred^T d)
        e_rot = np.degrees(np.arccos(np.clip((cross - 1.0) / 2.0, -1.0, 1.0)))
        # 剔除阈值渐进收紧：首轮外点重污染会把内点残差也抬过 pos_tol（全员被剔→放弃），
        # 先保住残差较小的多数派（60 分位），后续轮再收到目标阈值。
        tol = max(pos_tol_m, float(np.percentile(e_pos, 60)))
        new_keep = (e_pos <= tol) & (e_rot <= rot_tol_deg)
        if new_keep.sum() < 3:
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep
    if keep.sum() < 3:
        return None
    e_pos = np.linalg.norm(p_pred - (p_new @ R_G.T + t_G), axis=1)
    d = np.einsum("ij,njk->nik", R_G, R_new)
    cross = np.einsum("nij,nij->n", R_pred, d)
    e_rot = np.degrees(np.arccos(np.clip((cross - 1.0) / 2.0, -1.0, 1.0)))
    yaw_dev = math.degrees(math.atan2(R_G[1, 0], R_G[0, 0]))
    return {"R_G": R_G, "t_G": t_G, "n": n, "n_inlier": int(keep.sum()),
            "inlier": keep,
            "pos_med": float(np.median(e_pos[keep])), "pos_p90": float(np.percentile(e_pos[keep], 90)),
            "pos_max": float(e_pos[keep].max()),
            "rot_med": float(np.median(e_rot[keep])), "rot_p90": float(np.percentile(e_rot[keep], 90)),
            "R_dev_deg": float(abs(yaw_dev)),
            "e_pos": e_pos, "e_rot": e_rot}


def _elastic_merge(T: np.ndarray, dist: np.ndarray, anchors: list[dict[str, Any]],
                   R_G: np.ndarray, t_G: np.ndarray, *, elastic: float, tail_m: float,
                   holdout_frac: float, seed: int, irls_rounds: int) -> dict[str, Any]:
    """P0.2 弹性锚定位姿图：节点=(x,y,yaw)（z 不动），里程边保形 + 跨会话锚绝对观测。

    ⚠️ 两个学费（Docs/P0.2）：yaw 残差全程无 wrap（unwrap+锚分支对齐，±π 缝隙毁有限差分
    雅可比）；里程边 σ 必须按步长弹性（0.05+elastic·step），恒定刚性拉不动累积漂移。
    返回 dict（xy/yaw/训练留出统计/逐锚残差），调用方负责落盘。
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    N = len(T)
    NE = N - 1
    step = np.linalg.norm(T[1:, :2, 3] - T[:-1, :2, 3], axis=1)
    sig_od = np.column_stack([0.05 + elastic * step, 0.05 + elastic * step,
                              0.01 + 0.2 * elastic * step])
    sig_an = np.array([0.35, 0.35, 0.05])
    d_meas = np.einsum("rij,rj->ri", np.array([T[r][:2, :2].T for r in range(NE)]),
                       T[1:, :2, 3] - T[:-1, :2, 3])
    head_u = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
    dyaw_meas = np.diff(head_u)
    xy0 = T[:, :2, 3] @ R_G[:2, :2].T + t_G[:2]
    yaw0 = head_u + math.atan2(R_G[1, 0], R_G[0, 0])
    for a in anchors:                                 # 锚 yaw 分支对齐（无缠绕）
        d = a["ayaw"] - yaw0[a["row"]]
        a["ayaw_adj"] = yaw0[a["row"]] + (d + math.pi) % (2 * math.pi) - math.pi

    # 留出：尾段锚按 new_kf 分组（A/B 铁律——验证集不参与优化）
    rows_tail = sorted({a["row"] for a in anchors if a["dist"] > tail_m})
    te_rows: set[int] = set()
    if rows_tail and holdout_frac > 0:
        rng = np.random.default_rng(seed)
        te_rows = set(rng.choice(rows_tail, size=max(1, int(holdout_frac * len(rows_tail))),
                                 replace=False).tolist())
    tr_an = [a for a in anchors if a["row"] not in te_rows]
    te_an = [a for a in anchors if a["row"] in te_rows]

    def sparsity(an_list):
        S = lil_matrix((NE * 3 + len(an_list) * 3, 3 * N), dtype=np.int8)
        for r in range(NE):
            S[3 * r: 3 * r + 3, 3 * r: 3 * r + 6] = 1
        for k, a in enumerate(an_list):
            S[NE * 3 + 3 * k: NE * 3 + 3 * k + 3, 3 * a["row"]: 3 * a["row"] + 3] = 1
        return S.tocsr()

    def residuals(x, an_list, aw):
        X = x.reshape(N, 3)
        xy, th = X[:, :2], X[:, 2]
        d = xy[1:] - xy[:NE]
        c, s = np.cos(th[:NE]), np.sin(th[:NE])
        r = np.empty(NE * 3 + len(an_list) * 3)
        r[0::3][:NE] = ((c * d[:, 0] + s * d[:, 1]) - d_meas[:, 0]) / sig_od[:, 0]
        r[1::3][:NE] = ((-s * d[:, 0] + c * d[:, 1]) - d_meas[:, 1]) / sig_od[:, 1]
        r[2::3][:NE] = (np.diff(th) - dyaw_meas) / sig_od[:, 2]
        rows = np.array([a["row"] for a in an_list])
        base = NE * 3
        r[base + 0::3] = (xy[rows, 0] - np.array([a["ax"] for a in an_list])) / sig_an[0] * aw
        r[base + 1::3] = (xy[rows, 1] - np.array([a["ay"] for a in an_list])) / sig_an[1] * aw
        r[base + 2::3] = (th[rows] - np.array([a["ayaw_adj"] for a in an_list])) / sig_an[2] * aw
        return r

    xy, yaw = xy0.copy(), yaw0.copy()
    ws = np.array([a["w"] for a in tr_an])
    aw = np.sqrt(ws / ws.mean())
    cau = np.ones(len(tr_an))
    for _ in range(max(1, irls_rounds)):
        sol = least_squares(residuals, np.column_stack([xy, yaw]).ravel(),
                            args=(tr_an, aw * cau), jac_sparsity=sparsity(tr_an),
                            method="trf", max_nfev=200)
        X = sol.x.reshape(N, 3)
        xy, yaw = X[:, :2].copy(), X[:, 2].copy()
        e = np.hypot(xy[[a["row"] for a in tr_an], 0] - [a["ax"] for a in tr_an],
                     xy[[a["row"] for a in tr_an], 1] - [a["ay"] for a in tr_an])
        cau = 1.0 / (1.0 + (e / 0.8) ** 2)            # Cauchy IRLS 抗坏锚

    def residuals_of(alist, xy_, yaw_):
        return np.array([math.hypot(xy_[a["row"], 0] - a["ax"], xy_[a["row"], 1] - a["ay"])
                         for a in alist])

    e_tr = residuals_of(tr_an, xy, yaw)
    e_te = residuals_of(te_an, xy, yaw) if te_an else np.zeros(0)
    return {"xy": xy, "yaw": yaw, "xy0": xy0,
            "n_train": len(tr_an), "n_holdout": len(te_an),
            "train_med": float(np.median(e_tr)) if len(e_tr) else None,
            "train_p90": float(np.percentile(e_tr, 90)) if len(e_tr) else None,
            "holdout_med": float(np.median(e_te)) if len(e_te) else None,
            "holdout_p90": float(np.percentile(e_te, 90)) if len(e_te) else None,
            "e_train": e_tr,
            "moved": np.linalg.norm(xy - xy0, axis=1)}


def align_into(world_dir: Path, new_sid: str, old_sid: str, *, tail_m: float = 100.0,
               holdout_frac: float = 0.3, elastic: float = 0.15, seed: int = 7,
               irls_rounds: int = 4, write: bool = True,
               min_constraints: int = 8) -> dict[str, Any]:
    """把新会话对齐进旧会话世界系（P0.1 gauge 初值 + P0.2 弹性位姿图），产物与原表并存。

    ``min_constraints``：采纳门槛（默认 8，与 ``XSessionConfig.align_min_constraints`` 同源）——
    **原始约束数**与**有效锚数**都少于此数就拒绝（世界树 pass 用它当桥门槛）。
    产物（write=True 时，均在世界 ``xsession/`` 下，**不碰 sessions/*/poses.npz 原表**）：
      * ``align_<new>_into_<old>.json``：gauge/优化/留出统计 + 逐锚内外点掩码；
      * ``merged_<new>_into_<old>.npz``：修正整表（ids/T_map/dist_m + base_sid/method）。
    返回 dict（ok/reason + 统计）；输入缺口返回 ok=False（不抛，供后台线程直接消费）。
    """
    world_dir = Path(world_dir)
    # 新会话**必须**读原表：它可能已被对齐过，拿 merged 当输入会把同一修正叠第二次。
    try:
        with np.load(world_dir / "sessions" / new_sid / "poses.npz") as z:
            ids = np.asarray(z["ids"], np.int64)
            T = np.asarray(z["T_map"], np.float64)
            dist = np.asarray(z["dist_m"], np.float64)
    except (OSError, KeyError, ValueError) as exc:
        return {"ok": False, "reason": f"poses_unreadable:{type(exc).__name__}"}
    # 旧会话（基准）：优先已采纳的 merged 表 ⇒ 链式采纳（C→B、B→A）时 C 落在最终基准系，
    # base_sid 随之往上传（写出的 merged 记录的是**最终**基准而不是中间会话）。
    old_tab = session_pose_table(world_dir, old_sid)
    if old_tab is None:
        return {"ok": False, "reason": "poses_unreadable:OSError"}
    T_old = {int(k): Tm for k, Tm in zip(old_tab["ids"], old_tab["T_map"])}
    base_sid = old_tab.get("base_sid") or old_sid
    cpath = world_dir / "xsession" / "constraints.jsonl"
    if not cpath.is_file():
        return {"ok": False, "reason": "no_constraints"}
    try:
        cons = [json.loads(l) for l in cpath.read_text(encoding="utf-8").strip().splitlines()]
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"constraints_unreadable:{type(exc).__name__}"}
    cons = [c for c in cons if c.get("new_sid") == new_sid and c.get("old_sid") == old_sid]
    if len(cons) < int(min_constraints):
        return {"ok": False, "reason": "too_few_constraints", "n": len(cons),
                "threshold": int(min_constraints)}

    # 锚：约束 → 旧 map 系绝对观测（p_pred/yaw_pred）
    row_of = {int(k): r for r, k in enumerate(ids)}
    anchors = []
    sel = []
    for c in cons:
        Tj = T_old.get(c["old_kf"])
        r = row_of.get(c["new_kf"])
        if Tj is None or r is None:
            continue
        R_ab = np.asarray(c["R_ab"], np.float64)
        t_ab = np.asarray(c["t_ab"], np.float64)
        R_pred = Tj[:3, :3] @ R_ab
        p_pred = Tj[:3, :3] @ t_ab + Tj[:3, 3]
        anchors.append({"row": r, "ax": p_pred[0], "ay": p_pred[1],
                        "ayaw": math.atan2(R_pred[1, 0], R_pred[0, 0]),
                        "w": float(c["inliers"]), "dist": float(dist[r])})
        sel.append(c)
    if len(anchors) < int(min_constraints):
        return {"ok": False, "reason": "too_few_valid_anchors", "n": len(anchors),
                "threshold": int(min_constraints)}

    # P0.1 gauge 初值
    R_old = np.array([T_old[c["old_kf"]][:3, :3] for c in sel])
    p_old = np.array([T_old[c["old_kf"]][:3, 3] for c in sel])
    R_ab = np.array([np.asarray(c["R_ab"], np.float64) for c in sel])
    t_ab = np.array([np.asarray(c["t_ab"], np.float64) for c in sel])
    R_new = np.array([T[row_of[c["new_kf"]]][:3, :3] for c in sel])
    p_new = np.array([T[row_of[c["new_kf"]]][:3, 3] for c in sel])
    wts = np.array([float(c["inliers"]) for c in sel])
    g = estimate_gauge(R_old, p_old, R_ab, t_ab, R_new, p_new, wts)
    if g is None:
        return {"ok": False, "reason": "gauge_degenerate"}

    m = _elastic_merge(T, dist, anchors, g["R_G"], g["t_G"], elastic=elastic, tail_m=tail_m,
                       holdout_frac=holdout_frac, seed=seed, irls_rounds=irls_rounds)

    out: dict[str, Any] = {
        "ok": True, "new_sid": new_sid, "old_sid": old_sid,
        "n_constraints": len(cons), "n_anchors": len(anchors),
        "gauge": {"R_dev_deg": round(g["R_dev_deg"], 2), "pos_med_m": round(g["pos_med"], 3),
                  "n_inlier": g["n_inlier"], "n": g["n"],
                  "t_G": [round(float(v), 4) for v in g["t_G"]]},
        "train": {"n": m["n_train"], "med_m": round(m["train_med"], 3) if m["train_med"] else None,
                  "p90_m": round(m["train_p90"], 3) if m["train_p90"] else None},
        "holdout": {"n": m["n_holdout"],
                    "med_m": round(m["holdout_med"], 3) if m["holdout_med"] is not None else None,
                    "p90_m": round(m["holdout_p90"], 3) if m["holdout_p90"] is not None else None},
        "moved_m": {"head_med": round(float(np.median(m["moved"][dist <= tail_m])), 3),
                    "tail_med": round(float(np.median(m["moved"][dist > tail_m])), 3)
                    if (dist > tail_m).any() else None,
                    "max": round(float(m["moved"].max()), 3)}}

    if write:
        N = len(ids)
        T_out = T.copy()
        T_out[:, :2, 3] = m["xy"]
        T_out[:, :3, :3] = np.stack([np.array([[math.cos(t), -math.sin(t), 0],
                                               [math.sin(t), math.cos(t), 0], [0, 0, 1]])
                                     for t in m["yaw"]])
        xdir = world_dir / "xsession"
        xdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(xdir / f"merged_{new_sid}_into_{old_sid}.npz",
                            ids=ids, T_map=T_out, dist_m=dist,
                            base_sid=base_sid, method="anchored_pose_graph_v1")
        thr = max(2.0 * (m["train_med"] or 1.0), 1.0)   # 内外点掩码（供证据加权参考）
        rec = {"wall": round(time.time(), 1), "method": "anchored_pose_graph_v1",
               "new_sid": new_sid, "old_sid": old_sid, "base_sid": base_sid,
               **{k: v for k, v in out.items() if k not in ("ok",)},
               "gauge_R_G": [[round(float(v), 5) for v in row] for row in g["R_G"]],
               # 掩码只对训练锚（len = train.n）；留出锚不参与优化故无残差
               "inlier_note": "per TRAIN anchor, len==train.n; holdout anchors excluded",
               "inlier": [bool(v) for v in (m["e_train"] < thr)]}
        jp = xdir / f"align_{new_sid}_into_{old_sid}.json"
        tmp = jp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(jp)                               # 原子写
        out["merged"] = str(xdir / f"merged_{new_sid}_into_{old_sid}.npz")
        out["align"] = str(jp)
    return out


def align_into_auto(world_dir: Path, new_sid: str, cfg: XSessionConfig) -> dict[str, Any]:
    """会话末自动采纳入口：约束里挑确认最多的旧会话作基准，够门槛才对齐。不抛异常。"""
    try:
        cpath = Path(world_dir) / "xsession" / "constraints.jsonl"
        if not cpath.is_file():
            return {"ok": False, "reason": "no_constraints"}
        cons = [json.loads(l) for l in cpath.read_text(encoding="utf-8").strip().splitlines()]
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"constraints_unreadable:{type(exc).__name__}"}
    per: dict[str, int] = {}
    for c in cons:
        if c.get("new_sid") == new_sid:
            per[c["old_sid"]] = per.get(c["old_sid"], 0) + 1
    if not per:
        return {"ok": False, "reason": "no_constraints_for_session"}
    old_sid, n = max(per.items(), key=lambda kv: kv[1])
    if n < int(cfg.align_min_constraints):
        return {"ok": False, "reason": "too_few_constraints",
                "best_old_sid": old_sid, "n": n,
                "threshold": int(cfg.align_min_constraints)}
    return align_into(world_dir, new_sid, old_sid,
                      tail_m=cfg.align_tail_m, holdout_frac=cfg.align_holdout_frac,
                      elastic=cfg.align_elastic,
                      min_constraints=int(cfg.align_min_constraints))


# ---- 世界树 pass（P0 §洞4：spanning tree —— 把多棵树并进同一坐标系）------------
# 跨会话采纳（align_into）是**逐对**的：世界里会各自长成多棵"树"（每棵各一个 gauge），
# 新会话落在哪棵取决于它看见了谁、看见多少。世界树 pass 补上"世界级"的那一步：
# 按约束图把够格的桥**链式**采纳进同一坐标系，产出一棵指向根的树 + 一份诚实的报告
#（分量 / 对外缺口 / 逐操作证据）。弱桥（低于 bridge_min 或过不了质量闸）**只报告不落盘**
# ——v1 铁律：宁可只报告，也不伪造修正。

_TREE_WRITE_GATE = {"min_holdout": 2, "holdout_med_max": 1.0}   # 落盘前质量闸（force 可关）


def _effective_table_path(world_dir: Path, sid: str) -> Path | None:
    """该会话**实际生效**的位姿表文件（口径与 ``session_pose_table`` 一致）：
    合法 merged 就是它，否则原表；都没有返回 None。"""
    raw = Path(world_dir) / "sessions" / sid / "poses.npz"
    mp = _merged_table_path(world_dir, sid)
    if mp is not None:
        tab = session_pose_table(world_dir, sid)
        if tab is not None and tab["source"] == "merged":
            return mp
    return raw if raw.is_file() else None


def _chain_sids(world_dir: Path, sid: str, max_hops: int = 16) -> list[str] | None:
    """"已采纳链"上的 sid 列表（含自己、以坐标系终点结尾）；过期/成环返回 None。

    每一环都做**过期检查**：``merged_u`` 必须不早于 base 当前有效表——base 在 u 被采纳
    之后又换过表的话，u 的坐标还停在 base 的旧系里，整条链作废（调用方必须重算 u）。
    """
    chain, cur = [sid], sid
    seen = {sid}
    for _ in range(max_hops):
        tab = session_pose_table(world_dir, cur)
        if tab is None:
            return None
        base = tab.get("base_sid")
        if not base or str(base) == cur:
            return chain
        mp = _merged_table_path(world_dir, cur)
        bp = _effective_table_path(world_dir, str(base))
        if mp is None or bp is None:
            return None
        try:
            if mp.stat().st_mtime_ns < bp.stat().st_mtime_ns:
                return None
        except OSError:                              # pragma: no cover - 竞态删除
            return None
        if str(base) in seen:
            return None
        seen.add(str(base))
        chain.append(str(base))
        cur = str(base)
    return None


def _constraint_edges(world_dir: Path) -> dict[tuple[str, str], int]:
    """扫 ``xsession/constraints.jsonl`` → 有向边计数 {(new, old): 条数}。

    方向即采纳方向：``new → old`` = "new 的帧看见了 old 的帧"，``align_into(new, old)`` 可用。
    计数口径与 align_into 的门槛一致（它就是逐条拿去当锚的）。坏行跳过（报告件不拖垮谁）。
    """
    out: dict[tuple[str, str], int] = {}
    try:
        lines = (Path(world_dir) / "xsession" / "constraints.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
    except OSError:
        return out
    for ln in lines:
        try:
            c = json.loads(ln)
            k = (str(c["new_sid"]), str(c["old_sid"]))
        except (ValueError, KeyError, TypeError):
            continue
        out[k] = out.get(k, 0) + 1
    return out


def _plan_tree(root: str, members: list[str], out_edges: dict[str, list[tuple[str, int]]],
               kf_of: dict[str, int], chains: dict[str, list[str] | None]) -> dict[str, Any]:
    """从 root 反向长出**指向 root 的树**（in-arborescence）。纯计算、不碰盘，选根与执行复用。

    * 只走"能采纳"方向的有向边 u→v（u 的帧看见过 v），边权 = 约束条数；
    * u 若已有新鲜**链**（u 被采纳进 b、边仍够格）⇒ 父 = b 并**等 b 先落位**：强链整段
      跟着基走（99 条的那条不会被 7 条的弱桥顶掉）；基落不了位，u 一起不动（只报告）；
    * 其余节点：到根距离（反向 BFS）近的先落位，父 = 已落位节点里边权最大者（树的最弱边
      尽量大）；同层先放"被别的会话链指着的基"，平手取会话规模、sid 升序（确定性）。
    """
    rev: dict[str, list[str]] = {}
    for u, lst in out_edges.items():
        for v, _c in lst:
            rev.setdefault(v, []).append(u)
    dist = {root: 0}
    dq = deque([root])
    while dq:
        v = dq.popleft()
        for u in rev.get(v, []):
            if u not in dist:
                dist[u] = dist[v] + 1
                dq.append(u)
    mem = set(members)
    bases = {ch[1] for ch in chains.values() if ch and len(ch) >= 2}
    chain_base: dict[str, str] = {}
    for u in members:
        ch = chains.get(u) or []
        if (len(ch) >= 2 and ch[1] != u and ch[1] in mem
                and any(v == ch[1] for v, _c in out_edges.get(u, []))):
            chain_base[u] = ch[1]

    def key(u: str) -> tuple:
        best = max((c for v, c in out_edges.get(u, []) if dist.get(v, 1 << 30) < dist.get(u, 1 << 30)),
                   default=0)
        return (dist.get(u, 1 << 30), 0 if u in bases else 1, -best, -kf_of.get(u, 0), u)

    order: list[str] = [root]
    parent: dict[str, tuple[str, int]] = {}
    pending = sorted((s for s in members if s != root), key=key)
    while pending:
        rest, progressed = [], False
        for u in pending:
            b = chain_base.get(u)
            if b is not None and b not in order:       # 链基还没落位：等它（链整段走）
                rest.append(u)
                continue
            if b is not None:                          # 挂回自己的链基（不另投强父，保链完整）
                parent[u] = (b, next(c for v, c in out_edges[u] if v == b))
            else:
                cand = [(c, v) for v, c in out_edges.get(u, []) if v in order]
                if not cand:                           # 等更强的父落位（或最后判不可达）
                    rest.append(u)
                    continue
                c, p = max(cand, key=lambda t: (t[0], t[1]))
                parent[u] = (p, c)
            order.append(u)
            progressed = True
        if not progressed:
            break
        pending = rest
    used = [c for _p, c in parent.values()]
    return {"root": root, "order": order, "parent": parent,
            "unreachable": sorted(set(members) - set(order)),
            "min_edge": min(used) if used else None, "total": sum(used)}


def _best_edge_of(sid: str, edges: dict[tuple[str, str], int],
                  bmin: int) -> dict[str, Any] | None:
    """该会话最强的一条跨会话边（含弱边）：给 unreachable 一个诚实解释（差多少才够桥）。"""
    best = None
    for (u, v), c in sorted(edges.items()):
        if sid not in (u, v):
            continue
        rec = ({"dir": "out", "peer": v, "count": c} if u == sid
               else {"dir": "in", "peer": u, "count": c})
        if best is None or c > best["count"]:
            best = {**rec, "meets_min": c >= bmin}
    return best


def _component_bridge(members: list[str], edges: dict[tuple[str, str], int],
                      bmin: int) -> dict[str, Any] | None:
    """该分量对外最强的一条边 = **合树的缺口**（含低于门槛的弱边）：还差多少才够桥。"""
    mem = set(members)
    best = None
    for (u, v), c in sorted(edges.items()):
        if (u in mem) == (v in mem):
            continue
        if best is None or c > best["count"]:
            best = {"new": u, "old": v, "count": c, "meets_min": c >= bmin}
    return best


def _tree_gate(out: dict[str, Any]) -> tuple[bool, str]:
    """落盘质量闸：预览够格吗（留出锚 ≥2 且留出中位 ≤1 m）。不够 ⇒ 只报告不落盘。"""
    ho = out.get("holdout") or {}
    n, med = int(ho.get("n") or 0), ho.get("med_m")
    if n < int(_TREE_WRITE_GATE["min_holdout"]):
        return False, f"holdout_too_few:{n}"
    if med is None or float(med) > float(_TREE_WRITE_GATE["holdout_med_max"]):
        return False, f"holdout_med:{med}"
    return True, "ok"


def _brief_align(o: dict[str, Any]) -> dict[str, Any]:
    """align_into 结果的精简证据（报告里每条操作留这些）。"""
    return {"ok": bool(o.get("ok")), "reason": o.get("reason"),
            "n_constraints": o.get("n_constraints"), "n_anchors": o.get("n_anchors"),
            "gauge": o.get("gauge"), "train": o.get("train"),
            "holdout": o.get("holdout"), "moved_m": o.get("moved_m")}


def align_world_tree(world_dir: Path, cfg: XSessionConfig | None = None, *,
                     bridge_min: int | None = None, root: str | None = None,
                     write: bool = True, force: bool = False) -> dict[str, Any]:
    """世界树 pass：把约束图里够格的桥**链式采纳**成"一个分量一个坐标系"（P0 §洞4）。

    步骤：1) 建有向约束图（``new → old`` 可采纳方向，边权 = 确认条数）；
    2) 弱连通分量（只认 ≥ ``bridge_min`` 的边；孤立会话自成分量）→ 逐候选根模拟，
       取（覆盖率, 树瓶颈, 树总权, 会话规模，平手取 sid 最早）最大者为根 → 长出指向根的树；
    3) 按树的次序逐个 ``align_into(子, 父)``：链新鲜且已在该系 ⇒ ``already`` 跳过；
       父这轮没落位 ⇒ ``blocked_parent``；否则先**预演**（算不落盘）过质量闸，够格才真落盘
       （``force=True`` 关闸；``write=False`` 只出计划+预演，不写任何 merged）。

    报告落 ``<world>/xsession/world_tree.json``（分量/缺口/树边/逐操作证据）。不抛异常。
    """
    world_dir = Path(world_dir)
    cfg = cfg or XSessionConfig()
    bmin = int(cfg.align_min_constraints if bridge_min is None else bridge_min)
    if not (world_dir / "xsession" / "constraints.jsonl").is_file():
        return {"ok": False, "reason": "no_constraints"}
    edges_all = _constraint_edges(world_dir)
    feats = {s["sid"]: int(s["features"]) for s in _scan_sessions(world_dir, cfg.min_session_kf)}
    if root is not None and root not in feats:
        return {"ok": False, "reason": f"root_not_found:{root}"}
    nodes = sorted(feats)
    excluded = sorted({s for pair in edges_all for s in pair} - set(nodes))
    out_edges: dict[str, list[tuple[str, int]]] = {}
    for (u, v), c in edges_all.items():
        if u in feats and v in feats and c >= bmin:
            out_edges.setdefault(u, []).append((v, c))
    adj: dict[str, set[str]] = {s: set() for s in nodes}
    for u, lst in out_edges.items():
        for v, _c in lst:
            adj[u].add(v)
            adj[v].add(u)
    comps: list[list[str]] = []
    seen: set[str] = set()
    for s in nodes:
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    # 每个会话的"已采纳链"（新鲜性/成环由 _chain_sids 判；None = 过期，按无链处理）
    chains: dict[str, list[str] | None] = {s: _chain_sids(world_dir, s) for s in nodes}

    report: dict[str, Any] = {
        "wall": round(time.time(), 1), "world": str(world_dir), "bridge_min": bmin,
        "write": bool(write), "force": bool(force), "dry_run": not write,
        "sessions": {s: feats[s] for s in nodes}, "excluded_sessions": excluded,
        "edges": {f"{u}->{v}": c for (u, v), c in sorted(edges_all.items())},
        "components": []}
    for comp in comps:
        cands = [root] if (root is not None and root in comp) else comp
        best_sc: tuple | None = None
        best_plan: dict[str, Any] | None = None
        for r in cands:                              # cands 有序 ⇒ 平手时 sid 最早的赢
            plan = _plan_tree(r, comp, out_edges, feats, chains)
            sc = (len(plan["order"]), plan["min_edge"] or 0, plan["total"], feats.get(r, 0))
            if best_sc is None or sc > best_sc:
                best_sc, best_plan = sc, plan
        assert best_plan is not None
        root_sid = str(best_plan["root"])
        ch = _chain_sids(world_dir, root_sid)
        root_gauge = ch[-1] if ch else root_sid        # 根的"坐标系终点"
        ops: list[dict[str, Any]] = []
        ok_sids = {root_sid}                       # 执行后（干跑=计划里）应在根系的会话
        previewable = {root_sid}                   # **此刻**真在根系的会话（预演只对它们成立）
        for u in best_plan["order"][1:]:
            p, cnt = best_plan["parent"][u]
            op: dict[str, Any] = {"sid": u, "parent": p, "count": cnt}
            chu = _chain_sids(world_dir, u)
            if chu is not None and chu[-1] == root_gauge:
                op["status"] = "already"              # 链新鲜且已在该系：不重复算
                ok_sids.add(u)
                previewable.add(u)
                ops.append(op)
                continue
            if p not in ok_sids:
                op["status"] = "blocked_parent"       # 父这轮没落位，链断在这里
                ops.append(op)
                continue
            if p not in previewable:
                op["status"] = "planned"              # 干跑里的链下游：父还没写，预演不成立
                ok_sids.add(u)
                ops.append(op)
                continue
            prev = align_into(world_dir, u, p, tail_m=cfg.align_tail_m,
                              holdout_frac=cfg.align_holdout_frac, elastic=cfg.align_elastic,
                              min_constraints=bmin, write=False)
            op["preview"] = _brief_align(prev)
            if not prev.get("ok"):
                op["status"] = "align_failed"
                ops.append(op)
                continue
            gate_ok, why = _tree_gate(prev)
            if not write:
                if gate_ok or force:
                    op["status"] = "would_align"
                    ok_sids.add(u)
                else:
                    op["status"] = "would_refuse"
                op["gate"] = why
            elif gate_ok or force:
                real = align_into(world_dir, u, p, tail_m=cfg.align_tail_m,
                                  holdout_frac=cfg.align_holdout_frac,
                                  elastic=cfg.align_elastic, min_constraints=bmin, write=True)
                op["status"] = "aligned" if real.get("ok") else "align_failed"
                op["merged"] = real.get("merged")
                if real.get("ok"):
                    ok_sids.add(u)
                    previewable.add(u)
            else:
                op["status"] = "refused_gate"
                op["gate"] = why
            ops.append(op)
        report["components"].append({
            "root": root_sid, "nodes": comp,
            "coverage": f"{len(best_plan['order'])}/{len(comp)}",
            "tree": [{"sid": u, "parent": best_plan["parent"][u][0],
                      "count": best_plan["parent"][u][1]} for u in best_plan["order"][1:]],
            "bridge": _component_bridge(comp, edges_all, bmin),
            "unreachable": [{"sid": u, "best_edge": _best_edge_of(u, edges_all, bmin)}
                            for u in best_plan["unreachable"]],
            "ops": ops})
    report["aligned"] = sum(1 for c in report["components"] for o in c["ops"]
                            if o["status"] == "aligned")
    report["skipped"] = sum(1 for c in report["components"] for o in c["ops"]
                            if o["status"] == "already")
    report["refused"] = sum(1 for c in report["components"] for o in c["ops"]
                            if o["status"] in ("refused_gate", "would_refuse"))
    xdir = world_dir / "xsession"
    xdir.mkdir(parents=True, exist_ok=True)
    jp = xdir / "world_tree.json"
    tmp = jp.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(jp)                               # 原子写
        report["report"] = str(jp)
    except OSError:                                   # pragma: no cover - 报告写不出去不影响结论
        report["report"] = None
    return {"ok": True, **report}


class XSessionTracker:
    """会话侧：装载世界索引（后台线程）→ 每关键帧查询+验证（建图线程）→ 会话末写回。

    * 任何失败都不抛：错误进 status，跨会话检索只是候选/约束来源，坏了导航照常。
    * v1 隔离：确认对只写约束 jsonl，**不碰位姿图**（世界系采纳是 P0.1）。
    """

    def __init__(self, cfg: XSessionConfig, world_dir: Path, sid: str,
                 session_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.world_dir = Path(world_dir)
        self.sid = str(sid)
        self.session_dir = Path(session_dir) if session_dir is not None else None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._index: XSessionIndex | None = None
        self._info: dict[str, Any] = {}
        self._reason = "disabled" if not cfg.enabled else "loading"
        self._n_queries = self._n_verified = self._n_confirmed = self._n_errors = 0
        self._rejects: dict[str, int] = {}
        self._recent: deque = deque(maxlen=8)
        self._last_confirm_dist: dict[int, float] = {}
        self._yaw_pool: dict[tuple[str, str], deque] = {}   # (new, old) → 带符号 yaw 残差样本
        self._base_cache: dict[str, str | None] = {}        # old_sid → 其位姿表坐标系根（table_base）
        self._query_ms: float | None = None
        self._written: dict[str, Any] = {"written": False}
        self._words: dict[int, np.ndarray] = {}       # k → des3d 的 word id（写回收集，建图线程写）
        self._vocab: BowVocabulary | None = None
        self._vocab_sha: str | None = None
        self._thread: threading.Thread | None = None
        if not cfg.enabled:
            return
        vpath = _vocab_path(cfg.vocab)
        if vpath is None:
            self._reason = "vocab_missing"
            return
        try:
            self._vocab = BowVocabulary.load(str(vpath))
            self._vocab_sha = _sha1(vpath)
        except Exception:                             # pragma: no cover
            self._vocab, self._reason = None, "vocab_load_error"
            return
        self._thread = threading.Thread(target=self._load_bg, name="navmesh-xsession", daemon=True)
        self._thread.start()

    # ---- 装载（后台线程）----
    def _load_bg(self) -> None:
        try:
            idx, info = load_index(self.cfg, self.world_dir, exclude_sid=self.sid)
            with self._lock:
                self._index, self._info = idx, info
                self._reason = "ok" if idx is not None else str(info.get("reason", "no_sessions"))
        except Exception as exc:                      # pragma: no cover
            with self._lock:
                self._index, self._reason = None, f"load_error:{type(exc).__name__}"
        finally:
            self._ready.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._index is not None

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """等后台装载结束（不保证成功，用 ready/status 判断结果）。"""
        return self._ready.wait(timeout=timeout)

    def join(self, timeout: float = 30.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ---- 只读查询（在线 gauge 估计 / 先验消费用；建图线程调）----
    def old_pose(self, sid: str, kf: int) -> tuple[np.ndarray, np.ndarray] | None:
        """旧关键帧在**本索引快照**系里的 ``(R(3,3), xy(2,))``；查不到返回 None。

        取向与会话末采纳一致：索引的 xy/R 来自 ``session_pose_table``（merged 优先），
        即"该会话位姿表所在的坐标系"——约束验证用的正是这份坐标。在线估 gauge 必须拿
        同一份，而不是重新读盘（盘上可能已是另一张表 ⇒ 混 gauge）。
        """
        with self._lock:
            idx = self._index
        if idx is None:
            return None
        i = idx.index_of(sid, kf)
        if i is None:
            return None
        _sid, _k, xy, R = idx.doc(i)
        return np.asarray(R, np.float64), np.asarray(xy, np.float64)

    def table_base(self, sid: str) -> str | None:
        """``sid`` 的位姿表在**当前索引快照**里所处坐标系的根：merged → 其 base_sid；raw → 自己。

        给"绝不混 gauge"的闸用（``nav_prior.PriorConsumer`` 要求先验坐标系根与约束引用
        的旧会话同根）。会话不在快照里 / merged 文件读不到 ⇒ None（无据可依，不猜）。
        """
        with self._lock:
            idx = self._index
        if idx is None:
            return None
        if sid in self._base_cache:
            return self._base_cache[sid]
        marker: str | None = None
        for s in idx.manifest.get("sessions", []):
            if s.get("sid") == sid:
                marker = str(s.get("table", "raw"))
                break
        if marker is None:
            base: str | None = None
        elif marker.startswith("merged:"):
            name = marker.split(":", 2)[1]
            try:
                with np.load(self.world_dir / "xsession" / name) as z:
                    base = str(z["base_sid"]) if "base_sid" in z.files else sid
            except (OSError, KeyError, ValueError):
                base = None
        else:
            base = sid
        self._base_cache[sid] = base
        return base

    # ---- 每关键帧（建图线程）----
    def on_keyframe(self, k: int, feat: Any, dist_m: float, R_map: np.ndarray, refresh: bool) -> list[dict[str, Any]]:
        """收集本会话写回数据 + 跨会话查询验证。返回本帧确认的约束（通常 0 或 1 条）。"""
        if refresh and (k - 1) in self._words:        # 原地补帧：与 SessionWriter/mapper 同口径
            del self._words[k - 1]
        if self._vocab is None or feat is None or len(feat.des3d) == 0:
            return []
        try:
            self._words[int(k)] = self._vocab.transform(feat.des3d).astype(np.int16)
        except Exception:                             # pragma: no cover
            return []
        if not self.ready:
            return []
        t0 = time.perf_counter()
        confirmed: list[dict[str, Any]] = []
        try:
            ranked = self._index.query(self._words[int(k)], top_n=int(self.cfg.query_top))
            tries = 0
            for doc_i, score in ranked:
                if tries >= int(self.cfg.verify_max):
                    break
                last = self._last_confirm_dist.get(doc_i)
                if last is not None and dist_m - last < self.cfg.reconfirm_path_m:
                    continue                            # 原地重复确认不花验证预算
                tries += 1
                with self._lock:
                    self._n_verified += 1
                rec = self._verify_one(int(doc_i), float(score), feat, R_map, float(dist_m), int(k))
                if rec is not None:
                    confirmed.append(rec)
                    self._last_confirm_dist[doc_i] = dist_m
        except Exception as exc:                      # 瞬时 IO/检索错不禁用，计数留痕
            with self._lock:
                self._n_errors += 1
                self._reason = f"query_error:{type(exc).__name__}"
        with self._lock:
            self._n_queries += 1
            self._query_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return confirmed

    def _reject(self, why: str) -> None:
        with self._lock:
            self._rejects[why] = self._rejects.get(why, 0) + 1

    def _verify_one(self, doc_i: int, score: float, feat: Any, R_map: np.ndarray,
                    dist_m: float, k: int) -> dict[str, Any] | None:
        cfg = self.cfg.loop
        sid, old_k, _xy_old, R_old = self._index.doc(doc_i)
        xyz, des3d = self._index.load_kf(doc_i)
        if len(des3d) < cfg.min_inliers:
            self._reject("short_map_kf")
            return None
        # relative_pose 对 map 侧只用 des3d + xyz；uv/des/K/size/eye_y 都是 query 侧的。
        map_kf = KeyframeFeatures(np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8),
                                  xyz, des3d, feat.K, feat.size, feat.eye_y)
        rel, why = relative_pose(map_kf, feat, cfg)
        if rel is None:
            self._reject(why)
            return None
        # HMD 朝向不漂（跨会话也成立：旋转与平移 gauge 无关；play space 被重置会先被 yaw_jump 留痕）。
        # 2026-10-06：绝对 6° 门实测吞真重合 ⇒ yaw_consensus 用"相对该会话对中位数"的窗（见 cfg 注释）。
        R_err = (np.asarray(R_old, np.float64) @ rel["R_ab"]).T @ R_map
        yaw_err = rotation_angle_deg(R_err)
        signed = math.degrees(math.atan2(float(R_err[1, 0]), float(R_err[0, 0])))
        theta = None
        if self.cfg.yaw_consensus:
            pool = self._yaw_pool.setdefault((self.sid, sid), deque(maxlen=256))
            if len(pool) >= int(self.cfg.yaw_consensus_min):
                theta = float(np.median(np.fromiter(pool, float)))
            pool.append(signed)                       # 先收样：被拒的样本同样进池（中位数靠池收敛）
        if yaw_err > float(self.cfg.yaw_cap_deg):
            self._reject("rotation_mismatch")         # 硬顶：重置/坏会话（如 025013）照拦
            return None
        over = (yaw_err > cfg.yaw_tol_deg) if theta is None else (abs(signed - theta) > cfg.yaw_tol_deg)
        if over:
            self._reject("rotation_mismatch")
            return None
        t_ab = rel["t_ab"]
        off = float(np.hypot(*t_ab[:2]))
        if off > cfg.max_offset_m:
            self._reject("too_far")
            return None
        rec = {"wall": round(time.time(), 3), "new_sid": self.sid, "new_kf": k,
               "old_sid": sid, "old_kf": old_k, "score": round(score, 3),
               "inliers": rel["inliers"], "coverage": rel["coverage"], "reproj_px": rel["reproj_px"],
               "rot_err_deg": round(yaw_err, 2), "offset_m": round(off, 3),
               "new_dist_m": round(dist_m, 2),
               "t_ab": [round(float(v), 4) for v in t_ab],
               "R_ab": [[round(float(v), 5) for v in row] for row in rel["R_ab"]]}
        try:                                          # 约束落盘：append-only，坏不了先验
            xdir = self.world_dir / "xsession"
            xdir.mkdir(parents=True, exist_ok=True)
            with open(xdir / "constraints.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass                                      # 写盘失败只丢留痕，不影响内存态
        with self._lock:
            self._n_confirmed += 1
            self._recent.append({k2: rec[k2] for k2 in ("new_kf", "old_sid", "old_kf", "inliers",
                                                        "rot_err_deg", "offset_m")})
        return rec

    # ---- 会话末写回（stop() 里调用，装载线程已可 join）----
    def write_back(self, table: dict[str, np.ndarray] | None) -> dict[str, Any]:
        """把本会话关键帧的直方图 + 终态位姿并入世界索引（原子重写缓存）。"""
        out: dict[str, Any] = {"written": False, "n_words": len(self._words)}
        if self._vocab is None:
            out["reason"] = self._reason
            self._written = out
            return out
        if len(self._words) < int(self.cfg.min_session_kf):
            out["reason"] = "too_few_keyframes"
            self._written = out
            return out
        try:
            if self._thread is not None:
                self._thread.join(timeout=60.0)       # 装载完成再写，避免与后台重建竞争
            if self._index is None:                   # 首个会话 / 装载失败：重建一次（含本会话）
                idx, binfo = build_index(self.cfg, self.world_dir, exclude_sid=None)
                with self._lock:
                    self._index, self._info = idx, binfo
                if idx is None:
                    out["reason"] = str(binfo.get("reason", "no_sessions"))
                    self._written = out
                    return out
                out["rebuilt"] = True
            idx = self._index
            if idx.manifest.get("vocab_sha1") != self._vocab_sha:
                out["reason"] = "vocab_changed"
                self._written = out
                return out
            if any(s.get("sid") == self.sid for s in idx.manifest.get("sessions", [])):
                out["reason"] = "already_written"     # 双重 stop 防护
                self._written = out
                return out
            ids = np.asarray(table["ids"], np.int64) if table else np.zeros(0, np.int64)
            T_map = np.asarray(table["T_map"], np.float64) if table else np.zeros((0, 4, 4))
            dist = np.asarray(table["dist_m"], np.float64) if table else np.zeros(0)
            row_of = {int(k): r for r, k in enumerate(ids)}
            ks = sorted(k for k in self._words if k in row_of)
            if len(ks) < int(self.cfg.min_session_kf):
                out["reason"] = "poses_missing"
                self._written = out
                return out
            wparts = [self._words[k] for k in ks]
            n0 = idx.n_docs
            base = int(idx.indptr[-1]) if len(idx.indptr) else 0
            idx.sids = idx.sids + [self.sid] * len(ks)
            idx.kfs = idx.kfs + ks
            idx.xy = np.concatenate([idx.xy, np.asarray([T_map[row_of[k]][:2, 3] for k in ks], np.float32)])
            idx.R = np.concatenate([idx.R, np.asarray([T_map[row_of[k]][:3, :3] for k in ks], np.float32)])
            idx.dist_m = np.concatenate([idx.dist_m, np.asarray([dist[row_of[k]] for k in ks], np.float32)])
            idx.words = np.concatenate([idx.words, np.concatenate(wparts).astype(np.int16)])
            counts = np.asarray([len(w) for w in wparts], np.int64)
            idx.indptr = np.concatenate([idx.indptr, base + np.cumsum(counts)])
            for j, w in enumerate(wparts):
                idx._bow.add_words(n0 + j, w)
            if self.session_dir is not None:
                idx.dirs[self.sid] = self.session_dir
            idx.manifest = {**idx.manifest, "built_wall": time.time(),
                            "sessions": list(idx.manifest.get("sessions", []))
                            + [{"sid": self.sid, "features": len(ks)}]}
            _save_index(self.world_dir, idx, idx.manifest)
            out.update(written=True, docs=len(ks), sessions=idx.manifest["sessions"])
        except Exception as exc:                      # noqa: BLE001 - 写回失败不能拖垮 stop
            out["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        self._written = out
        return out

    # ---- 遥测 ----
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"active": self._vocab is not None,
                    "reason": self._reason,
                    "vocab_sha1": (self._vocab_sha or "")[:12],
                    "sessions": int(self._info.get("sessions") or 0),
                    "docs": int(self._info.get("docs") or 0),
                    "collected": len(self._words),
                    "queries": self._n_queries, "verified": self._n_verified,
                    "confirmed": self._n_confirmed, "errors": self._n_errors,
                    "rejects": dict(self._rejects), "query_ms": self._query_ms,
                    "recent": list(self._recent), "write_back": dict(self._written)}
