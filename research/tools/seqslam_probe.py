"""SeqSLAM 思路的离线基线探针（不接实时链路）。

背景
----
全局描述子（``_context_descriptor`` / ``appearance_descriptor`` / OSNet 整帧）
已被 ``world_mapping_probe.py`` 实测否定：地点重访对的相似度**低于**异地点对，
排序反转，任何阈值都分不开。根因是第一人称视角朝向主导单帧全局相似度。

SeqSLAM 的思路是：不比单帧，比**一段连续轨迹**（沿序列取一串帧，与另一段序列
逐位比对后取均值）。同一地点即使朝向相反，沿同一条路走过时"画面变化的轨迹"
仍然相似。本工具用自相似度矩阵（self-similarity matrix）+ 速度搜索实现这一点，
并在"原路返回"素材上做回环检测评测。

素材与真值
----------
``2026-09-18 07-31-11.mkv``：64.83 s，单个 VRChat 世界，玩家明显移动。
区域标注来自 ``research/tools/world_region_similarity.py`` 作者对 1fps 65 帧的逐帧肉眼核验，
是一条**去程 + 原路返回**路线：

    A 大厅 1-9 / 58-63      B 舞台 10-14 / 37-43    C 岩台 15-19 / 53-57
    D 广场 20-27 / 44-52    E 舞蹈地板 28-32(单访)  F 花园 33-36(单访)
    M 菜单遮挡 64-65(剔除)

→ 有 4 组真值回环（A/B/C/D 各一对），E/F 是**单访负样本**（任何命中都是误报）。
→ 注意：返回段是**反向**行进的（去 A→B→C→D→E→F，回 B→D→C→A），
  因此回环对应自相似矩阵的**反对角线**，必须允许负速度才能匹配。这一点是本
  实验的关键，也直接决定运行时设计（agent 走回头路时朝向相反）。

用法
----
  # 1) 抽帧（顺序解码，AV1 不 seek）
  .venv/Scripts/python.exe research/tools/seqslam_probe.py extract \
      --video "2026-09-18 07-31-11.mkv" --fps 4 --out .tmp/seq_frames

  # 2) 跑 SeqSLAM 基线
  .venv/Scripts/python.exe research/tools/seqslam_probe.py run \
      --frames-dir .tmp/seq_frames --seq-len 10 --min-sep 8 \
      --json-out .tmp/seqslam_report.json

输出：单帧基线(ds=1) vs 序列匹配(ds=N) 的回环召回率、单访段误报率、PR/AUC。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# 区域真值（秒，含端点）。标签重复出现 = 该地点被重访。
# 【2026-09-20 修正】此前 agent 的标注有系统性错误（"舞台重访"实为蘑菇花园新区域、
# "大厅重访"实为泳池）。本表由主 agent 逐帧肉眼核验 13 张联络表重建：
#   hall        大厅（Avatar Mirror + Miku 海报墙室内）
#   portals     舞台/世界传送门平台（YOZORA 面板 + 世界球）
#   lounge      沙发休息区（水族箱 + 红帘）
#   statue      初音雕像底座区
#   walkway     广告牌/水族箱走道
#   plaza       室外沙地广场（红曲线地面，含亭子接近段）
#   rocks       岩石 + 规则牌窄缝
#   pooldeck    泳池平台（大厅外墙 Miku 海报 + Avatar Mirror 一侧）
#   dancepool   六边形发光舞池
#   garden      蘑菇花园 + 管道
# 真实重访：hall(×3) / statue(×2) / pooldeck(×3) / dancepool(×2)；
# 单访（负样本）：portals / lounge / walkway / plaza / rocks / garden
GT_SEGMENTS = [
    ("hall", 0.0, 3.0),
    ("portals", 3.25, 6.75),
    ("lounge", 7.0, 9.75),
    ("statue", 10.0, 16.5),
    ("walkway", 16.75, 17.75),
    ("plaza", 18.0, 31.75),
    ("rocks", 32.0, 33.25),
    ("pooldeck", 33.5, 36.25),
    ("dancepool", 36.5, 38.0),
    ("garden", 38.25, 45.75),
    ("pooldeck", 46.0, 47.75),
    ("hall", 48.0, 52.75),
    ("statue", 53.0, 55.5),
    ("pooldeck", 55.75, 57.75),
    ("dancepool", 58.0, 60.75),
    ("hall", 61.0, 63.0),
]
GT_EXCLUDE = [("M", 63.25, 65.0)]  # VRChat 菜单遮挡，不参与统计


# ---------------------------------------------------------------- 抽帧

def cmd_extract(args: argparse.Namespace) -> int:
    import cv2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[error] 打不开视频: {args.video}", file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[info] src_fps={src_fps:.2f} frames={total}")
    step = max(1, int(round(src_fps / args.fps)))

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    idx = 0
    next_t = 0.0
    times: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        if idx >= int(round(next_t * src_fps)):
            p = out / f"seq_{saved:04d}.jpg"
            cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            times.append(round(t, 3))
            saved += 1
            next_t += 1.0 / args.fps
        idx += 1
        if args.max_frames and saved >= args.max_frames:
            break
        if args.max_seconds and t > args.max_seconds:
            break
    cap.release()
    (out / "times.json").write_text(json.dumps(times), encoding="utf-8")
    print(f"[info] saved={saved} -> {out}")
    return 0


# ---------------------------------------------------------------- 描述子

def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


def _context_desc(frame) -> np.ndarray:
    from backend.avatar_identity import _context_descriptor
    d = _context_descriptor(frame)
    a = np.asarray(d, dtype=np.float32).reshape(-1)
    return _l2(a)


def _gray_desc(frame, gw: int = 16, gh: int = 12) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = g.reshape(-1)
    g = g - g.mean()                      # patch 归一化：抗整体亮度变化
    return _l2(g)


def _color_desc(frame) -> np.ndarray:
    from backend.avatar_identity import appearance_descriptor
    d = appearance_descriptor(frame, (0.0, 0.0, 1.0, 1.0))
    return _l2(np.asarray(d, dtype=np.float32).reshape(-1))


DESCRIPTORS = {
    "context": _context_desc,
    "gray": _gray_desc,
    "color": _color_desc,
}


# ---------------------------------------------------------------- 序列匹配

def dist_matrix(desc: np.ndarray) -> np.ndarray:
    """余弦距离矩阵 D[i,j] = 1 - cos(i,j)。desc 已 L2 归一化。"""
    s = desc @ desc.T
    return (1.0 - s).astype(np.float32)


def sequence_score(D: np.ndarray, seq_len: int, velocities: list[float]) -> np.ndarray:
    """SeqSLAM 序列匹配：S[i,j] = min_v mean_k D[i+k, j + round(k*v)]。

    负速度允许匹配**反向**行进的回环（原路返回时朝向相反，对应反对角线）。
    """
    M = D.shape[0]
    best = np.full((M, M), np.inf, dtype=np.float32)
    for v in velocities:
        acc = np.zeros((M, M), dtype=np.float32)
        cnt = np.zeros((M, M), dtype=np.float32)
        for k in range(seq_len):
            off = int(round(k * v))
            rows = M - k
            if rows <= 0:
                continue
            c_lo = max(0, off)
            c_hi = M + min(0, off)
            j_lo = max(0, -off)
            j_hi = M - max(0, off)
            if c_hi <= c_lo or j_hi <= j_lo:
                continue
            src = D[k:M, c_lo:c_hi]
            acc[:rows, j_lo:j_hi] += src
            cnt[:rows, j_lo:j_hi] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(cnt > 0, acc / np.maximum(cnt, 1e-6), np.inf)
        best = np.minimum(best, s.astype(np.float32))
    return best


def _row_contrast(S: np.ndarray) -> np.ndarray:
    """SeqSLAM 的行内对比度增强：减去行均值再除以行标准差（越小越好→取负）。"""
    m = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    return (S - m) / np.maximum(sd, 1e-6)


# ---------------------------------------------------------------- 评测

def gt_label(t: float, tol: float = 0.0) -> str | None:
    for lab, a, b in GT_SEGMENTS:
        if a - tol <= t <= b + tol:
            return lab
    for lab, a, b in GT_EXCLUDE:
        if a - tol <= t <= b + tol:
            return None
    return None


def gt_segments_of(label: str) -> list[tuple[float, float]]:
    return [(a, b) for lab, a, b in GT_SEGMENTS if lab == label]


def evaluate(S: np.ndarray, times: list[float], min_sep: float,
             tol: float = 1.0) -> dict:
    """对每个查询帧找最佳匹配（排除时间邻域），按真值判定命中/误报。"""
    N = len(times)
    min_far = int(min_sep * N / max(times[-1], 1e-6)) if N > 1 else 0
    min_far = max(min_far, 1)

    rows = []
    for i in range(N):
        ti = times[i]
        li = gt_label(ti)
        if li is None:
            continue
        # 排除时间邻域内的自匹配
        valid = [j for j in range(N) if abs(times[j] - ti) >= min_sep]
        if not valid:
            continue
        scores = S[i, valid]
        order = np.argsort(scores, kind="stable")
        best_j = valid[int(order[0])]
        top5 = [valid[int(x)] for x in order[:5]]
        topk = [valid[int(x)] for x in order[:20]]
        tb = times[best_j]

        segs = gt_segments_of(li)
        revisited = len(segs) >= 2
        # 命中：匹配帧落在**另一个**同标签段内（容差 tol 秒）
        def _hit(t: float) -> bool:
            for a, b in segs:
                if a - tol <= ti <= b + tol:
                    continue  # 自身所在段，跳过
                if a - tol <= t <= b + tol:
                    return True
            return False

        hit = revisited and _hit(tb)
        rows.append({
            "t": ti, "label": li, "revisited": revisited,
            "best_t": tb, "score": float(scores[order[0]]),
            "hit_top1": hit,
            "hit_topk": [revisited and any(_hit(times[j]) for j in topk[:k])
                         for k in (1, 5, 10, 20)],
        })

    pos = [r for r in rows if r["revisited"]]
    neg = [r for r in rows if not r["revisited"]]
    res: dict = {
        "n_query": len(rows), "n_pos": len(pos), "n_neg": len(neg),
        "recall_top1": (sum(r["hit_top1"] for r in pos) / len(pos)) if pos else None,
        "recall_topk": {f"@{k}": (sum(r["hit_topk"][i] for r in pos) / len(pos))
                        if pos else None
                        for i, k in enumerate((1, 5, 10, 20))},
    }
    # 逐标签召回（看是哪个地点认出来了）
    by_label: dict[str, list[bool]] = {}
    for r in pos:
        by_label.setdefault(r["label"], []).append(r["hit_top1"])
    res["recall_top1_by_label"] = {
        k: round(sum(v) / len(v), 4) for k, v in sorted(by_label.items())}
    byk: dict[str, dict[str, list[bool]]] = {}
    for r in pos:
        d = byk.setdefault(r["label"], {"@1": [], "@5": [], "@10": [], "@20": []})
        for i, k in enumerate(("@1", "@5", "@10", "@20")):
            d[k].append(r["hit_topk"][i])
    res["recall_topk_by_label"] = {
        k: {kk: round(sum(vv) / len(vv), 4) for kk, vv in v.items()}
        for k, v in sorted(byk.items())}

    # 阈值扫描：PR / AUC / 最佳平衡准确率
    if pos and neg:
        sp = np.array([r["score"] for r in pos])
        sn = np.array([r["score"] for r in neg])
        lo = float(min(sp.min(), sn.min()))
        hi = float(max(sp.max(), sn.max()))
        grid = np.linspace(lo, hi, 200)
        tpr = np.array([(sp <= t).mean() for t in grid])
        fpr = np.array([(sn <= t).mean() for t in grid])
        bal = (tpr + (1 - fpr)) / 2
        bi = int(np.argmax(bal))
        # AUC（梯形，按 fpr 升序）
        o = np.argsort(fpr)
        auc = float(np.trapezoid(tpr[o], fpr[o]))
        res["pr"] = {
            "best_balanced_acc": round(float(bal[bi]), 4),
            "best_threshold": round(float(grid[bi]), 4),
            "tpr_at_best": round(float(tpr[bi]), 4),
            "fpr_at_best": round(float(fpr[bi]), 4),
            "auc": round(auc, 4),
        }
        res["score_pos"] = {"min": round(float(sp.min()), 4),
                            "median": round(float(np.median(sp)), 4),
                            "max": round(float(sp.max()), 4)}
        res["score_neg"] = {"min": round(float(sn.min()), 4),
                            "median": round(float(np.median(sn)), 4),
                            "max": round(float(sn.max()), 4)}
        res["separable_at_zero_fpr"] = bool(sp.max() < sn.min())
    res["rows"] = rows
    return res


# ---------------------------------------------------------------- run

def cmd_run(args: argparse.Namespace) -> int:
    import cv2

    d: Path = args.frames_dir
    files = sorted(d.glob("*.jpg"))
    if not files:
        print(f"[error] 没有帧: {d}", file=sys.stderr)
        return 2
    times: list[float]
    tp = d / "times.json"
    if tp.exists():
        times = json.loads(tp.read_text(encoding="utf-8"))
    else:
        times = [i / args.fps for i in range(len(files))]
    times = times[:len(files)]

    frames = []
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            print(f"[warn] 读不了 {p.name}", file=sys.stderr)
            continue
        frames.append(im)
    print(f"[info] loaded {len(frames)} frames, t=[{times[0]}, {times[-1]}]s")

    vel = [round(float(x), 2) for x in np.arange(0.6, 1.45, 0.05)]
    if args.allow_reverse:
        vel = vel + [-v for v in vel]
    print(f"[info] velocities={vel}")

    report: dict = {"seq_len": args.seq_len, "min_sep": args.min_sep,
                    "allow_reverse": args.allow_reverse,
                    "n_frames": len(frames), "descriptors": {}}

    for name in args.descriptors:
        if name not in DESCRIPTORS:
            print(f"[warn] 未知描述子 {name}", file=sys.stderr)
            continue
        fn = DESCRIPTORS[name]
        try:
            vecs = [fn(f) for f in frames]
        except Exception as e:  # 依赖缺失时如实报错，不伪造
            print(f"[warn] 描述子 {name} 不可用: {e}", file=sys.stderr)
            report["descriptors"][name] = {"available": False, "error": str(e)}
            continue
        desc = np.stack(vecs).astype(np.float32)
        D = dist_matrix(desc)
        out: dict = {"available": True, "dim": int(desc.shape[1])}
        # 单帧基线（等价于 ds=1）
        out["single_frame"] = evaluate(D, times, args.min_sep)
        # 序列匹配
        S = sequence_score(D, args.seq_len, vel)
        out["seqslam"] = evaluate(S, times, args.min_sep)
        Sc = _row_contrast(S)
        out["seqslam_contrast"] = evaluate(Sc, times, args.min_sep)
        report["descriptors"][name] = out

    def _brief(o):
        """去掉逐帧 rows，只留统计；非 dict 原样返回。"""
        if isinstance(o, dict):
            return {k: _brief(v) for k, v in o.items() if k != "rows"}
        return o

    print(json.dumps(_brief(report["descriptors"]), ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\n[saved] {args.json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--video", type=Path, required=True)
    e.add_argument("--fps", type=float, default=4.0)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-frames", type=int, default=0)
    e.add_argument("--max-seconds", type=float, default=0.0)
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("run")
    r.add_argument("--frames-dir", type=Path, required=True)
    r.add_argument("--fps", type=float, default=4.0)
    r.add_argument("--seq-len", type=int, default=10)
    r.add_argument("--min-sep", type=float, default=8.0)
    r.add_argument("--descriptors", nargs="+",
                   default=["gray", "context", "color"])
    r.add_argument("--no-reverse", dest="allow_reverse", action="store_false")
    r.add_argument("--json-out", type=Path, default=None)
    r.set_defaults(func=cmd_run, allow_reverse=True)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
