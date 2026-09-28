"""离线时间基准校验 harness（阶段一）。

契约（**必须原样保留**）
------------------------
本 harness 的产出是 **``offline_timebase_validation``**，用**合成数据**固定阶段一里
最容易出错、也最适合自动化的那段流水线：

    帧源 → VideoTimebase → frame_index / frame_timestamp
         → action_timeline.jsonl → episode_action_summary()
         → route_history_seed_edges() → consensus_assign()

它**不是**、也**不得**被引作：

- ✗ 真实路线历史验证
- ✗ 真实 VRChat 动作验证
- ✗ 真实 AnyaDance 反馈验证

理由：合成数据能证明"时间基准自洽 + 字段能完整传递"，但证明不了任何真实设备行为。
它不需要伪造真实 VRChat 结果，却能先把阶段一中**与视频真伪无关**的那半截钉死。

真实环境只需要补一件事：录一段 VRChat 视频，同时保存帧索引和动作日志；
届时复用**同一个检查器**（本文件），比较真实数据是否通过同样的时间基准与路线边验收。

覆盖的 9 条
-----------
1. 固定 ``fps=20``、已知 ``anchor`` 与已知动作时间 ⇒ 帧号/帧时间戳正确；
2. 同一事件在不同 ``anchor`` 下得到**相同帧号**（帧号只依赖相对时间）；
3. 动作时间线与帧索引使用**同一时间基准**（合成 + 真实 ``.tmp/loop_frame_index.npz``）；
4. 已知速度片段按**时间积分**，而不是按帧数积分（换 fps 距离按 1/fps 缩放）；
5. ``NaN`` / ``±inf`` / ``<=0`` / ``bool`` 的 fps 必须**拒绝**；
6. 坏 JSONL 行**跳过但不破坏**后续读取；
7. 缺失动作记录时**明确返回** ``insufficient_action_records``（以及无日志时的显式降级）；
8. ``driver_ack=none`` / ``unknown`` **不影响路线摘要**（路线闸门只读 n_records / 距离）；
9. ``episode_action_summary → route_history_seed_edges → consensus_assign`` 字段**完整传递**。

用法
----
    .venv/Scripts/python.exe research/tools/offline_timebase_harness.py \
        --json-out .tmp/offline_timebase_validation.json

退出码 0 = 全部通过；1 = 有任一断言失败。所有中间产物写在仓库 ``.tmp/`` 下（**不落 C 盘**）。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# `driver_log.py` 位于包根并使用了相对导入（`from .config import ...`），
# 必须按包名导入；这里按 `tests/_bootstrap.py` 同样的方式注册一个合成包模块。
if "neko_anyadance_body" not in sys.modules:
    _pkg = types.ModuleType("neko_anyadance_body")
    _pkg.__path__ = [str(REPO)]  # type: ignore[attr-defined]
    sys.modules["neko_anyadance_body"] = _pkg

from neko_anyadance_body.driver_log import (  # noqa: E402
    ActionLogRecorder,
    ActionTimeline,
    VideoTimebase,
    episode_action_summary,
    load_action_timeline,
)
from research.tools.episode_loop_eval import (  # noqa: E402
    build_episode_evidence,
    consensus_assign,
    load_route_history,
    route_history_seed_edges,
    visual_seed_edges,
)

VALIDATION_KIND = "offline_timebase_validation"
DISCLAIMER = [
    "本结果是 offline_timebase_validation（离线时间基准校验），使用合成数据。",
    "它【不是】真实路线历史验证。",
    "它【不是】真实 VRChat 动作验证。",
    "它【不是】真实 AnyaDance 反馈验证。",
]

FPS = 20.0          # 本素材标定用帧率（默认与 driver_log 默认一致）
ANCHOR = 1000.0     # 合成锚点（秒）；用假时钟精确可控


# ------------------------------------------------------------------ 工具

class FakeClock:
    """可控单调时钟：让 anchor 与动作时间完全确定。"""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def _action(forward: float = 0.0, strafe: float = 0.0, yaw: float = 0.0,
            sent: bool = True, ack: str = "none",
            vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
            episode: str = "", goal: str = "", source: str = "harness") -> dict:
    """构造一条 `ActionLogRecorder.record(action=...)` 需要的动作字典。"""
    return {
        "goal_id": goal,
        "episode_id": episode,
        "input_command": {"forward": float(forward), "strafe": float(strafe),
                          "source": source},
        "actual_send_result": "sent" if sent else "failed",
        "driver_ack": ack,
        "osc_velocity": {"vx": float(vx), "vy": float(vy), "vz": float(vz)},
        "turn_intent": {"yaw_delta": float(yaw), "source": source},
    }


def write_log(path: Path, fps: float, anchor: float, specs) -> Path:
    """用**真实写入口** `ActionLogRecorder` 落盘。

    ``specs`` = ``[(frame, action_dict), ...]``；事件时刻由 ``anchor + frame/fps``
    换算，因此断言里能直接预期帧号。
    """
    tb = VideoTimebase.from_record({"fps": float(fps), "anchor_monotonic": float(anchor)})
    rec = ActionLogRecorder(path, tb, clock=lambda: float(anchor))
    if not rec.start():
        raise RuntimeError(f"无法打开动作日志: {path} ({rec.last_error})")
    for frame, act in specs:
        t = float(anchor) + int(frame) / float(fps)
        rec.record(action=act, monotonic_time=t)
    rec.stop()
    return path


# ------------------------------------------------------------------ 9 条检查

def check_1_fixed_fps_anchor(work: Path):
    """① 固定 fps=20、已知 anchor、已知动作时间 ⇒ 帧号/帧时间戳正确。"""
    path = work / "c1.jsonl"
    clk = FakeClock(ANCHOR)
    tl = ActionTimeline(path, FPS, clock=clk)
    if not tl.begin():
        return False, "ActionTimeline.begin() 失败"
    if tl.to_record() != {"fps": FPS, "anchor_monotonic": ANCHOR}:
        return False, f"timebase 记录不符: {tl.to_record()}"

    clk.t = ANCHOR + 7.25
    tl.record_command(forward=1.0, strafe=0.0, sent=True)
    clk.t = ANCHOR + 10.0
    tl.note_velocity(0.0, 0.0, 0.4)          # 速度应附着在其后的行上
    tl.record_command(forward=0.0, strafe=1.0, sent=True)
    tl.close()

    tb, rows = load_action_timeline(path)
    if tb is None or len(rows) != 2:
        return False, f"读回异常: tb={tb} rows={len(rows)}"

    f0, f1 = rows[0]["frame_index"], rows[1]["frame_index"]
    if f0 != 145 or f1 != 200:
        return False, f"帧号错: 7.25s->{f0}(期望145), 10s->{f1}(期望200)"
    if abs(rows[0]["frame_timestamp"] - 1007.25) > 1e-6:
        return False, f"帧时间戳错: {rows[0]['frame_timestamp']} != 1007.25"
    if abs(rows[1]["frame_timestamp"] - 1010.0) > 1e-6:
        return False, f"帧时间戳错: {rows[1]['frame_timestamp']} != 1010.0"
    if tb.frame_index_at(1010.0) != 200 or abs(tb.frame_timestamp(200) - 1010.0) > 1e-9:
        return False, "tb.frame_index_at / frame_timestamp 不一致"
    if abs(rows[1]["osc_velocity"]["vz"] - 0.4) > 1e-9:
        return False, "速度未附着到其后行"
    return True, f"anchor={ANCHOR} fps={FPS} 7.25s->f145 10s->f200; 速度已附着"


def check_2_anchor_invariance(work: Path):
    """② 同一相对时间在不同 anchor 下 → 相同帧号；绝对时间戳按差平移。"""
    a = VideoTimebase.from_record({"fps": FPS, "anchor_monotonic": 1000.0})
    b = VideoTimebase.from_record({"fps": FPS, "anchor_monotonic": 5000.0})
    offs = [0.0, 0.25, 7.25, 33.3, 100.0]
    for off in offs:
        fa = a.frame_index_at(1000.0 + off)
        fb = b.frame_index_at(5000.0 + off)
        if fa != fb:
            return False, f"offset={off}: 帧号不一致 {fa} != {fb}"
        da = b.frame_timestamp(fb) - a.frame_timestamp(fa)
        if abs(da - 4000.0) > 1e-6:
            return False, f"offset={off}: 绝对平移错 {da} != 4000.0"
    return True, f"5 个相对时刻帧号一致；绝对时间戳整体平移 4000s"


def check_3_shared_timebase(work: Path, real_index: Path, *, allow_real: bool):
    """③ 动作时间线与帧索引使用同一时间基准。"""
    # 3a 合成：帧时间戳与帧号必须互逆，且等同 anchor + f/fps
    tb = VideoTimebase.from_record({"fps": FPS, "anchor_monotonic": ANCHOR})
    n = 50
    times = [tb.frame_timestamp(f) for f in range(n)]
    for f in range(n):
        if abs(times[f] - (ANCHOR + f / FPS)) > 1e-9:
            return False, f"synthetic f={f}: {times[f]} != {ANCHOR + f / FPS}"
        if tb.frame_index_at(times[f]) != f:
            return False, f"synthetic f={f}: 帧号不可逆 -> {tb.frame_index_at(times[f])}"

    # 3b 真实帧索引：它的时间轴 `times`（秒）必须落在 src_fps 网格上，且同一个
    #     VideoTimebase 能把该时间轴正反变换而无失真——这正是"同一时间基准"。
    #     注意：`times = idx/src_fps`，idx 是**源帧号**，不一定等于行号，所以
    #     只能验证"网格 + 往返"，不能假设 times[f] == f/fps。
    if not allow_real or not real_index.exists():
        return True, "合成通过；真实帧索引缺失，3b 跳过"
    z = np.load(real_index, allow_pickle=True)
    src_fps = float(z["src_fps"])
    times = z["times"].astype(float)
    tb_real = VideoTimebase.from_record(
        {"fps": src_fps, "anchor_monotonic": float(times[0])})
    if tb_real.fps != src_fps:
        return False, f"真实索引 src_fps 回填失败: {tb_real.fps} != {src_fps}"
    probes = (0, 1, 37, 250, len(times) - 1)
    for p in probes:
        t = float(times[p])
        rt = tb_real.frame_timestamp(tb_real.frame_index_at(t))
        if abs(rt - t) > 1e-6:
            return False, f"真实索引 t={t}: 同基准往返失败 {rt} != {t}"
    # 判别力：故意用错 fps 必须失败（证明该检查不是恒真）
    wrong = VideoTimebase.from_record(
        {"fps": src_fps / 2.0, "anchor_monotonic": float(times[0])})
    if not any(abs(wrong.frame_timestamp(wrong.frame_index_at(float(times[p])))
                   - float(times[p])) > 1e-6 for p in probes):
        return False, "换错 fps 竟然也通过：该检查无判别力"
    # to_record / from_record 往返必须保持同一基准
    rtx = VideoTimebase.from_record(tb_real.to_record())
    if rtx.fps != tb_real.fps or rtx.anchor != tb_real.anchor:
        return False, "to_record/from_record 往返改变了时间基准"
    return True, (f"合成 50 帧互逆；真实索引 src_fps={src_fps} n={len(times)} "
                  f"同基准往返通过，且对错 fps 敏感")


def check_4_time_integration(work: Path):
    """④ 速度按**时间**积分（换 fps 距离按 1/fps 缩放），不按帧数。"""
    n_frames = 200
    d = {}
    for fps in (20.0, 40.0):
        path = work / f"c4_{int(fps)}.jsonl"
        specs = [(f, _action(forward=1.0, vz=1.0, episode="ep0"))
                 for f in range(n_frames)]
        write_log(path, fps, ANCHOR, specs)
        s = episode_action_summary(path, [[0, n_frames]])
        d[fps] = s[0]["osc_forward_distance"]
    # 解析值：n_frames / fps * |v| = 200/20=10.0, 200/40=5.0
    if abs(d[20.0] - 10.0) > 1e-6:
        return False, f"fps=20 距离 {d[20.0]} != 10.0（=200帧/20fps*1.0m/s）"
    if abs(d[40.0] - 5.0) > 1e-6:
        return False, f"fps=40 距离 {d[40.0]} != 5.0（=200帧/40fps*1.0m/s）"
    if abs(d[20.0] / d[40.0] - 2.0) > 1e-6:
        return False, f"换 fps 距离比 {d[20.0] / d[40.0]} != 2.0（说明是按帧数积分）"
    return True, f"fps=20→{d[20.0]}m fps=40→{d[40.0]}m，随 1/fps 缩放（时间积分）"


def check_5_fps_rejected(work: Path):
    """⑤ NaN / ±inf / <=0 / bool 的 fps 必须拒绝。"""
    bads = [float("nan"), float("inf"), float("-inf"), 0, 0.0, -5, -0.1, True, False]
    for bad in bads:
        try:
            VideoTimebase(bad)
            return False, f"VideoTimebase({bad!r}) 未拒绝"
        except ValueError:
            pass
        try:
            ActionTimeline(work / "c5.jsonl", bad)
            return False, f"ActionTimeline(fps={bad!r}) 未拒绝"
        except ValueError:
            pass
    # 坏 fps 的 timebase 头：读取时必须优雅降级为 None，而不是抛错
    p = work / "c5_badheader.jsonl"
    p.write_text(json.dumps({"record": "timebase", "fps": float("nan"),
                             "anchor_monotonic": 0.0}) + "\n", encoding="utf-8")
    tb, rows = load_action_timeline(p)
    if tb is not None:
        return False, "NaN fps 的时间基准头未被降级为 None"
    return True, f"{len(bads)} 个非法 fps 全部拒绝；坏头部降级为 None"


def check_6_bad_lines(work: Path):
    """⑥ 坏 JSONL 行跳过但不破坏后续读取。"""
    path = work / "c6.jsonl"
    write_log(path, FPS, ANCHOR, [
        (0, _action(forward=1.0, vz=1.0, episode="ep0")),
        (10, _action(forward=1.0, vz=1.0, episode="ep0")),
    ])
    good = path.read_text(encoding="utf-8").splitlines()
    # 注意：`{"record": "action"}` 是**合法**动作行，不能当坏行用；用别的 record 名。
    polluted = [good[0], "{not json", good[1], "[1,2,3]", "", "null",
                "123", good[2], "   ", "{\"record\": \"heartbeat\"}"]
    path.write_text("\n".join(polluted) + "\n", encoding="utf-8")
    tb, rows = load_action_timeline(path)
    if tb is None:
        return False, "坏行破坏了时间基准头读取"
    if len(rows) != 2:
        return False, f"坏行过滤后动作行数 {len(rows)} != 2"
    s = episode_action_summary(path, [[0, 20]])
    if s[0]["n_records"] != 2:
        return False, f"坏行之后汇总失败: n_records={s[0]['n_records']} != 2"
    return True, "10 行污染中 2 条有效行仍被完整读出，汇总不受影响"


def check_7_insufficient(work: Path):
    """⑦ 缺失动作记录 ⇒ insufficient_action_records；无日志 ⇒ 显式降级。"""
    path = work / "c7.jsonl"
    write_log(path, FPS, ANCHOR, [
        (0, _action(forward=1.0, vz=1.0, episode="ep0")),          # ep0 只有 1 条
        (200, _action(forward=1.0, vz=1.0, episode="ep1")),
        (201, _action(forward=1.0, vz=1.0, episode="ep1")),
        (202, _action(forward=1.0, vz=1.0, episode="ep1")),
    ])
    summaries = episode_action_summary(path, [[0, 200], [200, 210]])
    if summaries[0]["n_records"] != 1 or summaries[1]["n_records"] != 3:
        return False, f"构造数据不符: {summaries[0]['n_records']}, {summaries[1]['n_records']}"

    vs = {1: 0}
    rh = route_history_seed_edges(vs, summaries)
    if not rh["available"] or rh["n_kept"] != 0:
        return False, f"应过滤掉该边: available={rh['available']} kept={rh['n_kept']}"
    reasons = {d["reason"] for d in rh["dropped"]}
    if "insufficient_action_records" not in reasons:
        return False, f"丢弃原因不含 insufficient_action_records: {reasons}"

    for empty in (None, []):
        rh2 = route_history_seed_edges(vs, empty)
        if rh2["available"] is not False or rh2["source"] != "visual_proxy":
            return False, f"无日志未显式降级: {rh2}"
        if rh2["seed_edges"] != vs:
            return False, "无日志时视觉边未原样保留（可能伪造了边）"
    return True, "n_records<2 → insufficient_action_records；无日志 → visual_proxy 原样保留"


def check_8_ack_invariance(work: Path):
    """⑧ driver_ack=none/unknown 不影响路线摘要（闸门不读 ack）。"""
    eps = [[0, 200], [200, 210]]
    vs = {1: 0}
    res = {}
    for ack in ("none", "unknown", "accepted"):
        path = work / f"c8_{ack}.jsonl"
        specs = [(f, _action(forward=1.0, vz=1.0, ack=ack, episode="ep0"))
                 for f in range(200)]
        specs += [(f, _action(forward=1.0, vz=1.0, ack=ack, episode="ep1"))
                  for f in range(200, 205)]
        write_log(path, FPS, ANCHOR, specs)
        res[ack] = episode_action_summary(path, eps)

    ref = res["none"]
    for ack in ("unknown", "accepted"):
        for k in ("n_records", "osc_forward_distance", "forward_frames", "commands_sent"):
            if ref[0][k] != res[ack][0][k] or ref[1][k] != res[ack][1][k]:
                return False, f"ack={ack} 改变了路线字段 {k}"
        if route_history_seed_edges(vs, ref)["seed_edges"] != \
                route_history_seed_edges(vs, res[ack])["seed_edges"]:
            return False, f"ack={ack} 改变了路线闸门输出"

    if ref[0]["n_records"] != 200:
        return False, f"driver_ack=none 的行被误删: n_records={ref[0]['n_records']}"
    # ack 字段本身仍被读（证明不是"根本没读"，而是"读了但不影响路线闸门"）
    if ref[0]["commands_acked"] != 0:
        return False, "none 的 ack 计数应为 0"
    if res["accepted"][0]["commands_acked"] != 200:
        return False, "accepted 的 ack 计数应被读出"
    return True, "none/unknown/accepted 的路线字段与闸门输出一致；ack 仅影响 ack_rate"


def check_9_chain(work: Path, ep_path: Path, idx_path: Path, *, allow_real: bool):
    """⑨ summary → route_history_seed_edges → consensus_assign 字段完整传递。"""
    if not allow_real or not (ep_path.exists() and idx_path.exists()):
        return None, "真实 artifacts 缺失，9 跳过（不做结构伪造）"

    G = build_episode_evidence(ep_path, idx_path, 8.0)
    keys = list(range(int(G["n_ep"])))
    vis = visual_seed_edges(G, keys, 20, 40, 25, 2, 8.0)

    z = np.load(idx_path, allow_pickle=True)
    fps = float(z["src_fps"])
    eps = G["eps"]
    specs = []
    for a, b in np.asarray(eps).tolist():
        a, b = int(a), int(b)
        span = max(1, b - a)
        step = max(1, span // 30)          # 每个 episode 约 30 行，够过 n_records 与距离门
        for f in range(a, b, step):
            specs.append((f, _action(forward=1.0, vz=1.0, episode=f"ep{a}")))
    log = work / "c9.jsonl"
    write_log(log, fps, ANCHOR, specs)

    # 字段契约 1：summary 必须带闸门要读的键
    summaries = load_route_history(log, eps)
    need = {"episode", "n_records", "osc_forward_distance"}
    for s in summaries:
        if not need <= set(s):
            return False, f"summary 缺字段: {need - set(s)}"
    # 字段契约 2：闸门只过滤、不新增，且输出可被共识直接消费
    rh = route_history_seed_edges(vis, summaries)
    if not set(rh["seed_edges"]) <= set(vis):
        return False, "闸门新增了视觉上不存在的边"
    asg, hist, cdet = consensus_assign(G, keys, 20, 40, 25, 2, 8.0,
                                       seed_edges=rh["seed_edges"])
    if not isinstance(asg, dict) or not asg:
        return False, f"consensus_assign 未消费 seed_edges: {type(asg)}"
    return True, (f"episodes={G['n_ep']} visual_seed={len(vis)} "
                  f"kept={rh['n_kept']} iterations={hist} → consensus 正常返回")


CHECKS = [
    ("1 固定fps/anchor/动作时间", check_1_fixed_fps_anchor),
    ("2 anchor 无关性", check_2_anchor_invariance),
    ("3 共享时间基准", check_3_shared_timebase),
    ("4 速度按时间积分", check_4_time_integration),
    ("5 非法fps拒绝", check_5_fps_rejected),
    ("6 坏行跳过", check_6_bad_lines),
    ("7 缺记录=insufficient_action_records", check_7_insufficient),
    ("8 driver_ack 不影响路线摘要", check_8_ack_invariance),
    ("9 字段端到端传递", check_9_chain),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--episodes", type=Path, default=REPO / ".tmp" / "episodes.npz")
    ap.add_argument("--index", type=Path, default=REPO / ".tmp" / "loop_frame_index.npz")
    ap.add_argument("--no-real-artifacts", action="store_true",
                    help="只跑合成检查（跳过 3b / 9），用于快速 CI")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    # Concurrency-safe isolation (2026-09-22):
    #   * default -> a UNIQUE per-run temp dir, removed at exit;
    #   * explicit --workdir -> honored; a dir WE create is removed at exit, a
    #     pre-existing dir is PRESERVED (never delete what we did not create);
    #   * in ALL cases we clear only the harness's OWN artifacts (c*.jsonl), so a
    #     reused dir can never accumulate rows (ActionLogRecorder appends), and
    #     no unrelated user file is ever touched.
    if args.workdir is None:
        (REPO / ".tmp").mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="offline_harness_", dir=REPO / ".tmp"))
        owns = True
    else:
        work = Path(args.workdir)
        owns = not work.exists()
        work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("c*.jsonl"):
        try:
            stale.unlink()
        except OSError:
            pass
    allow_real = not args.no_real_artifacts

    results = []
    for name, fn in CHECKS:
        try:
            if name.startswith("3"):
                ok, detail = fn(work, args.index, allow_real=allow_real)
            elif name.startswith("9"):
                ok, detail = fn(work, args.episodes, args.index, allow_real=allow_real)
            else:
                ok, detail = fn(work)
        except Exception as exc:                       # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"check": name, "ok": ok, "detail": detail})

    passed = sum(1 for r in results if r["ok"] is True)
    failed = sum(1 for r in results if r["ok"] is False)
    skipped = sum(1 for r in results if r["ok"] is None)

    report = {
        "validation_kind": VALIDATION_KIND,
        "disclaimer": DISCLAIMER,
        "fps": FPS,
        "anchor": ANCHOR,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "ok": failed == 0,
        "checks": results,
    }

    print(f"=== {VALIDATION_KIND} ===")
    for d in DISCLAIMER:
        print(f"  {d}")
    print("-" * 60)
    for r in results:
        mark = "PASS" if r["ok"] is True else ("SKIP" if r["ok"] is None else "FAIL")
        print(f"  [{mark}] {r['check']}")
        print(f"         {r['detail']}")
    print("-" * 60)
    print(f"  passed={passed} failed={failed} skipped={skipped} "
          f"=> {'OK' if report['ok'] else 'FAIL'}")

    if args.json_out is not None:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [saved] {args.json_out}")

    # Cleanup: only remove the workdir THIS process created. A pre-existing
    # explicit --workdir (owns == False) is always preserved.
    if owns:
        shutil.rmtree(work, ignore_errors=True)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
