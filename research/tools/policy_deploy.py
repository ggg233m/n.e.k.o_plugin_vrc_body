# -*- coding: utf-8 -*-
"""VLA 策略的部署 manifest / preflight / replay（2026-11-05）。

为什么需要它：
  现有运行时（backend/policy_head.py + research/tools/policy_{runtime,runner,executor}.py）
  的配置方式是一组 `VRC_POLICY_*` **环境变量**，没有任何落盘物记录「这次部署到底用了
  哪个 checkpoint、什么口径、什么标定」。训练侧我们已经两次被「参数漏传 → 静默错口径」
  咬过（omega_src_hz、omega_agg），推理侧同理：
    * 喂错 schema（drop_wp=False）形状对得上、不报错，只是分布错；
    * omega 符号反了，模型每次转向都朝反方向，而日志里数字「看起来正常」。

本工具**不新建第二套运行时**，只做三件事：
  make      从训练 run 生成 deployment manifest（含 SHA256 / schema / 标定 / 数据口径）
  preflight 纯只读校验：文件、hash、schema、维度、omega 符号、安全参数、设备；
            任何一项不满足**以非零码退出**，绝不「带病启动」。
  replay    用已有录制的图像跑通 PolicyRuntime + SafetyGate + omega 换算，
            验证 3 通道输出 / omega 右正 / 限幅 / stale 归零 / ESTOP，
            **不启动 VRChat**。

⚠️ replay 通过 ≠ 真机通过。它验证的是「模型与安全链路」，不是「VRChat 闭环」。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

K_STEPS = 12
STEP_S = 0.1

# 部署必须满足的口径。任何一个不符就拒绝上线 —— 这些正是「形状对但语义错」的那些字段。
REQUIRED_SCHEMA = {
    "act_dim": 3,
    "steps": K_STEPS,
    "step_s": STEP_S,
    "omega_unit": "rad_s",
    "history_frame_s": 0.2,
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_adapter_file(run: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = run / "adapter" / name
        if p.exists():
            return p
    return None


class Fail(Exception):
    pass


def _ck(cond, msg, fails):
    """记录一条检查结果。cond 为假时进 fails 列表，不抛（一次报全）。"""
    if not cond:
        fails.append(msg)
    return bool(cond)


# ------------------------------------------------------------------ make
def cmd_make(a) -> int:
    run = Path(a.run).resolve()
    model = Path(a.model).resolve()
    if not run.exists():
        print("[make] run 不存在: %s" % run, file=sys.stderr)
        return 2
    head = run / "head.pt"
    schema_p = run / "input_schema.json"
    adapter = find_adapter_file(run)
    missing = [str(p) for p in (head, schema_p, adapter) if p is None or not Path(p).exists()]
    if missing:
        print("[make] 缺文件: %s" % missing, file=sys.stderr)
        return 2

    schema = json.loads(schema_p.read_text(encoding="utf-8"))
    args_p = run.parent / "args.json"
    train_args = json.loads(args_p.read_text(encoding="utf-8")) if args_p.exists() else {}

    ds = train_args.get("dataset")
    ds_manifest = None
    if ds:
        mp = Path(ds) / "manifest.json"
        if mp.exists():
            ds_manifest = json.loads(mp.read_text(encoding="utf-8"))

    man = {
        "kind": "vrc_policy_deployment",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": ("由 policy_deploy.py make 生成。真机未验证；GameScaling 仅通用 3D 先验。"),
        "model_dir": str(model),
        "run_dir": str(run),
        "files": {
            "head": {"path": str(head), "sha256": sha256(head), "bytes": head.stat().st_size},
            "adapter": {"path": str(adapter), "sha256": sha256(adapter),
                        "bytes": adapter.stat().st_size},
            "input_schema": {"path": str(schema_p), "sha256": sha256(schema_p)},
        },
        "schema": schema,
        "training": {
            "dataset": ds,
            "dataset_omega": (ds_manifest or {}).get("omega"),
            "dataset_wp_mode": (ds_manifest or {}).get("wp_mode"),
            "seed": train_args.get("seed"),
            "epochs": train_args.get("epochs"),
            "batch": train_args.get("batch"),
            "lora_r": train_args.get("lora_r"),
            "lr_head": train_args.get("lr_head"),
            "drop_wp": train_args.get("drop_wp"),
            "omega_clip": train_args.get("omega_clip"),
        },
        "calibration": {
            # 这些是**角色/世界标定**，不是模型参数。必须人工确认后写死。
            "v_max_mps": a.v_max,
            "dv_max_mps2": a.dv_max,
            "w_max_rad_s": a.w_max,
            "dw_max_rad_s2": a.dw_max,
            "omega_sign": "right_positive",
            "turn_channel_sign": "service_left_positive（omega_to_delta_deg 内部反号）",
            # ⚠️ 2026-11-06 修订：wp≈0 **不再是**停车条件。
            # 旧行为（2026-11-05 审计）是 `gated = hypot(wp) < wp_stop → 整块归零`，
            # 结果是「未定位/地图外/自由探索」时策略被外部静默停死，与「已到终点」
            # 不可区分。新行为：wp≈0 只表示「没有路径先验」，模型照常出块；
            # 停止只来自 policy_modes.decide_stop 的五种来源。
            "wp_stop_m": a.wp_stop,
            "wp_stop_semantics": "annotation_only_wp_never_stops",
            "wp_stop_zero_means": ["没有路径先验（A* 未请求/失效/未定位）",
                                   "自由探索或纯局部提示词任务",
                                   "模型应更多依赖画面，而不是被停车"],
            "stop_sources": ["estop", "stale", "safety_gate", "nav_state", "stop_head"],
            "saturation_measured": None,  # 下面填
        },
        "runtime_env": {
            "VRC_POLICY_HEAD": "1",
            "VRC_POLICY_MODEL": str(model),
            "VRC_POLICY_RUN": str(run),
            "VRC_POLICY_DRYRUN": "1",
            "VRC_POLICY_HZ": str(a.hz),
            "VRC_POLICY_VMAX": str(a.v_max),
            "VRC_POLICY_STALE": str(a.stale_s),
            "VRC_POLICY_FLOOR": str(a.floor_frac),
            "VRC_POLICY_GRAPH": "0",
        },
    }

    # 饱和率：把「选定 w_max 下多大比例的标签会被打到满舵」记录下来。
    # 这是部署决策的**可审计依据** —— peak 模型比 mean 更饱和，不能凭感觉选 1.57。
    sat = None
    if ds:
        try:
            import numpy as np
            import train_policy_lora as T
            idx = [x for x in T.build_index(Path(ds), None) if x["split"] == "val"]
            om = []
            for x in idx:
                act = np.asarray(x["act"])[:8, 2]
                msk = np.asarray(x["mask"])[:8]
                om.append(np.abs(act[msk > 0]))
            om = np.concatenate(om)
            sat = dict(n=int(om.size),
                       p50=round(float(np.percentile(om, 50)), 4),
                       p95=round(float(np.percentile(om, 95)), 4),
                       max=round(float(om.max()), 4),
                       sat_pct_at_w_max=round(float(100 * (om > a.w_max).mean()), 2))
        except Exception as e:  # noqa: BLE001 - 统计失败不阻止生成 manifest
            sat = {"error": repr(e)}
    man["calibration"]["saturation_measured"] = sat

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[make] wrote %s" % out)
    print("[make] head   sha256=%s" % man["files"]["head"]["sha256"][:24])
    print("[make] adapter sha256=%s" % man["files"]["adapter"]["sha256"][:24])
    if sat and "sat_pct_at_w_max" in sat:
        print("[make] omega 饱和率 @ w_max=%.3f → %.2f%%（p95=%.2f rad/s）"
              % (a.w_max, sat["sat_pct_at_w_max"], sat["p95"]))
    return 0


# ------------------------------------------------------------------ preflight
def cmd_preflight(a) -> int:
    """纯只读校验。任何一项不过 → 非零退出。不加载模型（快、无副作用）。"""
    fails = []
    warns = []
    rows = []

    def row(name, ok, detail):
        rows.append((name, "PASS" if ok else "FAIL", detail))
        return ok

    man_path = Path(a.manifest)
    if not man_path.exists():
        print("[preflight] 找不到 manifest: %s" % man_path, file=sys.stderr)
        return 2
    man = json.loads(man_path.read_text(encoding="utf-8"))
    print("[preflight] manifest=%s" % man_path)
    print("[preflight] run=%s" % man.get("run_dir"))

    # 1) 文件存在 + hash 一致
    for key in ("head", "adapter", "input_schema"):
        ent = (man.get("files") or {}).get(key) or {}
        p = Path(ent.get("path") or "")
        exists = p.exists()
        if not _ck(exists, "%s 文件不存在: %s" % (key, p), fails):
            row("file:" + key, False, "missing")
            continue
        want = ent.get("sha256")
        got = sha256(p)
        ok = (want == got)
        if not _ck(ok, "%s SHA256 不匹配（manifest=%s… 实际=%s…）"
                   % (key, str(want)[:16], got[:16]), fails):
            row("sha:" + key, False, "mismatch")
        else:
            row("sha:" + key, True, got[:16])

    # 2) 模型目录
    model_dir = Path(man.get("model_dir") or "")
    row("model_dir", _ck(model_dir.is_dir(), "model_dir 不存在: %s" % model_dir, fails),
        str(model_dir))

    # 3) schema 口径（这些是「形状对但语义错」的字段）
    schema = man.get("schema") or {}
    for k, want in REQUIRED_SCHEMA.items():
        got = schema.get(k)
        ok = _ck(got == want, "schema.%s 期望 %r 实际 %r" % (k, want, got), fails)
        row("schema:" + k, ok, "%r" % (got,))
    # drop_wp 不强制某个值，但必须**显式存在**（缺失会退化成 False，静默换口径）
    row("schema:drop_wp", _ck("drop_wp" in schema, "schema 缺 drop_wp 字段", fails),
        "%r" % (schema.get("drop_wp"),))
    row("schema:drop_topo", _ck("drop_topo" in schema, "schema 缺 drop_topo 字段", fails),
        "%r" % (schema.get("drop_topo"),))

    # 4) head 输出维度 = K_STEPS × act_dim（不解包整个模型，只读 state_dict）
    head_p = Path(((man.get("files") or {}).get("head") or {}).get("path") or "")
    if head_p.exists():
        try:
            import torch
            sd = torch.load(head_p, map_location="cpu")
            out_dim = int(sd["fc.weight"].shape[0])
            act_dim = int(schema.get("act_dim", 0))
            ok = _ck(out_dim == K_STEPS * act_dim,
                     "head 输出 %d != K_STEPS(%d) × act_dim(%d)" % (out_dim, K_STEPS, act_dim),
                     fails)
            row("head_dim", ok, "%d = %d × %d" % (out_dim, K_STEPS, act_dim))
            del sd
        except Exception as e:  # noqa: BLE001
            _ck(False, "head.pt 读取失败: %r" % (e,), fails)
            row("head_dim", False, repr(e))

    # 5) omega 符号约定：右正 → omega_to_turn>0，omega_to_delta_deg<0（service 左正）
    try:
        from policy_executor import omega_to_turn, omega_to_delta_deg
        cal = man.get("calibration") or {}
        w_max = float(cal.get("w_max_rad_s", 0.0))
        _ck(w_max > 0, "calibration.w_max_rad_s 必须 >0，实际 %r" % (w_max,), fails)
        t_pos = omega_to_turn(+1.0, w_max)
        t_neg = omega_to_turn(-1.0, w_max)
        ok1 = _ck(t_pos > 0 > t_neg,
                  "omega_to_turn 方向错：+1→%.4f, -1→%.4f（应右正）" % (t_pos, t_neg), fails)
        row("omega:turn_sign", ok1, "+1→%.3f, -1→%.3f" % (t_pos, t_neg))
        d_pos = omega_to_delta_deg(+1.0, w_max, 100)
        d_neg = omega_to_delta_deg(-1.0, w_max, 100)
        ok2 = _ck(d_pos < 0 < d_neg,
                  "omega_to_delta_deg 未反号：+1→%.3f, -1→%.3f（service 需左正）"
                  % (d_pos, d_neg), fails)
        row("omega:turn_deg_sign", ok2, "+1→%.3f°, -1→%.3f°" % (d_pos, d_neg))
    except Exception as e:  # noqa: BLE001
        _ck(False, "omega 换算导入失败: %r" % (e,), fails)
        row("omega:sign", False, repr(e))

    # 6) 安全参数
    cal = man.get("calibration") or {}
    for k, must_pos in (("v_max_mps", True), ("dv_max_mps2", True),
                        ("w_max_rad_s", True), ("dw_max_rad_s2", True)):
        v = cal.get(k)
        ok = isinstance(v, (int, float)) and (v > 0 if must_pos else True)
        row("safety:" + k, _ck(ok, "calibration.%s 非法: %r" % (k, v), fails), "%r" % (v,))
    # ⚠️ 停车语义必须被**显式声明**（2026-11-05 审计，2026-11-06 改口径）。
    # 这条防的不是「代码写错」，而是「部署者按旧口径理解」：旧代码里 wp=(0,0)
    # （未定位/地图外的兜底值）会整块归零；现在它只表示「没有路径先验」，
    # 停与不停由 policy_modes.decide_stop 的六个来源决定。
    sem = cal.get("wp_stop_semantics")
    ok_sem = _ck(sem == "annotation_only_wp_never_stops",
                 "calibration.wp_stop_semantics 缺失或非预期（%r）；"
                 "当前口径必须是 annotation_only_wp_never_stops"
                 "（wp≈0 只表示无路径先验，不置零动作）" % (sem,), fails)
    row("safety:wp_stop_semantics", ok_sem, "%r" % (sem,))
    wps = cal.get("wp_stop_m")
    ok_wps = _ck(isinstance(wps, (int, float)) and float(wps) >= 0.0,
                 "calibration.wp_stop_m 非法: %r" % (wps,), fails)
    row("safety:wp_stop_m", ok_wps, "%r" % (wps,))
    # 定位安全门必须独立声明：部署者要能一眼看出「未定位会停」这条来自
    # state+pose_age，而不是来自 wp。缺了它就会出现「以为到点门兜底、其实没有」。
    loc = cal.get("localization_gate") or {}
    ok_loc = _ck(bool(loc) and bool(loc.get("independent_of_wp", False)),
                 "calibration.localization_gate 缺失或未声明 independent_of_wp=True"
                 "（未定位/位姿过期必须由 state+pose_age 独立判定）", fails)
    row("safety:localization_gate", ok_loc, "%r" % (loc.get("source"),))

    env = man.get("runtime_env") or {}
    stale = env.get("VRC_POLICY_STALE")
    row("safety:stale_s", _ck(stale is not None and float(stale) > 0,
                             "VRC_POLICY_STALE 缺失或非正: %r" % (stale,), fails),
        "%r" % (stale,))

    # 7) 设备
    try:
        import torch
        xpu = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
        row("device", True, "xpu" if xpu else "cpu(未检测到 XPU，回退 CPU)")
        if not xpu:
            warns.append("未检测到 XPU，将回退 CPU（延迟会显著变差）")
    except Exception as e:  # noqa: BLE001
        _ck(False, "torch 导入失败: %r" % (e,), fails)
        row("device", False, repr(e))

    # 8) 目录可写（executor 要写 STOP 文件与 policy_live.jsonl）
    rp = Path(man.get("run_dir") or ".")
    row("writable:run_dir", _ck(os.access(rp, os.W_OK), "run_dir 不可写: %s" % rp, fails), str(rp))

    # 9) 显式声明：preflight 不需要 VRChat
    row("vrchat", True, "preflight 不需要 VRChat（未启动）")

    # 输出
    print()
    print("  %-26s %-6s %s" % ("check", "state", "detail"))
    print("  " + "-" * 68)
    for n, st, d in rows:
        print("  %-26s %-6s %s" % (n, st, d))
    print()
    if warns:
        for w in warns:
            print("  [WARN] %s" % w)
    if fails:
        print("  [FAIL] %d 项未通过：" % len(fails))
        for m in fails:
            print("    - %s" % m)
        out = Path(a.out) if a.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(dict(ok=False, fails=fails, warns=warns,
                                        rows=rows), indent=1, ensure_ascii=False), encoding="utf-8")
        return 3
    print("  [OK] preflight 全部通过（%d 项）" % len(rows))
    out = Path(a.out) if a.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(ok=True, fails=[], warns=warns, rows=rows),
                                    indent=1, ensure_ascii=False), encoding="utf-8")
        print("  wrote %s" % out)
    return 0


# ------------------------------------------------------------------ env
def cmd_env(a) -> int:
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    env = man.get("runtime_env") or {}
    if a.dry_run:
        env["VRC_POLICY_DRYRUN"] = "1"
    for k, v in env.items():
        print("$env:%s = %r" % (k, str(v)))
    return 0



# ------------------------------------------------------------------ replay
def cmd_replay(a) -> int:
    """用已有录制的图像回放，验证模型与安全链路。**不启动 VRChat。**

    验证项（每项都会写进产物 json）：
      1. 模型能加载（含 adapter 与 schema 一致性）
      2. 输出 3 通道（vx, vz, omega）
      3. omega 右正：omega_to_turn(+x)>0，omega_to_delta_deg(+x)<0（service 左正）
      4. 限幅生效：|v|<=v_max、|w|<=w_max、|dw|<=dw_max
      5. stale/gated 归零
      6. ESTOP 的 STOP 文件可建可删
    """
    import numpy as np
    from PIL import Image
    from policy_runtime import PolicyRuntime
    from policy_executor import SafetyGate, omega_to_turn, omega_to_delta_deg

    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    cal = man.get("calibration") or {}
    v_max = float(cal.get("v_max_mps", 1.2))
    dv_max = float(cal.get("dv_max_mps2", 0.5))
    w_max = float(cal.get("w_max_rad_s", 1.57))
    dw_max = float(cal.get("dw_max_rad_s2", 3.14))

    print("[replay] 载入 runtime ...", flush=True)
    # 2026-11-06：wp_stop 不再置零动作（旧代码 `gated = hypot(wp) < wp_stop` 会
    # 整块归零；causal 数据集的 wp 恒为 (0,0)，那样测出来全是 0 块）。
    # 现在它只是一个日志注记阈值；这里仍传 0 以保持 replay 数字与历史可比。
    rt = PolicyRuntime(man["model_dir"], man["run_dir"],
                       size=(448, 256), max_len=a.max_len, compile=False,
                       wp_stop=0.0)
    print("[replay] runtime ok: act_dim=%d drop_wp=%s drop_topo=%s"
          % (rt.act_dim, rt.drop_wp, rt.drop_topo), flush=True)

    import train_policy_lora as T
    ds = Path((man.get("training") or {}).get("dataset"))
    if not ds.exists():
        print("[replay] 数据集不存在: %s" % ds, file=sys.stderr)
        return 2
    samples = [x for x in T.build_index(ds, None) if x["split"] == "val"][:a.n]
    if not samples:
        print("[replay] 没有可用 val 样本", file=sys.stderr)
        return 2

    gate = SafetyGate(v_max=v_max, dv_max=dv_max, w_max=w_max, dw_max=dw_max)
    lat, raws, gats = [], [], []
    for i, s in enumerate(samples):
        cur = np.asarray(Image.open(s["img"]).convert("RGB"))
        prev = np.asarray(Image.open(s["prev_img"]).convert("RGB"))
        out = rt.act(cur, prev, s["wp"], s["body"], preprocessed=True)
        blk = np.asarray(out["block"], dtype=np.float64).reshape(-1, 3)
        lat.append(float(out["latency_ms"]))
        raws.append(blk)
        gats.append(gate.apply(blk, False, False, 0.0))
        if i == 0 and a.verbose:
            print("[replay] first block[0]=%s" % (blk[0].tolist(),), flush=True)

    raw = np.asarray(raws)
    gat = np.asarray(gats)
    om = gat[..., 2]

    c = {}
    c["channels_are_3"] = bool(raw.shape[-1] == 3)
    c["act_dim_is_3"] = bool(rt.act_dim == 3)
    c["v_clamped"] = bool(np.linalg.norm(gat[..., :2], axis=-1).max() <= v_max + 1e-6)
    c["w_clamped"] = bool(np.abs(om).max() <= w_max + 1e-6)
    c["dw_limited"] = bool(np.abs(np.diff(om, axis=1)).max() <= dw_max + 1e-6)
    c["omega_has_variance"] = bool(float(np.std(om)) > 1e-6)
    c["omega_turn_right_positive"] = bool(omega_to_turn(0.5 * w_max, w_max) > 0)
    c["omega_deg_negated"] = bool(omega_to_delta_deg(0.5 * w_max, w_max, 100) < 0)
    c["stale_zeroes_block"] = bool(np.allclose(gate.apply(np.ones((12, 3)), True, False, 0.0), 0.0))
    c["gated_zeroes_block"] = bool(np.allclose(gate.apply(np.ones((12, 3)), False, True, 0.0), 0.0))
    # wp≈0 契约（2026-11-06 反转）：wp=(0,0) **不得**置零动作，也不得判成停止。
    # 旧版本在这里验证「roi 到点门 → 整块归零」，那个行为正是本轮要修掉的：
    # 未定位/地图外/自由探索时 wp 兜底为 (0,0)，旧逻辑会让策略永久静默不动。
    try:
        _rt_probe = PolicyRuntime(man["model_dir"], man["run_dir"],
                                  size=(448, 256), max_len=a.max_len, compile=False,
                                  wp_stop=0.02)
        _s = samples[0]
        _cur = np.asarray(Image.open(_s["img"]).convert("RGB"))
        _prev = np.asarray(Image.open(_s["prev_img"]).convert("RGB"))
        _o = _rt_probe.act(_cur, _prev, (0.0, 0.0), _s["body"], preprocessed=True)
        c["wp_near_zero_does_not_stop"] = bool(
            not _o["gated"] and not _o["stop"]
            and bool((_o.get("stop_detail") or {}).get("wp_near_zero")))
        # 同一图、同一 body，只把 wp_stop 改成 0：块必须逐位相同，
        # 证明这个阈值已经与动作输出完全解耦（只剩日志注记）。
        _rt_probe.wp_stop = 0.0
        _o2 = _rt_probe.act(_cur, _prev, (0.0, 0.0), _s["body"], preprocessed=True)
        c["wp_stop_threshold_never_changes_block"] = bool(
            np.allclose(np.asarray(_o["block"]), np.asarray(_o2["block"])))
        del _rt_probe
    except Exception as _e:  # noqa: BLE001
        c["wp_near_zero_does_not_stop"] = False
        c["wp_stop_threshold_never_changes_block"] = False

    stop_p = Path(a.stop_probe)
    try:
        stop_p.parent.mkdir(parents=True, exist_ok=True)
        stop_p.write_text("probe", encoding="utf-8")
        c["estop_file_creatable"] = stop_p.exists()
    finally:
        if stop_p.exists():
            stop_p.unlink()
    c["estop_file_removable"] = not stop_p.exists()

    res = dict(
        n=len(samples),
        checks=c,
        summary=dict(
            latency_p50=float(np.percentile(lat, 50)),
            latency_p95=float(np.percentile(lat, 95)),
            latency_max=float(max(lat)),
            omega_abs_mean=float(np.abs(om).mean()),
            omega_abs_max=float(np.abs(om).max()),
            sat_rate_pct=float(100.0 * (np.abs(om) >= w_max - 1e-9).mean()),
        ),
        omega_sample=om[:3].tolist(),
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    print()
    print("  %-30s %s" % ("check", "result"))
    print("  " + "-" * 46)
    bad = 0
    for k, v in c.items():
        print("  %-30s %s" % (k, "PASS" if v else "FAIL"))
        if not v:
            bad += 1
    print()
    print("  latency p50=%.1f ms p95=%.1f ms max=%.1f ms"
          % (res["summary"]["latency_p50"], res["summary"]["latency_p95"], res["summary"]["latency_max"]))
    print("  omega |mean|=%.4f |max|=%.4f 饱和率=%.2f%%"
          % (res["summary"]["omega_abs_mean"], res["summary"]["omega_abs_max"],
             res["summary"]["sat_rate_pct"]))
    print("  wrote %s" % out)
    if bad:
        print()
        print("  [FAIL] %d 项检查未通过" % bad, file=sys.stderr)
        return 3
    print()
    print("  [OK] replay 全部通过 —— 但**不等于**真机通过：")
    print("       未验证 VRChat/OSC 发送、WGC 采集、真实延迟闭环")
    return 0


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VLA 策略部署 manifest / preflight / replay")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="从训练 run 生成 deployment manifest")
    m.add_argument("--run", required=True)
    m.add_argument("--model", default=r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3")
    m.add_argument("--out", required=True)
    m.add_argument("--v-max", type=float, default=1.2)
    m.add_argument("--dv-max", type=float, default=0.5)
    m.add_argument("--w-max", type=float, default=1.57)
    m.add_argument("--dw-max", type=float, default=3.14)
    m.add_argument("--hz", type=float, default=2.0)
    m.add_argument("--stale-s", type=float, default=0.8)
    m.add_argument("--floor-frac", type=float, default=0.6)
    m.add_argument("--wp-stop", type=float, default=0.02,
                   help="到点门阈值（米）；wp 模长小于它则整块归零")
    m.set_defaults(fn=cmd_make)

    p = sub.add_parser("preflight", help="只读校验，失败非零退出")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_preflight)

    e = sub.add_parser("env", help="打印可用的 VRC_POLICY_* 环境变量")
    e.add_argument("--manifest", required=True)
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(fn=cmd_env)

    r = sub.add_parser("replay", help="用录制图像回放验证（不启动 VRChat）")
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=8)
    r.add_argument("--max-len", type=int, default=192)
    r.add_argument("--stop-probe", default=".venv_xpu/runs/deploy/STOP_probe")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(fn=cmd_replay)

    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
