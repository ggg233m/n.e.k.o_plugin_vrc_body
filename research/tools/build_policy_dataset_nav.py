# -*- coding: utf-8 -*-
"""把 nav_online 旧格式场次（kf_img + events/osc/hmd）适配成 policy 测试集。

与 ``build_policy_dataset.py`` 的口径相同，区别只有取样点：
旧录制没有 5 Hz 流，只有关键帧，所以样本时刻 = ``events.jsonl`` 里 ``kind=kf`` 的 ``t``，
图像 = ``kf_img/{k:06d}_L.png``；标签 / wp / mask 全部复用同一套函数。
所有样本 ``split="val"``，供 ``eval_policy_lora.py`` 直接评估。

用法::

    .venv\\Scripts\\python.exe research/tools/build_policy_dataset_nav.py \\
        --sess navmesh_recordings/20261031_001200 \\
        --out  G:\\vrc_recording\\vrc_model\\dataset\\m0_holdout_001200 --size 448 256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_policy_dataset import (  # noqa: E402
    DEFAULT_LAG_S, DEFAULT_LOOKAHEAD_M, GAP_SUSPECT_S, HMD_STILL_S, IDLE_SILENCE_S,
    JPEG_Q, K_STEPS, STEP_S, V_STILL, build_dr_path, encode_jpeg, hmd_forward,
    hmd_static, policy_geometry, to_policy_rgb, waypoint_body,
)


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256])
    ap.add_argument("--lag", type=float, default=DEFAULT_LAG_S)
    ap.add_argument("--lookahead", type=float, default=DEFAULT_LOOKAHEAD_M)
    ap.add_argument("--q", type=int, default=JPEG_Q)
    a = ap.parse_args(argv)
    sess = Path(a.sess)
    out = Path(a.out)
    size = (int(a.size[0]), int(a.size[1]))
    meta = json.loads((sess / "meta.json").read_text(encoding="utf-8"))
    sid = sess.name

    # 关键帧时刻 + 图像
    kfs = [(int(r["k"]), float(r["t"])) for r in read_jsonl(sess / "events.jsonl")
           if r.get("kind") == "kf" and (sess / "kf_img" / f"{int(r['k']):06d}_L.png").exists()]
    kfs.sort(key=lambda x: x[1])
    t_kf = np.array([t for _, t in kfs])
    k_kf = [k for k, _ in kfs]

    # nav_online 的 osc.jsonl 是合并格式：{"t": ..., "vx": ..., "vz": ...}
    osc = read_jsonl(sess / "osc.jsonl")
    t_m = np.array([r["t"] for r in osc])
    vx_m = np.array([float(r["vx"]) for r in osc])
    vz_m = np.array([float(r["vz"]) for r in osc])
    vx_t, vz_t = t_m, t_m
    # HMD：新格式 hmd_frames.jsonl（四元数），旧 nav 格式 hmd.jsonl（3×3 旋转矩阵）
    hp = sess / "hmd_frames.jsonl"
    if hp.exists():
        hmd = read_jsonl(hp)
        h_t = np.array([r["t"] for r in hmd])
        quat = np.array([r["hmd"]["rotation_xyzw"] for r in hmd])
    else:
        from scipy.spatial.transform import Rotation
        hmd = read_jsonl(sess / "hmd.jsonl")
        h_t = np.array([r["t"] for r in hmd])
        R = np.array([r["R"] for r in hmd], dtype=np.float64)
        quat = Rotation.from_matrix(R).as_quat()  # x, y, z, w
    fwd = hmd_forward(quat)
    streams = dict(t_m=t_m, vx_m=vx_m, vz_m=vz_m, h_t=h_t, quat=quat, fwd=fwd,
                   vx_t=vx_t, vz_t=vz_t)
    path = build_dr_path(streams)

    def zoh(tq: float):
        i = int(np.searchsorted(t_m, tq, side="right") - 1)
        if i < 0:
            return None
        return float(streams["vx_m"][i]), float(streams["vz_m"][i]), float(tq - t_m[i])

    first_osc = float(min(vx_t[0], vz_t[0]))
    last_hmd = float(h_t[-1])
    samples, dropped = [], dict(preosc=0, idle=0, gap=0, n=0, noimg=0)
    for k, t in kfs:
        q0 = t + a.lag
        if q0 < first_osc:
            dropped["preosc"] += 1; continue
        z0 = zoh(t)
        if z0 is not None and z0[2] > IDLE_SILENCE_S and hmd_static(t, streams):
            dropped["idle"] += 1; continue
        act, mask = [], []
        for step in range(K_STEPS):
            q = q0 + step * STEP_S
            z = zoh(q)
            if q > last_hmd or z is None or (z[2] > GAP_SUSPECT_S and np.hypot(z[0], z[1]) > V_STILL):
                act.append([0.0, 0.0]); mask.append(0)
            else:
                act.append([round(z[0], 4), round(z[1], 4)]); mask.append(1)
        valid = [i for i in range(K_STEPS) if mask[i]]
        if not valid:
            continue
        cls = ("move" if mask[0] and np.hypot(*act[0]) > V_STILL
               else "idle" if all(np.hypot(*act[i]) <= V_STILL for i in valid) else "trans")
        hi = int(np.clip(np.searchsorted(h_t, t, side="right") - 1, 0, len(h_t) - 1))
        fx, fz = fwd[hi]
        wp = waypoint_body(t, path, np.array([fx, fz]), a.lookahead)
        # 0.4 s 前最近关键帧
        j = int(np.searchsorted(t_kf, t - 0.4, side="right") - 1)
        cand = [i for i in (j, j + 1) if 0 <= i < len(t_kf)]
        prev_k = k
        if cand:
            best = min(cand, key=lambda i: abs(t_kf[i] - (t - 0.4)))
            if abs(t_kf[best] - (t - 0.4)) <= 0.15:
                prev_k = k_kf[best]
        samples.append(dict(k=k, t=t, act=act, mask=mask, cls=cls,
                            wp=[0.0, 0.0] if wp is None else wp, prev_k=prev_k))
        dropped["n"] += 1

    out = Path(a.out) / sid  # 与 build_policy_dataset 一致：<dataset>/<sid>/...
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    need = sorted({s["k"] for s in samples} | {s["prev_k"] for s in samples})
    for k in need:
        png = sess / "kf_img" / f"{k:06d}_L.png"
        rgb = cv2.cvtColor(cv2.imread(str(png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        (out / "images" / f"{k:06d}_L.jpg").write_bytes(encode_jpeg(to_policy_rgb(rgb, size), a.q))
    t0 = samples[0]["t"] if samples else 0.0
    with (out / "policy_obs.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"t": s["t"], "t_rel": round(s["t"] - t0, 6), "i": s["k"],
                                "img_L": f"images/{s['k']:06d}_L.jpg", "prev": f"images/{s['prev_k']:06d}_L.jpg",
                                "sid": sid, "world": "", "split": "val"}, ensure_ascii=False) + "\n")
    with (out / "policy_labels.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"t": s["t"], "act": s["act"], "mask": s["mask"], "cls": s["cls"],
                                "wp": s["wp"], "split": "val"}, ensure_ascii=False) + "\n")
    stats = dict(sid=sid, kf=len(kfs), samples=len(samples), size=list(size),
                 world_scale=meta.get("config", {}).get("world_scale"), dropped=dropped,
                 cls={c: sum(1 for s in samples if s["cls"] == c) for c in ("move", "trans", "idle")})
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
