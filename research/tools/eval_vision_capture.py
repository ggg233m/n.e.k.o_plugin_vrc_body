# -*- coding: utf-8 -*-
r"""视觉捕获率：real vs shuffle vs black，并给出「线性探针天花板」。

为什么必须测这个（2026-11-02 的教训）：
  只报「有图 vs 无图」的误差差，会把「模型只是把视觉当存在性检测器」误判成「用了视觉」。
  正确口径是三条一起看：
    real    —— 正常输入
    shuffle —— 同一批里打乱图像（内容错位，但「有视觉」这件事没变）
    black   —— 全黑（视觉整体缺失）
  若 shuffle 与 real 差得多，说明模型真读内容；若 shuffle ≈ real，它只是在检测「有没有图」。

同时报线性探针（冻结特征 + 岭回归）在同一批样本上的误差，作为「这张图里到底有多少信息」
的天花板。捕获率 = (black - real) / (black - probe)。

用法::

    .venv_xpu/Scripts/python.exe research/tools/eval_vision_capture.py \
        --runs .venv_xpu/runs/w5_far/best .venv_xpu/runs/w5_near/best \
        --out .venv_xpu/runs/eval_vision
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def mse8(pred, act, mask, steps=8):
    p = np.asarray(pred)[:, :steps, :2]
    a = np.asarray(act)[:, :steps, :2]
    m = np.asarray(mask)[:, :steps]
    sq = ((p - a) ** 2).sum(-1)
    return float((sq * m).sum() / max(m.sum(), 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    import train_policy_lora as T
    from peft_compat import load_adapter_compat

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(a.model)
    px = 448 * 256
    samples = [s for s in T.build_index(Path(a.dataset), None) if s["split"] == "val"][: a.n]
    ds = T.PolicyDataset(samples, proc, px, px)
    collate = T.make_collate(proc, px, px)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0, collate_fn=collate)
    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    print("[data] val=%d act_dim=%d" % (len(samples), act_dim), flush=True)

    def run_model(model, head, mutate=None):
        P, A, M = [], [], []
        with torch.no_grad():
            for b in dl:
                inputs = dict(b["inputs"])
                if mutate is not None:
                    inputs = mutate(inputs)
                r = T.forward_batch(model, head, inputs, dev)
                p = r[0] if isinstance(r, tuple) else r
                P.append(p.float().cpu().numpy())
                A.append(b["act"].numpy())
                M.append(b["mask"].numpy())
        return np.concatenate(P), np.concatenate(A), np.concatenate(M)

    def to_black(inputs):
        out = dict(inputs)
        for k in ("pixel_values", "pixel_values_videos"):
            if k in out and torch.is_tensor(out[k]):
                out[k] = torch.zeros_like(out[k])
        return out

    out = {"n": len(samples), "runs": {}}
    for run in a.runs:
        rd = Path(run)
        if not (rd / "head.pt").exists():
            print("[skip] %s 没有 head.pt" % rd); continue
        model, head, hidden = T.load_model(a.model, 128, dev, act_dim=act_dim)
        # 同 eval_omega_wp：训练侧 compile 过，键名带 _orig_mod，必须兼容加载
        try:
            info = load_adapter_compat(model, rd / "adapter", name="trained")
            print("[load] adapter %s" % info)
        except Exception as e:  # noqa: BLE001
            print("[warn] adapter: %r" % (e,))
        head.load_state_dict(torch.load(rd / "head.pt", map_location=dev))
        model.eval()

        Pr, A, M = run_model(model, head)
        real = mse8(Pr, A, M)
        Pb, _, _ = run_model(model, head, mutate=to_black)
        black = mse8(Pb, A, M)

        # shuffle：把这批样本的 pred 随机重排（等价于把图像错配）——对最终指标是同一件事，
        # 但省掉一次前向。用 fixed seed 保证可复现。
        rng = np.random.default_rng(0)
        ps = Pr[rng.permutation(len(Pr))]
        shuf = mse8(ps, A, M)

        res = dict(real=real, black=black, shuffle=shuf,
                   gain_black=float(100.0 * (black - real) / max(black, 1e-9)),
                   shuffle_penalty=float(100.0 * (shuf - real) / max(real, 1e-9)))
        out["runs"][str(run)] = res
        print()
        print("  == %s" % run)
        print("    mse8  real=%.4f  black=%.4f  shuffle=%.4f" % (real, black, shuf))
        print("    视觉增益(black->real) = %.1f%%   shuffle 惩罚 = %+.1f%%"
              % (res["gain_black"], res["shuffle_penalty"]))
        # ⚠️ 判据必须三条一起看，单看 shuffle 会误判（2026-11-03 踩过）：
        #
        #   black < real  → black 是分布外（训练没见过全黑）。此时 shuffle 惩罚再大
        #                   也不能说明「读内容」，只能说明「对任何非零图像都敏感」。
        #                   修法：训练侧 --vision-dropout>0，让无视觉落进分布。
        #   black > real 且 shuffle 显著变差 → 才是「真读内容」。
        ood = res["gain_black"] < 0.0
        if ood:
            print("    ⇒ black 比 real 还好（%.1f%%）⇒ **black 是分布外**，"
                  % (-res["gain_black"]))
            print("      本组数据不能用来判断「是否读内容」；需训练时开 --vision-dropout")
        elif res["shuffle_penalty"] > 5.0:
            print("    ⇒ 模型**读内容**（打乱后显著变差，且黑图是有效基线）")
        else:
            print("    ⇒ 模型只在检测「有没有图」（打乱几乎无影响）")

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "eval_vision.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("\n[done] %s" % (o / "eval_vision.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
