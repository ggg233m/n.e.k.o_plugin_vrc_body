# -*- coding: utf-8 -*-
r"""ω + wp 去泄漏 的评测：三个数一次出。

要回答的三件事（2026-11-03）：

  1. **omega8**：ω 通道回归得怎么样（rad/s 口径的 MSE）；
  2. **方向角分布是否松开**：机体系动作方向角的分布。之前是 p50=0°、|角|>20° 只占 2.92%
     （「几乎从不拐弯」）。ω 放出来后如果仍钉在 0°，说明模型没接管方向；
     如果松开、甚至出现双峰，说明该上「模式头」。
  3. **视觉捕获率**：同一批样本上 real vs shuffle vs black 的 mse8，
     与「线性探针能达到的天花板」比。之前是 25.0% 可用、模型只吃到 6.2%。

用法::

    .venv_xpu/Scripts/python.exe research/tools/eval_omega_wp.py \
        --runs .venv_xpu/runs/w5_far/best .venv_xpu/runs/w5_near/best \
        --out .venv_xpu/runs/eval_omega
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

K_STEPS = 12


def pct(x, p):
    return float(np.percentile(np.asarray(x, float), p))


def dir_stats(pred: np.ndarray, mask: np.ndarray) -> dict:
    """机体系动作方向角（度）。只统计速度非零且 mask 有效的步。"""
    v = pred[..., :2]
    m = mask > 0
    spd = np.linalg.norm(v, axis=-1)
    ok = m & (spd > 0.1)
    if ok.sum() < 10:
        return dict(n=0)
    # 机体系：x=右, z=前 → 相对正前方的偏角，右正
    ang = np.degrees(np.arctan2(v[..., 0][ok], v[..., 1][ok]))
    return dict(n=int(ok.sum()),
                p05=pct(ang, 5), p25=pct(ang, 25), p50=pct(ang, 50),
                p75=pct(ang, 75), p95=pct(ang, 95),
                abs_mean=float(np.mean(np.abs(ang))),
                gt20=float(np.mean(np.abs(ang) > 20)),
                gt45=float(np.mean(np.abs(ang) > 45)),
                std=float(np.std(ang)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--n", type=int, default=400, help="快速子集条数（val 前 N）")
    ap.add_argument("--full", action="store_true", default=True,
                    help="同时报「val 全部」口径（默认开）。两组差很大，必须都报")
    ap.add_argument("--no-full", dest="full", action="store_false")
    ap.add_argument("--numeric-bypass", action="store_true",
                    help="被评模型带数值旁路时必须开，否则 head 维度不符")
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

    all_val = [s for s in T.build_index(Path(a.dataset), None) if s["split"] == "val"]
    # 选样必须与 probe_ridge_masked 完全一致，否则跨模型对比无意义
    samples = T.select_val_subset(Path(a.dataset), a.n)
    if not samples:
        print("[data] 没有 val 样本"); return 2
    dss = {
        "前%d" % min(a.n, len(samples)): T.PolicyDataset(samples, proc, px, px),
    }
    if a.full and len(all_val) > len(samples):
        dss["全部%d" % len(all_val)] = T.PolicyDataset(all_val, proc, px, px)
    collate = T.make_collate(proc, px, px)
    dls = {k: DataLoader(v, batch_size=a.batch, shuffle=False, num_workers=0,
                         collate_fn=collate) for k, v in dss.items()}
    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    print("[data] val=%d(全部) %d(子集) act_dim=%d proto=%s"
          % (len(all_val), len(samples), act_dim, list(dls)), flush=True)

    out = {"n_all": len(all_val), "n_sub": len(samples),
           "act_dim": act_dim, "runs": {}}
    for run in a.runs:
        rd = Path(run)
        if not rd.exists():
            print("[skip] %s 不存在" % rd); continue
        meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
        model, head, hidden = T.load_model(a.model, 128, dev, act_dim=act_dim)
        # ⚠️ 必须用兼容加载器：训练侧对内层模型做了 torch.compile，存下来的键名带
        # `_orig_mod`；直接 load_adapter 会把 372 个键全部静默跳过，评测出来的模型
        # 等于没训练（实测 mse8 从 1.78 变成 11.5、adapter 前后输出差 0.0）。
        try:
            info = load_adapter_compat(model, rd / "adapter", name="trained")
            print("[load] adapter %s" % info)
        except Exception as e:  # noqa: BLE001
            print("[warn] %s adapter: %r" % (rd, e))
        sd = torch.load(rd / "head.pt", map_location=dev)
        try:
            head.load_state_dict(sd)
        except Exception as e:  # noqa: BLE001
            print("[skip] %s head 形状不匹配：%r" % (rd, e)); continue
        model.eval()

        if a.numeric_bypass:
            # 被评模型带数值旁路时，head 维度是 1024+32，必须构造同样的旁路
            bypass = T.NumericBypass().to(dev)
            _m = np.asarray(T.numeric_features(
                [s for s in T.build_index(Path(a.dataset), None) if s["split"] == "train"]),
                dtype=np.float64)
            bypass.set_stats(_m.mean(0), _m.std(0))
            head = T.ActionHead(hidden, out=T.K_STEPS * act_dim,
                                extra=bypass.out_dim).to(dev)
            head.load_state_dict(sd)
        else:
            bypass = None

        per_proto = {}
        for pname, dl in dls.items():
            preds, acts, masks = [], [], []
            with torch.no_grad():
                for b in dl:
                    r = T.forward_batch(model, head, b["inputs"], dev,
                                        bypass=bypass, numeric=b.get("nums"))
                    p = r[0] if isinstance(r, tuple) else r
                    preds.append(p.float().cpu().numpy())
                    acts.append(b["act"].numpy())
                    masks.append(b["mask"].numpy())
            per_proto[pname] = (np.concatenate(preds), np.concatenate(acts),
                                np.concatenate(masks))
        # 两组口径必须都报：实测同一模型在「val 前400」和「val 全部」上差 59%
        #（ridge 1.2371 vs 0.7796），只报一组会得出错误结论。
        res = {"meta_act_dim": meta.get("act_dim"), "protocols": {}}
        print()
        print("  == %s" % run)
        for pname, (P, A, M) in per_proto.items():
            pr = {}
            pr["n"] = int(len(P))
            pr["mse8_speed"] = float(((((P[..., :2] - A[..., :2]) ** 2).sum(-1)) * M).sum()
                                      / max(M.sum(), 1))
            # ⚠️ copy8 必须与 ridge 同口径：**前 8 步**，不是全部 12 步。
            # 之前写 P.shape[1]（=12）导致 copy 基线 2.7754，而 ridge 在 8 步上是 1.8026，
            # 差 54%，让人误以为两批样本不同（2026-11-03 排查）。
            STEPS = 8
            _p, _a, _m = P[:, :STEPS, :2], A[:, :STEPS, :2], M[:, :STEPS]
            pr["copy8"] = float(
                ((((np.repeat(_a[:, :1], STEPS, axis=1) - _a) ** 2)
                  .sum(-1)) * _m).sum() / max(_m.sum(), 1))
            pr["mse8_speed_8step"] = float(
                ((((_p - _a) ** 2).sum(-1)) * _m).sum() / max(_m.sum(), 1))
            if P.shape[-1] > 2:
                pr["mse8_omega"] = float(((((P[..., 2] - A[..., 2]) ** 2)) * M).sum()
                                          / max(M.sum(), 1))
                sel = M > 0
                pr["omega_true_std"] = float(np.std(A[..., 2][sel]))
                pr["omega_pred_std"] = float(np.std(P[..., 2][sel]))
                pr["omega_corr"] = float(np.corrcoef(A[..., 2][sel], P[..., 2][sel])[0, 1])
            pr["dir_pred"] = dir_stats(P, M)
            pr["dir_true"] = dir_stats(A, M)
            res["protocols"][pname] = pr
            print("    [%s] n=%d  speed=%.4f  copy=%.4f  vs_copy=%+.1f%%"
                  % (pname, pr["n"], pr["mse8_speed"], pr["copy8"],
                     100.0 * (pr["copy8"] - pr["mse8_speed"]) / max(pr["copy8"], 1e-9)))
            if "mse8_omega" in pr:
                print("         omega=%.4f corr=%.3f std真=%.2f std预=%.2f (比=%.2f)"
                      % (pr["mse8_omega"], pr["omega_corr"], pr["omega_true_std"],
                         pr["omega_pred_std"],
                         pr["omega_pred_std"] / max(pr["omega_true_std"], 1e-9)))
        out["runs"][str(run)] = res

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "eval_omega.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("\n[done] %s" % (o / "eval_omega.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
