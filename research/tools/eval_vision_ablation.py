# -*- coding: utf-8 -*-
"""视觉消融：模型到底有没有在用图像？

**问题**（2026-11-02，用户提问「不应该让模型理解场景吗」）：
现在的输入是「两帧图像 + 4 个数字（body vx/vz、wp dx/dz）」，输出 12 步 (vx,vz)。
而 wp 与 A* 的 pure-pursuit 恒等 —— 也就是说所有「该往哪走、该走多快」的决定性
信息都在那 4 个数字里，图像可能只是陪跑。

若真是这样，这个模型就不是 VLA，是一个带图片装饰的几何控制器；
那么加再多几何标量都不会让它「理解场景」，只会让它更像查表。

**做法**：固定 (wp, body) 与文本不变，只替换视频张量，看动作块怎么变。
四个条件都跑同一批样本：

* base      —— 真实两帧（当前 + 0.2 s 前）
* black     —— 两帧全黑（保留形状与 token 数，视觉特征恒定）
* shuffle   —— 换成**别的样本**的图像（场景内容全错，但仍是自然图像）
* same      —— 两帧都用当前帧（抹掉时间差分）

**判据**：和 base 比动作块的差异。差异大 ⇒ 模型依赖视觉；
差异接近 0 ⇒ 视觉没进决策，那这个链路就不是 VLA。

用法::

    .venv_xpu/Scripts/python.exe research/tools/eval_vision_ablation.py \
        --runs .venv_xpu/runs/m0_s3_448/best --n 400 --out .venv_xpu/runs/vision_ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_policy_lora import (  # noqa: E402
    K_STEPS, ActionHead, PolicyDataset, build_index, forward_batch, load_model,
    make_collate,
)


class AblateDataset(Dataset):
    """在 PolicyDataset 外面套一层，只改 video，不改文本（所以 wp/body 逐字不变）。"""

    def __init__(self, base: PolicyDataset, n: int, mode: str, seed: int = 0):
        self.base = base
        self.n = min(n, len(base))
        self.mode = mode
        rng = np.random.default_rng(seed)
        # shuffle 的置换固定下来，四个条件用同一份，保证可比
        self.perm = rng.permutation(len(base))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        item = self.base[i]
        v = item["video"]
        if self.mode == "base":
            pass
        elif self.mode == "black":
            item["video"] = np.zeros_like(v)
        elif self.mode == "shuffle":
            j = int(self.perm[i % len(self.perm)])
            if j == i:
                j = int((j + 1) % len(self.perm))
            item["video"] = self.base[j]["video"].copy()
        elif self.mode == "same":
            item["video"] = np.stack([v[1], v[1]])
        else:
            raise ValueError(self.mode)
        return item


@torch.no_grad()
def predict(model, head, loader, device, injector=None, want_labels=False):
    """返回预测块；want_labels 时连 (act, mask) 一起返回，用于算真 mse。

    ⚠️ 只比 cos 是不够的：cos 低只说明「对图像敏感」，不说明「读对了」。
    判据必须是对标签的误差：mse8(real) 应当显著小于 mse8(shuffle)。
    """
    out, acts, masks = [], [], []
    for batch in loader:
        r = forward_batch(model, head, batch["inputs"], device, injector=injector)
        if isinstance(r, tuple):
            r = r[0]
        out.append(r.float().cpu().numpy())
        if want_labels:
            acts.append(batch["act"].numpy())
            masks.append(batch["mask"].numpy())
    pred = np.concatenate(out)
    if want_labels:
        return pred, np.concatenate(acts), np.concatenate(masks)
    return pred


def mse8_of(pred, act, mask, steps=8):
    """**只取速度两维**，与 train_policy_lora.masked_mse 的 mse8 口径对齐。

    ⚠️ 2026-11-04 修正：此前写成 pred[:, :steps, :]，把 ω 也算了进去。
    ω 是 rad/s（std=1.58），量级与速度（±4）不同但远大于「静止样本」的 0，
    会把 mse8 从 2.12 拉到 19.11（差 9 倍），导致本工具的数和训练日志对不上。
    训练日志里的 mse8 一律是 p[..., :2] 与 a[..., :2]。
    """
    err = ((pred[:, :steps, :2] - act[:, :steps, :2]) ** 2).sum(-1)
    m = mask[:, :steps]
    return float((err * m).sum() / max(m.sum(), 1.0))


def compare(base: np.ndarray, other: np.ndarray) -> dict:
    """动作块层面的差异：绝对值、余弦、以及「第 0 步速度变了多少」。"""
    d = other - base
    b = base.reshape(len(base), -1)
    o = other.reshape(len(other), -1)
    nb = np.linalg.norm(b, axis=-1)
    no = np.linalg.norm(o, axis=-1)
    denom = nb * no
    cos = np.where(denom > 1e-12, (b * o).sum(-1) / np.maximum(denom, 1e-12), 1.0)
    spd_b = np.linalg.norm(base[:, 0, :], axis=-1)
    spd_o = np.linalg.norm(other[:, 0, :], axis=-1)
    return dict(
        absdiff_mean=float(np.abs(d).mean()),
        absdiff_p95=float(np.percentile(np.abs(d), 95)),
        cosine_mean=float(cos.mean()),
        cosine_p05=float(np.percentile(cos, 5)),
        frac_cosine_lt_0_9=float((cos < 0.9).mean()),
        step0_speed_base_p50=float(np.percentile(spd_b, 50)),
        step0_speed_other_p50=float(np.percentile(spd_o, 50)),
        step0_speed_absdiff_mean=float(np.abs(spd_o - spd_b).mean()),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="视觉消融")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = int(a.size[0]) * int(a.size[1])

    samples = build_index(Path(a.dataset), None)
    val = [s for s in samples if s["split"] == "val"]
    print("[data] val=%d, 用前 %d 条" % (len(val), a.n), flush=True)
    ds = PolicyDataset(val, processor, min_px, max_px)   # 占位；每个 run 另行按格式重建
    collate = make_collate(processor, min_px, max_px)

    modes = ["base", "black", "shuffle", "same"]

    def make_loaders(dataset):
        return {m: DataLoader(AblateDataset(dataset, a.n, m), batch_size=a.batch,
                              shuffle=False, num_workers=0, collate_fn=collate)
                for m in modes}

    def run_prompt_format(run: Path) -> str:
        """按 run 的 meta.json 判定该模型训练时用的 prompt 格式。

        2026-11-02 换了记忆卡格式之后，用新格式评旧 checkpoint 会得出错误的绝对误差
        （同一模型 real mse8 从 1.2467 涨到 1.9644）。旧 run 的 meta.json 里没有
        这个字段，据此回落 legacy。
        """
        mp = run / "meta.json"
        if mp.exists():
            try:
                fmt = json.loads(mp.read_text(encoding="utf-8")).get("prompt_format")
                if fmt in ("card", "legacy"):
                    return fmt
            except Exception:  # noqa: BLE001
                pass
        return "legacy"

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"n": a.n, "runs": {}}
    for run in a.runs:
        run = Path(run)
        print("[load] %s" % run, flush=True)
        # ⚠️ act_dim 必须从 checkpoint 反推，不能默认 2。
        # 训练侧 2026-11-03 起是 act_dim=3（vx, vz, ω）× K_STEPS=12 = 36 维；
        # 默认 act_dim=2 → head 是 24 维，load_state_dict 直接 size mismatch。
        _sd0 = torch.load(Path(run) / "head.pt", map_location="cpu")
        _out = int(_sd0["fc.weight"].shape[0])
        _act_dim = 3 if (_out % 12 == 0 and _out // 12 == 3) else 2
        model, head, hidden = load_model(a.model, 128, device, act_dim=_act_dim)
        # ⚠️ 必须用 compat 版。训练侧对**内层模型**做了 torch.compile，
        # save_pretrained 存下的 372 条 LoRA 键全部带 `._orig_mod.` 前缀，
        # 而这里 load_model() 不编译 → 键名对不上 → PEFT 静默跳过。
        # 结果是「载入成功但权重没进去」，等于在评测一个完全没训练的 adapter。
        # 实测后果：mse8 从 ~2.1 变成 14.79（差 7 倍）。
        from peft_compat import load_adapter_compat
        _st = load_adapter_compat(model, run / "adapter", name="trained")
        print("[adapter] total=%d stripped=%d loaded=%d"
              % (_st["total"], _st["stripped"], _st["loaded"]), flush=True)
        assert _st["loaded"] == _st["total"], "adapter 键没全部写入"

        # 视觉注入模型：run 目录里有 injector.pt 就必须重建注入器，并把 head 换成
        # 加宽后的形状。否则 head.pt 是 (2048,24)、head 是 (1024,24)，load 直接报
        # size mismatch —— 串联任务会在这里挂掉。
        head_sd = torch.load(run / "head.pt", map_location=device)
        injector = None
        inj_path = run / "injector.pt"
        if inj_path.exists():
            from train_policy_lora import VisualInjector
            sd = torch.load(inj_path, map_location=device)
            proj_w = sd["proj.weight"]
            proj_dim, dim = int(proj_w.shape[0]), int(proj_w.shape[1])
            # ⚠️ 不能从 proj.bias 推 out_dim —— bias 的长度就是 proj_dim。
            # 正确来源是 head.pt 的第一层宽度：extra = in_features - hidden。
            extra = int(head_sd["fc.weight"].shape[1]) - int(hidden)
            if extra <= 0 or proj_dim <= 0 or extra % proj_dim != 0:
                raise RuntimeError("注入器维度对不上：extra=%d proj=%d" % (extra, proj_dim))
            blocks = int(round((extra // proj_dim) ** 0.5))
            injector = VisualInjector(dim, grid=(8, 14), blocks=blocks,
                                      proj=proj_dim).to(device)
            injector.load_state_dict(sd)
            injector.eval()
            head = ActionHead(hidden, extra=injector.out_dim).to(device)
            print("[inject] proj_dim=%d blocks=%d extra=%d"
                  % (proj_dim, blocks, injector.out_dim), flush=True)

        head.load_state_dict(head_sd)
        model.eval()

        fmt = run_prompt_format(run)
        # ⚠️ 必须从 run 的 args.json 回放输入口径。2026-11-04 踩到：
        # r8_nowp 训练时 --drop-wp（输入里 wp 恒 0），但这里默认 drop_wp=False，
        # 等于在评测时喂了模型**训练中从未见过**的 wp，bench 数字全部作废。
        _run_args = {}
        _ap = run.parent / "args.json"
        if _ap.exists():
            _run_args = json.loads(_ap.read_text(encoding="utf-8"))
        ds_run = PolicyDataset(val, processor, min_px, max_px, prompt_format=fmt,
                               drop_wp=bool(_run_args.get("drop_wp", False)),
                               quad_feats=bool(_run_args.get("quad_feats", False)))
        print("[caliber] drop_wp=%s quad_feats=%s"
              % (_run_args.get("drop_wp", False), _run_args.get("quad_feats", False)),
              flush=True)
        loaders = make_loaders(ds_run)
        print("[prompt] format=%s" % fmt, flush=True)

        preds = {}
        label_mse = {}
        for m in modes:
            preds[m], act_l, mask_l = predict(model, head, loaders[m], device,
                                              injector=injector, want_labels=True)
            label_mse[m] = mse8_of(preds[m], act_l, mask_l)
            print("  [%s] done  mse8=%.4f" % (m, label_mse[m]), flush=True)

        # 关键判据：对**标签**的误差。base 是真实图，理应最好。
        base_mse = label_mse["base"]
        vision_helps = {
            "mse_real": base_mse,
            "mse_black": label_mse["black"],
            "mse_shuffle": label_mse["shuffle"],
            "gain_vs_black": label_mse["black"] - base_mse,
            "gain_vs_shuffle": label_mse["shuffle"] - base_mse,
            "reads_content": bool(label_mse["shuffle"] > base_mse),
        }
        print("  >>> mse8 real=%.4f black=%.4f shuffle=%.4f | "
              "real 比 black 好 %.4f, 比 shuffle 好 %.4f, 读内容=%s"
              % (base_mse, label_mse["black"], label_mse["shuffle"],
                 vision_helps["gain_vs_black"], vision_helps["gain_vs_shuffle"],
                 vision_helps["reads_content"]), flush=True)

        base = preds["base"]
        rows = {}
        for m in modes[1:]:
            rows[m] = compare(base, preds[m])
            r = rows[m]
            print("  %-8s |d|=%.3f  cos=%.3f (p05 %.3f)  step0 speed %.2f vs %.2f"
                  % (m, r["absdiff_mean"], r["cosine_mean"], r["cosine_p05"],
                     r["step0_speed_base_p50"], r["step0_speed_other_p50"]), flush=True)
        # 顺带报一下 base 自身输出的离散度：如果模型对任何场景都输出几乎一样的东西，
        # 那「依赖视觉」这件事本身就不成立。
        rows["base_spread"] = dict(
            step0_speed_p50=float(np.percentile(np.linalg.norm(base[:, 0, :], axis=-1), 50)),
            step0_speed_p05=float(np.percentile(np.linalg.norm(base[:, 0, :], axis=-1), 5)),
            step0_speed_p95=float(np.percentile(np.linalg.norm(base[:, 0, :], axis=-1), 95)),
            block_std_across_samples=float(base.reshape(len(base), -1).std(axis=0).mean()),
        )
        rows["label_mse"] = label_mse
        rows["vision_helps"] = vision_helps
        result["runs"][str(run)] = rows
    (out_dir / "vision_ablation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (out_dir / "vision_ablation.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
