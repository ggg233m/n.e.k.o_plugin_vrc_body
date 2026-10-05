# -*- coding: utf-8 -*-
r"""房间尺子：用**世界里的实物刻度**给"1 个地图单位 = 多少世界米"定标。

    python research/tools/room_ruler.py .slam_probe/stereo_seq/run9_ruler_front

为什么要它
----------
`Docs/漂移形态诊断（2026-10-05）.md` §十 用 SGBM+PnP 自建了几何链，但那个工具**自证不了
绝对米制**（它量的是两条链之间的比，且短弦上有系统收缩）。这里的房间自带一把**按世界米
刻的竖尺**（厘米刻度、每 10 cm 一格、2m/150/1m 是红线），于是有一条**完全外部**的锚点：
站在尺子正前方拍一段，就能把"地图单位"钉到"世界米"。

两条独立途径（互相印证，别只信一条）
------------------------------------
记 ``u`` = 1 个地图单位 = 1 个双目单位（`b=0.126` 追踪米算出来的深度单位）值多少世界米。
**要写进配置的 `world_scale` 就是 u。**（`nav_online.DeadReckoner._advance` 把 OSC 的
世界米位移除以 `world_scale` 换成地图单位 ⇒ 除出来的必须是 u，两条链才同尺。）

A. **地面途径**：画面竖直中心行对应"与相机等高的物点"（前提：视线水平。虚拟驱动的 HMD
   俯仰恒 0，内参 cx/cy 又正好是画面正中 ⇒ 成立）。读出那一行在尺子上的值 = 相机在
   **世界米**下的高度 H；双目量"地面在相机下方"得到 cam_h 个**双目单位**。⇒ ``u = H/cam_h``。
   ⚠️ 这一条依赖"双目量的那层地面 = 相机脚下那层地面"。站着不动、视野里只有一层地板时成立；
   房间有台阶/平台时会把另一层量的进来（run8/run9 同房间的 cam_h 就差 7%）。

B. **墙面途径**（不碰地面，通常更硬）：尺子贴在竖直墙面上。尺子自己给出"1 世界米 = 多少
   像素"（红线 200↔100 相距 px_per_m 个像素）；而墙的深度 Z 满足 ``px_per_m = fx/Z_m``
   ⇒ 墙在世界米下的距离 = fx/px_per_m。双目在那片墙上的视差 d 给出同一个距离的**单位**值
   ``Z_u = fx·b/d``。⇒ ``u = Z_m/Z_u``。
   ⚠️ 这条假设"墙面垂直于视线"。墙微斜会让 px_per_m 沿画面上下变化（run9 实测 286–300 px/m，
   即 ±2%），所以工具把 200/150/100 三条红锚的残差一并报出来，供判断。

⚠️ 别看错参照系（这是 §十 里反复踩的坑）
----------------------------------------
* **HMD 遥测里的 1.5 是"追踪系"的数**，不是世界米；它也**不是**"相机在世界里多高"。
  相机实际在哪，要由尺子读出来（run9 实测 1.34 世界米）。
* **别拿 VRChat 面板报的"模型身高"当尺子**：run9 面板报 1.26 m，而尺子/双目实测视点在
  1.34 m（差 6%）。要绝对锚点就用**房间里的实物刻度**。
* 相对比（几何链 vs DR 的斜率）**随弦长变化**（短弦被 PnP 解小），那不是尺度，别拿来定标。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "research" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402

from backend.nav_mapping import make_sgbm, stereo_disparity, stereo_points  # noqa: E402
from stereo_geom_chain import camh_from_points                              # noqa: E402


def ruler_geometry(bgr: np.ndarray, band: tuple[int, int],
                   red_values: tuple[float, ...]) -> dict:
    """从一帧里读尺子：红色绝对锚点的行、刻度线的行、以及中心行对应的厘米值。

    红锚（2m/150/1m）是**红字/红线**，颜色可判 ⇒ 用它们把"第几格是哪个值"钉死，
    不必人工估行号。刻度线是深色细线 ⇒ 行灰度剖面的局部极小。
    """
    x0, x1 = band
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    prof = gray[:, x0:x1].mean(axis=1)
    base = float(np.median(prof))
    dark = np.where(base - prof > 6.0)[0]
    ticks: list[float] = []
    if dark.size:
        cur = [dark[0]]
        for v in dark[1:]:
            if v - cur[-1] <= 2:
                cur.append(v)
            else:
                ticks.append(float(cur[int(np.argmin(prof[cur]))]))
                cur = [v]
        ticks.append(float(cur[int(np.argmin(prof[cur]))]))

    b, g, r = (bgr[:, x0:x1, i].astype(np.float64) for i in range(3))
    cnt = ((r - 0.5 * (g + b)) > 20.0).sum(axis=1)
    cand = np.where(cnt >= 3)[0]
    reds: list[float] = []
    if cand.size:
        cur = [cand[0]]
        for v in cand[1:]:
            if v - cur[-1] <= 4:
                cur.append(v)
            else:
                reds.append(float(np.mean(cur)))
                cur = [v]
        reds.append(float(np.mean(cur)))

    # 红锚按行号升序 = 高度降序 ⇒ 值也从大到小（上层是 200、下层是 100）
    vals = list(red_values)[:len(reds)]
    anchors = list(zip(vals, reds))
    # 红字/红线的行与它那条刻度线通常差几像素：优先用"最近的刻度行"当锚，更准
    matched = []
    for v, row in anchors:
        if ticks:
            near = min(ticks, key=lambda t: abs(t - row))
            if abs(near - row) <= 8.0:
                matched.append((float(v), near))
                continue
        matched.append((float(v), row))
    return {"ticks": ticks, "reds": reds, "matched": matched}


def cm_at_cy(matched: list[tuple[float, float]], cy: float) -> tuple[float, float, float]:
    """(中心行的厘米值, 每米像素数, 拟合最大残差)。"""
    if len(matched) < 2:
        return float("nan"), float("nan"), float("nan")
    cm = np.array([m[0] for m in matched], float)
    rr = np.array([m[1] for m in matched], float)
    slope, inter = np.polyfit(cm, rr, 1)          # row = inter + slope*cm
    resid = float(np.abs(np.polyval([slope, inter], cm) - rr).max())
    px_per_m = float(abs(slope) * 100.0)
    return float((cy - inter) / slope), px_per_m, resid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("seq", type=Path)
    ap.add_argument("--band", default="266:336",
                    help="尺子面板上的取样列（要避开左侧数字、右侧人物/背景）")
    ap.add_argument("--red-values", default="200,150,100",
                    help="红色锚点刻度从上行到下行对应的厘米值（本房间 2m/150/1m）")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    band = tuple(int(v) for v in args.band.split(":"))
    red_values = tuple(float(v) for v in args.red_values.split(","))

    seq = args.seq
    meta = json.loads((seq / "meta.json").read_text(encoding="utf-8"))
    fx, cx, cy = float(meta["fx"]), float(meta["cx"]), float(meta["cy"])
    b = float(meta["baseline_tracking_m"])
    print(f"序列 {seq.name}：fx={fx:g} cx={cx:g} cy={cy:g} 基线={b:g}；"
          f"尺子取样列 {band[0]}–{band[1]}，红锚值 {red_values}")

    frames = sorted((seq / "mav0" / "cam0" / "data").glob("*.png"))
    matcher = make_sgbm()
    eyes, pxs, resids, camhs, walld = [], [], [], [], []
    for f in frames:
        fr = f.parent.parent.parent / "cam1" / "data" / f.name
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        il = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(str(fr), cv2.IMREAD_GRAYSCALE)
        if bgr is None or il is None or ir is None:
            continue
        geo = ruler_geometry(bgr, band, red_values)
        eye, px_per_m, resid = cm_at_cy(geo["matched"], cy)
        if np.isfinite(eye):
            eyes.append(eye)
            pxs.append(px_per_m)
            resids.append(resid)
        disp = stereo_disparity(il, ir, matcher)
        ch = camh_from_points(stereo_points(il, ir, fx=fx, cx=cx, cy=cy, baseline_m=b,
                                           max_range_m=6.0, step=2, disp=disp))
        if ch is not None:
            camhs.append(ch)
        reg = disp[:, band[0]:band[1]]
        good = reg[reg > 1.0]
        if good.size > 200:
            walld.append(float(np.median(good)))

    if not eyes:
        print("❌ 没读到尺子（取样列或红锚值不对？）")
        return 1
    h_m = float(np.median(eyes)) / 100.0
    px_per_m = float(np.median(pxs))
    print(f"\n① 尺子读数：画面中心行（第 {cy:g} 行）= **{h_m:.3f} 世界米**"
          f"（{len(eyes)} 帧，IQR {np.percentile(eyes,25)/100:.3f}–"
          f"{np.percentile(eyes,75)/100:.3f}；每米 {px_per_m:.1f} px，"
          f"红锚拟合残差最大 {max(resids):.1f} px）")

    out: dict = {"seq": seq.name, "eye_world_m": h_m, "px_per_m": px_per_m,
                 "n_frames": len(eyes), "red_anchor_resid_px": max(resids)}
    u_vals = []
    if camhs:
        ch = float(np.median(camhs))
        u_a = h_m / ch
        u_vals.append(u_a)
        print(f"② A 地面途径：cam_h 中位 {ch:.3f} 单位（{len(camhs)} 帧，IQR "
              f"{np.percentile(camhs,25):.3f}–{np.percentile(camhs,75):.3f}）"
              f" ⇒ u = {h_m:.3f}/{ch:.3f} = **{u_a:.4f}**")
        out.update({"cam_h_units": ch, "u_floor": u_a})
    if walld:
        d = float(np.median(walld))
        z_u = fx * b / d
        z_m = fx / px_per_m
        u_b = z_m / z_u
        u_vals.append(u_b)
        print(f"③ B 墙面途径：尺子面板视差中位 {d:.2f} px（{len(walld)} 帧）"
              f" ⇒ Z_u={z_u:.3f} 单位，Z_m={z_m:.3f} 世界米 ⇒ u = **{u_b:.4f}**")
        out.update({"wall_disp_px": d, "z_units": z_u, "z_world_m": z_m, "u_wall": u_b})
    if u_vals:
        lo, hi = min(u_vals), max(u_vals)
        mid = float(np.mean(u_vals))
        print(f"\n⇒ 1 个地图单位 = **{mid:.4f} 世界米**（两条途径 {lo:.4f}–{hi:.4f}，"
              f"即 ±{100*(hi-lo)/2/mid:.1f}%）")
        try:
            from backend.nav_online import OnlineNavConfig
            cur = float(OnlineNavConfig().world_scale)
            print(f"   现役配置 world_scale = {cur:.4f}"
                  f" ⇒ 差 {100*(cur/mid-1):+.1f}%（落在测量误差内即「这个常数是对的」）")
            out["config_world_scale"] = cur
        except Exception:
            pass
        out["u_mean"] = mid
    print("\n⚠️ 别把这里的数跟 HMD 遥测的 1.5 直接比：那是**追踪系**的高度，"
          "而相机在世界里多高要由尺子读出。也别拿 VRChat 面板报的模型身高当尺子。")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())