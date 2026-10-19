
# -*- coding: utf-8 -*-
"""WLS / SGBM 后处理基准（合成真值 + 计时），用于评估 WLS 值不值得接进在线管线。

     python research/tools/wls_bench.py
     python research/tools/wls_bench.py --images left.png right.png   # 用真实截图（无真值）

合成场景 = 720×405，视差 5~23 px，右侧 35% 无纹理墙，含遮挡带，亚像素 GT。

⚠️ **2026-10-19 的本地实测结论（真数据，本脚本没跑）**：WLS 确实降孤立率
（10.76%→7.03% / 10.41%→8.37% / 11.88%→8.28%，1.24–1.53×），但代价是
**中位 49.3–55.3 ms、最差 103.4 ms**（超过 33.3 ms 帧预算），且 λ=500 只保留
60.8% 的真实边缘台阶。⇒ **暂不接进 `stereo_disparity`**；若要接只能关键帧专属。

另一个已踩的坑：`createDisparityWLSFilter` 需要 `createRightMatcher`，且 λ 的
量纲跟输入单位绑定（Q4 定点 vs 像素差 16×）—— 本脚本的
`lambda_needs_matching_units` 段就是量这个的。
"""
import numpy as np, cv2, time, json, sys, argparse
rng = np.random.default_rng(77)
W, H = 720, 405
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def make_synthetic():
    tex = cv2.GaussianBlur(rng.random((H, W, 3)).astype(np.float32) * 255, (3, 3), 0.8)
    L = tex.copy()
    wall = xx > 470
    for c in range(3):
        L[:, :, c][wall] = 132.0
    vig = (1.0 - 0.25 * ((xx - W/2)**2 + (yy - H/2)**2) / (W*W))[:, :, None]
    L = np.clip(L * vig + rng.normal(0, 2.0, (H, W, 3)), 0, 255).astype(np.uint8)
    L = cv2.GaussianBlur(L, (3, 3), 0.6)
    d = (5 + 18 * (yy / float(H - 1))).astype(np.float32)
    d[(xx > 150) & (xx < 340) & (yy > 160) & (yy < 330)] = 20.0
    d[wall] = 8.0
    gt = cv2.GaussianBlur(d, (3, 5), 1.0)
    dat = cv2.remap(gt, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    for _ in range(4):
        dat = cv2.remap(gt, xx + dat, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    R = cv2.remap(L, (xx + dat).astype(np.float32), yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return L, R, gt, wall

def mk(minD=0, nd=32, bs=5, spk=100):
    return cv2.StereoSGBM_create(minDisparity=minD, numDisparities=nd, blockSize=bs,
        P1=24*bs*bs, P2=96*bs*bs, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=spk, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def bench(fn, n=15, warm=4):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(ts)), 2), round(float(np.percentile(ts, 90)), 2)

def metric_px(d_px, gt, mask, name):
    """d_px: 已经是像素为单位的 float32 视差; 无效值(<=0)按 5px 惩罚。"""
    e = np.abs(d_px - gt); e = np.where(d_px > 0, e, 5.0)
    return {name: dict(texMAE=round(float(e[mask].mean()), 3),
                       bad1_pct=round(float((e[mask] > 1).mean() * 100), 1),
                       fill_pct=round(float((d_px > 0).mean() * 100), 1))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs=2, default=None, help="left.png right.png (未标注则用合成场景)")
    a = ap.parse_args()
    if a.images:
        L, R = cv2.imread(a.images[0]), cv2.imread(a.images[1])
        gt, wall = None, None
    else:
        L, R, gt, wall = make_synthetic()
        wall = wall & (xx < 700)
        tex = ~wall

    res = {"timing_ms": {}, "quality": {}}
    ml = mk(); mr = cv2.ximgproc.createRightMatcher(ml)
    tl, pl = bench(lambda: ml.compute(L, R)); dl = ml.compute(L, R)
    tr, pr = bench(lambda: mr.compute(R, L)); dr = mr.compute(R, L)
    w = cv2.ximgproc.createDisparityWLSFilter(ml); w.setLambda(8000); w.setSigmaColor(1.5)
    tw, pw = bench(lambda: w.filter(dl, L, None, dr))
    g = cv2.ximgproc.createDisparityWLSFilterGeneric(False); g.setLambda(8000); g.setSigmaColor(1.5)
    tg, pg = bench(lambda: g.filter(dl, L, None, None))
    res["timing_ms"] = dict(sgbm_left=[tl, pl], sgbm_right=[tr, pr],
        wls_with_conf=[tw, pw], wls_generic_no_conf=[tg, pg],
        pipeline_proper=round(tl + tr + tw, 2), pipeline_generic=round(tl + tg, 2))

    if gt is not None:
        res["quality"].update(metric_px(dl.astype(np.float32)/16.0, gt, tex, "sgbm_raw"))
        res["quality"].update(metric_px(w.filter(dl, L, None, dr).astype(np.float32)/16.0, gt, tex, "wls_with_conf"))
        res["quality"].update(metric_px(g.filter(dl, L, None, None).astype(np.float32)/16.0, gt, tex, "wls_generic_noconf"))
        res["quality"].update(metric_px(cv2.medianBlur(dl.astype(np.float32)/16.0, 5), gt, tex, "median5_direct"))
        res["quality"].update(metric_px(w.filter(dl, L, None, dr).astype(np.float32)/16.0, gt, wall, "WLS_on_lowtexture_wall"))

    mp = mk(); before = (mp.getDisp12MaxDiff(), mp.getSpeckleWindowSize(), mp.getUniquenessRatio())
    cv2.ximgproc.createDisparityWLSFilter(mp)
    after = (mp.getDisp12MaxDiff(), mp.getSpeckleWindowSize(), mp.getUniquenessRatio())
    res["factory_mutates_matcher"] = dict(before=before, after=after)

    w2 = cv2.ximgproc.createDisparityWLSFilter(ml); w2.setSigmaColor(1.5)
    unit = {}
    for lam in (500, 32000, 128000):
        w2.setLambda(lam)
        q4 = metric_px(w2.filter(dl, L, None, dr).astype(np.float32)/16.0, gt, tex, "x") if gt is not None else None
        px = metric_px(w2.filter(dl.astype(np.float32)/16.0, L, None, dr.astype(np.float32)/16.0), gt, tex, "x") if gt is not None else None
        unit[str(lam)] = dict(Q4_input=q4, PX_input=px)
    res["lambda_needs_matching_units"] = unit

    th = {}
    for n in (1, 4, 20):
        cv2.setNumThreads(n); th[str(n)] = bench(lambda: ml.compute(L, R), n=5)[0]
    cv2.setNumThreads(0)
    res["threads_ms"] = th
    res["cpu"] = "Xeon E5-2666 v3 @2.90GHz, 20 logical" if not a.images else "?"
    print(json.dumps(res, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
