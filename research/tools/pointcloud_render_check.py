
# -*- coding: utf-8 -*-
"""把 .tmp 的离线光栅化工具搬进 research/tools，作为 viewer 改动的回归检查。

为什么需要它：viewer 是纯前端，浏览器外没有任何测试能覆盖它。
本脚本用**同一份字节、同一套 MVP 数学**在 numpy 里复现渲染，用来回答：
  1) 深度精度够不够（16 位深度单位 vs 点云局部厚度）；
  2) 三个按钮（会话/高度/时间）是否真的产生不同颜色；
  3) 改动前后（near/far）的差异有多大。
"""
from __future__ import annotations

import argparse
import base64
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "tools"))

from prior_topview import write_png  # noqa: E402

XYO, XYS, ZS, ZO = 32.0, 1000.0, 50.0, 1.0
PAT = re.compile(r'const RAW = atob\("([A-Za-z0-9+/=]*)"\)')
PALETTE = np.array([[79, 156, 240], [242, 163, 60], [89, 201, 141], [224, 101, 95],
                    [169, 122, 224], [63, 184, 196], [229, 192, 75], [143, 155, 179]],
                   np.float32)
BG = np.array([15, 17, 21], np.float32)


def load(html_path: Path):
    """解出点云字节。7 字节/点：u16 x, u16 y, u8 z, u8 sid, u8 time。"""
    h = html_path.read_text(encoding="utf-8", errors="replace")
    m = PAT.search(h)
    if not m:
        raise SystemExit("no payload in %s" % html_path)
    raw = base64.b64decode(m.group(1))
    if len(raw) % 7:
        raise SystemExit("payload %d bytes is not a multiple of 7" % len(raw))
    a = np.frombuffer(raw, np.uint8).reshape(-1, 7)
    x = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)) / XYS - XYO
    y = (a[:, 2].astype(np.int32) | (a[:, 3].astype(np.int32) << 8)) / XYS - XYO
    z = a[:, 4].astype(np.float64) / ZS - ZO
    return np.column_stack([x, y, z]), a[:, 5].astype(np.int64), a[:, 6].astype(np.int64)


def depth_unit(near: float, far: float, z_ref: float, bits: int = 16) -> float:
    """16 位深度缓冲在 z_ref（距相机米数）处 1 个单位等于多少米。

    这个数**大于点云的局部厚度**时，同一像素里的点深度全部相等，
    谁最后画谁赢 ⇒ 满屏闪烁。所以它必须显著小于面厚。
    """
    def d01(zm):
        z = -abs(zm)
        zc = ((far + near) / (near - far)) * z + (2 * far * near / (near - far))
        return (zc / abs(z) + 1.0) * 0.5
    return 1.0 / ((1 << bits) - 1) / max(abs(d01(z_ref + 1.0) - d01(z_ref)), 1e-15)


def hsv_rainbow(t, s=0.60, v0=0.80, v1=1.00):
    """与 shader 里的时间彩虹逐字同式 —— 图例和画面必须用同一个公式。"""
    c = np.clip(np.asarray(t, np.float64), 0, 1)
    h = 0.62 * (1.0 - c)
    v = v0 + (v1 - v0) * c
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    w = v * (1.0 - s * (1.0 - f))
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, w, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [w, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, w, v, v, q])
    return np.stack([r, g, b], -1) * 255.0


def render(P, col, out_png, W=1200, H=700, yaw=0.7, pitch=0.50,
           fov=0.9, dist=None, near=None, far=None, bits=16):
    """离线复现 viewer 的渲染：MVP → 量化深度 → z-buffer → 圆盘点。

    bits=None 表示不量化（浮点深度），作为"完美渲染"的对照。
    """
    lo, hi = P.min(0), P.max(0)
    cen = (lo + hi) / 2.0
    span = float(max(hi - lo))
    R = float(np.linalg.norm(hi - cen))
    dist = (span * 1.6 + 4.0) if dist is None else dist
    near = max(0.05, 0.5 * max(dist - R, 0.1)) if near is None else near
    far = (dist + R + 20.0) if far is None else far
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)
    ex = cen[0] + dist * cp * syw; ey = cen[1] - dist * cp * cyw; ez = cen[2] + dist * sp
    zx, zy, zz = cen[0] - ex, cen[1] - ey, cen[2] - ez
    L = math.sqrt(zx * zx + zy * zy + zz * zz) or 1.0
    zx /= L; zy /= L; zz /= L
    xx, xy, xz = cyw, syw, 0.0
    yx, yy, yz = zz * xy - zy * xz, zx * xz - zz * xx, zy * xx - zx * xy
    L = math.sqrt(yx * yx + yy * yy + yz * yz) or 1.0
    yx /= L; yy /= L; yz /= L
    xx = zy * yz - zz * yy; xy = zz * yx - zx * yz; xz = zx * yy - zy * yx
    Rv = np.array([[xx, xy, xz], [yx, yy, yz], [-zx, -zy, -zz]])
    t = np.array([-(xx * ex + xy * ey + xz * ez), -(yx * ex + yy * ey + yz * ez),
                  (zx * ex + zy * ey + zz * ez)])
    cam = P @ Rv.T + t
    w = -cam[:, 2]
    tt = 1.0 / math.tan(fov / 2.0); ar = W / H
    ndx = (tt / ar) * cam[:, 0] / np.where(w > 0, w, 1)
    ndy = tt * cam[:, 1] / np.where(w > 0, w, 1)
    kp = (w >= near) & (w <= far) & (np.abs(ndx) <= 1) & (np.abs(ndy) <= 1)
    idx = np.where(kp)[0]
    info = dict(unit=depth_unit(near, far, max(dist, 1), 16), visible=int(len(idx)),
                clipped=int((w < near).sum()), near=near, far=far, dist=dist, px=0)
    if not len(idx):
        img = np.tile(BG, (H, W, 1)).astype(np.uint8)
        if out_png is not None:
            write_png(out_png, img)
        return img, info
    zv = cam[idx, 2]
    zc = ((far + near) / (near - far)) * zv + (2 * far * near / (near - far))
    d01 = (zc / w[idx] + 1.0) * 0.5
    # ⚠️ 浮点基准必须是**真实深度**（d01 本身），不是 np.arange(点序号)。
    # 写成 arange 时"最小 dep"退化成"像素内首个点"，基准本身就错了，
    # 于是任何位深与它的差异都只是个与精度无关的常数（实测恒为 1.87%）。
    dep = d01 if bits is None else np.round(d01 * ((1 << bits) - 1))
    ci = np.clip(((ndx * 0.5 + 0.5) * W).astype(np.int64), 0, W - 1)[idx]
    ri = np.clip(((0.5 - ndy * 0.5) * H).astype(np.int64), 0, H - 1)[idx]
    pix = ri.astype(np.int64) * W + ci
    order = np.lexsort((idx, dep, pix))                    # 像素 → 深度 → 序号
    oi, od, op = idx[order], dep[order], pix[order]
    edges = np.searchsorted(op, np.arange(W * H + 1))
    has = np.diff(edges) > 0
    mnp = np.where(has)[0]
    starts = edges[:-1][has]
    counts = np.diff(edges)[has]
    # ⚠️ 必须取**深度最小**（离相机最近）的点，这是 z-buffer 的定义；同深度时后画的赢。
    # 这里踩过两个坑，都表现为"两个参数渲染出完全相同的图"（A/B 恒等于 0）：
    #   ① 直接取组内最后一个 ⇒ 那是**最远**的点，与深度量化无关 ⇒ 精度成了无关变量；
    #   ② 用 np.searchsorted(od, dmin+1) 定位并列段右端 ⇒ od 只在**像素内部**有序、
    #      全局无序，searchsorted 的返回值无意义，再被 clamp 回来就退化成 ①。
    # 正确做法：按组求最小深度（reduceat 要求组首索引，正好是 starts），
    # 再在组内找"深度等于该最小值"的最后一个位置（maximum.at 累积）。
    dmin = np.minimum.reduceat(od, starts)
    gid = np.repeat(np.arange(len(starts)), counts)
    pos = np.arange(len(od)) - np.repeat(starts, counts)
    sel = od == np.repeat(dmin, counts)
    last = np.zeros(len(starts), np.int64)
    np.maximum.at(last, gid[sel], pos[sel])
    pick = oi[starts + last]
    buf = np.tile(BG, (H, W, 1)).reshape(-1, 3)
    size = float(np.clip(240.0 / max(np.median(w[idx]), 1e-6), 1.5, 5.0))
    r = max(1, int(round(size / 2.0)))
    offs = [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1)
            if a * a + b * b <= size * size / 4.0 or size <= 2]
    rr, cc = mnp // W, mnp % W
    for a, b in offs:
        buf[np.clip(rr + a, 0, H - 1) * W + np.clip(cc + b, 0, W - 1)] = col[pick]
    img = buf.reshape(H, W, 3)
    if out_png is not None:
        write_png(out_png, np.clip(img, 0, 255).astype(np.uint8))
    info["px"] = int(mnp.size)
    return np.clip(img, 0, 255).astype(np.uint8), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("html")
    ap.add_argument("--out", default=".tmp/pointcloud")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dist", type=float, default=None, help="相机距离（默认 = 复位档）")
    ap.add_argument("--old-near", action="store_true",
                    help="用改动前的 near=0.05 / far=dist*8+50（对照用）")
    a = ap.parse_args()

    src = Path(a.html)
    P, sid, tim = load(src)
    tag = a.tag or src.stem
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("[load ] %s  %d 点  sid 值 %d 个  time 值 %d 个 [%d,%d]"
          % (src.name, len(P), len(set(sid.tolist())), len(set(tim.tolist())),
             int(tim.min()), int(tim.max())))
    lo, hi = P.min(0), P.max(0)
    span = float(max(hi - lo))
    R = float(np.linalg.norm(hi - (lo + hi) / 2.0))
    dist = a.dist if a.dist is not None else (span * 1.6 + 4.0)
    near = 0.05 if a.old_near else max(0.05, 0.5 * max(dist - R, 0.1))
    far = (dist * 8 + 50) if a.old_near else (dist + R + 20.0)
    u = depth_unit(near, far, max(dist, 1))
    print("[cam  ] dist=%.1f  near=%.2f  far=%.0f  16 位单位=%.4f m (%.0f mm) @%.0fm"
          % (dist, near, far, u, u * 1000, dist))
    print("[check] 点云局部厚度量级 ~25 mm ⇒ %s"
          % ("OK（单位 << 面厚）" if u < 0.008 else
             "**单位大于面厚，必然 z-fighting**" if u > 0.025 else "临界"))

    nz = np.clip((P[:, 2] - np.percentile(P[:, 2], 1)) /
                 max(np.percentile(P[:, 2], 99) - np.percentile(P[:, 2], 1), 1e-6), 0, 1)
    hcol = ((np.array([0.13, 0.42, 0.87]) * (1 - nz)[:, None]
             + np.array([0.98, 0.72, 0.20]) * nz[:, None]) * 255.0)
    modes = {"MODE0_sess": PALETTE[sid % len(PALETTE)],
             "MODE1_height": hcol,
             "MODE2_time": hsv_rainbow(tim / 254.0)}
    imgs, infos = {}, {}
    for name, col in modes.items():
        img, info = render(P, np.asarray(col, np.float32),
                           out / ("%s_%s.png" % (tag, name)), dist=dist, near=near, far=far)
        imgs[name] = img; infos[name] = info
        print("[png  ] %-14s 覆盖像素 %7d  可见 %8d"
              % (name, info["px"], info["visible"]))
    print()
    print("[diff ] 三模式的共同像素颜色差异（都很大 ⇒ 三个按钮真的在改着色）：")
    names = list(modes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = imgs[names[i]], imgs[names[j]]
            bg = np.abs(A.astype(int) - BG).sum(-1) > 6
            both = bg & (np.abs(B.astype(int) - BG).sum(-1) > 6)
            if not both.any():
                print("   %-13s vs %-13s 无共同像素" % (names[i], names[j])); continue
            d = np.abs(A[both].astype(int) - B[both].astype(int)).max(1)
            print("   %-13s vs %-13s 共同 %7d  颜色不同 %5.1f%%  差>80 %.1f%%"
                  % (names[i], names[j], int(both.sum()), 100.0 * (d > 0).mean(),
                     100.0 * (d > 80).mean()))


if __name__ == "__main__":
    main()
