# -*- coding: utf-8 -*-
r"""导出指纹：两份 pointcloud_view 导出「到底是不是同一回事」。

为什么需要它（2026-10-27 实测）：我曾把 final_poses（610 帧）与 final_poses_full（759 帧）
的导出并排比较，得出「回环校正把回访从 435 压到 80.5 cm」的结论 —— 实际上 final_poses 只是
full 的**帧子集**（共同帧位置差 0.0000 m，100% 相同），根本没有「回环校正」这个变量。
两个数不可比，结论作废。

本工具打印两份导出的元信息 + 空间格重叠率，用来在比较**之前**断言它们同源。
重叠率 < 95% 就说明至少差了一个变量（帧集 / 体素 / 距离闸 / 位姿表 / 地面口径）。

用法：
    .venv/Scripts/python.exe research/tools/export_fingerprint.py a.html
    .venv/Scripts/python.exe research/tools/export_fingerprint.py a.html b.html   # 对照
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import numpy as np

STRIDE = 8
XYO, XYS, ZS, ZO = 32.0, 1000.0, 50.0, 1.0


def load(path: Path):
    h = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'const RAW = atob\("([A-Za-z0-9+/=]*)"\)', h)
    if not m:
        raise SystemExit(f"[fail] {path} 没有 RAW 载荷")
    a = np.frombuffer(base64.b64decode(m.group(1)), np.uint8).reshape(-1, STRIDE)
    x = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)) / XYS - XYO
    y = (a[:, 2].astype(np.int32) | (a[:, 3].astype(np.int32) << 8)) / XYS - XYO
    z = a[:, 4].astype(np.float64) / ZS - ZO
    P = np.column_stack([x, y, z])
    vox = re.search(r"gl\.uniform1f\(uVox, ([0-9.]+)\)", h)
    sess = re.search(r"const SESS = (\[.*?\]);", h, re.S)
    return {
        "path": path,
        "n": len(P),
        "pts": P,
        "vox": float(vox.group(1)) if vox else None,
        "sess": sess.group(1)[:200] if sess else None,
        "new_sprite": "uVox*uFocal" in h,
        "hit_filter": "显示门槛" in h,
        "gc_mode": "gc_mode" in h,
    }


def cells(P, v):
    q = np.floor(P / v).astype(np.int64)
    return q[:, 0] * (1 << 42) + q[:, 1] * (1 << 21) + q[:, 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("html", nargs="+", help="一个或多个导出 HTML")
    ap.add_argument("--grid", type=float, default=0.15, help="重叠率用的空间格边长（米）")
    a = ap.parse_args()

    ifo = []
    for p in a.html:
        fp = Path(p)
        if not fp.is_file():
            raise SystemExit(f"[fail] 没有这个文件：{fp}")
        d = load(fp)
        ifo.append(d)
        P_ = d["pts"]
        print(f"=== {fp.name} ===")
        print(f"  点数        {d['n']:,}")
        print(f"  体素        {d['vox']}")
        print(f"  覆盖        {P_[:,0].max()-P_[:,0].min():.1f} × {P_[:,1].max()-P_[:,1].min():.1f} × {P_[:,2].max()-P_[:,2].min():.1f} m")
        print(f"  中心        ({(P_[:,0].min()+P_[:,0].max())/2:.2f}, {(P_[:,1].min()+P_[:,1].max())/2:.2f})")
        print(f"  渲染特征    投影点径={d['new_sprite']}  显示门槛={d['hit_filter']}  gc_mode注释={d['gc_mode']}")
        print(f"  SESS        {d['sess']}")

    if len(ifo) > 1:
        print("=== 两两空间格重叠（<95% ⇒ 至少差一个变量，不可直接比较）===")
        sets = [set(cells(d["pts"], a.grid).tolist()) for d in ifo]
        for i in range(len(ifo)):
            for j in range(i + 1, len(ifo)):
                inter = len(sets[i] & sets[j])
                r1 = 100.0 * inter / max(len(sets[i]), 1)
                r2 = 100.0 * inter / max(len(sets[j]), 1)
                tag = "OK" if min(r1, r2) >= 95 else "⚠️ 不可比"
                print(f"  {ifo[i]['path'].name} ∩ {ifo[j]['path'].name}: "
                      f"{ifo[i]['n']:,}格 -> 覆盖{ifo[j]['n']:,}格的 {r2:.1f}% / 反向 {r1:.1f}%  {tag}")
        print()
        print("判读：重叠率低不一定是错，但**比较前**必须知道差在哪（帧集/体素/闸/位姿）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
