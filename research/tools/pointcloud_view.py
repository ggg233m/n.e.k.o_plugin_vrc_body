#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""录制档稠密点云 → **可交互 3D 查看器**（自包含 HTML，离线可开）。

为什么要有它
------------
2D 栅格看不出"重建得准不准"：两张糊图在 0.1 m 格上都是灰的，但一个是对的、一个是漂移吹大的。
多场点云按**会话着色**投到同一世界系后，对齐错没错一眼可见——错的会分层/重影，对的会重合。
这是目前唯一能让人（而不是另一套自证脚本）判断精度的手段。

数据来源的硬边界（2026-10-06 查证）
------------------------------------
* **稠密点云只存在于 ``navmesh_recordings/<rec>/kf/*.npz`` 的 ``pts``**（每帧 3–4.6 万点）。
* 在线会话 ``navmesh_memory/<world>/sessions/<sid>/kf/*.npz`` **只有稀疏特征** ``xyz``（~280 点/帧），
  没有 ``pts`` ⇒ 在线跑**不产生几何**，只产生跨会话约束。想让地图长高必须走录制档。
* 位姿取 ``session_pose_table`` 的 merged（世界树并树后的同一坐标系）；会话没并树 ⇒ 拒（不猜）。

用法
----
    python research/tools/pointcloud_view.py --world wrld_home-7cf435ea \
        --rec 20261001_044153 20261005_235232 20261006_001523 \
        --session 20261001_044153 20261005_235237 20261006_001523 \
        --vox 0.06 --out .tmp/pointcloud/world.html

``--session`` 与 ``--rec`` 一一对应（录制名与会话 sid 不同名时必须指名，同 offline_fusion）。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_mapping import MapperConfig, _ground_model  # noqa: E402
from backend.nav_xsession import session_pose_table  # noqa: E402

_CFG = MapperConfig()

# 量化参数：xy 用 mm（Uint16 上限 65.5 m），z 用 2 cm（Uint8 覆盖 5.1 m）
XY_OFFSET_M = 32.0
XY_SCALE = 1000.0
Z_SCALE = 50.0
# ⚠️ z 必须是**有符号**编码。原来 `clip(z*50,0,255)` 把地板以下的点全压成 0 层：
# 三场实测 h<0 占 20–63%（见 §地面口径），负高度会被永久糊成一张平板，误导所有目视判断。
# 偏移 1.0 m ⇒ 可表示 −1.00 … +4.10 m，够房间尺度。
Z_OFFSET_M = 1.0
Z_MIN_M = -Z_OFFSET_M
Z_MAX_M = (255.0 / Z_SCALE) - Z_OFFSET_M


def voxelize(xyz: np.ndarray, sid: np.ndarray, vox: float,
             extra: np.ndarray | None = None) -> tuple:
    """体素降采样：同格取**质心**，sid/extra 取格内首次出现的（稳定、可复现）。"""
    if not len(xyz):
        return (xyz, sid) if extra is None else (xyz, sid, extra)
    q = np.floor(xyz / vox).astype(np.int64)
    # 三轴各留 21 位（±1M 格 ⇒ ±64 km @1 cm），够任何房间尺度
    key = (q[:, 0] * (1 << 42)) + (q[:, 1] * (1 << 21)) + q[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    acc = np.zeros((n, 3), np.float64)
    np.add.at(acc, inv, xyz)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    center = acc / cnt[:, None]
    # sid 众数：把 sid 编进同一个 bincount 的权重里不可靠 ⇒ 取每格最小 sid（稳定、可复现）
    sid_out = np.zeros(n, np.uint8)
    sid_out[inv[::-1]] = sid[::-1]        # 逆序写 ⇒ 最终留下首次出现的（可复现）
    if extra is None:
        return center, sid_out
    ex_out = np.zeros(n, np.float32)
    ex_out[inv[::-1]] = np.asarray(extra, np.float32)[::-1]
    return center, sid_out, ex_out


def estimate_cam_h(rec_dir: Path, every: int = 10) -> float:
    """逐场估计相机离地高度——**必须逐场估，不能全场共用一个常数**。

    三场实测 1.756 / 1.483 / 1.438 m（见 prior.json 的 cam_h）：共用一个值会让地面
    整体抬高/沉底（差 0.3 m ≈ 2 个体素），按会话着色时表现为**整场分层**，看起来
    像对齐错了，其实只是标定常数错了。

    算法与 ``offline_fusion.estimate_cam_h`` 同口径：取相机下方 [1.2,2.6] m 带内
    ``pts[:,2]`` 的 1 cm 直方图众数，跨帧按点数加权取中位。
    """
    cands, wts = [], []
    kfs = sorted(rec_dir.glob("kf/*.npz"), key=lambda p: int(p.stem))
    for f in kfs[::max(1, every)]:
        with np.load(f) as z:
            if "pts" not in z.files:
                continue
            rz = np.asarray(z["pts"], np.float32)[:, 2]
        m = (rz >= -2.6) & (rz <= -1.2)
        if int(m.sum()) < 100:
            continue
        bins = np.floor(rz[m] / 0.01).astype(np.int64)
        u, cnt = np.unique(bins, return_counts=True)
        band = np.abs(bins - u[np.argmax(cnt)]) <= 6          # 众数 ±6 cm
        cands.append(-float(np.median(rz[m][band])))
        wts.append(float(m.sum()))
    if not cands:
        return 1.6
    cands = np.asarray(cands)
    wts = np.asarray(wts, float)
    order = np.argsort(cands)
    cw = np.cumsum(wts[order])
    return float(cands[order][np.searchsorted(cw, cw[-1] / 2)])


def prior_cam_h(world_dir: Path) -> dict[str, float]:
    """从 ``<world>/prior/prior.json`` 取已标定的逐场相机高（与融合管线同口径）。

    ``offline_fusion`` 建先验时把每场的 cam_h 记在 ``cam_h`` 数组里，顺序与 ``sessions``
    一致。查看器复用它才能与先验对齐；没有先验（或该场不在先验里）才退回就地估计。
    """
    fp = world_dir / "prior" / "prior.json"
    if not fp.exists():
        return {}
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        sids = [s.get("sid") for s in d.get("sessions", [])]
        hs = [float(h) for h in d.get("cam_h", [])]
    except (ValueError, TypeError, OSError):
        return {}
    if len(sids) != len(hs):
        return {}
    return {s: h for s, h in zip(sids, hs) if s}


def frame_ground_offset(pts: np.ndarray, T: np.ndarray, cam_h: float,
                        every: int = 4) -> float:
    """单关键帧的**相机正下方地面修正量**（追踪米），与 mapper `_ray_voxels` 同一口径。

    为什么必须有它（2026-10-06 实测，本文档 §地面口径）
    --------------------------------------------------
    `cam_h` 是**全局**量（跨关键帧共享、带 0.06 m 滞回），逐场会被估偏：三场实测
    `cam_h = 1.756 / 1.483 / 1.438`。而 `_ground_model` 的第二返回值（相机正下方修正）
    恰好把这个偏差吸收掉 —— `cam_h − gc` = **1.761 / 1.759 / 1.762**（±2 mm，同一 avatar）。

    ⇒ `cam_h` 与 `gc` 是**一对**：`h = (rel_z + cam_h) − gc`。只用 `cam_h` 会让各场地面
    差 0.27 m（按场着色就是**分层**，看起来像对齐错了）；只用 `gc` 则丢掉绝对基准。
    `offline_fusion.kf_rows` 两者都用（对），本查看器原先只用 `cam_h`（错）。
    """
    p = np.asarray(pts[::max(1, int(every))], np.float64)
    if len(p) < 300:
        return 0.0
    vkey = (np.floor(p[:, 0] / _CFG.vox_xy_m).astype(np.int64) * (1 << 42)
            + np.floor(p[:, 1] / _CFG.vox_xy_m).astype(np.int64) * (1 << 21)
            + np.floor(p[:, 2] / _CFG.vox_z_m).astype(np.int64))
    u, inv, cnt = np.unique(vkey, return_inverse=True, return_counts=True)
    cen = np.zeros((len(u), 3), np.float64)
    np.add.at(cen, inv, p)
    cen /= cnt[:, None]
    R = np.asarray(T, np.float64)[:3, :3]
    pw = cen @ R.T
    _gc, gc_cam = _ground_model(pw[:, 2] + cam_h, cnt.astype(np.float64), pw[:, :2],
                                _CFG.ground_offset_max_m, _CFG.ground_offset_min_range_m,
                                _CFG.ground_plane_max_deg, _CFG.ground_plane_band_m)
    return float(gc_cam)


def pose_table_rigid(world_dir: Path, sid: str) -> tuple[dict, str]:
    """会话原表位姿 **只加并树的刚体分量**（与 `offline_fusion --pose rigid` 逐位同口径）。

    ⚠️ 默认必须用 rigid，不是 merged（2026-10-06 实测 + 仓库自己的 A/B）
    ----------------------------------------------------------------
    `ROADMAP` 2026-10-06 的 A/B 结论是「**刚体换系零代价、弹性形变会拉坏内部自洽会话的
    多视一致性**」⇒ 融合/固化档（`prior.json` 的 `pose_mode` 就是 `rigid`）都用 rigid。
    本查看器原先硬编码 `session_pose_table`（= merged 弹性位姿），于是**同一份数据、
    两张不一样的图**：对同一个人来说这是最坏的一种不一致。
    实测两者范围只差 ±5%（不是本次"一坨"的主因），但口径必须与先验一致。
    """
    raw_path = world_dir / "sessions" / sid / "poses.npz"
    got = session_pose_table(world_dir, sid)
    if got is None:
        raise SystemExit(f"[fail] 会话 {sid} 没有位姿表（没并树？先跑 tools/xsession_align.py --world-tree）")
    if str(got.get("source", "")).startswith("raw"):
        raise SystemExit(f"[fail] 会话 {sid} 的位姿是 raw（未并进世界树）⇒ 不猜坐标系，先并树")
    with np.load(raw_path) as z:
        raw = {int(k): np.asarray(T, np.float64) for k, T in zip(z["ids"], z["T_map"])}
    merged = {int(k): np.asarray(T, np.float64) for k, T in zip(got["ids"], got["T_map"])}
    common = sorted(set(raw) & set(merged))
    if len(common) < 10:
        raise SystemExit(f"[fail] 会话 {sid} 原表与 merged 表交集太少（{len(common)}）")
    P = np.array([raw[k][:2, 3] for k in common])
    Q = np.array([merged[k][:2, 3] for k in common])
    sol, *_ = np.linalg.lstsq(np.column_stack([P, np.ones(len(P))]), Q, rcond=None)
    M, t2 = sol[:2], sol[2]
    U, _s, Vt = np.linalg.svd(M)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        R2 = U @ np.diag([1.0, -1.0]) @ Vt
    t2 = Q.mean(0) - P.mean(0) @ R2                 # 旋转定死后取平移（与 offline_fusion 一致）
    Rz = np.eye(3)
    Rz[:2, :2] = R2.T
    out = {}
    for k, T in raw.items():
        M4 = T.copy()
        M4[:2, 3] = T[:2, 3] @ R2 + t2
        M4[:3, :3] = Rz @ T[:3, :3]
        out[k] = M4
    resid = float(np.median(np.linalg.norm(P @ R2 + t2 - Q, axis=1)))
    return out, f"rigid({got.get('source')}, 非刚体残余中位 {resid:.3f} m, base={str(got.get('base_sid'))[-6:]})"


def load_session(world_dir: Path, rec: str, sid: str, vox: float, cam_h: float,
                 z_min: float, z_max: float, every: int, pose_mode: str = "rigid",
                 range_m: float | None = None):
    """一场录制 → 世界系点云（已体素降采样）。返回 (xyz, sid_idx, 统计)。"""
    if pose_mode == "merged":
        tab = session_pose_table(world_dir, sid)
        if tab is None:
            raise SystemExit(f"[fail] 会话 {sid} 没有位姿表（没并树？先跑 tools/xsession_align.py --world-tree）")
        if str(tab.get("source", "")).startswith("raw"):
            raise SystemExit(f"[fail] 会话 {sid} 的位姿是 raw（未并进世界树）⇒ 不猜坐标系，先并树")
        pose_of = {int(k): np.asarray(T, np.float64) for k, T in zip(tab["ids"], tab["T_map"])}
        pose_src = f"merged({tab.get('source')}, base={str(tab.get('base_sid'))[-6:]})"
    else:
        pose_of, pose_src = pose_table_rigid(world_dir, sid)

    rec_dir = ROOT / "navmesh_recordings" / rec
    if not rec_dir.is_dir():
        raise SystemExit(f"[fail] 没有这个录制：{rec_dir}")
    kfs = sorted(rec_dir.glob("kf/*.npz"), key=lambda p: int(p.stem))
    if not kfs:
        raise SystemExit(f"[fail] 录制 {rec} 的 kf/ 是空的")

    chunks, sids, tnorm, n_pts, n_used = [], [], [], 0, 0
    n_neg_raw = n_gc_used = n_far_drop = 0
    gc_vals = []
    t0 = time.perf_counter()
    for kf in kfs:
        k = int(kf.stem)
        if k % every:
            continue
        T = pose_of.get(k)
        if T is None:
            continue
        with np.load(kf) as z:
            if "pts" not in z.files:
                continue
            p = np.asarray(z["pts"], np.float32)
            # 时间进度用累积路程（kf 自带 dist_m），不用帧号：帧间隔随运动快慢不均
            dm = float(np.asarray(z["dist_m"]).reshape(-1)[0]) if "dist_m" in z.files else float(k)
        n_pts += len(p)
        # ⚠️ 相机距离闸：mapper 建图只在 range_m(=5.0 m) 内取点，**导出必须同口径**。
        # 实测（2026-10-06）：23–26% 的点落在 5–8 m、另有 2–2.7% 在 8 m 以上。
        # 双目深度误差按 z² 增长（§10.8：5 m 处约 0.1 m/像素），远带那批点既稀疏又偏，
        # 投到地图上就是**从相机位置向外发散的尖刺**（俯视图里最扎眼的那一圈）。
        # 它们不是几何，是噪声；mapper 早就把它们挡在门外了，导出必须一致。
        if range_m is not None:
            rng_ok = np.isfinite(p).all(axis=1) & (np.linalg.norm(p, axis=1) <= float(range_m))
            n_far_drop += int(len(p) - rng_ok.sum())
            p = p[rng_ok]
            if len(p) == 0:
                continue
        # h = (rel_z + cam_h) − gc：cam_h 与 gc 是**一对**，只用 cam_h 会让各场地面差 0.27 m
        gc = frame_ground_offset(p, T, cam_h)
        gc_vals.append(gc)
        pw = p @ T[:3, :3].T                       # 相机系 → 世界（旋转）
        xy = pw[:, :2] + T[:2, 3]                  # 平移只取 xy（z 走地面高度口径）
        zw = pw[:, 2] + cam_h - gc                 # 与 offline_fusion.kf_rows 同口径
        sel = (zw > z_min) & (zw < z_max)
        sel &= np.isfinite(xy).all(axis=1) & np.isfinite(zw)
        if not sel.any():
            continue
        n_neg_raw += int((zw < 0).sum())
        n_gc_used += int(sel.sum())
        n_used += int(sel.sum())
        chunks.append(np.column_stack([xy[sel], zw[sel]]))
        tnorm.append(np.full(int(sel.sum()), dm, np.float32))
    if not chunks:
        raise SystemExit(f"[fail] 录制 {rec} 没有可用点（位姿表与 kf 对不上？）")
    allp = np.concatenate(chunks)
    allt = np.concatenate(tnorm)
    allt = np.clip(allt / max(float(allt.max()), 1e-6), 0.0, 1.0)   # 归一化到 0..1
    c, s, t = voxelize(allp, np.zeros(len(allp), np.uint8), vox, allt)
    return c, s, t, dict(rec=rec, sid=sid, n_kf=len(kfs), n_pts_raw=n_pts, n_pts_kept=n_used,
                      n_vox=len(c), s=round(time.perf_counter() - t0, 1),
                      pose_src=pose_src, base=str(pose_src).split("base=")[-1],
                      cam_h=round(float(cam_h), 3),
                      gc_median=round(float(np.median(gc_vals)), 3) if gc_vals else None,
                      cam_h_eff=round(float(cam_h - (np.median(gc_vals) if gc_vals else 0.0)), 3),
                      neg_share=round(n_neg_raw / max(n_gc_used, 1), 4),
                      range_m=(float(range_m) if range_m is not None else None),
                      far_dropped=n_far_drop)


_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:__BG__;color:__FG__;font:13px/1.6 system-ui,"Segoe UI",sans-serif}
#cv{display:block;width:100vw;height:100vh;cursor:grab}
#cv:active{cursor:grabbing}
#hud{position:fixed;left:14px;top:14px;background:__PANEL__;border:1px solid __BORDER__;
     border-radius:10px;padding:12px 14px;max-width:330px;backdrop-filter:blur(8px)}
#hud h1{margin:0 0 8px;font-size:14px;font-weight:600}
.kv{display:flex;justify-content:space-between;gap:12px;color:__MUTED__}
.kv b{color:__FG__;font-weight:600}
#legend{margin-top:9px;border-top:1px solid __BORDER__;padding-top:8px}
.row{display:flex;align-items:center;gap:7px;margin:3px 0}
.sw{width:11px;height:11px;border-radius:3px;flex:none}
#keys{margin-top:9px;border-top:1px solid __BORDER__;padding-top:8px;color:__MUTED__;font-size:12px}
button{background:__BTN__;color:__FG__;border:1px solid __BORDER__;border-radius:6px;
       padding:4px 9px;margin:2px 3px 0 0;cursor:pointer;font:inherit;font-size:12px}
button.on{background:__BTNON__;border-color:__ACCENT__;color:__ACCENT__}
</style></head><body>
<canvas id="cv"></canvas>
<div id="hud">
  <h1>__TITLE__</h1>
  <div class="kv"><span>点数</span><b id="np">–</b></div>
  <div class="kv"><span>体素</span><b>__VOX__ m</b></div>
  <div class="kv"><span>覆盖</span><b id="ext">–</b></div>
  <div id="legend"></div>
  <div id="keys">
    <button id="bSid" class="on">按会话</button><button id="bH">按高度</button><button id="bT">按时间</button>
    <button id="bTop">俯视</button><button id="bSide">侧视</button>
    <button id="bFit">复位</button><br>
    拖拽旋转 · 滚轮缩放 · 右键平移 · <span style="color:__FG__">G</span> 网格
  </div>
</div>
<script>
const RAW = atob("__DATA__");
const buf = new Uint8Array(RAW.length); for (let i=0;i<RAW.length;i++) buf[i]=RAW.charCodeAt(i);
const N = buf.length/6|0;
const POS = new Float32Array(N*3), SID = new Float32Array(N);
const XYO=__XYO__, XYS=__XYS__, ZS=__ZS__, ZO=__ZO__;
for (let i=0;i<N;i++){
  const o=i*6, u=buf[o]|buf[o+1]<<8, v=buf[o+2]|buf[o+3]<<8;
  POS[i*3]   = u/XYS - XYO;
  POS[i*3+1] = v/XYS - XYO;
  POS[i*3+2] = buf[o+4]/ZS - ZO;      // 有符号：编码前加了 +ZO
  SID[i]     = buf[o+5];
}
document.getElementById('np').textContent = N.toLocaleString();
let mnx=1e9,mxx=-1e9,mny=1e9,mxy=-1e9,mnz=1e9,mxz=-1e9;
for(let i=0;i<N;i++){const x=POS[i*3],y=POS[i*3+1],z=POS[i*3+2];
  if(x<mnx)mnx=x; if(x>mxx)mxx=x; if(y<mny)mny=y; if(y>mxy)mxy=y; if(z<mnz)mnz=z; if(z>mxz)mxz=z;}
document.getElementById('ext').textContent =
  (mxx-mnx).toFixed(1)+' × '+(mxy-mny).toFixed(1)+' × '+(mxz-mnz).toFixed(1)+' m';

const SESS = __SESS__;
const PAL = __PAL__;
const lg = document.getElementById('legend');
SESS.forEach((s,i)=>{const d=document.createElement('div');d.className='row';
  const sw=document.createElement('span');sw.className='sw';sw.style.background=PAL[i%PAL.length];
  const t=document.createElement('span');
  t.textContent=s.name+' · '+s.n.toLocaleString()+' 点'+(s.h!==undefined?' · 相机高 '+s.h.toFixed(2)+' m':'');
  d.appendChild(sw);d.appendChild(t);lg.appendChild(d);});
if(2===__MODE__){const d=document.createElement('div');d.className='row';d.style.marginTop='4px';
  const bar=document.createElement('span');
  bar.style.cssText='width:110px;height:8px;border-radius:4px;flex:none;'+
    'background:linear-gradient(90deg,#2b6fd6,#3fc3ea,#57dba2,#e6d24e,#ef8b3a,#df4436)';
  const lb=document.createElement('span');lb.textContent='早 → 晚（各场内部按累积路程）';
  d.appendChild(bar);d.appendChild(lb);lg.appendChild(d);}

const cv = document.getElementById('cv'), gl = cv.getContext('webgl');
if(!gl){document.body.innerHTML='<p style="padding:20px">此浏览器不支持 WebGL</p>';}
const VS = `
attribute vec3 aP; attribute float aS;
uniform mat4 uMVP; uniform float uMode; uniform vec3 uPal[8]; uniform float uZ0,uZ1;
varying vec3 vC;
vec3 hsv2rgb(vec3 c){ vec4 K=vec4(1.0,2.0/3.0,1.0/3.0,3.0);
  vec3 p=abs(fract(c.xxx+K.xyz)*6.0-K.www);
  return c.z*mix(vec3(1.0),clamp(p-K.xxx,0.0,1.0),c.y); }
void main(){
  gl_Position = uMVP * vec4(aP,1.0);
  float t = clamp((aP.z-uZ0)/max(uZ1-uZ0,1e-6),0.0,1.0);
  vec3 hc = mix(vec3(0.13,0.42,0.87), vec3(0.98,0.72,0.20), t);   // 低→蓝 高→黄
  int k = int(aS+0.5); vec3 sc = uPal[0];
  for(int i=0;i<8;i++){ if(i==k) sc = uPal[i]; }
  // uMode: 0=按会话(调色板) 1=按高度 2=按时间(彩虹:早→蓝,晚→红)
  vec3 tc = hsv2rgb(vec3(0.62*(1.0-clamp(aS/254.0,0.0,1.0)), 0.85, 0.95));
  vC = uMode < 0.5 ? sc : (uMode < 1.5 ? hc : tc);
  gl_PointSize = clamp(240.0/max(gl_Position.w,1.0), 1.5, 5.0);   // 近大远小
}`;
const FS = `precision mediump float; varying vec3 vC;
void main(){ vec2 d=gl_PointCoord-0.5; if(dot(d,d)>0.25) discard; gl_FragColor=vec4(vC,1.0); }`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o)); return o;}
const P = gl.createProgram();
gl.attachShader(P, sh(gl.VERTEX_SHADER,VS)); gl.attachShader(P, sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(P); gl.useProgram(P);
const bP=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,bP); gl.bufferData(gl.ARRAY_BUFFER,POS,gl.STATIC_DRAW);
const aP=gl.getAttribLocation(P,'aP'); gl.enableVertexAttribArray(aP); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
const bS=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,bS); gl.bufferData(gl.ARRAY_BUFFER,SID,gl.STATIC_DRAW);
const aS=gl.getAttribLocation(P,'aS'); gl.enableVertexAttribArray(aS); gl.vertexAttribPointer(aS,1,gl.FLOAT,false,0,0);
const uMVP=gl.getUniformLocation(P,'uMVP'), uMode=gl.getUniformLocation(P,'uMode'),
      uPal=gl.getUniformLocation(P,'uPal'), uZ0=gl.getUniformLocation(P,'uZ0'), uZ1=gl.getUniformLocation(P,'uZ1');
const pal=[]; for(let i=0;i<8;i++){const c=PAL[i%PAL.length].replace('#','');
  pal.push(parseInt(c.slice(0,2),16)/255, parseInt(c.slice(2,4),16)/255, parseInt(c.slice(4,6),16)/255);}
gl.uniform3fv(uPal, new Float32Array(pal)); gl.uniform1f(uZ0,mnz); gl.uniform1f(uZ1,mxz);
gl.enable(gl.DEPTH_TEST);

// 网格（1 m 线，贴在 z 最低处）
let gridB=null,gridN=0,showGrid=true;
{ const z=mnz-0.02, v=[];
  for(let x=Math.floor(mnx);x<=Math.ceil(mxx);x++){v.push(x,mny,z,x,mxy,z);}
  for(let y=Math.floor(mny);y<=Math.ceil(mxy);y++){v.push(mnx,y,z,mxx,y,z);}
  gridN=v.length/3; gridB=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,gridB);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(v),gl.STATIC_DRAW); }

let yaw=0.7, pitch=0.50, dist=0, cx=0, cy=0, cz=0, mode=__MODE__;   // pitch>0 ⇒ 相机在上方俯视
function fit(){ cx=(mnx+mxx)/2; cy=(mny+mxy)/2; cz=(mnz+mxz)/2;
  dist = Math.max(mxx-mnx, mxy-mny, mxz-mnz)*1.6 + 4; }
fit();
function resize(){ const r=window.devicePixelRatio||1;
  cv.width=Math.floor(innerWidth*r); cv.height=Math.floor(innerHeight*r); gl.viewport(0,0,cv.width,cv.height); }
addEventListener('resize',resize); resize();
function persp(f,ar,n,fa){const t=1/Math.tan(f/2);return new Float32Array(
  [t/ar,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0]);}
function draw(){
  const ar=cv.width/cv.height;
  const cp=Math.cos(pitch),sp=Math.sin(pitch),cyw=Math.cos(yaw),syw=Math.sin(yaw);
  const ex=cx+dist*cp*syw, ey=cy-dist*cp*cyw, ez=cz+dist*sp;
  // 右手系、up=+z。V 第三行是 backward(=eye-center)；正交基 x×y=backward ⇒ y=backward×x。
  // ⚠️ 勿改成 z(=center-eye)×x：那是反号，世界 +z 会在屏幕上朝下（2026-10-06 实测翻车）。
  let zx=cx-ex,zy=cy-ey,zz=cz-ez; let l=Math.hypot(zx,zy,zz)||1; zx/=l;zy/=l;zz/=l;
  let xx=cyw, xy=syw, xz=0;
  let yx=zz*xy-zy*xz, yy=zx*xz-zz*xx, yz=zy*xx-zx*xy;   // = (−z)×x
  l=Math.hypot(yx,yy,yz)||1; yx/=l;yy/=l;yz/=l;
  xx=yy*zz-yz*zy; xy=yz*zx-yx*zz; xz=yx*zy-yy*zx;
  const V=new Float32Array([xx,yx,-zx,0, xy,yy,-zy,0, xz,yz,-zz,0,
     -(xx*ex+xy*ey+xz*ez), -(yx*ex+yy*ey+yz*ez), (zx*ex+zy*ey+zz*ez), 1]);
  const Pm=persp(0.9,ar,0.05,dist*8+50);
  const M=new Float32Array(16);                      // 列主序 M = Pm · V
  for(let r=0;r<4;r++)for(let c=0;c<4;c++){let s=0;
    for(let k=0;k<4;k++) s+=Pm[r+k*4]*V[k+c*4]; M[r+c*4]=s;}
  gl.clearColor(__CLR__); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(uMVP,false,M); gl.uniform1f(uMode,mode);
  gl.bindBuffer(gl.ARRAY_BUFFER,bP); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,bS); gl.vertexAttribPointer(aS,1,gl.FLOAT,false,0,0);
  gl.drawArrays(gl.POINTS,0,N);
  if(showGrid&&gridN){ gl.bindBuffer(gl.ARRAY_BUFFER,gridB);
    gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0); gl.vertexAttrib1f(aS,0);
    gl.uniform1f(uMode,0.86); gl.drawArrays(gl.LINES,0,gridN); }
  requestAnimationFrame(draw);
}
draw();
let drag=0,lx=0,ly=0;
cv.addEventListener('mousedown',e=>{drag=e.button===2?2:1;lx=e.clientX;ly=e.clientY;});
addEventListener('mouseup',()=>drag=0);
cv.addEventListener('contextmenu',e=>e.preventDefault());
addEventListener('mousemove',e=>{ if(!drag)return; const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
  if(drag===1){ yaw+=dx*0.006; pitch=Math.max(-0.35,Math.min(1.50,pitch+dy*0.006)); }
  else { const s=dist*0.0016; cx-=Math.cos(yaw)*dx*s+Math.sin(yaw)*dy*s; cy-=Math.sin(yaw)*dx*s-Math.cos(yaw)*dy*s; } });
cv.addEventListener('wheel',e=>{e.preventDefault();dist=Math.max(0.6,dist*(1+Math.sign(e.deltaY)*0.12));},{passive:false});
// ⚠️ 勿用 bS/bH：那是上面的 WebGL buffer 名，重复声明会让整段脚本 SyntaxError（全黑）
const btnSid=document.getElementById('bSid'), btnH=document.getElementById('bH'),
      btnT=document.getElementById('bT');
function setMode(m){mode=m;btnSid.classList.toggle('on',m===0);btnH.classList.toggle('on',m===1);
  btnT.classList.toggle('on',m===2);}
btnSid.onclick=()=>setMode(0); btnH.onclick=()=>setMode(1); btnT.onclick=()=>setMode(2);
setMode(__MODE__);
btnH.onclick=()=>{mode=1;btnH.classList.add('on');btnSid.classList.remove('on');};
document.getElementById('bTop').onclick =()=>{pitch=1.45;yaw=0;};   // 不到 1.5，避免 up 与视线平行退化
document.getElementById('bSide').onclick=()=>{pitch=0.08;yaw=0.7;};
document.getElementById('bFit').onclick =()=>{fit();};
addEventListener('keydown',e=>{ if(e.key==='g'||e.key==='G') showGrid=!showGrid; });
</script></body></html>
"""

_PALETTE = ["#4f9cf0", "#f2a33c", "#59c98d", "#e0655f", "#a97ae0",
            "#3fb8c4", "#e5c04b", "#8f9bb3"]


def build_html(points: np.ndarray, sid: np.ndarray, sess: list[dict], vox: float,
               title: str, default_mode: int = 0) -> str:
    """点云 → 自包含 HTML。布局：每点 6 字节（u16 x, u16 y, u8 z, u8 sid）。"""
    x = np.clip((points[:, 0] + XY_OFFSET_M) * XY_SCALE, 0, 65535).astype(np.uint16)
    y = np.clip((points[:, 1] + XY_OFFSET_M) * XY_SCALE, 0, 65535).astype(np.uint16)
    z = np.clip((points[:, 2] + Z_OFFSET_M) * Z_SCALE, 0, 255).astype(np.uint8)   # 有符号编码
    s = sid.astype(np.uint8)
    # 交错布局必须用结构化 dtype：直接 stack (N,2)+(N,1) 会因形状不同报错
    dt = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "u1"), ("s", "u1")])
    arr = np.empty(len(x), dtype=dt)
    arr["x"], arr["y"], arr["z"], arr["s"] = x, y, z, s
    raw = arr.tobytes()
    b64 = base64.b64encode(raw).decode("ascii")
    html = (_HTML
            .replace("__TITLE__", title)
            .replace("__DATA__", b64)
            .replace("__VOX__", f"{vox:g}")
            .replace("__SESS__", json.dumps(sess, ensure_ascii=False))
            .replace("__PAL__", json.dumps(_PALETTE))
            .replace("__MODE__", str(int(default_mode)))
            .replace("__XYO__", repr(float(XY_OFFSET_M)))
            .replace("__XYS__", repr(float(XY_SCALE)))
            .replace("__ZS__", repr(float(Z_SCALE)))
            .replace("__ZO__", repr(float(Z_OFFSET_M)))
            .replace("__BG__", "#0f1115").replace("__FG__", "#e8eaed")
            .replace("__PANEL__", "rgba(24,27,33,.88)").replace("__BORDER__", "#2c313a")
            .replace("__MUTED__", "#98a0ac").replace("__BTN__", "#1b1f26")
            .replace("__BTNON__", "#20304a").replace("__ACCENT__", "#7fb4ff")
            .replace("__CLR__", "0.059, 0.067, 0.082, 1"))
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True, help="世界名（navmesh_memory/ 下）或绝对路径")
    ap.add_argument("--rec", nargs="+", required=True, help="录制目录名（navmesh_recordings/ 下）")
    ap.add_argument("--session", nargs="+", default=None,
                    help="会话 sid，与 --rec 一一对应（空格或逗号分隔均可）")
    ap.add_argument("--vox", type=float, default=0.06, help="体素边长（米），默认 0.06")
    ap.add_argument("--every", type=int, default=1, help="每 N 帧取 1 帧（抽稀，默认全取）")
    ap.add_argument("--cam-h", default="auto",
                    help="相机离地高度：auto=逐场估计（推荐，默认）；或显式米数，逗号/空格分隔，"
                         "给 1 个则全场共用，给 N 个则与 --rec 一一对应")
    ap.add_argument("--z-min", type=float, default=-1.0, help="保留的地面高度下限（米，默认 −1.0）")
    ap.add_argument("--z-max", type=float, default=4.0, help="保留的地面高度上限（米，默认 4.0）")
    ap.add_argument("--pose", choices=("rigid", "merged"), default="rigid",
                    help="位姿口径：rigid=原表只加并树刚体分量（默认，与 prior.json 的 pose_mode 一致）；"
                         "merged=弹性并树位姿（仅用于对照，A/B 已裁定它会拉坏多视一致性）")
    ap.add_argument("--range-m", type=float, default=5.0,
                    help="相机距离上限（米），与 mapper.range_m 同口径；默认 5.0。"
                         "双目深度误差按 z² 增长，远带点会变成放射状尖刺。给 0 表示不过滤（仅诊断用）")
    ap.add_argument("--max-pts", type=int, default=1_200_000, help="点数上限，超了自动加粗体素")
    ap.add_argument("--color-by", choices=("session", "time"), default="session",
                    help="着色：session=按场次（看跨场对齐）；time=按本场累积路程（看漂移展开，"
                         "同一处若被多次经过而画在不同位置，会呈现不同颜色叠在一起）")
    ap.add_argument("--out", default=None, help="输出 HTML（默认 .tmp/pointcloud/<world>.html）")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    wdir = Path(args.world)
    if not wdir.is_absolute():
        wdir = ROOT / "navmesh_memory" / wdir
    if not wdir.is_dir():
        raise SystemExit(f"[fail] 没有这个世界目录：{wdir}")
    sids: list[str] = []
    for chunk in (args.session or []):
        sids.extend(s.strip() for s in chunk.split(",") if s.strip())
    if not sids:
        sids = list(args.rec)
    if len(sids) != len(args.rec):
        raise SystemExit(f"[fail] --session 给了 {len(sids)} 个，--rec 有 {len(args.rec)} 个（一一对应）")

    # cam_h 必须逐场估：三场实测 1.756 / 1.483 / 1.438，共用常数 ⇒ 整场地面抬高/沉底 ⇒ 假分层
    if str(args.cam_h).strip().lower() == "auto":
        pm = prior_cam_h(wdir)
        cam_hs, srcs = [], []
        for rec, sid in zip(args.rec, sids):
            if sid in pm:
                cam_hs.append(pm[sid]); srcs.append("prior")
            else:
                cam_hs.append(estimate_cam_h(ROOT / "navmesh_recordings" / rec)); srcs.append("估计")
        print("[cam_h] " + ", ".join(f"{r[-6:]}={h:.3f}({s})" for r, h, s in zip(args.rec, cam_hs, srcs)))
    else:
        vals = [float(v) for v in str(args.cam_h).replace(",", " ").split() if v]
        if len(vals) == 1:
            cam_hs = vals * len(args.rec)
        elif len(vals) != len(args.rec):
            raise SystemExit(f"[fail] --cam-h 给了 {len(vals)} 个，--rec 有 {len(args.rec)} 个")
        else:
            cam_hs = vals

    vox = float(args.vox)
    by_time = str(args.color_by).strip().lower() == "time"
    allp, alls, allc, sess, stats = [], [], [], [], []
    for i, (rec, sid) in enumerate(zip(args.rec, sids)):
        pts, s, tt, st = load_session(wdir, rec, sid, vox, float(cam_hs[i]),
                                      float(args.z_min), float(args.z_max), max(1, int(args.every)),
                                      pose_mode=str(args.pose),
                                      range_m=(None if float(args.range_m) <= 0 else float(args.range_m)))
        # 着色通道：按时间 = 本场累积路程进度 0..254；按会话 = 场索引（shader 取调色板）
        col = (tt * 254.0).astype(np.float32) if by_time else np.full(len(pts), float(i), np.float32)
        allp.append(pts)
        alls.append(np.full(len(pts), i, np.uint8))
        allc.append(col)
        sess.append({"name": f"{sid[-6:]}（{rec[-6:]}）", "n": len(pts), "h": round(float(st["cam_h_eff"]), 3)})
        stats.append(st)
        print(f"[load ] {rec} → {sid}  cam_h={st['cam_h']:.3f} − gc={st['gc_median']:+.3f} "
              f"⇒ 有效离地 {st['cam_h_eff']:.3f}  kf={st['n_kf']}  原始点 {st['n_pts_raw']:,}  "
              f"距离闸丢 {st['far_dropped']:,}（{st['far_dropped']/max(st['n_pts_raw'],1)*100:.1f}%，"
              f"range={st['range_m']} m）  保留 {st['n_pts_kept']:,}  "
              f"(h<0 占 {st['neg_share']*100:.1f}%)  体素后 {st['n_vox']:,}  "
              f"({st['s']}s)  pose={st['pose_src']}", flush=True)

    P = np.concatenate(allp)
    S = np.concatenate(alls)
    C = np.concatenate(allc)

    # 🚨 跨场「有效离地高」一致性闸（2026-10-06 新增，起因见本文档 §地面口径）
    # cam_h 是**全局估计**，逐场可以被估偏 0.3 m；gc 会把它补回来，于是 cam_h−gc 才是
    # 真正跨场可比的量。同一个 avatar 下它必须几乎相同 —— 差得多只有两种解释：
    # ① 真的换了 avatar/身高（那 world_scale 也该跟着重标，见 漂移形态诊断 §10.10）；
    # ② 地面修正失效（地面证据不足、或该场地面被 UGC 挡住）。
    # 两种情况都不许静默出图。
    eff = [(st["sid"], float(st["cam_h_eff"])) for st in stats if st.get("cam_h_eff") is not None]
    if len(eff) >= 2:
        lo = min(v for _s, v in eff)
        hi = max(v for _s, v in eff)
        spread = hi - lo
        tag = "OK  " if spread <= 0.05 else ("WARN" if spread <= 0.15 else "FAIL")
        detail = ", ".join(f"{s[-6:]}={v:.3f}" for s, v in eff)
        print(f"[cam_h] {tag} 有效离地高跨场 {detail} ⇒ 极差 {spread*1000:.0f} mm")
        if spread > 0.05:
            print(f"[cam_h] ⚠️ 极差 {spread:.3f} m 超出量化噪声（2 cm）。先查是不是换了 avatar/身高"
                  f"（换身高要重标 world_scale = 0.755×(眼高/1.26)），再查地面证据是否够。"
                  f"**不要**直接改 cam_h 去凑 —— 它和 gc 是一对。")
    # 跨会话再并一次体素：多场重叠区只留一份（sid 取先到的 ⇒ 图例按场次顺序稳定）
    while len(P) > int(args.max_pts):
        vox *= 1.5
        P, S, C = voxelize(P, S, vox, C)
        print(f"[thin ] 超过上限 {int(args.max_pts):,} ⇒ 体素加粗到 {vox:.3f} m（现 {len(P):,} 点）", flush=True)
    P, S, C = voxelize(P, S, vox, C)

    # 图例数字必须是**合并后**各场实际显示的点数：合并时同格只留一份（sid 取先到的），
    # 所以各场"体素后点数"之和 ≠ 显示点数（重叠区被先加载的场吃掉）。
    for i, s in enumerate(sess):
        s["n"] = int(np.count_nonzero(S == i))

    out = Path(args.out) if args.out else (ROOT / ".tmp" / "pointcloud" / f"{wdir.name}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or f"{wdir.name} · {len(sess)} 场稠密点云"
    out.write_text(build_html(P, C.astype(np.uint8), sess, vox, title,
                              default_mode=2 if by_time else 0), encoding="utf-8")
    print(f"[out  ] {out}  ({out.stat().st_size/1048576:.1f} MB · {len(P):,} 点 · 体素 {vox:.3f} m)")
    print(json.dumps({"sessions": stats, "n_pts": len(P), "vox_m": round(vox, 4)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
