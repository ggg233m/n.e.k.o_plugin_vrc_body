#!/usr/bin/env python
"""先验栅格俯视图渲染器（2D 三色 + 轨迹叠加）。

为什么要有这个工具：稠密点云是**未过滤的原始观测**（空中噪声占一半），
拿它判断"地图对不对"会被误导。经过判据过滤的产物是 ``prior.npz`` 的
``labels``（UNKNOWN / FREE / OCC 三态），那才是"真正的地图"。

产出：
  * 自包含 HTML（canvas 渲染，滚轮缩放 / 拖拽平移 / 鼠标读世界坐标）
  * 可选 PNG（纯 zlib 手写，零第三方依赖）

用法::

    python research/tools/prior_topview.py --world wrld_home-7cf435ea
    python research/tools/prior_topview.py --world wrld_home-7cf435ea --png --crop

坐标系：``labels`` 行=y、列=x，格心 = origin + (i+0.5)·res。
渲染时世界 y 向上（与数学惯例一致，也与 nav_grid 一致）。
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_prior import L_FREE, L_OCC, L_UNK, PRIOR_JSON, PRIOR_NPZ, prior_dir  # noqa: E402
from backend.nav_xsession import session_pose_table  # noqa: E402

# 三态配色（light theme）
COL_UNK = (236, 236, 240)    # 未知：浅灰
COL_FREE = (255, 255, 255)   # 可走：白
COL_OCC = (47, 59, 82)       # 障碍：深蓝灰

# 轨迹配色（与 pointcloud_view.py 的调色板一致，便于跨工具对照）
TRAJ_PAL = ["#4a7fd4", "#f2a33c", "#4fb06a", "#c75b8a", "#8a6fd4",
            "#3fa8b0", "#d4803c", "#7a8a3c"]


# --------------------------------------------------------------------------- IO

def load_prior(world_dir: Path) -> dict:
    d = prior_dir(world_dir)
    npz_p, json_p = d / PRIOR_NPZ, d / PRIOR_JSON
    if not npz_p.exists() or not json_p.exists():
        raise SystemExit(f"[fail] 缺先验文件：{d}（先跑 offline_fusion.py --write-prior）")
    z = np.load(npz_p, allow_pickle=True)
    meta = json.loads(json_p.read_text(encoding="utf-8"))
    lab = np.asarray(z["labels"], np.uint8)
    if lab.ndim != 2:
        raise SystemExit(f"[fail] labels 形状异常 {lab.shape}")
    # npz 与 json 的 origin 必须一致（不同口径 ⇒ 拒绝渲染，避免静默画歪）
    o_npz = np.asarray(z["origin_xy"], np.float64).reshape(2)
    o_json = np.asarray(meta.get("origin_xy", o_npz), np.float64).reshape(2)
    if not np.allclose(o_npz, o_json, atol=1e-6):
        raise SystemExit(f"[fail] origin_xy 不一致：npz={o_npz.tolist()} json={o_json.tolist()}")
    if list(lab.shape) != list(meta.get("shape", lab.shape)):
        raise SystemExit(f"[fail] shape 不一致：npz={lab.shape} json={meta.get('shape')}")
    return dict(labels=lab, origin=o_npz, res=float(z["res_m"]), meta=meta,
                base_sid=str(z["base_sid"]), scale=float(z["world_scale"]),
                npz_path=npz_p, json_path=json_p, dir=d)


def load_trajs(world_dir: Path, meta: dict) -> list[dict]:
    """按 prior.json 的 sessions 顺序取各场轨迹（世界系 xy）。"""
    out = []
    for i, s in enumerate(meta.get("sessions", [])):
        sid = str(s.get("sid", ""))
        if not sid:
            continue
        try:
            tab = session_pose_table(Path(world_dir), sid)
        except Exception as e:                                  # noqa: BLE001
            print(f"[warn ] {sid} 位姿读取失败：{type(e).__name__}: {e}")
            continue
        if not tab or tab.get("T_map") is None or len(tab.get("ids", [])) == 0:
            print(f"[warn ] {sid} 无 T_map（未并树？）⇒ 轨迹跳过")
            continue
        T = np.asarray(tab["T_map"], np.float64)
        if T.ndim != 3 or T.shape[1] < 3:
            print(f"[warn ] {sid} T_map 形状异常 {T.shape} ⇒ 跳过")
            continue
        xy = T[:, :2, 3]
        if len(xy) < 2:
            continue
        out.append(dict(name=f"{sid[-6:]}", sid=sid, rec=str(s.get("rec", "")),
                        color=TRAJ_PAL[i % len(TRAJ_PAL)],
                        pts=np.round(xy, 3).tolist(),
                        src=str(tab.get("source", "?"))))
    return out


# ------------------------------------------------------------------- PNG writer

def write_png(path: Path, rgb: np.ndarray) -> None:
    """纯 zlib 手写 PNG（RGB8），零第三方依赖。"""
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def rasterize(lab: np.ndarray, trajs: list[dict], origin: np.ndarray, res: float,
              bbox: tuple[float, float, float, float] | None, px_per_m: float,
              draw_traj: bool) -> np.ndarray:
    """把栅格（+轨迹）光栅化成 RGB 图像，用于 PNG 输出。"""
    H, W = lab.shape
    x0, y0, x1, y1 = bbox if bbox else (origin[0], origin[1],
                                        origin[0] + W * res, origin[1] + H * res)
    iw = max(8, int(round((x1 - x0) * px_per_m)))
    ih = max(8, int(round((y1 - y0) * px_per_m)))

    # 逐像素反查格下标（最近邻、无整除误差）。图像行向下 ⇒ 世界 y 递减
    px_x = np.arange(iw, dtype=np.float64)
    wx = x0 + (px_x + 0.5) / px_per_m
    cc = np.floor((wx - origin[0]) / res).astype(np.int64)
    py_y = np.arange(ih, dtype=np.float64)
    wy = y0 + (ih - 1 - py_y + 0.5) / px_per_m
    rr = np.floor((wy - origin[1]) / res).astype(np.int64)
    CC, RR = np.meshgrid(cc, rr)                       # (ih, iw)
    inside = (CC >= 0) & (CC < W) & (RR >= 0) & (RR < H)
    idx = np.clip(RR, 0, H - 1) * W + np.clip(CC, 0, W - 1)
    v = lab.reshape(-1)[idx]
    img = np.where(inside[..., None],
                   np.where(v[..., None] == L_FREE, COL_FREE,
                            np.where(v[..., None] == L_OCC, COL_OCC, COL_UNK)),
                   COL_UNK).astype(np.uint8)

    if draw_traj:
        for t in trajs:
            col = np.array([int(t["color"][k:k + 2], 16) for k in (1, 3, 5)], np.uint8)
            pts = np.asarray(t["pts"], np.float64)
            px = ((pts[:, 0] - x0) * px_per_m).astype(np.int32)
            py = (ih - 1 - ((pts[:, 1] - y0) * px_per_m)).astype(np.int32)
            for a, b in zip(zip(px[:-1], py[:-1]), zip(px[1:], py[1:])):
                _line(img, a, b, col)
    return img


def _line(img: np.ndarray, a, b, col: np.ndarray) -> None:
    """Bresenham，粗 2px（画两次错开）。"""
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    H, W = img.shape[:2]
    while True:
        for ox, oy in ((0, 0), (1, 0), (0, 1)):
            xx, yy = x0 + ox, y0 + oy
            if 0 <= xx < W and 0 <= yy < H:
                img[yy, xx] = col
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


# ------------------------------------------------------------------------ HTML

HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root{--bg:#f7f7f9;--panel:#ffffff;--line:#dcdce2;--tx:#22222a;--dim:#6b6b78}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--tx);
       font:13px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  header{padding:12px 18px 10px;border-bottom:1px solid var(--line);background:var(--panel)}
  h1{margin:0 0 4px;font-size:15px;font-weight:600}
  .sub{color:var(--dim);font-size:12px}
  .sub b{color:var(--tx);font-weight:600}
  #wrap{display:flex;height:calc(100vh - 92px)}
  #cv{flex:1 1 0;min-width:0;height:100%;display:block;cursor:grab;background:#e9e9ee}
  #cv.drag{cursor:grabbing}
  aside{width:236px;flex:none;border-left:1px solid var(--line);background:var(--panel);
        padding:12px 14px;overflow:auto}
  .lg{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px}
  .sw{width:16px;height:16px;border:1px solid var(--line);border-radius:3px;flex:none}
  h2{font-size:12px;margin:14px 0 6px;color:var(--dim);font-weight:600;
     text-transform:uppercase;letter-spacing:.05em}
  h2:first-child{margin-top:0}
  .kv{display:flex;justify-content:space-between;gap:8px;font-size:12px;padding:2px 0}
  .kv span:last-child{font-variant-numeric:tabular-nums;color:var(--dim)}
  .btns{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
  button{border:1px solid var(--line);background:#fff;color:var(--tx);border-radius:5px;
         padding:4px 9px;font-size:12px;cursor:pointer}
  button:hover{background:#f0f0f5}
  button.on{background:#2f3b52;color:#fff;border-color:#2f3b52}
  .tip{font-size:11px;color:var(--dim);margin-top:10px;line-height:1.7}
  #pos{position:fixed;left:50%;transform:translateX(-50%);bottom:12px;background:rgba(34,34,42,.86);
       color:#fff;padding:4px 12px;border-radius:14px;font-size:12px;pointer-events:none;
       font-variant-numeric:tabular-nums;opacity:0;transition:opacity .12s}
  #pos.show{opacity:1}
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">先验栅格 · <b>__SHAPE__</b> 格 @ <b>__RES__</b> m ·
    尺度 <b>__SCALE__</b> · 根会话 <b>__BASE__</b> · 产出 __BUILT__ · __NPATH__</div>
</header>
<div id="wrap">
  <canvas id="cv"></canvas>
  <aside>
    <h2>图例</h2>
    <div class="lg"><i class="sw" style="background:#fff"></i>可走 FREE __NFREE__</div>
    <div class="lg"><i class="sw" style="background:#2f3b52"></i>障碍 OCC __NOCC__</div>
    <div class="lg"><i class="sw" style="background:#ececf0"></i>未知 UNKNOWN __NUNK__</div>
    <h2>轨迹</h2>
    <div id="tlg"></div>
    <h2>图层</h2>
    <div class="btns">
      <button id="bT" class="on">轨迹</button>
      <button id="bG" class="on">网格</button>
      <button id="bR">重置</button>
    </div>
    <h2>统计</h2>
    <div id="stats"></div>
    <div class="tip">滚轮缩放（以光标为锚）· 拖拽平移 · 移动鼠标读世界坐标（米，y 向上）</div>
  </aside>
</div>
<div id="pos"></div>
<script>
const W=__W__, H=__H__, RES=__RES__, OX=__OX__, OY=__OY__;
const LBL=Uint8Array.from(atob("__DATA__"), c=>c.charCodeAt(0));
const TRAJ=__TRAJ__;
const BB=__BB__;
const STATS=__STATS__;

// ---- 离屏栅格图（世界 y 向上 ⇒ 行翻转）----
const off=document.createElement('canvas'); off.width=W; off.height=H;
const octx=off.getContext('2d'); const im=octx.createImageData(W,H);
for(let r=0;r<H;r++){ const yr=H-1-r;
  for(let c=0;c<W;c++){ const v=LBL[r*W+c];
    let R=236,G=236,B=240;
    if(v===1){R=255;G=255;B=255;} else if(v===2){R=47;G=59;B=82;}
    const i=(yr*W+c)*4; im.data[i]=R;im.data[i+1]=G;im.data[i+2]=B;im.data[i+3]=255; } }
octx.putImageData(im,0,0);

const cvs=document.getElementById('cv'), g=cvs.getContext('2d');
let cx=(BB[0]+BB[2])/2, cy=(BB[1]+BB[3])/2, s=10;   // s = px / m
let showT=true, showG=true;
let dpr=Math.min(2, window.devicePixelRatio||1);

function fit(){
  const r=cvs.getBoundingClientRect();
  cvs.width=Math.round(r.width*dpr); cvs.height=Math.round(r.height*dpr);
  const sx=r.width/(BB[2]-BB[0]), sy=r.height/(BB[3]-BB[1]);
  s=Math.min(sx,sy)*0.94;
}
const SX=x=>cvs.width/2+(x-cx)*s*dpr;
const SY=y=>cvs.height/2-(y-cy)*s*dpr;
const WX=px=>((px-cvs.width/2)/(s*dpr))+cx;
const WY=py=>cy-((py-cvs.height/2)/(s*dpr));

function draw(){
  const r=cvs.getBoundingClientRect();
  g.setTransform(1,0,0,1,0,0);
  g.fillStyle='#e9e9ee'; g.fillRect(0,0,cvs.width,cvs.height);
  g.imageSmoothingEnabled=false;
  g.drawImage(off, SX(OX), SY(OY+H*RES), W*RES*s*dpr, H*RES*s*dpr);

  if(showG){
    const step = s*dpr>26?1:(s*dpr>9?5:10);
    g.strokeStyle='rgba(120,120,140,.20)'; g.lineWidth=1;
    g.fillStyle='rgba(90,90,110,.75)'; g.font=Math.round(11*dpr)+'px sans-serif';
    for(let x=Math.ceil(BB[0]/step)*step; x<=BB[2]; x+=step){
      g.beginPath(); g.moveTo(SX(x),0); g.lineTo(SX(x),cvs.height); g.stroke(); }
    for(let y=Math.ceil(BB[1]/step)*step; y<=BB[3]; y+=step){
      g.beginPath(); g.moveTo(0,SY(y)); g.lineTo(cvs.width,SY(y)); g.stroke();
      g.fillText(y.toFixed(0), 3*dpr, SY(y)-3*dpr); }
  }
  if(showT){
    g.lineWidth=Math.max(1.4,1.6*dpr); g.lineJoin='round';
    for(const t of TRAJ){
      g.strokeStyle=t.color; g.beginPath();
      for(let i=0;i<t.pts.length;i++){ const p=t.pts[i];
        if(i===0) g.moveTo(SX(p[0]),SY(p[1])); else g.lineTo(SX(p[0]),SY(p[1])); }
      g.stroke();
      // 起点小方块
      const p0=t.pts[0]; g.fillStyle=t.color;
      g.fillRect(SX(p0[0])-3*dpr,SY(p0[1])-3*dpr,6*dpr,6*dpr);
    }
  }
}

// ---- 交互 ----
let drag=null;
cvs.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,cx:cx,cy:cy};cvs.classList.add('drag');});
window.addEventListener('mouseup',()=>{drag=null;cvs.classList.remove('drag');});
window.addEventListener('mousemove',e=>{
  const r=cvs.getBoundingClientRect();
  if(drag){
    const px=(e.clientX-drag.x)*dpr, py=(e.clientY-drag.y)*dpr;
    cx=drag.cx-px/(s*dpr); cy=drag.cy+py/(s*dpr); draw(); return; }
  if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom){
    document.getElementById('pos').classList.remove('show'); return; }
  const wx=WX((e.clientX-r.left)*dpr), wy=WY((e.clientY-r.top)*dpr);
  const cc=Math.floor((wx-OX)/RES), rr=Math.floor((wy-OY)/RES);
  let lab='—';
  if(cc>=0&&cc<W&&rr>=0&&rr<H){ const v=LBL[rr*W+cc];
    lab = v===1?'FREE':(v===2?'OCC':'UNKNOWN'); }
  const el=document.getElementById('pos');
  el.textContent=`x ${wx.toFixed(2)}  y ${wy.toFixed(2)} m   ·   ${lab}`;
  el.classList.add('show');
});
cvs.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=cvs.getBoundingClientRect();
  const mx=(e.clientX-r.left)*dpr, my=(e.clientY-r.top)*dpr;
  const wx=WX(mx), wy=WY(my);
  const k=Math.exp(-e.deltaY*0.0016);
  s=Math.max(0.4, Math.min(600, s*k));
  cx=wx-((mx-cvs.width/2)/(s*dpr)); cy=wy+((my-cvs.height/2)/(s*dpr));
  draw();
},{passive:false});

const bT=document.getElementById('bT'), bG=document.getElementById('bG');
bT.onclick=()=>{showT=!showT;bT.classList.toggle('on',showT);draw();};
bG.onclick=()=>{showG=!showG;bG.classList.toggle('on',showG);draw();};
document.getElementById('bR').onclick=()=>{fit();draw();};
window.addEventListener('resize',()=>{fit();draw();});

const tlg=document.getElementById('tlg');
for(const t of TRAJ){
  const d=document.createElement('div'); d.className='lg';
  const sw=document.createElement('i'); sw.className='sw'; sw.style.background=t.color;
  const tx=document.createElement('span');
  tx.textContent=t.name+' · '+t.pts.length+' 帧'+(t.src?' · '+t.src:'');
  d.appendChild(sw); d.appendChild(tx); tlg.appendChild(d);
}
const st=document.getElementById('stats');
for(const k of Object.keys(STATS)){
  const d=document.createElement('div'); d.className='kv';
  const a=document.createElement('span'); a.textContent=k;
  const b=document.createElement('span'); b.textContent=STATS[k];
  d.appendChild(a); d.appendChild(b); st.appendChild(d);
}
fit(); draw();
</script></body></html>
"""


# ------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="先验栅格俯视图渲染（三态 + 轨迹叠加）")
    ap.add_argument("--world", required=True, help="世界目录名，如 wrld_home-7cf435ea")
    ap.add_argument("--out", default=None, help="输出 HTML 路径（默认 .tmp/topview/<world>.html）")
    ap.add_argument("--png", default=None, help="额外输出 PNG 路径（可选）")
    ap.add_argument("--px-per-m", type=float, default=20.0, help="PNG 像素密度，默认 20")
    ap.add_argument("--no-traj", action="store_true", help="不叠加轨迹")
    ap.add_argument("--no-crop", action="store_true", help="不裁剪到内容包围盒（画整张栅格）")
    ap.add_argument("--margin-m", type=float, default=2.0, help="裁剪留边（米），默认 2")
    args = ap.parse_args()

    t0 = time.perf_counter()
    wdir = ROOT / "navmesh_memory" / args.world
    if not wdir.is_dir():
        raise SystemExit(f"[fail] 世界目录不存在：{wdir}")

    pr = load_prior(wdir)
    lab, origin, res, meta = pr["labels"], pr["origin"], pr["res"], pr["meta"]
    H, W = lab.shape
    print(f"[load ] {pr['npz_path']}  labels {H}×{W}  res={res} m  origin=({origin[0]:.2f},{origin[1]:.2f})")

    trajs = [] if args.no_traj else load_trajs(wdir, meta)
    for t in trajs:
        print(f"[traj ] {t['sid']} ← rec {t['rec'] or '?'}  {len(t['pts'])} 帧  src={t['src']}")

    # 包围盒：已知格 ∪ 轨迹
    m = float(args.margin_m)
    if args.no_crop:
        bb = (float(origin[0]), float(origin[1]),
              float(origin[0] + W * res), float(origin[1] + H * res))
    else:
        nz = np.argwhere(lab != L_UNK)
        if len(nz):
            r0, c0 = nz.min(0)
            r1, c1 = nz.max(0) + 1
            bb = [origin[0] + c0 * res, origin[1] + r0 * res,
                  origin[0] + c1 * res, origin[1] + r1 * res]
        else:
            bb = [float(origin[0]), float(origin[1]),
                  float(origin[0] + W * res), float(origin[1] + H * res)]
        for t in trajs:
            p = np.asarray(t["pts"], np.float64)
            bb[0] = min(bb[0], float(p[:, 0].min()))
            bb[1] = min(bb[1], float(p[:, 1].min()))
            bb[2] = max(bb[2], float(p[:, 0].max()))
            bb[3] = max(bb[3], float(p[:, 1].max()))
        bb = (bb[0] - m, bb[1] - m, bb[2] + m, bb[3] + m)
    print(f"[bbox ] x [{bb[0]:.1f},{bb[2]:.1f}]  y [{bb[1]:.1f},{bb[3]:.1f}]"
          f"  ⇒ {(bb[2]-bb[0]):.1f} × {(bb[3]-bb[1]):.1f} m")

    n_free = int((lab == L_FREE).sum())
    n_occ = int((lab == L_OCC).sum())
    n_unk = int((lab == L_UNK).sum())
    tot = H * W
    built = meta.get("built_wall")
    built_s = (_dt.datetime.fromtimestamp(built).strftime("%Y-%m-%d %H:%M")
               if isinstance(built, (int, float)) else "?")

    stats = {
        "栅格总数": f"{tot:,}",
        "可走占比": f"{100.0*n_free/tot:.1f}%",
        "障碍占比": f"{100.0*n_occ/tot:.1f}%",
        "未知占比": f"{100.0*n_unk/tot:.1f}%",
        "已知面积": f"{(n_free+n_occ)*res*res:,.1f} m²",
        "收录会话": f"{len(meta.get('sessions', []))} 场",
        "融合前 OCC": f"{meta.get('transitions', {}).get('occ_baseline', '?')}",
        "融合后 OCC": f"{meta.get('transitions', {}).get('occ_fused', '?')}",
        "清除(有地面证据)": f"{meta.get('transitions', {}).get('cleared_free_with_gnd_evid', '?')}",
    }

    html = (HTML
            .replace("__TITLE__", f"{args.world} · 先验栅格俯视图")
            .replace("__SHAPE__", f"{H}×{W}")
            .replace("__RES__", f"{res:g}")
            .replace("__SCALE__", f"{pr['scale']:g}")
            .replace("__BASE__", pr["base_sid"][-6:])
            .replace("__BUILT__", built_s)
            .replace("__NPATH__", f"{pr['npz_path'].name}（{pr['npz_path'].stat().st_size//1024} KB）")
            .replace("__NFREE__", f"{n_free:,}（{100.0*n_free/tot:.1f}%）")
            .replace("__NOCC__", f"{n_occ:,}（{100.0*n_occ/tot:.1f}%）")
            .replace("__NUNK__", f"{n_unk:,}（{100.0*n_unk/tot:.1f}%）")
            .replace("__W__", str(W)).replace("__H__", str(H))
            .replace("__OX__", repr(float(origin[0]))).replace("__OY__", repr(float(origin[1])))
            .replace("__RES__", repr(float(res)))
            .replace("__DATA__", base64.b64encode(lab.tobytes()).decode("ascii"))
            .replace("__TRAJ__", json.dumps([{k: t[k] for k in ("name", "color", "pts", "src")} for t in trajs]))
            .replace("__BB__", json.dumps([round(float(v), 3) for v in bb]))
            .replace("__STATS__", json.dumps(stats, ensure_ascii=False)))
    # __RES__ 被用了两次（副标题 + JS 常量），第二次 replace 后已耗尽；再兜一次
    html = html.replace("__RES__", f"{res:g}")

    out = Path(args.out) if args.out else (ROOT / ".tmp" / "topview" / f"{args.world}_prior.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[out  ] {out}  ({out.stat().st_size/1048576:.2f} MB, {time.perf_counter()-t0:.1f}s)")

    if args.png:
        img = rasterize(lab, trajs, origin, res, bb, float(args.px_per_m), bool(trajs))
        pp = Path(args.png)
        write_png(pp, img)
        print(f"[png  ] {pp}  {img.shape[1]}×{img.shape[0]} px")

    # 残留占位符自检
    left = [k for k in ("__DATA__", "__TRAJ__", "__BB__", "__STATS__", "__W__", "__H__",
                        "__OX__", "__OY__", "__TITLE__", "__NFREE__", "__NOCC__", "__NUNK__")
            if k in html]
    if left:
        print(f"[warn ] 未替换占位符：{left}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
