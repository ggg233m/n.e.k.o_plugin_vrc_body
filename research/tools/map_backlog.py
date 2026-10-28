# -*- coding: utf-8 -*-
r'''用 mtime 反演建图线程的真实队列：到达/服务/delay 曲线。

为什么不能只看 events.jsonl（旧录制）：那里只有**采集时刻** ``t``，没有建图线程什么时候吃下这一帧。
2026-10-29 起 kf 行带 ``done_t``（建图线程消费完成时刻，clock 域），有它优先用；旧录制退回 mtime。
``kf/<k>.npz`` 的 mtime 就是 ``rec.keyframe()`` 真正跑完的时刻（它在建图线程里同步调用，
见 nav_online.py:1948），所以 mtime 序列 = 服务完成时刻序列。

前提（本工具会自动检查）：写盘延迟单调增长 ⇒ 线程从不空转 ⇒ 相邻 mtime 间隔就是服务时间。
'''
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(rec):
    meta = json.loads((rec/'meta.json').read_text(encoding='utf-8'))
    W0, C0 = meta['started_wall'], meta['started_clock']
    ev = [json.loads(x) for x in (rec/'events.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    kf = sorted([e for e in ev if e.get('kind')=='kf'], key=lambda e: e['k'])
    out = []
    # done_t（nav_online 2026-10-29 新增）：建图线程消费完成的时刻（clock 域，与 t 同源）。
    # 有它就用它（不再受 mtime 被复制/解压改写的影响）；旧录制没有 → 退回 mtime 口径。
    use_done_t = all('done_t' in e for e in kf) and all(e.get('done_t') is not None for e in kf)
    for e in kf:
        p = rec/'kf'/('%06d.npz' % int(e['k']))
        if not p.exists():
            continue
        done = (e['done_t']-C0) if use_done_t else (p.stat().st_mtime-W0)
        out.append(dict(k=int(e['k']), cap=e['t']-C0, done=done,
                        unmapped=bool(e.get('unmapped')), dist=float(e.get('dist_m', 0.0))))
    return meta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recs', nargs='*', default=['20261026_184318'])
    args = ap.parse_args()
    for name in args.recs:
        rec = ROOT/'navmesh_recordings'/name
        if not rec.exists():
            print('[skip] ' + name, file=sys.stderr); continue
        meta, rows = load(rec)
        mp = [r for r in rows if not r['unmapped']]
        un = [r for r in rows if r['unmapped']]
        cap = np.array([r['cap'] for r in mp]); done = np.array([r['done'] for r in mp])
        svc = np.diff(done); arr = np.diff(cap)
        delay = done - cap
        print()
        print('=== %s ===  入图 %d 帧 / 补写 %d 帧' % (name, len(mp), len(un)))
        print('  到达率 lambda = %.3f kf/s（中位间隔 %.0f ms  mean %.0f ms）'
              % (1/arr.mean(), np.median(arr)*1000, arr.mean()*1000))
        print('  服务时间 median %.0f ms  mean %.0f ms  p90 %.0f ms  p99 %.0f ms'
              % (np.median(svc)*1000, svc.mean()*1000, np.percentile(svc,90)*1000, np.percentile(svc,99)*1000))
        mu = 1/svc.mean()
        rho = (1/arr.mean())/mu
        print('  服务率 mu = %.3f kf/s  =>  rho = %.2f  %s'
              % (mu, rho, '[UNSTABLE: 延迟线性增长]' if rho > 1 else '[stable]'))
        print('  队列延迟 median %.1f s  p90 %.1f s  max %.1f s'
              % (np.median(delay), np.percentile(delay,90), delay.max()))
        # 单调性检查（决定 svc 是否可信）
        trend = np.polyfit(done-done[0], delay, 1)[0] if len(delay) > 10 else 0.0
        print('  延迟趋势 %.3f s/s（>0.05 视为单调积压，此时 svc=相邻mtime间隔 可信）' % trend)
        print('  墙钟：首帧写盘 %.1fs，末个入图帧 %.1fs，最后一帧写盘 %.1fs'
              % (done[0], done[-1], rows[-1]['done']))
        if un:
            print('  未入图 %d 帧：采集到 %.0f s，补写窗口 %.1f→%.1f s'
                  % (len(un), un[-1]['cap'], un[0]['done'], un[-1]['done']))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    main()