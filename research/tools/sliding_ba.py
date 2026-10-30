#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Sliding-window local BA over dense ICP odometry edges.
#
# Why the middle ground:
#   global least squares (vo_g6/joint_poses) -> 10.2/10.1 cm, good visual residuals
#     get diluted across the 80% of frames with no visual constraint;
#   per-pair direct fix (local_vo_fix) -> 5.7 cm but breaks trajectory continuity;
#   industry standard (Stereo DSO / BALM / GO-SLAM) is a LOCAL bundle adjustment
#     on a sliding window: keeps continuity via overlapping frames, does not let
#     unconstrained frames dilute the constraint.
#
# Model: 2-D pose graph (x, y). yaw comes from HMD and does not drift, so only
# translation is solved. Edges:
#   * dense ICP relative translation (high weight, measured)
#   * soft prior to the base table (low weight, keeps global reference)
#   * overlap frames couple adjacent windows
import sys
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

ROOT = Path(r'H:/AI/neko-music/vrc/pc-vr/n.e.k.o_plugin_vrc_body')
RECNAME = sys.argv[1] if len(sys.argv) > 1 else '20261029_060258'
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 60
OVERLAP = int(sys.argv[3]) if len(sys.argv) > 3 else 20
W_ICP = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
W_PRIOR = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
REC = ROOT / 'navmesh_recordings' / RECNAME

with np.load(REC / 'poses_replay.npz') as z:
    ids = np.asarray(z['ids'], np.int64)
    T = np.asarray(z['T'], np.float64)
POSE = {int(k): T[i] for i, k in enumerate(ids)}
KS = sorted(POSE)

ep = ROOT / '.tmp' / ('icp_edges_' + RECNAME + '.npz')
if not ep.exists():
    raise SystemExit('[fail] missing ' + str(ep))
EDGES = {}
with np.load(ep) as z:
    for key in z.files:
        _, a, b = key.split('_')
        EDGES[(int(a), int(b))] = np.asarray(z[key], np.float64)
print('[load] %s  %d frames  %d ICP edges' % (RECNAME, len(KS), len(EDGES)), flush=True)

idx = {k: i for i, k in enumerate(KS)}
XY0 = np.array([[POSE[k][0, 3], POSE[k][1, 3]] for k in KS], np.float64)
XY = XY0.copy()

def solve_window(lo, hi):
    sub = KS[lo:hi]
    n = len(sub)
    if n < 3:
        return
    lidx = {k: i for i, k in enumerate(sub)}
    rows = []
    rhs = []
    wts = []
    for (a, b), t in EDGES.items():
        if a in lidx and b in lidx:
            # x: -xa + xb = t_x ; y likewise
            r = np.zeros(2 * n); r[2 * lidx[a]] = -1.0; r[2 * lidx[b]] = 1.0
            rows.append(r); rhs.append(t[0]); wts.append(W_ICP)
            r2 = np.zeros(2 * n); r2[2 * lidx[a] + 1] = -1.0; r2[2 * lidx[b] + 1] = 1.0
            rows.append(r2); rhs.append(t[1]); wts.append(W_ICP)
    for i, k in enumerate(sub):
        r = np.zeros(2 * n); r[2 * i] = W_PRIOR
        rows.append(r); rhs.append(W_PRIOR * XY0[idx[k], 0]); wts.append(1.0)
        r2 = np.zeros(2 * n); r2[2 * i + 1] = W_PRIOR
        rows.append(r2); rhs.append(W_PRIOR * XY0[idx[k], 1]); wts.append(1.0)
    A = np.asarray(rows); b = np.asarray(rhs); w = np.asarray(wts)
    sol = lsqr(coo_matrix(A * w[:, None]).tocsr(), b * w, damp=1e-4, iter_lim=5000)
    xy = sol[0].reshape(-1, 2)
    # 重叠帧加权融合（保证窗口间连续）
    for i, k in enumerate(sub):
        gi = idx[k]
        if k in OVERLAP_SET:
            XY[gi] = 0.5 * (XY[gi] + xy[i])
        else:
            XY[gi] = xy[i]

OVERLAP_SET = set()
step = max(1, WIN - OVERLAP)
t0 = 0
n_win = 0
while t0 < len(KS):
    t1 = min(t0 + WIN, len(KS))
    if t1 - t0 < 3:
        break
    OVERLAP_SET = set(KS[max(t0, t1 - OVERLAP):t1]) if n_win > 0 else set()
    solve_window(t0, t1)
    n_win += 1
    if t1 >= len(KS):
        break
    t0 += step
print('[solve] %d windows' % n_win, flush=True)

# 真值对检验
truth = ROOT / '.tmp' / 'truth3d_060258.tsv'
if truth.exists():
    prs = []
    for line in truth.open(encoding='utf-8'):
        if line.startswith('a'):
            continue
        f = line.split()
        prs.append((int(f[0]), int(f[1])))
    prs = sorted({tuple(sorted(p)) for p in prs})
    def d_before_after(a, b):
        d0 = float(np.linalg.norm(XY0[idx[a]] - XY0[idx[b]]))
        d1 = float(np.linalg.norm(XY[idx[a]] - XY[idx[b]]))
        return d0, d1
    print()
    print('%-14s %9s %9s' % ('pair', 'before', 'after'))
    b0 = []; b1 = []
    for a, b in prs:
        if a in idx and b in idx:
            d0, d1 = d_before_after(a, b)
            b0.append(d0); b1.append(d1)
            print('%-14s %9.2f %9.2f' % ('%d-%d' % (a, b), d0, d1), flush=True)
    if b0:
        print()
        print('before median %.2f  after median %.2f' % (np.median(b0), np.median(b1)), flush=True)

# 连续性检查：相邻帧步长不应跳变
steps0 = np.linalg.norm(np.diff(XY0, axis=0), axis=1)
steps1 = np.linalg.norm(np.diff(XY, axis=0), axis=1)
print()
print('step length: before p50 %.3f max %.3f | after p50 %.3f max %.3f' % (
    np.median(steps0), steps0.max(), np.median(steps1), steps1.max()), flush=True)

out = REC / 'poses_icp.npz'
T2 = T.copy()
T2[:, 0, 3] = XY[:, 0]
T2[:, 1, 3] = XY[:, 1]
np.savez_compressed(out, ids=ids, T=T2)
print('[out  ] %s' % out, flush=True)