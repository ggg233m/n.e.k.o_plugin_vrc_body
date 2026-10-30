#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Dense ICP odometry edges for all adjacent keyframe pairs.
#
# Why dense ICP: adjacent keyframes overlap almost completely (0.4 m apart).
# Verified 16/16 convergence, 68-99% inliers, RMS 0.010-0.115 m.
# Sparse ORB (kf_visual_odom) only solves 26% of adjacent pairs because des3d
# has a median of 351 points per frame, while dense pts has about 84k.
import sys, time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(r'H:/AI/neko-music/vrc/pc-vr/n.e.k.o_plugin_vrc_body')
RECNAME = sys.argv[1] if len(sys.argv) > 1 else '20261029_060258'
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
REC = ROOT / 'navmesh_recordings' / RECNAME

with np.load(REC / 'poses_replay.npz') as z:
    ids = np.asarray(z['ids'], np.int64)
    T = np.asarray(z['T'], np.float64)
POSE = {int(k): T[i] for i, k in enumerate(ids)}
KS = sorted(POSE)
print('[load] %s  %d frames' % (RECNAME, len(KS)), flush=True)


def dense(k, cap=CAP, seed=0):
    p = REC / ('kf/%06d.npz' % k)
    if not p.exists():
        return np.zeros((0, 3))
    with np.load(p) as z:
        q = np.asarray(z['pts'], np.float64)
    ok = np.isfinite(q).all(1) & (np.hypot(q[:, 0], q[:, 1]) <= 5.0)
    q = q[ok]
    if len(q) > cap:
        q = q[np.random.default_rng(seed).choice(len(q), cap, replace=False)]
    return q


def batch_normals(pts, nb_idx):
    nb = pts[nb_idx]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', c, c)
    _, v = np.linalg.eigh(cov)
    return v[:, :, 0]


def icp(PA, PB, R0, t0, iters=15, dist=0.35, kk=10):
    if len(PA) < 2000 or len(PB) < 2000:
        return None
    treeA = cKDTree(PA)
    _, nbA = treeA.query(PA, k=kk)
    nA = batch_normals(PA, nbA)
    R = np.asarray(R0, np.float64).copy()
    t = np.asarray(t0, np.float64).copy()
    rms = np.inf
    used = 0
    for it in range(iters):
        Pt = PB @ R.T + t
        d, idx = treeA.query(Pt)
        m = d < dist
        if int(m.sum()) < 500:
            return None
        src = Pt[m]
        tgt = PA[idx[m]]
        nrm = nA[idx[m]]
        res = np.einsum('ij,ij->i', src - tgt, nrm)
        s = float(np.std(res))
        if s > 1e-9:
            keep = np.abs(res - np.median(res)) < 3.0 * s
            if int(keep.sum()) < 500:
                return None
            res = res[keep]
            nrm = nrm[keep]
            src = src[keep]
        J = np.hstack([np.cross(src, nrm), nrm])
        H = J.T @ J + 1e-6 * np.eye(6)
        g = J.T @ res
        try:
            x = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return None
        w = x[:3]
        ang = float(np.linalg.norm(w))
        if ang > 1e-8:
            ax = w / ang
            K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
            dR = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
        else:
            dR = np.eye(3)
        R = dR @ R
        t = dR @ t + x[3:]
        rms = float(np.sqrt(np.mean(res ** 2)))
        used = int(len(res))
        if np.linalg.norm(x) < 1e-5:
            break
    return R, t, rms, used


OUT = ROOT / '.tmp' / ('icp_edges_' + RECNAME + '.npz')
edges = {}
t_all = time.time()
ok = 0
tot = 0
for k0, k1 in zip(KS, KS[1:]):
    if k1 - k0 > 3:
        continue
    tot += 1
    PA = dense(k0)
    PB = dense(k1)
    if not len(PA) or not len(PB):
        continue
    T0, T1 = POSE[k0], POSE[k1]
    R0 = T0[:3, :3].T @ T1[:3, :3]
    t0 = T0[:3, :3].T @ (T1[:3, 3] - T0[:3, 3])
    r = icp(PA, PB, R0, t0)
    if r is None:
        continue
    RR, tt, rms, used = r
    edges[(k0, k1)] = tt.astype(np.float64)
    ok += 1
    if tot % 100 == 0:
        print('  %4d  ok %4d  (%.0fs)' % (tot, ok, time.time() - t_all), flush=True)

print()
print('[done] solved %d / %d (%.0f%%)  %.0fs' % (ok, tot, 100.0 * ok / max(tot, 1), time.time() - t_all), flush=True)
np.savez_compressed(OUT, **{('e_%d_%d' % k): v for k, v in edges.items()})
print('[save] %s  %d edges' % (OUT, len(edges)), flush=True)