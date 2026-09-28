# -*- coding: utf-8 -*-
# (4) incremental-geometry verification.
# Question: does triangulating IMAGE features (ORB) with the METRIC OSC+HMD pose
# produce wall geometry flatter than the depth-fusion floor (~0.070 m)?
#
# Design (per user spec):
#   1. relative pose between frames comes ONLY from OSC distance x HMD yaw (DR, metric)
#   2. ORB tracks wall features between frame c and c+k
#   3. triangulate -> 3D world points (no Depth Anything averaging of any kind)
#   4. filter by reproj error / parallax / cheirality / multi-frame agreement
#   5. A/B: at the SAME matched pixels, compare against the depth-model route.
#
# Stop-loss: if triangulated wall is ~7 cm (cannot beat 0.070) -> abandon precise 3D walls.
import os, sys, math, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.spatial import cKDTree

REPO = r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body"
REC = os.path.join(REPO, ".slam_probe", "offline_probe", "recorder")
RUN = os.path.join(REC, "runs", "20260920-233456")
CACHE = os.path.join(REPO, ".tmp", "_depth_cache20.npz")
OUTP = os.path.join(REPO, ".tmp", "_incr_geom.txt")
sys.path.insert(0, REC)

import cv2
import video_frames as VF

OUT = []
def log(s=""):
    OUT.append(str(s)); print(s, flush=True)

STRIDE = int(os.environ.get("IG_STRIDE", "4"))
KS = tuple(int(x) for x in os.environ.get("IG_KS", "20,40,80").split(","))   # 20Hz: 1s/2s/4s
FEAT = os.environ.get("IG_FEAT", "orb")
NFEAT = int(os.environ.get("IG_NFEAT", "1500"))
RATIO = float(os.environ.get("IG_RATIO", "0.75"))
REPROJ = float(os.environ.get("IG_REPROJ", "2.0"))      # px
ZMIN, ZMAX = 0.4, 12.0
PAR_MIN, PAR_MAX = 0.5, 25.0                            # deg

# ---------- load cached depth + intrinsics ----------
C = np.load(CACHE, allow_pickle=True)
D16 = C["d16"]; KK = C["k"]; VH = C["vh"]; USABLE = C["usable"].astype(np.int64)
T_TELE = C["t_tele"]; fx = float(C["fx"]); fy = float(C["fy"]); cy = float(C["cy"])
cam_h = float(C["cam_h"]); W = int(C["W"]); H = int(C["H"]); nF = D16.shape[0]
cx = (W - 1) / 2.0
CEN = np.where(np.isfinite(VH), VH, cy).astype(np.float64)
video = str(C["video"]); fps = float(C["fps"]); offset = float(C["offset"])
log("cache: nF=%d  fps=%g  WxH=%dx%d  fx=%.2f  cam_h=%.2f" % (nF, fps, W, H, fx, cam_h))
log("video=%s  offset=%.2f" % (video, offset))

# ---------- metric pose from OSC+HMD (DR, uncorrupted by loop optimisation) ----------
pg = json.load(open(os.path.join(RUN, "pose_graph.json"), encoding="utf-8"))
nd = pg["nodes"]
NT = np.array([n["t"] for n in nd], float)
NX0 = np.array([n["x0"] for n in nd], float)
NZ0 = np.array([n["z0"] for n in nd], float)
NYAW = np.array([n["yaw"] for n in nd], float)
PX = np.interp(T_TELE, NT, NX0)
PZ = np.interp(T_TELE, NT, NZ0)
YAW = np.radians(np.interp(T_TELE, NT, NYAW))           # geometric heading (already signed)
ok_pose = np.isfinite(KK) & np.isfinite(PX) & np.isfinite(PZ)
log("DR pose span: x %.2f..%.2f  z %.2f..%.2f  path n=%d" %
    (NX0.min(), NX0.max(), NZ0.min(), NZ0.max(), len(nd)))
log("frames with finite depth-scale K: %d/%d" % (int(ok_pose.sum()), nF))

def R_wc(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, -1.0, 0.0], [-s, 0.0, c]], float)

def PMAT(i):
    R = R_wc(YAW[i]); pos = np.array([PX[i], cam_h, PZ[i]])
    Rt = np.hstack([R.T, (-R.T @ pos).reshape(3, 1)])
    Ki = np.array([[fx, 0.0, cx], [0.0, fy, CEN[i]], [0.0, 0.0, 1.0]])
    return Ki @ Rt, R, pos

def Zdep(i, u, v):
    ui = np.clip(np.round(u).astype(np.int64), 0, W - 1)
    vi = np.clip(np.round(v).astype(np.int64), 0, H - 1)
    d = D16[i, vi, ui].astype(np.float64)
    return np.where(d > 1e-6, KK[i] / np.maximum(d, 1e-6), np.nan)

# ---------- video frames (same fiducial as the depth cache) ----------
t0 = time.time()
frames, fmeta = VF.grab(video, fps=fps, size=(W, H), gray=True, verbose=True)
nAll = frames.shape[0]
t_all = np.arange(nAll, dtype=np.float64) / fps + offset
TLO, THI = float(T_TELE[0]), float(T_TELE[-1])
us2 = np.where((t_all >= TLO - 1e-9) & (t_all <= THI + 1e-9))[0]
log("grabbed %d frames (%.1fs, %s)  usable(cache)=%d  usable(recomputed)=%d" %
    (nAll, time.time() - t0, fmeta.get("ms_per_frame"), USABLE.size, us2.size))
if us2.size == USABLE.size:
    FIDX = us2
else:
    log("  WARN index mismatch, using cache 'usable'")
    FIDX = USABLE

# ---------- features ----------
if FEAT == "sift":
    det = cv2.SIFT_create(nfeatures=NFEAT, contrastThreshold=0.02)
    NORM = cv2.NORM_L2
else:
    det = cv2.ORB_create(nfeatures=NFEAT, scaleFactor=1.2, nlevels=8, fastThreshold=8)
    NORM = cv2.NORM_HAMMING
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
t0 = time.time()
KPS = [None] * nF; DES = [None] * nF
nfeat = []
for c in range(nF):
    if not ok_pose[c]:
        continue
    g = frames[FIDX[c]]
    kp, des = det.detectAndCompute(g, None)
    if des is None or len(kp) == 0:
        continue
    pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 1, 2)
    pts = cv2.cornerSubPix(g, pts, (4, 4), (-1, -1), crit)
    KPS[c] = pts.reshape(-1, 2); DES[c] = des; nfeat.append(len(kp))
    if (c + 1) % 400 == 0:
        log("  feat %d/%d  median=%d  %.0fs" % (c + 1, nF, int(np.median(nfeat)), time.time() - t0))
log("features: %s nfeatures=%d  frames=%d  median kp=%d  (%.0fs)" %
    (FEAT, NFEAT, len(nfeat), int(np.median(nfeat)) if nfeat else 0, time.time() - t0))

matcher = cv2.BFMatcher(NORM, crossCheck=False)

def local_plan_med(P, radius=0.40, minn=6):
    P = P.astype(np.float64)
    if P.shape[0] < minn:
        return float("nan"), 0
    nb = cKDTree(P).query_ball_point(P, radius, workers=1)
    res = []
    for nl in nb:
        if len(nl) < minn:
            continue
        A = P[nl] - P[nl].mean(0)
        sv = np.linalg.svd(A, compute_uv=False)
        res.append(sv[-1] / math.sqrt(len(nl)))
    if not res:
        return float("nan"), 0
    return float(np.median(res)), len(res)

# ---------- per-pair triangulation ----------
pair_rows = []          # (c, j, k, n_match, n_pass, med_par, med_Z, med_reproj, med_ratio)
cloud = []              # world points (triangulated, passing)
cloud_dep = []          # world points (depth route) at the SAME pixels, both endpoints
nfail_pose = nfail_des = 0
SKIP = {"no_match": 0, "no_pass": 0, "zrange": 0, "reproj": 0, "par": 0, "cheir": 0}
RATIO_ALL = []; ZDEP_ALL = []; ZTRI_ALL = []; N_NAN_DEP = []; N_FAR_DEP = []; PAIR_PM = []; K_ALL = []
t0 = time.time()
bases = [c for c in range(0, nF, STRIDE) if KPS[c] is not None]
for bnum, c in enumerate(bases):
    Pc, Rc, posc = PMAT(c)
    for k in KS:
        j = c + k
        if j >= nF or KPS[j] is None:
            continue
        Pj, Rj, posj = PMAT(j)
        try:
            mm = matcher.knnMatch(DES[c], DES[j], k=2)
        except cv2.error:
            continue
        good = [m for m, n in mm if m.distance < RATIO * n.distance]
        if len(good) < 8:
            SKIP["no_match"] += 1
            continue
        ic = np.array([good[t].queryIdx for t in range(len(good))], np.int64)
        jj = np.array([good[t].trainIdx for t in range(len(good))], np.int64)
        pc = KPS[c][ic].T.astype(np.float64)   # 2xN
        pj = KPS[j][jj].T.astype(np.float64)
        # ---- triangulate in world ----
        X4 = cv2.triangulatePoints(Pc, Pj, pc, pj)
        w4 = X4[3]
        valid = np.abs(w4) > 1e-9
        X = np.full((3, pc.shape[1]), np.nan)
        X[:, valid] = X4[:3, valid] / w4[valid]
        # ---- cheirality + depth (camera frame) ----
        Xc = Rc.T @ (X - posc.reshape(3, 1)); Zc = Xc[2]
        Xj = Rj.T @ (X - posj.reshape(3, 1)); Zj = Xj[2]
        # ---- reprojection ----
        def reproj(P, uv, Xw):
            q = P @ np.vstack([Xw, np.ones((1, Xw.shape[1]))])
            uv2 = q[:2] / q[2]
            return np.linalg.norm(uv2 - uv, axis=0)
        rc = reproj(Pc, pc, X)
        rj = reproj(Pj, pj, X)
        # ---- parallax ----
        v1 = X - posc.reshape(3, 1); v2 = X - posj.reshape(3, 1)
        n1 = np.linalg.norm(v1, axis=0); n2 = np.linalg.norm(v2, axis=0)
        cosang = np.sum(v1 * v2, axis=0) / np.maximum(n1 * n2, 1e-12)
        par = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        # ---- depth route at the same pixels ----
        Zw_c = Zdep(c, pc[0], pc[1])
        Zw_j = Zdep(j, pj[0], pj[1])
        Zc_dep = np.where(np.isfinite(Zw_c), np.maximum(Zw_c, 1e-6), np.nan)
        ratio = Zc / Zc_dep
        m = (np.isfinite(X[0]) & np.isfinite(Zc) & np.isfinite(Zj)
             & (Zc > ZMIN) & (Zc < ZMAX) & (Zj > ZMIN) & (Zj < ZMAX)
             & (rc < REPROJ) & (rj < REPROJ) & (par > PAR_MIN) & (par < PAR_MAX))
        npass = int(m.sum())
        if npass == 0:
            SKIP["no_pass"] += 1
            SKIP["zrange"] += int(np.sum(~((Zc > ZMIN) & (Zc < ZMAX) & (Zj > ZMIN) & (Zj < ZMAX))))
            SKIP["reproj"] += int(np.sum(~((rc < REPROJ) & (rj < REPROJ))))
            SKIP["par"] += int(np.sum(~((par > PAR_MIN) & (par < PAR_MAX))))
            SKIP["cheir"] += int(np.sum(~(np.isfinite(Zc) & np.isfinite(Zj))))
            continue
        Xm = X[:, m]
        cloud.append(Xm.T)
        pm, nv = local_plan_med(Xm.T)
        if np.isfinite(pm):
            PAIR_PM.append(pm)
        # --- depth-model agreement at the SAME pixels ---
        r_ok = np.isfinite(ratio[m])
        RATIO_ALL.append(ratio[m][r_ok])
        ZDEP_ALL.append(Zc_dep[m][r_ok])
        ZTRI_ALL.append(Zc[m][r_ok])
        N_NAN_DEP.append(int((~np.isfinite(Zc_dep[m])).sum()))
        N_FAR_DEP.append(int((np.isfinite(Zc_dep[m]) & (Zc_dep[m] > 12)).sum()))
        K_ALL.append(np.full(int(r_ok.sum()), k))
        # depth-route 3D points at same pixels (base frame side)
        Xdc = np.column_stack([pc[0][m], pc[1][m]]); Zdc = Zc_dep[m]
        gg = np.isfinite(Zdc) & (Zdc > ZMIN) & (Zdc < ZMAX)
        if gg.any():
            Zg = Zdc[gg];             Xcg = (Xdc[gg, 0] - cx) * Zg / fx
            Ycg = (Xdc[gg, 1] - CEN[c]) * Zg / fy
            Pw = posc.reshape(3, 1) + Rc @ np.vstack([Xcg, Ycg, Zg])
            cloud_dep.append(Pw.T)
        pair_rows.append((c, j, k, len(good), npass, float(np.median(par[m])),
                          float(np.median(Zc[m])), float(np.median(rc[m])),
                          float(np.nanmedian(ratio[m]))))
    if (bnum + 1) % 60 == 0:
        log("  pairs %d/%d  %d pts  %.0fs" % (bnum + 1, len(bases),
            sum(len(x) for x in cloud), time.time() - t0))

PR = np.array(pair_rows, float)
log("")
log("pair scan: bases=%d  rows=%d  skip=%s  total_tri_pts=%d" %
    (len(bases), len(pair_rows), SKIP, sum(len(x) for x in cloud)))
log("")
log("=== per-k pair diagnostics (metric pose) ===")
log("%5s %6s %8s %8s %9s %9s %9s %9s" %
    ("k", "pairs", "med_mtch", "pass%", "par_deg", "Z_tri", "reproj_px", "Ztri/Zdep"))
for k in KS:
    s = PR[PR[:, 2] == k]
    if not len(s):
        continue
    log("%5d %6d %8.1f %7.1f%% %8.2f %9.2f %9.3f %9.3f" %
        (k, len(s), np.median(s[:, 3]), 100 * np.median(s[:, 4] / np.maximum(s[:, 3], 1)),
         np.median(s[:, 5]), np.median(s[:, 6]), np.median(s[:, 7]), np.median(s[:, 8])))

# ---------- does triangulation agree with the depth model at the SAME pixels? ----------
RA = np.concatenate(RATIO_ALL) if RATIO_ALL else np.zeros(0)
ZD = np.concatenate(ZDEP_ALL) if ZDEP_ALL else np.zeros(0)
ZT = np.concatenate(ZTRI_ALL) if ZTRI_ALL else np.zeros(0)
log("")
log("=== triangulated depth vs DA-V2 depth model, at the SAME matched pixels ===")
log("matched px compared=%d ; depth model NaN here=%d ; depth model >12 m (out of range)=%d" %
    (RA.size, int(np.sum(N_NAN_DEP)), int(np.sum(N_FAR_DEP))))
if RA.size:
    log("overall  Z_tri/Z_dep: p25=%.3f  median=%.3f  p75=%.3f   (1.0 = agree)" %
        (np.nanpercentile(RA, 25), np.nanmedian(RA), np.nanpercentile(RA, 75)))
    log("%-14s %7s %9s %9s %9s" % ("Z_dep bucket", "n", "med_ratio", "med_Zdep", "med_Ztri"))
    for lo, hi in ((0.4, 1.5), (1.5, 3), (3, 6), (6, 9), (9, 12)):
        b = (ZD >= lo) & (ZD < hi)
        if b.sum() < 30:
            continue
        log("%-14s %7d %9.3f %9.2f %9.2f" %
            ("[%.1f,%.1f)" % (lo, hi), int(b.sum()), float(np.nanmedian(RA[b])),
             float(np.nanmedian(ZD[b])), float(np.nanmedian(ZT[b]))))
    # depth-model self-consistency (base vs neighbour at matched pixels) - NOT used for fusion
    log("note: ratio<1 => triangulation puts the point CLOSER than DA-V2 (pose baseline too large or DA says too far)")
    KA = np.concatenate(K_ALL) if K_ALL else np.zeros(0)
    log("")
    log("--- separates 'depth model degrades at range' from 'baseline k error grows' ---")
    log("%5s %-13s %6s %9s" % ("k", "Z_dep bucket", "n", "med_ratio"))
    for k in KS:
        for lo, hi in ((3, 6), (6, 9), (9, 12)):
            b = (KA == k) & (ZD >= lo) & (ZD < hi)
            if b.sum() < 30:
                continue
            log("%5d %-13s %6d %9.3f" % (k, "[%.0f,%.0f)" % (lo, hi), int(b.sum()), float(np.nanmedian(RA[b]))))

# ---------- surface flatness of the triangulated cloud vs depth cloud ----------
def metric(P, npts=20000):
    if P.shape[0] < 200:
        return dict(n=int(P.shape[0]), plan_med=float("nan"), plan_p90=float("nan"),
                    ransac=0, frac=0.0, nvalid=int(P.shape[0]))
    xyz = P.astype(np.float64)
    S = min(npts, xyz.shape[0]); rs = np.random.default_rng(2)
    Q = xyz[rs.choice(xyz.shape[0], S, replace=False)]
    nb = cKDTree(Q).query_ball_point(Q, 0.40, workers=1)
    res = np.full(len(Q), np.nan)
    for i, nl in enumerate(nb):
        if len(nl) < 6:
            continue
        A = Q[nl] - Q[nl].mean(0)
        sv = np.linalg.svd(A, compute_uv=False)
        res[i] = sv[-1] / math.sqrt(len(nl))
    best = 0
    for _ in range(1500):
        p0, p1, p2 = Q[rs.choice(len(Q), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if abs(n[1]) > 0.35:
            continue
        inl = int((np.abs((Q - p0) @ n) <= 0.05).sum())
        if inl > best:
            best = inl
    return dict(n=int(xyz.shape[0]), plan_med=float(np.nanmedian(res)),
                plan_p90=float(np.nanpercentile(res, 90)), ransac=int(best),
                frac=float(best / len(Q)), nvalid=int(np.isfinite(res).sum()))

Pt = np.concatenate(cloud) if cloud else np.zeros((0, 3))
Pd = np.concatenate(cloud_dep) if cloud_dep else np.zeros((0, 3))
log("")
log("=== surface flatness (0.40 m neighbourhood, same metric as prior eval) ===")
log("reference floors: depth-fusion single-frame ~0.0291 m ; fused 0.070 (4Hz&20Hz) ; optimised pose 0.0788")
for nm, P in (("triangulated(ORB,metric pose)", Pt), ("depth-route(same pixels)", Pd)):
    mt = metric(P)
    log("%-30s pts=%7d valid=%6d plan_med=%.4f plan_p90=%.4f RANSAC_inl=%6d (%.1f%%)" %
        (nm, mt["n"], mt["nvalid"], mt["plan_med"], mt["plan_p90"], mt["ransac"], 100 * mt["frac"]))
if Pt.shape[0] and np.isfinite(Pt).all(axis=1).sum() > 100:
    y = Pt[:, 1]
    log("  tri cloud height: p05=%.2f p50=%.2f p95=%.2f m (cam_h=%.2f)" %
        (np.percentile(y, 5), np.percentile(y, 50), np.percentile(y, 95), cam_h))

# ---------- decomposition: within-pair flatness vs across-pair (pose) smear ----------
if PAIR_PM:
    PPa = np.array(PAIR_PM, float)
    log("")
    log("=== decomposition ===")
    log("within-pair triangulated flatness: n_pairs=%d  median=%.4f m  p25=%.4f  p75=%.4f" %
        (PPa.size, float(np.median(PPa)), float(np.percentile(PPa, 25)), float(np.percentile(PPa, 75))))
    log("across-pair (whole cloud) triangulated flatness = see table above")
    log("=> if within-pair << across-pair, the wall smear is CROSS-PAIR POSE INCONSISTENCY, not ORB noise")

np.save(os.path.join(REPO, ".tmp", "_incr_tri_pts.npy"), Pt.astype(np.float32))
np.save(os.path.join(REPO, ".tmp", "_incr_dep_pts.npy"), Pd.astype(np.float32))
np.save(os.path.join(REPO, ".tmp", "_incr_pairs.npy"), PR)
open(OUTP, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
log("done")
