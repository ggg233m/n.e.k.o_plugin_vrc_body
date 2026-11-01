import json
from pathlib import Path
import numpy as np

p = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\policy\policy_live.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
ok = [r for r in rows if "latency_ms" in r]
noframe = [r for r in rows if r.get("reason") == "no_frame"]

print("total rows     :", len(rows))
print("no_frame       :", len(noframe))
print("real inferences:", len(ok))
print("errors         :", len([r for r in rows if str(r.get("reason","")).startswith("error")]))
if ok:
    lat = np.array([r["latency_ms"] for r in ok])
    print("latency p50/p95/max : %.1f / %.1f / %.1f ms" % (np.percentile(lat,50), np.percentile(lat,95), lat.max()))
    print("stale count    :", sum(1 for r in ok if r.get("stale")))
    wp = np.array([r["wp"] for r in ok])
    dist = np.hypot(wp[:,0], wp[:,1])
    print("wp dist        : mean=%.3f  frac>0.1=%.3f" % (dist.mean(), (dist>0.1).mean()))
    gated = np.array([bool(r.get("gated")) for r in ok])
    print("gated frac     : %.3f" % gated.mean())
    sp = np.array([np.hypot(r["block"][0][0], r["block"][0][1]) for r in ok])
    print("step0 speed    : mean=%.3f p95=%.3f max=%.3f" % (sp.mean(), np.percentile(sp,95), sp.max()))
    print("vetoed rows    :", sum(1 for r in ok if r.get("vetoed")))
    nz = sp[~gated] if gated.any() else sp
    if len(nz):
        print("non-gated speed: mean=%.3f max=%.3f" % (nz.mean(), nz.max()))
