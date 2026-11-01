import json
from pathlib import Path
import numpy as np

p = Path(r'H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\policy\policy_live.jsonl')
rows = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
ok = [r for r in rows if 'latency_ms' in r and 'wp' in r]
print('total inference rows:', len(ok))
print()
print('last 12 non-gated rows (newest last):')
shown = []
for r in ok:
    if not r.get('gated'):
        shown.append(r)
for r in shown[-12:]:
    s = np.hypot(r['block'][0][0], r['block'][0][1])
    print('  wp=[%+.2f,%+.2f] step0=%.2f m/s latency=%.0fms' % (r['wp'][0], r['wp'][1], s, r['latency_ms']))
print()
if shown:
    spd = np.array([np.hypot(r['block'][0][0], r['block'][0][1]) for r in shown])
    wp = np.array([r['wp'] for r in shown])
    print('non-gated: n=%d  speed mean=%.2f max=%.2f' % (len(shown), spd.mean(), spd.max()))
    print('wp dist: mean=%.2f max=%.2f' % (np.hypot(wp[:,0],wp[:,1]).mean(), np.hypot(wp[:,0],wp[:,1]).max()))