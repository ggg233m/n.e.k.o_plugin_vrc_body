import json
from pathlib import Path
p = Path(r'H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\policy\policy_live.jsonl')
lines = p.read_text(encoding='utf-8').splitlines()
print('rows:', len(lines))
n_real = sum(1 for l in lines if 'latency_ms' in l)
print('real inference rows:', n_real)
last = json.loads(lines[-1])
print('last row wp:', last.get('wp'), 'gated:', last.get('gated'), 'latency:', last.get('latency_ms'))