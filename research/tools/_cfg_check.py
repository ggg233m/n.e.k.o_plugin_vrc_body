import sys
sys.path.insert(0, r'H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body')
from backend.config import load_config
c = load_config({})
print('input.primary =', repr(c.input.primary))
print('osc host/port:', c.osc.send_host, c.osc.send_port)
print('driver:', c.driver_log)