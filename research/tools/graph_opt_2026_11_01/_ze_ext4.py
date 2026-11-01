# -*- coding: utf-8 -*-
"""struct 布局修正：version 字段可能是 uint32，不是 char[256]。"""
import ctypes
U32 = ctypes.c_uint32
dll = ctypes.WinDLL(r'C:\Windows\System32\ze_loader.dll')
dll.zeInit(1)
n = U32(0); dll.zeDriverGet(ctypes.byref(n), None)
drv = (ctypes.c_void_p * n.value)(); dll.zeDriverGet(ctypes.byref(n), drv)

for layout, fields in (
    ('char256+uint32', [('name', ctypes.c_char*256), ('version', U32)]),
    ('char256+char256', [('name', ctypes.c_char*256), ('version', ctypes.c_char*256)]),
):
    class ExtProp(ctypes.Structure):
        _fields_ = fields
    hdrv = ctypes.c_void_p(drv[0])
    ec = U32(0)
    dll.zeDriverGetExtensionProperties(hdrv, ctypes.byref(ec), None)
    props = (ExtProp * ec.value)()
    dll.zeDriverGetExtensionProperties(hdrv, ctypes.byref(ec), props)
    names = []
    for j in range(ec.value):
        nm = props[j].name.decode('utf-8','ignore').strip(chr(0))
        if nm: names.append(nm)
    print('---', layout, '---', flush=True)
    print('  非空扩展数 =', len(names), '/ ', ec.value, flush=True)
    print('  has record_replay_graph =', 'ZE_extension_record_replay_graph' in names, flush=True)
    for nm in names:
        if any(k in nm.lower() for k in ('graph','replay','record','capture')):
            print('   >>>', nm, flush=True)
    print('  sample:', names[:6], flush=True)