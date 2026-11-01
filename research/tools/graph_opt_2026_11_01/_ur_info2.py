# -*- coding: utf-8 -*-
"""用正确的 API 查 UR 层对 0x2080 的回答。"""
import ctypes
U32 = ctypes.c_uint32; SZ = ctypes.c_size_t
ur = ctypes.WinDLL(r'D:\Program Files (x86)\Intel\oneAPI\2026.0\bin\ur_loader.dll')
print('urLoaderInit rc =', ur.urLoaderInit(0, None), flush=True)
na = U32(0)
ur.urAdapterGet(0, None, ctypes.byref(na))
print('adapters =', na.value, flush=True)
ad = (ctypes.c_void_p * max(1,na.value))()
ur.urAdapterGet(na.value, ad, ctypes.byref(na))
nd = U32(0)
ur.urDeviceGet(ctypes.c_void_p(ad[0]), 0, ctypes.byref(nd), 0, None)
print('devices =', nd.value, flush=True)
devs = (ctypes.c_void_p * max(1,nd.value))()
ur.urDeviceGet(ctypes.c_void_p(ad[0]), 0, ctypes.byref(nd), 0, devs)
for i in range(nd.value):
    h = ctypes.c_void_p(devs[i])
    val = U32(0); sz = SZ(0)
    rc = ur.urDeviceGetInfo(h, 0x2080, ctypes.sizeof(U32), ctypes.byref(val), ctypes.byref(sz))
    print(f'dev[{i}] 0x2080 GRAPH_RECORD_AND_REPLAY rc={rc} value={val.value}', flush=True)
    v2 = U32(0); s2 = SZ(0)
    rc2 = ur.urDeviceGetInfo(h, 0x1000, ctypes.sizeof(U32), ctypes.byref(v2), ctypes.byref(s2))
    print(f'dev[{i}] 0x1000 对照 rc={rc2} value={v2.value}', flush=True)
    # 枚举 device extensions
    ne = U32(0)
    class Ext(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char*256), ('version', U32)]
    try:
        rc = ur.urDeviceGetInfo(h, 0x1030, 0, None, ctypes.byref(SZ(0)))
        print(f'  ext 查询 rc={rc}', flush=True)
    except Exception as e:
        print('   ext 查询失败', type(e).__name__, flush=True)