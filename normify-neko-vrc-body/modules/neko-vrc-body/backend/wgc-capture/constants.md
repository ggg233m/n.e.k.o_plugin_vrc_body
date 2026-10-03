---
uid: 9e41001d
id: neko-vrc-body.backend.wgc-capture.constants
parent: neko-vrc-body.backend.wgc-capture
name: {zh: "WGC 与 D3D11 常量", en: "WGC and D3D11 Constants"}
description:
  zh: >
      D3D11 与 WGC 所需的 IID、驱动类型、用途标志与虚表索引。
      
  en: >
      The IIDs, driver types, usage flags and vtable indexes needed by D3D11 and the Windows Graphics Capture path.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.772Z"
fingerprint: 0ff7a755ae59a0b73e3394a4115de54c92c1e6cd083b2f71b449a093a7cca46a
source:
  - path: "backend/wgc_capture.py"
    line: 29
    end_line: 39
apis:
  - protocol: file
    path: "backend/wgc_capture.py#_IID_ID3D11TEXTURE2D"
    description:
      zh: >
          ID3D11Texture2D 的接口标识符。
          
      en: >
          The interface identifier of ID3D11Texture2D.
          
  - protocol: file
    path: "backend/wgc_capture.py#_VTBL_CREATE_TEXTURE2D"
    description:
      zh: >
          ID3D11Device 虚表中 CreateTexture2D 的槽位索引。
          
      en: >
          The vtable slot index of CreateTexture2D on ID3D11Device.
          
---
