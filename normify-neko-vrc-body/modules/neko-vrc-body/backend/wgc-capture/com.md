---
uid: 9e41001e
id: neko-vrc-body.backend.wgc-capture.com
parent: neko-vrc-body.backend.wgc-capture
name: {zh: "COM 绑定层", en: "COM Interop Layer"}
description:
  zh: >
      ctypes/COM 结构体与虚表调用，查询失败返回 None 而不抛异常。
      
  en: >
      ctypes/COM structures and vtable calls where a failed query returns None instead of raising.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.771Z"
fingerprint: 0ff7a755ae59a0b73e3394a4115de54c92c1e6cd083b2f71b449a093a7cca46a
source:
  - path: "backend/wgc_capture.py"
    line: 51
    end_line: 120
apis:
  - protocol: file
    path: "backend/wgc_capture.py#_vtbl_call"
    description:
      zh: >
          按虚表索引调用 COM 方法。
          
      en: >
          Invokes a COM method by vtable index.
          
  - protocol: file
    path: "backend/wgc_capture.py#wgc_supported"
    description:
      zh: >
          判断当前系统是否支持 Windows Graphics Capture。
          
      en: >
          Determines whether Windows Graphics Capture is supported on the current system.
          
  - protocol: file
    path: "backend/wgc_capture.py#_query_interface"
    description:
      zh: >
          查询接口，失败时返回 None 而不抛异常。
          
      en: >
          Queries for an interface, returning None on failure instead of raising.
          
---
