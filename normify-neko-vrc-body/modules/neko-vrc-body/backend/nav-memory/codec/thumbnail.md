---
uid: 9e70000b
id: neko-vrc-body.backend.nav-memory.codec.thumbnail
parent: neko-vrc-body.backend.nav-memory.codec
name: {zh: "缩略图", en: "Thumbnail"}
description:
  zh: >
      缩略图缩放，在感知线程完成因为它很便宜；JPEG 编码留给写盘线程。
      
  en: >
      Thumbnail downscaling, done on the perception thread because it is cheap; the JPEG encode is left to the writer thread.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.638Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 112
    end_line: 117
apis:
  - protocol: file
    path: "backend/nav_memory.py#make_thumbnail"
    description:
      zh: >
          在感知线程里缩放缩略图。
          
      en: >
          Downscale a thumbnail on the perception thread.
          
---
