---
uid: "9e600015"
id: neko-vrc-body.backend.nav-memory.codec
parent: neko-vrc-body.backend.nav-memory
name: {zh: "记忆编解码", en: "Memory Codec"}
description:
  zh: >
      目录命名、原子写入、特征编解码与缩略图缩放；缩放留在感知线程，编码留给写盘线程。
      
  en: >
      Directory naming, atomic writes, feature encoding and thumbnail downscaling; the downscale stays on the perception thread while encoding is left to the writer.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.637Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 40
    end_line: 117
  - path: "backend/nav_memory.py"
    line: 515
    end_line: 517
---
