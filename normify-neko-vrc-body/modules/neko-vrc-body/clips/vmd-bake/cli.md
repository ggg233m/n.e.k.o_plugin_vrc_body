---
uid: 9e3a0394
id: neko-vrc-body.clips.vmd-bake.cli
parent: neko-vrc-body.clips.vmd-bake
name: {zh: "烘焙命令行", en: "Bake CLI"}
description:
  zh: >
      命令行烘焙入口：读入 VMD、重定向并写出 .nya 片段。
      
  en: >
      The command-line bake entry point: read a VMD, retarget it and write a .nya clip.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.809Z"
fingerprint: ff195c90f5686de6f36a074e773db33f01c0cb0477451740a419014ee45f511b
source:
  - path: "vmd_bake.py"
    line: 481
    end_line: 596
apis:
  - protocol: file
    path: "vmd_bake.py#bake_vmd_file"
    description:
      zh: >
          把 VMD 文件直接烘焙成 .nya 片段。
          
      en: >
          Bake a VMD file straight into a .nya clip.
          
  - protocol: file
    path: "vmd_bake.py#_main"
    description:
      zh: >
          命令行入口。
          
      en: >
          Command-line entry point.
          
---
