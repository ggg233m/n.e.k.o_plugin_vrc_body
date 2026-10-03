---
uid: 9e3a0393
id: neko-vrc-body.clips.vmd-bake.document
parent: neko-vrc-body.clips.vmd-bake
name: {zh: "文档重定向", en: "Document Retargeting"}
description:
  zh: >
      把已解算的动作文档转成运行时回放所用的 v1 版 .nya 布局。
      
  en: >
      Converting an already-solved motion document into the version-1 .nya layout the runtime plays back.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.810Z"
fingerprint: ff195c90f5686de6f36a074e773db33f01c0cb0477451740a419014ee45f511b
source:
  - path: "vmd_bake.py"
    line: 263
    end_line: 478
apis:
  - protocol: file
    path: "vmd_bake.py#retarget_solved_document"
    description:
      zh: >
          把已解算的动作文档转成 v1 版本的 .nya 文档。
          
      en: >
          Convert a solved motion document into a version-1 .nya document.
          
  - protocol: file
    path: "vmd_bake.py#convert_solved_file"
    description:
      zh: >
          把已解算的文档文件转成 .nya 文件。
          
      en: >
          Convert a solved document file into a .nya file.
          
---
