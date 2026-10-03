---
uid: 9e3a0033
id: neko-vrc-body.packaging.build
parent: neko-vrc-body.packaging
name: {zh: "打包构建", en: "Package Build"}
description:
  zh: >
      驱动宿主官方打包器，并在隔离目录里验证产出的安装包。
      
  en: >
      Drives the host's official packager and verifies the resulting install in an isolated directory.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.842Z"
fingerprint: b4c930198a52c05adb15f6bc7b23530e8cc9a585d2a5d930075ea61b8d9614cc
source:
  - path: "packaging/build_neko.py"
    line: 22
    end_line: 114
apis:
  - protocol: file
    path: "packaging/build_neko.py#prune_stale_install_dirs"
    description:
      zh: >
          删除上一轮构建遗留的隔离安装目录。
          
      en: >
          Delete leftover isolated install directories from the previous build.
          
  - protocol: file
    path: "packaging/build_neko.py#main"
    description:
      zh: >
          执行完整的构建并验证流水线。
          
      en: >
          Run the whole build-and-verify pipeline.
          
---
