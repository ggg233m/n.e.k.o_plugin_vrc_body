---
uid: 9e3c0100
id: neko-vrc-body.backend.ipc.constants
parent: neko-vrc-body.backend.ipc
name: {zh: "常量与路径解析", en: "Constants and Path Resolution"}
description:
  zh: >
      回环服务进程的路径解析、包名解析、vendor 目录、UI 目录与静态资源表。
      
  en: >
      Path resolution, package-name resolution, vendor directory, UI directory and the static asset table of the loopback service process.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.602Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 31
    end_line: 120
apis:
  - protocol: file
    path: "backend/process.py#resolve_plugin_package_name"
    description:
      zh: >
          解析插件在 VRChat 下的包名。
          
      en: >
          Resolves the plugin package name used under VRChat.
          
  - protocol: file
    path: "backend/process.py#UI_DIRECTORY"
    description:
      zh: >
          回环服务对外暴露的 UI 静态资源目录。
          
      en: >
          The UI static asset directory served by the loopback service.
          
  - protocol: file
    path: "backend/process.py#_UI_ASSETS"
    description:
      zh: >
          UI 路由名到磁盘资源文件的静态资源表。
          
      en: >
          The static asset table mapping UI route names to files on disk.
          
---
