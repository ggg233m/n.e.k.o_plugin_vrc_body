---
uid: 9e3a0032
id: neko-vrc-body.tools.xsession-align
parent: neko-vrc-body.tools
name: {zh: "跨会话对齐命令行", en: "Cross-Session Align CLI"}
description:
  zh: >
      跨会话对齐入口的命令行封装，让已保存的会话可以离线对齐。
      
  en: >
      Command-line wrapper around the cross-session alignment entry point, so a saved session can be aligned offline.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.944Z"
fingerprint: 80bf40ca66dac544928935fc4a7b67c3a4c925dbed2213d314233482077d14d5
source:
  - path: "tools/xsession_align.py"
    line: 23
    end_line: 59
apis:
  - protocol: file
    path: "tools/xsession_align.py#main"
    description:
      zh: >
          在命令行把一个已保存会话对齐进另一个会话的世界系。
          
      en: >
          Align one saved session into another session's world frame from the command line.
          
---
