---
uid: 9e42001c
id: neko-vrc-body.backend.nav-online.navigator.commands
parent: neko-vrc-body.backend.nav-online.navigator
name: {zh: "导航命令入口", en: "Navigation command entry"}
description:
  zh: >
      导航命令入口：去某处、探索与取消。
      
  en: >
      Navigation command entry points: go somewhere, explore, and cancel.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.645Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 1050
    end_line: 1074
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.goto"
    description:
      zh: >
          发起去某处的导航命令。
          
      en: >
          Starts a goto navigation command.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.explore"
    description:
      zh: >
          发起探索导航命令。
          
      en: >
          Starts an explore navigation command.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.cancel"
    description:
      zh: >
          取消正在执行的导航命令。
          
      en: >
          Cancels the running navigation command.
          
---
