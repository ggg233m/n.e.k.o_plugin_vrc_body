---
uid: 9e3a020e
id: neko-vrc-body.plugin.debug.command-dispatch
parent: neko-vrc-body.plugin.debug
name: {zh: "面板与调试分发", en: "Panel and Debug Dispatch"}
description:
  zh: >
      宿主 UI 调用的 panel_command 与 debug_command 入口，及其背后的开关/命令名表。
      
  en: >
      The panel_command and debug_command entry points the host UI calls, and the switch/command name tables behind them.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.866Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2072
    end_line: 2181
  - path: "__init__.py"
    line: 131
    end_line: 180
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin.panel_command"
    description:
      zh: >
          宿主面板下发的开关命令。
          
      en: >
          Panel switch commands issued from the host UI.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin.debug_command"
    description:
      zh: >
          宿主面板下发的调试命令。
          
      en: >
          Debug commands issued from the host UI.
          
---
