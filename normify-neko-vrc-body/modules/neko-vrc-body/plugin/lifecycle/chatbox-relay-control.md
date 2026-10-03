---
uid: 9e3a0205
id: neko-vrc-body.plugin.lifecycle.chatbox-relay-control
parent: neko-vrc-body.plugin.lifecycle
name: {zh: "聊天框中继控制", en: "Chatbox Relay Control"}
description:
  zh: >
      插件外壳持有的后台聊天框中继的启动与停止。
      
  en: >
      Starting and stopping the background chatbox relay owned by the plugin shell.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.873Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 260
    end_line: 282
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._start_chatbox_relay"
    description:
      zh: >
          启动后台聊天框中继轮询器。
          
      en: >
          Start the background chatbox relay poller.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._stop_chatbox_relay"
    description:
      zh: >
          停止后台聊天框中继轮询器。
          
      en: >
          Stop the background chatbox relay poller.
          
---
