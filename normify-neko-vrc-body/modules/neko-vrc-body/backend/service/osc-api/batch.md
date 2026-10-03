---
uid: 9e3e001e
id: neko-vrc-body.backend.service.osc-api.batch
parent: neko-vrc-body.backend.service.osc-api
name: {zh: "批量、聊天框与取消", en: "Batch, Chatbox and Cancel"}
description:
  zh: >
      批量 OSC 下发、聊天框输入，以及取消仍在队列中的内容。
      
  en: >
      Batched OSC dispatch, chatbox typing and the cancel path for anything still queued.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.749Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2979
    end_line: 3057
apis:
  - protocol: file
    path: "backend/service.py#BackendService.send_osc_batch"
    description:
      zh: >
          按顺序发送一批 OSC 命令。
          
      en: >
          Send a batch of OSC commands in order.
          
  - protocol: file
    path: "backend/service.py#BackendService.send_chatbox"
    description:
      zh: >
          在 VRChat 聊天框输入一行文本。
          
      en: >
          Type a line into the VRChat chatbox.
          
  - protocol: file
    path: "backend/service.py#BackendService.cancel_inputs"
    description:
      zh: >
          取消全部待执行的调度输入。
          
      en: >
          Cancel every pending scheduled input.
          
---
