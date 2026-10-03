---
uid: 9e3a0370
id: neko-vrc-body.device-io.chatbox-relay.line
parent: neko-vrc-body.device-io.chatbox-relay
name: {zh: "聊天框文本与读取助手", en: "Chatbox Line and Readers"}
description:
  zh: >
      聊天框文本类型，以及兼容宿主版本的读取助手 —— 字段既可能来自 dataclass 也可能来自原始映射。
      
  en: >
      The chatbox line type and the host-version-tolerant readers that pull a field off either a dataclass or a raw mapping.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.817Z"
fingerprint: 8446eceac711a656f2cb14da577f38da57d62b7c6a8ec1866e9f0f6a1f22df34
source:
  - path: "chatbox_relay.py"
    line: 30
    end_line: 94
apis:
  - protocol: file
    path: "chatbox_relay.py#ChatboxLine"
    description:
      zh: >
          一条已通过过滤、准备发送的聊天框文本。
          
      en: >
          One filtered chatbox line ready to send.
          
  - protocol: file
    path: "chatbox_relay.py#_normalize_text"
    description:
      zh: >
          把记录正文压成单行。
          
      en: >
          Collapse a record body into a single line.
          
  - protocol: file
    path: "chatbox_relay.py#_record_id"
    description:
      zh: >
          导出一个用于去重的稳定记录 id。
          
      en: >
          Derive a stable record id for de-duplication.
          
---
