---
uid: 9e3a0213
id: neko-vrc-body.plugin.agent.journal-changes
parent: neko-vrc-body.plugin.agent
name: {zh: "增量日志", en: "Delta Journalling"}
description:
  zh: >
      把世界增量 diff 成日志条目，让 agent 读到的是事件而不是裸状态。
      
  en: >
      Diffing a world delta into journal entries the agent can read as events rather than raw state.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.848Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 744
    end_line: 786
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._journal_changes"
    description:
      zh: >
          把世界增量转成给 agent 的日志条目。
          
      en: >
          Turn a world delta into journal entries for the agent.
          
---
