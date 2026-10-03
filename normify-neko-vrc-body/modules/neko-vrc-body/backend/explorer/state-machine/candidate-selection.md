---
uid: 9e50001e
id: neko-vrc-body.backend.explorer.state-machine.candidate-selection
parent: neko-vrc-body.backend.explorer.state-machine
name: {zh: "候选挑选", en: "Candidate Selection"}
description:
  zh: >
      按置信度与方位挑选搜索候选。
      
  en: >
      Picks the search candidate by confidence and bearing.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.599Z"
fingerprint: 3ceb238589b7465b0375c659e6db1f5f984144bfe544a81d29d28423b1db0ad9
source:
  - path: "backend/explorer.py"
    line: 167
    end_line: 188
apis:
  - protocol: file
    path: "backend/explorer.py#ExplorerStateMachine._select_candidate"
    description:
      zh: >
          挑选要搜索的实体。
          
      en: >
          Picks the entity to search for.
          
---
