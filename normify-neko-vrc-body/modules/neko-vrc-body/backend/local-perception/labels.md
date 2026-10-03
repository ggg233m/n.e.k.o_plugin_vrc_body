---
uid: "9e410002"
id: neko-vrc-body.backend.local-perception.labels
parent: neko-vrc-body.backend.local-perception
name: {zh: "标签加载", en: "Label Loading"}
description:
  zh: >
      从 JSON 数组或 ultralytics 风格的对象里加载标签列表。
      
  en: >
      Loads the label list from either a JSON array or an ultralytics-style object.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.615Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 313
    end_line: 374
apis:
  - protocol: file
    path: "backend/local_perception.py#_load_labels"
    description:
      zh: >
          解析并校验类别标签，格式不符时抛出可读错误。
          
      en: >
          Parses and validates the class labels, raising a readable error when the shape is wrong.
          
---
