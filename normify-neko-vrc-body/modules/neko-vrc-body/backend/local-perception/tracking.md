---
uid: "9e410001"
id: neko-vrc-body.backend.local-perception.tracking
parent: neko-vrc-body.backend.local-perception
name: {zh: "目标跟踪", en: "Object Tracking"}
description:
  zh: >
      按类别匹配的边界框跟踪器，刻意不用外观特征向量以保持延迟与隐私边界可预测。
      
  en: >
      A per-class bounding-box tracker that deliberately avoids appearance feature vectors so latency and the privacy boundary stay predictable.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.616Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 201
    end_line: 310
apis:
  - protocol: file
    path: "backend/local_perception.py#_IoUTracker.update"
    description:
      zh: >
          把新检测按交并比与类别关联到已有轨迹并完成更新。
          
      en: >
          Associates new detections to existing tracks by IoU and class, then advances the tracker state.
          
  - protocol: file
    path: "backend/local_perception.py#_iou"
    description:
      zh: >
          计算两个边界框的交并比，供关联与去重共用。
          
      en: >
          Computes the intersection-over-union of two boxes, shared by association and de-duplication.
          
---
