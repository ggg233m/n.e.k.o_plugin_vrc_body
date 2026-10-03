---
uid: "9e700002"
id: neko-vrc-body.backend.direction-memory.store.predict-refuse
parent: neko-vrc-body.backend.direction-memory.store
name: {zh: "预测与拒绝", en: "Prediction and Refusal"}
description:
  zh: >
      预测、动态障碍标记、受阻扇区清单与直接拒绝判定。
      
  en: >
      Prediction, dynamic-obstacle marking, the list of blocked sectors and the outright-refusal decision.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.596Z"
fingerprint: 10cac7e571fabcdb167f981ba4b09c03bf51d330c67e6c7e3f92b5caac9a3eee
source:
  - path: "backend/direction_memory.py"
    line: 257
    end_line: 382
apis:
  - protocol: file
    path: "backend/direction_memory.py#DirectionMemory.predict"
    description:
      zh: >
          预测某扇区受阻的概率。
          
      en: >
          Predict the probability that a sector is blocked.
          
  - protocol: file
    path: "backend/direction_memory.py#DirectionMemory.mark_dynamic_obstacle"
    description:
      zh: >
          标记某扇区被动态障碍挡住。
          
      en: >
          Mark a sector blocked by a dynamic obstacle.
          
  - protocol: file
    path: "backend/direction_memory.py#DirectionMemory.should_refuse"
    description:
      zh: >
          是否应当直接拒绝某个方位。
          
      en: >
          Whether a bearing should be refused outright.
          
---
