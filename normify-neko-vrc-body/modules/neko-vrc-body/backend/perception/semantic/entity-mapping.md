---
uid: "9e400011"
id: neko-vrc-body.backend.perception.semantic.entity-mapping
parent: neko-vrc-body.backend.perception.semantic
name: {zh: "语义实体映射", en: "Semantic Entity Mapping"}
description:
  zh: >
      把 VLM 返回的语义实体对齐到检测器产出的框上。
      
  en: >
      Aligns the semantic entities returned by the VLM onto the boxes produced by the detector.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.716Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1751
    end_line: 1847
apis:
  - protocol: file
    path: "backend/vision.py#_semantic_entity_mapping"
    description:
      zh: >
          把 VLM 返回的语义实体按重叠度与类型对齐到检测框。
          
      en: >
          Aligns the semantic entities returned by the VLM to detection boxes by overlap and type.
          
  - protocol: file
    path: "backend/vision.py#_semantic_iou"
    description:
      zh: >
          计算两个矩形的交并比，用于判断语义实体与检测框是否同一目标。
          
      en: >
          Computes the intersection-over-union of two rectangles to decide whether a semantic entity and a detection box are the same target.
          
---
