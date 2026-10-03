---
uid: 9e3a0029
id: neko-vrc-body.world-salience.bands
parent: neko-vrc-body.world-salience
name: {zh: "距离与方位分档", en: "Proximity and Bearing Bands"}
description:
  zh: >
      量化的距离档与八分方位；检测框贴边时拒绝假装知道远近。
      
  en: >
      Quantised proximity bands and eight-way bearing sectors, deliberately refusing to guess when the detection box is clipped.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.944Z"
fingerprint: 9514e5be7514a04afbe83acf4af23398f6939c9ad2d179fbcb3a12d4af685844
source:
  - path: "world_salience.py"
    line: 62
    end_line: 98
apis:
  - protocol: file
    path: "world_salience.py#proximity_band"
    description:
      zh: >
          把表观高度量化成距离档；检测框贴边时返回 unknown。
          
      en: >
          Quantise an apparent height into a proximity band; returns unknown when the box is clipped.
          
  - protocol: file
    path: "world_salience.py#bearing_sector"
    description:
      zh: >
          把方位角量化成八个带左右的扇区。
          
      en: >
          Quantise a bearing angle into one of eight left/right sectors.
          
---
