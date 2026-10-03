---
uid: 9e3a0350
id: neko-vrc-body.device-io.vmc.bones
parent: neko-vrc-body.device-io.vmc
name: {zh: "骨骼表", en: "Bone Tables"}
description:
  zh: >
      骨骼表：骨架根节点、一帧必须包含的骨骼、设备映射与手指链。
      
  en: >
      Bone tables: the skeleton root, the bones a frame must contain, the device mapping and the finger chains.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.831Z"
fingerprint: a49ba23fb5ec0e9ebaa1ef04e8b0a3ded0b99c3a792d7b1568bce88cb271c578
source:
  - path: "vmc_idle.py"
    line: 33
    end_line: 110
apis:
  - protocol: file
    path: "vmc_idle.py#_ROOT"
    description:
      zh: >
          其余骨骼都相对它解析的骨架根节点。
          
      en: >
          The skeleton root every other bone is resolved against.
          
  - protocol: file
    path: "vmc_idle.py#_REQUIRED_BONES"
    description:
      zh: >
          一帧可用所必须全部存在的骨骼。
          
      en: >
          Bones that must all be present for a frame to be usable.
          
  - protocol: file
    path: "vmc_idle.py#_DEVICE_BONES"
    description:
      zh: >
          骨骼到 AnyaDance 设备的映射。
          
      en: >
          Bone to AnyaDance device mapping.
          
  - protocol: file
    path: "vmc_idle.py#_CANONICAL_HEIGHT_M"
    description:
      zh: >
          求解所参照的规范化身身高。
          
      en: >
          The canonical avatar height the solve is scaled against.
          
---
