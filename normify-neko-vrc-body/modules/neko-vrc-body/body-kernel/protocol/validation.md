---
uid: 9e3a0304
id: neko-vrc-body.body-kernel.protocol.validation
parent: neko-vrc-body.body-kernel.protocol
name: {zh: "帧校验", en: "Frame Validation"}
description:
  zh: >
      单设备位姿与整帧的安全校验，未通过不得上线。
      
  en: >
      Safety validation of individual device poses and of a whole frame before it is allowed on the wire.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.789Z"
fingerprint: 4a5943879019db1637b9a60ce6e1045d4061f57b2413cf0c5466cce1190e304d
source:
  - path: "protocol.py"
    line: 15
    end_line: 62
apis:
  - protocol: file
    path: "protocol.py#_validate_device_pose"
    description:
      zh: >
          按安全限位校验单个设备位姿。
          
      en: >
          Validate one device pose against the safety limits.
          
  - protocol: file
    path: "protocol.py#validate_frame"
    description:
      zh: >
          校验整帧，遇到第一处违规即抛错。
          
      en: >
          Validate a whole frame, raising on the first violation.
          
---
