---
uid: 9e3a03a0
id: neko-vrc-body.clips.expression-motion.overlay
parent: neko-vrc-body.clips.expression-motion
name: {zh: "叠加层描述符", en: "Overlay Descriptor"}
description:
  zh: >
      叠加层描述符及它可驱动的手势集合，含进度、取消权重与过期判定。
      
  en: >
      The overlay descriptor and the gesture set it may drive, with progress, cancel weight and expiry.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.806Z"
fingerprint: c259af45ca5ae068f69fda4bf52ec86545024c5ec90b3354281011688df04f0f
source:
  - path: "expression_motion.py"
    line: 20
    end_line: 50
apis:
  - protocol: file
    path: "expression_motion.py#ExpressionOverlay"
    description:
      zh: >
          一个表情叠加层：手势、侧别、能量与存活期。
          
      en: >
          One expression overlay: gesture, side, energy and lifetime.
          
  - protocol: file
    path: "expression_motion.py#EXPRESSION_GESTURES"
    description:
      zh: >
          表情层会驱动的那些手势。
          
      en: >
          The gestures the expression layer will drive.
          
  - protocol: file
    path: "expression_motion.py#ExpressionOverlay.expired"
    description:
      zh: >
          叠加层进度，以及它是否已过期。
          
      en: >
          Overlay progress and whether it has expired.
          
---
