---
uid: 9e42000c
id: neko-vrc-body.backend.navigator.target-observation
parent: neko-vrc-body.backend.navigator
name: {zh: "目标观测缓存", en: "Target observation cache"}
description:
  zh: >
      同一目标的近期可靠观测缓存与平滑。
      
  en: >
      Cache and smoothing of recent reliable observations of the same target.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.682Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1779
    end_line: 1857
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._record_target_observation"
    description:
      zh: >
          记录一次目标观测并按可靠性更新缓存。
          
      en: >
          Records one target observation and updates the cache according to its reliability.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._cached_target_for_goal"
    description:
      zh: >
          取出该目标平滑后的缓存观测。
          
      en: >
          Returns the smoothed cached observation for the goal.
          
---
