---
uid: "9e410021"
id: neko-vrc-body.backend.webui.pages
parent: neko-vrc-body.backend.webui
name: {zh: "静态页面", en: "Static Pages"}
description:
  zh: >
      四个无需构建的静态页面：主控制台、navmesh 视图、navmesh 记忆管理与覆盖热力图。
      
  en: >
      Four build-free static pages: the main console, the navmesh view, navmesh memory management and the coverage heatmap.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.771Z"
fingerprint: 9e25b2e402df4d10113db2b37c54850fa2e16408a46a5b7bdc33a43f0f55db85
source:
  - path: "backend/standalone_ui/index.html"
  - path: "backend/standalone_ui/app.js"
  - path: "backend/standalone_ui/styles.css"
  - path: "backend/standalone_ui/navmesh.html"
  - path: "backend/standalone_ui/navmesh.js"
  - path: "backend/standalone_ui/navmesh_memory.html"
  - path: "backend/standalone_ui/navmesh_memory.js"
  - path: "backend/standalone_ui/coverage.html"
  - path: "backend/standalone_ui/coverage.js"
apis:
  - protocol: file
    path: "backend/standalone_ui/app.js"
    description:
      zh: >
          主控制台脚本：状态展示、功能开关与配置编辑。
          
      en: >
          Main control console: status, toggles and config editing.
          
  - protocol: file
    path: "backend/standalone_ui/navmesh.js"
    description:
      zh: >
          navmesh 视图脚本：渲染当前 navmesh 与其运行状态。
          
      en: >
          Navmesh view: renders the current navmesh and its runtime state.
          
  - protocol: file
    path: "backend/standalone_ui/navmesh_memory.js"
    description:
      zh: >
          navmesh 记忆页脚本：浏览并清理已保存的 navmesh 记忆。
          
      en: >
          Navmesh memory page: browses and prunes stored navmesh memories.
          
  - protocol: file
    path: "backend/standalone_ui/coverage.js"
    description:
      zh: >
          覆盖页脚本：绘制探索覆盖热力图。
          
      en: >
          Coverage page: draws the exploration coverage heatmap.
          
---
