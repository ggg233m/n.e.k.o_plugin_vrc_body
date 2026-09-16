# Avatar 外观重识别模型

本目录携带 `osnet_x0_25_msmt17_dynamic.onnx`：OSNet x0_25（Omni-Scale Feature
Learning for Person Re-Identification, ICCV 2019）在 MSMT17 行人数据集上训练的
重识别网络，输出 512 维 L2 归一化外观嵌入。由
[anriha/osnet_x0_25_msmt17](https://huggingface.co/anriha/osnet_x0_25_msmt17)
（MIT 许可）的固定 batch=16 导出改写输入为动态 batch 得到，权重字节未改动；
改写后与原模型在 batch 1/3/16 下逐元素输出差为 0。

- 输入：`input`，`[batch, 3, 256, 128]`，RGB，/255 后按 ImageNet 均值方差归一化
  （mean 0.485/0.456/0.406，std 0.229/0.224/0.225）。归一化不可省略：裸 /255
  输入实测让跨身份相似度分布明显变糊。
- 输出：`output`，`[batch, 512]`，使用前需 L2 归一化，点积即余弦相似度。
- 消费方：`backend/reid_embedder.py`（`OsnetReidEmbedder`），由
  `vision.identity_reid_model_path` 启用，匹配阈值用
  `vision.identity_reid_model_similarity`（默认 0.75，与直方图的 0.90 不通用）。
- 实测工作点（本仓 `build/_probe` 两轮测试）：真实 VRChat 截图上不同 Avatar
  ≤0.72、同人同视角 ≥0.83；192 个同画风动漫角色、18336 对负样本的极端拼图上
  0.75 阈值误并率 0.80%、扰动（调暗/冷色灯光/框抖动）召回 96.7%。
- 隐私边界与直方图路径一致：嵌入仅在当前后端进程内存中用于会话级 Re-ID，
  不落盘、不做真人生物识别、不冒充 VRChat `usr_`/`avtr_` 身份。
- `checksums.json`：模型文件的字节数与 SHA-256。
