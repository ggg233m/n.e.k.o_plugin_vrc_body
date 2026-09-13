# 人物检测模型

本目录随插件携带当前使用的 `person_detect_v1.3_s` 模型，配置路径相对于插件配置目录解析，不再依赖外部 `anime_models` 目录。

- `person_detect_v1.3_s_model.onnx`：原有 YOLOv8s ONNX 权重，输入尺寸以模型图和插件配置为准，当前配置为 640×640。
- `person_detect_v1.3_s_labels.json`：单类别 `person` 标签。
- `person_detect_v1.3_s_model_artifacts.json`：原有模型结构及导出版本记录。
- `person_detect_v1.3_s_threshold.json`：原始阈值记录，仅留作参考；运行时仍使用 `plugin.toml` 的 `vision.confidence_threshold`，本次不调整。
- `checksums.json`：上述四个原始文件的字节数和 SHA-256。

本次从本机已有 `anime_models` 资源复制，未转换、量化或重新训练权重。推理继续使用插件的 OpenVINO／ONNX Runtime 适配器；本目录不包含 Python 运行环境或 GPU 驱动。
