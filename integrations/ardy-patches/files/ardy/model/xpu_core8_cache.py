"""按运行环境和源码指纹持久化编译产物；不包含模型权重。"""
import hashlib
import json
import os
import platform
from pathlib import Path
import torch

def _folder(model):
    here = Path(__file__).parent
    signature = dict(torch=torch.__version__, python=platform.python_version(),
                     gpu=torch.xpu.get_device_name(), platform=platform.version(),
                     shapes=[list(p.shape) for p in model.parameters()])
    signature['sources'] = {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                            for name in ('xpu_core8_pipeline.py', 'xpu_core8_runtime.py', 'xpu_inverse.py', 'xpu_core8_warmup.py', 'xpu_core8_cache.py')}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    # 未配置时使用当前用户缓存，避免依赖开发电脑的盘符。
    default_cache = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.cache') / 'ardy' / 'core8-compile'
    folder = Path(os.environ.get('ARDY_CORE8_ARTIFACT_DIR') or default_cache) / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'environment.json').write_text(json.dumps(signature, indent=2), encoding='utf-8')
    return folder

def restore_core8_cache(model):
    folder = _folder(model)
    model._core8_artifact_folder = folder
    path = folder/'artifacts.bin'
    if not path.exists():
        print('[Core8 缓存] 无匹配的产物包，将复用磁盘内核缓存并在预热后保存。', flush=True)
        return False
    try:
        info = torch.compiler.load_cache_artifacts(path.read_bytes())
        print(f'[Core8 缓存] 恢复产物包：{info}', flush=True)
        return info is not None
    except Exception as exc:
        print(f'[Core8 缓存] 产物包不可用，重新预热：{type(exc).__name__}', flush=True)
        return False

def save_core8_cache(model):
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            return
        folder = model._core8_artifact_folder
        temporary = folder/f'artifacts.{os.getpid()}.tmp'
        temporary.write_bytes(artifacts[0])
        temporary.replace(folder/'artifacts.bin')
        print(f'[Core8 缓存] 已保存 {len(artifacts[0])/1048576:.1f} MiB 编译产物，下次自动恢复。', flush=True)
    except Exception as exc:
        print(f'[Core8 缓存] 保存失败：{type(exc).__name__}；本次生成仍可继续。', flush=True)
