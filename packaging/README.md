# 发行包重建

使用 Windows x64、Python 3.11 和可导入 N.E.K.O SDK 的宿主开发环境。
先将 `uv.lock` 中的依赖同步到本项目 `vendor/`（不要安装到共享宿主环境）：

```powershell
uv export --frozen --no-dev --no-emit-project --format requirements-txt --output-file build/dependencies.lock
uv pip install --python <Python3.11路径> --target vendor --require-hashes --only-binary :all: -r build/dependencies.lock
<宿主Python路径> packaging/build_neko.py --host <N.E.K.O源码根目录>
```

开始前创建 `build/`。更换 Python 小版本、依赖或平台时，使用新的暂存目录重建 vendor；不要混合不同 ABI 的 `.pyd` 文件。
构建使用宿主官方打包器、包检查器和安装器，测试安装目的地仅为本项目的 `build/安装验证 空格-*`。
交付使用文件名带 `-profile-safe` 的包：移除打包器自动生成、无业务参数的默认 profile，并重新计算 payload 哈希，避免旧宿主同名 profile 归属不明时导入失败。插件代码、模型、vendor、元数据均保持不变，现有 profile 不被接管或覆盖。
可运行 `packaging/verify_profile_conflict.py --host <N.E.K.O源码根目录>`，复现旧包冲突并验证修订包安装后原有 profile 字节不变。
读取 `build/package-verification.json` 的 `installed_plugin`，再用无第三方库的 Python 3.11 验证安装结果：

```powershell
<干净Python路径> -I -S -B packaging/verify_runtime.py <installed_plugin路径>
```

`-I -S` 禁用用户目录和系统 site-packages，只允许后端入口加载包内 vendor。
该验证不启动屏幕捕获，不输出设备控制指令；完成后停止测试后端。
