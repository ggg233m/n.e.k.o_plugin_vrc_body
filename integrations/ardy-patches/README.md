# ARDY 补丁与安装

这里是ARDY接入补丁的维护位置。包含49个覆盖文件、官方基线及SHA-256清单、应用工具、原项目许可证与归属说明；不包含官方完整源码、模型、虚拟环境或编译缓存。此目录随插件源码提交，但由 `pyproject.toml` 的 `integrations` 排除规则阻止进入宿主插件安装包。

## 推荐安装：使用轻量接入包

1. 准备Windows x64、64位Python 3.11、对应显卡驱动及Visual Studio C++ Build Tools。
2. 下载官方ARDY源码，使用已验证基线 `693f74d13b3d04a0a22ce127ee79c929dd89756b`。不要用未知版本或其他修改直接覆盖。
3. 解压维护者从当前仓库生成的轻量接入包，将源码放在包内 `ardy/`。已有源码和模型也可通过包内 `ardy.local.json` 指定位置。
4. 按[完整资源与安装说明](../../Docs/ARDY/PORTABLE.md)准备Core40、Llama及两套LLM2Vec权重。
5. 双击接入包的 `安装环境.cmd`，选择NVIDIA CUDA或Intel XPU。安装器会先校验、备份并应用本目录的补丁，再在接入包自己的 `.venv` 中安装依赖和编译ARDY。
6. 在NEKO宿主面板启用YUI插件及ARDY选项，保存并按提示重载。双击接入包的 `启动NPC动作.cmd`，等待模型就绪和世界握手。无需令牌或宿主命令行。

插件仓库目前只提供补丁与动作服务源码，完整安装器、图重放运行工具及viser依赖仍由轻量接入包提供。仅应用补丁不等于完成后端安装。

## 从源码仓库单独应用补丁

用于已有后端环境、需要更新补丁的开发者；以下PowerShell在插件仓库根目录运行，不是宿主命令行：

```powershell
$ArdySource = Read-Host '输入官方ARDY源码目录（包含setup.py）'
py -3.11 ./integrations/ardy-patches/apply.py $ArdySource --check
if ($LASTEXITCODE -ne 0) { throw '预检失败，请检查源码版本和修改' }
py -3.11 ./integrations/ardy-patches/apply.py $ArdySource
if ($LASTEXITCODE -ne 0) { throw '补丁应用失败' }
```

也可在任意目录用Python调用本目录的 `apply.py`，参数传入源码目录；补丁位置按脚本自身解析。

已安装的ARDY环境不会自动跟随源码变化。停止ARDY后端，使用**后端自己的Python**重新安装，不能使用NEKO宿主Python：

```powershell
$KitRoot = Read-Host '输入已完成安装的接入包目录'
& (Join-Path $KitRoot 'portable/install.ps1') -Reuse -Device xpu -ArdySource $ArdySource
```

NVIDIA用户把 `xpu` 改成 `cuda`。该操作重跑依赖检查与源码编译；保留本机模型。随后双击接入包的 `启动NPC动作.cmd`。旧接入包若仍自带旧补丁，应先换成从当前仓库生成的新包，避免旧安装器覆盖新版本。

## 校验、备份与回退

- 全部覆盖文件先校验，再写入源码；不匹配的本地修改会明确失败，不强制覆盖。
- `--check` 不修改源码；重复应用相同补丁返回 `pending=0`。
- 接受官方基线，以及清单中明确列出的旧补丁内容指纹；不接受任意修改。
- 原文件备份在ARDY源码的 `.ardy-integration-backups/时间戳/`，`added-files.json` 记录本次新增文件。
- 手工回退时先停止后端，恢复对应备份，并核对 `added-files.json` 后移除该次新增文件。已有后端环境仍需重新安装回退后的源码。不要混用不同批次备份。

XPU在本机验证；CUDA安装配置保留，但不代表已完成另一台电脑的硬件验收。补丁应用通过不等于模型生成或Unity动作质量通过。
