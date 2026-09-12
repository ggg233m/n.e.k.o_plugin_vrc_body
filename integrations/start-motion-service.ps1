param(
    [string]$KitRoot,
    [switch]$Smoke,
    [switch]$Check,
    [ValidateSet('auto','cuda','xpu')][string]$Device,
    [ValidateSet('Core40','Core8','Core40Eager','Core8Eager','Eager')][string]$Profile
)
$ErrorActionPreference='Stop'
# 插件位于接入包的 neko-plugin 下时自动发现；独立源码仓库显式指定接入包。
if(-not $KitRoot){$KitRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)}
$launcher=Join-Path $KitRoot 'start-npc-motion.ps1'
if(-not(Test-Path -LiteralPath $launcher -PathType Leaf)){
    throw '未找到接入包。请直接双击接入包中的 启动NPC动作.cmd；开发调试可用 -KitRoot 指定接入包目录。'
}
$options=@{}
if($Smoke){$options.Smoke=$true}
if($Check){$options.Check=$true}
if($Device){$options.Device=$Device}
if($Profile){$options.Profile=if($Profile -eq 'Eager'){'Core8Eager'}else{$Profile}}
& $launcher @options
exit $LASTEXITCODE
