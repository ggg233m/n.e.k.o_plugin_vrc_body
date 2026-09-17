@echo off
setlocal enabledelayedexpansion

:: 一键打包脚本 - 使用 N.E.K.O 宿主的 Python 环境构建插件
:: 用法: build.bat

set "HOST_ROOT=H:\AI\neko-music\N.E.K.O"
set "PYTHON_EXE=%HOST_ROOT%\.venv\Scripts\python.exe"
set "BUILD_SCRIPT=packaging\build_neko.py"

if not exist "%PYTHON_EXE%" (
    echo [错误] 找不到 N.E.K.O 的 Python 环境: %PYTHON_EXE%
    echo 请检查宿主路径是否正确
    exit /b 1
)

if not exist "%BUILD_SCRIPT%" (
    echo [错误] 找不到构建脚本: %BUILD_SCRIPT%
    echo 请在插件根目录下运行此脚本
    exit /b 1
)

echo.
echo ========================================
echo N.E.K.O 插件一键打包
echo ========================================
echo 宿主路径: %HOST_ROOT%
echo Python:    %PYTHON_EXE%
echo.

"%PYTHON_EXE%" "%BUILD_SCRIPT%" --host "%HOST_ROOT%"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo 构建成功
    echo ========================================
    echo 插件包位置: dist\
    dir /B dist\*.neko-plugin 2>nul
) else (
    echo.
    echo ========================================
    echo 构建失败 (退出码: %ERRORLEVEL%)
    echo ========================================
    exit /b %ERRORLEVEL%
)
