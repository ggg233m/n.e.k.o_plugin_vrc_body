@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PYTHONUTF8=1
set TMP=H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.tc_cache\tmp
set TEMP=%TMP%
set TORCHINDUCTOR_CACHE_DIR=H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.tc_cache\ind_cpp4
set TRITON_CACHE_DIR=H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.tc_cache\tri_cpp4
set TORCHINDUCTOR_COMPILE_THREADS=1
set CC=clang-cl
set CXX=clang-cl
set PATH=D:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\compiler;D:\Program Files (x86)\Intel\oneAPI\2025.3\bin;%PATH%
"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.venv_xpu\Scripts\python.exe" %*
