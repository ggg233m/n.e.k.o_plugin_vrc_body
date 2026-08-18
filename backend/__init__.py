"""AnyaDance 身体后端的独立运行时边界。

本包刻意不导入 N.E.K.O 插件 SDK。宿主项目可以复制此目录，为自身的身体与
传输模块提供适配器，然后将 :mod:`backend.process` 作为本机独立进程运行。
"""

__all__: list[str] = []
