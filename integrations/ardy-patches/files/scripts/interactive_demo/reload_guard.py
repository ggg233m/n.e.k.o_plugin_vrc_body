"""重新加载模型时串行化会话生成，避免旧任务写回新会话。"""
from contextlib import contextmanager


@contextmanager
def pause_for_model_reload(session):
    playing = session.playing
    session.playing = False
    try:
        # 等待正在执行的生成完成；持锁覆盖模型替换、文本更新与首段生成。
        with session.replan_lock:
            yield
    finally:
        session.playing = playing
