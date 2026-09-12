"""有限大小的 CPU 显示快照，角色和相机共享；不改动作历史存储。"""
FIELDS = ('joints_pos', 'joints_rot', 'foot_contacts', 'root_velocities')

def _key(session):
    result = []
    for name in FIELDS:
        value = getattr(session, name, None)
        try:
            version = value._version if value is not None else None
        except RuntimeError:
            version = None
        result.append((id(value), version))
    return tuple(result)

def publish_display_cache_locked(session, start):
    """调用方持有 motion_tensor_lock，单次最多回传 64 帧。"""
    start = max(0, min(int(start), session.joints_pos.shape[1] - 1))
    end = min(start + 64, session.joints_pos.shape[1])
    data = {}
    for name in FIELDS:
        value = getattr(session, name, None)
        data[name] = None if value is None else value[:, start:end].detach().to('cpu', copy=True)
    session._display_frame_cache = (_key(session), start, end, data)
    session._display_cache_copies = getattr(session, '_display_cache_copies', 0) + 1

def display_frame(session, frame):
    with session.motion_tensor_lock:
        source = {name: None if getattr(session, name, None) is None else getattr(session, name)[:, frame] for name in FIELDS}
        if not getattr(session, '_display_cache_enabled', True):
            return dict(source, _source=source)
        cached = getattr(session, '_display_frame_cache', None)
        if cached is None or cached[0] != _key(session) or not cached[1] <= frame < cached[2]:
            publish_display_cache_locked(session, frame)
            cached = session._display_frame_cache
        offset = frame - cached[1]
        return dict({name: None if value is None else value[:, offset] for name, value in cached[3].items()}, _source=source)
