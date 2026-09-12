"""读取 PMX 的贴图引用，打包给浏览器，不修改原文件。"""
import io
import struct
import zipfile
from pathlib import Path

LIMIT = 256 * 1024 * 1024

def pack_pmx(filename):
    path = Path(filename.strip().strip('"')).resolve()
    if path.suffix.lower() != ".pmx" or not path.is_file():
        raise ValueError("请选择存在的 PMX 文件")
    if path.stat().st_size > LIMIT:
        raise ValueError("PMX 文件超过 256 MiB")
    data = path.read_bytes()
    stream = io.BytesIO(data)
    def read(n):
        if n < 0 or n > len(data): raise ValueError("PMX 数据长度无效")
        value = stream.read(n)
        if len(value) != n: raise ValueError("PMX 数据被截断")
        return value
    def integer(): return struct.unpack("<i", read(4))[0]
    if read(4) != b"PMX ": raise ValueError("不是 PMX 文件")
    version = struct.unpack("<f", read(4))[0]
    if not 1.99 < version < 2.11: raise ValueError("支持 PMX 2.0 / 2.1")
    header = read(read(1)[0])
    if len(header) < 8 or header[0] not in (0,1) or header[1] > 4 or any(x not in (1,2,4) for x in header[2:8]):
        raise ValueError("PMX 头部无效")
    def string(): return read(integer()).decode("utf-16-le" if header[0] == 0 else "utf-8")
    title = string()
    for _ in range(3): string()
    vertices = integer()
    if not 0 <= vertices <= 2000000: raise ValueError("PMX 顶点数超出范围")
    for _ in range(vertices):
        read(32 + header[1] * 16)
        deform = read(1)[0]
        sizes = {0:header[5], 1:header[5]*2+4, 2:header[5]*4+16, 3:header[5]*2+40, 4:header[5]*4+16}
        if deform not in sizes: raise ValueError("不支持的 PMX 蒙皮类型")
        read(sizes[deform]+4)
    read(integer() * header[2])
    count = integer()
    if not 0 <= count <= 1024: raise ValueError("PMX 贴图数量超出范围")
    names = [string().replace("\\", "/") for _ in range(count)]
    payload = {"model.pmx": data}
    total = len(data)
    missing = []
    for name in names:
        if not name: continue
        texture = (path.parent / name).resolve()
        if not texture.is_relative_to(path.parent): raise ValueError("贴图必须位于 PMX 所在目录或子目录：" + name)
        if not texture.is_file():
            missing.append(name)
            continue
        total += texture.stat().st_size
        if total > LIMIT: raise ValueError("模型与贴图总计超过 256 MiB")
        payload[name] = texture.read_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(output,"w",zipfile.ZIP_STORED) as archive:
        for name, blob in payload.items(): archive.writestr(name,blob)
    warning = "；缺失贴图使用占位：" + ", ".join(missing) if missing else ""
    return output.getvalue(), f"{title or path.stem}；{vertices} 顶点，{count} 张贴图" + warning
