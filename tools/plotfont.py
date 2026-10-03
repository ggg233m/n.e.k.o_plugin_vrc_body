# -*- coding: utf-8 -*-
"""matplotlib 中文字体——省得每张图都画成口口。

    from tools.plotfont import use_cjk
    use_cjk()            # 之后 plt 的中文/减号都正常

## 为什么不用 ``rcParams['font.sans-serif'] = ['Microsoft YaHei']`` 就完事

那样设**在这台机器上不生效**：matplotlib 只认它 font_manager 缓存里注册过的字体，
而 ``.ttc``（msyh.ttc / simsun.ttc / msjh.ttc）这类字体集合 matplotlib 支持不稳，
设了名字仍然回退到默认的 DejaVu Sans，中文全是口方块。

所以这里直接用 ``font_manager.fontManager.addfont()`` 把**单个 TTF 文件**塞进去，
再取它真实的 family name。列表按优先级排：黑体（笔画粗，图上小字更清楚）优先。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 优先级从高到低。simhei 是纯 TTF，最稳；msyh.ttc 只作后备。
_CANDIDATES = (
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_done = False


def use_cjk() -> str | None:
    """注册一个能显示中文的字体并设为默认。返回字体名；一个都找不到时返回 None。

    找不到**不抛异常**——图还是能出，只是中文会是口方块，标题里会提示。
    """
    global _done
    if _done:
        return use_cjk.__dict__.get("name")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    for path in _CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
        except Exception:                                   # noqa: BLE001 - 字体坏了就换下一个
            continue
        rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        rcParams["font.family"] = "sans-serif"
        # 默认 DejaVu 没有 U+2212，中文一换字体负号就变口口；这个开关只在 DejaVu 下需要，
        # 换成中文字体后它是多余的，所以按实际字体定。
        rcParams["axes.unicode_minus"] = name.startswith("DejaVu")
        use_cjk.__dict__["name"] = name
        _done = True
        return name

    print("[plotfont] 没找到中文字体，图里的中文会是口方块。"
          f"试过：{', '.join(_CANDIDATES)}", file=sys.stderr)
    use_cjk.__dict__["name"] = None
    _done = True
    return None
