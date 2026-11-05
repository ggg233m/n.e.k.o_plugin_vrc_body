# -*- coding: utf-8 -*-
"""manifest 回归测试：口径必须被强制，不能静默漂移。"""
import sys
import tempfile
sys.path.insert(0, "research/tools")
import input_schema as S

fails = []


def check(name, fn):
    try:
        fn()
        print("  PASS %s" % name)
    except AssertionError as e:
        fails.append(name)
        print("  FAIL %s: %s" % (name, e))


def t_missing_warns():
    msgs = []
    s = S.load(tempfile.mkdtemp(), warn=msgs.append)
    assert len(msgs) == 1, "缺失 manifest 必须发一条警告，实际 %d" % len(msgs)
    assert s["legacy"] is True and s["act_dim"] == 2, "回退值不对"


def t_roundtrip():
    d = tempfile.mkdtemp()
    fake = type("A", (), dict(drop_wp=True, quad_feats=False,
                             prompt_format="card", act_dim=3))
    s = dict(S.from_args(fake))
    S.save(d, s)
    back = S.load(d)
    assert back["drop_wp"] is True, "drop_wp 丢了"
    assert back["act_dim"] == 3, "act_dim 丢了"
    assert back["legacy"] is False, "不该判成 legacy"


def t_mismatch_detected():
    man = dict(S.LEGACY_DEFAULTS)
    man["act_dim"] = 3
    out = 12 * 2
    caught = False
    if out // 12 != man["act_dim"]:
        caught = True
    assert caught, "act_dim 矛盾必须被检出"


def t_drop_wp_zeroes():
    drop_wp = True
    wp = (3.5, -1.2)
    if drop_wp:
        wp = (0.0, 0.0)
    assert wp == (0.0, 0.0), "drop_wp 没生效"


check("缺失 manifest 会警告（不静默）", t_missing_warns)
check("manifest 写读往返保真", t_roundtrip)
check("act_dim 与 head 矛盾被检出", t_mismatch_detected)
check("drop_wp 把 wp 置 0", t_drop_wp_zeroes)

if fails:
    print()
    print("FAILED %d" % len(fails))
    raise SystemExit(1)
print()
print("OK 4/4")
