"""校验三个Udon接收器的固定CRC表与线上CRC16完全一致。"""
import binascii
import random
import re
from pathlib import Path
import pytest


@pytest.mark.parametrize('name',['NekoArdyPoseLab','NekoArdyWorldBridge','NekoArdyStreamBridge'])
def test_nibble_table_matches_wire_crc_and_detects_single_bit_errors(name):
    source=(Path(__file__).parents[1]/'unity/Assets/NEKO/ArdyLab'/f'{name}.cs').read_text(encoding='utf-8-sig')
    table=[int(v,16) for v in re.search(r'_crcNibbles=\{([^}]+)',source).group(1).split(',')]
    assert len(table)==16
    def crc(data):
        value=65535
        for byte in data:
            value^=byte<<8
            for _ in range(2):value=((value<<4)^table[(value>>12)&15])&65535
        return value
    rng=random.Random(20260912)
    for data in [b'123456789',bytes(131),bytes([255])*131]+[rng.randbytes(131) for _ in range(128)]:
        assert crc(data)==binascii.crc_hqx(data,65535)
    data=rng.randbytes(131);expected=crc(data)
    for bit in range(len(data)*8):
        changed=bytearray(data);changed[bit//8]^=1<<(bit%8)
        assert crc(changed)!=expected
