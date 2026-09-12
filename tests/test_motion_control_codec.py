"""冻结准备/武装控制包字段，防止和姿态或结束包混用。"""
import binascii
import struct
import pytest
import _bootstrap
from yui_npc_controller.runtime.pose_codec import encode_motion_control,midi_events


def test_prepare_and_begin_share_bounded_credit_packet():
    for version,sequence,anchor in [(4,51,3),(5,1,None)]:
        packet=encode_motion_control(version,7,19,sequence,"a"*32,anchor_id=anchor)
        assert len(packet)==133 and len(midi_events(packet))==78
        assert packet[:3]==b"AP"+bytes([version])
        assert struct.unpack_from('<iii',packet,3)==(7,19,sequence)
        assert packet[23:55]==b'a'*32
        assert packet[55:131]==bytes(76)
        assert struct.unpack_from('<H',packet,131)[0]==binascii.crc_hqx(packet[:131],0xffff)
        assert packet[15:17]==(bytes([1,3]) if anchor is not None else bytes(2))


@pytest.mark.parametrize("version,sequence,op_id,anchor",[(3,1,"a"*32,None),(5,2,"a"*32,None),(4,1,"g"*32,1),(4,1,"a"*32,True),(5,1,"a"*32,0)])
def test_invalid_control_requests_are_not_encoded(version,sequence,op_id,anchor):
    with pytest.raises(ValueError): encode_motion_control(version,7,19,sequence,op_id,anchor_id=anchor)
