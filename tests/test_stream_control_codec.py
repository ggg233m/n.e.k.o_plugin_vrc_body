"""持续控制帧的模式、身份与有界MIDI事务契约。"""
import binascii
import struct
import pytest
import _bootstrap
from yui_npc_controller.runtime.pose_codec import encode_stream_control,midi_events


def test_stream_intent_has_independent_operation_version_and_plan():
    packet=encode_stream_control(8,9,3,12,'a'*32,first_sequence=11,intent_version=7,
                                 end_sequence=50,mode=2,plan_request=10)
    assert len(packet)==133 and len(midi_events(packet))==78
    assert packet[15]==2 and packet[16]==127
    assert struct.unpack_from('<I',packet,17)==(11,)
    assert packet[23:55]==b'a'*32
    assert struct.unpack_from('<III',packet,55)==(7,50,10)
    assert not any(packet[67:131])
    assert struct.unpack_from('<H',packet,131)[0]==binascii.crc_hqx(packet[:131],0xffff)


@pytest.mark.parametrize('values',[
    dict(first_sequence=0,intent_version=1),
    dict(first_sequence=1,intent_version=0),
    dict(first_sequence=5,intent_version=1,end_sequence=4),
    dict(first_sequence=1,intent_version=1,mode=2),
    dict(first_sequence=1,intent_version=1,mode=True),
    dict(first_sequence=1,intent_version=1,plan_request=-1),
])
def test_invalid_or_unverified_stream_boundaries_rejected(values):
    with pytest.raises(ValueError):encode_stream_control(8,9,3,12,'a'*32,**values)
