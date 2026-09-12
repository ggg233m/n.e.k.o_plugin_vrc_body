"""长距离坐标保持旧包长，只有新世界能力允许发送滚动坐标。"""
import pytest
import _bootstrap
from yui_npc_controller.runtime.pose_codec import encode,decode,midi_events,rebind_pose_epoch
from yui_npc_controller.runtime.motion_space import MotionSpace


@pytest.mark.parametrize('x,expected',[(32.767,32.767),(32.768,-32.768),(-32.769,32.767),(65.536,0),(131.073,.001),(-131.073,-.001)])
def test_wrapped_positions_keep_millimeter_precision_without_extra_events(x,expected):
    packet=encode(1,2,1,0,[x,.9544,-x],[[0,0,0,1]]*27,[1]*4,wrap_root=True)
    decoded=decode(packet)
    assert len(packet)==133 and len(midi_events(packet))==78
    assert decoded['root_wrapped'] and decoded['root'][0]==pytest.approx(expected)
    assert decoded['root'][1]==.954
    rebound=decode(rebind_pose_epoch(packet,3))
    assert rebound['epoch']==3 and rebound['root']==decoded['root']


def test_legacy_encoder_still_rejects_out_of_window():
    with pytest.raises(ValueError,match='root_out_of_range'):
        encode(1,2,1,0,[33,.9544,0],[[0,0,0,1]]*27,[1]*4)


def test_long_verified_path_round_trip_with_explicit_new_window():
    space=MotionSpace(origin=[20,3,10],yaw=73,scale=.7841774,max_distance=10000)
    target=[220,3,-100]
    assert space.world_point(space.source_point(target))==pytest.approx(target)
    with pytest.raises(ValueError,match='path_out_of_range'):
        MotionSpace(origin=[20,3,10],yaw=73,scale=.7841774).source_point(target)
