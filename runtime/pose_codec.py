"""隔离姿态包：Core27 四元数 smallest-three，CRC 与 14 位 MIDI NoteOff 装载。"""
import binascii
import math
import struct

CHANNEL = 15
PACKET_BYTES = 133


def pack_quaternion(q):
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError('invalid_quaternion')
    length = math.sqrt(sum(x*x for x in q))
    if length < 1e-8:
        raise ValueError('zero_quaternion')
    q = [x/length for x in q]
    largest = max(range(4), key=lambda i: abs(q[i]))
    if q[largest] < 0:
        q = [-x for x in q]
    value, shift = largest, 2
    limit = 1 / math.sqrt(2)
    for i in range(4):
        if i == largest:
            continue
        code = round((q[i] + limit) * 1023 / (2 * limit))
        value |= min(1023, max(0, code)) << shift
        shift += 10
    return value


def unpack_quaternion(value):
    largest, shift = value & 3, 2
    q = [0.] * 4
    limit = 1 / math.sqrt(2)
    for i in range(4):
        if i == largest:
            continue
        q[i] = ((value >> shift) & 1023) / 1023 * (2 * limit) - limit
        shift += 10
    q[largest] = math.sqrt(max(0., 1 - sum(x*x for x in q)))
    return q


def encode(session, epoch, sequence, frame_ms, root, rotations, contacts=None, *, wrap_root=False):
    if len(rotations) != 27 or len(root) != 3:
        raise ValueError('expected_core27')
    for value in (session, epoch, sequence):
        if type(value) is not int or not 1 <= value <= 0x7fffffff:
            raise ValueError('invalid_identity')
    if type(frame_ms) is not int or not 0 <= frame_ms <= 65535:
        raise ValueError('invalid_frame_time')
    if not all(math.isfinite(x) and (-1000000<=x<=1000000 if wrap_root and i!=1 else -32.768<=x<=32.767) for i,x in enumerate(root)):
        raise ValueError('root_out_of_range')
    version = 1
    if contacts is not None:
        if len(contacts) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in contacts):
            raise ValueError('invalid_contacts')
        # v2 高四位依次为左踝、左趾、右踝、右趾；保持 133 字节/78 个事件。
        version = 2 | sum((int(x >= .5) << (i+4)) for i, x in enumerate(contacts))
    if wrap_root:
        if contacts is None:raise ValueError('wrapped_pose_requires_contacts')
        version=(version&240)|12
    millimeters=[round(x*1000) for x in root]
    if wrap_root:
        for i in (0,2):millimeters[i]=(millimeters[i]+32768)%65536-32768
    data = struct.pack('<2sBIIIH3h', b'AP', version, session, epoch, sequence, frame_ms,
                       *millimeters)
    data += b''.join(struct.pack('<I', pack_quaternion(q)) for q in rotations)
    return data + struct.pack('<H', binascii.crc_hqx(data, 0xffff))


def decode(data):
    if len(data) != PACKET_BYTES or data[:2] != b'AP' or not (data[2] == 1 or data[2] & 15 in (2,12)):
        raise ValueError('invalid_packet')
    if binascii.crc_hqx(data[:-2], 0xffff) != struct.unpack('<H', data[-2:])[0]:
        raise ValueError('crc_mismatch')
    session, epoch, sequence, frame_ms, x, y, z = struct.unpack_from('<IIIH3h', data, 3)
    result=dict(session=session, epoch=epoch, sequence=sequence, frame_ms=frame_ms,
                contacts=None if data[2] == 1 else [(data[2] >> (i+4)) & 1 for i in range(4)],
                root=[x/1000, y/1000, z/1000],
                rotations=[unpack_quaternion(struct.unpack_from('<I', data, 23+i*4)[0]) for i in range(27)])
    if data[2]&15==12:result['root_wrapped']=True
    return result


def rebind_pose_epoch(data, epoch):
    """对本地已编码姿态仅重绑定世界代次；不重复计算27个关节矩阵。"""
    if (len(data)!=PACKET_BYTES or data[:2]!=b'AP' or not (data[2]==1 or data[2]&15 in (2,12))
        or type(epoch) is not int or not 1<=epoch<=0x7fffffff):
        raise ValueError('invalid_pose_rebind')
    if binascii.crc_hqx(data[:-2],0xffff)!=struct.unpack('<H',data[-2:])[0]:
        raise ValueError('crc_mismatch')
    result=data[:7]+struct.pack('<I',epoch)+data[11:-2]
    return result+struct.pack('<H',binascii.crc_hqx(result,0xffff))


def encode_finish(session, epoch, sequence):
    data=bytearray(encode(session,epoch,sequence,1,[0,0,0],[[0,0,0,1]]*27))
    data[2]=3
    data[-2:]=struct.pack('<H',binascii.crc_hqx(data[:-2],0xffff))
    return bytes(data)


def encode_motion_control(version,session,epoch,sequence,op_id,*,anchor_id=None):
    if version not in (4,5) or not isinstance(op_id,str) or len(op_id)!=32 or any(c not in '0123456789abcdef' for c in op_id):
        raise ValueError('invalid_motion_control')
    if anchor_id is not None and (version!=4 or type(anchor_id) is not int or not 0<=anchor_id<=126):
        raise ValueError('invalid_motion_anchor')
    if version==5 and sequence!=1:
        raise ValueError('invalid_begin_sequence')
    data=bytearray(encode(session,epoch,sequence,0,[0,0,0],[[0,0,0,1]]*27))
    data[2]=version
    data[15:131]=bytes(116)
    if anchor_id is not None:
        data[15]=1;data[16]=anchor_id
    data[23:55]=op_id.encode('ascii')
    data[-2:]=struct.pack('<H',binascii.crc_hqx(data[:-2],0xffff))
    return bytes(data)


def midi_events(data):
    if len(data) != PACKET_BYTES:
        raise ValueError('invalid_packet_length')
    # 不使用 NoteOn，避免被旧路由器的跨通道 ESTOP 音符解释为急停。
    events = [('cc', CHANNEL, 110, 1)]
    value, bits = 0, 0
    for byte in data:
        value |= byte << bits
        bits += 8
        while bits >= 14:
            word = value & 16383
            events.append(('note_off', CHANNEL, word >> 7, word & 127))
            value >>= 14
            bits -= 14
    if bits:
        events.append(('note_off', CHANNEL, value >> 7, value & 127))
    events.append(('cc', CHANNEL, 111, 1))
    return events


def encode_stream_control(version,session,epoch,request,op_id,*,first_sequence=0,intent_version=0,end_sequence=0,anchor_id=None,mode=0,plan_request=0):
    """持续会话控制帧仍为133字节，不扩大MIDI事务或滚动预算。"""
    if version not in (6,7,8,9,10) or not isinstance(op_id,str) or len(op_id)!=32 or any(c not in '0123456789abcdef' for c in op_id):
        raise ValueError('invalid_stream_control')
    for value in (first_sequence,intent_version,end_sequence,plan_request):
        if type(value) is not int or not 0<=value<=0x7fffffff:raise ValueError('invalid_stream_boundary')
    if version==8 and (first_sequence<1 or intent_version<1 or (end_sequence and end_sequence<first_sequence)):
        raise ValueError('invalid_intent_boundary')
    if anchor_id is not None and (version!=10 or type(anchor_id) is not int or not 0<=anchor_id<=126):
        raise ValueError('invalid_stream_anchor')
    if type(mode) is not int or not 0<=mode<=4 or (version!=8 and (mode or plan_request)):
        raise ValueError('invalid_stream_mode')
    if version==8 and mode in (2,3) and plan_request<=0:
        raise ValueError('stream_path_receipt_required')
    data=bytearray(encode(session,epoch,request,0,[0,0,0],[[0,0,0,1]]*27))
    data[2]=version;data[15:131]=bytes(116);data[15]=mode
    data[16]=127 if anchor_id is None else anchor_id
    struct.pack_into('<I',data,17,first_sequence)
    data[23:55]=op_id.encode('ascii')
    struct.pack_into('<III',data,55,intent_version,end_sequence,plan_request)
    data[-2:]=struct.pack('<H',binascii.crc_hqx(data[:-2],0xffff))
    return bytes(data)
