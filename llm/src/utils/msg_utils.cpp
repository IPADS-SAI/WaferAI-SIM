#include "utils/msg_utils.h"
#include "defs/global.h"
#include "defs/spec.h"

sc_bv<256> SerializeMsg(Msg msg) {
    sc_bv<256> serialized_msg;

    int pos = 0;

    serialized_msg.range(pos + M_D_IS_END - 1, pos) =
        sc_bv<M_D_IS_END>(msg.is_end_);
    pos += M_D_IS_END;
    serialized_msg.range(pos + M_D_MSG_TYPE - 1, pos) =
        sc_bv<M_D_MSG_TYPE>(int(msg.msg_type_));
    pos += M_D_MSG_TYPE;
    serialized_msg.range(pos + M_D_SEQ_ID - 1, pos) =
        sc_bv<M_D_SEQ_ID>(msg.seq_id_);
    pos += M_D_SEQ_ID;
    serialized_msg.range(pos + M_D_DES - 1, pos) = sc_bv<M_D_DES>(msg.des_);
    pos += M_D_DES;
    serialized_msg.range(pos + M_D_OFFSET - 1, pos) =
        sc_bv<M_D_OFFSET>(msg.offset_);
    pos += M_D_OFFSET;
    serialized_msg.range(pos + M_D_TAG_ID - 1, pos) =
        sc_bv<M_D_TAG_ID>(msg.tag_id_);
    pos += M_D_TAG_ID;
    serialized_msg.range(pos + M_D_SOURCE - 1, pos) =
        sc_bv<M_D_SOURCE>(msg.source_);
    pos += M_D_SOURCE;
    serialized_msg.range(pos + M_D_LENGTH - 1, pos) =
        sc_bv<M_D_LENGTH>(msg.length_);
    pos += M_D_LENGTH;
    serialized_msg.range(pos + M_D_REFILL - 1, pos) =
        sc_bv<M_D_REFILL>(msg.refill_);
    pos += M_D_REFILL;
    serialized_msg.range(pos + M_D_ROOFLINE - 1, pos) =
        sc_bv<M_D_ROOFLINE>(msg.roofline_packets_);
    pos += M_D_ROOFLINE;
    serialized_msg.range(pos + M_D_CONF_END - 1, pos) = msg.config_end_;
    pos += M_D_CONF_END;
    serialized_msg.range(pos + M_D_DATA - 1, pos) = msg.data_;
    pos += M_D_DATA;
    serialized_msg.range(255, pos) = sc_bv<32>(0);

    return serialized_msg;
}

Msg DeserializeMsg(sc_bv<256> buffer) {
    Msg msg;
    int pos = 0;

    msg.is_end_ = buffer.range(pos + M_D_IS_END - 1, pos).to_uint64(),
    pos += M_D_IS_END;
    msg.msg_type_ =
        MSG_TYPE(buffer.range(pos + M_D_MSG_TYPE - 1, pos).to_uint64()),
    pos += M_D_MSG_TYPE;
    msg.seq_id_ = buffer.range(pos + M_D_SEQ_ID - 1, pos).to_uint64(),
    pos += M_D_SEQ_ID;
    msg.des_ = buffer.range(pos + M_D_DES - 1, pos).to_uint64(), pos += M_D_DES;
    msg.offset_ = buffer.range(pos + M_D_OFFSET - 1, pos).to_uint64(),
    pos += M_D_OFFSET;
    msg.tag_id_ = buffer.range(pos + M_D_TAG_ID - 1, pos).to_uint64(),
    pos += M_D_TAG_ID;
    msg.source_ = buffer.range(pos + M_D_SOURCE - 1, pos).to_uint64(),
    pos += M_D_SOURCE;
    msg.length_ = buffer.range(pos + M_D_LENGTH - 1, pos).to_uint64(),
    pos += M_D_LENGTH;
    msg.refill_ = buffer.range(pos + M_D_REFILL - 1, pos).to_uint64(),
    pos += M_D_REFILL;
    msg.roofline_packets_ =
        buffer.range(pos + M_D_ROOFLINE - 1, pos).to_uint64(),
    pos += M_D_ROOFLINE;
    msg.config_end_ = buffer.range(pos + M_D_CONF_END - 1, pos).to_uint64(),
    pos += M_D_CONF_END;
    msg.data_ = buffer.range(pos + M_D_DATA - 1, pos);

    return msg;
}

void CalculatePacketNum(int output_size, int weight, int data_byte,
                        int &packet_num, int &end_length) {
    int slice_size = (output_size % weight) ? (output_size / weight + 1)
                                            : (output_size / weight);

    int slice_size_in_bit = slice_size * data_byte * 8;
    packet_num = (slice_size_in_bit % M_D_DATA)
                     ? (slice_size_in_bit / M_D_DATA + 1)
                     : (slice_size_in_bit / M_D_DATA);
    end_length = slice_size_in_bit - (packet_num - 1) * M_D_DATA;

    packet_num = packet_num % HW_NOC_PAYLOAD_PER_CYCLE
                     ? packet_num / HW_NOC_PAYLOAD_PER_CYCLE + 1
                     : packet_num / HW_NOC_PAYLOAD_PER_CYCLE;
}

bool IsBlockableMsgType(MSG_TYPE type) {
    switch (type) {
    case MSG_TYPE::DATA:
        return true;
    default:
        return false;
    }
}