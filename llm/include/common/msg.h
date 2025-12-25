#pragma once
#include "systemc.h"

#include "defs/enums.h"
#include "macros/macros.h"

// 以下为数据包相关
class Msg {
public:
    bool is_end_;          // 是否为最后一个包
    MSG_TYPE msg_type_;    // 消息类型
    int seq_id_;           // 包序号
    int des_;              // 目标id
    int offset_;           // 目标地址偏移
    int tag_id_;           // send & recv对应的tag编号
    int source_;           // 发送此msg的core id
    int length_;           // 真实数据的长度，避免end包覆盖
    bool refill_;          // 在end包中表示是否需要refill
    bool config_end_;      // 是否为一个原语config的最后一个包
    int roofline_packets_; // 视作发送X个数据包，加快模拟速度
    sc_bv<128> data_;

    Msg(bool e, MSG_TYPE m, int seq, int des, int offset, int tag, int length,
        sc_bv<128> d)
        : is_end_(e),
          msg_type_(m),
          seq_id_(seq),
          des_(des),
          offset_(offset),
          tag_id_(tag),
          length_(length),
          data_(d) {}
    Msg(bool e, MSG_TYPE m, int s, int des, sc_bv<128> d)
        : is_end_(e), msg_type_(m), seq_id_(s), des_(des), data_(d) {
        config_end_ = true;
    }

    Msg(bool e, MSG_TYPE m, int s, int des, bool conf_end, sc_bv<128> d)
        : is_end_(e),
          msg_type_(m),
          seq_id_(s),
          des_(des),
          config_end_(conf_end),
          data_(d) {} // 用于CONFIG中的计算原语

    Msg(MSG_TYPE m, int des, int tag, int source)
        : msg_type_(m),
          des_(des),
          tag_id_(tag),
          source_(source) {} // 用于REQ和ACK
    Msg(MSG_TYPE m, int des, int source)
        : msg_type_(m), des_(des), source_(source) {} // 用于DONE
    Msg() {
        seq_id_ = -1;
        des_ = -1;
        is_end_ = false;
        msg_type_ = CONFIG;
        data_ = sc_bv<128>(0x1);
    }

    bool operator<(const Msg &other) const { return other.seq_id_ < seq_id_; }

    //判断消息是否为控制信号 (ACK/REQ/DONE)
    bool IsControlMsg() const {
        return msg_type_ == ACK || msg_type_ == REQUEST || msg_type_ == DONE;
    }
};
