#pragma once
#include "systemc.h"

#include "common/memory.h"
#include "common/pd.h"
#include "prims/base.h"

class Clear_sram : public PrimBase {
public:
    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void printSelf();

    Clear_sram() { name = "Clear_sram"; }
};


class Load_prim : public PrimBase {
public:
    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void printSelf();

    Load_prim() { name = "Load_prim"; }
};


class Recv_prim : public PrimBase {
public:
    RECV_TYPE type;
    int tag_id;   // 和send原语对应的tag
    int recv_cnt; // 需要接收到的end包数量（用于多发一）

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void printSelf();

    Recv_prim() { name = "Recv_prim"; }
    Recv_prim(RECV_TYPE type) : type(type) { name = "Recv_prim"; }
    Recv_prim(RECV_TYPE type, int tag, int cnt)
        : type(type), tag_id(tag), recv_cnt(cnt) {
        name = "Recv_prim";
    }
};


class Send_prim : public PrimBase {
public:
    SEND_TYPE type;
    int des_id;                        // 目标id
    string output_label = UNSET_LABEL; // 需要从哪一个数据块标签获取结果，并发送
    int max_packet;                    // 需要发送的包裹数量
    int tag_id;                        // send_tag，用于与recv原语对应
    int end_length;                    // 尾包长度，避免覆盖

    int data_packet_id; // 已经发送的包裹数量

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);

    void printSelf();

    Send_prim() { name = "Send_prim"; }
    Send_prim(SEND_TYPE type) : type(type) { name = "Send_prim"; }
    Send_prim(SEND_TYPE type, int des, int tag)
        : type(type), des_id(des), tag_id(tag) {
        name = "Send_prim";
    } // 用于SEND_ACK
    Send_prim(SEND_TYPE type, int des, int max_packet, int tag)
        : des_id(des), type(type), max_packet(max_packet), tag_id(tag) {
        name = "Send_prim";
    }
};


class Set_addr : public PrimBase {
public:
    AddrDatapassLabel datapass_label;

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);

    void printSelf();
    Set_addr() { name = "Set_addr"; }
};

class Set_batch : public PrimBase {
public:
    vector<Stage> batch_info;
    int auto_pd;

    int taskCoreDefault(TaskCoreContext &context);

    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);

    void printSelf();

    Set_batch() {
        name = "Set_batch";
        auto_pd = 0;
    }

    Set_batch(vector<Stage> batchInfo) {
        name = "Set_batch";
        this->batch_info = batchInfo;
        auto_pd = 0;
    }

    Set_batch(vector<Stage> batchInfo, int auto_pd) {
        name = "Set_batch";
        this->batch_info = batchInfo;
        this->auto_pd = auto_pd;
    }
};

class Store_prim : public PrimBase {
public:
    int dram_addr;
    int sram_addr;
    int size;

    int taskCoreDefault(TaskCoreContext &context);

    void deserialize(vector<sc_bv<128>> buffer);
    vector<sc_bv<128>> serialize();

    void printSelf();

    Store_prim() { name = "Store_prim"; }
};