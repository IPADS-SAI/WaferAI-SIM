#pragma once
#include "systemc.h"
#include <iostream>
#include <queue>

#include "common/msg.h"
#include "defs/const.h"
#include "defs/global.h"
#include "macros/macros.h"
#include "trace/Event_engine.h"
#include "utils/memory_utils.h"
#include "utils/msg_utils.h"
#include "utils/router_utils.h"
#include "utils/system_utils.h"

using namespace std;

class RouterUnit;

class RouterMonitor : public sc_module {
public:
    RouterUnit **routers;

    Event_engine *event_engine;

    SC_HAS_PROCESS(RouterMonitor);
    RouterMonitor(const sc_module_name &n, Event_engine *event_engine);
    ~RouterMonitor();
};

class RouterUnit : public sc_module {
public:
    int rid;

    // 从core发送过来：用于控制core_is_ready（数据信道）
    sc_in<bool> core_busy_i;
    // 从core发送过来：用于控制控制信道的core_is_ready
    sc_in<bool> ctrl_core_busy_i;

    // 传递数据的真正信道
    sc_out<sc_bv<256>> channel_o[DIRECTIONS];
    sc_in<sc_bv<256>> channel_i[DIRECTIONS];

    // 输入，输出缓存区
    queue<sc_bv<256>> buffer_i[DIRECTIONS];
    queue<sc_bv<256>> buffer_o[DIRECTIONS];

    // 通道未满的握手信号，只有收到该信号为true，才可向目标发送数据，input信号缺少的一个由core_is_ready担任
    sc_out<bool> channel_avail_o[DIRECTIONS];
    sc_in<bool> channel_avail_i[DIRECTIONS - 1];

    // 通道使能信号，router接收到之后才可查看channel内容
    sc_out<bool> data_sent_o[DIRECTIONS];
    sc_in<bool> data_sent_i[DIRECTIONS];

    /* -------------------Handshake--------------------- */
    // 在发送方握手信号发送的时候，不建立单独通路，而是在ack
    // return包返回的时候建立单独通路，此时将会lock一个input channel和output
    // channel， 并标记对应的channel_avail为其ack
    // tag(由config文件决定)，channel_avail从bool改为int，如果为0,则表示没有上锁且可发送，如果为-1，表示通道已满
    // 如果为大于0的数，则表示已上锁，数字等于正在连接的通路tag编号。router在发送包的时候，如果该包的tag等于上锁通道的tag，则可以发送，否则不予发送。
    // 在发送当前通路上的end包的时候，当前router上锁的通道自动解除。
    // 现在有一个问题，如果有两个ack返回包在相邻时间返回，其中一个可能会被另一个阻挡，导致部分router会被无意义锁住。
    int input_lock[5];
    int input_lock_ref[5];
    int output_lock[5];
    int output_lock_ref[5];

    /* -----------------Host-Interface------------------ */
    // 只有位置最靠边缘的router才会注册这些端口
    sc_in<bool> *host_data_sent_i;
    sc_out<bool> *host_data_sent_o;

    sc_in<sc_bv<256>> *host_channel_i;
    sc_out<sc_bv<256>> *host_channel_o;

    queue<sc_bv<256>> *host_buffer_i;
    queue<sc_bv<256>> *host_buffer_o;        // 数据信道：只存数据包
    queue<sc_bv<256>> *host_ctrl_buffer_o;   // 控制信道：只存控制包

    sc_out<bool> *host_channel_avail_o;
    /* ------------------------------------------------- */

    /* ---------------Control Channel------------------- */
    // 控制信道 - 用于传输 ACK/REQ/DONE 信号
    sc_out<sc_bv<256>> ctrl_channel_o[DIRECTIONS];
    sc_in<sc_bv<256>> ctrl_channel_i[DIRECTIONS];

    // 控制信道缓冲区
    queue<sc_bv<256>> ctrl_buffer_i[DIRECTIONS];
    queue<sc_bv<256>> ctrl_buffer_o[DIRECTIONS];

    // 控制信道空闲信号
    sc_out<bool> ctrl_channel_avail_o[DIRECTIONS];
    sc_in<bool> ctrl_channel_avail_i[DIRECTIONS - 1];

    // 控制信道发送使能信号
    sc_out<bool> ctrl_sent_o[DIRECTIONS];
    sc_in<bool> ctrl_sent_i[DIRECTIONS];

    // Host 控制信道接口 (仅边缘 router)
    sc_out<bool> *host_ctrl_sent_o;
    sc_out<sc_bv<256>> *host_ctrl_channel_o;
    /* ------------------------------------------------- */

    // 触发execute函数的信号
    sc_event need_next_trigger;

    Event_engine *event_engine;

    SC_HAS_PROCESS(RouterUnit);
    RouterUnit(const sc_module_name &n, int s_rid, Event_engine *event_engine);
    ~RouterUnit();

    void move_data();

    void router_execute();
    void monitor_core_status();
    void trans_next_trigger();

    void end_of_elaboration();
};