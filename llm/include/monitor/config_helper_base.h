#pragma once
#include "systemc.h"

#include "common/config.h"
#include "common/msg.h"

using namespace std;

class config_helper_base {
public:
    int end_cores;

    int g_recv_done_cnt; // 累计接收到的done信号数量，当到达指定数量，将会清零
    int g_recv_ack_cnt;  // 累计接收到的ack信号数量，当达到指定数量，将会清零

    // 在mem-interface中，当recv_helper读到消息的时候，将读到的消息放入这两个数组中，随后通知对应的处理函数。
    vector<Msg> g_temp_done_msg;
    vector<Msg> g_temp_ack_msg;

    vector<pair<int, int>> source_info; // 记录计算图开始需要推入才能触发的data
    vector<CoreConfig> coreconfigs;     // 记录所有核的工作配置，包括所有原语

    int pipeline; // 是否进行input连续输入，从而增加pipeline并行度，数值大小为需要连续进行的pipe段数
    int end_count_sources; // 此变量协助记录汇节点个数（done接收次数），参考source中的is_end字段

    virtual void fill_queue_config(queue<Msg> *q) = 0; // 下发原语配置
    virtual void fill_queue_start(queue<Msg> *q) = 0;  // 下发初始数据
    void fill_queue_data(queue<Msg> *q);               // 下发权重数据

    bool judge_is_end_core(int i);
    bool judge_is_end_work(CoreJob work);

    virtual void parse_ack_msg(Event_engine *event_engine, int flow_id,
                               sc_event *notify_event) = 0;
    virtual void parse_done_msg(Event_engine *event_engine,
                                sc_event *notify_event) = 0;

    string name() { return "Config helper"; }

    virtual void generate_prims(int i) = 0;

    virtual void printSelf() = 0;
    virtual config_helper_base *clone() const = 0;
    virtual ~config_helper_base() = default;
};