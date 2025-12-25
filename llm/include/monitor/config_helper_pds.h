#pragma once
#include "systemc.h"
#include <vector>

#include "common/pd.h"
#include "monitor/config_helper_base.h"

using namespace std;

class config_helper_pds : public config_helper_base {
public:
    json json_template_p, json_template_d;
    vector<CoreStatus> coreStatus;
    vector<RequestRecord> requestRecords;
    int decode_done;                // 收到decode的eof完成次数
    vector<Msg> temp_config;        // 存放所有还没有发出去的config
    vector<queue<int>> idle_decode; // 由于超过credit而需要被stall的decode
    queue<int> req_decode; // 做完prefill之后，等待进行decode的请求

    vector<vector<double>>
        token_record; // 记录每一个req的每一个token的处理完毕时间

    bool busy_p; // 此次iteration是否已经开始
    bool busy_d;
    bool wait_send_start_prefill;
    bool wait_send_start_decode;
    bool need_trigger_send_start; // 是否需要触发start data的发送
    bool wait_schedule_p;
    bool wait_schedule_d;
    int g_recv_ack_cnt_p;
    int g_recv_ack_cnt_d;
    int g_recv_done_cnt_p;
    int g_recv_done_cnt_d;
    vector<int> arrival_time; // 记录所有req到达的时间
    vector<Msg> g_done_msg_p; // 收集
    vector<Msg> g_done_msg_d; // 收集

    // 模型配置
    int heads;
    double eof_chance;
    int prefill_stage, decode_stage;
    int prefill_core, decode_core;
    vector<int> stage_index;
    int batch_size;
    int head_size;
    int kv_heads;
    int prefill_iters;
    int hidden_size;
    int intermediate_size;

    int tp_size;

    config_helper_pds(string filename, sc_event *ev_sig,
                      int config_chip_id = 0);

    config_helper_pds *clone() const override {
        return new config_helper_pds(*this);
    }

    void generate_prims(int i) {}
    void generate_prims(int i, vector<Msg> &temp_buffer);

    void printSelf();

    void random_core();

    void fill_queue_start(queue<Msg> *q);
    void fill_queue_config(queue<Msg> *q);

    void parse_ack_msg(Event_engine *event_engine, int flow_id,
                       sc_event *notify_event);
    void parse_done_msg(Event_engine *event_engine, sc_event *notify_event);

    void iter_start(PD_JOB type); // 填充原语，发送在meminterface完成
    void iter_done(PD_JOB type);

    void set_global_vars(int T, int tp_size = 1);
};