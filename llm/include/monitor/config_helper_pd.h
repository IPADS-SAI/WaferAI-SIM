#pragma once
#include "systemc.h"
#include <vector>

#include "common/pd.h"
#include "monitor/config_helper_base.h"

using namespace std;

class config_helper_pd : public config_helper_base {
public:
    json json_template;
    vector<CoreStatus> coreStatus;
    vector<RequestRecord> requestRecords;
    int decode_done;                // 收到decode的eof完成次数
    vector<Msg> temp_config;        // 存放所有还没有发出去的config
    vector<queue<int>> idle_decode; // 由于超过credit而需要被stall的decode
    vector<queue<int>> unfinished_prefill; // 存储还没有完成的prefill任务
    vector<vector<double>> token_record;

    bool busy;                // 此次iteration是否已经开始
    vector<int> arrival_time; // 记录所有req到达的时间
    vector<Msg> g_done_msg;   // 收集

    // 模型配置
    int heads;
    int head_size;
    double eof_chance;
    int model_stage;
    int batch_size;
    int kv_heads;
    int attend_cores; // 参与仿真的核数量，不一定等于总核数
    int prefill_iters; // prefill的总分块数
    int hidden_size;
    int intermediate_size;

    int tp_size; // tp组的大小

    config_helper_pd(string filename, sc_event *ev_sig,
                     int config_chip_id = 0);

    config_helper_pd *clone() const override {
        return new config_helper_pd(*this);
    }

    void generate_prims(int i);

    void printSelf();

    void random_core();

    void fill_queue_start(queue<Msg> *q);
    void fill_queue_config(queue<Msg> *q);

    void parse_ack_msg(Event_engine *event_engine, int flow_id,
                       sc_event *notify_event);
    void parse_done_msg(Event_engine *event_engine, sc_event *notify_event);

    void iter_start(); // 填充原语，发送在meminterface完成
    void iter_done(vector<Msg> done_msg);

    void set_global_vars(int T, int tp_size = 1);
    void printResults();
};