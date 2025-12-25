#pragma once
#include "common/pd.h"
#include "monitor/config_helper_base.h"
#include "prims/base.h"

class config_helper_gpu_pd : public config_helper_base {
public:
    // serving 相关
    json json_template;
    CoreStatus iter_status;                 // 所有core共用一个CoreStatus
    vector<RequestRecord> requestRecords; // 所有请求的完成情况
    int decode_done;                      // 收到decode的eof完成次数
    vector<Msg> temp_config;              // 存放所有还没有发出去的config
    queue<int> idle_decode;               // 由于超过credit而需要被stall的decode
    queue<int> unfinished_prefill;        // 存储还没有完成的prefill任务

    GpuPosLocator *gpu_pos_locator;

    int prim_index;               // 正在执行的原语编号
    vector<PrimBase *> prim_list; // 所有需要执行的原语内容（每一个iter刷新）

    // 模型配置
    int heads;
    int head_size;
    double eof_chance;
    int model_stage;
    int batch_size;
    int kv_heads;
    vector<pair<string, string>> source_info;

    bool busy;                // 此次iteration是否已经开始
    vector<int> arrival_time; // 记录所有req到达的时间
    vector<Msg> g_done_msg;   // 收集

    config_helper_gpu_pd(string filename, sc_event *ev_sig,
                         int config_chip_id = 0);

    config_helper_gpu_pd *clone() const override {
        return new config_helper_gpu_pd(*this);
    }

    void generate_prims();
    void generate_prims(int i);

    void parse_ack_msg(Event_engine *event_engine, int flow_id,
                       sc_event *notify_event);
    void parse_done_msg(Event_engine *event_engine, sc_event *notify_event);

    void printSelf();

    void fill_queue_config(queue<Msg> *q);
    void fill_queue_start(queue<Msg> *q);

    void iter_start();
    void iter_done(vector<Msg> done_msg);

    void set_global_vars(int T, int tp_size = 1);
};