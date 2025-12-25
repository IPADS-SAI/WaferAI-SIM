#pragma once
#include "monitor/config_helper_base.h"

class config_helper_gpu : public config_helper_base {
public:
    vector<StreamConfig> streams;
    int gpu_index; // 正在发放第几个gpu start data，仅在GPU模式下使用
    int done_loop; // 已经完成多少loop了，针对loop字段
    GpuPosLocator *gpu_pos_locator;

    config_helper_gpu(string filename, int config_chip_id = 0);
    config_helper_gpu *clone() const override {
        return new config_helper_gpu(*this);
    }

    void generate_prims(int i);

    void parse_ack_msg(Event_engine *event_engine, int flow_id,
                       sc_event *notify_event);
    void parse_done_msg(Event_engine *event_engine, sc_event *notify_event);

    void printSelf();

    void fill_queue_config(queue<Msg> *q);
    void fill_queue_start(queue<Msg> *q);
};