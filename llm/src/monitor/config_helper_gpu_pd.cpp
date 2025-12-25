#include "monitor/config_helper_gpu_pd.h"
#include "prims/gpu_prims.h"
#include "prims/norm_prims.h"
#include "utils/config_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

config_helper_gpu_pd::config_helper_gpu_pd(string filename, sc_event *ev_sig,
                                           int config_chip_id) {
    LOG_INFO(CONFIG) << "Loading config file " << filename;

    json j;
    ifstream jfile(filename);
    jfile >> j;

    decode_done = 0;
    prim_index = 0;

    auto config_reqs = j["requests"];
    int req_cnt = config_reqs["count"];

    heads = config_reqs["heads"];
    head_size = config_reqs["head_size"];
    kv_heads = config_reqs["kv_heads"];
    eof_chance = config_reqs["eof_chance"];
    batch_size = config_reqs["batch_size"];

    auto config_source = j["source"];
    for (int i = 0; i < config_source.size(); i++) {
        source_info.push_back(
            make_pair(config_source[i]["label"], config_source[i]["size"]));
    }

    iter_status = CoreStatus(0, JOB_BOTH);

    int arr_size = config_reqs["arrival"].size();
    if (arr_size < req_cnt) {
        for (int i = 0; i < arr_size; i++)
            arrival_time.push_back(config_reqs["arrival"][i]);

        for (int i = arr_size; i < req_cnt; i++)
            arrival_time.push_back(config_reqs["arrival"][arr_size - 1]);
    } else {
        for (int i = 0; i < req_cnt; i++)
            arrival_time.push_back(config_reqs["arrival"][i]);
    }

    for (int i = 0; i < req_cnt; i++) {
        RequestRecord record =
            RequestRecord(i, config_reqs["seq_len"], heads, arrival_time[i]);
        requestRecords.push_back(record);
    }

    // 建立原语模板
    json_template = j["chips"][0]["streams"][0];
    busy = false;
    g_recv_ack_cnt = 0;
    g_recv_done_cnt = 0;

    ev_sig->notify(0, SC_NS);
}

void config_helper_gpu_pd::fill_queue_config(queue<Msg> *q) {
    // 将temp中的所有内容搬运到q中，并清空temp
    for (auto msg : temp_config) {
        auto des = msg.des_;
        int index = des / GRID_X;
        q[index].push(msg);
    }

    temp_config.clear();
}

void config_helper_gpu_pd::fill_queue_start(queue<Msg> *q) {
    LOG_INFO(NETWORK) << "Config helper start START data distribution, phase "
                      << prim_index;

    // 如果是第一个原语且有prefill任务，则需要预先发送数据
    bool has_prefill = false;
    for (auto stage : iter_status.batchInfo) {
        if (stage.type == PREFILL)
            has_prefill = true;
    }

    // 发送数据的大小等于通过source查找
    if (prim_index == 0 && has_prefill) {
        for (auto source : source_info) {
            AddrPosKey source_key =
                AddrPosKey(0, GetDefinedParam(source.second));
            gpu_pos_locator->addPair(source.first, source_key);
        }
    }

    // 直接获取这一个prim有几个核参加
    int sms = ((GpuBase *)prim_list[prim_index])->req_sm;
    for (int i = 0; i < min(sms, GRID_SIZE); i++) {
        int index = i / GRID_X;
        int pkg_index = 0;

        // 这里相当于quick start，实际上也只有第一个原语需要初始数据
        sc_bv<128> d(0x1);
        Msg m = Msg(true, MSG_TYPE::S_DATA, pkg_index + 1, i, 0, i, 0, d);
        m.source_ = GRID_SIZE;
        q[index].push(m);
    }
}

void config_helper_gpu_pd::iter_done(vector<Msg> done_msg) {
    // 更新prim_index，如果prim_index等于prim_list的长度，则说明所有原语已经完成
    // 则根据iter_status更新requestRecords
    prim_index++;

    if (prim_index == prim_list.size()) {
        for (auto &stage : iter_status.batchInfo) {
            auto &record = requestRecords[stage.req_id];
            switch (record.phase) {
            case PREFILL:
                if (++record.prefill_counter == record.prefill_iters) {
                    stage.type = record.phase = DECODE;
                    stage.token_num = 1;
                }
                break;
            case DECODE:
                record.decode_counter++;
                if (record.decode_counter >= (1.5) / (eof_chance)) {
                    stage.type = record.phase = PD_DONE;

                    if (++decode_done == requestRecords.size()) {
                        LOG_INFO(SYSTEM) << "All requests finished";
                        LOG_INFO(CATCH_TEST) << "Catch test finished";

                        ofstream outfile("simulation_result.txt", ios::app);
                        if (outfile.is_open()) {
                            outfile << "[CATCH TEST] " << sc_time_stamp()
                                    << "L1CACHESIZE " << L1CACHESIZE
                                    << " L2CACHESIZE " << L2CACHESIZE
                                    << " BANDWIDTH " << GPU_DRAM_BANDWIDTH
                                    << endl;
                            outfile.close();
                        } else {
                            LOG_ERROR(config_helper_gpu_pd.cpp)
                                << "Failed to open simulation_result_gpu.txt";
                        }
                        sc_stop();
                    }
                }
                break;
            }
        }

        prim_index = 0;
    }

    busy = false;
}

void config_helper_gpu_pd::iter_start() {
    if (busy)
        return;

    // 如果prim_index是0，则首先清空prim_list，然后生成新的
    // 随后，根据现在的prim_index生成原语（只生成一个原语，如果该原语涉及多轮SM计算，也需要一并生成）
    if (prim_index == 0) {
        prim_list.clear();

        // 根据上一轮的状态生成新的原语
        vector<Stage> new_stage;
        int credit = 0;

        // 优先将上一个iter中的decode加入这个iter。如果塞不下，则放入idle_decode中
        for (auto stage : iter_status.batchInfo) {
            switch (stage.type) {
            case PREFILL:
                break;
            case DECODE:
                if (credit < HW_CORE_CREDIT) {
                    credit += 1;
                    new_stage.push_back(stage);
                } else {
                    idle_decode.push(stage.req_id);
                }
                break;
            case PD_DONE:
                break;
            }
        }

        // 如果此时还放得下，则优先从idle_decode中取
        bool new_reqs = true;

        while (credit < HW_CORE_CREDIT) {
            if (idle_decode.size()) {
                // 这里从idle_decode中取
                int req_id = idle_decode.front();
                idle_decode.pop();
                credit += 1;
                new_stage.push_back(Stage(req_id, DECODE, 1));
            }

            else if (HW_CORE_CREDIT - credit >= HW_PD_RATIO &&
                     unfinished_prefill.size()) {
                // 这里选取还没有做完的prefill任务
                int req_id = unfinished_prefill.front();
                unfinished_prefill.pop();
                credit += HW_PD_RATIO;

                auto &record = requestRecords[req_id];
                new_stage.push_back(Stage(
                    record.id, PREFILL, record.seq_len / record.prefill_iters));

                if (++record.prefill_distribute < record.prefill_iters)
                    unfinished_prefill.push(req_id);
            }

            else if (HW_CORE_CREDIT - credit >= HW_PD_RATIO && new_reqs) {
                // 统计现在可以被指派的请求个数
                new_reqs = false;

                for (auto &req : requestRecords) {
                    sc_core::sc_time arv_time(req.arrival_time, sc_core::SC_NS);
                    if (req.phase == UNTOUCHED && arv_time <= sc_time_stamp()) {
                        credit += HW_PD_RATIO;
                        new_stage.push_back(Stage(
                            req.id, PREFILL, req.seq_len / req.prefill_iters));
                        req.phase = PREFILL;

                        if (++req.prefill_distribute < req.prefill_iters)
                            unfinished_prefill.push(req.id);
                        new_reqs = true;
                        break;
                    }
                }
            } else
                break;
        }

        // 开始生成原语，填入prim_list中
        LOG_DEBUG(SCHEDULE) << "Schedule for this iteration";
        iter_status.batchInfo = new_stage;
        generate_prims();

        for (auto stage : iter_status.batchInfo) {
            LOG_DEBUG(SCHEDULE)
                << "    REQ: " << stage.req_id << ", TYPE: " << stage.type
                << ", finished iter: "
                << ((requestRecords[stage.req_id].phase == PREFILL)
                        ? requestRecords[stage.req_id].prefill_counter
                        : requestRecords[stage.req_id].decode_counter)
                << ", iter count "
                << requestRecords[stage.req_id].prefill_iters;
        }
    }

    // 随后按照prim_index为每一个核分配原语
    generate_prims(prim_index);

    if (iter_status.batchInfo.size() == 0) {
        // 如果当前iter没有任何core有工作，则不发放config
        temp_config.clear();
        busy = false;
        LOG_DEBUG(SCHEDULE) << "Complete idle";
    } else
        busy = true;
}

void config_helper_gpu_pd::generate_prims() {
    // 根据iter_status填满prim_list，这里不包含任何收发原语，只有计算原语
    int B = 1, NH = heads, T = 0, C = heads * head_size;
    for (auto stage : iter_status.batchInfo) {
        auto record = requestRecords[stage.req_id];
        switch (stage.type) {
        case PREFILL:
            T += record.seq_len / record.prefill_iters;
            break;
        case DECODE:
            T += 1;
            break;
        }
    }

    set_global_vars(T, 1);
    StreamConfig stream = json_template;
    prim_list = stream.prims;
}

void config_helper_gpu_pd::generate_prims(int i) {
    GpuBase *prim = (GpuBase *)prim_list[i];
    int sms = prim->req_sm;

    for (int c = 0; c < GRID_SIZE; c++) {
        int prim_seq = 0;

        // 若c大于sms，则跳过
        if (c >= sms)
            continue;

        PrimBase *recv_data_1 = new Recv_prim(RECV_TYPE::RECV_START, c, 1);
        temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq, c,
                                  recv_data_1->serialize()[0]));

        PrimBase *set_batch = new Set_batch(iter_status.batchInfo, false);
        auto segments = set_batch->serialize();
        for (int seg = 0; seg < segments.size(); seg++)
            temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq, c,
                                      seg == segments.size() - 1,
                                      segments[seg]));

        // 只需要看单个原语重复次数
        int repeat = sms / GRID_SIZE + (sms % GRID_SIZE > c);
        PrimBase *set_addr = PrimFactory::getInstance().createPrim("Set_addr");
        auto label = set_addr->prim_context->datapass_label_;

        for (int r = 0; r < repeat; r++) {
            for (int i = 0; i < MAX_SPLIT_NUM; i++) {
                label->indata[i] =
                    prim->prim_context->datapass_label_->indata[i];
            }
            label->outdata = prim->prim_context->datapass_label_->outdata;

            auto segments = set_addr->serialize();
            for (int seg = 0; seg < segments.size(); seg++)
                temp_config.push_back(
                    Msg(false, MSG_TYPE::CONFIG, ++prim_seq, c,
                        seg == segments.size() - 1, segments[seg]));

            prim->fetch_index = c + r * GRID_SIZE;
            segments = prim->serialize();
            for (int seg = 0; seg < segments.size(); seg++)
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          c, seg == segments.size() - 1,
                                          segments[seg]));
            prim->fetch_index = 0;
        }

        // 发送DONE信号
        PrimBase *send_done = new Send_prim(SEND_TYPE::SEND_DONE);
        Msg m = Msg(true, MSG_TYPE::CONFIG, ++prim_seq, c,
                    send_done->serialize()[0]);
        m.refill_ = false;
        temp_config.push_back(m);
    }
}

void config_helper_gpu_pd::set_global_vars(int T, int tp_size) {
    int C = heads * head_size;
    vtable.clear();
    vtable.push_back(make_pair("B", 1));
    vtable.push_back(make_pair("T", T));
    vtable.push_back(make_pair("C", C));
    vtable.push_back(make_pair("NH", heads));
    vtable.push_back(make_pair("DH", head_size));
    vtable.push_back(make_pair("R", heads / kv_heads));
    vtable.push_back(make_pair("3C", 3 * C));
    vtable.push_back(make_pair("4C", 4 * C));
    vtable.push_back(make_pair("BTC", T * C));
    vtable.push_back(make_pair("2BTC", 2 * T * C));
    vtable.push_back(make_pair("3BTC", 3 * T * C));
    vtable.push_back(make_pair("4BTC", 4 * T * C));
    vtable.push_back(make_pair("CR", head_size * kv_heads));
    vtable.push_back(make_pair("3CR", 3 * kv_heads * head_size));

    // 根据 tp_size 生成对应的除以 2 的幂次方的版本
    // tp_size 一定是 2 的幂，所以可以计算需要生成多少个版本
    int log2_tp_size = 0;
    int temp_tp_size = tp_size;
    while (temp_tp_size > 1) {
        log2_tp_size++;
        temp_tp_size /= 2;
    }

    // 为所有涉及 T 的参数生成除以 2、4、8 等的版本
    for (int i = 1; i <= log2_tp_size; i++) {
        int divisor = 1 << i;  // 2, 4, 8, ...
        string suffix = "/" + to_string(divisor);
        
        // T 的版本
        vtable.push_back(make_pair("T" + suffix, T / divisor));
        
        // BTC 相关参数的版本
        vtable.push_back(make_pair("BTC" + suffix, (T * C) / divisor));
        vtable.push_back(make_pair("2BTC" + suffix, (2 * T * C) / divisor));
        vtable.push_back(make_pair("3BTC" + suffix, (3 * T * C) / divisor));
        vtable.push_back(make_pair("4BTC" + suffix, (4 * T * C) / divisor));
    }

    // 为所有涉及 C 的参数生成除以 2、4、8 等的版本
    for (int i = 1; i <= log2_tp_size; i++) {
        int divisor = 1 << i;  // 2, 4, 8, ...
        string suffix = "/" + to_string(divisor);
        
        // C 的版本
        vtable.push_back(make_pair("C" + suffix, C / divisor));
        
        // 3C 相关参数的版本
        vtable.push_back(make_pair("3C" + suffix, (3 * C) / divisor));
        vtable.push_back(make_pair("4C" + suffix, (4 * C) / divisor));
        vtable.push_back(make_pair("3CR" + suffix, (3 * kv_heads * head_size) / divisor));
        vtable.push_back(make_pair("CR" + suffix, (head_size * kv_heads) / divisor));
    }
}

void config_helper_gpu_pd::parse_ack_msg(Event_engine *event_engine,
                                         int flow_id, sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting Recv Ack", "B",
                            Trace_event_util());

    // 计算本iter参与计算的core数量
    int sms = ((GpuBase *)prim_list[prim_index])->req_sm;
    int attend_cores = sms >= GRID_SIZE ? GRID_SIZE : sms;

    for (auto m : g_temp_ack_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- ACK <- " << cid << ", total "
                           << g_recv_ack_cnt + 1 << " / " << attend_cores;

        g_recv_ack_cnt++;
    }

    g_temp_ack_msg.clear();
    event_engine->add_event(this->name(), "Waiting Recv Ack", "E",
                            Trace_event_util());

    if (g_recv_ack_cnt >= attend_cores) {
        g_recv_ack_cnt = 0;
        notify_event->notify(CYCLE, SC_NS);
    }
}

void config_helper_gpu_pd::parse_done_msg(Event_engine *event_engine,
                                          sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting Core busy", "B",
                            Trace_event_util());

    // 计算本iter参与计算的core数量
    int sms = ((GpuBase *)prim_list[prim_index])->req_sm;
    int attend_cores = sms >= GRID_SIZE ? GRID_SIZE : sms;

    for (auto m : g_temp_done_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- DONE <- " << cid << ", total "
                           << g_recv_done_cnt + 1 << " / " << attend_cores;

        g_recv_done_cnt++;
        g_done_msg.push_back(m);
    }
    g_temp_done_msg.clear();
    event_engine->add_event(this->name(), "Waiting Core busy", "E",
                            Trace_event_util());

    if (g_recv_done_cnt >= attend_cores) {
        iter_done(g_done_msg);

        g_done_msg.clear();
        g_recv_done_cnt = 0;
        notify_event->notify(CYCLE, SC_NS);
    }
}

void config_helper_gpu_pd::printSelf() {}