#include "monitor/config_helper_pd.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

config_helper_pd::config_helper_pd(string filename, sc_event *ev_sig,
                                   int config_chip_id) {
    LOG_INFO(CONFIG) << "Loading config file " << filename;

    json j;
    ifstream jfile(filename);
    jfile >> j;

    decode_done = 0;

    // 收集相关参数
    auto config_reqs = j["requests"];
    auto config_model = j["model"];

    int req_cnt = config_reqs["count"];
    heads = config_model["heads"];
    head_size = config_model["head_size"];
    eof_chance = config_reqs["eof_chance"];
    model_stage = config_model["stage"];
    batch_size = 1;
    kv_heads = config_model["kv_heads"];
    hidden_size = config_model["hidden_size"];
    intermediate_size = config_model["intermediate_size"];
    if (config_model.contains("prefill_iters"))
        prefill_iters = config_model["prefill_iters"];
    else
        prefill_iters = 4;

    json_template = j["chips"][0]["cores"];
    tp_size = json_template.size();

    for (int i = 0; i < req_cnt; i++) {
        vector<double> v;
        token_record.push_back(v);
    }

    // 分配TP组
    attend_cores = GRID_SIZE / (tp_size * model_stage) * model_stage;
    for (int i = 0; i < attend_cores; i++) {
        CoreStatus status = CoreStatus(i * tp_size, JOB_BOTH);
        coreStatus.push_back(status);
    }

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
        record.prefill_iters = prefill_iters;
        requestRecords.push_back(record);
    }

    for (int i = 0; i < attend_cores / model_stage; i++) {
        queue<int> p;
        idle_decode.push_back(p);
    }

    for (int i = 0; i < attend_cores; i++) {
        queue<int> p;
        unfinished_prefill.push_back(p);
    }

    // 建立原语模板
    busy = false;
    g_recv_ack_cnt = 0;
    g_recv_done_cnt = 0;

    ev_sig->notify(0, SC_NS);
}

void config_helper_pd::fill_queue_config(queue<Msg> *q) {
    // 将temp中的所有内容搬运到q中，并清空temp
    for (auto msg : temp_config) {
        auto des = msg.des_;
        int index = des / GRID_X;
        q[index].push(msg);
    }

    temp_config.clear();
}

void config_helper_pd::fill_queue_start(queue<Msg> *q) {
    LOG_INFO(NETWORK) << "Config helper start START data distribution";

    // 所有tp组的初始核需要发送start_data
    for (auto status : coreStatus) {
        int index = status.id / GRID_X;
        int total_pkg = 0;

        for (int i = 0; i < status.batchInfo.size(); i++) {
            auto stage = status.batchInfo[i];
            if (stage.type == PREFILL) {
                auto record = requestRecords[stage.req_id];
                int size =
                    record.seq_len / record.prefill_iters * heads * head_size;
                int send_size_in_bit = size * sizeof(float) * 8;
                int pkg_num = (send_size_in_bit % M_D_DATA)
                                  ? (send_size_in_bit / M_D_DATA + 1)
                                  : (send_size_in_bit / M_D_DATA);
                pkg_num = pkg_num % HW_NOC_PAYLOAD_PER_CYCLE
                              ? pkg_num / HW_NOC_PAYLOAD_PER_CYCLE + 1
                              : pkg_num / HW_NOC_PAYLOAD_PER_CYCLE;

                if (SPEC_USE_BEHA_NOC) {
                    sc_bv<M_D_DATA> d(0x1);
                    int length = M_D_DATA;
                    Msg m = Msg(false, MSG_TYPE::S_DATA, ++total_pkg, status.id,
                                0, status.id, M_D_DATA, d);
                    m.source_ = GRID_SIZE;
                    m.roofline_packets_ = pkg_num;
                    q[index].push(m);
                } else {
                    for (int j = 1; j <= pkg_num; j++) {
                        sc_bv<M_D_DATA> d(0x1);

                        Msg m = Msg(false, MSG_TYPE::S_DATA, j + total_pkg,
                                    status.id, M_D_DATA * (j - 1), status.id,
                                    M_D_DATA, d);
                        m.source_ = GRID_SIZE;
                        m.roofline_packets_ = 1;
                        q[index].push(m);
                    }

                    total_pkg += pkg_num;
                }
            }
        }

        sc_bv<M_D_DATA> d(0x1);
        q[index].push(Msg(true, MSG_TYPE::S_DATA, ++total_pkg, status.id, 0,
                          status.id, 1, d));

        LOG_DEBUG(NETWORK) << "Config helper -> START data -> " << status.id;
    }
}

void config_helper_pd::iter_done(vector<Msg> done_msg) {
    // 按照coreStatus更新requestRecords，理论来说只要获取所有core的batch的req_id即可
    // 如果其中有DECODE done的话就额外更新一次
    // 只有最后一个stage的core才能够更新
    for (auto msg : done_msg) {
        int id = msg.source_ / tp_size;
        if ((id + 1) % model_stage)
            continue;

        auto &status = coreStatus[id];
        int stage_count = 0;
        for (auto &stage : status.batchInfo) {
            auto &record = requestRecords[stage.req_id];
            switch (record.phase) {
            case PREFILL:
                if (++record.prefill_counter == record.prefill_iters) {
                    token_record[record.id].push_back(
                        sc_time_stamp().to_double());
                    stage.type = record.phase = DECODE;
                    stage.token_num = 1;
                }
                break;
            case DECODE:
                record.decode_counter++;
                token_record[record.id].push_back(sc_time_stamp().to_double());
                if (record.decode_counter >= 2 / (eof_chance)) {
                    stage.type = record.phase = PD_DONE;

                    char format_label_k[100];
                    sprintf(format_label_k, "%s%sk#%d", ETERNAL_PREFIX,
                            KVCACHE_PREFIX, stage.req_id);
                    string label_k = format_label_k;
                    for (int i = 0; i < attend_cores; i++) {
                        g_dram_kvtable[i]->remove(label_k);
                    }

                    char format_label_v[100];
                    sprintf(format_label_v, "%s%sv#%d", ETERNAL_PREFIX,
                            KVCACHE_PREFIX, stage.req_id);
                    string label_v = format_label_v;
                    for (int i = 0; i < attend_cores; i++) {
                        g_dram_kvtable[i]->remove(label_v);
                    }

                    if (++decode_done == requestRecords.size())
                        printResults();
                }
                break;
            }

            stage_count++;
        }
    }

    busy = false;
}

void config_helper_pd::iter_start() {
    if (busy)
        return;

    // 为每一个核进行schedule，如果这个核不是第一个stage，则复制前一个stage上一个iter的任务
    vector<vector<Stage>> temp_stage;
    for (auto status : coreStatus) {
        int id = status.id / tp_size;
        if ((id + 1) % model_stage != 1) {
            temp_stage.push_back(coreStatus[id - 1].batchInfo);
        } else {
            // 为stage1核分配任务，取决于前一个iter的最后一个stage核的执行情况。如果任务打不满，主动寻找新的req任务
            int credit = 0;
            vector<Stage> new_stage_1;

            for (auto stage : coreStatus[id + model_stage - 1].batchInfo) {
                switch (stage.type) {
                case PREFILL:
                    break;
                case DECODE:
                    if (credit < HW_CORE_CREDIT) {
                        credit += 1;
                        new_stage_1.push_back(stage);
                    } else {
                        idle_decode[id / model_stage].push(stage.req_id);
                    }
                    break;
                case PD_DONE:
                    break;
                }
            }

            bool new_reqs = true;
            queue<int> &decode_waiting_list = idle_decode[id / model_stage];
            queue<int> &prefill_waiting_list = unfinished_prefill[id];
            LOG_DEBUG(SCHEDULE) << "Schedule for core " << id * tp_size
                                << ", current credit: " << credit;

            while (credit < HW_CORE_CREDIT) {
                // PREFILL new iter > UNTOUCHED

                if (decode_waiting_list.size()) {
                    // 这里从idle_decode中取
                    int req_id = decode_waiting_list.front();
                    decode_waiting_list.pop();
                    credit += 1;
                    new_stage_1.push_back(Stage(req_id, DECODE, 1));
                }

                else if (HW_CORE_CREDIT - credit >= HW_PD_RATIO &&
                         prefill_waiting_list.size()) {
                    // 这里选取还没有做完的prefill任务
                    int req_id = prefill_waiting_list.front();
                    prefill_waiting_list.pop();
                    credit += HW_PD_RATIO;

                    auto &record = requestRecords[req_id];
                    new_stage_1.push_back(
                        Stage(record.id, PREFILL,
                              record.seq_len / record.prefill_iters));

                    if (++record.prefill_distribute < record.prefill_iters)
                        prefill_waiting_list.push(req_id);
                }

                else if (HW_CORE_CREDIT - credit >= HW_PD_RATIO && new_reqs) {
                    // 统计现在可以被指派的请求个数
                    new_reqs = false;

                    for (auto &req : requestRecords) {
                        sc_core::sc_time arv_time(req.arrival_time,
                                                  sc_core::SC_NS);
                        if (req.phase == UNTOUCHED &&
                            arv_time <= sc_time_stamp()) {
                            credit += HW_PD_RATIO;
                            new_stage_1.push_back(
                                Stage(req.id, PREFILL,
                                      req.seq_len / req.prefill_iters));
                            req.phase = PREFILL;

                            if (++req.prefill_distribute < req.prefill_iters)
                                prefill_waiting_list.push(req.id);
                            new_reqs = true;
                            break;
                        }
                    }
                } else
                    break;
            }

            temp_stage.push_back(new_stage_1);
        }
    }

    // 统一更新所有的batchInfo，生成原语
    bool complete_idle = true;

    LOG_DEBUG(SCHEDULE) << "Schedule for this iteration";
    for (int s = 0; s < coreStatus.size(); s++) {
        auto &status = coreStatus[s];
        status.batchInfo = temp_stage[s];

        LOG_DEBUG(SCHEDULE) << "  Core " << status.id;
        for (auto stage : status.batchInfo) {
            complete_idle = false;

            LOG_DEBUG(SCHEDULE)
                << "    REQ: " << stage.req_id << ", TYPE: " << stage.type
                << ", finished iter: "
                << ((requestRecords[stage.req_id].phase == PREFILL)
                        ? requestRecords[stage.req_id].prefill_counter
                        : requestRecords[stage.req_id].decode_counter)
                << ", iter count "
                << requestRecords[stage.req_id].prefill_iters;
        }

        generate_prims(status.id);
    }

    if (complete_idle) {
        // 如果当前iter没有任何core有工作，则不发放config
        temp_config.clear();
        busy = false;
        LOG_DEBUG(SCHEDULE) << "Complete idle";
    } else
        busy = true;
}

void config_helper_pd::printSelf() {}

void config_helper_pd::generate_prims(int i) {
    auto status = coreStatus[i / tp_size];

    // 计算input token大小
    int B = 1, NH = heads, T = 0, C = heads * head_size;
    for (auto stage : status.batchInfo) {
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

    // TODO: 其他decoder模型适配？
    set_global_vars(T, tp_size);

    // lambda函数
    auto add_recv = [&](int &prim_seq, bool start, int recv_tag, int recv_cnt,
                        int core_id) {
        // 如果是tp组的第一个核的第一个work，则为RECV_START，否则为RECV_DATA
        Recv_prim *recv_data =
            (Recv_prim *)PrimFactory::getInstance().createPrim("Recv_prim");
        recv_data->type = start ? RECV_TYPE::RECV_START : RECV_TYPE::RECV_DATA;
        recv_data->recv_cnt = recv_cnt;
        recv_data->tag_id = recv_tag;

        Msg m = Msg(false, MSG_TYPE::CONFIG, ++prim_seq, core_id,
                    recv_data->serialize()[0]);

        temp_config.push_back(m);
    };

    // 处理tp的模板核
    vector<CoreConfig> template_cores;
    for (auto &j : json_template) {
        CoreConfig core = j;
        template_cores.push_back(core);
    }

    // 对于tp组的第一个核，标记输出标签
    string output_label = "";

    // 为每一个核做相同的原语生成
    // 为每一个work生成前后的send和recv原语
    for (int core_id = i; core_id < i + tp_size; core_id++) {
        int index = core_id / GRID_X;
        int prim_seq = 0;

        // 每个核生成一个set_batch
        PrimBase *set_batch = new Set_batch(status.batchInfo);
        g_prim_stash.push_back(set_batch);
        auto segments = set_batch->serialize();
        for (int seg = 0; seg < segments.size(); seg++)
            temp_config.push_back(
                Msg(!status.batchInfo.size() && core_id % tp_size &&
                        seg == segments.size() - 1,
                    MSG_TYPE::CONFIG, ++prim_seq, core_id,
                    seg == segments.size() - 1, segments[seg]));

        if (status.batchInfo.size()) {
            for (int w = 0; w < template_cores[core_id - i].worklist.size();
                 w++) {
                auto &work = template_cores[core_id - i].worklist[w];
                add_recv(prim_seq, (w == 0 && core_id == i), work.recv_tag + i,
                         work.recv_cnt, core_id);

                // work的所有计算原语
                for (int p = 0; p < work.prims.size(); p++) {
                    auto &prim = work.prims[p];
                    PrimBase *set_addr =
                        PrimFactory::getInstance().createPrim("Set_addr");
                    auto label = set_addr->prim_context->datapass_label_;

                    if (prim->prim_type & COMP_PRIM) {
                        for (int i = 0; i < MAX_SPLIT_NUM; i++)
                            label->indata[i] =
                                prim->prim_context->datapass_label_->indata[i];
                        label->outdata =
                            prim->prim_context->datapass_label_->outdata;
                    }

                    auto segments = set_addr->serialize();
                    for (int seg = 0; seg < segments.size(); seg++)
                        temp_config.push_back(
                            Msg(false, MSG_TYPE::CONFIG, ++prim_seq, core_id,
                                seg == segments.size() - 1, segments[seg]));

                    segments = prim->serialize();
                    for (int seg = 0; seg < segments.size(); seg++)
                        temp_config.push_back(
                            Msg(false, MSG_TYPE::CONFIG, ++prim_seq, core_id,
                                seg == segments.size() - 1, segments[seg]));

                    if (w == template_cores[core_id - i].worklist.size() &&
                        p == work.prims.size() - 1 && i % tp_size == 0)
                        output_label = label->outdata;
                }

                // 需要计算send_data的发送包裹数，首先找到这个work的最后一个计算原语
                CompBase *last_comp = (CompBase *)work.prims.back();

                // 发送原语，遵循work中的cast，编号和tag需要自定义
                for (int c = 0; c < work.cast.size(); c++) {
                    auto ca = work.cast[c];
                    int next_id = ca.dest + i;
                    Send_prim *send_req =
                        new Send_prim(SEND_TYPE::SEND_REQ, next_id, ca.tag + i);
                    Recv_prim *recv_ack = new Recv_prim(RECV_TYPE::RECV_ACK);
                    Send_prim *send_data = new Send_prim(SEND_TYPE::SEND_DATA,
                                                         next_id, ca.tag + i);
                    g_prim_stash.push_back(send_req);
                    g_prim_stash.push_back(recv_ack);
                    g_prim_stash.push_back(send_data);

                    CalculatePacketNum(
                        last_comp->out_size, ca.weight, last_comp->data_byte,
                        send_data->max_packet, send_data->end_length);
                    send_data->output_label =
                        last_comp->prim_context->datapass_label_->outdata;

                    temp_config.push_back(Msg(false, MSG_TYPE::CONFIG,
                                              ++prim_seq, core_id,
                                              send_req->serialize()[0]));
                    temp_config.push_back(Msg(false, MSG_TYPE::CONFIG,
                                              ++prim_seq, core_id,
                                              recv_ack->serialize()[0]));
                    temp_config.push_back(Msg(
                        core_id != i && c == work.cast.size() - 1 &&
                            w ==
                                template_cores[core_id - i].worklist.size() - 1,
                        MSG_TYPE::CONFIG, ++prim_seq, core_id,
                        send_data->serialize()[0]));
                }
            }
        } else if (!(core_id % tp_size)) {
            add_recv(prim_seq, true, core_id, 1, core_id);
        }

        // 处理数据流向下一个core
        // 这里只有tp_group的第一个核才需要发送
        if (!(core_id % tp_size)) {
            int group_i = core_id / tp_size;

            int send_dest = group_i + 1;
            if (send_dest % model_stage == 0)
                send_dest -= model_stage;
            send_dest *= tp_size;
            int recv_source = group_i - 1;
            if (recv_source < 0)
                recv_source += model_stage;
            else if (recv_source % model_stage == model_stage - 1)
                recv_source += model_stage;
            recv_source *= tp_size;

            int send_tag = core_id + tp_size * send_dest;
            int recv_tag = recv_source + core_id * tp_size;

            PrimBase *recv_data_2 =
                new Recv_prim(RECV_TYPE::RECV_DATA, recv_tag, 1);
            PrimBase *send_req =
                new Send_prim(SEND_TYPE::SEND_REQ, send_dest, send_tag);
            PrimBase *recv_ack = new Recv_prim(RECV_TYPE::RECV_ACK);
            Send_prim *send_data =
                new Send_prim(SEND_TYPE::SEND_DATA, send_dest, send_tag);
            send_data->output_label = output_label;
            g_prim_stash.push_back(recv_data_2);
            g_prim_stash.push_back(send_req);
            g_prim_stash.push_back(recv_ack);
            g_prim_stash.push_back(send_data);

            int output_size = max(int(C * T * B), 1);
            CalculatePacketNum(output_size, 1, 1, send_data->max_packet,
                               send_data->end_length);

            if ((core_id / tp_size + 1) % model_stage != 1) {
                // 不是stage 1 就是接收上一个 stage 传过来的中间结果
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id,
                                          recv_data_2->serialize()[0]));
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_req->serialize()[0]));
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, recv_ack->serialize()[0]));
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_data->serialize()[0]));
            } else {
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_req->serialize()[0]));
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, recv_ack->serialize()[0]));
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_data->serialize()[0]));
                // stage 1 的话又要接受新的prefilling recv_data1
                // 这里的recv_data2 是来自decoding 的数据
                temp_config.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id,
                                          recv_data_2->serialize()[0]));
            }

            // tp组的第一个核需要向memInterface发送DONE信号
            PrimBase *send_done = new Send_prim(SEND_TYPE::SEND_DONE);
            g_prim_stash.push_back(send_done);
            Msg m = Msg(true, MSG_TYPE::CONFIG, ++prim_seq, core_id,
                        send_done->serialize()[0]);
            m.refill_ = false;
            temp_config.push_back(m);
        }
    }
}

void config_helper_pd::parse_ack_msg(Event_engine *event_engine, int flow_id,
                                     sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting Recv Ack", "B",
                            Trace_event_util());

    for (auto m : g_temp_ack_msg) {
        int cid = m.source_;

        LOG_DEBUG(NETWORK) << "Config helper <- ACK <- " << cid << ", total "
                           << g_recv_ack_cnt + 1 << "/"
                           << coreStatus.size() * tp_size;

        g_recv_ack_cnt++;
    }

    g_temp_ack_msg.clear();
    event_engine->add_event(this->name(), "Waiting Recv Ack", "E",
                            Trace_event_util());

    if (g_recv_ack_cnt >= coreStatus.size() * tp_size) {
        g_recv_ack_cnt = 0;
        notify_event->notify(CYCLE, SC_NS);
    }
}

void config_helper_pd::parse_done_msg(Event_engine *event_engine,
                                      sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting Core busy", "B",
                            Trace_event_util());

    for (auto m : g_temp_done_msg) {
        int cid = m.source_;

        LOG_DEBUG(NETWORK) << "Config helper <- DONE <- " << cid << ", total "
                           << g_recv_done_cnt + 1 << " / " << coreStatus.size();

        g_recv_done_cnt++;
        g_done_msg.push_back(m);
    }
    g_temp_done_msg.clear();
    event_engine->add_event(this->name(), "Waiting Core busy", "E",
                            Trace_event_util());

    // 只有tp的第一个核会发送done信号
    if (g_recv_done_cnt >= coreStatus.size()) {
        iter_done(g_done_msg);

        for (PrimBase *p : g_prim_stash) {
            delete p;
        }
        g_prim_stash.clear();

        g_done_msg.clear();
        g_recv_done_cnt = 0;

        notify_event->notify(CYCLE, SC_NS);
    }
}

void config_helper_pd::set_global_vars(int T, int tp_size) {
    int C = heads * head_size;
    int P = hidden_size;
    int J = intermediate_size;
    vtable = {{"B", 1},
              {"T", T},
              {"chunk", 1},
              {"T/2", T / 2},
              {"C", C},
              {"P", P},
              {"J", J},
              {"PJ", P * J},
              {"JP", J * P},
              {"CP", C * P},
              {"PC", P * C},
              {"3PC", 3 * P * C},
              {"BTP", T * P},
              {"BTJ", T * J},
              {"NH", heads},
              {"DH", head_size},
              {"R", heads / kv_heads},
              {"3C", 3 * C},
              {"3C/2", 3 * C / 2},
              {"3CC/2", 3 * C * C / 2},
              {"4C", 4 * C},
              {"BTC", T * C},
              {"2BTC", 2 * T * C},
              {"3BTC", 3 * T * C},
              {"4BTC", 4 * T * C},
              {"3C-R", C * (2 + heads / kv_heads) / (heads / kv_heads)},
              {"CHUNK", prefill_iters}};

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
        int divisor = 1 << i; // 2, 4, 8, ...
        string suffix = "/" + to_string(divisor);

        // T 的版本
        vtable.push_back({"T" + suffix, T / divisor});
        vtable.push_back({"NH" + suffix, heads / divisor});
        vtable.push_back({"P" + suffix, P / divisor});
        vtable.push_back({"J" + suffix, J / divisor});
        vtable.push_back({"JP" + suffix, J * P / divisor});
        vtable.push_back({"PJ" + suffix, J * P / divisor});
        vtable.push_back({"CP" + suffix, J * P / divisor});
        vtable.push_back({"PC" + suffix, J * P / divisor});
        vtable.push_back({"3PC" + suffix, 3 * C * P / divisor});

        // BTC 相关参数的版本
        vtable.push_back({"BTC" + suffix, (T * C) / divisor});
        vtable.push_back({"2BTC" + suffix, (2 * T * C) / divisor});
        vtable.push_back({"3BTC" + suffix, (3 * T * C) / divisor});
        vtable.push_back({"4BTC" + suffix, (4 * T * C) / divisor});
        vtable.push_back({"BTP" + suffix, (T * P) / divisor});
        vtable.push_back({"BTJ" + suffix, (T * J) / divisor});
    }

    // 为所有涉及 C 的参数生成除以 2、4、8 等的版本
    for (int i = 1; i <= log2_tp_size; i++) {
        int divisor = 1 << i; // 2, 4, 8, ...
        string suffix = "/" + to_string(divisor);

        // C 的版本
        vtable.push_back({"C" + suffix, C / divisor});

        // 3C 相关参数的版本
        vtable.push_back({"3C" + suffix, (3 * C) / divisor});
        vtable.push_back({"4C" + suffix, (4 * C) / divisor});
        vtable.push_back({"3CC" + suffix, (3 * C * C) / divisor});
        vtable.push_back(
            {"3C-R" + suffix,
             (C * (2 + heads / kv_heads) / (heads / kv_heads)) / divisor});
    }

    for (auto &pair : vtable) {
        if (pair.second == 0)
            pair.second = 1;
    }
}

void config_helper_pd::printResults() {
    LOG_INFO(SYSTEM) << "All requests finished";
    LOG_INFO(CATCH_TEST) << "Catch test finished";

    ofstream outfile("simulation_result_df_pd.txt", ios::app);
    if (outfile.is_open()) {
        outfile << "[CATCH TEST] " << sc_time_stamp() << "HW_SRAM_SIZE "
                << HW_SRAM_SIZE << " BANDWIDTH " << HW_DRAM_DEFAULT_BITWIDTH
                << endl;
        outfile.close();
    } else
        LOG_ERROR(config_helper_pd.cpp)
            << "Failed to open file simulation_result_df_pd.txt";

    ofstream file("token_records.txt", ios::app);

    if (!file.is_open())
        LOG_ERROR(config_helper_pd.cpp)
            << "Failed to open file token_records.txt";

    // 设置输出格式，避免科学计数法
    file << fixed << setprecision(6); // 设置小数点后6位精度，可根据需要调整

    file << "*"
         << "*\n";
    for (int i = 0; i < token_record.size(); i++) {
        file << "Request " << i << ": \n";
        for (int j = 0; j < token_record[i].size(); j++) {
            file << "Token " << j << ": " << token_record[i][j] << "\n";
        }
    }

    file << "\n\n";
    file.close();
    sc_stop();
}