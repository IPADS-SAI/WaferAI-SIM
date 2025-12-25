#include "monitor/config_helper_pds.h"
#include "defs/spec.h"
#include "prims/norm_prims.h"
#include "prims/pd_prims.h"
#include "utils/config_utils.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

config_helper_pds::config_helper_pds(string filename, sc_event *ev_sig,
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
    for (int i = 0; i < req_cnt; i++) {
        vector<double> v;
        token_record.push_back(v);
    }


    heads = config_model["heads"];
    head_size = config_model["head_size"];
    kv_heads = config_model["kv_heads"];
    hidden_size = config_model["hidden_size"];
    intermediate_size = config_model["intermediate_size"];
    eof_chance = config_reqs["eof_chance"];
    // prefill 的 pp 阶数
    prefill_stage = config_model["prefill_stage"];
    decode_stage = config_model["decode_stage"];
    // 总共参与prefill的核数，pp * dp 不包括 tp 的数量
    prefill_core = config_model["prefill_cores"];
    decode_core = config_model["decode_cores"];
    batch_size = 1;
    // prefill_iter 表示 prefill 的 chunk 数量
    if (config_model.contains("prefill_iters"))
        prefill_iters = config_model["prefill_iters"];
    else
        prefill_iters = 4;

    // 建立原语模板
    json_template_p = j["chips"][0]["cores"]["prefill"];
    json_template_d = j["chips"][0]["cores"]["decode"];
    tp_size = json_template_p.size();

    for (int i = 0; i < prefill_core + decode_core; i++) {
        if (i < prefill_core) {
            stage_index.push_back(i % prefill_stage + 1);
            coreStatus.push_back(CoreStatus(i * tp_size, JOB_PREFILL));
        } else {
            stage_index.push_back((i - prefill_core) % decode_stage + 1);
            coreStatus.push_back(CoreStatus(i * tp_size, JOB_DECODE));
        }
    }

    int arr_size = config_reqs["arrival"].size();
    if (arr_size < req_cnt) {
        for (int i = 0; i < arr_size; i++)
            arrival_time.push_back(config_reqs["arrival"][i]);
        // 后续 都按照 最后要给 到达时间来计算
        for (int i = arr_size; i < req_cnt; i++)
            arrival_time.push_back(config_reqs["arrival"][arr_size - 1]);
    } else {
        for (int i = 0; i < req_cnt; i++)
            arrival_time.push_back(config_reqs["arrival"][i]);
    }

    // 检查batch_size参数的合理性，同时依此修改arrive时间
    // 能放的下 prefill
    if (batch_size * HW_PD_RATIO > HW_CORE_CREDIT) {
        LOG_ERROR(config_helper_pds.cpp) << "Batch size too large";
    } else {
        for (int i = 0; i < req_cnt; i++) {
            int target = min((i / batch_size + 1) * batch_size - 1, req_cnt);
            // 按照 batch 重新调整arrival_time
            arrival_time[i] = arrival_time[target];
        }
    }

    for (int i = 0; i < decode_core / decode_stage; i++) {
        queue<int> q;
        idle_decode.push_back(q);
    }

    for (int i = 0; i < req_cnt; i++) {
        RequestRecord record =
            RequestRecord(i, config_reqs["seq_len"], heads, arrival_time[i]);
        record.prefill_iters = prefill_iters;
        requestRecords.push_back(record);
    }

    busy_d = busy_p = false;
    g_recv_ack_cnt_d = g_recv_ack_cnt_p = g_recv_done_cnt_d =
        g_recv_done_cnt_p = 0;
    wait_schedule_p = true;
    wait_schedule_d = false;
    wait_send_start_prefill = false;
    wait_send_start_decode = false;
    need_trigger_send_start = false;

    ev_sig->notify(0, SC_NS);
}

void config_helper_pds::fill_queue_config(queue<Msg> *q) {
    // 将temp中的所有内容搬运到q中，并清空temp
    LOG_INFO(NETWORK) << "Config helper start CONFIG distribution";
    for (auto msg : temp_config) {
        auto des = msg.des_;
        int index = des / GRID_X;
        q[index].push(msg);
    }

    temp_config.clear();
}


// 每一拍都是config ack start done
void config_helper_pds::fill_queue_start(queue<Msg> *q) {
    // 只有在stage 1的core进行prefill的时候，才需要发送start data
    // 在调用这个函数的时候，已经完成对core的config发放
    LOG_INFO(NETWORK) << "Config helper start START data distribution";
    if (!wait_send_start_prefill && !wait_send_start_decode)
        return;
    // 为什么这里 start 都需要发
    for (auto status : coreStatus) {
        int index = status.id / GRID_X;
        int total_pkg = 0;
        // 是 prefill 的核 但是prefill已经结束了
        if (!wait_send_start_prefill && status.id / tp_size < prefill_core)
            continue;

        if (!wait_send_start_decode && status.id / tp_size >= prefill_core)
            continue;

        for (int i = 0; i < status.batchInfo.size(); i++) {
            auto stage = status.batchInfo[i];
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
                Msg m = Msg(false, MSG_TYPE::S_DATA, ++total_pkg, status.id, 0,
                            status.id, M_D_DATA, d);
                m.source_ = GRID_SIZE;
                m.roofline_packets_ = pkg_num;
                q[index].push(m);
            } else {
                for (int j = 1; j <= pkg_num; j++) {
                    sc_bv<M_D_DATA> d(0x1);

                    Msg m =
                        Msg(false, MSG_TYPE::S_DATA, j + total_pkg, status.id,
                            M_D_DATA * (j - 1), status.id, M_D_DATA, d);
                    m.source_ = GRID_SIZE;
                    m.roofline_packets_ = 1;
                    q[index].push(m);
                }

                total_pkg += pkg_num;
            }
        }

        sc_bv<M_D_DATA> d(0x1);
        q[index].push(Msg(true, MSG_TYPE::S_DATA, ++total_pkg, status.id, 0,
                          status.id, 1, d));

        LOG_DEBUG(NETWORK) << "Config helper -> START data -> " << status.id;
    }

    wait_send_start_prefill = wait_send_start_decode = false;
}

void config_helper_pds::iter_done(PD_JOB type) {
    // 按照coreStatus更新requestRecords，理论来说只要获取所有core的batch的req_id即可
    // 如果其中有DECODE done的话就额外更新一次
    // 只有最后一个stage的core才能够更新
    vector<Msg> done_msg;
    if (type == JOB_PREFILL)
        done_msg = g_done_msg_p;
    else if (type == JOB_DECODE)
        done_msg = g_done_msg_d;

    for (auto msg : done_msg) {
        int id = msg.source_ / tp_size;
        // 不是prefil 和 decoding 最后一个核发过来的
        // 比如最开始几拍，可能只有前面几个stage core 会有 done 信号
        if (id < prefill_core && stage_index[id] != prefill_stage ||
            id >= prefill_core && stage_index[id] != decode_stage)
            continue;

        auto &status = coreStatus[id];
        int stage_count = 0;
        for (auto &stage : status.batchInfo) {
            auto &record = requestRecords[stage.req_id];
            switch (record.phase) {
            case PREFILL:
                if (++record.prefill_counter == record.prefill_iters) {
                    stage.type = record.phase = DECODE;
                    token_record[record.id].push_back(
                        sc_time_stamp().to_double());
                    stage.token_num = 1;
                    req_decode.push(stage.req_id);
                    if (!busy_d)
                        wait_schedule_d = true;
                }
                break;
            case DECODE:
                record.decode_counter++;
                // 记录decoding 的某一个token的时间
                token_record[record.id].push_back(sc_time_stamp().to_double());
                // decoding 阶段结束
                if (record.decode_counter >= (2) / (eof_chance)) {
                    stage.type = record.phase = PD_DONE;

                    if (++decode_done == requestRecords.size()) {
                        LOG_INFO(SYSTEM) << "All requests finished";

                        ofstream file("token_records.txt", ios::app);
                        if (!file.is_open()) {
                            LOG_ERROR(config_helper_pds.cpp)
                                << "Failed to open token_records.txt";
                            return;
                        }

                        // 设置输出格式，避免科学计数法
                        file << fixed
                             << setprecision(
                                    6); // 设置小数点后6位精度，可根据需要调整

                        file << "*"
                             << "*\n";
                        for (int i = 0; i < token_record.size(); i++) {
                            file << "Request " << i << ": \n";
                            for (int j = 0; j < token_record[i].size(); j++) {
                                file << "Token " << j << ": "
                                     << token_record[i][j] << "\n";
                            }
                        }

                        file << "\n\n";
                        file.close();
                        LOG_INFO(CATCH_TEST) << "Catch test finished";
                        sc_stop();
                    }
                }
                break;
            }

            stage_count++;
        }
    }
    // 无论chunk prefil 是否结束，当前的p_busy 都为false
    if (type == JOB_PREFILL)
        busy_p = false;
    else if (type == JOB_DECODE)
        busy_d = false;
}

void config_helper_pds::iter_start(PD_JOB type) {
    if (type == JOB_PREFILL && busy_p || type == JOB_DECODE && busy_d)
        return;

    // 为每一个核进行schedule，如果这个核不是第一个stage，则复制前一个stage上一个iter的任务
    vector<pair<int, vector<Stage>>> temp_stage;

    if (type == JOB_PREFILL) {
        for (auto status : coreStatus) {
            int id = status.id / tp_size;
            if (id >= prefill_core)
                continue;
            // prefill 和 decoding 中分别的stage id
            // 如果都不是第一个
            if (stage_index[id] != 1)
                temp_stage.push_back(
                    make_pair(id, coreStatus[id - 1].batchInfo));
            else {
                // 为stage1核分配任务，如果是prefill核，则只能做prefill任务
                int done = 0;
                vector<Stage> new_stage_1;

                // 优先做上个iter没有做完的prefill任务
                for (auto stage : coreStatus[id].batchInfo) {
                    auto &record = requestRecords[stage.req_id];
                    if (record.prefill_distribute < record.prefill_iters) {
                        record.prefill_distribute++;
                        new_stage_1.push_back(stage);
                        done++;
                    }
                }

                // 最后一个阶段的prefill是否完成，于第一阶段的prefill核没有关系，直接跳过

                // 如果还能放进来新的 prefill， 说明上一个stage已经有prefill 的
                // req都被放完了
                if (done < batch_size) {
                    for (auto &req : requestRecords) {
                        sc_core::sc_time arv_time(req.arrival_time,
                                                  sc_core::SC_NS);
                        if (req.phase == PREFILL &&
                            req.prefill_distribute < req.prefill_iters) {
                            req.prefill_distribute++;
                            new_stage_1.push_back(
                                Stage(req.id, PREFILL,
                                      req.seq_len / req.prefill_iters));

                            if (++done == batch_size)
                                break;
                        } else if (req.phase == UNTOUCHED &&
                                   arv_time <= sc_time_stamp()) {
                            new_stage_1.push_back(
                                Stage(req.id, PREFILL,
                                      req.seq_len / req.prefill_iters));
                            req.phase = PREFILL;
                            req.prefill_distribute++;

                            if (++done == batch_size)
                                break;
                        }
                    }
                }

                temp_stage.push_back(make_pair(id, new_stage_1));
            }
        }

        wait_schedule_p = false;
    }

    else if (type == JOB_DECODE) {
        for (auto status : coreStatus) {
            int id = status.id / tp_size;
            if (id < prefill_core)
                continue;

            if (stage_index[id] != 1)
                temp_stage.push_back(
                    make_pair(id, coreStatus[id - 1].batchInfo));
            else {
                // 为stage1核分配任务，如果是decode核，则只能做decode任务
                int credit = 0;
                vector<Stage> new_stage_1;

                // 优先看从最后一个阶段下来的任务
                for (auto stage : coreStatus[id + decode_stage - 1].batchInfo) {
                    if (stage.type == PD_DONE)
                        continue;

                    if (credit < HW_CORE_CREDIT) {
                        credit += 1;
                        new_stage_1.push_back(stage);
                    } else {
                        idle_decode[(id - prefill_core) / decode_stage].push(
                            stage.req_id);
                    }
                }

                // 如果此时还有空余，则查看是否有等待队列中的decode
                auto &waiting_list =
                    idle_decode[(id - prefill_core) / decode_stage];
                while (waiting_list.size() && credit < HW_CORE_CREDIT) {
                    int req_id = waiting_list.front();
                    waiting_list.pop();
                    credit += 1;
                    new_stage_1.push_back(Stage(req_id, DECODE, 1));
                }

                // 最后检查是否有新转为decode的请求
                while (req_decode.size() && credit < HW_CORE_CREDIT) {
                    int req_id = req_decode.front();
                    req_decode.pop();
                    credit += 1;
                    new_stage_1.push_back(Stage(req_id, DECODE, 1));
                }

                temp_stage.push_back(make_pair(id, new_stage_1));
            }
        }

        wait_schedule_d = false;
    }

    // 统一更新所有的batchInfo，生成原语
    bool complete_idle = true;
    vector<Msg> temp_buffer;

    LOG_DEBUG(SCHEDULE) << "Schedule for this iteration";
    for (auto pair : temp_stage) {
        auto &status = coreStatus[pair.first];
        status.batchInfo = pair.second;

        LOG_DEBUG(SCHEDULE) << "  Core " << status.id;
        for (auto stage : status.batchInfo) {
            complete_idle = false;

            LOG_DEBUG(SCHEDULE)
                << "    Request id: " << stage.req_id
                << ", type: " << stage.type << ", finished iters: "
                << ((requestRecords[stage.req_id].phase == PREFILL)
                        ? requestRecords[stage.req_id].prefill_counter
                        : requestRecords[stage.req_id].decode_counter);
        }

        generate_prims(status.id, temp_buffer);
    }

    if (complete_idle) {
        // 如果当前iter没有任何core有工作，则不发放config
        if (type == JOB_PREFILL) {
            busy_p = false;
            wait_schedule_p = true;
        } else if (type == JOB_DECODE) {
            busy_d = false;
            wait_schedule_d = true;
        }

        LOG_DEBUG(SCHEDULE) << "Complete idle";
        temp_config.clear();
    } else {
        for (auto msg : temp_buffer)
            temp_config.push_back(msg);

        if (type == JOB_PREFILL)
            busy_p = true;
        else if (type == JOB_DECODE)
            busy_d = true;
    }
}

void config_helper_pds::printSelf() {}

void config_helper_pds::generate_prims(int i, vector<Msg> &temp_buffer) {
    // 一个iter中有stage个core参与执行，id 1要流向id end，id end要传回id 1
    // core中原语为单个corejob，需要配置收发规则
    auto status = coreStatus[i / tp_size];

    int B = 1, NH = heads, T = 0, C = heads * head_size;
    bool exist_prefill = false;
    for (auto stage : status.batchInfo) {
        auto record = requestRecords[stage.req_id];
        switch (stage.type) {
        case PREFILL:
            T += record.seq_len / record.prefill_iters;
            exist_prefill = true;
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

    if (status.job_type == JOB_PREFILL) {
        for (auto &j : json_template_p) {
            CoreConfig core = j;
            template_cores.push_back(core);
        }
    } else {
        for (auto &j : json_template_d) {
            CoreConfig core = j;
            template_cores.push_back(core);
        }
    }

    // 对于tp组的第一个核，标记输出标签
    string output_label = "";

    // 为每一个核做相同的原语生成
    // 为每一个work生成前后的send和recv原语
    for (int core_id = i; core_id < i + tp_size; core_id++) {
        int index = core_id / GRID_X;
        int prim_seq = 0;

        // 每个核生成一个set_batch
        // 如果本迭代没有工作，且不为tp组的第一个核，则作为最后一个原语
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
                if (status.batchInfo.size()) {
                    for (int p = 0; p < work.prims.size(); p++) {
                        auto prim = work.prims[p];
                        PrimBase *set_addr =
                            PrimFactory::getInstance().createPrim("Set_addr");
                        auto label = set_addr->prim_context->datapass_label_;

                        if (prim->prim_type & COMP_PRIM) {
                            for (int i = 0; i < MAX_SPLIT_NUM; i++)
                                label->indata[i] =
                                    prim->prim_context->datapass_label_
                                        ->indata[i];
                            label->outdata =
                                prim->prim_context->datapass_label_->outdata;
                        }

                        auto segments = set_addr->serialize();
                        for (int seg = 0; seg < segments.size(); seg++)
                            temp_config.push_back(Msg(
                                false, MSG_TYPE::CONFIG, ++prim_seq, core_id,
                                seg == segments.size() - 1, segments[seg]));

                        segments = prim->serialize();
                        for (int seg = 0; seg < segments.size(); seg++)
                            temp_config.push_back(Msg(
                                false, MSG_TYPE::CONFIG, ++prim_seq, core_id,
                                seg == segments.size() - 1, segments[seg]));

                        if (w == template_cores[core_id - i].worklist.size() &&
                            p == work.prims.size() - 1 && i % tp_size == 0)
                            output_label = label->outdata;
                    }
                }

                // 需要计算send_data的发送包裹数，首先找到这个work的最后一个计算原语
                CompBase *last_comp = (CompBase *)work.prims.back();
                if (tp_size == 1)
                    continue;

                // 发送原语，遵循work中的cast
                for (auto ca : work.cast) {
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

                    // 如果为最后一个work，且不为tp组第一个核，则为最后一条原语
                    temp_config.push_back(Msg(
                        core_id != i &&
                            w ==
                                template_cores[core_id - i].worklist.size() - 1,
                        MSG_TYPE::CONFIG, ++prim_seq, core_id,
                        send_data->serialize()[0]));
                }
            }
        } else if (!(core_id % tp_size)) {
            // 如果本迭代没有工作，且为tp组第一个核，则需要加入RECV_START空转一轮
            add_recv(prim_seq, true, core_id, 1, core_id);
        }

        // 处理数据流向下一个cores
        // 这里只有tp_group的第一个核才需要发送
        if (!(core_id % tp_size)) {
            int group_i = core_id / tp_size;
            int send_dest = group_i + 1;
            if (group_i >= prefill_core) {
                // prefill core只需要向下加一即可，只有decode core需要进行处理
                if (send_dest >= decode_core + prefill_core)
                    send_dest -= decode_stage;
                else if (stage_index[send_dest] == 1)
                    send_dest -= decode_stage;
            }
            send_dest *= tp_size;
            int recv_source = group_i - 1;
            if (group_i >= prefill_core) {
                // 当为decode core的时候，决定发送方的编号
                if (recv_source < prefill_core)
                    recv_source += decode_stage;
                else if (stage_index[recv_source] == decode_stage)
                    recv_source += decode_stage;
            } else {
                // 当为prefill core的时候，决定发送方的编号
                if (recv_source < 0)
                    recv_source += prefill_stage;
                else if (stage_index[recv_source] == prefill_stage)
                    recv_source += prefill_stage;
            }
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

            if (core_id / tp_size < prefill_core &&
                stage_index[core_id / tp_size] == 1) {
                // 如果是第一个核，则只发不收
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_req->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, recv_ack->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_data->serialize()[0]));
            } else if (core_id / tp_size < prefill_core &&
                       stage_index[core_id / tp_size] == prefill_stage) {
                // 如果是prefill最后一个核，则只收不发
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id,
                                          recv_data_2->serialize()[0]));
            } else if (core_id / tp_size >= prefill_core &&
                       stage_index[core_id / tp_size] == 1) {
                // 如果是decode的第一个核，则先发后收
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_req->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, recv_ack->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_data->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id,
                                          recv_data_2->serialize()[0]));
            } else {
                // 其余的核，统一先收后发
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id,
                                          recv_data_2->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_req->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, recv_ack->serialize()[0]));
                temp_buffer.push_back(Msg(false, MSG_TYPE::CONFIG, ++prim_seq,
                                          core_id, send_data->serialize()[0]));
            }

            // tp组的第一个核需要向memInterface发送DONE信号
            PrimBase *send_done = new Send_prim(SEND_TYPE::SEND_DONE);
            g_prim_stash.push_back(send_done);
            Msg m = Msg(true, MSG_TYPE::CONFIG, ++prim_seq, core_id,
                        send_done->serialize()[0]);
            m.refill_ = false;
            temp_buffer.push_back(m);
        }
    }
}

void config_helper_pds::parse_ack_msg(Event_engine *event_engine, int flow_id,
                                      sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting_Recv_Ack", "B",
                            Trace_event_util());

    for (auto m : g_temp_ack_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- ACK <- " << cid << ", type "
                           << coreStatus[cid / tp_size].job_type;

        if (coreStatus[cid / tp_size].job_type == JOB_PREFILL) {
            g_recv_ack_cnt_p++;
            LOG_DEBUG(NETWORK) << "Total " << g_recv_ack_cnt_p << " / "
                               << prefill_core * tp_size;
        } else if (coreStatus[cid / tp_size].job_type == JOB_DECODE) {
            g_recv_ack_cnt_d++;
            LOG_DEBUG(NETWORK) << "Total " << g_recv_ack_cnt_d << " / "
                               << decode_core * tp_size;
        }
    }

    LOG_INFO(NETWORK) << "g_recv_ack_cnt_p: " << g_recv_ack_cnt_p
                      << ", prefill_core * tp_size: " << prefill_core * tp_size;
    LOG_INFO(NETWORK) << "g_recv_ack_cnt_d: " << g_recv_ack_cnt_d
                      << ", decode_core * tp_size: " << decode_core * tp_size;

    g_temp_ack_msg.clear();
    // wait(sc_core::sc_time(10, sc_core::SC_NS));
    event_engine->add_event(this->name(), "Waiting Recv Ack", "E",
                            Trace_event_util(), sc_time(2, SC_NS));

    if (g_recv_ack_cnt_p >= prefill_core * tp_size) {
        g_recv_ack_cnt_p = 0;
        wait_send_start_prefill = true;
        notify_event->notify(SC_ZERO_TIME);
    }

    if (g_recv_ack_cnt_d >= decode_core * tp_size) {
        g_recv_ack_cnt_d = 0;
        wait_send_start_decode = true;
        notify_event->notify(SC_ZERO_TIME);
    }
}

void config_helper_pds::parse_done_msg(Event_engine *event_engine,
                                       sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting_Core_busy", "B",
                            Trace_event_util());

    for (auto m : g_temp_done_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- DONE <- " << cid
                           << ", type: " << coreStatus[cid / tp_size].job_type;

        if (coreStatus[cid / tp_size].job_type == JOB_PREFILL) {
            g_recv_done_cnt_p++;
            LOG_DEBUG(NETWORK) << "  Total " << g_recv_done_cnt_p << " / "
                               << prefill_core * tp_size;
            g_done_msg_p.push_back(m);
        } else if (coreStatus[cid / tp_size].job_type == JOB_DECODE) {
            g_recv_done_cnt_d++;
            LOG_DEBUG(NETWORK)
                << "  Total " << g_recv_done_cnt_d << " / " << decode_core;
            g_done_msg_d.push_back(m);
        }
    }
    g_temp_done_msg.clear();
    event_engine->add_event(this->name(), "Waiting_Core_busy", "E",
                            Trace_event_util(), sc_time(2, SC_NS));

    if (g_recv_done_cnt_p >= prefill_core) {
        iter_done(JOB_PREFILL);

        for (PrimBase *p : g_prim_stash) {
            delete p;
        }
        g_prim_stash.clear();

        g_done_msg_p.clear();
        g_recv_done_cnt_p = 0;
        wait_schedule_p = true;
        notify_event->notify(CYCLE, SC_NS);
    }

    if (g_recv_done_cnt_d >= decode_core) {
        iter_done(JOB_DECODE);

        for (PrimBase *p : g_prim_stash) {
            delete p;
        }
        g_prim_stash.clear();

        g_done_msg_d.clear();
        g_recv_done_cnt_d = 0;
        wait_schedule_d = true;
        notify_event->notify(CYCLE, SC_NS);
    }
}

void config_helper_pds::set_global_vars(int T, int tp_size) {
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