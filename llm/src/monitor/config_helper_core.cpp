#include "nlohmann/json.hpp"
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <sstream>

#include "common/system.h"
#include "defs/spec.h"
#include "monitor/config_helper_core.h"
#include "prims/base.h"
#include "utils/config_utils.h"
#include "utils/display_utils.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

using json = nlohmann::json;

CoreConfig *config_helper_core::get_core(int id) {
    for (int i = 0; i < coreconfigs.size(); i++) {
        if (coreconfigs[i].id == id)
            return &(coreconfigs[i]);
    }

    LOG_ERROR(config_helper_core.cpp) << "Core " << id << " not found";
    return nullptr;
}

void config_helper_core::printSelf() {}

void config_helper_core::random_core() {
    int o2r[GRID_SIZE];
    int r2o[GRID_SIZE];
    for (int i = 0; i < GRID_SIZE; i++) {
        o2r[i] = r2o[i] = -1;
    }

    std::srand(std::time(nullptr));
    for (auto config : coreconfigs) {
        int id = config.id;
        int rand = 0;
        do {
            rand = std::rand() % GRID_SIZE;
        } while (r2o[rand] != -1);
        o2r[id] = rand;
        r2o[rand] = id;
    }

    // 改写
    for (auto &config : coreconfigs) {
        int oid = config.id;
        config.id = o2r[oid];
        if (config.prim_copy != -1)
            config.prim_copy = o2r[config.prim_copy];

        if (config.send_global_mem != -1)
            config.send_global_mem = o2r[config.send_global_mem];

        for (auto &work : config.worklist) {
            if (work.recv_tag < GRID_SIZE)
                work.recv_tag = o2r[oid];
            for (auto &cast : work.cast) {
                if (cast.tag < GRID_SIZE && cast.dest >= 0)
                    cast.tag = o2r[cast.tag];
                if (cast.dest < GRID_SIZE && cast.dest >= 0)
                    cast.dest = o2r[cast.dest];
            }
        }

        config.printSelf();
    }

    // 改写source
    for (auto &pair : source_info) {
        pair.first = o2r[pair.first];
    }

    // 重新绘图
    set<int> source_ids;
    for (auto &pair : source_info) {
        source_ids.emplace(pair.first);
    }

    unordered_map<int, Display::Core> cores;
    int core_id = 0;
    for (const auto &core_j : coreconfigs) {
        Display::Core core;
        core.id = core_j.id;
        core.x = core.id % GRID_X; // X 坐标
        core.y = core.id / GRID_X; // Y 坐标

        // 提取每个 core 的 dest 信息
        for (const auto &work : core_j.worklist) {
            vector<int> temp_cast;
            for (const auto &cast : work.cast) {
                if (cast.critical) {
                    int d = cast.dest;
                    temp_cast.push_back(1e5 + d);
                } else
                    temp_cast.push_back(cast.dest);
            }

            core.dests.push_back(temp_cast);
        }

        cores[core.id] = core;
        core_id++;
    }

    plot_dataflow(cores, source_ids);
}

config_helper_core::config_helper_core(string filename, int config_chip_id) {
    LOG_INFO(CONFIG) << "Loading config file " << filename;

    plot_dataflow(filename);
    ifstream jfile(filename);
    if (!jfile.is_open()) {
        LOG_ERROR(CONFIG) << "Could not open config file " << filename;
    }

    json j;
    try {
        jfile >> j;
    } catch (const json::parse_error &e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
    }

    // 收集相关参数
    auto config_vars = j["vars"];
    for (auto var : config_vars.items()) {
        vtable.push_back(make_pair(var.key(), var.value()));
    }

    SetParamFromJson(config_vars, "B", &batch_size, 1);
    SetParamFromJson(config_vars, "T", &seq_len, 128);

    auto config_source = j["source"];
    end_count_sources = 0;
    for (auto source : config_source) {
        int source_loop = 0;
        bool is_end = false;
        SetParamFromJson(source, "loop", &source_loop, 1);
        SetParamFromJson(source, "is_end", &is_end, false);

        if (is_end)
            end_count_sources += source_loop;

        for (; source_loop > 0; source_loop--)
            source_info.push_back(
                make_pair(source["dest"], GetDefinedParam(source["size"])));
    }

    auto config_cores = j["chips"][config_chip_id]["cores"];
    for (int i = 0; i < config_cores.size(); i++) {
        // 调用 config_helper_base中的from_json
        CoreConfig core = config_cores[i]; // 这里不直接转化prims
        coreconfigs.push_back(core);
    }

    // if (j.contains("random") && j["random"]) {
    //     random_core();
    // }

    CoreConfigRemap(source_info, coreconfigs);

    SetParamFromJson(j, "pipeline", &pipeline, 1);

    // 检查是否需要复制原语的核，config书写要求：需要重新写明所有work的cast、recv_cnt,数量等同于需要复制的那个核的work数量
    for (int i = 0; i < coreconfigs.size(); i++) {
        if (coreconfigs[i].prim_copy != -1) {
            auto worklist_temp = coreconfigs[i].worklist;
            coreconfigs[i].worklist.clear();
            for (int j = 0; j < worklist_temp.size(); j++) {
                auto prev_job = worklist_temp[j];

                auto target_core = get_core(coreconfigs[i].prim_copy);
                auto target_work = target_core->worklist[j];
                for (int c = 0; c < prev_job.cast.size(); c++) {
                    if (c >= target_work.cast.size()) {
                        target_work.cast.push_back(prev_job.cast[c]);
                        continue;
                    }
                    if (target_work.cast[c].tag == target_work.cast[c].dest ||
                        prev_job.cast[c].tag != prev_job.cast[c].dest)
                        target_work.cast[c].tag = prev_job.cast[c].tag;
                    target_work.cast[c].dest = prev_job.cast[c].dest;
                    target_work.cast[c].loopout = prev_job.cast[c].loopout;
                }
                target_work.recv_cnt = prev_job.recv_cnt;
                if (target_work.recv_tag == coreconfigs[i].prim_copy ||
                    prev_job.recv_tag != coreconfigs[i].id)
                    target_work.recv_tag = prev_job.recv_tag;
                coreconfigs[i].worklist.push_back(target_work);
            }
        }
    }

    end_cores = 0;
    g_recv_ack_cnt = 0;
    g_recv_done_cnt = 0;

    for (int i = 0; i < coreconfigs.size(); i++)
        generate_prims(i);

    // 再去重新填写send的收发地址
    calculate_address(true);
    calculate_address(false);
}

void config_helper_core::fill_queue_config(queue<Msg> *q) {
    for (auto &config : coreconfigs) {
        int index = config.id / GRID_X;

        auto build_msgs = [&](const vector<PrimBase *> &prims,
                              bool adjust_recv = false) {
            vector<Msg> msgs;
            msgs.reserve(prims.size());
            for (auto *prim : prims) {
                if (adjust_recv) {
                    if (auto *recv_prim = dynamic_cast<Recv_prim *>(prim)) {
                        if (recv_prim->type == RECV_TYPE::RECV_START)
                            recv_prim->type = RECV_TYPE::RECV_DATA;
                    }
                }

                auto segments = prim->serialize();
                for (int seg = 0; seg < segments.size(); seg++)
                    msgs.emplace_back(
                        Msg(false, MSG_TYPE::CONFIG, msgs.size() + 1, config.id,
                            seg == segments.size() - 1, segments[seg]));
            }
            return msgs;
        };

        // 三类循环消息
        vector<Msg> in_loop, next_loop, last_loop;
        for (auto &work : config.worklist) {
            // 第一个循环接受 start 数据包
            auto tmp_in = build_msgs(work.prims_in_loop);
            // 中间的循环 原本接受start数据包的接受data数据包
            auto tmp_next = build_msgs(work.prims_in_loop, true);
            // 最后一个循环，如果只有一个循环就是最后一个循环
            auto tmp_last = build_msgs(work.prims_last_loop);

            in_loop.insert(in_loop.end(), tmp_in.begin(), tmp_in.end());
            next_loop.insert(next_loop.end(), tmp_next.begin(), tmp_next.end());
            last_loop.insert(last_loop.end(), tmp_last.begin(), tmp_last.end());
        }

        // queue push helper
        int seq_cnt = 1;
        auto push_msg = [&](Msg m) {
            m.seq_id_ = seq_cnt++;
            q[index].push(m);
        };

        // RECV_WEIGHT
        PrimBase *recv_weight = new Recv_prim(RECV_TYPE::RECV_WEIGHT,
                                              config.worklist[0].recv_tag, 0);
        push_msg(Msg(false, MSG_TYPE::CONFIG, 0, config.id,
                     recv_weight->serialize()[0]));

        // Set_batch
        vector<Stage> batchInfo;
        for (int i = 0; i < batch_size; i++)
            batchInfo.emplace_back(i + 1, PREFILL, seq_len);
        PrimBase *set_batch = new Set_batch(batchInfo, pipeline);

        // 主循环，将pipeline视为一种循环
        for (int j = 0; j < pipeline; j++) {
            // 如果 默认的 loop = 1 其实 in_loop 和 next_loop 都不会执行
            // 这里的loop 不为 1 就是 decoding 的数量
            for (int i = 0; i < config.loop - 1; i++) {
                auto segments = set_batch->serialize();
                for (int seg = 0; seg < segments.size(); seg++)
                    push_msg(Msg(false, MSG_TYPE::CONFIG, 0, config.id,
                                 seg == segments.size() - 1, segments[seg]));

                auto &reps = (i == 0) ? in_loop : next_loop;
                for (auto m : reps)
                    push_msg(m);
            }
            // 默认执行最后一个循环
            auto segments = set_batch->serialize();
            for (int seg = 0; seg < segments.size(); seg++)
                push_msg(Msg(false, MSG_TYPE::CONFIG, 0, config.id,
                             seg == segments.size() - 1, segments[seg]));

            for (size_t k = 0; k < last_loop.size(); k++) {
                Msg m = last_loop[k];
                // 最后一个原语， 然后循环重填
                m.refill_ = m.is_end_ =
                    (k + 1 == last_loop.size() && j == pipeline - 1);
                push_msg(m);
            }
        }
    }
}

void config_helper_core::generate_prims(int i) {
    CoreConfig *c = &coreconfigs[i];

    bool is_source = any_of(source_info.begin(), source_info.end(),
                            [&](auto &src) { return src.first == c->id; });

    auto add_recv = [&](vector<PrimBase *> &prims, bool start, int tag,
                        int cnt) {
        prims.push_back(new Recv_prim(
            start ? RECV_TYPE::RECV_START : RECV_TYPE::RECV_DATA, tag, cnt));
    };

    auto add_comps = [&](vector<PrimBase *> &prims,
                         const vector<PrimBase *> &works) {
        for (auto *prim : works) {
            PrimBase *p = PrimFactory::getInstance().createPrim("Set_addr");
            auto label = p->prim_context->datapass_label_;
            if (prim->prim_type & PRIM_TYPE::COMP_PRIM) {
                for (int i = 0; i < MAX_SPLIT_NUM; i++)
                    label->indata[i] =
                        prim->prim_context->datapass_label_->indata[i];
                label->outdata = prim->prim_context->datapass_label_->outdata;
            }
            prims.push_back(p);
            prims.push_back(prim);
        }
    };

    auto add_sends = [&](vector<PrimBase *> &prims, const vector<Cast> &casts,
                         bool loopout) {
        for (auto &ca : casts) {
            // loopout pipeline的最后一个核是发给 0 核还是发给host
            if ((loopout && ca.loopout == FALSE) ||
                (!loopout && ca.loopout == TRUE))
                continue;
            prims.push_back(
                new Send_prim(SEND_TYPE::SEND_REQ, ca.dest, ca.tag));
            prims.push_back(new Recv_prim(RECV_TYPE::RECV_ACK));
            prims.push_back(
                new Send_prim(SEND_TYPE::SEND_DATA, ca.dest, ca.tag));
        }
    };

    for (int w = 0; w < c->worklist.size(); w++) {
        auto &work = c->worklist[w];
        bool is_end = judge_is_end_work(work);
        if (is_end)
            end_cores++;

        // 非最后循环
        // loop = 1 的 时候只有last_loop 循环
        add_recv(work.prims_in_loop, (is_source && w == 0), work.recv_tag,
                 work.recv_cnt);
        add_comps(work.prims_in_loop, work.prims);
        add_sends(work.prims_in_loop, work.cast, false);

        // 最后循环
        add_recv(work.prims_last_loop, (is_source && w == 0 && c->loop == 1),
                 work.recv_tag, work.recv_cnt);
        add_comps(work.prims_last_loop, work.prims);

        if (is_end) {
            work.prims_last_loop.push_back(new Send_prim(SEND_TYPE::SEND_DONE));
            // work.prims_last_loop.push_back(
            //     PrimFactory::getInstance().createPrim("Clear_sram"));
            continue;
        }

        // 现在不会有不是is_end 的核心有 loopout

        add_sends(work.prims_last_loop, work.cast, true);

        // if (w == c->worklist.size() - 1) {
        //     work.prims_last_loop.push_back(
        //         PrimFactory::getInstance().createPrim("Clear_sram"));
        // }
    }
}

void config_helper_core::calculate_address(bool do_loop) {
    // 自动设置 send 和 receive 的地址
    for (int i = 0; i < coreconfigs.size(); i++) {
        for (auto &work : coreconfigs[i].worklist) {
            // 遍历每一个核中的send原语
            vector<PrimBase *> *v = nullptr;
            if (do_loop)
                v = &(work.prims_in_loop);
            else
                v = &(work.prims_last_loop);

            int output_size = 0;
            int output_offset = 0;
            int index = 0;
            string output_label = "";

            if (!do_loop && judge_is_end_work(work))
                continue; // 汇节点

            // 拿到这个corejob的output size
            for (int j = v->size() - 1; j >= 0; j--) {
                auto p = (*v)[j];

                if (p->prim_type & PRIM_TYPE::COMP_PRIM) {
                    CompBase *cp = (CompBase *)p;
                    output_size = cp->out_size;
                    // output_offset = cp->out_offset;
                    output_label = cp->prim_context->datapass_label_->outdata;
                    break;
                }
            }

            vector<string> output_label_split;
            stringstream ss(output_label);
            string word;

            while (ss >> word)
                output_label_split.push_back(word);

            for (auto &prim : (*v)) {
                if (typeid(*prim) == typeid(Send_prim)) {
                    Send_prim *temp = (Send_prim *)prim;
                    if (temp->type != SEND_DATA)
                        continue;

                    CalculatePacketNum(output_size, work.cast[index].weight,
                                       (prim->datatype ? 2 : 1),
                                       temp->max_packet, temp->end_length);

                    temp->output_label = output_label_split.size() == 1
                                             ? output_label_split[0]
                                             : output_label_split[index];
                    index++;
                }
            }
        }
    }
}

void config_helper_core::fill_queue_start(queue<Msg> *q) {
    LOG_INFO(NETWORK) << "Config helper start START data distribution";

    for (int pipe = 0; pipe < pipeline; pipe++) {
        for (auto source : source_info) {
            // 从这里看pipeline 和 source_loop 的功能是一样的
            // start 数据包一次性都下发完成 但是可以分阶段使用
            // source loop 的循环是靠prim refill 实现的
            int i = source.first;
            int size = source.second;

            int index = i / GRID_X;
            int send_offset = 0;
            for (auto config : coreconfigs) {
                if (config.id == i)
                    send_offset =
                        ((NpuBase *)config.worklist[0].prims[0])->inp_offset;
            }

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
                Msg m = Msg(true, MSG_TYPE::S_DATA, 1, i, send_offset, i,
                            length, d);
                m.source_ = GRID_SIZE;
                m.roofline_packets_ = pkg_num;
                q[index].push(m);
            } else {
                for (int j = 1; j <= pkg_num; j++) {
                    sc_bv<M_D_DATA> d(0x1);
                    int length = M_D_DATA;
                    bool is_end_packet = j == pkg_num;
                    if (is_end_packet)
                        length =
                            size * sizeof(float) - M_D_DATA * (pkg_num - 1);

                    Msg m = Msg(j == pkg_num, MSG_TYPE::S_DATA, j, i,
                                send_offset + M_D_DATA * (j - 1), i, length, d);
                    m.source_ = GRID_SIZE;
                    m.roofline_packets_ = 1;
                    q[index].push(m);
                }
            }
        }
    }
}

void config_helper_core::parse_ack_msg(Event_engine *event_engine, int flow_id,
                                       sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting Recv Ack", "B",
                            Trace_event_util());

    for (auto m : g_temp_ack_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- ACK <- " << cid << ", total "
                           << g_recv_ack_cnt + 1 << "/" << coreconfigs.size();

        g_recv_ack_cnt++;
    }

    g_temp_ack_msg.clear();
    event_engine->add_event(this->name(), "Waiting Recv Ack", "E",
                            Trace_event_util());


    if (g_recv_ack_cnt >= coreconfigs.size()) {
        notify_event->notify(CYCLE, SC_NS);
        g_recv_ack_cnt = 0;

        // 使用唯一的flow ID替换名称
        std::string flow_name = "flow_" + std::to_string(flow_id);
        event_engine->add_event(this->name(), "Waiting Recv Ack", "f",
                                Trace_event_util(flow_name), sc_time(0, SC_NS),
                                100, "e");
        LOG_INFO(NETWORK) << "Config helper received all ACK";
    }
}

void config_helper_core::parse_done_msg(Event_engine *event_engine,
                                        sc_event *notify_event) {
    notify_event = nullptr; // 无需触发任何信号
    event_engine->add_event(this->name(), "Waiting Core busy", "B",
                            Trace_event_util());

    for (auto m : g_temp_done_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- DONE <- " << cid << ", total "
                           << g_recv_done_cnt + 1 << " / "
                           << end_cores * pipeline * max(1, end_count_sources);

        g_recv_done_cnt++;
        // g_done_msg.push_back(m);
    }
    g_temp_done_msg.clear();
    event_engine->add_event(this->name(), "Waiting Core busy", "E",
                            Trace_event_util());

    if (g_recv_done_cnt >= end_cores * pipeline * max(1, end_count_sources)) {
        LOG_INFO(SYSTEM) << "All requests finished";
        LOG_INFO(SYSTEM) << "  end_cores: " << end_cores
                         << ", total pipe: " << pipeline
                         << ", end_count_sources: " << end_count_sources;

        g_recv_done_cnt = 0;
        LOG_INFO(CATCH_TEST) << "Catch test finished";
        sc_stop();
    }
}