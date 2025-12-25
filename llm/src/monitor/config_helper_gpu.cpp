#include "monitor/config_helper_gpu.h"
#include "common/config.h"
#include "utils/config_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

config_helper_gpu::config_helper_gpu(string filename, int config_chip_id) {
    LOG_INFO(CONFIG) << "Loading config file " << filename;

    json j;
    // plot_dataflow(filename);
    ifstream jfile(filename);
    jfile >> j;

    // 收集相关参数
    auto config_vars = j["vars"];
    for (auto var : config_vars.items()) {
        vtable.push_back(make_pair(var.key(), var.value()));
    }

    auto config_streams = j["chips"][0]["streams"];
    if (config_streams.size() != 1) {
        LOG_ERROR(SYSTEM) << "More than 1 stream is not supported";
    }

    for (int i = 0; i < config_streams.size(); i++) {
        StreamConfig stream = config_streams[i];
        streams.push_back(stream);
    }

    // 将stream的原语放入coreconfigs中
    for (int i = 0; i < GRID_SIZE; i++) {
        CoreConfig core;
        core.id = i;
        core.prim_copy = -1;
        core.send_global_mem = -1;
        core.loop = 1;

        coreconfigs.push_back(core);
    }

    // 处理stream的原语
    for (int i = 0; i < streams.size(); i++) {
        auto stream = streams[i];
        auto prims = stream.prims;

        for (int j = 0; j < prims.size(); j++) {
            GpuBase *prim = (GpuBase *)prims[j];
            int sms = prim->req_sm;

            int cycles = sms / GRID_SIZE;
            int rest = sms - cycles * GRID_SIZE;
            for (int c = 0; c < GRID_SIZE; c++) {
                auto &core = coreconfigs[c];

                CoreJob new_job(1, c, cycles + (c < rest));
                auto prim_copy = prim->clone();
                new_job.prims.push_back(prim_copy);
                core.worklist.push_back(new_job);
            }
        }
    }

    // 处理收发原语（启动）
    for (int i = 0; i < coreconfigs.size(); i++) {
        generate_prims(i);
    }

    end_cores = GRID_SIZE;
    pipeline = 1;
    g_recv_ack_cnt = 0;
    g_recv_done_cnt = 0;
    gpu_index = 0;
    done_loop = 0;

    printSelf();
}

void config_helper_gpu::fill_queue_config(queue<Msg> *q) {
    for (auto config : coreconfigs) {
        int index = config.id / GRID_X;
        vector<Msg> single_rep;

        auto recv_weight = new Recv_prim(RECV_TYPE::RECV_WEIGHT,
                                         config.worklist[0].recv_tag, 0);
        q[index].push(Msg(false, MSG_TYPE::CONFIG, 1, config.id,
                          recv_weight->serialize()[0]));

        vector<Stage> batchInfo;
        for (int i = 0; i < GetDefinedParam("B"); i++)
            batchInfo.push_back(Stage(i + 1, PREFILL, GetDefinedParam("T")));

        PrimBase *set_batch = new Set_batch(batchInfo, 1);
        auto segments = set_batch->serialize();
        for (int seg = 0; seg < segments.size(); seg++)
            single_rep.push_back(
                Msg(false, MSG_TYPE::CONFIG, single_rep.size() + 1, config.id,
                    seg == segments.size() - 1, segments[seg]));

        for (auto work : config.worklist) {
            for (auto prim : work.prims_last_loop) {
                auto segments = prim->serialize();
                for (int seg = 0; seg < segments.size(); seg++)
                    single_rep.push_back(Msg(
                        false, MSG_TYPE::CONFIG, single_rep.size() + 1,
                        config.id, seg == segments.size() - 1, segments[seg]));
            }
        }

        for (int i = 0; i < streams[0].loop; i++) {
            for (int j = 1; j <= single_rep.size(); j++) {
                Msg m = single_rep[j - 1];
                m.seq_id_ = j + single_rep.size() * i + 1;

                if (i == streams[0].loop - 1 && j == single_rep.size()) {
                    m.is_end_ = true;
                    m.refill_ = false;
                }

                q[index].push(m);
            }
        }
    }
}

void config_helper_gpu::generate_prims(int i) {
    CoreConfig *c = &coreconfigs[i];

    for (auto &work : c->worklist) {
        // 不向in_loop推入任何原语，只操作last_loop
        work.prims_last_loop.push_back(
            new Recv_prim(RECV_TYPE::RECV_START, work.recv_tag, work.recv_cnt));

        for (auto prim : work.prims) {
            GpuBase *gp = (GpuBase *)prim;
            int sms = gp->req_sm;

            // 只需要看单个原语重复次数
            int repeat = sms / GRID_SIZE + (sms % GRID_SIZE > i);

            for (int r = 0; r < repeat; r++) {
                PrimBase *p = PrimFactory::getInstance().createPrim("Set_addr");
                auto label = p->prim_context->datapass_label_;

                // Set_addr 的label 指向其后面的那条原语
                for (int i = 0; i < MAX_SPLIT_NUM; i++) {
                    label->indata[i] =
                        gp->prim_context->datapass_label_->indata[i];
                }
                label->outdata = gp->prim_context->datapass_label_->outdata;

                // 这里直接推入字符串形式的label，之后会在序列化的时候转化为整形label
                work.prims_last_loop.push_back(p);

                gp->fetch_index = i + r * GRID_SIZE;
                work.prims_last_loop.push_back(prim);
            }
        }

        work.prims_last_loop.push_back(new Send_prim(SEND_TYPE::SEND_DONE));
    }
}

void config_helper_gpu::fill_queue_start(queue<Msg> *q) {
    LOG_INFO(NETWORK) << "Config helper start START data distribution, phase "
                      << gpu_index;

    int sms = ((GpuBase *)(streams[0].prims[gpu_index]))->req_sm;
    for (auto stream : streams) {
        for (auto source : stream.sources) {
            AddrPosKey source_key = AddrPosKey(0, source.second);
            gpu_pos_locator->addPair(source.first, source_key);
        }
    }

    for (int i = 0; i < min(sms, GRID_SIZE); i++) {
        auto config = coreconfigs[i];
        int index = config.id / GRID_X;
        int pkg_index = 0;

        // 这里相当于quick start，实际上也只有第一个原语需要初始数据
        sc_bv<128> d(0x1);
        Msg m = Msg(true, MSG_TYPE::S_DATA, pkg_index + 1, config.id, 0,
                    config.id, 0, d);
        m.source_ = GRID_SIZE;
        q[index].push(m);
    }

    gpu_index++;
}

void config_helper_gpu::printSelf() {}

void config_helper_gpu::parse_ack_msg(Event_engine *event_engine, int flow_id,
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

        // 使用唯一的flow ID替换名称
        std::string flow_name = "flow_" + std::to_string(flow_id);
        event_engine->add_event(this->name(), "Waiting Recv Ack", "f",
                                Trace_event_util(flow_name), sc_time(0, SC_NS),
                                100, "e");
        LOG_INFO(NETWORK) << "Config helper received all ACK";

        g_recv_ack_cnt = 0;
    }
}

void config_helper_gpu::parse_done_msg(Event_engine *event_engine,
                                       sc_event *notify_event) {
    event_engine->add_event(this->name(), "Waiting Core busy", "B",
                            Trace_event_util());

    for (auto m : g_temp_done_msg) {
        int cid = m.source_;
        LOG_DEBUG(NETWORK) << "Config helper <- DONE <- " << cid << ", total "
                           << g_recv_done_cnt + 1 << " / " << coreconfigs.size();

        g_recv_done_cnt++;
        // g_done_msg.push_back(m);
    }
    g_temp_done_msg.clear();
    event_engine->add_event(this->name(), "Waiting Core busy", "E",
                            Trace_event_util());

    auto prim = streams[0].prims[gpu_index - 1];
    auto core_inv = ((GpuBase *)prim)->req_sm;

    if (core_inv >= GRID_SIZE)
        core_inv = GRID_SIZE;
    if (g_recv_done_cnt >= core_inv) {
        LOG_INFO(SYSTEM) << "Work done " << gpu_index << " of "
                         << streams[0].prims.size();

        if (gpu_index == streams[0].prims.size()) {
            gpu_index = 0;
            done_loop++;
            LOG_INFO(SYSTEM)
                << "Loop done " << done_loop << " of " << streams[0].loop;

            for (auto &pair : vtable) {
                if (pair.first == "T")
                    pair.second = 1;
            }

            if (done_loop == streams[0].loop) {
                LOG_INFO(SYSTEM) << "All requests finished";
                LOG_INFO(CATCH_TEST) << "Catch test finished";

                ofstream outfile("simulation_result_gpu.txt", ios::app);
                if (outfile.is_open()) {
                    outfile << "[CATCH TEST] " << sc_time_stamp()
                            << "L1CACHESIZE " << L1CACHESIZE << " L2CACHESIZE "
                            << L2CACHESIZE << " BANDWIDTH "
                            << GPU_DRAM_BANDWIDTH << endl;
                    outfile.close();
                } else {
                    LOG_ERROR(config_helper_gpu.cpp)
                        << "Failed to open simulation_result_gpu.txt";
                }
                sc_stop();
            }
        }

        g_recv_done_cnt = 0;
        notify_event->notify(CYCLE, SC_NS);
    }
}