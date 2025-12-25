#include "common/system.h"
#include "systemc.h"
#include <deque>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <typeinfo>

#include "defs/const.h"
#include "defs/global.h"
#include "link/nb_global_memif_v2.h"
#include "memory/dram/GPUNB_DcacheIF.h"
#include "memory/gpu/GPU_L1L2_Cache.h"
#include "memory/sram/Mem_access_unit.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "prims/moe_prims.h"
#include "prims/norm_prims.h"
#include "prims/pd_prims.h"
#include "trace/Event_engine.h"
#include "utils/memory_utils.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"
#include "workercore/workercore.h"

using namespace std;

// workercore
WorkerCore::WorkerCore(const sc_module_name &n, int s_cid,
                       Event_engine *event_engine, string dram_config_name)
    : sc_module(n), cid(s_cid), event_engine(event_engine) {
    // systolic_config = new HardwareTaskConfig();
    // other_config = new HardwareTaskConfig();
    dcache = new DCache(sc_gen_unique_name("dcache"), cid, (int)cid / GRID_X,
                        (int)cid % GRID_X, this->event_engine, dram_config_name,
                        "../DRAMSys/configs");

    LOG_DEBUG(SYSTEM) << "Core " << cid << " dram config path "
                      << dram_config_name;
    LOG_DEBUG(SYSTEM)
        << " max address "
        << dcache->dramSysWrapper->dramsys->getAddressDecoder().maxAddress();

    auto sram_bitw = GetCoreHWConfig(cid)->sram_bitwidth;
    ram_array = new DynamicBandwidthRamRow<sc_bv<SRAM_BITWIDTH>, SRAM_BANKS>(
        sc_gen_unique_name("ram_array"), 0,
        HW_SRAM_SIZE * 8 / sram_bitw / SRAM_BANKS, SIMU_READ_PORT,
        SIMU_WRITE_PORT, BANK_PORT_NUM + SRAM_BANKS, BANK_PORT_NUM,
        BANK_HIGH_READ_PORT_NUM, event_engine);
    temp_ram_array =
        new DynamicBandwidthRamRow<sc_bv<SRAM_BITWIDTH>, SRAM_BANKS>(
            sc_gen_unique_name("temp_ram_array"), 0,
            HW_SRAM_SIZE * 8 / sram_bitw / SRAM_BANKS, SIMU_READ_PORT,
            SIMU_WRITE_PORT, BANK_PORT_NUM + SRAM_BANKS, BANK_PORT_NUM,
            BANK_HIGH_READ_PORT_NUM, event_engine);

    executor = new WorkerCoreExecutor(sc_gen_unique_name("workercore-exec"),
                                      cid, this->event_engine);
    // executor->MaxDramAddr =
    //     dcache->dramSysWrapper->dramsys->getAddressDecoder().maxAddress();
    executor->MaxDramAddr =
        dcache->dramSysWrapper->dramsys->getMemSpec().memorySizeBytes;
    executor->defaultDataLength =
        dcache->dramSysWrapper->dramsys->getMemSpec().defaultBytesPerBurst;
    if (SYSTEM_MODE != SIM_GPU && SYSTEM_MODE != SIM_GPU_PD) {
        GPU_DRAM_ALIGNED = executor->defaultDataLength;
    }
    assert(dataset_words_per_tile <
           dcache->dramSysWrapper->dramsys->getMemSpec().memorySizeBytes);
    g_dram_kvtable[cid] =
        new DramKVTable(executor->MaxDramAddr, (uint64_t)50 * 1024 * 1024, 20);
#if USE_NB_DRAMSYS == 1
    executor->nb_dcache_socket->socket.bind(dcache->socket);
#else
    executor->dcache_socket->isocket.bind(dcache->socket);
#endif
    executor->mem_access_port->mem_read_port(*ram_array);
    executor->mem_access_port->mem_write_port(*ram_array);
    executor->high_bw_mem_access_port->mem_read_port(*ram_array);

    executor->temp_mem_access_port->mem_read_port(*temp_ram_array);
    executor->temp_mem_access_port->mem_write_port(*temp_ram_array);
    executor->high_bw_temp_mem_access_port->mem_read_port(*temp_ram_array);
}

WorkerCore::~WorkerCore() {
    delete executor;
    delete dcache;
    delete ram_array;
    delete temp_ram_array;
}

// workercore executor
WorkerCoreExecutor::WorkerCoreExecutor(const sc_module_name &n, int s_cid,
                                       Event_engine *event_engine)
    : sc_module(n), cid(s_cid), event_engine(event_engine) {
    prim_refill = false;

    SC_THREAD(catch_channel_avail_i);
    sensitive << channel_avail_i.pos();
    dont_initialize();

    SC_THREAD(catch_data_sent_i);
    sensitive << data_sent_i.pos();
    dont_initialize();

    //  控制信道相关线程
    SC_THREAD(catch_ctrl_channel_avail_i);
    sensitive << ctrl_channel_avail_i.pos();
    dont_initialize();

    SC_THREAD(catch_ctrl_sent_i);
    sensitive << ctrl_sent_i.pos();
    dont_initialize();

    SC_THREAD(poll_ctrl_buffer_i);

    SC_THREAD(next_write_clear);
    sensitive << ev_next_write_clear;
    dont_initialize();

    SC_THREAD(switch_prim_block);
    sensitive << ev_block;

    SC_THREAD(worker_core_execute);

    SC_THREAD(send_logic);
    sensitive << ev_send;
    dont_initialize();

    SC_THREAD(send_para_logic);
    sensitive << ev_para_send;
    dont_initialize();

    SC_THREAD(recv_logic);
    sensitive << ev_recv;
    dont_initialize();

    SC_THREAD(send_helper);
    sensitive << ev_send_helper;
    dont_initialize();

    SC_THREAD(task_logic);
    sensitive << ev_comp;
    dont_initialize();

    SC_THREAD(req_logic);
    // req_logic 只处理 REQUEST 消息，只需要监听：
    // 1. ev_recv_msg_type_[REQUEST] - 收到 REQUEST 时触发（数据信道或控制信道都会触发）
    // 2. ev_prim_recv_notice - 执行 recv_data 原语时触发，需要检查是否有匹配的 REQUEST
    sensitive << ev_recv_msg_type_[MSG_TYPE::REQUEST] << ev_prim_recv_notice;
    dont_initialize();

    SC_THREAD(poll_buffer_i);
    sram_addr = new int(0);

    // 初始化PrimCoreContext
    core_context = make_shared<PrimCoreContext>(cid);

    send_done = true;
    send_last_packet = false;

    start_global_mem_event = new sc_event();
    end_global_mem_event = new sc_event();
    start_nb_dram_event = new sc_event();
    start_nb_gpu_dram_event = new sc_event();
    start_sram_event = new sc_event();
    end_sram_event = new sc_event();

    end_nb_dram_event = new sc_event();
    end_nb_gpu_dram_event = new sc_event();

    sram_writer = new SRAMWriteModule("sram_writer", end_sram_event);
#if USE_NB_DRAMSYS == 1
    nb_dcache_socket =
        new NB_DcacheIF(cid, sc_gen_unique_name("nb_dcache"),
                        start_nb_dram_event, end_nb_dram_event, event_engine);
#else
    dcache_socket = new DcacheCore(sc_gen_unique_name("dcache"), event_engine);
#endif
#if USE_L1L2_CACHE == 1
    core_lv1_cache = new L1Cache(("l1_cache_" + to_string(cid)).c_str(), cid,
                                 L1CACHESIZE, L1CACHELINESIZE, 4, 8);
    gpunb_dcache_if = new GPUNB_dcacheIF(sc_gen_unique_name("nb_dcache_if"),
                                         cid, start_nb_gpu_dram_event,
                                         end_nb_gpu_dram_event, event_engine);
#else
#endif
    mem_access_port = new mem_access_unit(sc_gen_unique_name("mem_access_unit"),
                                          event_engine);
    high_bw_mem_access_port = new high_bw_mem_access_unit(
        sc_gen_unique_name("high_bw_mem_access_unit"), event_engine);
    temp_mem_access_port = new mem_access_unit(
        sc_gen_unique_name("temp_mem_access_unit"), event_engine);
    high_bw_temp_mem_access_port = new high_bw_mem_access_unit(
        sc_gen_unique_name("high_bw_temp_mem_access_unit"), event_engine);
}

void WorkerCoreExecutor::init_global_mem() {
    nb_global_mem_socket = new NB_GlobalMemIF(
        sc_gen_unique_name("nb_global_mem"), start_global_mem_event,
        end_global_mem_event, event_engine);
}

void WorkerCoreExecutor::end_of_elaboration() {
    // 在构造函数之后设置信号的初始值
    data_sent_o.write(false);
    core_busy_o.write(false);
    ctrl_sent_o.write(false);
    ctrl_core_busy_o.write(false);
}

void WorkerCoreExecutor::worker_core_execute() {
    while (true) {
        PrimBase *p = nullptr;    // 下一个要执行的原语
        bool conf_delete = false; // 是否自动填充了一个recv_conf原语

        if (prim_queue.size() == 0) {
            // 队列中没有指令，意味着现在是初始状态或者所有原语都被执行完了（假设所有原语只做一轮），默认作recv，直到config发进来
            p = new Recv_prim(RECV_TYPE::RECV_CONF);
            prim_queue.emplace_front(p);
            conf_delete = true;
        } else {
            p = prim_queue.front();
        }

        // NOTE:
        // send原语和recv原语和其他计算原语不同，需要涉及core中信号的处理，所以需要在core这个文件内部处理相关逻辑，否则会出现依赖问题。
        // 需要等待 switch_prim_block 将 prim_block 置为 false，然后再执行
        // switch_prim_block 收到 ev_block 触发 ev_block 在 send_logic 和
        // recv_logic 中触发

        if (typeid(*p) == typeid(Send_prim)) {
            // 触发 send_logic
            if (!SPEC_SEND_RECV_PARALLEL) {
                ev_send.notify(CYCLE, SC_NS);
                event_engine->add_event(
                    "Core " + ToHexString(cid), "Send_prim", "B",
                    Trace_event_util(
                        "Send_prim" +
                        GetEnumSendType(dynamic_cast<Send_prim *>(p)->type)));
                wait(prim_block.negedge_event());
                event_engine->add_event(
                    "Core " + ToHexString(cid), "Send_prim", "E",
                    Trace_event_util(
                        "Send_prim" +
                        GetEnumSendType(dynamic_cast<Send_prim *>(p)->type)));
            } else {
                while (!send_done) {
                    wait(CYCLE, SC_NS);
                }

                // send 模块处理的四条指令
                while ((typeid(*p) == typeid(Recv_prim) &&
                        ((Recv_prim *)p)->type == RECV_ACK) ||
                       (typeid(*p) == typeid(Send_prim) &&
                        ((Send_prim *)p)->type == SEND_DATA) ||
                       (typeid(*p) == typeid(Send_prim) &&
                        ((Send_prim *)p)->type == SEND_REQ) ||
                       (typeid(*p) == typeid(Send_prim) &&
                        ((Send_prim *)p)->type == SEND_DONE)) {
                    prim_queue.pop_front();
                    send_para_queue.push(p);
                    if (prim_refill) {
                        prim_queue.emplace_back(p);
                    }
                    if (!prim_queue.size())
                        break;
                    // 这里会pop出来RECV_ACK
                    p = prim_queue.front();
                }

                send_done = false;

                // 触发 send_logic
                ev_para_send.notify(CYCLE, SC_NS);
                continue;
            }
        } else if (typeid(*p) == typeid(Recv_prim)) {
            ev_recv.notify(CYCLE, SC_NS);
            event_engine->add_event(
                "Core " + ToHexString(cid), "Receive_prim", "B",
                Trace_event_util(
                    "Receive_prim" +
                    GetEnumRecvType(dynamic_cast<Recv_prim *>(p)->type)));
            wait(prim_block.negedge_event());
            event_engine->add_event(
                "Core " + ToHexString(cid), "Receive_prim", "E",
                Trace_event_util(
                    "Receive_prim" +
                    GetEnumRecvType(dynamic_cast<Recv_prim *>(p)->type)));
        } else {
            // 检查队列中p的下一个原语是否还是计算原语
            ev_comp.notify(CYCLE, SC_NS);
            event_engine->add_event("Core " + ToHexString(cid), "Comp_prim",
                                    "B", Trace_event_util(p->name));
            wait(prim_block.negedge_event());

            // 发送信号让send发送最后一个包
            if (prim_queue.size() >= 2 &&
                !(prim_queue[1]->prim_type & COMP_PRIM)) {
                send_last_packet = true;
                ev_send_last_packet.notify(CYCLE, SC_NS);
            }

            event_engine->add_event("Core " + ToHexString(cid), "Comp_prim",
                                    "E", Trace_event_util(p->name));
        }

        // 将原语重新填充到队列中
        if (prim_refill) {
            bool flag = false;
            if (typeid(*p) == typeid(Recv_prim)) {
                Recv_prim *rp = (Recv_prim *)p;
                if (rp->type == RECV_CONF || rp->type == RECV_WEIGHT) {
                    flag = true;
                }
            }

            if (!flag)
                prim_queue.emplace_back(p);
        }

        if (conf_delete)
            delete p;

        prim_queue.pop_front();
        wait(CYCLE, SC_NS);
    }
}

void WorkerCoreExecutor::switch_prim_block() {
    while (true) {
        prim_block.write(true);
        wait();

        prim_block.write(false);
        wait(CYCLE, SC_NS);
    }
}

// 指令被 RECV_CONF发送过来后，会在本地核实例化对应的指令类
PrimBase *WorkerCoreExecutor::parse_prim(vector<sc_bv<128>> segments) {
    int type = segments[0].range(7, 0).to_uint64();
    PrimBase *task = PrimFactory::getInstance().createPrim(type, false);

    task->deserialize(segments);
    task->prim_context = core_context;

    return task;
}

// 数据信道接收
void WorkerCoreExecutor::poll_buffer_i() {
    MSG_TYPE block_mark = MSG_TYPE::MSG_TYPE_NUM;

    while (true) {
        if (!data_sent_i.read()) {
            if (block_mark < MSG_TYPE::MSG_TYPE_NUM) {
                if (msg_buffer_[block_mark].size() < MAX_BUFFER_PACKET_SIZE) {
                    block_mark = MSG_TYPE::MSG_TYPE_NUM;
                    core_busy_o.write(false);
                    continue;
                }

                wait(ev_msg_process_end);
                continue;
            } else {
                wait(ev_data_sent_i);
            }
        }

        Msg m = DeserializeMsg(channel_i.read());
        msg_buffer_[m.msg_type_].push(m);
        ev_recv_msg_type_[m.msg_type_].notify(0, SC_NS);

        if (IsBlockableMsgType(m.msg_type_) &&
            msg_buffer_[m.msg_type_].size() >= MAX_BUFFER_PACKET_SIZE) {
            core_busy_o.write(true);
            block_mark = m.msg_type_;
        } else
            core_busy_o.write(false);

        wait(CYCLE, SC_NS);
    }
}

// 控制信道接收
void WorkerCoreExecutor::poll_ctrl_buffer_i() {
    while (true) {
        if (!ctrl_sent_i.read()) {
            // 控制信道空闲时，检查当前所有控制消息类型的 buffer 是否有空间
            // 控制消息包括 REQUEST、ACK、DONE，它们不会被 IsBlockableMsgType 标记
            // 如果任何一个控制消息类型的 buffer 满了，就设置 busy
            if (msg_buffer_[MSG_TYPE::REQUEST].size() >= MAX_BUFFER_PACKET_SIZE ||
                msg_buffer_[MSG_TYPE::ACK].size() >= MAX_BUFFER_PACKET_SIZE ||
                msg_buffer_[MSG_TYPE::DONE].size() >= MAX_BUFFER_PACKET_SIZE) {
                ctrl_core_busy_o.write(true);
            } else {
                ctrl_core_busy_o.write(false);
            }

            wait(ev_ctrl_sent_i);
            continue;
        }

        Msg m = DeserializeMsg(ctrl_channel_i.read());
        msg_buffer_[m.msg_type_].push(m);
        ev_ctrl_msg_recv.notify(0, SC_NS);
        // 同时触发对应消息类型的event，兼容原有逻辑
        ev_recv_msg_type_[m.msg_type_].notify(0, SC_NS);

        // 检查所有控制消息类型的 buffer 是否满
        // 如果任何一个满了，就设置 busy，阻止接收更多消息
        if (msg_buffer_[MSG_TYPE::REQUEST].size() >= MAX_BUFFER_PACKET_SIZE ||
            msg_buffer_[MSG_TYPE::ACK].size() >= MAX_BUFFER_PACKET_SIZE ||
            msg_buffer_[MSG_TYPE::DONE].size() >= MAX_BUFFER_PACKET_SIZE) {
            ctrl_core_busy_o.write(true);
        } else {
            ctrl_core_busy_o.write(false);
        }

        wait(CYCLE, SC_NS);
    }
}

// 捕获控制信道空闲信号
void WorkerCoreExecutor::catch_ctrl_channel_avail_i() {
    while (true) {
        ev_ctrl_channel_avail_i.notify(CYCLE, SC_NS);
        wait();
    }
}

// 捕获控制信道发送信号
void WorkerCoreExecutor::catch_ctrl_sent_i() {
    while (true) {
        ev_ctrl_sent_i.notify(CYCLE, SC_NS);
        wait();
    }
}

/*
 在workercore executor中添加了一把锁，用于lock住write helper，
 因为同时运行send和recv原语会在同一个时钟周期内access write helper函数
*/

/*
send_helper_write >= 2, that data_sent_o = true send a msg to router
send_helper_write < 2, that data_sent_o = false, reset signal

present_time the most recent time that the helper is try to lock

try = present_time some one has try to lock the helper before in the same
time

status = 0, If a new cycle begins and no other module requires the helper,
reset send_helper_write to 0. pool down data_sent_o status = 1, 表示
准备执行send taskCoreDefault （会有delay） 一般在status 0 之后
同一个周期内，行为和 0 一致

status = 2 表示send 从 sram 里面已经拿到数据了，可以开始发送了

status = 1 2 都只出现一次



*/

bool WorkerCoreExecutor::atomic_helper_lock(sc_time try_time, int status,
                                            bool force) {
    bool res;

    if (try_time < present_time)
        res = false;

    if (try_time == present_time) {
        if (status == 0) {
            if (force == true) {
                send_helper_write = status;
                return true;
            }
            return false;
        }

        if (status == 1) { // send prepare
            // status 1 只会出现在这里
            if (send_helper_write == 0) {
                send_helper_write = 1;
                res = true;
            } else {
                res = false;
            }
        }
        if (status == 2) { // send ready
            if (send_helper_write == 1) {
                send_helper_write = 2;
                res = true;
            } else {
                res = false;
            }
        }
        if (status == 3) { // other pass cond.
            if (send_helper_write == 0) {
                send_helper_write = 3;
                res = true;
            } else {
                res = false;
            }
        }
    }

    if (try_time > present_time) {
        if (try_time - present_time < sc_time(CYCLE, SC_NS))
            return false;

        present_time = try_time;
        // status 1 不会进入到这里，因为status 1 之前肯定会有 status 0
        // 修改了 present_time
        if (status == 2) { // send ready
            if (send_helper_write == 1) {
                send_helper_write = 2;
                res = true;
            } else {
                res = false;
            }
        } else {
            // 这里应该只会进 status 0,
            // 1 和 3 ( 3 除了返回给host ack 因为while循环)
            // 都会在0后处理，且不会有延迟，所以pre=try_time
            // status 1 后面只能紧跟status 2
            // 防止status 3 把 原本 status 2 抢占 了
            if (send_helper_write == 1 && force == false)
                res = false;
            else {
                // 0 或者 3 是当前周期第一来的状态
                // 2 只会出现上面一种情况 成为当前周期第一来的状态

                send_helper_write = status;
                res = true;
            }
        }
    }

    return res;
}
// data_sent_o pos trigger router && later router can self trigger if
// data_sent_o is true 是否拉低不重要，只要 data_sent_o 是高就能发送
// 根据消息类型选择数据信道或控制信道
void WorkerCoreExecutor::send_helper() {
    while (true) {
        bool flag = SPEC_ROUTER_PIPE ? (send_helper_write >= 1)
                                     : (send_helper_write >= 2);

        if (flag) {
            auto ser = SerializeMsg(send_buffer);
            // 根据消息类型选择信道
            if (send_buffer.IsControlMsg()) {
                // 控制消息走控制信道
                ctrl_channel_o.write(ser);
                ctrl_sent_o.write(true);
                data_sent_o.write(false);
            } else {
                // 数据消息走数据信道
                channel_o.write(ser);
                data_sent_o.write(true);
                ctrl_sent_o.write(false);
            }
            ev_next_write_clear.notify(CYCLE, SC_NS);
        } else {
            data_sent_o.write(false);
            ctrl_sent_o.write(false);
        }

        wait();
    }
}

void WorkerCoreExecutor::catch_channel_avail_i() {
    while (true) {
        ev_channel_avail_i.notify(CYCLE, SC_NS);

        wait();
    }
}

void WorkerCoreExecutor::next_write_clear() {
    while (true) {
        if (atomic_helper_lock(sc_time_stamp(), 0))
            ev_send_helper.notify(0, SC_NS);

        wait();
    }
}

void WorkerCoreExecutor::catch_data_sent_i() {
    while (true) {
        ev_data_sent_i.notify(CYCLE, SC_NS);

        wait();
    }
}

WorkerCoreExecutor::~WorkerCoreExecutor() {
    delete sram_addr;
#if USE_NB_DRAMSYS == 1
    delete nb_dcache_socket;
#else
    delete dcache_socket;
#endif
    delete mem_access_port;
    delete high_bw_mem_access_port;
    delete temp_mem_access_port;
    delete high_bw_temp_mem_access_port;
    delete start_nb_dram_event;
    delete end_nb_dram_event;
    delete start_sram_event;
    delete end_sram_event;
    delete start_global_mem_event;
    delete end_global_mem_event;
    delete sram_writer;
    delete g_dram_kvtable;
    delete g_dram_kvtable[cid];
}