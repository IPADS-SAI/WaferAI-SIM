#pragma once
#include "systemc.h"

#include "../router/router.h"
#include "../workercore/workercore.h"
#include "link/chip_global_memory.h"
#include "monitor/config_helper_base.h"
#include "monitor/gpu_cache_system.h"
#include "monitor/mem_interface.h"
#include "trace/Event_engine.h"
#include "link/global_mem_interface.h"

using namespace std;

class Monitor : public sc_module {
public:
    // signals
    sc_signal<bool> *core_busy;
    sc_signal<sc_bv<256>> *channel[DIRECTIONS];
    sc_signal<sc_bv<256>> *rc_channel;
    sc_signal<bool> *channel_avail[DIRECTIONS];
    sc_signal<bool> *data_sent[DIRECTIONS];
    sc_signal<bool> *rc_data_sent;

    sc_signal<bool> *host_channel_avail;
    sc_signal<bool> *host_data_sent_i;
    sc_signal<bool> *host_data_sent_o;
    sc_signal<sc_bv<256>> *host_channel_i;
    sc_signal<sc_bv<256>> *host_channel_o;

    /* ---------------Control Channel Signals------------------- */
    // Router-Router 控制信道信号//qzl添加
    sc_signal<sc_bv<256>> *ctrl_channel[DIRECTIONS];
    sc_signal<bool> *ctrl_channel_avail[DIRECTIONS];
    sc_signal<bool> *ctrl_sent[DIRECTIONS];
    sc_signal<bool> *ctrl_core_busy;  

    // Router-Core 控制信道信号
    sc_signal<sc_bv<256>> *rc_ctrl_channel;
    sc_signal<bool> *rc_ctrl_sent;

    // Host-Router 控制信道信号

    sc_signal<bool> *host_ctrl_sent_i;
    sc_signal<sc_bv<256>> *host_ctrl_channel_i;
    /* --------------------------------------------------------- */

    sc_signal<bool> star;
    sc_signal<bool> config_done;
    sc_out<bool> start_o;
    sc_in<bool> preparations_done_i;

    // components
    RouterMonitor *routerMonitor;
    WorkerCore **workerCores;
    MemInterface *memInterface;
    
    GlobalMemInterface *globalMemInterface;
    // ChipGlobalMemory *chipGlobalMemory;

#if USE_L1L2_CACHE == 1
    L1L2CacheSystem *cacheSystem;
    GpuPosLocator *gpu_pos_locator;
#else
#endif

    Event_engine *event_engine;
    const char *config_name;

    SC_HAS_PROCESS(Monitor);
    Monitor(const sc_module_name &n, Event_engine *event_engine,
            const char *config_name);
    Monitor(const sc_module_name &n, Event_engine *event_engine,
            config_helper_base *input_config);
    ~Monitor();

    void start_simu();

private:
    void init();
};