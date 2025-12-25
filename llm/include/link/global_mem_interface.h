#pragma once
#include "nlohmann/json.hpp"
#include <fstream>
#include <string>
#include <atomic>
#include <vector>

#include "monitor/config_helper_core.h"
#include "monitor/config_helper_gpu.h"
#include "monitor/config_helper_pd.h"
#include "monitor/mem_interface.h"
#include "prims/comp_prims.h"
#include "prims/norm_prims.h"
#include "utils/print_utils.h"
#include "utils/msg_utils.h"
#include "utils/prim_utils.h"
#include "trace/Event_engine.h"
#include "monitor/config_helper_base.h"

#include "link/chip_global_memory.h"
#include "link/chip_config_helper.h"

class Event_engine;
class config_helper_base;

class GlobalMemInterface : public sc_module {
public:

    ChipGlobalMemory *chipGlobalMemory;
    NB_GlobalMemIF *nb_global_mem_socket;
    Event_engine *event_engine;
    chip_config_helper *config_helper;

    int cid;
    bool chip_prim_refill = true;

    std::vector<chip_instr_base*> global_instrs;
    sc_signal<bool> chip_prim_block;

    deque<chip_instr_base*> global_instrs_queue;

    sc_event ev_block;
    sc_event ev_task;

    SC_HAS_PROCESS(GlobalMemInterface);  // Enable SystemC processes for this module
    GlobalMemInterface(const sc_module_name &n, Event_engine *event_engine,
                        const char *config_name);

    GlobalMemInterface(const sc_module_name &n, Event_engine *event_engine,
                config_helper_base *input_config);

    GlobalMemInterface();

    void init();
    void load_global_prims(const char *config_name);
    void task_logic();
    void switch_chip_prim_block();
    void instr_executor();
    void init_prim();

    // GlobalMemInterfaceExecutor *executor;

    // // Load and execute global_interface primitives from the config JSON
    // void load_global_prims(const char *config_name);
    // void execute_prims();

    // ChipGlobalMemory *chipGlobalMemory;
    // NB_GlobalMemIF *nb_global_mem_socket;

    // // Global interface instructions
    // std::vector<chip_instr_base*> global_instrs;
};
