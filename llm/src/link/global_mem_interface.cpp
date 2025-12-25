#include <atomic>
#include <vector>

#include "link/chip_config_helper.h"
#include "link/global_mem_interface.h"
#include "nlohmann/json.hpp"
#include <fstream>

#include <typeinfo>
#include "link/instr/recv_global_mem.h"
#include "link/instr/print_msg.h"
#include "link/instr/wait_event.h"

GlobalMemInterface::GlobalMemInterface(const sc_module_name &n, Event_engine *event_engine,const char *config_name) 
    : sc_module(n), event_engine(event_engine) {
    config_helper = new chip_config_helper(config_name, 0);
    init(); 
    
    // Load any global_interface primitives from the config
    load_global_prims(config_name);
}

GlobalMemInterface::GlobalMemInterface(const sc_module_name &n, Event_engine *event_engine, config_helper_base *input_config){
    assert(0);
}

GlobalMemInterface::GlobalMemInterface() {
    init();
}

void GlobalMemInterface::init() {

    chipGlobalMemory = new ChipGlobalMemory(sc_gen_unique_name("chip-global-memory"), "../DRAMSys/configs/ddr4-example-8bit.json", "../DRAMSys/configs");
    // // assert(0 && "task_logic's is not impl and sc_env is not implemented");
    SC_THREAD(switch_chip_prim_block);
    sensitive << ev_block;

    SC_THREAD(instr_executor);

    SC_THREAD(task_logic);
    sensitive << ev_task;
    dont_initialize();
}

void GlobalMemInterface::load_global_prims(const char *config_name) {
    global_instrs.clear();
    global_instrs_queue.clear();

    for(auto instr : config_helper->instr_list){
        global_instrs.push_back(instr);
        global_instrs_queue.emplace_back(instr);
    }
}

void GlobalMemInterface::switch_chip_prim_block() {
    while(true){
        chip_prim_block.write(true);
        wait();

        chip_prim_block.write(false);
        wait(CYCLE, SC_NS);
    }
}

void GlobalMemInterface::task_logic() {
    while(true) {
        chip_instr_base *p = global_instrs_queue.front();        
        int delay = 0;

        TaskChipContext context = generate_chip_context(this);

        if (typeid(*p) == typeid(Recv_global_mem) && chipGlobalMemory) {
            wait(chipGlobalMemory->write_received_event);
        } 

        // p->datapass_label = *next_datapass_label;
        delay = p->taskCoreDefault(context);
        wait(sc_time(delay, SC_NS));

        ev_block.notify(CYCLE, SC_NS);
        wait();
        // TaskChipContext context = generate_chip_context(this);
    }

    //

//     void GlobalMemInterface::task_logic() {
//     // Continuously execute global instructions
//     while (true) {
//         // If no instructions, wait for load or reload
//         if (global_instrs_queue.empty()) {
//             wait(ev_task);
//             continue;
//         }
//         // Fetch next instruction
//         chip_instr_base *p = global_instrs_queue.front();
//         global_instrs_queue.pop_front();

//         // Dispatch based on instruction type
//         if (typeid(*p) == typeid(Recv_global_mem)) {
//             auto *prim = static_cast<Recv_global_mem*>(p);
//             std::cout << "[GlobalMemInterface] Recv_global_mem seq=" << prim->seq << std::endl;
//             // TODO: insert actual global memory receive logic here
//         } else if (typeid(*p) == typeid(Print_msg)) {
//             auto *prim = static_cast<Print_msg*>(p);
//             std::cout << "[GlobalMemInterface] Print_msg: " << prim->message << std::endl;
//         } else if (typeid(*p) == typeid(Wait_event)) {
//             auto *prim = static_cast<Wait_event*>(p);
//             std::cout << "[GlobalMemInterface] Wait_event event: " << prim->event << std::endl;
//         } else {
//             std::cout << "[GlobalMemInterface] Unknown instruction: " << p->name << std::endl;
//         }
//         // Throttle to next system cycle
//         wait(CYCLE, SC_NS);
//     }
// }


    // using namespace std;
    // // Execute each global instruction in order
    // while (!global_instrs_queue.empty()) {
    //     chip_instr_base* p = global_instrs_queue.front();
    //     global_instrs_queue.pop_front();
    //     // Dispatch based on instruction type
    //     if (typeid(*p) == typeid(Recv_global_mem)) {
    //         Recv_global_mem* prim = static_cast<Recv_global_mem*>(p);
    //         cout << "[GlobalMemInterface] Recv_global_mem seq=" << prim->seq << endl;
    //         // TODO: insert global memory receive logic here
    //     } else if (typeid(*p) == typeid(Print_msg)) {
    //         Print_msg* prim = static_cast<Print_msg*>(p);
    //         cout << "[GlobalMemInterface] Print_msg: " << prim->message << endl;
    //     } else {
    //         cout << "[GlobalMemInterface] Unknown instruction: " << p->name << endl;
    //     }
    //     // Throttle between instructions
    //     wait(CYCLE, SC_NS);
    // }
    // // No more instructions, suspend thread
    // while (true) wait();

}

void GlobalMemInterface::instr_executor() {
    while(true) {
        chip_instr_base *p = global_instrs_queue.front();
        // global_instrs_queue.pop_front();

        if(global_instrs_queue.size() == 0){
            //
            wait();
        } else{
            p = global_instrs_queue.front();
            ev_task.notify(CYCLE, SC_NS);
            event_engine->add_event("Chip " + ToHexString(cid), "Comp_prim",
                                    "B", Trace_event_util(p->name));
            wait(chip_prim_block.negedge_event());
            event_engine->add_event("Chip " + ToHexString(cid), "Comp_prim",
                                    "E", Trace_event_util(p->name));
                                    
            if(chip_prim_refill){
                bool flag = false;
                global_instrs_queue.emplace_back(p);
            }

            global_instrs_queue.pop_front();
            wait(CYCLE, SC_NS);
        }
    }
}

void GlobalMemInterface::init_prim(){
    //开始时，将config_core里面的所有原语压入栈
    assert(0);
    global_instrs_queue.clear();

    for(auto instr : config_helper->instr_list){
        global_instrs_queue.emplace_back(instr);
    }
}