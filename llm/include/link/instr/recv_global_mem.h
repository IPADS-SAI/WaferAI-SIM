#pragma once
#include "chip_instr.h"
#include "link/chip_task_context.h"
#include <tlm>
#include <systemc>

class Recv_global_mem : public chip_instr_base{
public:

    Recv_global_mem(){
        name = "Recv_global_mem";
        instr_type = SEQ_EXEC;
        id = -1;
    }

    Recv_global_mem(int i){
        name = "Recv_global_mem";
        instr_type = SEQ_EXEC;
        id = i;
    }

    void parseJson(json j) override {
        seq = j.contains("seq") ? j["seq"].get<int>() : -1;
    }

    void printSelf(std::string prefix) override{
        std::cout << prefix << "<Recv_global_mem>" << std::endl;
    }

    int taskCoreDefault(TaskChipContext &context) override {
        std::cout << "Executing <Recv_global_mem>" << std::endl;
        return 0;
    }
};