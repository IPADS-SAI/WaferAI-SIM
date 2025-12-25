#pragma once
#include "chip_instr.h"

class Print_msg : public chip_instr_base{
public:
    std::string message;

    Print_msg() {
        name = "Print_msg";
        instr_type = SEQ_EXEC;
        id = -1;
        message = "";
    }

    Print_msg(int i, std::string msg){
        message = msg;
        id = i;
        name = "Print_msg";
        instr_type = SEQ_EXEC;
    }

    void parseJson(json j) override {
        // Basic implementation - can be expanded based on needs
        seq = j.contains("seq") ? j["seq"].get<int>() : -1;
        if (j.contains("message")) {
            message = j["message"];
        }
    }

    void printSelf(std::string prefix) override{
        std::cout << prefix << "<Print_msg>" << std::endl;
        std::cout << prefix << "\tmsg: " << message << std::endl;
    }

    int taskCoreDefault(TaskChipContext &context) override {
        std::cout << "Executing <Print_msg>" << std::endl;
        std::cout << "\t" << "message: " << message << std::endl;
        return 10000;
    }
};