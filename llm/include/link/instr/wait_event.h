#pragma once
#include "chip_instr.h"

class Wait_event : public chip_instr_base{
public:
    std::string event;

    Wait_event() {
        name = "Wait_event";
        instr_type = SEQ_EXEC;
        id = -1;
        event = "";
    }

    Wait_event(int i, std::string e){
        event = e;
        id = i;
        name = "Wait_event";
        instr_type = SEQ_EXEC;
    }

    void parseJson(json j) override {
        seq = j.contains("seq") ? j["seq"].get<int>() : -1;
        if (j.contains("event")) {
            event = j["event"];
        }
    }

    void printSelf(std::string prefix) override{
        std::cout << prefix << "<Wait_event>" << std::endl;
    }

    int taskCoreDefault(TaskChipContext &context) override {
        return 0;
    }
};