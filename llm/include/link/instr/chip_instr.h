#pragma once
#include "common/system.h"
#include "defs/const.h"
#include "defs/enums.h"
#include "defs/global.h"
#include "systemc.h"

#include "nlohmann/json.hpp"
#include <vector>
#include <unordered_set>

#include "link/chip_task_context.h"

class TaskChipContext;

using json = nlohmann::json;

enum INSTR_TYPE{
    SEQ_EXEC = 0, //顺序执行，不考虑乱序情况
    OOO_EXEC,
};

class chip_instr_base{
public:

    int id;
    int seq;
    INSTR_TYPE instr_type;
    std::string name;

    std::unordered_set<int> dependencies; 
    
    chip_instr_base(){
        
    }

    chip_instr_base(int i){
        id = i;
        name = "chip_instr_base";
        instr_type = SEQ_EXEC;
    }

    virtual void parseJson(json j) = 0;
    virtual int taskCoreDefault(TaskChipContext &context) = 0;

    virtual ~chip_instr_base() = default;

    virtual void printSelf(std::string prefix) = 0;
};
