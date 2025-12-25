#pragma once
#include "systemc.h"

#include "common/config.h"
#include "common/msg.h"

#include "link/instr/chip_instr.h"

using namespace std;

class chip_config_helper{
public:
    int cid;
    chip_config_helper(string filename, int cid = 0);

    vector<chip_instr_base*> instr_list;

    void printSelf();
};