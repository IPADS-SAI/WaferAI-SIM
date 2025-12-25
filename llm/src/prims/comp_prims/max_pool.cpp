#include "systemc.h"

#include "defs/global.h"
#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Max_pool);

void Max_pool::initialize() {
    auto &p = param_value;
    int oH = (p["H"] + 2 * p["pY"] - p["kY"]) / p["sY"] + 1;
    int oW = (p["W"] + 2 * p["pX"] - p["kX"]) / p["sX"] + 1;
    int oC = p["C"];
    data_size_input = {p["B"] * p["C"] * p["H"] * p["W"]};
    data_chunk = {{"output", p["B"] * oC * oH * oW}};
}

void Max_pool::taskCore(TaskCoreContext &context, string prim_name,
                       u_int64_t &dram_time, u_int64_t &exu_ops,
                       u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    int oH = (p["H"] + 2 * p["pY"] - p["kY"]) / p["sY"] + 1;
    int oW = (p["W"] + 2 * p["pX"] - p["kX"]) / p["sX"] + 1;
    int oC = p["C"];
    exu_ops = 0;
    sfu_ops = (u_int64_t)p["B"] * oC * oH * oW * p["kX"] * p["kY"];
    vec_ops = 0;
}
