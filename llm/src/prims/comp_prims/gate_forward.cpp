#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(gate_forward);

void gate_forward::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"output", p["B"] * p["T"] * p["K"]}};
}

void gate_forward::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    exu_ops = (uint64_t)p["B"] * p["T"] * p["C"] * p["E_N"];
    sfu_ops = 0;
    vec_ops = 0;
}