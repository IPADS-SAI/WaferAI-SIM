#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(silu_forward);

void silu_forward::initialize() {
    auto &p = param_value;
    data_size_input = {p["N"]};
    data_chunk = {{"output", p["N"]}};
}

void silu_forward::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    exu_ops = 0;
    sfu_ops = (u_int64_t)p["N"];
    vec_ops = (u_int64_t)p["N"] * 3;
}