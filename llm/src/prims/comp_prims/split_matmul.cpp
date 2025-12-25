#include "systemc.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Split_matmul);

void Split_matmul::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"output", p["B"] * p["T"] * p["C"]}};
}

void Split_matmul::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    exu_ops = 0;
    sfu_ops = 0;
    vec_ops = 0;
}