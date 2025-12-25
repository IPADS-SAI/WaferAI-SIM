#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(switch_data);

void switch_data::initialize() {
    auto &p = param_value;
    data_size_input = {p["IN"]};
    data_chunk = {{"output", p["OUT"]}};
}

void switch_data::taskCore(TaskCoreContext &context, string prim_name,
                          u_int64_t &dram_time, u_int64_t &exu_ops,
                          u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    exu_ops = 0;
    sfu_ops = 0;
    vec_ops = 0;
}