#include "systemc.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Merge_matmul);

void Merge_matmul::initialize() {
    data_chunk.clear();
    data_size_input.clear();

    auto &p = param_value;
    if (p["dim"] == 1)
        data_chunk.push_back({"output", p["B"] * p["T"] * p["C"]});
    else if (p["dim"] == 2)
        data_chunk.push_back({"output", p["B"] * p["T"] * p["C"] * p["slice"]});
        
    for (int i = 0; i < p["slice"]; i++)
        data_size_input.push_back(p["B"] * p["T"] * p["C"]);
}

void Merge_matmul::taskCore(TaskCoreContext &context, string prim_name,
                            u_int64_t &dram_time, u_int64_t &exu_ops,
                            u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    exu_ops = (u_int64_t)p["B"] * p["T"] * p["C"];
    sfu_ops = 0;
    vec_ops = 0;
}