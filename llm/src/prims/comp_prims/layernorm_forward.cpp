#include "systemc.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Layernorm_f)

void Layernorm_f::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"weight", p["C"]},
                  {"bias", p["C"]},
                  {"output", p["B"] * p["T"] * p["C"]}};
}

void Layernorm_f::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read weight";

    auto label_weight = ETERNAL_PREFIX + prim_name + "_w";
    checkStaticData(context, dram_time, data_chunk_addr["weight"],
                    GetFromPairedVector(data_chunk, "weight"), label_weight);

    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid << " read bias";

    auto label_bias = ETERNAL_PREFIX + prim_name + "_b";
    checkStaticData(context, dram_time, data_chunk_addr["bias"],
                    GetFromPairedVector(data_chunk, "bias"), label_bias);

    auto &p = param_value;
    exu_ops = 0;
    sfu_ops = (u_int64_t)p["B"] * p["T"];
    vec_ops = (u_int64_t)p["B"] * p["T"] * (8 * p["C"] + 3);
}