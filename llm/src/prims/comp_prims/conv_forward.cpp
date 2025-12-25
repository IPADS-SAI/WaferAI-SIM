#include "systemc.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Conv_f);

void Conv_f::initialize() {
    auto &p = param_value;
    int oH = (p["H"] + 2 * p["pY"] - p["kY"]) / p["sY"] + 1;
    int oW = (p["W"] + 2 * p["pX"] - p["kX"]) / p["sX"] + 1;
    int oC = p["F"];

    data_size_input = {p["B"] * p["C"] * p["H"] * p["W"]};
    data_chunk = {{"weight", p["F"] * p["C"] * p["kY"] * p["kX"]},
                  {"bias", p["F"]},
                  {"output", p["B"] * p["F"] * oH * oW}};
}

void Conv_f::taskCore(TaskCoreContext &context, string prim_name,
                     u_int64_t &dram_time, u_int64_t &exu_ops,
                     u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto label_weight = ETERNAL_PREFIX + prim_name + "_w";
    checkStaticData(context, dram_time, data_chunk_addr["weight"],
                    GetFromPairedVector(data_chunk, "weight"), label_weight);

    auto label_bias = ETERNAL_PREFIX + prim_name + "_b";
    checkStaticData(context, dram_time, data_chunk_addr["bias"],
                    GetFromPairedVector(data_chunk, "bias"), label_bias);

    auto &p = param_value;
    int oH = (p["H"] + 2 * p["pY"] - p["kY"]) / p["sY"] + 1;
    int oW = (p["W"] + 2 * p["pX"] - p["kX"]) / p["sX"] + 1;
    int oC = p["F"];
    exu_ops = (uint64_t)p["B"] * p["C"] * p["kY"] * p["kX"] * 2 * oH * oW * oC;
    sfu_ops = 0;
    vec_ops = 0;
}