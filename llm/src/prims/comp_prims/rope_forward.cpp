#include "prims/comp_prims.h"
#include "utils/memory_utils.h"

REGISTER_PRIM(rope_forward);

void rope_forward::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"sincos", p["B"] * (p["C"] / p["NH"]) * 2 * p["T"]},
                  {"output", p["B"] * p["T"] * p["C"]}};
}

void rope_forward::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    // 此时默认已经分好注意力头了。对于每一个注意力头，对应的sincos数据大小均为B
    // * T * (C / NH) (最后一个维度已扩展)
    // 读出需要用到的sincos数据
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read degree data";

    auto label_sincos = ETERNAL_PREFIX + prim_name + "_sc";
    checkStaticData(context, dram_time, data_chunk_addr["sincos"],
                    GetFromPairedVector(data_chunk, "sincos"), label_sincos);

    auto &p = param_value;
    exu_ops = 0;
    sfu_ops = 0;
    vec_ops = 3 * p["T"] * p["C"];
}