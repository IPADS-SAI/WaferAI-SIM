#include "systemc.h"
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include "common/system.h"
#include "defs/global.h"
#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Matmul_f);

void Matmul_f::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"weight", p["C"] * p["OC"]},
                  {"bias", p["OC"]},
                  {"output", p["B"] * p["T"] * p["OC"]}};
}

void Matmul_f::taskCore(TaskCoreContext &context, string prim_name,
                        u_int64_t &dram_time, u_int64_t &exu_ops,
                        u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read weight";

    auto label_weight = ETERNAL_PREFIX + prim_name + "_w";
    if (SPEC_LOAD_STATIC == "layer") {
        // 直接加载一整层的权重。这里模拟为读取单个完整tensor。spill时优先排出最旧访问权重。
        checkStaticData(context, dram_time, data_chunk_addr["weight"],
                        GetFromPairedVector(data_chunk, "weight"), label_weight,
                        false);
    } else if (SPEC_LOAD_STATIC == "single") {
        // 加载单个完整权重。这里模拟为读取单个完整tensor。spill时优先排出最新访问权重。
        checkStaticData(context, dram_time, data_chunk_addr["weight"],
                        GetFromPairedVector(data_chunk, "weight"), label_weight,
                        false);
    } else if (SPEC_LOAD_STATIC == "partial") {
        // 加载部分权重。这里模拟为分批读取权重的一部分。spill时优先排出最新访问权重。
        int mac_size = 64 * 1024;
        LOG_DEBUG(MEMORY) << "mac_size " << mac_size;

        checkStaticDataTile(context, dram_time, data_chunk_addr["weight"],
                            GetFromPairedVector(data_chunk, "weight"),
                            label_weight, false, mac_size);
    }

    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid << " read bias";

    auto label_bias = ETERNAL_PREFIX + prim_name + "_b";
    checkStaticData(context, dram_time, data_chunk_addr["bias"],
                    GetFromPairedVector(data_chunk, "bias"), label_bias, false);

    auto &p = param_value;

    ExuConfig *exu = GetCoreHWConfig(context.cid)->exu;

    uint64_t weight_tile_x = (p["C"] + exu->x_dims - 1) / exu->x_dims;
    uint64_t weight_tile_y = (p["OC"] + exu->x_dims - 1) / exu->x_dims;

    uint64_t padding_input_x =
        (p["T"] * p["B"]) > exu->x_dims ? p["T"] * p["B"] : exu->x_dims;

    uint64_t performance_cycle = (exu->x_dims + exu->x_dims + padding_input_x) *
                                 weight_tile_x * weight_tile_y;

    uint64_t performance_comp =
        performance_cycle * exu->x_dims * exu->x_dims * HW_COMP_UTIL;

    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " performance_cycle " << performance_cycle
                    << " performance_comp " << performance_comp;

    int loop_input_count =
        weight_tile_y - 1; // read loop_input_count Repetitive input

    for (int loop = 0; loop < loop_input_count; loop++) {
        for (int p = 0; p < data_size_input.size(); p++) {
            if (prim_context->datapass_label_->indata[p].find(DRAM_LABEL) ==
                0) {
                prefReadData(context, dram_time, data_size_input[p],
                             prim_context->datapass_label_->indata[p]);
            }
        }
    }

    exu_ops = performance_comp * 2;
    sfu_ops = 0;
    vec_ops = (uint64_t)p["B"] * p["OC"] * p["T"] * p["C"] * 2;

    // 比较使用vector core是否更快
    VectorConfig *vec = GetCoreHWConfig(context.cid)->vec;

    int exu_cycle = 0;
    exu_cycle += exu_ops /
                 (exu->x_dims * exu->x_dims * 2 * exu->count * HW_COMP_UTIL) *
                 CYCLE;
    int vec_cycle = vec_ops / vec->x_dims / vec->count * CYCLE;
    if (vec_cycle < exu_cycle) {
        exu_ops = 0;
        sfu_ops = 0;
    } else {
        vec_ops = 0;
        sfu_ops = 0;
    }
}