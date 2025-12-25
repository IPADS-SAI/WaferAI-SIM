#include "common/pd.h"
#include "prims/pd_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(matmul_forward_pd);

void matmul_forward_pd::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"weight", p["C"] * p["OC"]},
                  {"bias", p["OC"]},
                  {"output", p["B"] * p["T"] * p["OC"] / 3}};
}

void matmul_forward_pd::taskCore(TaskCoreContext &context, string prim_name,
                                 u_int64_t &dram_time, u_int64_t &exu_ops,
                                 u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    // 空转一轮，直接退出（PD模式）
    auto &p = param_value;
    if (p["T"] == 0)
        return;

    bool need_multiply = false;
    for (auto stage : prim_context->batch_info_) {
        if (stage.type == DECODE) {
            need_multiply = true;
            break;
        }
    }

    int chunk_ratio = need_multiply ? 1 : p["chunk"];
    auto label_weight = ETERNAL_PREFIX + prim_name + "_w";

    if (SPEC_LOAD_STATIC == "layer") {
        // 直接加载一整层的权重。这里模拟为读取单个完整tensor。spill时优先排出最旧访问权重。
        checkStaticData(context, dram_time, data_chunk_addr["weight"],
                        GetFromPairedVector(data_chunk, "weight") / chunk_ratio,
                        label_weight, false);
    } else if (SPEC_LOAD_STATIC == "single") {
        // 加载单个完整权重。这里模拟为读取单个完整tensor。spill时优先排出最新访问权重。
        checkStaticData(context, dram_time, data_chunk_addr["weight"],
                        GetFromPairedVector(data_chunk, "weight") / chunk_ratio,
                        label_weight, false);
    } else if (SPEC_LOAD_STATIC == "partial") {
        // 加载部分权重。这里模拟为分批读取权重的一部分。spill时优先排出最新访问权重。
        int mac_size = 64 * 1024;
        LOG_DEBUG(MEMORY) << "mac_size " << mac_size;

        checkStaticDataTile(context, dram_time, data_chunk_addr["weight"],
                            GetFromPairedVector(data_chunk, "weight") /
                                chunk_ratio,
                            label_weight, false, mac_size);
    }

    auto label_bias = ETERNAL_PREFIX + prim_name + "_b";
    checkStaticData(context, dram_time, data_chunk_addr["bias"],
                    GetFromPairedVector(data_chunk, "bias") / chunk_ratio,
                    label_bias);

    // 写入kvcache，根据batchInfo确定
    for (auto stage : prim_context->batch_info_) {
        int size = 0;
        switch (p["job_type"]) {
        case JOB_PREFILL:
        case JOB_BOTH:
            size = data_byte * p["B"] * p["OC"] * stage.token_num / 3;
            break;
        case JOB_DECODE:
            size = data_byte * p["B"] * p["OC"] / 3 * p["chunk"];
            break;
        default:
            assert(false && "Unsupported job type");
        }

        char format_label_k[100];
        sprintf(format_label_k, "%s%sk#%d", ETERNAL_PREFIX, KVCACHE_PREFIX,
                stage.req_id);
        string label_k = format_label_k;

        char format_label_v[100];
        sprintf(format_label_v, "%s%sv#%d", ETERNAL_PREFIX, KVCACHE_PREFIX,
                stage.req_id);
        string label_v = format_label_v;

        // 如果没有对应的kvcache，则创建一个标签；如果已经有了，则直接更新大小
#if USE_SRAM_MANAGER == 1
        sram_update_cache(context, label_k, prim_context->sram_pos_locator_,
                          size, dram_time, prim_context->cid);
#else
        sram_write_append_generic(context, size, dram_time);
        prim_context->sram_pos_locator_->updatePair(label_k, size, context,
                                                    dram_time);
#endif

#if USE_SRAM_MANAGER == 1
        sram_update_cache(context, label_v, prim_context->sram_pos_locator_,
                          size, dram_time, prim_context->cid);
#else
        sram_write_append_generic(context, size, dram_time);
        prim_context->sram_pos_locator_->updatePair(label_v, size, context,
                                                    dram_time);
#endif
    }

    // 决定是否终止（需要放在别的原语中）
    prim_context->decode_done_.clear();
    for (auto stage : prim_context->batch_info_) {
        if (stage.type == DECODE && RandResult(2))
            prim_context->decode_done_.push_back(true);
        else
            prim_context->decode_done_.push_back(false);
    }


    ExuConfig *exu = GetCoreHWConfig(context.cid)->exu;

    uint64_t weight_tile_x = (p["C"] + exu->x_dims - 1) / exu->x_dims;
    uint64_t weight_tile_y = (p["OC"] + exu->x_dims - 1) / exu->x_dims;

    uint64_t padding_input_x =
        (p["B"] * p["T"]) > exu->x_dims ? p["B"] * p["T"] : exu->x_dims;

    uint64_t performance_cycle = (exu->x_dims + exu->x_dims + padding_input_x) *
                                 weight_tile_x * weight_tile_y;

    uint64_t performance_comp =
        performance_cycle * exu->x_dims * exu->x_dims * HW_COMP_UTIL;
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " performance_cycle " << performance_cycle;

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
