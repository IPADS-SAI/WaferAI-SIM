#include "prims/pd_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"

REGISTER_PRIM(rope_forward_pd);

void rope_forward_pd::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"sincos", p["B"] * (p["C"] / p["NH"]) * 2},
                  {"output", p["B"] * p["T"] * p["C"] / (1 + 2 / p["R"])}};
}

void rope_forward_pd::taskCore(TaskCoreContext &context, string prim_name,
                               u_int64_t &dram_time, u_int64_t &exu_ops,
                               u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    // 此时默认已经分好注意力头了。对于每一个注意力头，对应的sincos数据大小均为B
    // * T * (C / NH) (最后一个维度已扩展)
    // 读出需要用到的sincos数据
    auto &p = param_value;
    int max_token_num = 0;
    for (auto stage : prim_context->batch_info_)
        max_token_num = max(max_token_num, stage.token_num);
    if (p["job_type"] == JOB_DECODE)
        max_token_num = 1;

    // 读入sincos数据
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read degree data";
    auto label_sincos = ETERNAL_PREFIX + prim_name + "_sc";
    checkStaticData(context, dram_time, data_chunk_addr["sincos"],
                    GetFromPairedVector(data_chunk, "sincos") * max_token_num,
                    label_sincos);

    // 在这里写回kvcache
    int total_tokens = 0;

    for (auto stage : prim_context->batch_info_) {
        int size = 0;
        switch (p["job_type"]) {
        case JOB_PREFILL:
        case JOB_BOTH:
            size = data_byte * p["B"] * p["C"] * stage.token_num;
            break;
        case JOB_DECODE:
            size = data_byte * p["B"] * p["C"] * p["chunk"];
            break;
        default:
            assert(false && "Unsupported job type");
        }

        total_tokens += stage.token_num;

        char format_label_k[100];
        sprintf(format_label_k, "%s%sk#%d", ETERNAL_PREFIX, KVCACHE_PREFIX,
                stage.req_id);
        string label_k = format_label_k;

        char format_label_v[100];
        sprintf(format_label_v, "%s%sv#%d", ETERNAL_PREFIX, KVCACHE_PREFIX,
                stage.req_id);
        string label_v = format_label_v;

        // 如果没有对应的kvcache，则创建一个标签；如果已经有了，则直接更新大小
        LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                        << " write K cache";
#if USE_SRAM_MANAGER == 1
        sram_update_cache(context, label_k, prim_context->sram_pos_locator_,
                          size, dram_time, prim_context->cid);
#else
        sram_write_append_generic(context, size, dram_time);
        prim_context->sram_pos_locator_->updatePair(label_k, size, context,
                                                    dram_time);
#endif

        LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                        << " write V cache";
#if USE_SRAM_MANAGER == 1
        sram_update_cache(context, label_v, prim_context->sram_pos_locator_,
                          size, dram_time, prim_context->cid);
#else
        sram_write_append_generic(context, size, dram_time);
        prim_context->sram_pos_locator_->updatePair(label_v, size, context,
                                                    dram_time);
#endif
    }

    exu_ops = 0;
    sfu_ops = 0;
    vec_ops = (u_int64_t)p["C"] * total_tokens * 3;
}