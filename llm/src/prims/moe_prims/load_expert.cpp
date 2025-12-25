#include "prims/moe_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(load_expert);

void load_expert::initialize() {
    auto &p = param_value;
    data_size_input = {0};
    data_chunk = {{"output", 0}};

    for (int i = 1; i <= p["E_N"]; i++) {
        for (int j = 1; j <= 3; j++) {
            data_chunk.push_back({"weight_" + to_string(j) + "_" + to_string(i),
                                  p["C"] * p["OC"]});
            data_chunk.push_back(
                {"bias_" + to_string(j) + "_" + to_string(i), p["OC"]});
        }
    }
}

void load_expert::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;
    int exp_1;

    if (p["strategy"] == MOE_LOAD_STRATEGY_NONE) {
        LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                        << " no expert to load";
        return;
    }

    if (p["strategy"] == MOE_LOAD_STRATEGY_HOT) {
        exp_1 = 0;
        int max_cnt = -1;
        for (int i = 0; i < prim_context->selected_experts_.size(); i++) {
            int cnt = prim_context->selected_freq_[i];
            if (cnt > max_cnt) {
                max_cnt = cnt;
                exp_1 = i;
            }
        }
    }

    else if (p["strategy"] == MOE_LOAD_STRATEGY_RANDOM) {
        exp_1 = rand() % p["E_N"];
    }

    prim_context->prefetched_experts_.clear();
    prim_context->prefetched_experts_.push_back(exp_1);

    // 将单个专家的所有数据load到sram中
    for (int i = 1; i <= 3; i++) {
        checkStaticData(
            context, dram_time,
            data_chunk_addr["weight_" + to_string(i) + "_" + to_string(exp_1)],
            GetFromPairedVector(data_chunk, "weight_" + to_string(i) + "_" +
                                                to_string(exp_1)),
            prim_context->datapass_label_->indata[i] + "_weight_" +
                to_string(exp_1));
        checkStaticData(
            context, dram_time,
            data_chunk_addr["bias_" + to_string(i) + "_" + to_string(exp_1)],
            GetFromPairedVector(data_chunk, "bias_" + to_string(i) + "_" +
                                                to_string(exp_1)),
            prim_context->datapass_label_->indata[i] + "_bias_" +
                to_string(exp_1));
    }

    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " Prefetch expert: " << exp_1;

    exu_ops = 0;
    sfu_ops = 0;
    vec_ops = 0;
}