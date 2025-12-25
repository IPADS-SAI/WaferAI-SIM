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

REGISTER_PRIM(Matmul_f_mla);

void Matmul_f_mla::initialize() {
    auto &p = param_value;
    data_size_input = {p["B"] * p["T"] * p["C"]};
    data_chunk = {{"weight", p["C"] * p["OC"]},
                  {"bias", p["OC"]},
                  {"output", p["B"] * p["T"] * 2 * p["NH"] * p["DH"]}};
}

void Matmul_f_mla::taskCore(TaskCoreContext &context, string prim_name,
                            u_int64_t &dram_time, u_int64_t &exu_ops,
                            u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    auto &p = param_value;

    // PD 模式空转
    if (p["T"] == 0)
        return;

    int chunk_ratio = p["chunk"];

    /* =========================
     * 1. 读取 weight
     * ========================= */
    auto label_weight = ETERNAL_PREFIX + prim_name + "_w";

    if (SPEC_LOAD_STATIC == "layer") {
        checkStaticData(context, dram_time, data_chunk_addr["weight"],
                        GetFromPairedVector(data_chunk, "weight") / chunk_ratio,
                        label_weight, false);
    } else if (SPEC_LOAD_STATIC == "single") {
        checkStaticData(context, dram_time, data_chunk_addr["weight"],
                        GetFromPairedVector(data_chunk, "weight") / chunk_ratio,
                        label_weight, false);
    } else if (SPEC_LOAD_STATIC == "partial") {
        int mac_size = 64 * 1024;
        checkStaticDataTile(context, dram_time, data_chunk_addr["weight"],
                            GetFromPairedVector(data_chunk, "weight") /
                                chunk_ratio,
                            label_weight, false, mac_size);
    }

    /* =========================
     * 2. per-request MLA low-rank
     * ========================= */
    for (auto stage : prim_context->batch_info_) {
        /* ---- 2.1 计算该 request 的 T ---- */
        int T_req = 0;
        switch (p["job_type"]) {
        case JOB_PREFILL:
        case JOB_BOTH:
            T_req = stage.token_num;
            break;
        case JOB_DECODE:
            T_req = 1;
            break;
        default:
            assert(false && "Unsupported job type");
        }

        if (T_req == 0)
            continue;

        /* ---- 2.2 low-rank label（sprintf 风格） ---- */
        char format_label_lr[100];
        sprintf(format_label_lr, "%s%s#%d", ETERNAL_PREFIX,
                (prim_name + "_low_rank").c_str(), stage.req_id);
        string label_low_rank = format_label_lr;

        int append_size = data_byte * p["B"] * T_req * p["OC"];

#if USE_SRAM_MANAGER == 1
        sram_update_cache(context, label_low_rank,
                          prim_context->sram_pos_locator_, append_size,
                          dram_time, prim_context->cid);
#else
        sram_write_append_generic(context, append_size, dram_time);
        prim_context->sram_pos_locator_->updatePair(label_low_rank, append_size,
                                                    context, dram_time);
#endif

        AddrPosKey lrcache;
        prim_context->sram_pos_locator_->findPair(label_low_rank, lrcache);
        int low_rank_size = lrcache.size / (p["B"] * p["OC"]);

        uint64_t X = 0;
        if (low_rank_size > 0) {
            X = low_rank_size / (data_byte * p["B"] * p["OC"]);
        }

        exu_ops +=
            (uint64_t)p["B"] * T_req * p["C"] * p["OC"] * 2 +
            (uint64_t)p["B"] * T_req * p["C"] * X * p["NH"] * p["DH"] * 4;
    }

    sfu_ops = 0;
    vec_ops = 0;
}