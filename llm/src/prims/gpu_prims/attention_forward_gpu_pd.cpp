#include "defs/enums.h"
#include "defs/global.h"
#include "prims/gpu_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"
#include <regex>

REGISTER_PRIM(attention_forward_gpu_pd);

void attention_forward_gpu_pd::initialize() {
    if (datatype == INT8)
        data_byte = 1;
    else if (datatype == FP16)
        data_byte = 2;

    auto &p = param_value;
    data_size_input = {(int)(data_byte * (u_int64_t)p["B"] * p["T"] * p["C"])};
    data_chunk = {{"preatt", (int)(data_byte * (u_int64_t)p["B"] * p["NH"] * p["T"] * p["T"])},
                  {"att", (int)(data_byte * (u_int64_t)p["B"] * p["NH"] * p["T"] * p["T"])},
                  {"output", (int)(data_byte * (u_int64_t)p["B"] * p["NH"] * p["T"] * p["C"] /
                                 (p["slice_x"] * p["slice_y"]))}};
}

int attention_forward_gpu_pd::taskCoreDefault(TaskCoreContext &context) {
    if (prim_context->auto_pd_ &&
        prim_context->loop_cnt > prim_context->auto_pd_) {
        param_value["T"] = 1;
        initialize();
        initializeDefault();
    }

    auto &p = param_value;

    int mem_time = 0;
    auto input_mem_offset = 0;
    if (!prim_context->gpu_pos_locator_->findPair(
            prim_context->datapass_label_->indata[0], input_mem_offset)) {
        LOG_ERROR(attention_forward_gpu_pd.cpp)
            << name << " of Core " << context.cid << " cannot find "
            << prim_context->datapass_label_->indata[0];
    }

    // 获取前缀label
    std::size_t pos = prim_context->datapass_label_->outdata.find_last_of('_');
    std::string prefix;
    if (pos != std::string::npos) {
        prefix = prim_context->datapass_label_->outdata.substr(0, pos);
    } else {
        prefix = prim_context->datapass_label_->outdata;
    }

    auto label_preatt = prefix + "_preatt";
    AddrPosKey p_key = AddrPosKey(0, GetFromPairedVector(data_chunk, "preatt"));
    prim_context->gpu_pos_locator_->fetchPair(label_preatt, p_key);

    auto label_att = prefix + "_att";
    AddrPosKey a_key = AddrPosKey(0, GetFromPairedVector(data_chunk, "att"));
    prim_context->gpu_pos_locator_->fetchPair(label_att, a_key);

    int overlap_time = 0;
#if USE_L1L2_CACHE == 1
    for (auto stage : prim_context->batch_info_) {
        char format_label_k[1000];

        std::regex pattern("attention"); // 因为是字面量，不需要复杂正则

        // 替换为目标字符串
        std::string result = std::regex_replace(prefix, pattern, "matmul");


        sprintf(format_label_k, "%s%s%sk#%d", result.c_str(), ETERNAL_PREFIX,
                KVCACHE_PREFIX, stage.req_id);
        string label_k = format_label_k;
        // cout << "b label_k: " << label_k << endl;

        char format_label_v[1000];
        sprintf(format_label_v, "%s%s%sv#%d", result.c_str(), ETERNAL_PREFIX,
                KVCACHE_PREFIX, stage.req_id);
        string label_v = format_label_v;

        AddrPosKey k_key, v_key;
        prim_context->gpu_pos_locator_->findPair(label_k, k_key);
        prim_context->gpu_pos_locator_->findPair(label_v, v_key);

        gpu_read_generic(
            context,
            k_key.pos +
                k_key.size / (p["slice_x"] * p["slice_y"]) * fetch_index,
            k_key.size / (p["slice_x"] * p["slice_y"]), mem_time, true);
        gpu_read_generic(
            context,
            v_key.pos +
                v_key.size / (p["slice_x"] * p["slice_y"]) * fetch_index,
            v_key.size / (p["slice_x"] * p["slice_y"]), mem_time, true);
        break;
    }

    auto data_size_preatt = GetFromPairedVector(data_chunk, "preatt");
    auto data_size_att = GetFromPairedVector(data_chunk, "att");

    // LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
    //                 << " write preatt";
    // gpu_write_generic(
    //     context,
    //     p_key.pos +
    //         data_size_preatt / (p["slice_x"] * p["slice_y"]) * fetch_index,
    //     data_size_preatt / (p["slice_x"] * p["slice_y"]), mem_time);

    // LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
    //                 << " read preatt";
    // gpu_read_generic(
    //     context,
    //     p_key.pos +
    //         data_size_preatt / (p["slice_x"] * p["slice_y"]) * fetch_index,
    //     data_size_preatt / (p["slice_x"] * p["slice_y"]), mem_time);

    // LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid << " write att";
    // gpu_write_generic(
    //     context,
    //     a_key.pos + data_size_att / (p["slice_x"] * p["slice_y"]) * fetch_index,
    //     data_size_att / (p["slice_x"] * p["slice_y"]), mem_time);

    // LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid << " write att";
    // gpu_read_generic(context,
    //                  a_key.pos + data_size_att / (p["slice_x"] * p["slice_y"]) *
    //                                  fetch_index,
    //                  data_size_att / (p["slice_x"] * p["slice_y"]), mem_time);

    // Q
    gpu_read_generic(context,
                     input_mem_offset + input_size /
                                            (3 * p["slice_x"] * p["slice_y"]) *
                                            fetch_index,
                     input_size / (3 * p["slice_x"] * p["slice_y"]), mem_time);

    // overlap_time = 0;
    AddrPosKey out_key;
    prim_context->gpu_pos_locator_->updatePair(
        prim_context->datapass_label_->outdata,
        GetFromPairedVector(data_chunk, "output"));
    prim_context->gpu_pos_locator_->findPair(
        prim_context->datapass_label_->outdata, out_key);

    gpu_write_generic(context, out_key.pos,
                      GetFromPairedVector(data_chunk, "output"), mem_time);
    u_int64_t cycle = 0;
    int cid = context.cid;

    CoreHWConfig *hardware_config = GetCoreHWConfig(cid);
    ExuConfig *exu = hardware_config->exu;
    SfuConfig *sfu = hardware_config->sfu;

    if (exu->type == MAC_Array)
        cycle += (u_int64_t)p["B"] * p["NH"] * p["T"] * (p["T"] - 1) / 2 *
                 (4 * p["C"] / p["NH"] + 5) / (p["slice_x"] * p["slice_y"]) /
                 (exu->x_dims * exu->x_dims * 2 * HW_COMP_UTIL) * CYCLE;
    else
        assert(false && "Unsupported tile type");

    if (sfu->type == Linear)
        cycle += 0 / (p["slice_x"] * p["slice_y"]) / sfu->x_dims * CYCLE;
    else
        assert(false && "Unsupported tile type");


    if (mem_time > cycle) {
        // 因为dram 已经wait 过了，所以额外的 overlap_time = 0
        overlap_time = 0;
        LOG_INFO(PRIM) << name << " of Core " << context.cid << ": dram_time "
                        << mem_time  << ", compute cycle " 
                       << cycle ;
    } else {
        overlap_time = cycle - mem_time;
        LOG_INFO(PRIM) << name << " of Core " << context.cid << ": dram_time "
                        << mem_time  << ", compute cycle "
                        << cycle ;
    }
#endif

    return overlap_time;
}

GpuBase *attention_forward_gpu_pd::clone() {
    return new attention_forward_gpu_pd(*this);
}