#include "prims/gpu_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Layernorm_f_gpu);

void Layernorm_f_gpu::initialize() {
    if (datatype == INT8)
        data_byte = 1;
    else if (datatype == FP16)
        data_byte = 2;

    auto &p = param_value;
    data_size_input = {(int)(data_byte * (u_int64_t)p["B"] * p["T"] * p["C"])};
    data_chunk = {{"weight", (int)(data_byte * (u_int64_t)p["C"])},
                  {"bias", (int)(data_byte * (u_int64_t)p["C"])},
                  {"output", (int)(data_byte * (u_int64_t)p["B"] * p["T"] * p["C"] /
                                 (p["slice_x"] * p["slice_y"]))}};
}

int Layernorm_f_gpu::taskCoreDefault(TaskCoreContext &context) {
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
        LOG_ERROR(layernorm_forward_gpu.cpp)
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

    auto label_weight = prefix + "_w";
    AddrPosKey w_key;
    prim_context->gpu_pos_locator_->fetchPair(label_weight, w_key);

    auto label_bias = prefix + "_b";
    AddrPosKey b_key;
    prim_context->gpu_pos_locator_->fetchPair(label_bias, b_key);

    int overlap_time = 0;
#if USE_L1L2_CACHE == 1
    // 通过fetch_index计算位置
    int row_index = fetch_index / p["slice_x"];
    int col_index = fetch_index % p["slice_x"];

    // input 读入
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read input";
    gpu_read_generic(context,
                     input_mem_offset + input_size / p["slice_y"] * row_index,
                     input_size / p["slice_y"], mem_time);

    // weight 读入
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read weight";
    gpu_read_generic(context, w_key.pos + w_key.size / p["slice_x"] * col_index,
                     GetFromPairedVector(data_chunk, "weight") / p["slice_x"],
                     mem_time);

    // bias 读入
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid << " read bias";
    gpu_read_generic(context, b_key.pos + b_key.size / p["slice_x"] * col_index,
                     GetFromPairedVector(data_chunk, "bias") / p["slice_x"],
                     mem_time);

    // TODO: 模拟计算cycle数
    // overlap_time = mem_time;
    AddrPosKey out_key;
    prim_context->gpu_pos_locator_->updatePair(
        prim_context->datapass_label_->outdata,
        GetFromPairedVector(data_chunk, "output"));

    prim_context->gpu_pos_locator_->findPair(
        prim_context->datapass_label_->outdata, out_key);

    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " write output";
    gpu_write_generic(context,
                      out_key.pos + GetFromPairedVector(data_chunk, "output") *
                                        fetch_index,
                      GetFromPairedVector(data_chunk, "output"), mem_time);

    int cycle = 0;

    CoreHWConfig *hardware_config = GetCoreHWConfig(prim_context->cid);
    ExuConfig *exu = hardware_config->exu;
    SfuConfig *sfu = hardware_config->sfu;

    if (exu->type == MAC_Array)
        cycle += 0 / (exu->x_dims * exu->x_dims * 2 * HW_COMP_UTIL) * CYCLE;
    else
        assert(false && "Unsupported tile type");

    if (sfu->type == Linear)
        cycle += p["B"] * p["T"] * (8 * p["C"] + 5) /
                 (p["slice_x"] * p["slice_y"]) / sfu->x_dims * CYCLE;
    else
        assert(false && "Unsupported tile type");


    if (mem_time > cycle) {
        // 因为dram 已经wait 过了，所以额外的 overlap_time = 0
        overlap_time = 0;
        LOG_INFO(PRIM) << name << " of Core " << context.cid << ": dram_time "
                       << mem_time << ", compute cycle " << cycle;
    } else {
        overlap_time = cycle - mem_time;
        LOG_INFO(PRIM) << name << " of Core " << context.cid << ": dram_time "
                       << mem_time << ", compute cycle " << cycle;
    }
#endif
    return overlap_time;
}

GpuBase *Layernorm_f_gpu::clone() { return new Layernorm_f_gpu(*this); }