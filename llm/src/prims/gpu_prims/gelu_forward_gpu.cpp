#include "prims/gpu_prims.h"
#include "utils/memory_utils.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Gelu_f_gpu);

void Gelu_f_gpu::initialize() {
    if (datatype == INT8)
        data_byte = 1;
    else if (datatype == FP16)
        data_byte = 2;

    auto &p = param_value;
    input_size = {(int)(data_byte * (u_int64_t)p["N"])};
    data_chunk = {
        {"output", (int)(data_byte * (u_int64_t)p["N"] / (p["slice_x"] * p["slice_y"]))}};
}


int Gelu_f_gpu::taskCoreDefault(TaskCoreContext &context) {
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
        LOG_ERROR(gelu_forward_gpu.cpp)
            << name << " of Core " << context.cid << " cannot find "
            << prim_context->datapass_label_->indata[0];
    }

    int overlap_time = 0;
#if USE_L1L2_CACHE == 1
    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " read input";
    gpu_read_generic(context,
                     input_mem_offset + input_size /
                                            (p["slice_x"] * p["slice_y"]) *
                                            fetch_index,
                     input_size / (p["slice_x"] * p["slice_y"]), mem_time);

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
    int cid = context.cid;

    CoreHWConfig *hardware_config = GetCoreHWConfig(cid);
    ExuConfig *exu = hardware_config->exu;
    SfuConfig *sfu = hardware_config->sfu;

    if (exu->type == MAC_Array)
        cycle += 0 / (p["slice_x"] * p["slice_y"]) /
                 (exu->x_dims * exu->x_dims * 2 * HW_COMP_UTIL) * CYCLE;
    else
        assert(false && "Unsupported tile type");

    if (sfu->type == Linear)
        cycle += p["N"] / (p["slice_x"] * p["slice_y"]) / sfu->x_dims * CYCLE;
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

GpuBase *Gelu_f_gpu::clone() { return new Gelu_f_gpu(*this); }