#include "systemc.h"

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"

REGISTER_PRIM(Merge_conv);

void Merge_conv::initialize() {
    LOG_ERROR(PRIM) << "Merge_conv::initialize() not implemented";
}

void Merge_conv::taskCore(TaskCoreContext &context, string prim_name,
                         u_int64_t &dram_time, u_int64_t &exu_ops,
                         u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    LOG_ERROR(PRIM) << "Merge_conv::taskCore() not implemented";
}