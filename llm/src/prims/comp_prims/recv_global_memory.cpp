#include "systemc.h"
#include <tlm>

#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Recv_global_memory);

void Recv_global_memory::initialize() {
    LOG_ERROR(PRIM) << "Recv_global_memory::initialize() not implemented";
}

void Recv_global_memory::taskCore(TaskCoreContext &context, string prim_name,
                                 u_int64_t &dram_time, u_int64_t &exu_ops,
                                 u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    LOG_ERROR(PRIM) << "Recv_global_memory not implemented";
}