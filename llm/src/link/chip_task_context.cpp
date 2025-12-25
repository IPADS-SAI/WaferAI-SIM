#include "link/chip_task_context.h"
#include "link/global_mem_interface.h"

TaskChipContext generate_chip_context(GlobalMemInterface *global_mem_interface){
    TaskChipContext context;
    // Pass through the ChipGlobalMemory pointer for TLM transactions
    context.chipGlobalMemory = global_mem_interface->chipGlobalMemory;
    return context;
}