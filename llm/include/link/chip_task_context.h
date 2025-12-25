#pragma once 

#include "memory/sram/High_mem_access_unit.h"
#include "memory/sram/Mem_access_unit.h"
#include "link/chip_global_memory.h"  // for ChipGlobalMemory
// Forward-declare to avoid pulling in heavy headers
class GlobalMemInterface;
class ChipGlobalMemory;


class TaskChipContext{
public:

    TaskChipContext() : mau(nullptr), chipGlobalMemory(nullptr) {}
    mem_access_unit *mau;
    ChipGlobalMemory *chipGlobalMemory; // pointer to global memory interface
};

TaskChipContext generate_chip_context(GlobalMemInterface *global_mem_interface);