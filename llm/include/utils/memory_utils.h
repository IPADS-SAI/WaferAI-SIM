#pragma once
#include "common/system.h"
#include "workercore/workercore.h"

void sram_first_write_generic(TaskCoreContext &context, int data_size_in_byte,
                              u_int64_t global_addr, u_int64_t &dram_time,
                              float *dram_start,
                              std::string label_name = "default_key",
                              bool use_manager = false,
                              SramPosLocator *sram_pos_locator = nullptr,
                              bool dummy_alloc = false,
                              bool add_dram_addr = true);
void sram_spill_back_generic(TaskCoreContext &context, int data_size_in_byte,
                             u_int64_t global_addr, u_int64_t &dram_time);
void sram_read_generic_temp(TaskCoreContext &context, int data_size_in_byte,
                            int sram_addr_offset, u_int64_t &dram_time);
void sram_read_generic(TaskCoreContext &context, int data_size_in_byte,
                       int sram_addr_offset, u_int64_t &dram_time,
                       AllocationID alloc_id = 0, bool use_manager = false,
                       SramPosLocator *sram_pos_locator = nullptr,
                       int start_offset = 0);
void sram_write_append_generic(TaskCoreContext &context, int data_size_in_byte,
                               u_int64_t &dram_time,
                               std::string label_name = "default_key",
                               bool use_manager = false,
                               SramPosLocator *sram_pos_locator = nullptr,
                               u_int64_t global_addr = 0);
void sram_write_back_temp(TaskCoreContext &context, int data_size_in_byte,
                          int &temp_sram_addr, u_int64_t &dram_time);
void sram_update_cache(TaskCoreContext &context, string label_k,
                       SramPosLocator *sram_pos_locator, int data_size_in_byte,
                       u_int64_t &dram_time, int cid);
void check_freq(std::unordered_map<u_int64_t, u_int16_t> &freq, u_int64_t *tags,
                u_int32_t set, u_int64_t elem_tag);
void sram_write(TaskCoreContext &context, int dma_read_count,
                int sram_addr_temp, AllocationID alloc_id, bool use_manager);

#if DCACHE
int dcache_replacement_policy(u_int32_t tileid, u_int64_t *tags,
                              u_int64_t new_tag, u_int16_t &set_empty_lines);
#endif

#if USE_L1L2_CACHE == 1
void gpu_read_generic(TaskCoreContext &context, uint64_t addr, int size,
                      int &mem_time, bool cache_read = false);
void gpu_write_generic(TaskCoreContext &context, uint64_t addr, int size,
                       int &mem_time, bool cache_write = true);
#endif

TaskCoreContext generate_context(WorkerCoreExecutor *workercore);

// 在内存层级中使用
uint64_t get_bank_index(uint64_t address);