#pragma once
#include "defs/enums.h"
#include "macros/macros.h"
#include "systemc.h"

#include "../unit_module/dram_kvtable/dram_kvtable.h"
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// 原语数组
class PrimBase;
class chip_instr_base;
extern vector<PrimBase *> g_prim_stash;
extern vector<chip_instr_base *> g_chip_prim_stash;

// kvcache管理表
class DramKVTable;
extern DramKVTable **g_dram_kvtable;

// 收集所有原语
class AddrLabelTable;
extern AddrLabelTable g_addr_label_table;

// 所有计算核心的硬件配置
class ExuConfig;
class SfuConfig;
class CoreHWConfig;
extern vector<pair<int, CoreHWConfig *>> g_core_hw_config;

// 重新映射计算核的编号表
extern unordered_map<int, int> g_core_remap;

// 输出流，用于打印
extern std::unordered_map<int, std::ofstream *> g_log_streams;

// 记录所有在计算原语中的参数，由json读取
extern vector<pair<string, int>> vtable;

extern u_int64_t g_data_footprint_in_words;

extern bool correct_exit; // 程序是否正常退出

// 模拟dram数组
#if DUMMY == 1
extern uint32_t *dram_array;
#endif

// dcache相关
#if DCACHE == 1
extern std::unordered_map<u_int64_t, u_int16_t> dcache_freq_v2;

extern std::unordered_set<uint64_t> *dcache_dirty;
extern uint64_t **dcache_tags;
extern uint32_t *dcache_occupancy;
extern uint32_t *dcache_last_evicted;
#endif

extern u_int64_t dcache_hits;
extern u_int64_t dcache_misses;
extern u_int64_t dcache_evictions;
extern u_int64_t *mc_transactions;
extern u_int64_t *mc_latency;
extern u_int64_t *mc_writebacks;
extern u_int32_t ***frame_counters;