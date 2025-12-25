#include "defs/global.h"
#include "common/memory.h"
#include "defs/enums.h"

u_int64_t dcache_hits = 0;
u_int64_t dcache_misses = 0;
u_int64_t dcache_evictions = 0;

vector<pair<int, CoreHWConfig *>> g_core_hw_config;
vector<PrimBase *> g_prim_stash;
vector<chip_instr_base*> g_chip_prim_stash;
AddrLabelTable g_addr_label_table;
DramKVTable** g_dram_kvtable;
unordered_map<int, int> g_core_remap;

// 记录所有在计算原语中的参数，见test文件夹下的config文件
vector<pair<string, int>> vtable;

u_int64_t g_data_footprint_in_words;

bool correct_exit = true;

#if DUMMY == 1
uint32_t *dram_array;
#else
// used for DRAM on cores
// uint32_t *dram_array[GRID_SIZE];
#endif

#if DCACHE == 1
// u_int16_t * dcache_freq;
std::unordered_map<u_int64_t, u_int16_t> dcache_freq_v2;
std::unordered_set<uint64_t> *dcache_dirty;
uint64_t **dcache_tags;
uint32_t *dcache_occupancy;
uint32_t *dcache_last_evicted;
#endif

u_int64_t *mc_transactions;
u_int64_t *mc_latency;
u_int64_t *mc_writebacks;
u_int32_t ***frame_counters;

std::unordered_map<int, std::ofstream*> g_log_streams;