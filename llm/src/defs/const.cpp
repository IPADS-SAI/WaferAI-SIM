#include "defs/const.h"
#include "macros/macros.h"

const double pu_freq = 1.0;
const double mem_freq = 1.0;
const double noc_freq = 1.0;
const double pu_mem_ratio = pu_freq / mem_freq;
const double noc_to_pu_ratio = pu_freq / noc_freq;

u_int64_t dataset_words_per_tile = 1024 * 1024 * 16;
u_int32_t dcache_size = 128;
u_int32_t dcache_words_in_line_log2 = 4; // 16 words, i.e., 64B

u_int32_t hbm_channels = 8;
u_int32_t total_hbm_channels = hbm_channels * DIES;

bool dataset_cached = true;

u_int32_t dcache_lines = dcache_size >> dcache_words_in_line_log2;
u_int32_t dcache_sets = (dcache_lines >> CACHE_WAY_BITS);

#if BANK_DEPTH > 524288
u_int32_t sram_read_latency = ceil_macro(3 * pu_mem_ratio * CYCLE);
#elif BANK_DEPTH > 131072
u_int32_t sram_read_latency = ceil_macro(2 * pu_mem_ratio * CYCLE);
#else
u_int32_t sram_read_latency = ceil_macro(1 * pu_mem_ratio * CYCLE);
#endif

u_int32_t hbm_device_latency = ceil_macro(30 * pu_mem_ratio);
const u_int32_t average_die_to_edge_distance = hops_to_mc(DIE_W);
u_int32_t on_die_hop_latency = 1;

u_int32_t hbm_read_latency =
    hbm_device_latency +
    ceil_macro(noc_to_pu_ratio * average_die_to_edge_distance *
               on_die_hop_latency);

int RAM_READ_LATENCY = sram_read_latency;  // NS
int RAM_WRITE_LATENCY = sram_read_latency; // NS
const double RAM_READ_ENERGY = 0.1;
const double RAM_WRITE_ENERGY = 0.2;

// Visual
const int Men_usage_thre = 2;