#pragma once
#include "defs/enums.h"
#include <string>

// 模拟模式
extern SIM_MODE SYSTEM_MODE;

// 计算核心维度
extern int GRID_X;
extern int GRID_Y;
extern int GRID_SIZE;
extern int CORE_PER_SM;

// simulation config中包含的参数
extern bool SPEC_USE_BEHA_NOC;        // 是否使用行为级NoC
extern bool SPEC_USE_BEHA_SRAM;       // 是否使用行为级SRAM
extern bool SPEC_USE_BEHA_DRAM;       // 是否使用行为级DRAM
extern bool SPEC_USE_PERF_GEMM;       // 是否使用GEMM性能模式
extern bool SPEC_KVCACHE_SPILL;       // 在SRAM满时，是否优先溢出kvcache
extern std::string SPEC_LOAD_STATIC; // 向SRAM加载数据时，加载的策略
extern std::string SPEC_TTF_FILE;     // 字体文件路径
extern bool SPEC_USE_DRAMSYS;         // 是否使用DRAMSys
extern bool SPEC_FAST_WARMUP;         // 是否跳过初始数据发送
extern bool SPEC_ROUTER_PIPE;         // 是否开启路由并行
extern bool SPEC_SEND_RECV_PARALLEL;  // 发送与接收原语是否同时进行

// hardware config中包含的参数
extern int HW_CORE_CREDIT; //  在PD模式中，单核每一拍可安排的任务量
extern int HW_PD_RATIO;    // 在PD模式中，单次Prefill所占任务量（Decode默认为1）
extern int HW_DRAM_DEFAULT_BITWIDTH; // DRAM的默认位宽
extern int HW_SRAM_SIZE;             // SRAM的大小
extern int HW_NOC_PAYLOAD_PER_CYCLE; // 在1cycle内，NoC单信道可传输的数据包数量
extern float HW_COMP_UTIL;
extern float HW_BEHA_DRAM_UTIL;

// gpu相关参数
extern bool GPU_USE_INNER_MM;
extern int GPU_DRAM_BANDWIDTH;
extern bool GPU_CACHE_LOG;
extern std::string GPU_DRAM_CONFIG_FILE;
extern int GPU_DRAM_BURST_SIZE;
extern int GPU_DRAM_ALIGNED;

// log相关参数
extern int LOG_LEVEL;