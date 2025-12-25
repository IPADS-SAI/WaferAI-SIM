#include "GPU_L1L2_Cache.h"

int sc_main(int argc, char *argv[]) {
    // 创建缓存系统
    CacheSystem system("cache_system", NUML1Caches);

    // 开始仿真
    sc_start(1000, SC_US);

    return 0;
}