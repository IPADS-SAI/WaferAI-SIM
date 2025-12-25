#pragma once
#include "macros/macros.h"
#include "systemc.h"

#include "defs/global.h"
#include "memory/dramsys_wrapper.h"
#include "memory/gpu/GPU_L1L2_Cache.h"

class L1L2CacheSystem : public sc_module {
public:
    // vector<Processor*> processors;
    // vector<L1Cache*> l1Caches;
    L2Cache *l2Cache;
    MainMemory *mainMemory;
    Bus *bus;
    ::DRAMSys::Config::Configuration testConfig;
    gem5::memory::DRAMSysWrapper *dramSysWrapper;

    L1L2CacheSystem(sc_module_name name, int numProcessors,
                    vector<L1Cache *> l1caches,
                    vector<GPUNB_dcacheIF *> processors,
                    std::string_view configuration,
                    std::string_view resource_directory)
        : sc_module(name),
          testConfig(
              ::DRAMSys::Config::from_path(configuration, resource_directory)) {

        l2Cache = new L2Cache("l2_cache", L2CACHESIZE, L2CACHELINESIZE, 8, 16);

        dramSysWrapper = new gem5::memory::DRAMSysWrapper("DRAMSysWrapper",
                                                          testConfig, false);
        if (SYSTEM_MODE == SIM_GPU || SYSTEM_MODE == SIM_GPU_PD) {
            GPU_DRAM_ALIGNED =
                dramSysWrapper->dramsys->getMemSpec().defaultBytesPerBurst;
        }
        assert(DRAM_BURST_BYTE >=
               dramSysWrapper->dramsys->getMemSpec().defaultBytesPerBurst);
        // mainMemory = new MainMemory("main_memory");
        bus = new Bus("bus", numProcessors);

        // 连接组件
        for (int i = 0; i < numProcessors; i++) {
            processors[i]->socket.bind(l1caches[i]->cpu_socket);
            l1caches[i]->bus_socket.bind(*bus->l1_sockets[i]);
            bus->addL1Cache(l1caches[i]);
        }

        bus->l2_socket.bind(l2Cache->bus_socket);
        l2Cache->mem_socket.bind(dramSysWrapper->tSocket);
    }
};