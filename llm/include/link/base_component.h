#pragma once
#include "systemc.h"

#include "../router/router.h"
#include "../workercore/workercore.h"
#include "link/chip_global_memory.h"
#include "monitor/gpu_cache_system.h"
#include "monitor/mem_interface.h"
#include "trace/Event_engine.h"
using namespace std;

// Monitor类的基类
class BaseComponent {
public:
    enum Type {
        TYPE_TOP,
        TYPE_LINK,
        TYPE_CHIP,
    };

    virtual Type getType() const = 0;
};