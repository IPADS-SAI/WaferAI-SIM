#pragma once

#include "nlohmann/json.hpp"
#include "systemc.h"

#include <vector>

using json = nlohmann::json;

// 用于统一函数接口
class BaseConfig {
public:
    enum Type {
        TYPE_TOP,
        TYPE_CHIP,
        TYPE_NODE,
    };

    virtual Type getType() const = 0;

    virtual void printSelf() = 0;
    virtual ~BaseConfig() {};
};
