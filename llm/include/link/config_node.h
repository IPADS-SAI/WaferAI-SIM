#pragma once

#include "nlohmann/json.hpp"
#include "systemc.h"

#include "link/config_base.h"
#include "link/config_chip.h"
#include "monitor/config_helper_base.h"
#include "monitor/config_helper_core.h"

#include <vector>
class ChipConfig;

class NodeConfig : public BaseConfig {
public:
    int id;
    BaseConfig *parent_config;
    BaseConfig *top_config;
    std::vector<ChipConfig> chips;
    void printSelf();

    Type getType() const override { return TYPE_NODE; }
};

void from_json(const json &j, NodeConfig &c);