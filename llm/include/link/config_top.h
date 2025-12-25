#pragma once

#include "link/config_base.h"
#include "nlohmann/json.hpp"
#include "systemc.h"

#include "config_chip.h"
#include "config_node.h"
#include <string>
#include <vector>

using json = nlohmann::json;
class ChipConfig;
class TopConfig : public BaseConfig {
public:
    // TopConfig(std::string filename);
    TopConfig(std::string filename);

    std::string filename;
    // NodeConfig* node;
    // std::vector<ChipConfig*> chip_;
    std::vector<BaseConfig *> component_;

    int seq_index;
    int pipeline;
    bool sequential;
    std::vector<std::pair<int, int>> source_info; // 记录一些全局变量

    void printSelf();
    Type getType() const override { return Type::TYPE_TOP; }

    ~TopConfig();
};