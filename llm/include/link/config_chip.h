#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "systemc.h"

#include "link/config_base.h"
#include "link/config_top.h"
#include "monitor/config_helper_base.h"
#include "monitor/config_helper_core.h"

using json = nlohmann::json;
class TopConfig;

class ChipConfig : public BaseConfig {
public:
    int id;
    int GridX, GridY; //[MYONIE] TODO : 现在GridX和GridY都是全局变量
    BaseConfig *parent_config;
    TopConfig *top_config;
    std::vector<std::pair<int, int>> source_info;
    config_helper_base *chip;
    void printSelf();

    ChipConfig();
    ~ChipConfig();

    ChipConfig(TopConfig *top_config) : top_config(top_config) {}
    // virtual ~ChipConfig(){};

    ChipConfig(TopConfig *top_config, BaseConfig *parent_config)
        : top_config(top_config), parent_config(parent_config) {}

    ChipConfig *deep_copy() const {
        // ChipConfig* copy = new ChipConfig(top_config, parent_config);
        // copy->id = id;
        // copy->GridX = GridX;
        // copy->GridY = GridY;
        // copy->source_info = source_info;
        auto chip_ptr = new ChipConfig(*this);

        if (chip != nullptr) {
            chip_ptr->chip = chip->clone();
        } else {
            chip_ptr->chip = nullptr;
        }
        return chip_ptr;
    }

    // load json函数
    void load_json(const json &j);
    Type getType() const override { return TYPE_CHIP; }
};

void from_json(const json &j, ChipConfig &c);