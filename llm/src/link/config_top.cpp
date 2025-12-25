#include "link/config_top.h"

#include "nlohmann/json.hpp"
#include "utils/system_utils.h"
#include "utils/config_utils.h"
#include <string>
#include <vector>

using json = nlohmann::json;

TopConfig::TopConfig(std::string filename)
    : filename(filename) {
    std::cout << "Loading Top config from " << filename << std::endl;
    std::ifstream file(filename);
    json j;
    file >> j;

    auto config_vars = j["vars"];
    for (auto var : config_vars.items()) {
        vtable.push_back(make_pair(var.key(), var.value()));
    }

    auto config_source = j["source"];
    for (auto source : config_source) {
        if (source.contains("loop")) {
            int loop_cnt = GetDefinedParam(source["loop"]);
            for (int i = 0; i < loop_cnt; i++) {
                source_info.push_back(
                    make_pair(source["dest"], GetDefinedParam(source["size"])));
            }
        } else {
            source_info.push_back(
                make_pair(source["dest"], GetDefinedParam(source["size"])));
        }
    }

    if (j.contains("pipeline")) {
        j.at("pipeline").get_to(pipeline);
    } else {
        pipeline = 1;
    }

    if (j.contains("sequential")) {
        j.at("sequential").get_to(sequential);
    } else {
        sequential = false;
    }

    auto config_chips = j["chips"];
    for (int i = 0; i < config_chips.size(); i++) {
        //[MYONIE] TODO : 可以考虑换成unique_ptr
        json j = config_chips[i];
        if (j.contains("cores_copy")) {
            int cores_copy = j.at("cores_copy");
            assert(cores_copy >= 0 && cores_copy < component_.size() &&
                   "cores_copy is out of range");
            auto chip_ptr =
                dynamic_cast<ChipConfig *>(component_[cores_copy])->deep_copy();
            // auto chip_ptr = chip_[cores_copy]->deep_copy();
            // auto chip_ptr = new ChipConfig(*chip_[cores_copy]);
            chip_ptr->id = j.at("chip_id");
            // chip_.push_back(chip_ptr);
            component_.push_back(chip_ptr);
        } else {
            auto chip_ptr = new ChipConfig(this, this);
            chip_ptr->load_json(j);
            // chip_.push_back(chip_ptr);
            component_.push_back(chip_ptr);
        }
    }
}

TopConfig::~TopConfig() {
    // TODO
}

void from_json(const json &j, TopConfig &c) {
    auto config_vars = j["vars"];
    for (auto var : config_vars.items()) {
        vtable.push_back(make_pair(var.key(), var.value()));
    }
}

void TopConfig::printSelf() {
    std::cout << "TopConfig: " << std::endl;
    std::cout << "pipeline: " << pipeline << std::endl;
    std::cout << "sequential: " << sequential << std::endl;
    std::cout << "source_info: " << std::endl;
    for (auto source : source_info) {
        std::cout << "  " << source.first << " " << source.second << std::endl;
    }

    for (auto component_ptr : component_) {
        component_ptr->printSelf();
    }
}