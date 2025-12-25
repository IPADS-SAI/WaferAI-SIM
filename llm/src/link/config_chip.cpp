#include "link/config_chip.h"
#include "link/config_base.h"
#include "monitor/config_helper_core.h"
#include "monitor/config_helper_gpu.h"

#include "nlohmann/json.hpp"
#include "utils/system_utils.h"
#include <string>
#include <vector>

using json = nlohmann::json;

ChipConfig::ChipConfig() {}


ChipConfig::~ChipConfig() {}

void ChipConfig::printSelf() {
    // chip->printSelf();
    std::cout << "chip_id: " << id << std::endl;
    std::cout << "GridX: " << GridX << std::endl;
    std::cout << "GridY: " << GridY << std::endl;
}

void ChipConfig::load_json(const json &j) {
    // std::cout << "load_json" << std::endl;
    if (j.contains("cores_copy")) {
        assert(0 && "cores_copy should not be called here");
    }

    j.at("chip_id").get_to(id);
    j.at("GridX").get_to(GridX);
    j.at("GridY").get_to(GridY);

    // 如果设置了Config的Type，则按照Config的Type来初始化
    //[yicheng] TODO ：这里的初始化方式不太好，回头再看看
    if (j.contains("core_type")) {
        if (j.at("core_type") == "dataflow") {
            chip = new config_helper_core(top_config->filename, id);
        } else if (j.at("core_type") == "gpu") {
            chip = new config_helper_gpu(top_config->filename, id);
        }
    } else {
        if (SYSTEM_MODE == SIM_DATAFLOW) {
            chip = new config_helper_core(top_config->filename, id);
        } else if (SYSTEM_MODE == SIM_GPU) {
            chip = new config_helper_gpu(top_config->filename, id);
        }
    }
}

void from_json(const json &j, ChipConfig &c) {
    assert(0 && "not implemented yet");
}
