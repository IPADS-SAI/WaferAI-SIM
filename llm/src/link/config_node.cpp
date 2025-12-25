#include "link/config_node.h"

#include "nlohmann/json.hpp"
#include "utils/system_utils.h"
#include <string>
#include <vector>

using json = nlohmann::json;

void NodeConfig::printSelf() {
    for (auto chip : chips) {
        chip.printSelf();
    }
}

void from_json(const json &j, NodeConfig &c) {
    // TODO
}