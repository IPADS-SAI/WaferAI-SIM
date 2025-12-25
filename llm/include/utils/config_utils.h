#pragma once
#include "common/config.h"
#include "nlohmann/json.hpp"
#include "utils/print_utils.h"

using json = nlohmann::json;

int GetDefinedParam(string var);
void ParseSimulationType(json j);
void ParseSimulationConfig(json j);
void ParseHardwareConfig(json j);
void ParseMemorySpec(string filename);
void CoreConfigRemap(
    vector<pair<int, int>> &source_info, vector<CoreConfig> &coreconfigs);

template <typename T> void SetParamFromJson(json j, string field, T *target) {
    if (j.contains(field)) {
        json value = j[field];

        if (value.is_number_integer()) {
            *target = value;
            return;
        }

        if (value.is_boolean()) {
            *target = value.get<bool>();
            return;
        }

        for (auto v : vtable) {
            if (v.first == value) {
                *target = v.second;
                return;
            }
        }

        LOG_ERROR(config_utils.h) << "Undefined variable " << value;
    } else
        LOG_ERROR(config_utils.h) << "Undefined field " << field << " in json";
}

template <typename T>
void SetParamFromJson(json j, string field, T *target, T default_value) {
    if (j.contains(field)) {
        json value = j[field];

        if (value.is_number_integer()) {
            *target = value;
            return;
        }

        if (value.is_boolean()) {
            *target = value.get<bool>();
            return;
        }

        for (auto v : vtable) {
            if (v.first == value) {
                *target = v.second;
                return;
            }
        }

        *target = default_value;
    } else
        *target = default_value;
}