#include "prims/base.h"
#include "utils/config_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

vector<sc_bv<128>> GpuBase::serialize() {
    LOG_DEBUG(CONFIG_DEBUG) << "Start serialize " << name;

    vector<sc_bv<128>> segments;

    // metadata
    sc_bv<128> metadata;
    metadata.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata.range(8, 8) = sc_bv<1>(datatype);
    metadata.range(24, 9) = sc_bv<16>(fetch_index);
    metadata.range(56, 41) = sc_bv<16>(req_sm);
    segments.push_back(metadata);

    std::vector<std::pair<std::string, int>> vec(param_value.begin(),
                                                 param_value.end());
    std::sort(vec.begin(), vec.end(),
              [](auto &a, auto &b) { return a.first < b.first; });

    // 规定一个参数使用32位存储，即一个segment存储4个参数
    for (auto it = vec.begin(); it != vec.end();) {
        sc_bv<128> d;
        d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
        int pos = 8;
        for (int i = 0; i < 4 && it != vec.end(); i++, it++, pos += 30) {
            d.range(pos + 29, pos) = sc_bv<30>(it->second);
            LOG_DEBUG(CONFIG_DEBUG) << "In serialize " << name << ": " << it->first
                              << " = " << it->second;
        }

        segments.push_back(d);
    }

    return segments;
}

void GpuBase::deserialize(vector<sc_bv<128>> segments) {
    LOG_DEBUG(CONFIG_DEBUG) << "Start deserialize " << name;

    // 解析metadata
    auto buffer = segments[0];
    datatype = DATATYPE(buffer.range(8, 8).to_uint64());
    fetch_index = buffer.range(24, 9).to_uint64();
    req_sm = buffer.range(56, 41).to_uint64();

    vector<string> vec(param_name.begin(), param_name.end());
    sort(vec.begin(), vec.end());

    // 依次解析参数，每一个segment存储4个参数
    if (segments.size() - 1 != (vec.size() + 3) / 4)
        LOG_ERROR(gpu_base.cpp) << "In deserialize " << name
                                << ": the number of segments does not match "
                                   "the number of parameters";

    for (int i = 1; i < segments.size(); i++) {
        auto buffer = segments[i];
        for (int j = 0; j < 4; j++) {
            int index = (i - 1) * 4 + j;
            if (index >= vec.size())
                break;
            param_value[vec[index]] =
                buffer.range(29 + j * 30, j * 30 + 8).to_uint64();
            LOG_DEBUG(CONFIG_DEBUG) << "In deserialize " << name << ": " << vec[index]
                              << " = " << param_value[vec[index]];
        }
    }

    initialize();
    initializeDefault();

    LOG_DEBUG(CONFIG) << "Finish deserialize " << name;
}

void GpuBase::parseCompose(json j) {
    SetParamFromJson(j, "require_sm", &req_sm);
}

void GpuBase::parseAddress(json j) {
    string in_label = j["indata"];
    prim_context->datapass_label_->outdata = j["outdata"];

    std::vector<std::string> in_labels;

    std::istringstream iss(in_label);
    std::string word;
    std::string temp;

    // 保证DRAM_LABEL后面跟着另一个label
    while (iss >> word) {
        if (word == DRAM_LABEL || word == "_" + string(DRAM_LABEL)) {
            temp = word;
            if (iss >> word) {
                temp += " " + word;
            }
            in_labels.push_back(temp);
        } else {
            in_labels.push_back(word);
        }
    }

    for (int i = 0; i < in_labels.size(); i++) {
        prim_context->datapass_label_->indata[i] = in_labels[i];
    }
}

void GpuBase::parseJson(json j) {
    for (auto &param : param_name) {
        SetParamFromJson(j, param, &param_value[param]);
    }

    initialize();
    initializeDefault();

    if (j.contains("compose"))
        parseCompose(j["compose"]);

    if (j.contains("address"))
        parseAddress(j["address"]);
}

void GpuBase::initializeDefault() {
    if (datatype == INT8)
        data_byte = 1;
    else if (datatype == FP16)
        data_byte = 2;

    input_size = 0;
    for (auto &input : data_size_input)
        input_size += input;

    out_size = -1;
    for (const auto &chunk : data_chunk) {
        if (chunk.first == "output") {
            out_size = chunk.second;
            break;
        }
    }

    if (out_size < 0) {
        LOG_ERROR(gpu_base.cpp) << "No output chunk found";
    }
}

void GpuBase::printSelf() {}