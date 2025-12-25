#include "nlohmann/json.hpp"

#include "defs/const.h"
#include "defs/enums.h"
#include "defs/global.h"
#include "prims/base.h"
#include "prims/base.h"
#include "utils/prim_utils.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>


using namespace std;


using json_order = nlohmann::ordered_json;


class CorePrims {
public:
    vector<PrimBase *> prims;


    void printSelf();
    CorePrims() {}
};

void from_json(const json &j, CorePrims &c) {


    if (j.contains("prims")) {
        auto prims = j["prims"];
        for (auto prim : prims) {
            CompBase *p = nullptr;
            string type = prim.at("type");

            p = (CompBase *)new_prim(type);
            p->parseJson(prim);
            p->initialize();

            c.prims.push_back((PrimBase *)p);
        }
    }
}


struct prim_dram_info {
    string name;
    int input;
    int data;
    int out;
};


int sc_main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <workload_config_path>" << std::endl;
        return 1;
    }

    std::string configFilePath = argv[1];
    std::cout << "Config file path: " << configFilePath << std::endl;
    json j;

    ifstream jfile(configFilePath);

    if (!jfile.is_open()) {
        std::cerr << "Failed to open config file." << std::endl;
        return 1;
    }

    try {
        jfile >> j;
    } catch (const json::parse_error &e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return 1;
    }

    // vector<pair<string, int>> vtable;
    vector<prim_dram_info> prims_dram_config;

    // 收集相关参数
    auto config_vars = j["vars"];
    for (auto var : config_vars.items()) {
        vtable.push_back(make_pair(var.key(), var.value()));
    }

    CorePrims config_prims = j;


    int core_dram = 0;


    for (auto p : config_prims.prims) {

        if (p == config_prims.prims.front()) {

            prim_dram_info tmp;
            tmp.name = p->name;
            tmp.input = core_dram;
            core_dram += p->dram_inp_size;
            tmp.data = core_dram;
            core_dram += p->dram_data_size;
            tmp.out = core_dram;
            core_dram += p->dram_out_size;
            prims_dram_config.push_back(tmp);
        } else {

            prim_dram_info tmp;
            tmp.name = p->name;
            tmp.input = -1;
            tmp.data = core_dram;
            core_dram += p->dram_data_size;
            tmp.out = core_dram;
            core_dram += p->dram_out_size;
            prims_dram_config.push_back(tmp);
        }
    }


    json_order dram_json_array = json_order::array();

    for (const auto &info : prims_dram_config) {
        json_order prim_obj;
        json_order dram_address_obj;

        // 构建 dram_address 子对象
        dram_address_obj["input"] = info.input;
        dram_address_obj["data"] = info.data;
        dram_address_obj["out"] = info.out;

        // 设置顶层字段
        prim_obj["name"] = info.name;
        prim_obj["dram_address"] = dram_address_obj;

        // 添加到数组
        dram_json_array.push_back(prim_obj);
    }

    // Write to file
    std::ofstream out_file("dram_config.json");
    if (out_file.is_open()) {
        out_file << dram_json_array.dump(
            4); // pretty print with indent of 4 spaces
        out_file.close();
        std::cout << "Successfully saved dram_config.json" << std::endl;
    } else {
        std::cerr << "Failed to open output file for writing." << std::endl;
    }


    return 0;
}