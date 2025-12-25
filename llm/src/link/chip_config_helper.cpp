#include "link/chip_config_helper.h"
#include <fstream>
#include "link/instr/recv_global_mem.h"
#include "link/instr/print_msg.h"
#include "link/instr/chip_prim_utils.h"
chip_config_helper::chip_config_helper(string filename, int cid){
    this->cid = cid;

    cout << "Loading chip config " << filename << endl;
    json j;

    ifstream jfile(filename);
    jfile >> j;

    auto config_vars = j["vars"];
    for (auto var : config_vars.items()) {
        vtable.push_back(make_pair(var.key(), var.value()));
    }

    // Parse global_interface prims
    if (j.contains("chips") && j["chips"].is_array() && cid < (int)j["chips"].size()) {
        auto &chip = j["chips"][cid];
        if (chip.contains("global_interface") && chip["global_interface"].contains("prims")) {
            auto &prims = chip["global_interface"]["prims"];
            for (const auto &p : prims) {
                chip_instr_base *instr = nullptr;
                string type = p.at("type");

                instr = new_chip_prim(type);
                instr->parseJson(p);
                instr_list.push_back(instr);
                // instr->parseJson(p);

                // if (!p.contains("type")) continue;
                // string type = p["type"].get<string>();
                // if (type == "Recv_global_memory") {
                //     int seq = p.contains("seq") ? p["seq"].get<int>() : -1;
                //     Recv_global_mem *instr = new Recv_global_mem(seq);
                //     instr_list.push_back(instr);
                // } else if (type == "Print_msg") {
                //     int seq = p.contains("seq") ? p["seq"].get<int>() : -1;
                //     string msg = p.contains("message") ? p["message"].get<string>() : "";
                //     Print_msg *instr = new Print_msg(seq, msg);
                //     instr_list.push_back(instr);
                // }
            }
        }
    }

    printSelf();
}


void chip_config_helper::printSelf(){
    cout << "<ChipConfigHelper>\n";
    cout << "\tchip id: " << cid << endl;
    cout << "\t<instr_list>\n";
    for(auto instr : instr_list){
        instr->printSelf("\t\t");
    }
    cout << "\t</instr_list>\n";

    cout << "</ChipConfigHelper>\n";
}