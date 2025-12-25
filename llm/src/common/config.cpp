#include "common/config.h"
#include "common/msg.h"
#include "utils/config_utils.h"
#include "utils/prim_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

void CoreJob::printSelf() {}

void CoreConfig::printSelf() {}

void from_json(const json &j, Cast &c) {
    SetParamFromJson<int>(j, "dest", &(c.dest));
    SetParamFromJson<int>(j, "tag", &(c.tag), c.dest);
    SetParamFromJson<int>(j, "weight", &(c.weight), 1);
    SetParamFromJson<int>(j, "addr", &(c.addr), -1);
    SetParamFromJson<bool>(j, "critical", &(c.critical), false);

    if (!j.contains("loopout"))
        c.loopout = BOTH;
    else {
        if (j.at("loopout") == "false")
            c.loopout = FALSE;
        else if (j.at("loopout") == "true")
            c.loopout = TRUE;
        else
            c.loopout = BOTH;
    }
}

void from_json(const json &j, CoreJob &c) {
    if (j.contains("cast")) {
        auto casts = j["cast"];
        for (int i = 0; i < casts.size(); i++) {
            Cast temp = casts[i];
            c.cast.push_back(temp);
        }
    } else if (SYSTEM_MODE != SIM_DATAFLOW)
        LOG_ERROR(config.cpp) << "Undefined \'cast\' field in json";

    SetParamFromJson<int>(j, "recv_cnt", &(c.recv_cnt));
    SetParamFromJson<int>(j, "recv_tag", &(c.recv_tag), 0);

    if (j.contains("prims")) {
        auto prims = j["prims"];
        for (auto prim : prims) {
            CompBase *p = nullptr;
            string type = prim.at("type");

            p = (CompBase *)(PrimFactory::getInstance().createPrim(type));
            p->parseJson(prim);

            c.prims.push_back((PrimBase *)p);
        }
    }
}

void from_json(const json &j, CoreConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));

    if (c.id >= GRID_SIZE) {
        LOG_ERROR(config.cpp) << "Core id " << c.id << " out of range";
        return;
    }

    SetParamFromJson<int>(j, "prim_copy", &(c.prim_copy), -1);
    SetParamFromJson<int>(j, "send_global_mem", &(c.send_global_mem), -1);
    SetParamFromJson<int>(j, "loop", &(c.loop), 1);

    if (j.contains("worklist")) {
        for (int i = 0; i < j["worklist"].size(); i++) {
            CoreJob cjob = j["worklist"][i];

            if (!j["worklist"][i].contains("recv_tag")) {
                cjob.recv_tag = c.id;
            }

            c.worklist.push_back(cjob);
        }
    } else {
        // 如果是旧版config，没有worklist条目，则将所有内容作为一个单独的job
        CoreJob cjob = j;

        if (!j.contains("recv_tag")) {
            cjob.recv_tag = c.id;
        }

        c.worklist.push_back(cjob);
    }
}

void from_json(const json &j, LayerConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));

    for (int i = 0; i < j["cast"].size(); i++) {
        Cast temp = j["cast"][i];
        c.cast.push_back(temp);
    }

    // loop统一在外部填写
    if (j.contains("split")) {
        if (j["split"]["type"] == "TP")
            c.split = SPLIT_TP;
        else
            c.split = SPLIT_DP;

        c.split_dim = j["split"]["dim"];
        c.split_slice = j["split"]["slice"];
    } else {
        c.split = NO_SPLIT;
    }
}

void from_json(const json &j, StreamConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));
    SetParamFromJson<int>(j, "loop", &(c.loop), 1);

    if (j.contains("prims")) {
        auto prims = j["prims"];
        for (auto prim : prims) {
            string type = prim.at("type");
            LOG_DEBUG(CONFIG_DEBUG) << "Start parsing prim " << type;
            GpuBase *p =
                (GpuBase *)(PrimFactory::getInstance().createPrim(type));
            p->parseJson(prim);
            LOG_DEBUG(CONFIG_DEBUG) << "Parsing done for prim " << type;

            c.prims.push_back((PrimBase *)p);
        }
    }

    if (j.contains("source")) {
        auto sources = j["source"];
        for (auto source : sources) {
            c.sources.push_back(
                make_pair(source["label"], GetDefinedParam(source["size"])));
        }
    }
}

void from_json(const json &j, CoreHWConfig &c) {
    SetParamFromJson<int>(j, "id", &(c.id));

    int exu_x, sa_cnt;
    SetParamFromJson<int>(j, "exu_x", &exu_x, 128);
    SetParamFromJson<int>(j, "sa_cnt", &sa_cnt, 1);
    c.exu = new ExuConfig(MAC_Array, exu_x, sa_cnt);

    int sfu_x;
    SetParamFromJson<int>(j, "sfu_x", &sfu_x, 2048);
    c.sfu = new SfuConfig(Linear, sfu_x);

    int vec_x, vec_cnt;
    SetParamFromJson<int>(j, "vec_x", &vec_x, 64);
    SetParamFromJson<int>(j, "vec_cnt", &vec_cnt, 1);
    c.vec = new VectorConfig(vec_x, vec_cnt);

    SetParamFromJson<int>(j, "sram_bitwidth", &(c.sram_bitwidth), 128);
    SetParamFromJson<string>(j, "dram_config", &(c.dram_config),
                             DEFAULT_DRAM_CONFIG_PATH);
    SetParamFromJson<int>(j, "dram_bw", &(c.dram_bw), HW_DRAM_DEFAULT_BITWIDTH);
}