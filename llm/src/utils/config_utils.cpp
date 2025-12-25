#include <regex>

#include "common/config.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "utils/config_utils.h"
#include "utils/print_utils.h"

int GetDefinedParam(string var) {
    for (auto v : vtable) {
        if (v.first == var)
            return v.second;
    }

    LOG_ERROR(config_utils.cpp) << "Undefined variable " << var;
    return 1;
}

void ParseSimulationType(json j) {
    std::unordered_map<string, SIM_MODE> sim_mode_map = {
        {"dataflow", SIM_DATAFLOW},
        {"gpu", SIM_GPU},
        {"sched_pd", SIM_PD},
        {"sched_pds", SIM_PDS},
        {"gpu_pd", SIM_GPU_PD}};

    if (j.contains("mode")) {
        auto mode = j["mode"];

        if (sim_mode_map.find(mode) != sim_mode_map.end())
            SYSTEM_MODE = sim_mode_map[mode];
        else
            LOG_ERROR(config_utils.cpp)
                << "Unsupported simulation mode " << mode;
    } else
        SYSTEM_MODE = SIM_DATAFLOW;

    LOG_INFO(SYSTEM) << "System running in simulation mode "
                     << GetEnumSimulationMode(SYSTEM_MODE);

    if (SYSTEM_MODE == SIM_GPU) {
        if (j.contains("chips"))
            CORE_PER_SM = j["chips"][0]["core_per_sm"];
        if (USE_L1L2_CACHE != 1)
            LOG_ERROR(config_utils.cpp)
                << "L1L2 cache unavailable for GPU simulation.";
    }
}

void ParseHardwareConfig(json j) {
    if (j.contains("x"))
        GRID_X = j["x"];
    GRID_Y = GRID_X;
    GRID_SIZE = GRID_X * GRID_Y;

    if (j.contains("noc")) {
        auto conf_noc = j["noc"];
        if (conf_noc.contains("noc_payload_per_cycle"))
            HW_NOC_PAYLOAD_PER_CYCLE = conf_noc["noc_payload_per_cycle"];
    }

    if (j.contains("operand")) {
        auto conf_operand = j["operand"];
        if (conf_operand.contains("core_credit"))
            HW_CORE_CREDIT = conf_operand["core_credit"];
        if (conf_operand.contains("pd_ratio"))
            HW_PD_RATIO = conf_operand["pd_ratio"];
        if (conf_operand.contains("comp_util"))
            HW_COMP_UTIL = conf_operand["comp_util"];
    }

    if (j.contains("memory")) {
        auto conf_memory = j["memory"];
        if (conf_memory.contains("sram_size"))
            HW_SRAM_SIZE = conf_memory["sram_size"];

        if (conf_memory.contains("dram_default_bitwidth"))
            HW_DRAM_DEFAULT_BITWIDTH = conf_memory["dram_default_bitwidth"];

        if (conf_memory.contains("beha_dram_util"))
            HW_BEHA_DRAM_UTIL = conf_memory["beha_dram_util"];
    }

    if (j.contains("gpu")) {
        auto conf_gpu = j["gpu"];
        if (conf_gpu.contains("dram_bandwidth"))
            GPU_DRAM_BANDWIDTH = conf_gpu["dram_bandwidth"];
        if (conf_gpu.contains("dram_burst_size"))
            GPU_DRAM_BURST_SIZE = conf_gpu["dram_burst_size"];
        if (conf_gpu.contains("dram_aligned"))
            GPU_DRAM_ALIGNED = conf_gpu["dram_aligned"];
    }

    auto config_cores = j["cores"];
    CoreHWConfig sample = config_cores[0];

    for (auto core : config_cores) {
        CoreHWConfig c = core;
        for (int i = sample.id + 1; i < c.id; i++) {
            ExuConfig *exu =
                new ExuConfig(MAC_Array, sample.exu->x_dims, sample.exu->count);
            SfuConfig *sfu = new SfuConfig(Linear, sample.sfu->x_dims);
            VectorConfig *vec =
                new VectorConfig(sample.vec->x_dims, sample.vec->count);
            g_core_hw_config.push_back(make_pair(
                i, new CoreHWConfig(i, exu, sfu, vec, sample.dram_config,
                                    sample.dram_bw, sample.sram_bitwidth)));
        }

        ExuConfig *exu = new ExuConfig(MAC_Array, c.exu->x_dims, c.exu->count);
        SfuConfig *sfu = new SfuConfig(Linear, c.sfu->x_dims);
        VectorConfig *vec = new VectorConfig(c.vec->x_dims, c.vec->count);
        g_core_hw_config.push_back(
            make_pair(c.id, new CoreHWConfig(c.id, exu, sfu, vec, c.dram_config,
                                             c.dram_bw, c.sram_bitwidth)));

        delete sample.exu;
        delete sample.sfu;
        delete sample.vec;
        sample = c;
        sample.exu = new ExuConfig(MAC_Array, c.exu->x_dims, c.exu->count);
        sample.sfu = new SfuConfig(Linear, c.sfu->x_dims);
        sample.vec = new VectorConfig(c.vec->x_dims, c.vec->count);
    }

    for (int i = sample.id + 1; i < GRID_SIZE; i++) {
        ExuConfig *exu =
            new ExuConfig(MAC_Array, sample.exu->x_dims, sample.exu->count);
        SfuConfig *sfu = new SfuConfig(Linear, sample.sfu->x_dims);
        VectorConfig *vec =
            new VectorConfig(sample.vec->x_dims, sample.vec->count);
        g_core_hw_config.push_back(make_pair(
            i, new CoreHWConfig(i, exu, sfu, vec, sample.dram_config,
                                sample.dram_bw, sample.sram_bitwidth)));
    }

    for (auto core : g_core_hw_config)
        core.second->printSelf();
}

void ParseSimulationConfig(json j) {
    // 设置仿真相关参数
    if (j.contains("ttf_file"))
        SPEC_TTF_FILE = j["ttf_file"];

    if (j.contains("operand")) {
        auto conf_operand = j["operand"];
        if (conf_operand.contains("use_pref_gemm"))
            SPEC_USE_PERF_GEMM = conf_operand["use_perf_gemm"];
        if (conf_operand.contains("load_static_as_tile"))
            SPEC_LOAD_STATIC = conf_operand["load_static_as_tile"];
    }

    if (j.contains("memory")) {
        auto conf_memory = j["memory"];
        if (conf_memory.contains("use_beha_sram"))
            SPEC_USE_BEHA_SRAM = conf_memory["use_beha_sram"];
        if (conf_memory.contains("use_beha_dram"))
            SPEC_USE_BEHA_DRAM = conf_memory["use_beha_dram"];
        if (conf_memory.contains("kvcache_spill"))
            SPEC_KVCACHE_SPILL = conf_memory["kvcache_spill"];
        if (conf_memory.contains("use_dramsys"))
            SPEC_USE_DRAMSYS = conf_memory["use_dramsys"];
    }

    if (j.contains("noc")) {
        auto conf_noc = j["noc"];
        if (conf_noc.contains("use_beha_noc"))
            SPEC_USE_BEHA_NOC = conf_noc["use_beha_noc"];
        if (conf_noc.contains("router_pipe"))
            SPEC_ROUTER_PIPE = conf_noc["router_pipe"];
        if (conf_noc.contains("fast_warmup"))
            SPEC_FAST_WARMUP = conf_noc["fast_warmup"];
        if (conf_noc.contains("send_recv_parallel"))
            SPEC_SEND_RECV_PARALLEL = conf_noc["send_recv_parallel"];
    }

    if (j.contains("gpu")) {
        auto conf_gpu = j["gpu"];
        if (conf_gpu.contains("use_inner_mm"))
            GPU_USE_INNER_MM = conf_gpu["use_inner_mm"];
        if (conf_gpu.contains("cache_log"))
            GPU_CACHE_LOG = conf_gpu["cache_log"];
        if (conf_gpu.contains("dram_config_file"))
            GPU_DRAM_CONFIG_FILE = conf_gpu["dram_config_file"];
    }

    if (j.contains("log")) {
        auto conf_log = j["log"];
        if (conf_log.contains("log_level"))
            LogConfig::CONFIG_LOG_LEVEL = LogLevel(conf_log["log_level"]);
        if (conf_log.contains("verbose_debug"))
            LogConfig::CONFIG_VERBOSE_DEBUG = conf_log["verbose_debug"];
        if (conf_log.contains("colored"))
            LogConfig::CONFIG_LOG_COLORED = conf_log["colored"];
    }
}

void ParseMemorySpec(string filename) {
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        LOG_WARN(config_utils.cpp) << "Cannot open file: " << filename;
        return;
    }

    std::string line;

    while (std::getline(infile, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string x_str, y_str;

        if (std::getline(ss, x_str, ':') && std::getline(ss, y_str)) {
            int x = std::stoi(x_str);
            int y = std::stoi(y_str);
            g_core_remap[x] = y;
            LOG_DEBUG(CONFIG) << "Remap core " << x << " to " << y;
        }
    }

    infile.close();
}

void CoreConfigRemap(vector<pair<int, int>> &source_info,
                     vector<CoreConfig> &coreconfigs) {
    auto get_or_key = [&](int x) -> int {
        auto it = g_core_remap.find(x);
        return (it != g_core_remap.end()) ? it->second : x;
    };

    for (auto &config : coreconfigs) {
        int old_id = config.id;
        config.id = get_or_key(config.id);

        if (config.prim_copy != -1)
            config.prim_copy = get_or_key(config.prim_copy);

        for (auto &work : config.worklist) {
            if (work.recv_tag == old_id)
                work.recv_tag = config.id;

            for (auto &cast : work.cast) {
                if (cast.tag == cast.dest)
                    cast.tag = get_or_key(cast.dest);

                cast.dest = get_or_key(cast.dest);
            }
        }
    }

    for (auto &source : source_info) {
        source.first = get_or_key(source.first);
    }
}