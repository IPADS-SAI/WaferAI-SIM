#include "assert.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "monitor/monitor.h"
#include "systemc.h"
#include "trace/Event_engine.h"
#include "utils/print_utils.h"
#include "utils/simple_flags.h"
#include "utils/system_utils.h"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

// 假设 json.hpp 文件在当前目录或包含路径中
#include <nlohmann/json.hpp>
#include <string>

#include <SFML/Graphics.hpp>
using namespace std;

Define_bool_opt("--help", g_flag_help, false, "show these help information");

Define_string_opt("--workload-config", g_flag_workload_config,
                  "../llm/test/workload_config/gpu/pd_serving.json",
                  "workload config file");
Define_string_opt("--hardware-config", g_flag_hardware_config,
                  "../llm/test/hardware_config/core_4x4.json",
                  "hardware config file");
Define_string_opt("--simulation-config", g_flag_simulation_config,
                  "../llm/test/simulation_config/default_spec.json",
                  "simulation config file");
Define_string_opt("--mapping-config", g_flag_mapping_config,
                  "../llm/test/mapping_config/default_mapping.txt",
                  "mapping config file");

Define_int64_opt("--trace-window", g_flag_trace_window, 2, "Trace window size");

int sc_main(int argc, char *argv[]) {
    clock_t start = clock();

    srand((unsigned)time(NULL));
    std::cout.setf(std::ios::unitbuf);

    // 解析参数
    simple_flags::parse_args(argc, argv);
    if (!simple_flags::get_unknown_flags().empty()) {
        string content;
        for (auto it : simple_flags::get_unknown_flags()) {
            content += "'" + it + "', ";
        }
        content.resize(content.size() - 2); // remove last ', '
        content.append(".");
        LOG_ERROR(CONFIG) << "Unknown option(s): " << content;
        return -1;
    }

    if (g_flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    // 清理所有上一次运行后产生的log文件
    DeleteCoreLogFiles();
    DeleteMemoryLogFiles();

    // 收集所有配置文件，统一解析
    InitGrid(g_flag_workload_config, g_flag_hardware_config,
             g_flag_simulation_config, g_flag_mapping_config);
    InitGlobalMembers();
    InitializeMemorySpec();

    // init_dram_areas();
    // initialize_cache_structures();

    Event_engine *event_engine =
        new Event_engine("event-engine", g_flag_trace_window);
    Monitor monitor("monitor", event_engine, g_flag_workload_config.c_str());
    sc_trace_file *tf = sc_create_vcd_trace_file("Cchip_1");
    sc_clock clk("clk", CYCLE, SC_NS);

    sc_start();

    // destroy_dram_areas();
    // destroy_cache_structures();
    // event_engine->dump_traced_file();
    sc_close_vcd_trace_file(tf);

    SystemCleanup();
    CloseLogFiles();

    clock_t end = clock();

    if (correct_exit) {
        LOG_INFO(SYSTEM) << "Total Real-time Cost: "
                         << (double)(end - start) / CLOCKS_PER_SEC << "s";
    } else {
        LOG_WARN(SYSTEM) << "Simulation terminated abnormally";
    }

    ofstream outfile("simulation_result_df_pd.txt", ios::app);
    if (outfile.is_open()) {
        outfile << "Total Real-time Cost: "
                << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
        outfile.close();
    } else {
        LOG_ERROR(SYSTEM) << "Unable to open file for writing timestamp";
    }
    delete event_engine;
    return 0;
}