#include "systemc.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#include "defs/const.h"
#include "defs/enums.h"
#include "defs/global.h"
#include "utils/config_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

#include "nlohmann/json.hpp"

using json = nlohmann::json;

bool RandResult(int threshold) {
    int num = rand() % 100 + 1;
    return num <= threshold;
}

int GetFromPairedVector(vector<pair<string, int>> &vector, string key) {
    for (auto &pair : vector) {
        if (pair.first == key)
            return pair.second;
    }

    LOG_ERROR(system_utils.cpp)
        << "Key " << key << " does not exist in vector.";
    return -1;
}

CoreHWConfig *GetCoreHWConfig(int id) {
    for (auto pair : g_core_hw_config) {
        if (pair.first == id)
            return pair.second;
    }

    LOG_ERROR(system_utils.cpp)
        << "Core HW config for id " << id << " does not exist.";
    return new CoreHWConfig();
}

int CeilingDivision(int a, int b) {
    if (b == 0) {
        LOG_ERROR(system_utils.cpp) << "Division by zero.";
    }

    return (a + b - 1) / b;
}

void InitGrid(string workload_config_path, string hardware_config_path,
              string simulation_config_path, string mapping_config_path) {
    json j1;
    ifstream jfile1(workload_config_path);

    if (!jfile1.is_open())
        LOG_ERROR(system_utils.cpp)
            << "Failed to open file " << workload_config_path;

    jfile1 >> j1;
    ParseSimulationType(j1);

    json j2;
    ifstream jfile2(hardware_config_path);

    if (!jfile2.is_open())
        LOG_ERROR(system_utils.cpp)
            << "Failed to open file " << hardware_config_path;

    jfile2 >> j2;
    ParseHardwareConfig(j2);

    json j3;
    ifstream jfile3(simulation_config_path);

    if (!jfile3.is_open())
        LOG_ERROR(system_utils.cpp) << "Failed to parse simulation config file";

    jfile3 >> j3;
    ParseSimulationConfig(j3);

    ParseMemorySpec(mapping_config_path);
}

void SystemCleanup() {
    // 清理所有原语
    for (auto p : g_prim_stash) {
        delete p;
    }

    for (auto p : g_core_hw_config) {
        delete p.second;
    }

    delete[] dram_array;
#if DCACHE == 1
    delete[] dcache_tags;
    delete[] dcache_occupancy;
    delete[] dcache_last_evicted;
#endif
}

void InitGlobalMembers() {
#if DUMMY == 1
    dram_array = new uint32_t[GRID_SIZE];
#endif

#if DCACHE == 1
    dcache_tags = new uint64_t *[GRID_SIZE];
    dcache_occupancy = new uint32_t[GRID_SIZE];
    dcache_last_evicted = new uint32_t[GRID_SIZE];
#endif
}


void DeleteCoreLogFiles() {
    const std::string current_dir = ".";
    try {
        for (const auto &entry :
             std::filesystem::directory_iterator(current_dir)) {
            if (entry.is_regular_file() &&
                entry.path().filename().string().find("core_") == 0 &&
                entry.path().extension() == ".log") {
                std::filesystem::remove(entry.path());

                LOG_INFO(SYSTEM)
                    << "Deleted log file: " << entry.path().filename().string();
            }
        }
    } catch (const std::filesystem::filesystem_error &e) {
        LOG_ERROR(system_utils.cpp)
            << "Failed to delete core log files, " << e.what();
    }
}


void DeleteMemoryLogFiles() {
    auto delete_mem = [](std::string log_dir_str, std::string log_pattern_str) {
        try {
            std::filesystem::path log_dir(log_dir_str);

            if (!std::filesystem::exists(log_dir)) {
                std::filesystem::create_directory(log_dir);
                return;
            }

            std::regex log_pattern(log_pattern_str);

            for (const auto &entry :
                 std::filesystem::directory_iterator(log_dir)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();

                    if (std::regex_match(filename, log_pattern))
                        std::filesystem::remove(entry.path());
                }
            }
        } catch (const std::filesystem::filesystem_error &e) {
            LOG_ERROR(system_utils.cpp)
                << "Failed to delete memory log files, " << e.what();
        }
    };

    delete_mem("gpu_cache", "L1Cache_cid_\\d+\\.log");
    delete_mem("sram_util", "sram_manager_cid_\\d+\\.log");
}


void CloseLogFiles() {
    for (auto &pair : g_log_streams) {
        if (pair.second && pair.second->is_open()) {
            pair.second->close();
            delete pair.second;
        }
    }
    g_log_streams.clear();
}


const char *GetCoreColor(int core_id) {
    static const char *colors[] = {
        "\033[31m", // 红色
        "\033[32m", // 绿色
        "\033[33m", // 黄色
        "\033[34m", // 蓝色
        "\033[35m", // 品红
        "\033[36m", // 青色
        "\033[91m", // 亮红
        "\033[92m", // 亮绿
    };
    return colors[core_id % (sizeof(colors) / sizeof(colors[0]))];
}


void InitializeMemorySpec() {
    modifyNbrOfDevices(
        "../DRAMSys/configs/memspec/JEDEC_4Gb_DDR4-1866_8bit_A.json",
        "../DRAMSys/configs/memspec/JEDEC_4Gb_DDR4-1866_8bit_DF.json",
        HW_DRAM_DEFAULT_BITWIDTH);
    int bytecount_df = static_cast<int>(log2(HW_DRAM_DEFAULT_BITWIDTH));
    generateDFAddressMapping(
        bytecount_df,
        "../DRAMSys/configs/addressmapping/am_ddr4_8x4Gbx8_df.json");

    if (GPU_DRAM_BANDWIDTH == 512 || GPU_DRAM_BANDWIDTH == 1024 ||
        GPU_DRAM_BANDWIDTH == 256 || GPU_DRAM_BANDWIDTH == 128 ||
        GPU_DRAM_BANDWIDTH == 64) {
        int numDevices = 32 * GPU_DRAM_BANDWIDTH / 512; // 每个设备 32 个通道
        int bytecount = static_cast<int>(log2(GPU_DRAM_BANDWIDTH)) - 1;

        generateGPUCacheJsonFile(numDevices,
                                 "../DRAMSys/configs/memspec/HBM2_GPU.json");
        generateAddressMapping(
            bytecount, "../DRAMSys/configs/addressmapping/am_hbm2_gpu.json");
        GPU_DRAM_CONFIG_FILE = "../DRAMSys/configs/gpu_hbm2.json";
    } else if (!SPEC_USE_BEHA_DRAM) {
        assert(false && "gpu bandwidth must be 512 1024 256");
    }
}


/**
 * 修改JSON文件中的nbrOfDevices值
 * @param inputPath 原JSON文件路径
 * @param outputPath 新JSON文件路径
 * @param x 要设置的nbrOfDevices值
 * @return 成功返回true，失败返回false
 */
bool modifyNbrOfDevices(const std::string &inputPath,
                        const std::string &outputPath, int x) {
    // 读取原始 JSON 文件
    std::ifstream infile(inputPath);
    if (!infile.is_open()) {
        LOG_ERROR(SYSTEM) << "Failed to open JSON file " << inputPath;
        return false;
    }

    json j;
    try {
        infile >> j;
    } catch (json::parse_error &e) {
        LOG_ERROR(SYSTEM) << "JSON parse error: " << e.what();
        return false;
    }
    infile.close();

    // 修改 nbrOfDevices 的值
    j["memspec"]["memarchitecturespec"]["nbrOfDevices"] = x;

    // 写入新文件
    std::ofstream outfile(outputPath);
    if (!outfile.is_open()) {
        LOG_ERROR(SYSTEM) << "Failed to create output file " << outputPath;
        return false;
    }

    try {
        outfile << j.dump(4); // 格式化输出，缩进4个空格
    } catch (const std::exception &e) {
        LOG_ERROR(SYSTEM) << "Error occurred while writing to file: "
                          << e.what();
        return false;
    }

    outfile.close();
    return true;
}


void generateAddressMapping(int n, const std::string &outputFilename) {
    // ROW_BIT 占 15 位，从 n+12 开始，最高位是 n+26，必须 <= 34
    json addressmapping;

    int byte_start = 0;
    int column_start = n;
    int bank_start = n + 22;
    int bankgroup_start = n + 24;
    int pseudo_start = n + 26;
    int row_start = n + 7;

    // BYTE_BIT
    std::vector<int> byte_bits;
    for (int i = 0; i < n; ++i)
        byte_bits.push_back(i);
    addressmapping["BYTE_BIT"] = byte_bits;

    // COLUMN_BIT (7 bits)
    std::vector<int> column_bits;
    for (int i = 0; i < 7; ++i)
        column_bits.push_back(column_start + i);
    addressmapping["COLUMN_BIT"] = column_bits;

    // BANK_BIT (2 bits)
    std::vector<int> bank_bits = {bank_start, bank_start + 1};
    addressmapping["BANK_BIT"] = bank_bits;

    // BANKGROUP_BIT (2 bits)
    std::vector<int> bankgroup_bits = {bankgroup_start, bankgroup_start + 1};
    addressmapping["BANKGROUP_BIT"] = bankgroup_bits;

    // PSEUDOCHANNEL_BIT (1 bit)
    addressmapping["PSEUDOCHANNEL_BIT"] = std::vector<int>{pseudo_start};

    // ROW_BIT (15 bits)
    std::vector<int> row_bits;
    for (int i = 0; i < 15; ++i) {
        int bit = row_start + i;
        row_bits.push_back(bit);
    }
    addressmapping["ROW_BIT"] = row_bits;

    // 写入文件
    json root;
    root["addressmapping"] = addressmapping;

    std::ofstream outFile(outputFilename);
    if (!outFile.is_open()) {
        LOG_ERROR(system_utils.cpp)
            << "Cannot write to file " << outputFilename;
        return;
    }
    outFile << std::setw(4) << root << std::endl;
    outFile.close();

    LOG_INFO(SYSTEM) << "Address mapping with n=" << n << " saved to "
                     << outputFilename;
}


void generateDFAddressMapping(int n, const std::string &outputFilename) {
    // ROW_BIT 占 15 位，从 n+12 开始，最高位是 n+26，必须 <= 34
    json addressmapping;

    int byte_start = 0;
    int column_start = n;
    int bank_start = n + 27;
    int bankgroup_start = n + 25;
    int row_start = n + 10;

    // BYTE_BIT
    std::vector<int> byte_bits;
    for (int i = 0; i < n; ++i)
        byte_bits.push_back(i);
    addressmapping["BYTE_BIT"] = byte_bits;

    // COLUMN_BIT (10 bits)
    std::vector<int> column_bits;
    for (int i = 0; i < 10; ++i)
        column_bits.push_back(column_start + i);
    addressmapping["COLUMN_BIT"] = column_bits;

    // BANK_BIT (2 bits)
    std::vector<int> bank_bits = {bank_start, bank_start + 1};
    addressmapping["BANK_BIT"] = bank_bits;

    // BANKGROUP_BIT (2 bits)
    std::vector<int> bankgroup_bits = {bankgroup_start, bankgroup_start + 1};
    addressmapping["BANKGROUP_BIT"] = bankgroup_bits;

    // ROW_BIT (15 bits)
    std::vector<int> row_bits;
    for (int i = 0; i < 15; ++i) {
        int bit = row_start + i;
        row_bits.push_back(bit);
    }
    addressmapping["ROW_BIT"] = row_bits;

    // 写入文件
    json root;
    root["addressmapping"] = addressmapping;

    std::ofstream outFile(outputFilename);
    if (!outFile.is_open()) {
        LOG_ERROR(system_utils.cpp)
            << "Cannot write to file " << outputFilename;
        return;
    }
    outFile << std::setw(4) << root << std::endl;
    outFile.close();

    LOG_INFO(SYSTEM) << "Address mapping with n=" << n << " saved to "
                     << outputFilename;
}


/**
 * @brief 根据给定的 nbrOfDevices 值生成一个 JSON 文件。
 *
 * @param A 要写入 nbrOfDevices 字段的整数值。
 * @param filename 要生成的 JSON 文件的名称（可选，默认为 "output.json"）。
 * @return bool 如果文件成功生成则返回 true，否则返回 false。
 */
bool generateGPUCacheJsonFile(int numDevices, const std::string &filename) {
    try {
        // 1. 创建 JSON 对象结构并填充初始数据
        json j = {
            {"memspec",
             {{"memarchitecturespec",
               {{"burstLength", 4},
                {"dataRate", 2},
                {"nbrOfBankGroups", 4},
                {"nbrOfBanks", 16},
                {"nbrOfColumns", 128},
                {"nbrOfPseudoChannels", 2},
                {"nbrOfRows", 32768},
                {"width", 64},
                // 注意：这里初始化为 32，但会被参数 A 覆盖
                {"nbrOfDevices", 32},
                {"nbrOfChannels", 1}

               }},
              {"memoryId", "https://www.computerbase.de/2019-05/"
                           "amd-memory-tweak-vram-oc/#bilder"},
              {"memoryType", "HBM2"},
              {"memtimingspec", {{"CCDL", 3},  {"CCDS", 2},    {"CKE", 8},
                                 {"DQSCK", 1}, {"FAW", 16},    {"PL", 0},
                                 {"RAS", 28},  {"RC", 42},     {"RCDRD", 12},
                                 {"RCDWR", 6}, {"REFI", 3900}, {"REFISB", 244},
                                 {"RFC", 220}, {"RFCSB", 96},  {"RL", 17},
                                 {"RP", 14},   {"RRDL", 6},    {"RRDS", 4},
                                 {"RREFD", 8}, {"RTP", 5},     {"RTW", 18},
                                 {"WL", 7},    {"WR", 14},     {"WTRL", 9},
                                 {"WTRS", 4},  {"XP", 8},      {"XS", 216},
                                 {"tCK", 1000}}}}}};

        // 2. 使用传入的参数 A 替换 nbrOfDevices 的值
        j["memspec"]["memarchitecturespec"]["nbrOfDevices"] = numDevices;

        // 3. 打开文件流用于写入
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            LOG_ERROR(system_utils.cpp) << "Cannot write to file " << filename;
            return false;
        }

        // 4. 将 JSON 对象序列化为格式化的字符串并写入文件
        // setw(4) 用于美化输出，使其具有缩进
        outFile << j.dump(4);
        outFile.close();

        return true;
    } catch (const std::exception &e) {
        LOG_ERROR(system_utils.cpp)
            << "Error occurred while generating GPU cache JSON file: "
            << e.what();
        return false;
    }
}

// void initialize_cache_structures() {
// #if DCACHE == 1
//     // g_data_footprint_in_words = GRID_SIZE * dataset_words_per_tile;
//     //global
//     // variable 全局的darray的大小 所有的tile

//     g_data_footprint_in_words =
//         GRID_SIZE * dataset_words_per_tile; // global variable
//     printf("GRID_SIZE%d, ss%ld\n", GRID_SIZE, dataset_words_per_tile);
//     assert(g_data_footprint_in_words > 0);

//     // u_int64_t total_lines =
//     //     g_data_footprint_in_words >> dcache_words_in_line_log2;
//     u_int64_t total_lines = dataset_words_per_tile >>
//     dcache_words_in_line_log2; printf("dataset_words_per_tile %ld \n",
//     dataset_words_per_tile); printf("g_data_footprint_in_words %ld \n",
//     g_data_footprint_in_words); printf("total_lines %ld \n", total_lines);
//     // dcache_freq = (u_int16_t *)calloc(total_lines, sizeof(u_int16_t));
//     // dcache_dirty = (bool *)calloc(total_lines, sizeof(bool));
//     dcache_dirty =
//         new std::unordered_set<uint64_t>[GRID_SIZE]; // One set per tile

//     // dcache_size dcache size of each tile
//     u_int64_t lines_per_tile = dcache_size >> dcache_words_in_line_log2;
//     for (int i = 0; i < GRID_SIZE; i++) {
//         u_int64_t *array_uint =
//             (u_int64_t *)calloc(lines_per_tile, sizeof(u_int64_t));
//         // bool *dcache_dirty = (bool *)calloc(total_lines,
//         sizeof(bool)); for (int j = 0; j < lines_per_tile; j++) {
//             array_uint[j] = UINT64_MAX;
//         }
//         dcache_tags[i] = array_uint;
//         dcache_occupancy[i] = 0;
//         dcache_last_evicted[i] = 0;
//     }
// #endif
// }

// void init_dram_areas() {
//     for (int i = 0; i < GRID_SIZE; i++) {
// #if DUMMY == 1
// #else
//         dram_array[i] =
//             (uint32_t *)calloc(dataset_words_per_tile, sizeof(uint32_t));
//         if (dram_array[i] == NULL)
//             std::cout << "Failed to calloc dram.\n";
// #endif
//     }

//     for (int i = 0; i < GRID_SIZE; i++) {
// #if DUMMY == 1
// #else
//         float *a = (float *)(dram_array[i]);
//         for (int j = 0; j < dataset_words_per_tile; j++)
//             a[j] = 1;
// #endif
//     }
// }

// void destroy_dram_areas() {
//     // free all of the dram areas
// #if DUMMY == 1
// #else
//     for (int i = 0; i < GRID_X; i++) {
//         free(dram_array[i]);
//         dram_array[i] = NULL;
//     }
// #endif
//     std::cout << "DRAM destroyed.\n";
// }
