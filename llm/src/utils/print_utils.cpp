#include "utils/print_utils.h"
#include "defs/enums.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "systemc.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

std::string ToHexString(int value) {
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << value;
    return stream.str();
}

void PrintBar(int length) {
    for (int i = 0; i < length; ++i)
        std::cout << "=";

    std::cout << "\n";
}

void PrintRow(const std::string &label, int value) {
    std::cout << "| " << std::left << std::setw(20) << label << "| "
              << std::right << std::setw(15) << value << " |\n";
}

std::string GetEnumSendType(SEND_TYPE type) {
    const std::unordered_map<SEND_TYPE, std::string> SEND_TYPE_NAMES = {
        {SEND_ACK, "SEND_ACK"},   {SEND_REQ, "SEND_REQ"},
        {SEND_DATA, "SEND_DATA"}, {SEND_SRAM, "SEND_SRAM"},
        {SEND_DONE, "SEND_DONE"},
    };

    auto it = SEND_TYPE_NAMES.find(type);
    if (it != SEND_TYPE_NAMES.end()) {
        return it->second;
    }

    return "Unknown SEND_TYPE";
}

std::string GetEnumRecvType(RECV_TYPE type) {
    const std::unordered_map<RECV_TYPE, std::string> RECV_TYPE_NAMES = {
        {RECV_TYPE::RECV_CONF, "RECV_CONF"},
        {RECV_TYPE::RECV_ACK, "RECV_ACK"},
        {RECV_TYPE::RECV_FLAG, "RECV_FLAG"},
        {RECV_TYPE::RECV_DATA, "RECV_DATA"},
        {RECV_TYPE::RECV_SRAM, "RECV_SRAM"},
        {RECV_TYPE::RECV_WEIGHT, "RECV_WEIGHT"},
        {RECV_TYPE::RECV_START, "RECV_START"}};

    auto it = RECV_TYPE_NAMES.find(type);
    if (it != RECV_TYPE_NAMES.end()) {
        return it->second;
    }

    return "Unknown RECV_TYPE";
}

std::string GetEnumSimulationMode(SIM_MODE mode) {
    const std::unordered_map<SIM_MODE, std::string> SIM_MODE_NAMES = {
        {SIM_DATAFLOW, "SIM_DATAFLOW"},
        {SIM_GPU, "SIM_GPU"},
        {SIM_PD, "SIM_PD"},
        {SIM_PDS, "SIM_PDS"},
        {SIM_GPU_PD, "SIM_GPU_PD"}};

    auto it = SIM_MODE_NAMES.find(mode);
    if (it != SIM_MODE_NAMES.end()) {
        return it->second;
    }

    return "Unknown SIM_MODE";
}


std::string GetEnumDirectionType(Directions type) {
    const std::unordered_map<Directions, std::string> DIRECTION_TYPE_NAMES = {
        {NORTH, "NORTH"}, {SOUTH, "SOUTH"},   {EAST, "EAST"},
        {WEST, "WEST"},   {CENTER, "CENTER"}, {DIRECTIONS, "DIRECTIONS"},
        {HOST, "HOST"}};

    auto it = DIRECTION_TYPE_NAMES.find(type);
    if (it != DIRECTION_TYPE_NAMES.end()) {
        return it->second;
    }

    return "Unknown DIRECTION_TYPE";
}


void LogVerboseImpl(int level, int core_id, const std::string &message) {
    std::ostringstream oss;

#if ENABLE_COLORS == 1
    oss << get_core_color(core_id) << "[INFO] Core " << core_id << " "
        << message << " " << sc_time_stamp().to_string() << "\033[0m";
#else

    oss << "[INFO] Core " << core_id << " " << message << " "
        << sc_time_stamp().to_string();
#endif
    std::string log_msg = oss.str();

    // 控制台输出
    std::cout << log_msg << std::endl;

    // 文件输出
    auto it = g_log_streams.find(core_id);
    if (it == g_log_streams.end()) {
        std::string filename = "core_" + std::to_string(core_id) + ".log";
        g_log_streams[core_id] = new std::ofstream(filename, std::ios::app);
        if (*g_log_streams[core_id])
            *g_log_streams[core_id] << "-- New Session --\n";
    }
    if (g_log_streams[core_id] && g_log_streams[core_id]->is_open()) {
        *g_log_streams[core_id] << log_msg << std::endl;
    }
}