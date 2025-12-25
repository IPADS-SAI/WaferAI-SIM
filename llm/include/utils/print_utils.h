#pragma once
#include "defs/enums.h"
#include "defs/global.h"
#include "systemc.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

std::string ToHexString(int value);

// 打印表格
void PrintBar(int length);
void PrintRow(const std::string &label, int value);

// 获取枚举类型
std::string GetEnumSendType(SEND_TYPE type);
std::string GetEnumRecvType(RECV_TYPE type);
std::string GetEnumSimulationMode(SIM_MODE mode);
std::string GetEnumDirectionType(Directions type);

enum class LogLevel { LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR };

struct LogConfig {
    static inline LogLevel CONFIG_LOG_LEVEL = LogLevel::LOG_DEBUG;
    static inline bool CONFIG_VERBOSE_DEBUG = false;
    static inline int align_width = 150; // 左右分界宽度
    static inline int prefix_width = 40; // 前缀宽度
    static inline bool CONFIG_LOG_COLORED = true;
};

// ------------------------------------------------------------
// Logger类
// ------------------------------------------------------------
class Logger {
public:
    Logger(LogLevel lvl, const std::string &module, const char *file, int line)
        : level(lvl), module(module), file(file), line(line) {}

    ~Logger() {
        if (!shouldPrint(level))
            return;

        if (module.find("DEBUG") != std::string::npos && !LogConfig::CONFIG_VERBOSE_DEBUG)
            return;

        std::ostringstream left;
        left << colorPrefix(level) << "[" << levelToString(level) << "]"
             << "\033[0m";
        left << "\033[92m[" << module << "]\033[0m ";

        std::string prefix_str = left.str();
        if (prefix_str.size() < LogConfig::prefix_width)
            left << std::string(LogConfig::prefix_width - prefix_str.size(),
                                ' ');

        left << ss.str() << ".";

        // 格式化右侧：SystemC时间戳
        std::ostringstream right;
        right << sc_core::sc_time_stamp();

        // 左右分栏对齐
        std::ostringstream final;
        std::string left_str = left.str();
        if (left_str.size() < LogConfig::align_width)
            left_str.append(LogConfig::align_width - left_str.size(), ' ');
        final << colorPrefix(level) << left_str << " | \033[36m" << right.str()
              << "\033[0m";

        // 输出
        static std::mutex mu;
        std::lock_guard<std::mutex> lock(mu);
        std::cout << final.str() << std::endl;

        if (level == LogLevel::LOG_ERROR) {
            correct_exit = false;
            sc_stop();
        }
    }

    template <typename T> Logger &operator<<(const T &val) {
        ss << val;
        return *this;
    }

private:
    std::ostringstream ss;
    LogLevel level;
    std::string module;
    std::string file;
    int line;

    bool shouldPrint(LogLevel lvl) {
        return static_cast<int>(lvl) >=
               static_cast<int>(LogConfig::CONFIG_LOG_LEVEL);
    }

    const char *levelToString(LogLevel lvl) {
        switch (lvl) {
        case LogLevel::LOG_DEBUG:
            return "DEBUG";
        case LogLevel::LOG_INFO:
            return "INFO";
        case LogLevel::LOG_WARN:
            return "WARN";
        case LogLevel::LOG_ERROR:
            return "ERROR";
        }
        return "UNKNOWN";
    }

    const char *colorPrefix(LogLevel lvl) {
        if (!LogConfig::CONFIG_LOG_COLORED)
            return "";
        switch (lvl) {
        case LogLevel::LOG_DEBUG:
            return "\033[37m"; // gray
        case LogLevel::LOG_INFO:
            return "\033[32m"; // green
        case LogLevel::LOG_WARN:
            return "\033[33m"; // yellow
        case LogLevel::LOG_ERROR:
            return "\033[31m"; // red
        }
        return "";
    }
};

// ------------------------------------------------------------
// 宏接口
// ------------------------------------------------------------
#define LOG_DEBUG(MODULE)                                                      \
    Logger(LogLevel::LOG_DEBUG, #MODULE, __FILE__, __LINE__)
#define LOG_INFO(MODULE) Logger(LogLevel::LOG_INFO, #MODULE, __FILE__, __LINE__)
#define LOG_WARN(MODULE) Logger(LogLevel::LOG_WARN, #MODULE, __FILE__, __LINE__)
#define LOG_ERROR(MODULE)                                                      \
    Logger(LogLevel::LOG_ERROR, #MODULE, __FILE__, __LINE__)
