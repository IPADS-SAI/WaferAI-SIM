#pragma once
#include "systemc.h"

// 定义 Request 结构体
struct Request {
    uint64_t address;
    enum Command { Read, Write, Invalid } command;
    int length;
    sc_core::sc_time delay;
};